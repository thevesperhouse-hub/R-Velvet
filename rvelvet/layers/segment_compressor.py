"""
Learned compression via cross-attention. Learned query vectors attend to segment tokens,
forcing the model to extract salient information through the bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SegmentCompressor(nn.Module):
    """
    Compresses token segments into concept vectors via learned cross-attention queries.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        segment_size: int = 512,
        n_concepts: int = 1,
        n_refine_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.segment_size = segment_size
        self.n_concepts = n_concepts
        self.scale = self.head_dim ** -0.5

        self.concept_queries = nn.Parameter(
            torch.randn(n_concepts, d_model) * 0.02
        )

        self.refine_layers = nn.ModuleList([
            CompressorCrossAttention(d_model, n_heads, dropout)
            for _ in range(n_refine_layers)
        ])

        self.seg_pos_enc = nn.Parameter(
            torch.randn(1, segment_size, d_model) * 0.02
        )

        self.out_norm = RMSNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor:
        B, L, D = tokens.shape
        W = self.segment_size
        K = self.n_concepts

        pad_len = (W - L % W) % W
        if pad_len > 0:
            tokens = F.pad(tokens, (0, 0, 0, pad_len))

        _, L_padded, _ = tokens.shape
        n_segments = L_padded // W

        segments = tokens.view(B, n_segments, W, D)

        pos_enc = self.seg_pos_enc[:, :W, :]
        segments = segments + pos_enc

        concepts = self.concept_queries.unsqueeze(0).unsqueeze(0)
        concepts = concepts.expand(B, n_segments, -1, -1)

        all_weights = []
        for layer in self.refine_layers:
            concepts, w = layer(
                queries=concepts,
                keys_values=segments,
            )
            if return_weights:
                all_weights.append(w)

        concepts = self.out_norm(concepts)

        if return_weights:
            return concepts, all_weights

        return concepts

    def decompress(
        self,
        concepts: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        B, n_seg, K, D = concepts.shape
        W = self.segment_size

        expanded = concepts.mean(dim=2, keepdim=True)
        expanded = expanded.expand(-1, -1, W, -1)

        expanded = expanded.reshape(B, n_seg * W, D)

        expanded = expanded[:, :target_len, :]

        return expanded


class CompressorCrossAttention(nn.Module):
    """
    Cross-attention where queries are concept vectors and keys/values are segment tokens.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model, bias=False),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys_values: torch.Tensor,
    ) -> tuple:
        B, S, K, D = queries.shape
        _, _, W, _ = keys_values.shape
        H = self.n_heads
        hd = self.head_dim

        q_in = self.norm_q(queries)
        kv_in = self.norm_kv(keys_values)

        q = self.q_proj(q_in).view(B, S, K, H, hd).permute(0, 1, 3, 2, 4)
        k = self.k_proj(kv_in).view(B, S, W, H, hd).permute(0, 1, 3, 2, 4)
        v = self.v_proj(kv_in).view(B, S, W, H, hd).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        weights = attn
        attn = self.dropout(attn)

        out = attn @ v
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, S, K, D)
        out = self.out_proj(out)

        refined = queries + out

        refined = refined + self.ffn(self.ffn_norm(refined))

        return refined, weights


class AdaptiveSegmentCompressor(nn.Module):
    """
    Route-aware compressor with three modes: SKIM (1 concept), PROCESS (4 concepts), FOCUS (16 concepts).
    All routes use learned cross-attention queries.
    """

    N_CONCEPTS_SKIM = 1
    N_CONCEPTS_PROCESS = 4

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        segment_size: int = 512,
        n_concepts_focus: int = 16,
        n_refine_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.segment_size = segment_size
        self.n_concepts_focus = n_concepts_focus
        self.n_refine_layers = n_refine_layers

        self.concept_queries_skim = nn.Parameter(
            torch.randn(self.N_CONCEPTS_SKIM, d_model) * 0.02
        )

        self.concept_queries_process = nn.Parameter(
            torch.randn(self.N_CONCEPTS_PROCESS, d_model) * 0.02
        )

        self.concept_queries_focus = nn.Parameter(
            torch.randn(n_concepts_focus, d_model) * 0.02
        )

        self.refine_layers = nn.ModuleList([
            CompressorCrossAttention(d_model, n_heads, dropout)
            for _ in range(n_refine_layers)
        ])

        self.seg_pos_enc = nn.Parameter(
            torch.randn(1, segment_size, d_model) * 0.02
        )

    @property
    def k_max(self) -> int:
        return self.n_concepts_focus

    def compress_segment(
        self,
        local_out: torch.Tensor,
        route_name: str,
        n_refine: int = None,
    ) -> torch.Tensor:
        B, S, W, D = local_out.shape
        n_refine = n_refine or self.n_refine_layers

        pos_enc = self.seg_pos_enc[:, :W, :]
        segments = local_out + pos_enc

        if route_name == 'SKIM':
            queries = self.concept_queries_skim
        elif route_name == 'PROCESS':
            queries = self.concept_queries_process
        elif route_name == 'FOCUS':
            queries = self.concept_queries_focus
        else:
            raise ValueError(f"Unknown route: {route_name}")

        K = queries.shape[0]

        concepts = queries.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)

        for layer in self.refine_layers[:n_refine]:
            concepts, _ = layer(
                queries=concepts,
                keys_values=segments,
            )

        return concepts

    def forward(
        self,
        local_out: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        B, S, W, D = local_out.shape
        K_max = self.k_max

        if not self.training:
            return self._forward_hard(local_out, route_weights)

        focus_out = self.compress_segment(local_out, 'FOCUS')

        process_out = self.compress_segment(local_out, 'PROCESS')
        process_padded = F.pad(process_out, (0, 0, 0, K_max - self.N_CONCEPTS_PROCESS))

        skim_out = self.compress_segment(local_out, 'SKIM')
        skim_padded = F.pad(skim_out, (0, 0, 0, K_max - self.N_CONCEPTS_SKIM))

        w_skim = route_weights[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        w_process = route_weights[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        w_focus = route_weights[:, :, 2].unsqueeze(-1).unsqueeze(-1)

        blended = w_skim * skim_padded + w_process * process_padded + w_focus * focus_out
        return blended

    def _forward_hard(
        self,
        local_out: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        B, S, W, D = local_out.shape
        K_max = self.k_max
        device = local_out.device

        route_idx = route_weights.argmax(dim=-1)  # (B, S)
        output = torch.zeros(B, S, K_max, D, device=device)

        for route_id, route_name in enumerate(['SKIM', 'PROCESS', 'FOCUS']):
            mask = (route_idx == route_id)
            if not mask.any():
                continue

            batch_indices, seg_indices = mask.nonzero(as_tuple=True)
            if len(batch_indices) == 0:
                continue

            selected = local_out[batch_indices, seg_indices].unsqueeze(1)
            compressed = self.compress_segment(selected, route_name)
            K = compressed.shape[2]
            output[batch_indices, seg_indices, :K, :] = compressed.squeeze(1)

        return output


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
