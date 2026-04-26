"""
Segment Compressor: The novel piece.

Compresses a window of tokens into a single "concept" vector.
Not pooling. Not approximation. LEARNED compression.

Key idea: A small set of learned "summary queries" attend to all
tokens in a chunk via cross-attention. The bottleneck forces the
model to learn WHAT MATTERS.

2048 tokens → 1 concept vector (or k concept vectors)

This is fundamentally different from:
- Mean pooling (loses everything)
- CLS token (no explicit compression objective)
- Perceiver (uses it as whole architecture, we use it as one stage)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SegmentCompressor(nn.Module):
    """
    Compress segments of tokens into concept vectors.

    Uses learned query vectors that attend to chunk tokens via
    cross-attention. The extreme bottleneck (2048 → k) forces
    the model to learn meaningful compression.

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        segment_size: Number of tokens per segment (window)
        n_concepts: Number of concept vectors per segment
                    (1 = maximum compression, 4 = more detail)
        n_refine_layers: How many rounds of cross-attention refinement
        dropout: Dropout rate
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

        # Learned concept queries - these learn "what to extract"
        # Each query specializes on a different type of information
        self.concept_queries = nn.Parameter(
            torch.randn(n_concepts, d_model) * 0.02
        )

        # Cross-attention layers for refinement
        # Multiple rounds = progressively better compression
        self.refine_layers = nn.ModuleList([
            CompressorCrossAttention(d_model, n_heads, dropout)
            for _ in range(n_refine_layers)
        ])

        # Segment position encoding (relative position within chunk)
        self.seg_pos_enc = nn.Parameter(
            torch.randn(1, segment_size, d_model) * 0.02
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor:
        """
        Compress token sequences into concept vectors.

        Args:
            tokens: (B, L, D) - local-encoded tokens
            return_weights: If True, also return attention weights (for analysis)

        Returns:
            concepts: (B, n_segments, n_concepts, D) - compressed concepts
            weights: (optional) attention weights showing what each concept focuses on
        """
        B, L, D = tokens.shape
        W = self.segment_size
        K = self.n_concepts

        # Pad to multiple of segment_size
        pad_len = (W - L % W) % W
        if pad_len > 0:
            tokens = F.pad(tokens, (0, 0, 0, pad_len))

        _, L_padded, _ = tokens.shape
        n_segments = L_padded // W

        # Split into segments: (B, n_segments, W, D)
        segments = tokens.view(B, n_segments, W, D)

        # Add position encoding within each segment
        pos_enc = self.seg_pos_enc[:, :W, :]
        segments = segments + pos_enc

        # Initialize concept vectors from learned queries
        # (B, n_segments, K, D)
        concepts = self.concept_queries.unsqueeze(0).unsqueeze(0)
        concepts = concepts.expand(B, n_segments, -1, -1)

        # Refine concepts through cross-attention
        all_weights = []
        for layer in self.refine_layers:
            concepts, w = layer(
                queries=concepts,       # (B, n_segments, K, D)
                keys_values=segments,   # (B, n_segments, W, D)
            )
            if return_weights:
                all_weights.append(w)

        if return_weights:
            return concepts, all_weights

        return concepts

    def decompress(
        self,
        concepts: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """
        Expand concepts back to token-level (approximate reconstruction).

        Used when the global reasoner identifies a segment as relevant
        and needs to "zoom in" to get details.

        Args:
            concepts: (B, n_segments, K, D)
            target_len: Original sequence length

        Returns:
            expanded: (B, L, D) - expanded back to token level
        """
        B, n_seg, K, D = concepts.shape
        W = self.segment_size

        # Repeat each concept for its segment
        # (B, n_segments, K, D) -> (B, n_segments, W, D)
        expanded = concepts.mean(dim=2, keepdim=True)  # (B, n_seg, 1, D)
        expanded = expanded.expand(-1, -1, W, -1)      # (B, n_seg, W, D)

        # Reshape to sequence
        expanded = expanded.reshape(B, n_seg * W, D)

        # Trim to target length
        expanded = expanded[:, :target_len, :]

        return expanded


class CompressorCrossAttention(nn.Module):
    """
    Cross-attention for concept refinement.

    Queries = concept vectors (what we're building)
    Keys/Values = segment tokens (what we're compressing from)

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # Query projection (from concepts)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        # Key/Value projection (from tokens)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Norms
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)

        # FFN after cross-attention
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
        """
        Args:
            queries: (B, S, K, D) - concept vectors
            keys_values: (B, S, W, D) - segment tokens

        Returns:
            refined: (B, S, K, D) - refined concept vectors
            weights: (B, S, H, K, W) - attention weights
        """
        B, S, K, D = queries.shape
        _, _, W, _ = keys_values.shape
        H = self.n_heads
        hd = self.head_dim

        # Normalize
        q_in = self.norm_q(queries)
        kv_in = self.norm_kv(keys_values)

        # Project
        q = self.q_proj(q_in).view(B, S, K, H, hd).permute(0, 1, 3, 2, 4)  # (B, S, H, K, hd)
        k = self.k_proj(kv_in).view(B, S, W, H, hd).permute(0, 1, 3, 2, 4)  # (B, S, H, W, hd)
        v = self.v_proj(kv_in).view(B, S, W, H, hd).permute(0, 1, 3, 2, 4)  # (B, S, H, W, hd)

        # Cross-attention: queries attend to keys
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, S, H, K, W)
        attn = F.softmax(attn, dim=-1)
        weights = attn
        attn = self.dropout(attn)

        # Apply to values
        out = attn @ v  # (B, S, H, K, hd)
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, S, K, D)
        out = self.out_proj(out)

        # Residual
        refined = queries + out

        # FFN
        refined = refined + self.ffn(self.ffn_norm(refined))

        return refined, weights


class AdaptiveSegmentCompressor(nn.Module):
    """
    Route-aware segment compressor for ACR.

    Three compression modes sharing the same cross-attention layers:
    - SKIM: 1 concept query per segment (maximum compression, 512:1)
    - PROCESS: 4 concept queries per segment (moderate, 128:1)
    - FOCUS: 16 concept queries per segment (minimal compression, 32:1)

    All three routes use learned cross-attention queries (no pass-through).
    This keeps the concept count bounded and avoids zero-padding issues
    in the global reasoner.

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        segment_size: Tokens per segment
        n_concepts_focus: Number of concept queries for FOCUS route
        n_refine_layers: Cross-attention refinement rounds
        dropout: Dropout rate
    """

    # Concept counts per route (used for masking)
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

        # SKIM: 1 concept query (maximum compression)
        self.concept_queries_skim = nn.Parameter(
            torch.randn(self.N_CONCEPTS_SKIM, d_model) * 0.02
        )

        # PROCESS: 4 concept queries (moderate compression)
        self.concept_queries_process = nn.Parameter(
            torch.randn(self.N_CONCEPTS_PROCESS, d_model) * 0.02
        )

        # FOCUS: 16 concept queries (minimal compression, still compressed)
        self.concept_queries_focus = nn.Parameter(
            torch.randn(n_concepts_focus, d_model) * 0.02
        )

        # Shared cross-attention refinement layers
        self.refine_layers = nn.ModuleList([
            CompressorCrossAttention(d_model, n_heads, dropout)
            for _ in range(n_refine_layers)
        ])

        # Segment position encoding
        self.seg_pos_enc = nn.Parameter(
            torch.randn(1, segment_size, d_model) * 0.02
        )

    @property
    def k_max(self) -> int:
        """Maximum concepts per segment (= FOCUS count)."""
        return self.n_concepts_focus

    def compress_segment(
        self,
        local_out: torch.Tensor,
        route_name: str,
        n_refine: int = None,
    ) -> torch.Tensor:
        """
        Compress segments using the specified route.

        Args:
            local_out: (B, S, W, D) - locally-encoded segments
            route_name: 'SKIM', 'PROCESS', or 'FOCUS'
            n_refine: Override number of refinement layers (default: use all)

        Returns:
            compressed: (B, S, K, D) where K depends on route
                SKIM: K=1, PROCESS: K=4, FOCUS: K=n_concepts_focus
        """
        B, S, W, D = local_out.shape
        n_refine = n_refine or self.n_refine_layers

        # Add positional encoding within segment
        pos_enc = self.seg_pos_enc[:, :W, :]
        segments = local_out + pos_enc

        # Select concept queries
        if route_name == 'SKIM':
            queries = self.concept_queries_skim    # (1, D)
        elif route_name == 'PROCESS':
            queries = self.concept_queries_process  # (4, D)
        elif route_name == 'FOCUS':
            queries = self.concept_queries_focus    # (16, D)
        else:
            raise ValueError(f"Unknown route: {route_name}")

        K = queries.shape[0]

        # Expand queries: (B, S, K, D)
        concepts = queries.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)

        # Refine through cross-attention
        for layer in self.refine_layers[:n_refine]:
            concepts, _ = layer(
                queries=concepts,       # (B, S, K, D)
                keys_values=segments,   # (B, S, W, D)
            )

        return concepts  # (B, S, K, D)

    def forward(
        self,
        local_out: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adaptive compression: blend or select path based on route weights.

        During training (soft weights): compute all paths, blend by weight.
        During inference (hard weights): compute only the selected path.

        Args:
            local_out: (B, S, W, D) - locally-encoded segments
            route_weights: (B, S, 3) - route selection weights [SKIM, PROCESS, FOCUS]

        Returns:
            concepts: (B, S, K_max, D) - compressed concepts padded to K_max
        """
        B, S, W, D = local_out.shape
        K_max = self.k_max

        if not self.training:
            return self._forward_hard(local_out, route_weights)

        # Soft routing: compute all paths and blend
        # All paths now produce concepts (no pass-through)
        focus_out = self.compress_segment(local_out, 'FOCUS')      # (B, S, 16, D)

        process_out = self.compress_segment(local_out, 'PROCESS')  # (B, S, 4, D)
        process_padded = F.pad(process_out, (0, 0, 0, K_max - self.N_CONCEPTS_PROCESS))

        skim_out = self.compress_segment(local_out, 'SKIM')        # (B, S, 1, D)
        skim_padded = F.pad(skim_out, (0, 0, 0, K_max - self.N_CONCEPTS_SKIM))

        # Blend by route weights
        w_skim = route_weights[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        w_process = route_weights[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        w_focus = route_weights[:, :, 2].unsqueeze(-1).unsqueeze(-1)

        blended = w_skim * skim_padded + w_process * process_padded + w_focus * focus_out
        return blended  # (B, S, K_max, D)

    def _forward_hard(
        self,
        local_out: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Hard routing at inference: compute only the selected path."""
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
