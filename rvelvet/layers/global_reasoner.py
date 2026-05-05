"""
Full self-attention over concept vectors. Quadratic complexity is tractable due to compression.

Uses torch.nn.functional.scaled_dot_product_attention for the attention core
(Flash / mem-efficient backends auto-selected on supported GPUs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ._norm import RMSNorm


class GlobalReasoner(nn.Module):

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 8,
        ffn_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            GlobalBlock(d_model, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        self.relevance_head = nn.Linear(d_model, 1, bias=False)
        # Toggled by Trainer when cfg.training.gradient_checkpointing=True.
        self.gradient_checkpointing = False

    def forward(
        self,
        concepts: torch.Tensor,
        causal: bool = False,
        padding_mask: torch.Tensor = None,
        qkv_delta_fns: list = None,
    ) -> dict:
        for i, layer in enumerate(self.layers):
            qkv_delta_fn = qkv_delta_fns[i] if qkv_delta_fns is not None else None
            if self.gradient_checkpointing and self.training and qkv_delta_fn is None:
                concepts = torch.utils.checkpoint.checkpoint(
                    layer, concepts, causal, padding_mask, None,
                    use_reentrant=False,
                )
            else:
                concepts = layer(concepts, causal=causal, padding_mask=padding_mask, qkv_delta_fn=qkv_delta_fn)

        concepts = self.norm(concepts)

        relevance = self.relevance_head(concepts).squeeze(-1)
        relevance = torch.sigmoid(relevance)

        return {
            'concepts': concepts,
            'relevance': relevance,
        }


class GlobalBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_mult: float,
        dropout: float,
    ):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attn = GlobalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, int(d_model * ffn_mult), dropout)

    def forward(self, x: torch.Tensor, causal: bool = False, padding_mask: torch.Tensor = None, qkv_delta_fn=None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=causal, padding_mask=padding_mask, qkv_delta_fn=qkv_delta_fn)
        x = x + self.ffn(self.norm2(x))
        return x


class GlobalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor, causal: bool = False, padding_mask: torch.Tensor = None, qkv_delta_fn=None) -> torch.Tensor:
        B, N, D = x.shape
        H = self.n_heads
        hd = self.head_dim

        qkv = self.qkv(x)
        if qkv_delta_fn is not None:
            qkv = qkv + qkv_delta_fn(x)
        qkv = qkv.view(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Build attn_mask only when there's padding (otherwise rely on is_causal
        # fast path which dispatches to the fused causal kernel).
        attn_mask = None
        if padding_mask is not None:
            # padding_mask: (B, N) bool, True = ignore. Convert to a key mask
            # (B, 1, 1, N) where True = allowed. SDPA with bool mask uses
            # True=keep semantics.
            keep = (~padding_mask).view(B, 1, 1, N)
            if causal:
                # Combine causal + padding into a single bool mask. Causal:
                # (1,1,N,N) with True on/below diagonal.
                causal_mask = torch.ones(N, N, device=x.device, dtype=torch.bool).tril()
                attn_mask = causal_mask.view(1, 1, N, N) & keep
                use_is_causal = False
            else:
                attn_mask = keep
                use_is_causal = False
        else:
            use_is_causal = causal

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=use_is_causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
