"""
Global Reasoner: Attention on concepts, not tokens.

Takes compressed concept vectors and does full attention between them.
Since we went from 1M tokens to ~500 concepts, quadratic is fine here.

This is where high-level reasoning happens:
- "Does paragraph 3 contradict paragraph 47?"
- "The answer to this question is somewhere in section 12"
- "These two distant concepts are related"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GlobalReasoner(nn.Module):
    """
    Full attention over concept vectors.

    1M tokens → 512 concepts → full attention O(512²) = trivial.

    Also has the ability to "expand" back: when the global reasoner
    determines a concept is relevant, it can signal for decompression.

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of self-attention layers
        ffn_mult: FFN hidden multiplier
        dropout: Dropout rate
    """

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

        # Relevance head: predicts which concepts need expansion
        self.relevance_head = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        concepts: torch.Tensor,
        causal: bool = False,
        padding_mask: torch.Tensor = None,
        qkv_delta_fns: list = None,
    ) -> dict:
        """
        Args:
            concepts: (B, N, D) - concept vectors
                      N = number of concepts (e.g., 512 for 1M tokens / 2048 window)
            causal: Whether to apply causal mask
            padding_mask: (B, N) bool - True for positions to IGNORE (padded concepts)
            qkv_delta_fns: optional list of callables, one per layer (for LoRA injection)

        Returns:
            dict with:
                'concepts': (B, N, D) - refined concepts
                'relevance': (B, N) - relevance scores per concept
        """
        for i, layer in enumerate(self.layers):
            qkv_delta_fn = qkv_delta_fns[i] if qkv_delta_fns is not None else None
            concepts = layer(concepts, causal=causal, padding_mask=padding_mask, qkv_delta_fn=qkv_delta_fn)

        concepts = self.norm(concepts)

        # Compute relevance scores (which concepts need expansion)
        relevance = self.relevance_head(concepts).squeeze(-1)  # (B, N)
        relevance = torch.sigmoid(relevance)

        return {
            'concepts': concepts,
            'relevance': relevance,
        }


class GlobalBlock(nn.Module):
    """Standard transformer block for global reasoning."""

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
    """Standard multi-head self-attention (quadratic, but on few concepts)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = False, padding_mask: torch.Tensor = None, qkv_delta_fn=None) -> torch.Tensor:
        B, N, D = x.shape
        H = self.n_heads
        hd = self.head_dim

        qkv = self.qkv(x)
        if qkv_delta_fn is not None:
            qkv = qkv + qkv_delta_fn(x)
        qkv = qkv.view(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, N, hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        if causal:
            mask = torch.triu(
                torch.ones(N, N, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn = attn.masked_fill(mask, float('-inf'))

        # Mask padded concepts: prevent attending TO them (as keys)
        if padding_mask is not None:
            # padding_mask: (B, N) True = ignore
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2),  # (B, 1, 1, N)
                float('-inf'),
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, H, N, hd)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
