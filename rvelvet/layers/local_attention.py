"""
Windowed self-attention with O(w²) complexity per window, O(n*w) total.

Uses torch.nn.functional.scaled_dot_product_attention (Flash/mem-efficient
backends auto-selected) for the attention core.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ._norm import RMSNorm


class LocalAttention(nn.Module):
    """
    Splits input into non-overlapping windows and applies multi-head attention within each window.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        window_size: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, L, D = x.shape
        W = self.window_size

        pad_len = (W - L % W) % W
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        _, L_padded, _ = x.shape
        n_windows = L_padded // W

        x = x.view(B, n_windows, W, D)

        qkv = self.qkv(x)
        # (B, n_windows, W, 3, H, hd) -> (3, B, H, n_windows, W, hd)
        qkv = qkv.view(B, n_windows, W, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv.unbind(0)

        # SDPA broadcasts over leading dims; treats last two as (L, D).
        # Per-window attention => batch axes are (B, H, n_windows), seq=W.
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        # (B, H, n_windows, W, hd) -> (B, n_windows, W, H, hd) -> (B, n_windows, W, D)
        out = out.permute(0, 2, 3, 1, 4).contiguous()
        out = out.view(B, n_windows, W, D)
        out = self.out_proj(out)
        out = out.view(B, L_padded, D)

        if pad_len > 0:
            out = out[:, :L, :]

        return out


class LocalTransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        window_size: int = 512,
        ffn_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attn = LocalAttention(d_model, n_heads, window_size, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, int(d_model * ffn_mult), dropout)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=causal)
        x = x + self.ffn(self.norm2(x))
        return x


class LocalEncoder(nn.Module):

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        window_size: int = 512,
        ffn_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            LocalTransformerBlock(d_model, n_heads, window_size, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        # Toggled by Trainer when cfg.training.gradient_checkpointing=True.
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, causal, use_reentrant=False,
                )
            else:
                x = layer(x, causal=causal)
        return self.norm(x)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
