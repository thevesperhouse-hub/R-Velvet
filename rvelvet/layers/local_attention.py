"""
Local Attention: Standard quadratic attention on small windows.

Nothing fancy here - proven mechanism, works perfectly for local context.
Window size ~512-2048 tokens. O(w²) per window, O(n*w) total.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LocalAttention(nn.Module):
    """
    Windowed self-attention.

    Splits input into non-overlapping windows and applies
    standard multi-head attention within each window.

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        window_size: Size of each local window
        dropout: Attention dropout
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
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) - input sequence
            causal: Whether to use causal masking

        Returns:
            (B, L, D) - attended output
        """
        B, L, D = x.shape
        W = self.window_size

        # Pad to multiple of window_size
        pad_len = (W - L % W) % W
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        _, L_padded, _ = x.shape
        n_windows = L_padded // W

        # Reshape into windows: (B, n_windows, W, D)
        x = x.view(B, n_windows, W, D)

        # QKV projection
        qkv = self.qkv(x)  # (B, n_windows, W, 3*D)
        qkv = qkv.view(B, n_windows, W, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # (3, B, H, n_windows, W, head_dim)
        q, k, v = qkv.unbind(0)

        # Attention scores: (B, H, n_windows, W, W)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Causal mask (within each window)
        if causal:
            mask = torch.triu(
                torch.ones(W, W, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = attn @ v  # (B, H, n_windows, W, head_dim)
        out = out.permute(0, 2, 3, 1, 4).contiguous()  # (B, n_windows, W, H, head_dim)
        out = out.view(B, n_windows, W, D)

        # Project output
        out = self.out_proj(out)

        # Reshape back: (B, L_padded, D)
        out = out.view(B, L_padded, D)

        # Remove padding
        if pad_len > 0:
            out = out[:, :L, :]

        return out


class LocalTransformerBlock(nn.Module):
    """
    Local Transformer block: Local attention + FFN + RMSNorm

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        window_size: Local attention window
        ffn_mult: FFN hidden dim multiplier
        dropout: Dropout rate
    """

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
    """
    Stack of local transformer blocks.

    Processes input with windowed attention - captures local context.

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        window_size: Local attention window
        ffn_mult: FFN multiplier
        dropout: Dropout rate
    """

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

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, causal=causal)
        return self.norm(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class FFN(nn.Module):
    """SwiGLU Feed-Forward Network"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
