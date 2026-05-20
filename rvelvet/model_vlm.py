"""
VelvetLM — Modern transformer (Gemma 3 / LLaMA 3 style).

Architecture: pre-norm RMSNorm, RoPE, SwiGLU, FlashAttention via SDPA.
Sliding window 5:1 ratio markers for future long-context inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute RoPE cos/sin tables."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def apply_rope(q, k, cos, sin):
    """Apply rotary position embeddings to q and k."""
    L = q.shape[2]
    cos = cos[:L].unsqueeze(0).unsqueeze(0)  # (1, 1, L, hd/2)
    sin = sin[:L].unsqueeze(0).unsqueeze(0)

    q1, q2 = q[..., :q.shape[-1] // 2], q[..., q.shape[-1] // 2:]
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)

    k1, k2 = k[..., :k.shape[-1] // 2], k[..., k.shape[-1] // 2:]
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

    return q_rot.type_as(q), k_rot.type_as(k)


class VelvetLM(nn.Module):
    """
    Modern transformer LM with RoPE, SwiGLU, FlashAttention.

    Layers are annotated as 'global' (1 in 6) or 'local' (5 in 6)
    following the Gemma 3 5:1 ratio. During training on short contexts
    all layers use full causal attention; the annotation enables
    sliding-window KV-cache optimisation at inference time.
    """

    def __init__(
        self,
        vocab_size: int = 100000,
        d_model: int = 2048,
        n_layers: int = 22,
        n_heads: int = 16,
        ffn_mult: float = 2.6875,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        window_size: int = 1024,
        global_every: int = 6,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        head_dim = d_model // n_heads

        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Annotate each layer as global or local (Gemma 3 pattern)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            is_global = ((i + 1) % global_every == 0) or (i == n_layers - 1)
            self.layers.append(
                TransformerBlock(d_model, n_heads, ffn_mult, dropout, is_global, window_size)
            )

        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # weight tying

        # RoPE buffers
        cos, sin = precompute_rope(head_dim, max_seq_len, rope_theta)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        residual_scale = 1.0 / math.sqrt(2 * self.n_layers)
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                if 'out_proj' in name or name.endswith('w2.weight'):
                    nn.init.xavier_normal_(p, gain=residual_scale)
                else:
                    nn.init.xavier_normal_(p, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(p)
        nn.init.normal_(self.token_embed.weight, std=0.02)

    def forward(self, input_ids, **kwargs):
        B, L = input_ids.shape
        x = self.token_embed(input_ids)

        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)

        x = self.norm(x)
        logits = self.lm_head(x)
        return {'logits': logits}

    def count_parameters(self):
        n_global = sum(1 for layer in self.layers if layer.is_global)
        n_local = len(self.layers) - n_global
        embed = self.token_embed.weight.numel()
        transformer = sum(p.numel() for layer in self.layers for p in layer.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            'token_embed': embed,
            f'transformer ({n_local}L+{n_global}G)': transformer,
            'output_norm': self.norm.weight.numel(),
            'total': total,
        }


class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads, ffn_mult, dropout, is_global, window_size):
        super().__init__()
        self.is_global = is_global
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        d_ff = int(d_model * ffn_mult)
        self.ffn = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.ffn(self.norm2(x))
        return x


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout

    def forward(self, x, rope_cos, rope_sin):
        B, L, D = x.shape
        H, hd = self.n_heads, self.head_dim

        qkv = self.qkv(x).view(B, L, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = apply_rope(q, k, rope_cos, rope_sin)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_drop if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class SwiGLU(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
