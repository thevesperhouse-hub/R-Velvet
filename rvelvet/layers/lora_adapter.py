"""
LoRA Adapter for Iterative Reasoning.

Low-rank adapters applied to the QKV projection of GlobalSelfAttention.
Each iteration gets its own set of adapters (one per layer), so the
shared global reasoner behaves differently at each reasoning step.

Key design:
- Zero-init up_proj → model starts identical to base (no disruption)
- Kaiming init down_proj → good gradient flow from the start
- ~786K total params for 8 iterations × 8 layers × rank 8
"""

import torch
import torch.nn as nn
import math


class LoRAAdapter(nn.Module):
    """
    Low-rank adapter for QKV projection.

    Adds a delta to the QKV output: qkv_delta = up_proj(down_proj(x)) * scaling

    Args:
        d_model: Input dimension
        d_out: Output dimension (3 * d_model for fused QKV)
        rank: Low-rank bottleneck dimension
        alpha: Scaling factor (effective scale = alpha / rank)
    """

    def __init__(
        self,
        d_model: int,
        d_out: int,
        rank: int = 8,
        alpha: float = 8.0,
    ):
        super().__init__()

        self.rank = rank
        self.scaling = alpha / rank

        self.down_proj = nn.Linear(d_model, rank, bias=False)
        self.up_proj = nn.Linear(rank, d_out, bias=False)

        # Init: Kaiming for down, zeros for up → output starts at zero
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)

        Returns:
            delta: (B, N, d_out) — additive correction to QKV
        """
        return self.up_proj(self.down_proj(x)) * self.scaling


class IterationLoRABank(nn.Module):
    """
    Collection of LoRA adapters indexed by (iteration, layer).

    For max_iterations=8, n_layers=8, rank=8:
        8 × 8 × (384×8 + 8×1152) = ~786K params

    Args:
        d_model: Hidden dimension
        n_layers: Number of global reasoner layers
        max_iterations: Maximum reasoning iterations
        rank: LoRA rank
        alpha: LoRA alpha scaling
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        max_iterations: int = 8,
        rank: int = 8,
        alpha: float = 8.0,
    ):
        super().__init__()

        self.max_iterations = max_iterations
        self.n_layers = n_layers
        d_out = 3 * d_model  # Fused QKV

        # adapters[iter_idx][layer_idx] = LoRAAdapter
        self.adapters = nn.ModuleList([
            nn.ModuleList([
                LoRAAdapter(d_model, d_out, rank, alpha)
                for _ in range(n_layers)
            ])
            for _ in range(max_iterations)
        ])

    def get_adapter(self, iteration: int, layer: int) -> LoRAAdapter:
        """Get the adapter for a specific iteration and layer."""
        return self.adapters[iteration][layer]

    def get_qkv_delta_fn(self, iteration: int, layer: int):
        """
        Return a callable that computes qkv_delta for a given (iteration, layer).

        Usage in GlobalSelfAttention:
            qkv = self.qkv(x)
            if qkv_delta_fn is not None:
                qkv = qkv + qkv_delta_fn(x)
        """
        adapter = self.get_adapter(iteration, layer)
        return adapter
