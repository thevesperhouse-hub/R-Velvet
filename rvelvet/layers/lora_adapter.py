"""
LoRA Adapter for Iterative Reasoning.

Low-rank adapters applied to QKV projection of GlobalSelfAttention.
Each iteration gets its own set of adapters (one per layer), enabling
different behavior at each reasoning step.

Zero-init up_proj ensures model starts identical to base.
~786K total params for 8 iterations × 8 layers × rank 8.
"""

import torch
import torch.nn as nn
import math


class LoRAAdapter(nn.Module):
    """
    Low-rank adapter for QKV projection.
    Adds delta: qkv_delta = up_proj(down_proj(x)) * scaling
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

        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_proj(self.down_proj(x)) * self.scaling


class IterationLoRABank(nn.Module):
    """
    Collection of LoRA adapters indexed by (iteration, layer).
    8 iterations × 8 layers × rank 8 = ~786K params.
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
        d_out = 3 * d_model

        self.adapters = nn.ModuleList([
            nn.ModuleList([
                LoRAAdapter(d_model, d_out, rank, alpha)
                for _ in range(n_layers)
            ])
            for _ in range(max_iterations)
        ])

    def get_adapter(self, iteration: int, layer: int) -> LoRAAdapter:
        return self.adapters[iteration][layer]

    def get_qkv_delta_fn(self, iteration: int, layer: int):
        return self.get_adapter(iteration, layer)
