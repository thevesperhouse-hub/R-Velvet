"""Shared RMSNorm shim.

Uses torch.nn.RMSNorm (PyTorch 2.4+) when available — that path uses a fused
kernel on supported devices. Falls back to a manual implementation that matches
the same parameter layout (single `weight` of shape (d_model,)) so checkpoints
remain compatible across torch versions.
"""

import torch
import torch.nn as nn

# torch.nn.RMSNorm exists in 2.4+. Detect once at import.
_HAS_TORCH_RMSNORM = hasattr(nn, "RMSNorm")


if _HAS_TORCH_RMSNORM:
    class RMSNorm(nn.RMSNorm):
        """torch.nn.RMSNorm wrapper with the (d_model, eps) constructor used by R-Velvet."""

        def __init__(self, d_model: int, eps: float = 1e-6):
            super().__init__(normalized_shape=d_model, eps=eps, elementwise_affine=True)
else:
    class RMSNorm(nn.Module):
        """Manual fallback (parameter name `weight` matches torch.nn.RMSNorm)."""

        def __init__(self, d_model: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d_model))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * norm * self.weight


__all__ = ["RMSNorm"]
