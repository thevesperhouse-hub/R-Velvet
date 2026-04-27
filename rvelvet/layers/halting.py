"""
Halting Unit for Iterative Reasoning (PonderNet-style).

Predicts p(halt) at each iteration. Training runs all iterations with outputs
weighted by halting distribution. Inference exits early when p(halt) > threshold.

Halting loss is KL divergence against Geometric prior (λ_p=0.5), encouraging
~2 iterations on average. ~37K params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class HaltingUnit(nn.Module):
    """
    Predicts halting probability from concept representations.
    Pipeline: mean_pool(concepts) → RMSNorm → Linear → SiLU → Linear → Sigmoid
    Last linear init at zeros gives sigmoid(0) = 0.5 at start.
    """

    def __init__(self, d_model: int = 384, hidden_dim: int = 96):
        super().__init__()

        self.norm = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        nn.init.zeros_(self.mlp[2].weight)

    def forward(self, concepts: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            concepts: (B, N, D)
            padding_mask: (B, N) bool, True for positions to ignore

        Returns:
            p_halt: (B,)
        """
        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            n_valid = valid_mask.sum(dim=1).clamp(min=1.0)
            pooled = (concepts * valid_mask).sum(dim=1) / n_valid
        else:
            pooled = concepts.mean(dim=1)

        logit = self.mlp(self.norm(pooled))
        p_halt = torch.sigmoid(logit.squeeze(-1))

        return p_halt


def compute_halting_loss(p_halts: list, lambda_p: float = 0.5) -> torch.Tensor:
    """
    PonderNet halting loss: KL(halt_distribution || Geometric(λ_p)).
    Converts conditional p(halt|not halted yet) into proper distribution,
    then computes KL against Geometric prior. λ_p=0.5 expects ~2 iterations.

    Returns:
        loss: scalar
    """
    N = len(p_halts)
    B = p_halts[0].shape[0]
    device = p_halts[0].device
    eps = 1e-8

    halt_dist = []
    remaining = torch.ones(B, device=device)

    for i in range(N):
        p_i = p_halts[i].clamp(eps, 1.0 - eps)
        halt_prob = remaining * p_i
        halt_dist.append(halt_prob)
        remaining = remaining * (1.0 - p_i)

    halt_dist[-1] = halt_dist[-1] + remaining

    halt_dist = torch.stack(halt_dist, dim=1)
    halt_dist = halt_dist.clamp(min=eps)
    halt_dist = halt_dist / halt_dist.sum(dim=1, keepdim=True)

    geometric = torch.zeros(N, device=device)
    for i in range(N):
        geometric[i] = lambda_p * ((1.0 - lambda_p) ** i)
    geometric = geometric / geometric.sum()
    geometric = geometric.unsqueeze(0).expand(B, -1)

    kl = (halt_dist * (halt_dist.log() - geometric.log())).sum(dim=1)

    return kl.mean()
