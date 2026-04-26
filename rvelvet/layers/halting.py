"""
Halting Unit for Iterative Reasoning (PonderNet-style).

Predicts p(halt) at each iteration. During training, all iterations run
and outputs are weighted by the halting distribution. During inference,
early exit when p(halt) > threshold.

The halting loss is KL divergence against a Geometric prior, encouraging
the model to halt in ~2 iterations on average (λ_p=0.5).

~37K params: RMSNorm(384) + Linear(384,96) + Linear(96,1)
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

    Init: last linear at zeros → sigmoid(0) = 0.5 at start (neutral).

    Args:
        d_model: Hidden dimension of concepts
        hidden_dim: Internal MLP hidden dimension
    """

    def __init__(self, d_model: int = 384, hidden_dim: int = 96):
        super().__init__()

        self.norm = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        # Init last linear to zeros → sigmoid(0) = 0.5
        nn.init.zeros_(self.mlp[2].weight)

    def forward(
        self,
        concepts: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            concepts: (B, N, D) - concept vectors at current iteration
            padding_mask: (B, N) bool - True for positions to IGNORE

        Returns:
            p_halt: (B,) - halting probability per batch element, in [0, 1]
        """
        # Mean-pool concepts (masked if needed)
        if padding_mask is not None:
            # Mask out padded positions
            valid_mask = (~padding_mask).unsqueeze(-1).float()  # (B, N, 1)
            n_valid = valid_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
            pooled = (concepts * valid_mask).sum(dim=1) / n_valid  # (B, D)
        else:
            pooled = concepts.mean(dim=1)  # (B, D)

        # Predict halting probability
        logit = self.mlp(self.norm(pooled))  # (B, 1)
        p_halt = torch.sigmoid(logit.squeeze(-1))  # (B,)

        return p_halt


def compute_halting_loss(
    p_halts: list,
    lambda_p: float = 0.5,
) -> torch.Tensor:
    """
    PonderNet halting loss: KL(halt_distribution || Geometric(λ_p)).

    Converts per-iteration conditional p(halt|not halted yet) into a proper
    distribution over iterations, then computes KL against a Geometric prior.

    The Geometric prior with λ_p=0.5 expects ~2 iterations on average.

    Args:
        p_halts: list of (B,) tensors, one per iteration — conditional p(halt)
        lambda_p: Geometric distribution parameter (prior)

    Returns:
        loss: scalar — mean KL divergence across batch
    """
    N = len(p_halts)
    B = p_halts[0].shape[0]
    device = p_halts[0].device
    eps = 1e-8

    # Convert conditional halting probs to joint distribution
    # p(halt at step i) = p_halt[i] * prod_{j<i} (1 - p_halt[j])
    halt_dist = []
    remaining = torch.ones(B, device=device)

    for i in range(N):
        p_i = p_halts[i].clamp(eps, 1.0 - eps)
        halt_prob = remaining * p_i
        halt_dist.append(halt_prob)
        remaining = remaining * (1.0 - p_i)

    # Assign remaining probability mass to last iteration (truncation)
    halt_dist[-1] = halt_dist[-1] + remaining

    halt_dist = torch.stack(halt_dist, dim=1)  # (B, N)
    # Clamp to avoid log(0)
    halt_dist = halt_dist.clamp(min=eps)
    halt_dist = halt_dist / halt_dist.sum(dim=1, keepdim=True)  # renormalize

    # Geometric prior: p(halt at step i) = λ_p * (1-λ_p)^i, truncated at N
    geometric = torch.zeros(N, device=device)
    for i in range(N):
        geometric[i] = lambda_p * ((1.0 - lambda_p) ** i)
    # Truncate and renormalize
    geometric = geometric / geometric.sum()
    geometric = geometric.unsqueeze(0).expand(B, -1)  # (B, N)

    # KL(halt_dist || geometric)
    kl = (halt_dist * (halt_dist.log() - geometric.log())).sum(dim=1)  # (B,)

    return kl.mean()
