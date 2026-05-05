"""
Adaptive Computation Routing (ACR) for R-Velvet.

Routes each segment to one of three computation paths:
SKIM (2 local, 2 global layers, max compression), PROCESS (4 local, 6 global, moderate compression),
or FOCUS (6 local, 8 global, no compression).

Uses Gumbel-softmax during training (differentiable) and hard argmax at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ._norm import RMSNorm


class SegmentScanner(nn.Module):
    """
    Lightweight scanner that decides how much computation each segment needs.

    Uses a learned query to pool each segment into a summary vector, then
    predicts routing logits and write priority. ~370K params for d_model=384.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        segment_size: int = 512,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.segment_size = segment_size
        self.scale = self.head_dim ** -0.5

        self.scan_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        hidden_dim = d_model // 2
        self.route_norm = RMSNorm(d_model)
        self.route_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4, bias=False),
        )

    def forward(self, segments: torch.Tensor) -> dict:
        """
        Args:
            segments: (B, S, W, D)

        Returns:
            dict with 'route_logits' (B, S, 3), 'write_priority' (B, S), 'segment_summary' (B, S, D)
        """
        B, S, W, D = segments.shape
        H = self.n_heads
        hd = self.head_dim

        query = self.scan_query.unsqueeze(0).expand(B, S, -1)

        q = self.q_proj(query).view(B, S, 1, H, hd).permute(0, 1, 3, 2, 4)
        k = self.k_proj(segments).view(B, S, W, H, hd).permute(0, 1, 3, 2, 4)
        v = self.v_proj(segments).view(B, S, W, H, hd).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.squeeze(3)
        out = out.reshape(B, S, D)
        summary = self.out_proj(out)

        route_out = self.route_head(self.route_norm(summary))
        route_logits = route_out[:, :, :3]
        write_priority = torch.sigmoid(route_out[:, :, 3])

        return {
            'route_logits': route_logits,
            'write_priority': write_priority,
            'segment_summary': summary,
        }


class AdaptiveRouter(nn.Module):
    """
    Routes segments using Gumbel-softmax (training) or hard argmax (inference).
    """

    def __init__(
        self,
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        tau_anneal_steps: int = 50000,
    ):
        super().__init__()

        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_anneal_steps = tau_anneal_steps
        self.register_buffer('step', torch.tensor(0, dtype=torch.long))

    @property
    def tau(self) -> float:
        progress = min(self.step.item() / max(self.tau_anneal_steps, 1), 1.0)
        return self.tau_start + (self.tau_end - self.tau_start) * progress

    def forward(self, route_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            route_logits: (B, S, 3)

        Returns:
            route_weights: (B, S, 3)
        """
        if self.training:
            route_weights = F.gumbel_softmax(route_logits, tau=self.tau, hard=True)
            self.step += 1
        else:
            idx = route_logits.argmax(dim=-1)
            route_weights = F.one_hot(idx, num_classes=3).float()

        return route_weights


ROUTE_CONFIGS = {
    'SKIM': {'local_layers': 2, 'global_layers': 2, 'n_concepts': 1},
    'PROCESS': {'local_layers': 4, 'global_layers': 6, 'n_concepts': 4},
    'FOCUS': {'local_layers': 6, 'global_layers': 8, 'n_concepts': 16},
}

ROUTE_COSTS = torch.tensor([0.1, 0.5, 1.0])
TARGET_DISTRIBUTION = torch.tensor([0.6, 0.3, 0.1])


def compute_layer_gates(route_weights: torch.Tensor, n_layers: int) -> torch.Tensor:
    """
    Compute per-layer gates from route weights for residual gating.
    All routes share layers, but later layers are gated based on route.

    Args:
        route_weights: (B, S, 3)
        n_layers: total layers

    Returns:
        gates: (B, S, n_layers)
    """
    B, S, _ = route_weights.shape
    device = route_weights.device

    gates = torch.zeros(B, S, n_layers, device=device)

    boundary_low = n_layers // 3
    boundary_high = 2 * n_layers // 3

    gates[:, :, :boundary_low] = 1.0

    process_focus_gate = route_weights[:, :, 1] + route_weights[:, :, 2]
    gates[:, :, boundary_low:boundary_high] = process_focus_gate.unsqueeze(-1)

    focus_gate = route_weights[:, :, 2]
    gates[:, :, boundary_high:] = focus_gate.unsqueeze(-1)

    return gates


def compute_acr_losses(route_weights: torch.Tensor, route_logits: torch.Tensor) -> dict:
    """
    Compute ACR auxiliary losses.

    Returns dict with 'load_balance', 'entropy', 'compute_cost'.
    """
    device = route_weights.device

    avg_route = route_weights.mean(dim=(0, 1))
    target = TARGET_DISTRIBUTION.to(device)
    avg_route_clamped = avg_route.clamp(min=1e-8)
    target_clamped = target.clamp(min=1e-8)
    load_balance = F.kl_div(avg_route_clamped.log(), target_clamped, reduction='sum')

    probs = F.softmax(route_logits, dim=-1)
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()

    costs = ROUTE_COSTS.to(device)
    compute_cost = (route_weights * costs.unsqueeze(0).unsqueeze(0)).sum(dim=-1).mean()

    return {
        'load_balance': load_balance,
        'entropy': entropy,
        'compute_cost': compute_cost,
    }


