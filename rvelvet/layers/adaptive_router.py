"""
Adaptive Computation Routing (ACR) for R-Velvet.

Routes each segment to one of three computation paths:
- SKIM: minimal compute for trivial content (2 local, 2 global layers, max compression)
- PROCESS: normal compute for standard content (4 local, 6 global, moderate compression)
- FOCUS: full compute for critical content (6 local, 8 global, no compression)

The router uses Gumbel-softmax during training (differentiable) and hard argmax at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SegmentScanner(nn.Module):
    """
    Lightweight scanner that decides how much computation each segment needs.

    Uses a learned query to pool each segment into a summary vector, then
    predicts routing logits and write priority.

    ~370K params for d_model=384 (0.7% overhead).

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads for query pooling
        segment_size: Tokens per segment
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

        # Learned query: single vector that pools each segment
        self.scan_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # Projections for query-pooling cross-attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Route head: predict (SKIM, PROCESS, FOCUS) + write_priority
        hidden_dim = d_model // 2  # 192 for d_model=384
        self.route_norm = RMSNorm(d_model)
        self.route_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4, bias=False),  # 3 route logits + 1 write priority
        )

    def forward(
        self,
        segments: torch.Tensor,
    ) -> dict:
        """
        Scan segments and predict routing decisions.

        Args:
            segments: (B, S, W, D) - token embeddings split into segments

        Returns:
            dict with:
                'route_logits': (B, S, 3) - raw logits for route selection
                'write_priority': (B, S) - priority signal for memory gating
                'segment_summary': (B, S, D) - pooled summary per segment
        """
        B, S, W, D = segments.shape
        H = self.n_heads
        hd = self.head_dim

        # Expand learned query for all batch elements and segments
        query = self.scan_query.unsqueeze(0).expand(B, S, -1)  # (B, S, D)

        # Project: query from scan_query, keys/values from segment tokens
        q = self.q_proj(query).view(B, S, 1, H, hd).permute(0, 1, 3, 2, 4)  # (B, S, H, 1, hd)
        k = self.k_proj(segments).view(B, S, W, H, hd).permute(0, 1, 3, 2, 4)  # (B, S, H, W, hd)
        v = self.v_proj(segments).view(B, S, W, H, hd).permute(0, 1, 3, 2, 4)  # (B, S, H, W, hd)

        # Cross-attention: query attends to all tokens in segment
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, S, H, 1, W)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # (B, S, H, 1, hd)
        out = out.squeeze(3)  # (B, S, H, hd)
        out = out.reshape(B, S, D)
        summary = self.out_proj(out)  # (B, S, D)

        # Route prediction from summary
        route_out = self.route_head(self.route_norm(summary))  # (B, S, 4)
        route_logits = route_out[:, :, :3]  # (B, S, 3)
        write_priority = torch.sigmoid(route_out[:, :, 3])  # (B, S)

        return {
            'route_logits': route_logits,
            'write_priority': write_priority,
            'segment_summary': summary,
        }


class AdaptiveRouter(nn.Module):
    """
    Routes segments using Gumbel-softmax (training) or hard argmax (inference).

    No learnable parameters - just the routing mechanism.

    Args:
        tau_start: Initial Gumbel temperature
        tau_end: Final Gumbel temperature
        tau_anneal_steps: Steps to anneal from start to end
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
        """Current temperature based on annealing schedule."""
        progress = min(self.step.item() / max(self.tau_anneal_steps, 1), 1.0)
        return self.tau_start + (self.tau_end - self.tau_start) * progress

    def forward(self, route_logits: torch.Tensor) -> torch.Tensor:
        """
        Route segments.

        Args:
            route_logits: (B, S, 3) - raw route logits

        Returns:
            route_weights: (B, S, 3) - soft (training) or hard (inference) route weights
        """
        if self.training:
            # Gumbel-softmax: straight-through estimator
            route_weights = F.gumbel_softmax(route_logits, tau=self.tau, hard=True)
            # Increment step counter
            self.step += 1
        else:
            # Hard routing at inference
            idx = route_logits.argmax(dim=-1)  # (B, S)
            route_weights = F.one_hot(idx, num_classes=3).float()

        return route_weights


# Route configurations
# (local_layers, global_layers, compression)
ROUTE_CONFIGS = {
    'SKIM': {'local_layers': 2, 'global_layers': 2, 'n_concepts': 1},
    'PROCESS': {'local_layers': 4, 'global_layers': 6, 'n_concepts': 4},
    'FOCUS': {'local_layers': 6, 'global_layers': 8, 'n_concepts': 16},
}

# Compute cost weights for each route (for loss)
ROUTE_COSTS = torch.tensor([0.1, 0.5, 1.0])

# Target distribution: 60% SKIM, 30% PROCESS, 10% FOCUS
TARGET_DISTRIBUTION = torch.tensor([0.6, 0.3, 0.1])


def compute_layer_gates(route_weights: torch.Tensor, n_layers: int) -> torch.Tensor:
    """
    Compute per-layer gates from route weights for residual gating.

    The idea: all routes share the same layers, but later layers are gated.
    - Layers 0-1: always active (SKIM + PROCESS + FOCUS)
    - Layers 2-3: PROCESS + FOCUS only
    - Layers 4+: FOCUS only

    Args:
        route_weights: (B, S, 3) - route selection weights [SKIM, PROCESS, FOCUS]
        n_layers: total number of layers

    Returns:
        gates: (B, S, n_layers) - gate value per layer per segment
    """
    B, S, _ = route_weights.shape
    device = route_weights.device

    gates = torch.zeros(B, S, n_layers, device=device)

    # Boundary: first 1/3 always on, middle 1/3 for PROCESS+FOCUS, last 1/3 FOCUS only
    boundary_low = n_layers // 3       # e.g., 2 for 6 layers
    boundary_high = 2 * n_layers // 3  # e.g., 4 for 6 layers

    # Layers 0 to boundary_low-1: all routes
    gates[:, :, :boundary_low] = 1.0

    # Layers boundary_low to boundary_high-1: PROCESS + FOCUS
    process_focus_gate = route_weights[:, :, 1] + route_weights[:, :, 2]  # (B, S)
    gates[:, :, boundary_low:boundary_high] = process_focus_gate.unsqueeze(-1)

    # Layers boundary_high+: FOCUS only
    focus_gate = route_weights[:, :, 2]  # (B, S)
    gates[:, :, boundary_high:] = focus_gate.unsqueeze(-1)

    return gates


def compute_acr_losses(
    route_weights: torch.Tensor,
    route_logits: torch.Tensor,
) -> dict:
    """
    Compute ACR auxiliary losses.

    Args:
        route_weights: (B, S, 3) - route selection weights
        route_logits: (B, S, 3) - raw route logits

    Returns:
        dict with:
            'load_balance': scalar - KL divergence from target distribution
            'entropy': scalar - mean entropy per segment (minimize for sharp decisions)
            'compute_cost': scalar - weighted cost of chosen routes
    """
    device = route_weights.device

    # 1. Load balance loss: KL(actual_distribution || target_distribution)
    # Average route weights across batch and segments
    avg_route = route_weights.mean(dim=(0, 1))  # (3,)
    target = TARGET_DISTRIBUTION.to(device)
    # KL divergence (target as reference)
    avg_route_clamped = avg_route.clamp(min=1e-8)
    target_clamped = target.clamp(min=1e-8)
    load_balance = F.kl_div(
        avg_route_clamped.log(), target_clamped, reduction='sum'
    )

    # 2. Entropy loss: encourage sharp decisions
    probs = F.softmax(route_logits, dim=-1)  # (B, S, 3)
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()

    # 3. Compute cost: penalize expensive routes
    costs = ROUTE_COSTS.to(device)  # (3,)
    compute_cost = (route_weights * costs.unsqueeze(0).unsqueeze(0)).sum(dim=-1).mean()

    return {
        'load_balance': load_balance,
        'entropy': entropy,
        'compute_cost': compute_cost,
    }


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
