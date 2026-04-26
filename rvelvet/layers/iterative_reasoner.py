"""
Iterative Reasoner: Multi-pass reasoning with shared weights + LoRA adapters.

Inspired by TRM (7M params beating billions), PonderNet, and COCONUT.

The core idea: instead of one pass through GlobalReasoner + MemoryController,
we loop N times with:
- Shared global reasoner weights (no duplication)
- Per-iteration LoRA adapters (different behavior each pass)
- Learned halting (PonderNet-style early exit at inference)
- COCONUT: output of iteration i becomes input of iteration i+1

Training: all iterations run, final output = weighted sum by halt distribution
Inference: early exit when p(halt) > threshold for all batch elements

~826K new params (1.6% overhead for 50M model).
"""

import torch
import torch.nn as nn

from .lora_adapter import IterationLoRABank
from .halting import HaltingUnit, compute_halting_loss


class IterativeReasoner(nn.Module):
    """
    Orchestrates iterative reasoning over concepts.

    Holds its own: LoRA bank, halting unit, iteration embeddings.
    References (not copies): global_reasoner, memory_controller.

    Args:
        global_reasoner: Shared GlobalReasoner module (by reference)
        memory_controller: Shared MemoryController module (by reference)
        d_model: Hidden dimension
        n_layers: Number of global reasoner layers
        max_iterations: Maximum reasoning iterations
        lora_rank: LoRA bottleneck rank
        halt_threshold: p(halt) threshold for early exit at inference
    """

    def __init__(
        self,
        global_reasoner: nn.Module,
        memory_controller: nn.Module,
        d_model: int = 384,
        n_layers: int = 8,
        max_iterations: int = 8,
        lora_rank: int = 8,
        halt_threshold: float = 0.5,
    ):
        super().__init__()

        self.global_reasoner = global_reasoner  # Reference, not copy
        self.memory_controller = memory_controller  # Reference, not copy
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_iterations = max_iterations
        self.halt_threshold = halt_threshold

        # Per-iteration LoRA adapters
        self.lora_bank = IterationLoRABank(
            d_model=d_model,
            n_layers=n_layers,
            max_iterations=max_iterations,
            rank=lora_rank,
        )

        # Halting unit
        self.halting_unit = HaltingUnit(d_model=d_model)

        # Iteration embeddings (added to concepts at each iteration)
        self.iteration_embed = nn.Parameter(
            torch.randn(max_iterations, d_model) * 0.02
        )

    def forward(
        self,
        concepts: torch.Tensor,
        memory: torch.Tensor = None,
        causal: bool = False,
        padding_mask: torch.Tensor = None,
    ) -> dict:
        """
        Iterative reasoning loop.

        Training: runs all iterations, returns weighted outputs.
        Inference: early exit when p(halt) > threshold.

        Args:
            concepts: (B, N, D) - flattened concept vectors
            memory: (B, M, D) - external memory state (None = fresh)
            causal: Whether to apply causal mask
            padding_mask: (B, N) bool - True for positions to IGNORE

        Returns:
            dict with:
                'concepts': (B, N, D) - final refined concepts
                'relevance': (B, N) - relevance scores
                'memory': (B, M, D) - updated memory
                'iteration_outputs': list of (B, N, D) per iteration
                'p_halts': list of (B,) per iteration
                'halt_distribution': (B, max_iterations) - normalized halt weights
                'n_iterations': int - actual iterations run
        """
        B, N, D = concepts.shape
        device = concepts.device

        # Initialize memory if needed
        if memory is None:
            memory = self.memory_controller.init_memory(B, device)

        iteration_outputs = []
        iteration_relevances = []
        p_halts = []
        h = concepts  # COCONUT: output feeds back as input

        for i in range(self.max_iterations):
            # 1. Add iteration embedding
            h_iter = h + self.iteration_embed[i].unsqueeze(0).unsqueeze(0)  # (B, N, D)

            # 2. Build LoRA delta functions for this iteration (one per layer)
            qkv_delta_fns = [
                self.lora_bank.get_qkv_delta_fn(i, j)
                for j in range(self.n_layers)
            ]

            # 3. Run through global reasoner layers manually (to inject LoRA per layer)
            for j, layer in enumerate(self.global_reasoner.layers):
                h_iter = layer(
                    h_iter, causal=causal,
                    padding_mask=padding_mask,
                    qkv_delta_fn=qkv_delta_fns[j],
                )

            # 4. Apply final norm + relevance head
            h_normed = self.global_reasoner.norm(h_iter)
            relevance = torch.sigmoid(
                self.global_reasoner.relevance_head(h_normed).squeeze(-1)
            )

            # 5. Memory controller (read/write, memory accumulates across iterations)
            mem_out = self.memory_controller(h_normed, memory)
            enriched = mem_out['enriched']  # (B, N, D)
            memory = mem_out['memory']  # (B, M, D) — updated

            # 6. Halting probability
            p_halt = self.halting_unit(enriched, padding_mask)  # (B,)
            p_halts.append(p_halt)

            # 7. Store intermediate output (for deep supervision)
            iteration_outputs.append(enriched)
            iteration_relevances.append(relevance)

            # 8. COCONUT: enriched output → input of next iteration
            h = enriched

            # 9. Inference early exit
            if not self.training:
                if (p_halt > self.halt_threshold).all():
                    break

        n_iterations = len(iteration_outputs)

        # Compute halt distribution from conditional p_halts
        halt_distribution = self._compute_halt_distribution(p_halts, device)

        if self.training:
            # Weighted sum of iteration outputs by halt distribution
            final_concepts = torch.zeros_like(iteration_outputs[0])
            final_relevance = torch.zeros_like(iteration_relevances[0])
            for i in range(n_iterations):
                w = halt_distribution[:, i].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                final_concepts = final_concepts + w * iteration_outputs[i]
                w_rel = halt_distribution[:, i].unsqueeze(-1)  # (B, 1)
                final_relevance = final_relevance + w_rel * iteration_relevances[i]
        else:
            # Use last iteration output
            final_concepts = iteration_outputs[-1]
            final_relevance = iteration_relevances[-1]

        return {
            'concepts': final_concepts,
            'relevance': final_relevance,
            'memory': memory,
            'iteration_outputs': iteration_outputs,
            'p_halts': p_halts,
            'halt_distribution': halt_distribution,
            'n_iterations': n_iterations,
        }

    def _compute_halt_distribution(
        self,
        p_halts: list,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert conditional p(halt) to a proper distribution over iterations.

        p(halt at i) = p_halt[i] * prod_{j<i}(1 - p_halt[j])
        Last iteration gets remaining mass (truncation).

        Returns:
            halt_dist: (B, max_iterations) — zeros for iterations not run
        """
        N = len(p_halts)
        B = p_halts[0].shape[0]
        eps = 1e-8

        halt_dist = torch.zeros(B, self.max_iterations, device=device)
        remaining = torch.ones(B, device=device)

        for i in range(N):
            p_i = p_halts[i].clamp(eps, 1.0 - eps)
            halt_prob = remaining * p_i
            halt_dist[:, i] = halt_prob
            remaining = remaining * (1.0 - p_i)

        # Assign remaining mass to last iteration that was actually run
        halt_dist[:, N - 1] = halt_dist[:, N - 1] + remaining

        # Renormalize
        halt_dist = halt_dist / halt_dist.sum(dim=1, keepdim=True).clamp(min=eps)

        return halt_dist
