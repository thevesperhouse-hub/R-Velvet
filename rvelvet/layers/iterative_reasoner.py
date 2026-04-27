"""
Iterative Reasoner: Multi-pass reasoning with shared weights + LoRA adapters.

Loops N times with shared global reasoner weights, per-iteration LoRA adapters,
learned halting (PonderNet-style), and COCONUT feedback (output i → input i+1).

Training runs all iterations, final output = weighted sum by halt distribution.
Inference exits early when p(halt) > threshold. ~826K new params (1.6% overhead).
"""

import torch
import torch.nn as nn

from .lora_adapter import IterationLoRABank
from .halting import HaltingUnit, compute_halting_loss


class IterativeReasoner(nn.Module):
    """
    Orchestrates iterative reasoning over concepts.
    Owns LoRA bank, halting unit, iteration embeddings.
    References global_reasoner and memory_controller.
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

        self.global_reasoner = global_reasoner
        self.memory_controller = memory_controller
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_iterations = max_iterations
        self.halt_threshold = halt_threshold

        self.lora_bank = IterationLoRABank(
            d_model=d_model,
            n_layers=n_layers,
            max_iterations=max_iterations,
            rank=lora_rank,
        )

        self.halting_unit = HaltingUnit(d_model=d_model)

        self.iteration_embed = nn.Parameter(torch.randn(max_iterations, d_model) * 0.02)

    def forward(
        self,
        concepts: torch.Tensor,
        memory: torch.Tensor = None,
        causal: bool = False,
        padding_mask: torch.Tensor = None,
    ) -> dict:
        """
        Iterative reasoning loop.
        Training runs all iterations with weighted outputs.
        Inference exits early when p(halt) > threshold.
        """
        B, N, D = concepts.shape
        device = concepts.device

        if memory is None:
            memory = self.memory_controller.init_memory(B, device)

        iteration_outputs = []
        iteration_relevances = []
        p_halts = []
        h = concepts

        for i in range(self.max_iterations):
            h_iter = h + self.iteration_embed[i].unsqueeze(0).unsqueeze(0)

            qkv_delta_fns = [
                self.lora_bank.get_qkv_delta_fn(i, j)
                for j in range(self.n_layers)
            ]

            for j, layer in enumerate(self.global_reasoner.layers):
                h_iter = layer(
                    h_iter, causal=causal,
                    padding_mask=padding_mask,
                    qkv_delta_fn=qkv_delta_fns[j],
                )

            h_normed = self.global_reasoner.norm(h_iter)
            relevance = torch.sigmoid(
                self.global_reasoner.relevance_head(h_normed).squeeze(-1)
            )

            mem_out = self.memory_controller(h_normed, memory)
            enriched = mem_out['enriched']
            memory = mem_out['memory']

            p_halt = self.halting_unit(enriched, padding_mask)
            p_halts.append(p_halt)

            iteration_outputs.append(enriched)
            iteration_relevances.append(relevance)

            h = enriched

            if not self.training:
                if (p_halt > self.halt_threshold).all():
                    break

        n_iterations = len(iteration_outputs)

        halt_distribution = self._compute_halt_distribution(p_halts, device)

        if self.training:
            final_concepts = torch.zeros_like(iteration_outputs[0])
            final_relevance = torch.zeros_like(iteration_relevances[0])
            for i in range(n_iterations):
                w = halt_distribution[:, i].unsqueeze(-1).unsqueeze(-1)
                final_concepts = final_concepts + w * iteration_outputs[i]
                w_rel = halt_distribution[:, i].unsqueeze(-1)
                final_relevance = final_relevance + w_rel * iteration_relevances[i]
        else:
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

    def _compute_halt_distribution(self, p_halts: list, device: torch.device) -> torch.Tensor:
        """
        Convert conditional p(halt) to proper distribution over iterations.
        Last iteration gets remaining mass (truncation).
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

        halt_dist[:, N - 1] = halt_dist[:, N - 1] + remaining

        halt_dist = halt_dist / halt_dist.sum(dim=1, keepdim=True).clamp(min=eps)

        return halt_dist
