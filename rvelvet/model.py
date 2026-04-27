"""
R-Velvet: Multi-Scale Architecture for Unlimited Context.

Pipeline: tokens → Local Encoder → Segment Compressor → Global Reasoner → Memory Controller → Output

Complexity: Local attention O(n*w²) for windows of size w, compression reduces n tokens to k concepts
where k << n, global reasoning O(k²) on concepts, external memory for long-range dependencies.
"""

import torch
import torch.nn as nn
import math

from .layers.local_attention import LocalEncoder
from .layers.segment_compressor import SegmentCompressor, AdaptiveSegmentCompressor
from .layers.global_reasoner import GlobalReasoner
from .layers.memory_controller import MemoryController
from .layers.adaptive_router import (
    SegmentScanner, AdaptiveRouter, compute_layer_gates, compute_acr_losses,
)
from .layers.iterative_reasoner import IterativeReasoner
from .layers.halting import compute_halting_loss
import torch.nn.functional as F


class RVelvet(nn.Module):
    """
    Multi-scale transformer: local windowed attention on tokens, learned compression to concepts,
    global reasoning on concepts, and external memory for long-range dependencies. Concepts are
    expanded back to token level for generation.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 384,
        n_local_layers: int = 6,
        n_global_layers: int = 8,
        n_local_heads: int = 6,
        n_global_heads: int = 8,
        window_size: int = 512,
        segment_size: int = 512,
        n_concepts: int = 1,
        n_refine_layers: int = 2,
        memory_size: int = 256,
        n_read_steps: int = 2,
        ffn_mult: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        use_acr: bool = False,
        use_iterative_reasoning: bool = False,
        max_reasoning_iterations: int = 8,
        lora_rank: int = 8,
        halt_threshold: float = 0.5,
        lambda_p: float = 0.5,
    ):
        super().__init__()

        self.d_model = d_model
        self.segment_size = segment_size
        self.n_concepts = n_concepts
        self.use_acr = use_acr
        self.use_iterative_reasoning = use_iterative_reasoning
        self.lambda_p = lambda_p
        self.n_local_layers = n_local_layers
        self.n_global_layers = n_global_layers

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(dropout)

        self.local_encoder = LocalEncoder(
            d_model=d_model,
            n_heads=n_local_heads,
            n_layers=n_local_layers,
            window_size=window_size,
            ffn_mult=ffn_mult,
            dropout=dropout,
        )

        self.compressor = SegmentCompressor(
            d_model=d_model,
            n_heads=n_local_heads,
            segment_size=segment_size,
            n_concepts=n_concepts,
            n_refine_layers=n_refine_layers,
            dropout=dropout,
        )

        self.global_reasoner = GlobalReasoner(
            d_model=d_model,
            n_heads=n_global_heads,
            n_layers=n_global_layers,
            ffn_mult=ffn_mult,
            dropout=dropout,
        )

        self.memory_controller = MemoryController(
            d_model=d_model,
            n_heads=n_local_heads,
            memory_size=memory_size,
            n_read_steps=n_read_steps,
            dropout=dropout,
        )
        if use_acr:
            self.scanner = SegmentScanner(
                d_model=d_model,
                n_heads=n_local_heads,
                segment_size=segment_size,
            )
            self.router = AdaptiveRouter()
            self.adaptive_compressor = AdaptiveSegmentCompressor(
                d_model=d_model,
                n_heads=n_local_heads,
                segment_size=segment_size,
                n_refine_layers=n_refine_layers,
                dropout=dropout,
            )

        if use_iterative_reasoning:
            self.iterative_reasoner = IterativeReasoner(
                global_reasoner=self.global_reasoner,
                memory_controller=self.memory_controller,
                d_model=d_model,
                n_layers=n_global_layers,
                max_iterations=max_reasoning_iterations,
                lora_rank=lora_rank,
                halt_threshold=halt_threshold,
            )

        self.expansion = ExpansionLayer(
            d_model=d_model,
            n_heads=n_local_heads,
            dropout=dropout,
        )

        self.out_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_normal_(p, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(p)

        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor = None,
        causal: bool = True,
    ) -> dict:
        B, L = input_ids.shape

        if self.use_acr:
            return self.forward_acr(input_ids, memory=memory, causal=causal)

        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_drop(x)

        local_out = self.local_encoder(x, causal=causal)

        concepts = self.compressor(local_out)
        B_c, n_seg, K, D = concepts.shape
        concepts_flat = concepts.view(B, n_seg * K, D)

        if self.use_iterative_reasoning:
            iter_out = self.iterative_reasoner(
                concepts_flat, memory=memory, causal=causal,
            )
            enriched_concepts = iter_out['concepts']
            relevance = iter_out['relevance']
            updated_memory = iter_out['memory']
        else:
            global_out = self.global_reasoner(concepts_flat, causal=causal)
            refined_concepts = global_out['concepts']
            relevance = global_out['relevance']

            mem_out = self.memory_controller(refined_concepts, memory)
            enriched_concepts = mem_out['enriched']
            updated_memory = mem_out['memory']

        expanded = self.expansion(local_out, enriched_concepts)

        out = self.out_norm(expanded)
        logits = self.lm_head(out)

        result = {
            'logits': logits,
            'concepts': enriched_concepts,
            'memory': updated_memory,
            'relevance': relevance,
        }

        if self.use_iterative_reasoning:
            result['iteration_outputs'] = iter_out['iteration_outputs']
            result['p_halts'] = iter_out['p_halts']
            result['halt_distribution'] = iter_out['halt_distribution']
            result['n_iterations'] = iter_out['n_iterations']
            result['local_out'] = local_out

        return result

    def forward_acr(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor = None,
        causal: bool = True,
    ) -> dict:
        B, L = input_ids.shape
        D = self.d_model
        W = self.segment_size
        device = input_ids.device

        if self.use_iterative_reasoning:
            return self._forward_acr_iterative(input_ids, memory=memory, causal=causal)

        positions = torch.arange(L, device=device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_drop(x)

        pad_len = (W - L % W) % W
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x
        L_padded = x_padded.shape[1]
        n_seg = L_padded // W
        segments = x_padded.view(B, n_seg, W, D)

        scan_out = self.scanner(segments)
        route_logits = scan_out['route_logits']
        write_priority = scan_out['write_priority']

        route_weights = self.router(route_logits)

        local_gates = compute_layer_gates(route_weights, self.n_local_layers)

        h_flat = segments.reshape(B * n_seg, W, D)

        if not self.training:
            for i, layer in enumerate(self.local_encoder.layers):
                gate_flat = local_gates[:, :, i].reshape(B * n_seg)
                active = gate_flat > 0.5
                if not active.any():
                    continue
                h_flat[active] = layer(h_flat[active], causal=causal)
        else:
            for i, layer in enumerate(self.local_encoder.layers):
                gate_flat = local_gates[:, :, i].reshape(B * n_seg, 1, 1)
                h_new = layer(h_flat, causal=causal)
                delta = h_new - h_flat
                h_flat = h_flat + gate_flat * delta

        h_flat = self.local_encoder.norm(h_flat)
        local_out = h_flat.view(B, n_seg, W, D)

        concepts_4d = self.adaptive_compressor(local_out, route_weights)

        K_max = concepts_4d.shape[2]
        concepts_flat = concepts_4d.view(B, n_seg * K_max, D)

        n_per_route = torch.tensor(
            [self.adaptive_compressor.N_CONCEPTS_SKIM,
             self.adaptive_compressor.N_CONCEPTS_PROCESS,
             self.adaptive_compressor.n_concepts_focus],
            device=device, dtype=torch.float,
        )
        concepts_per_seg = (route_weights * n_per_route).sum(dim=-1)
        pos_idx = torch.arange(K_max, device=device).float()
        concept_valid = pos_idx < concepts_per_seg.unsqueeze(-1)
        padding_mask = ~concept_valid.view(B, n_seg * K_max)

        global_gates = compute_layer_gates(route_weights, self.n_global_layers)

        global_gates_expanded = global_gates.unsqueeze(2).expand(
            B, n_seg, K_max, self.n_global_layers
        ).reshape(B, n_seg * K_max, self.n_global_layers)

        h_global = concepts_flat
        for i, layer in enumerate(self.global_reasoner.layers):
            gate = global_gates_expanded[:, :, i].unsqueeze(-1)
            h_new = layer(h_global, causal=causal, padding_mask=padding_mask)
            delta = h_new - h_global
            h_global = h_global + gate * delta

        h_global = self.global_reasoner.norm(h_global)
        h_global = h_global * (~padding_mask).unsqueeze(-1).float()

        relevance = self.global_reasoner.relevance_head(h_global).squeeze(-1)
        relevance = torch.sigmoid(relevance)

        refined_concepts = h_global

        write_priority_expanded = write_priority.unsqueeze(2).expand(
            B, n_seg, K_max
        ).reshape(B, n_seg * K_max)

        mem_out = self.memory_controller(
            refined_concepts, memory, write_priority=write_priority_expanded
        )
        enriched_concepts = mem_out['enriched']
        updated_memory = mem_out['memory']

        valid_mask = (~padding_mask).unsqueeze(-1).float()
        enriched_concepts = enriched_concepts * valid_mask

        local_flat = local_out.view(B, L_padded, D)
        if pad_len > 0:
            local_flat = local_flat[:, :L, :]
        expanded = self.expansion(local_flat, enriched_concepts)

        out = self.out_norm(expanded)
        logits = self.lm_head(out)

        return {
            'logits': logits,
            'concepts': enriched_concepts,
            'memory': updated_memory,
            'relevance': relevance,
            'route_weights': route_weights,
            'route_logits': route_logits,
            'write_priority': write_priority,
        }

    def _forward_acr_iterative(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor = None,
        causal: bool = True,
    ) -> dict:
        B, L = input_ids.shape
        D = self.d_model
        W = self.segment_size
        device = input_ids.device
        ir = self.iterative_reasoner

        positions = torch.arange(L, device=device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_drop(x)

        pad_len = (W - L % W) % W
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x
        L_padded = x_padded.shape[1]
        n_seg = L_padded // W
        segments = x_padded.view(B, n_seg, W, D)

        scan_out = self.scanner(segments)
        route_logits = scan_out['route_logits']
        write_priority = scan_out['write_priority']
        route_weights = self.router(route_logits)

        local_gates = compute_layer_gates(route_weights, self.n_local_layers)
        h_flat = segments.reshape(B * n_seg, W, D)

        if not self.training:
            for i, layer in enumerate(self.local_encoder.layers):
                gate_flat = local_gates[:, :, i].reshape(B * n_seg)
                active = gate_flat > 0.5
                if not active.any():
                    continue
                h_flat[active] = layer(h_flat[active], causal=causal)
        else:
            for i, layer in enumerate(self.local_encoder.layers):
                gate_flat = local_gates[:, :, i].reshape(B * n_seg, 1, 1)
                h_new = layer(h_flat, causal=causal)
                delta = h_new - h_flat
                h_flat = h_flat + gate_flat * delta

        h_flat = self.local_encoder.norm(h_flat)
        local_out = h_flat.view(B, n_seg, W, D)

        concepts_4d = self.adaptive_compressor(local_out, route_weights)
        K_max = concepts_4d.shape[2]
        concepts_flat = concepts_4d.view(B, n_seg * K_max, D)

        n_per_route = torch.tensor(
            [self.adaptive_compressor.N_CONCEPTS_SKIM,
             self.adaptive_compressor.N_CONCEPTS_PROCESS,
             self.adaptive_compressor.n_concepts_focus],
            device=device, dtype=torch.float,
        )
        concepts_per_seg = (route_weights * n_per_route).sum(dim=-1)
        pos_idx = torch.arange(K_max, device=device).float()
        concept_valid = pos_idx < concepts_per_seg.unsqueeze(-1)
        padding_mask = ~concept_valid.view(B, n_seg * K_max)

        global_gates = compute_layer_gates(route_weights, self.n_global_layers)
        global_gates_expanded = global_gates.unsqueeze(2).expand(
            B, n_seg, K_max, self.n_global_layers
        ).reshape(B, n_seg * K_max, self.n_global_layers)

        write_priority_expanded = write_priority.unsqueeze(2).expand(
            B, n_seg, K_max
        ).reshape(B, n_seg * K_max)

        if memory is None:
            memory = self.memory_controller.init_memory(B, device)

        iteration_outputs = []
        iteration_relevances = []
        p_halts = []
        h = concepts_flat

        for it in range(ir.max_iterations):
            h_iter = h + ir.iteration_embed[it].unsqueeze(0).unsqueeze(0)

            for j, layer in enumerate(self.global_reasoner.layers):
                gate = global_gates_expanded[:, :, j].unsqueeze(-1)
                qkv_delta_fn = ir.lora_bank.get_qkv_delta_fn(it, j)
                h_new = layer(
                    h_iter, causal=causal,
                    padding_mask=padding_mask,
                    qkv_delta_fn=qkv_delta_fn,
                )
                delta = h_new - h_iter
                h_iter = h_iter + gate * delta

            h_normed = self.global_reasoner.norm(h_iter)
            h_normed = h_normed * (~padding_mask).unsqueeze(-1).float()

            relevance = torch.sigmoid(
                self.global_reasoner.relevance_head(h_normed).squeeze(-1)
            )

            mem_out = self.memory_controller(
                h_normed, memory, write_priority=write_priority_expanded
            )
            enriched = mem_out['enriched']
            memory = mem_out['memory']

            valid_mask = (~padding_mask).unsqueeze(-1).float()
            enriched = enriched * valid_mask

            p_halt = ir.halting_unit(enriched, padding_mask)
            p_halts.append(p_halt)

            iteration_outputs.append(enriched)
            iteration_relevances.append(relevance)

            h = enriched

            if not self.training:
                if (p_halt > ir.halt_threshold).all():
                    break

        n_iterations = len(iteration_outputs)

        halt_distribution = ir._compute_halt_distribution(p_halts, device)

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

        local_flat = local_out.view(B, L_padded, D)
        if pad_len > 0:
            local_flat = local_flat[:, :L, :]
        expanded = self.expansion(local_flat, final_concepts)

        out = self.out_norm(expanded)
        logits = self.lm_head(out)

        return {
            'logits': logits,
            'concepts': final_concepts,
            'memory': memory,
            'relevance': final_relevance,
            'route_weights': route_weights,
            'route_logits': route_logits,
            'write_priority': write_priority,
            'iteration_outputs': iteration_outputs,
            'p_halts': p_halts,
            'halt_distribution': halt_distribution,
            'n_iterations': n_iterations,
            'local_out': local_flat,
        }

    def count_parameters(self) -> dict:
        counts = {}
        components = {
            'token_embed': [self.token_embed, self.pos_embed],
            'local_encoder': [self.local_encoder],
            'compressor': [self.compressor],
            'global_reasoner': [self.global_reasoner],
            'memory_controller': [self.memory_controller],
            'expansion': [self.expansion],
            'output': [self.out_norm],  # lm_head shares embed weights
        }
        if self.use_acr:
            components['scanner'] = [self.scanner]
            components['router'] = [self.router]
            components['adaptive_compressor'] = [self.adaptive_compressor]
        if self.use_iterative_reasoning:
            components['lora_bank'] = [self.iterative_reasoner.lora_bank]
            components['halting_unit'] = [self.iterative_reasoner.halting_unit]
            components['iteration_embed'] = []
        total = 0
        for name, modules in components.items():
            count = sum(
                p.numel() for m in modules for p in m.parameters()
            )
            counts[name] = count
            total += count
        if self.use_iterative_reasoning:
            iter_embed_count = self.iterative_reasoner.iteration_embed.numel()
            counts['iteration_embed'] = iter_embed_count
            total += iter_embed_count
        counts['total'] = total
        return counts


class ExpansionLayer(nn.Module):
    """
    Cross-attention where local tokens (queries) attend to enriched concepts (keys/values).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        concepts: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = tokens.shape
        N = concepts.shape[1]
        H = self.n_heads
        hd = self.head_dim

        q = self.q_proj(self.norm_q(tokens)).view(B, L, H, hd).transpose(1, 2)
        k = self.k_proj(self.norm_kv(concepts)).view(B, N, H, hd).transpose(1, 2)
        v = self.v_proj(self.norm_kv(concepts)).view(B, N, H, hd).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        return tokens + out


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
