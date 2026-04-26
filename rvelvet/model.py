"""
R-Velvet: Multi-Scale Architecture for Unlimited Context.

The full pipeline:
    tokens → Local Encoder → Segment Compressor → Global Reasoner → Memory Controller → Output

Why this works for 1M+ tokens:
- Local attention: O(n * w²) where w = 512 → handles raw token detail
- Compression: 1M tokens → 512 concepts → quadratic is trivial
- Global reasoning: O(512²) = 262K operations → nothing
- Memory: persistent storage across chunks, no length limit

The key insight: you don't NEED to attend to all tokens at once.
You need to COMPRESS intelligently, REASON globally, and REMEMBER selectively.
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
    R-Velvet: Multi-scale transformer for unlimited context.

    Architecture:
        1. Token Embedding (vocab → d_model)
        2. Local Encoder (windowed attention on raw tokens)
        3. Segment Compressor (tokens → concepts)
        4. Global Reasoner (full attention on concepts)
        5. Memory Controller (read/write external memory)
        6. Expansion + Output Head

    For generation: concepts get expanded back and projected to vocab.
    For understanding: enriched concepts are the representation.

    Args:
        vocab_size: Vocabulary size
        d_model: Hidden dimension throughout
        n_local_layers: Number of local attention layers
        n_global_layers: Number of global reasoning layers
        n_local_heads: Number of heads in local attention
        n_global_heads: Number of heads in global reasoning
        window_size: Local attention window size
        segment_size: Segment size for compression
        n_concepts: Number of concept vectors per segment
        n_refine_layers: Cross-attention refinement rounds in compressor
        memory_size: Number of external memory slots
        n_read_steps: Multi-hop read steps in memory
        ffn_mult: FFN hidden dimension multiplier
        dropout: Dropout rate
        max_seq_len: Maximum sequence length (for positional encoding)
        use_acr: Enable Adaptive Computation Routing
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

        # --- Token Embedding ---
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(dropout)

        # --- Stage 1: Local Encoder ---
        # Windowed attention on raw tokens. Captures syntax, local semantics.
        self.local_encoder = LocalEncoder(
            d_model=d_model,
            n_heads=n_local_heads,
            n_layers=n_local_layers,
            window_size=window_size,
            ffn_mult=ffn_mult,
            dropout=dropout,
        )

        # --- Stage 2: Segment Compressor ---
        # Compress local tokens into concept vectors.
        # This is where 1M tokens become ~500 concepts.
        self.compressor = SegmentCompressor(
            d_model=d_model,
            n_heads=n_local_heads,
            segment_size=segment_size,
            n_concepts=n_concepts,
            n_refine_layers=n_refine_layers,
            dropout=dropout,
        )

        # --- Stage 3: Global Reasoner ---
        # Full attention between concepts. This is where
        # "paragraph 3 contradicts paragraph 47" gets caught.
        self.global_reasoner = GlobalReasoner(
            d_model=d_model,
            n_heads=n_global_heads,
            n_layers=n_global_layers,
            ffn_mult=ffn_mult,
            dropout=dropout,
        )

        # --- Stage 4: Memory Controller ---
        # Read/write external memory for ultra-long context.
        self.memory_controller = MemoryController(
            d_model=d_model,
            n_heads=n_local_heads,
            memory_size=memory_size,
            n_read_steps=n_read_steps,
            dropout=dropout,
        )

        # --- ACR Components (optional) ---
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

        # --- Iterative Reasoning (optional) ---
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

        # --- Expansion: concepts back to token-level ---
        # Cross-attention: local tokens attend to enriched concepts
        self.expansion = ExpansionLayer(
            d_model=d_model,
            n_heads=n_local_heads,
            dropout=dropout,
        )

        # --- Output Head ---
        self.out_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: embed and output share weights
        self.lm_head.weight = self.token_embed.weight

        # Init weights
        self._init_weights()

    def _init_weights(self):
        """Xavier-like init, small for stability."""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_normal_(p, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(p)

        # Special: embedding init
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor = None,
        causal: bool = True,
    ) -> dict:
        """
        Full forward pass.

        Args:
            input_ids: (B, L) - token indices
            memory: (B, M, D) - external memory state (None = fresh)
            causal: Whether to use causal masking

        Returns:
            dict with:
                'logits': (B, L, V) - next token predictions
                'concepts': (B, N_concepts, D) - concept representations
                'memory': (B, M, D) - updated memory state
                'relevance': (B, N_concepts) - concept relevance scores
        """
        B, L = input_ids.shape

        # Dispatch to ACR forward if enabled
        if self.use_acr:
            return self.forward_acr(input_ids, memory=memory, causal=causal)

        # --- Embed ---
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_drop(x)

        # --- Stage 1: Local Encoding ---
        # (B, L, D) → (B, L, D) with local context
        local_out = self.local_encoder(x, causal=causal)

        # --- Stage 2: Compress ---
        # (B, L, D) → (B, n_segments, n_concepts, D)
        concepts = self.compressor(local_out)
        # Flatten segment and concept dims: (B, n_segments * n_concepts, D)
        B_c, n_seg, K, D = concepts.shape
        concepts_flat = concepts.view(B, n_seg * K, D)

        # --- Stage 3 & 4: Global Reasoning + Memory ---
        if self.use_iterative_reasoning:
            iter_out = self.iterative_reasoner(
                concepts_flat, memory=memory, causal=causal,
            )
            enriched_concepts = iter_out['concepts']
            relevance = iter_out['relevance']
            updated_memory = iter_out['memory']
        else:
            global_out = self.global_reasoner(concepts_flat, causal=causal)
            refined_concepts = global_out['concepts']   # (B, N, D)
            relevance = global_out['relevance']          # (B, N)

            mem_out = self.memory_controller(refined_concepts, memory)
            enriched_concepts = mem_out['enriched']      # (B, N, D)
            updated_memory = mem_out['memory']           # (B, M, D)

        # --- Expand back to token level ---
        # Local tokens attend to enriched concepts
        expanded = self.expansion(local_out, enriched_concepts)

        # --- Output ---
        out = self.out_norm(expanded)
        logits = self.lm_head(out)  # (B, L, V)

        result = {
            'logits': logits,
            'concepts': enriched_concepts,
            'memory': updated_memory,
            'relevance': relevance,
        }

        # Add iterative reasoning extras
        if self.use_iterative_reasoning:
            result['iteration_outputs'] = iter_out['iteration_outputs']
            result['p_halts'] = iter_out['p_halts']
            result['halt_distribution'] = iter_out['halt_distribution']
            result['n_iterations'] = iter_out['n_iterations']
            result['local_out'] = local_out  # For deep supervision

        return result

    def forward_acr(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor = None,
        causal: bool = True,
    ) -> dict:
        """
        Forward pass with Adaptive Computation Routing.

        Each segment is scanned, routed, and processed with variable depth/compression.

        Args:
            input_ids: (B, L) - token indices
            memory: (B, M, D) - external memory state (None = fresh)
            causal: Whether to use causal masking

        Returns:
            dict with:
                'logits': (B, L, V) - next token predictions
                'concepts': (B, N_concepts, D) - concept representations
                'memory': (B, M, D) - updated memory state
                'relevance': (B, N_concepts) - concept relevance scores
                'route_weights': (B, S, 3) - route selection per segment
                'route_logits': (B, S, 3) - raw route logits
                'write_priority': (B, S) - priority signal per segment
        """
        B, L = input_ids.shape
        D = self.d_model
        W = self.segment_size
        device = input_ids.device

        # Dispatch to iterative ACR if both are enabled
        if self.use_iterative_reasoning:
            return self._forward_acr_iterative(input_ids, memory=memory, causal=causal)

        # --- Embed ---
        positions = torch.arange(L, device=device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_drop(x)

        # --- Pad and reshape into segments ---
        pad_len = (W - L % W) % W
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x
        L_padded = x_padded.shape[1]
        n_seg = L_padded // W
        segments = x_padded.view(B, n_seg, W, D)  # (B, S, W, D)

        # --- Scan: decide route per segment ---
        scan_out = self.scanner(segments)
        route_logits = scan_out['route_logits']      # (B, S, 3)
        write_priority = scan_out['write_priority']  # (B, S)

        # --- Route ---
        route_weights = self.router(route_logits)    # (B, S, 3)

        # --- Local Encoding with Residual Gating ---
        local_gates = compute_layer_gates(
            route_weights, self.n_local_layers
        )  # (B, S, n_local_layers)

        h_flat = segments.reshape(B * n_seg, W, D)

        if not self.training:
            # Inference: skip layers where no segment needs computation
            for i, layer in enumerate(self.local_encoder.layers):
                gate_flat = local_gates[:, :, i].reshape(B * n_seg)
                active = gate_flat > 0.5
                if not active.any():
                    continue
                h_flat[active] = layer(h_flat[active], causal=causal)
        else:
            # Training: residual gating for differentiable routing
            for i, layer in enumerate(self.local_encoder.layers):
                gate_flat = local_gates[:, :, i].reshape(B * n_seg, 1, 1)
                h_new = layer(h_flat, causal=causal)
                delta = h_new - h_flat
                h_flat = h_flat + gate_flat * delta

        h_flat = self.local_encoder.norm(h_flat)
        local_out = h_flat.view(B, n_seg, W, D)

        # --- Adaptive Compression ---
        concepts_4d = self.adaptive_compressor(
            local_out, route_weights
        )  # (B, S, K_max, D) where K_max = n_concepts_focus

        K_max = concepts_4d.shape[2]
        concepts_flat = concepts_4d.view(B, n_seg * K_max, D)

        # --- Build concept validity mask ---
        # Each route produces a different number of real concepts;
        # the rest are zero-padded and must be masked in attention.
        n_per_route = torch.tensor(
            [self.adaptive_compressor.N_CONCEPTS_SKIM,
             self.adaptive_compressor.N_CONCEPTS_PROCESS,
             self.adaptive_compressor.n_concepts_focus],
            device=device, dtype=torch.float,
        )  # (3,)
        concepts_per_seg = (route_weights * n_per_route).sum(dim=-1)  # (B, S)
        pos_idx = torch.arange(K_max, device=device).float()
        concept_valid = pos_idx < concepts_per_seg.unsqueeze(-1)  # (B, S, K_max)
        padding_mask = ~concept_valid.view(B, n_seg * K_max)  # True = ignore

        # --- Global Reasoning with Residual Gating ---
        global_gates = compute_layer_gates(
            route_weights, self.n_global_layers
        )  # (B, S, n_global_layers)

        global_gates_expanded = global_gates.unsqueeze(2).expand(
            B, n_seg, K_max, self.n_global_layers
        ).reshape(B, n_seg * K_max, self.n_global_layers)

        h_global = concepts_flat
        for i, layer in enumerate(self.global_reasoner.layers):
            gate = global_gates_expanded[:, :, i].unsqueeze(-1)  # (B, N, 1)
            h_new = layer(h_global, causal=causal, padding_mask=padding_mask)
            delta = h_new - h_global
            h_global = h_global + gate * delta

        h_global = self.global_reasoner.norm(h_global)

        # Zero out padded concepts so they don't leak into downstream stages
        h_global = h_global * (~padding_mask).unsqueeze(-1).float()

        relevance = self.global_reasoner.relevance_head(h_global).squeeze(-1)
        relevance = torch.sigmoid(relevance)

        refined_concepts = h_global  # (B, N, D)

        # --- Memory with Priority ---
        write_priority_expanded = write_priority.unsqueeze(2).expand(
            B, n_seg, K_max
        ).reshape(B, n_seg * K_max)

        mem_out = self.memory_controller(
            refined_concepts, memory, write_priority=write_priority_expanded
        )
        enriched_concepts = mem_out['enriched']
        updated_memory = mem_out['memory']

        # Re-zero padded concepts (memory read may have contaminated them)
        valid_mask = (~padding_mask).unsqueeze(-1).float()  # (B, N, 1)
        enriched_concepts = enriched_concepts * valid_mask

        # --- Expand back to token level ---
        local_flat = local_out.view(B, L_padded, D)
        if pad_len > 0:
            local_flat = local_flat[:, :L, :]
        expanded = self.expansion(local_flat, enriched_concepts)

        # --- Output ---
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
        """
        ACR forward pass with iterative reasoning.

        Combines ACR residual gating with iterative LoRA + halting.
        For each iteration i, for each layer j:
            qkv_delta_fn = lora_bank.get_qkv_delta_fn(i, j)
            h_new = layer(h, causal, padding_mask, qkv_delta_fn)
            h = h + gate[j] * (h_new - h)  # ACR gating still active
        """
        B, L = input_ids.shape
        D = self.d_model
        W = self.segment_size
        device = input_ids.device
        ir = self.iterative_reasoner

        # --- Embed ---
        positions = torch.arange(L, device=device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_drop(x)

        # --- Pad and reshape into segments ---
        pad_len = (W - L % W) % W
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x
        L_padded = x_padded.shape[1]
        n_seg = L_padded // W
        segments = x_padded.view(B, n_seg, W, D)

        # --- Scan + Route ---
        scan_out = self.scanner(segments)
        route_logits = scan_out['route_logits']
        write_priority = scan_out['write_priority']
        route_weights = self.router(route_logits)

        # --- Local Encoding with Residual Gating ---
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

        # --- Adaptive Compression ---
        concepts_4d = self.adaptive_compressor(local_out, route_weights)
        K_max = concepts_4d.shape[2]
        concepts_flat = concepts_4d.view(B, n_seg * K_max, D)

        # --- Build concept validity mask ---
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

        # --- Global gates for ACR ---
        global_gates = compute_layer_gates(route_weights, self.n_global_layers)
        global_gates_expanded = global_gates.unsqueeze(2).expand(
            B, n_seg, K_max, self.n_global_layers
        ).reshape(B, n_seg * K_max, self.n_global_layers)

        # --- Write priority expanded ---
        write_priority_expanded = write_priority.unsqueeze(2).expand(
            B, n_seg, K_max
        ).reshape(B, n_seg * K_max)

        # --- Initialize memory ---
        if memory is None:
            memory = self.memory_controller.init_memory(B, device)

        # --- Iterative Reasoning Loop with ACR gating ---
        iteration_outputs = []
        iteration_relevances = []
        p_halts = []
        h = concepts_flat  # COCONUT input

        for it in range(ir.max_iterations):
            # 1. Add iteration embedding
            h_iter = h + ir.iteration_embed[it].unsqueeze(0).unsqueeze(0)

            # 2. Process through layers with ACR gating + LoRA
            for j, layer in enumerate(self.global_reasoner.layers):
                gate = global_gates_expanded[:, :, j].unsqueeze(-1)  # (B, N, 1)
                qkv_delta_fn = ir.lora_bank.get_qkv_delta_fn(it, j)
                h_new = layer(
                    h_iter, causal=causal,
                    padding_mask=padding_mask,
                    qkv_delta_fn=qkv_delta_fn,
                )
                delta = h_new - h_iter
                h_iter = h_iter + gate * delta  # ACR gating

            # 3. Norm + relevance
            h_normed = self.global_reasoner.norm(h_iter)
            h_normed = h_normed * (~padding_mask).unsqueeze(-1).float()

            relevance = torch.sigmoid(
                self.global_reasoner.relevance_head(h_normed).squeeze(-1)
            )

            # 4. Memory with priority
            mem_out = self.memory_controller(
                h_normed, memory, write_priority=write_priority_expanded
            )
            enriched = mem_out['enriched']
            memory = mem_out['memory']

            # Re-zero padded
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            enriched = enriched * valid_mask

            # 5. Halting
            p_halt = ir.halting_unit(enriched, padding_mask)
            p_halts.append(p_halt)

            # 6. Store
            iteration_outputs.append(enriched)
            iteration_relevances.append(relevance)

            # 7. COCONUT
            h = enriched

            # 8. Early exit at inference
            if not self.training:
                if (p_halt > ir.halt_threshold).all():
                    break

        n_iterations = len(iteration_outputs)

        # Compute halt distribution
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

        # --- Expand back to token level ---
        local_flat = local_out.view(B, L_padded, D)
        if pad_len > 0:
            local_flat = local_flat[:, :L, :]
        expanded = self.expansion(local_flat, final_concepts)

        # --- Output ---
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
        """Count parameters per component."""
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
            # iteration_embed is a Parameter, count manually
            components['iteration_embed'] = []  # handled below
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
    Expand concepts back to token-level via cross-attention.

    Local tokens (queries) attend to enriched concepts (keys/values).
    This lets each token "pick up" global context from relevant concepts.

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
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
        """
        Args:
            tokens: (B, L, D) - local token representations
            concepts: (B, N, D) - enriched concept vectors

        Returns:
            expanded: (B, L, D) - tokens enriched with global context
        """
        B, L, D = tokens.shape
        N = concepts.shape[1]
        H = self.n_heads
        hd = self.head_dim

        q = self.q_proj(self.norm_q(tokens)).view(B, L, H, hd).transpose(1, 2)
        k = self.k_proj(self.norm_kv(concepts)).view(B, N, H, hd).transpose(1, 2)
        v = self.v_proj(self.norm_kv(concepts)).view(B, N, H, hd).transpose(1, 2)

        # Cross-attention: tokens attend to concepts
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, L, N)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, H, L, hd)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        # Residual connection with original tokens
        return tokens + out


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
