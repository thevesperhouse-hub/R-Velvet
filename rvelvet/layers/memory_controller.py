"""
Memory Controller: External memory with semantic addressing.

The idea: Some information needs to persist across very long contexts
but doesn't fit in the fixed concept budget. Instead of trying to
compress EVERYTHING into 512 concepts, we allow the model to
READ from and WRITE to an external memory bank.

This is NOT a KV cache (that's just attention history).
This is semantic memory: the model DECIDES what to store and what to retrieve.

Key differences from Memory Networks / NTM:
- Content-based addressing only (no location-based)
- Write is gated: model learns WHEN to write
- Memory entries have a "staleness" score for replacement
- Reads are multi-hop: can chain retrievals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryController(nn.Module):
    """
    External memory with learned read/write.

    Memory bank: (M, D) matrix where M = number of slots.
    Each slot stores a D-dimensional vector.

    Read: content-based attention (query → memory → weighted sum)
    Write: gated update (decide what to store, where to put it)

    Args:
        d_model: Hidden dimension
        n_heads: Number of read heads (parallel reads)
        memory_size: Number of memory slots
        n_read_steps: Number of multi-hop read steps
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 4,
        memory_size: int = 256,
        n_read_steps: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.memory_size = memory_size
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # Memory bank - initialized as learnable parameter
        # Will be overwritten at runtime but learned init gives good starting point
        self.memory_init = nn.Parameter(torch.randn(memory_size, d_model) * 0.02)

        # Read heads: project query → (n_heads) read keys
        self.read_query = nn.Linear(d_model, d_model, bias=False)
        self.read_key = nn.Linear(d_model, d_model, bias=False)
        self.read_value = nn.Linear(d_model, d_model, bias=False)
        self.read_out = nn.Linear(d_model, d_model, bias=False)
        self.read_norm = RMSNorm(d_model)

        # Write heads: decide what to write and where
        self.write_query = nn.Linear(d_model, d_model, bias=False)
        self.write_key = nn.Linear(d_model, d_model, bias=False)
        self.write_value = nn.Linear(d_model, d_model, bias=False)

        # Write gate: sigmoid → [0, 1] per slot, controls how much to update
        self.write_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, 1, bias=False),
            nn.Sigmoid(),
        )

        # Staleness tracker (not a parameter, just state)
        self.register_buffer(
            'staleness', torch.zeros(memory_size), persistent=False
        )

        # Multi-hop read
        self.n_read_steps = n_read_steps
        if n_read_steps > 1:
            self.hop_transform = nn.Linear(d_model * 2, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize memory for a new sequence.

        Returns:
            memory: (B, M, D) - fresh memory bank
        """
        memory = self.memory_init.unsqueeze(0).expand(batch_size, -1, -1)
        return memory.clone()  # Clone so each batch element is independent

    def read(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-hop content-based read from memory.

        Args:
            query: (B, N, D) - what we're looking for
            memory: (B, M, D) - memory bank

        Returns:
            retrieved: (B, N, D) - information read from memory
        """
        B, N, D = query.shape
        M = memory.shape[1]
        H = self.n_heads
        hd = self.head_dim

        current_query = query

        for step in range(self.n_read_steps):
            # Project query and memory
            q = self.read_query(self.read_norm(current_query))
            k = self.read_key(memory)
            v = self.read_value(memory)

            # Multi-head reshape
            q = q.view(B, N, H, hd).transpose(1, 2)   # (B, H, N, hd)
            k = k.view(B, M, H, hd).transpose(1, 2)    # (B, H, M, hd)
            v = v.view(B, M, H, hd).transpose(1, 2)    # (B, H, M, hd)

            # Content-based attention
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, M)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            # Read
            read_out = attn @ v  # (B, H, N, hd)
            read_out = read_out.transpose(1, 2).contiguous().view(B, N, D)
            read_out = self.read_out(read_out)

            if step == 0:
                retrieved = read_out
            else:
                # Multi-hop: combine previous read with new read
                combined = torch.cat([retrieved, read_out], dim=-1)
                retrieved = self.hop_transform(combined)

            # Next hop query = original query + what we found so far
            if step < self.n_read_steps - 1:
                current_query = query + retrieved

        return retrieved

    def write(
        self,
        concepts: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gated write to memory.

        The model decides:
        1. WHAT to write (content)
        2. WHERE to write (which slots, via content similarity)
        3. HOW MUCH to write (gate value per slot)

        Args:
            concepts: (B, N, D) - new information to potentially store
            memory: (B, M, D) - current memory bank

        Returns:
            updated_memory: (B, M, D) - memory after write
        """
        B, N, D = concepts.shape
        M = memory.shape[1]

        # Compute write address: which memory slots match the new content?
        write_q = self.write_query(concepts)      # (B, N, D)
        write_k = self.write_key(memory)           # (B, M, D)

        # Address scores: (B, N, M)
        address = torch.bmm(write_q, write_k.transpose(1, 2))
        address = address / (D ** 0.5)
        address = F.softmax(address, dim=-1)

        # What to write: transform concepts into write values
        write_v = self.write_value(concepts)  # (B, N, D)

        # Aggregate write content per memory slot: (B, M, D)
        write_content = torch.bmm(address.transpose(1, 2), write_v)

        # Compute write gate per slot
        # Gate sees both current memory and proposed content
        gate_input = torch.cat([memory, write_content], dim=-1)  # (B, M, 2D)
        gate = self.write_gate(gate_input)  # (B, M, 1)

        # Gated update: memory = (1 - gate) * old + gate * new
        updated_memory = (1 - gate) * memory + gate * write_content

        return updated_memory

    def write_with_priority(
        self,
        concepts: torch.Tensor,
        memory: torch.Tensor,
        write_priority: torch.Tensor,
    ) -> torch.Tensor:
        """
        Priority-gated write to memory.

        Same as write(), but the internal write gate is multiplied by an
        external priority signal from the Scanner. This means low-priority
        segments (SKIM) barely write to memory, while high-priority (FOCUS)
        segments write strongly.

        No additional parameters — reuses existing write infrastructure.

        Args:
            concepts: (B, N, D) - new information to potentially store
            memory: (B, M, D) - current memory bank
            write_priority: (B, N) - external priority per concept [0, 1]

        Returns:
            updated_memory: (B, M, D) - memory after priority-gated write
        """
        B, N, D = concepts.shape
        M = memory.shape[1]

        # Compute write address: which memory slots match the new content?
        write_q = self.write_query(concepts)      # (B, N, D)
        write_k = self.write_key(memory)           # (B, M, D)

        # Address scores: (B, N, M)
        address = torch.bmm(write_q, write_k.transpose(1, 2))
        address = address / (D ** 0.5)
        address = F.softmax(address, dim=-1)

        # Propagate priority to memory slots via addressing matrix
        # write_priority: (B, N) → weight the address contributions
        priority_expanded = write_priority.unsqueeze(-1)  # (B, N, 1)
        address_weighted = address * priority_expanded  # (B, N, M)

        # What to write: transform concepts into write values
        write_v = self.write_value(concepts)  # (B, N, D)

        # Aggregate write content per memory slot weighted by priority
        write_content = torch.bmm(address_weighted.transpose(1, 2), write_v)  # (B, M, D)

        # Compute write gate per slot (internal gate)
        gate_input = torch.cat([memory, write_content], dim=-1)  # (B, M, 2D)
        gate = self.write_gate(gate_input)  # (B, M, 1)

        # Combined gate = internal gate (no extra external multiplication needed
        # since priority is already baked into address_weighted)
        updated_memory = (1 - gate) * memory + gate * write_content

        return updated_memory

    def forward(
        self,
        concepts: torch.Tensor,
        memory: torch.Tensor = None,
        write_priority: torch.Tensor = None,
    ) -> dict:
        """
        Full memory interaction: read then write.

        Args:
            concepts: (B, N, D) - concept vectors from global reasoner
            memory: (B, M, D) - current memory state (None = initialize)
            write_priority: (B, N) - optional external priority for gated write

        Returns:
            dict with:
                'enriched': (B, N, D) - concepts enriched with memory reads
                'memory': (B, M, D) - updated memory state
        """
        B = concepts.shape[0]
        device = concepts.device

        # Initialize memory if needed
        if memory is None:
            memory = self.init_memory(B, device)

        # Read from memory (enrich concepts with stored info)
        retrieved = self.read(concepts, memory)
        enriched = concepts + retrieved

        # Write to memory (store new relevant info)
        if write_priority is not None:
            updated_memory = self.write_with_priority(enriched, memory, write_priority)
        else:
            updated_memory = self.write(enriched, memory)

        return {
            'enriched': enriched,
            'memory': updated_memory,
        }


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
