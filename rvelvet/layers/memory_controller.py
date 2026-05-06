"""
External semantic memory with content-based read/write. The model learns what to store and retrieve
via gated writes and multi-hop reads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryController(nn.Module):
    """
    External memory with content-based read (multi-hop attention) and gated write.
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

        self.memory_init = nn.Parameter(torch.randn(memory_size, d_model) * 0.02)

        self.read_query = nn.Linear(d_model, d_model, bias=False)
        self.read_key = nn.Linear(d_model, d_model, bias=False)
        self.read_value = nn.Linear(d_model, d_model, bias=False)
        self.read_out = nn.Linear(d_model, d_model, bias=False)
        self.read_norm = RMSNorm(d_model)

        self.write_query = nn.Linear(d_model, d_model, bias=False)
        self.write_key = nn.Linear(d_model, d_model, bias=False)
        self.write_value = nn.Linear(d_model, d_model, bias=False)

        self.write_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, 1, bias=False),
            nn.Sigmoid(),
        )

        self.register_buffer(
            'staleness', torch.zeros(memory_size), persistent=False
        )

        self.n_read_steps = n_read_steps
        if n_read_steps > 1:
            self.hop_transform = nn.Linear(d_model * 2, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        memory = self.memory_init.unsqueeze(0).expand(batch_size, -1, -1)
        return memory.clone()

    def read(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = query.shape
        M = memory.shape[1]
        H = self.n_heads
        hd = self.head_dim

        current_query = query

        for step in range(self.n_read_steps):
            q = self.read_query(self.read_norm(current_query))
            k = self.read_key(memory)
            v = self.read_value(memory)

            q = q.view(B, N, H, hd).transpose(1, 2)
            k = k.view(B, M, H, hd).transpose(1, 2)
            v = v.view(B, M, H, hd).transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            read_out = attn @ v
            read_out = read_out.transpose(1, 2).contiguous().view(B, N, D)
            read_out = self.read_out(read_out)

            if step == 0:
                retrieved = read_out
            else:
                combined = torch.cat([retrieved, read_out], dim=-1)
                retrieved = self.hop_transform(combined)

            if step < self.n_read_steps - 1:
                current_query = query + retrieved

        return retrieved

    def write(
        self,
        concepts: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = concepts.shape
        M = memory.shape[1]

        write_q = self.write_query(concepts)
        write_k = self.write_key(memory)

        address = torch.bmm(write_q, write_k.transpose(1, 2))
        address = address / (D ** 0.5)
        address = F.softmax(address, dim=-1)

        write_v = self.write_value(concepts)

        write_content = torch.bmm(address.transpose(1, 2), write_v)

        gate_input = torch.cat([memory, write_content], dim=-1)
        gate = self.write_gate(gate_input)

        updated_memory = (1 - gate) * memory + gate * write_content

        return updated_memory

    def write_with_priority(
        self,
        concepts: torch.Tensor,
        memory: torch.Tensor,
        write_priority: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = concepts.shape
        M = memory.shape[1]

        write_q = self.write_query(concepts)
        write_k = self.write_key(memory)

        address = torch.bmm(write_q, write_k.transpose(1, 2))
        address = address / (D ** 0.5)
        address = F.softmax(address, dim=-1)

        priority_expanded = write_priority.unsqueeze(-1)
        address_weighted = address * priority_expanded

        write_v = self.write_value(concepts)

        write_content = torch.bmm(address_weighted.transpose(1, 2), write_v)

        gate_input = torch.cat([memory, write_content], dim=-1)
        gate = self.write_gate(gate_input)

        updated_memory = (1 - gate) * memory + gate * write_content

        return updated_memory

    def forward(
        self,
        concepts: torch.Tensor,
        memory: torch.Tensor = None,
        write_priority: torch.Tensor = None,
    ) -> dict:
        B = concepts.shape[0]
        device = concepts.device

        if memory is None:
            memory = self.init_memory(B, device)

        retrieved = self.read(concepts, memory)
        enriched = concepts + retrieved

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
