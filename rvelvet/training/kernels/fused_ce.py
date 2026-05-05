"""Fused Cross-Entropy — Triton kernel (Unsloth-style, single-launch).

Standard cross-entropy materializes the FULL [B*S, V] logits tensor in memory.
For V=128K, batch=64, seq=512: that's 64*512*128K*4 = 17GB just for logits.

This kernel computes cross-entropy chunked along the vocab dimension WITHIN a
single kernel launch — never materializing the full softmax. Memory: O(B*S*chunk).

Key wins vs. the chunk-per-launch baseline:
- 1 forward + 1 backward kernel launch instead of 2*ceil(V/CHUNK) launches
  (e.g. 32× fewer launches for V=128K)
- max_logit / sum_exp stay in registers across chunks (no global memory transit)
- grad_loss passed as scalar arg (no per-row pointer load)
- use_lse constexpr: skips grad_lse path entirely when z-loss is unused

Based on the technique from Unsloth (Daniel & Michael Han).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Chunk size for vocab processing — fits in SRAM (constexpr, power-of-2)
VOCAB_CHUNK = 4096


@triton.jit
def _fused_ce_forward_kernel(
    logits_ptr,      # [N, V] — N = B*S, can be BF16 / FP16 / F32
    labels_ptr,      # [N]
    loss_ptr,        # [N] — output: per-row loss (F32)
    lse_ptr,         # [N] — output: per-row logsumexp (F32, for z-loss)
    max_logit_ptr,   # [N] — output: saved for backward (F32)
    sum_exp_ptr,     # [N] — output: saved for backward (F32)
    V,
    NUM_CHUNKS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """One program per row. Internal loop over vocab chunks.

    Online softmax: running max + sum_exp accumulate in registers across
    iterations — never spilled to global memory until final store.
    """
    row = tl.program_id(0).to(tl.int64)
    label = tl.load(labels_ptr + row).to(tl.int64)

    # Single load of the correct logit (label is constant per row)
    correct_logit = tl.load(logits_ptr + row * V + label).to(tl.float32)

    # Online softmax accumulators (registers)
    running_max = -float("inf")
    running_sum = 0.0

    offs = tl.arange(0, CHUNK_SIZE).to(tl.int64)
    for c in tl.range(NUM_CHUNKS):
        chunk_start = c * CHUNK_SIZE
        vocab_idx = chunk_start + offs
        mask = vocab_idx < V

        logit_ptrs = logits_ptr + row * V + vocab_idx
        logits = tl.load(
            logit_ptrs, mask=mask, other=-float("inf"),
            eviction_policy='evict_first',
        ).to(tl.float32)

        chunk_max = tl.max(logits, axis=0)
        new_max = tl.maximum(running_max, chunk_max)
        # Rescale previous sum to new_max basis, then add chunk's exp-sum
        rescaled_prev = running_sum * tl.exp(running_max - new_max)
        chunk_sum = tl.sum(tl.exp(logits - new_max), axis=0)
        running_sum = rescaled_prev + chunk_sum
        running_max = new_max

    # Finalize: lse = max + log(sum), loss = -correct_logit + lse
    lse_val = running_max + tl.log(running_sum)
    loss_val = -correct_logit + lse_val

    tl.store(loss_ptr + row, loss_val)
    tl.store(lse_ptr + row, lse_val)
    tl.store(max_logit_ptr + row, running_max)
    tl.store(sum_exp_ptr + row, running_sum)


@triton.jit
def _fused_ce_backward_kernel(
    logits_ptr,       # [N, V] — read AND write (gradients in-place)
    grad_logits_ptr,  # same as logits_ptr — separate ptr for clarity
    labels_ptr,       # [N]
    max_logit_ptr,    # [N] from forward
    sum_exp_ptr,      # [N] from forward
    grad_lse_ptr,     # [N] — F32, only loaded if use_lse
    grad_loss,        # scalar (F32) — passed by-value, no memory load
    V,
    inv_N,            # 1.0 / N (scalar)
    use_lse: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """One program per row. Internal loop over vocab chunks. Writes grads in-place."""
    row = tl.program_id(0).to(tl.int64)
    label = tl.load(labels_ptr + row).to(tl.int64)
    max_val = tl.load(max_logit_ptr + row)
    sum_val = tl.load(sum_exp_ptr + row)

    if use_lse:
        g_lse = tl.load(grad_lse_ptr + row)
    else:
        g_lse = 0.0

    offs = tl.arange(0, CHUNK_SIZE).to(tl.int64)
    for c in tl.range(NUM_CHUNKS):
        chunk_start = c * CHUNK_SIZE
        vocab_idx = chunk_start + offs
        mask = vocab_idx < V

        logit_ptrs = logits_ptr + row * V + vocab_idx
        logits = tl.load(
            logit_ptrs, mask=mask, other=-float("inf"),
            eviction_policy='evict_first',
        ).to(tl.float32)

        # Recompute softmax (no extra memory beyond saved max/sum)
        softmax = tl.exp(logits - max_val) / sum_val

        # CE gradient: (softmax - one_hot) * grad_loss / N
        is_label = (vocab_idx == label).to(tl.float32)
        grad = (softmax - is_label) * grad_loss * inv_N

        # LSE gradient: softmax * grad_lse  (z-loss backprop)
        if use_lse:
            grad = grad + softmax * g_lse

        # In-place write to logits memory (auto-converts to native dtype)
        grad_ptrs = grad_logits_ptr + row * V + vocab_idx
        tl.store(grad_ptrs, grad, mask=mask, eviction_policy='evict_last')


class _FusedCEFunction(torch.autograd.Function):
    """Autograd wrapper: makes (loss, lse) differentiable w.r.t. logits.

    The backward recomputes softmax per-chunk from saved (max, sum_exp) and
    writes gradients in-place into the logits buffer — no [N, V] grad allocation.
    Safe because each thread reads a chunk THEN writes the gradient at the same
    address (no cross-row, no cross-chunk dependencies).
    """

    @staticmethod
    def forward(ctx, logits, labels):
        N, V = logits.shape
        device = logits.device

        loss = torch.empty(N, device=device, dtype=torch.float32)
        lse = torch.empty(N, device=device, dtype=torch.float32)
        max_logit = torch.empty(N, device=device, dtype=torch.float32)
        sum_exp = torch.empty(N, device=device, dtype=torch.float32)

        num_chunks = triton.cdiv(V, VOCAB_CHUNK)

        _fused_ce_forward_kernel[(N,)](
            logits, labels, loss, lse, max_logit, sum_exp,
            V,
            NUM_CHUNKS=num_chunks, CHUNK_SIZE=VOCAB_CHUNK,
            num_warps=8,
        )

        loss_mean = loss.mean()

        ctx.save_for_backward(labels, max_logit, sum_exp)
        ctx.logits = logits  # direct ref so backward can write in-place
        ctx.N = N
        ctx.V = V

        return loss_mean, lse

    @staticmethod
    def backward(ctx, grad_loss, grad_lse):
        labels, max_logit, sum_exp = ctx.saved_tensors
        logits = ctx.logits
        N, V = ctx.N, ctx.V

        # use_lse constexpr: skip the entire grad_lse path when not needed
        if grad_lse is None:
            use_lse = False
            # Pass a 1-element tensor as a placeholder; kernel never loads it.
            grad_lse_t = torch.empty(1, device=logits.device, dtype=torch.float32)
        else:
            use_lse = True
            grad_lse_t = grad_lse.contiguous().to(torch.float32)

        # grad_loss is a scalar tensor; convert to Python float (scalar kernel arg)
        grad_loss_f = float(grad_loss.detach())
        inv_N = 1.0 / N

        num_chunks = triton.cdiv(V, VOCAB_CHUNK)

        _fused_ce_backward_kernel[(N,)](
            logits, logits, labels,        # read AND write in-place
            max_logit, sum_exp,
            grad_lse_t,
            grad_loss_f,
            V,
            inv_N,
            use_lse=use_lse,
            NUM_CHUNKS=num_chunks, CHUNK_SIZE=VOCAB_CHUNK,
            num_warps=8,
        )

        return logits, None  # logits buffer now holds gradients


def fused_cross_entropy(
    logits: torch.Tensor,   # [B, S, V] or [N, V]
    labels: torch.Tensor,   # [B, S] or [N]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked fused cross-entropy — O(B*S*chunk) memory instead of O(B*S*V).

    For V=128K, this saves ~17GB vs standard F.cross_entropy. Single-kernel
    fusion eliminates per-chunk launch overhead.

    Returns (loss, logsumexp). Both are differentiable w.r.t. logits.
    Falls back to PyTorch when not on CUDA or for very small vocab where
    cuBLAS-optimized softmax is faster.
    """
    if logits.dim() == 3:
        B, S, V = logits.shape
        logits = logits.reshape(B * S, V)
        labels = labels.reshape(B * S)
    else:
        N, V = logits.shape

    # CPU / very small vocab: PyTorch path. F.cross_entropy handles bf16/fp16
    # internally via f32 reductions — no need for an explicit .float() cast that
    # would allocate a full [N, V] f32 buffer.
    if not logits.is_cuda or V <= 1024:
        loss = F.cross_entropy(logits, labels, reduction="mean")
        # logsumexp on bf16/fp16 isn't numerically safe — upcast just the
        # reduction dim (still cheap because it's a single reduction over V).
        if logits.dtype in (torch.float16, torch.bfloat16):
            lse = torch.logsumexp(logits.float(), dim=-1)
        else:
            lse = torch.logsumexp(logits, dim=-1)
        return loss, lse

    return _FusedCEFunction.apply(logits, labels)
