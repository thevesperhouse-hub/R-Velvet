"""Fused Cross-Entropy — Triton kernel (Unsloth-style).

Standard cross-entropy materializes the FULL [B*S, V] logits tensor in memory.
For V=128K, batch=64, seq=512: that's 64*512*128K*4 = 17GB just for logits.

This kernel computes cross-entropy in chunks along the vocab dimension,
never materializing the full softmax. Memory: O(B*S*chunk) instead of O(B*S*V).

Based on the technique from Unsloth (Daniel & Michael Han).
"""

import torch
import triton
import triton.language as tl

# Chunk size for vocab processing — fits in SRAM
VOCAB_CHUNK = 4096


@triton.jit
def _fused_ce_forward_kernel(
    logits_ptr,      # [N, V] — N = B*S, can be BF16 or F32
    labels_ptr,      # [N]
    loss_ptr,        # [N] — per-token loss (F32)
    max_logit_ptr,   # [N] — running max for numerical stability (F32)
    sum_exp_ptr,     # [N] — running sum(exp) (F32)
    V,               # vocab size
    chunk_start,     # start index in vocab dim
    CHUNK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    label = tl.load(labels_ptr + row).to(tl.int64)

    # Load this chunk of logits (BF16 -> F32 per-chunk, no global copy)
    offs = tl.arange(0, CHUNK_SIZE).to(tl.int64)
    vocab_idx = chunk_start + offs
    mask = vocab_idx < V

    logit_ptrs = logits_ptr + row * V + vocab_idx
    logits = tl.load(logit_ptrs, mask=mask, other=-float("inf")).to(tl.float32)

    # Update running max
    chunk_max = tl.max(logits, axis=0)
    prev_max = tl.load(max_logit_ptr + row)
    new_max = tl.maximum(prev_max, chunk_max)

    # Rescale previous sum_exp and add this chunk
    prev_sum = tl.load(sum_exp_ptr + row)
    rescaled_prev = prev_sum * tl.exp(prev_max - new_max)
    chunk_sum = tl.sum(tl.exp(logits - new_max), axis=0)
    new_sum = rescaled_prev + chunk_sum

    tl.store(max_logit_ptr + row, new_max)
    tl.store(sum_exp_ptr + row, new_sum)

    # If the correct label falls in this chunk, store the logit
    in_chunk = (label >= chunk_start) & (label < chunk_start + CHUNK_SIZE)
    if in_chunk:
        correct_logit = tl.load(logits_ptr + row * V + label).to(tl.float32)
        tl.store(loss_ptr + row, correct_logit)


@triton.jit
def _fused_ce_finalize_kernel(
    loss_ptr,        # [N] — contains correct_logit, will be overwritten with loss
    lse_ptr,         # [N] — logsumexp output (for z-loss)
    max_logit_ptr,   # [N]
    sum_exp_ptr,     # [N]
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N

    correct_logit = tl.load(loss_ptr + offs, mask=mask)
    max_logit = tl.load(max_logit_ptr + offs, mask=mask)
    sum_exp = tl.load(sum_exp_ptr + offs, mask=mask)

    # logsumexp = max + log(sum_exp)  — reused for z-loss
    logsumexp = max_logit + tl.log(sum_exp)
    # loss = -correct_logit + logsumexp
    loss = -correct_logit + logsumexp
    tl.store(loss_ptr + offs, loss, mask=mask)
    tl.store(lse_ptr + offs, logsumexp, mask=mask)


@triton.jit
def _fused_ce_backward_kernel(
    logits_ptr,       # [N, V] — original logits (BF16 or F32)
    grad_logits_ptr,  # [N, V] — output gradient (same dtype as logits)
    labels_ptr,       # [N]
    max_logit_ptr,    # [N] — from forward
    sum_exp_ptr,      # [N] — from forward
    grad_loss_ptr,    # [1] — upstream gradient for loss_mean (F32)
    grad_lse_ptr,     # [N] — upstream gradient for lse (F32), zeros if unused
    V,
    inv_N,            # 1.0 / N (float)
    chunk_start,
    CHUNK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    label = tl.load(labels_ptr + row).to(tl.int64)

    offs = tl.arange(0, CHUNK_SIZE).to(tl.int64)
    vocab_idx = chunk_start + offs
    mask = vocab_idx < V

    # Recompute softmax from saved max/sum_exp (no extra memory)
    logit_ptrs = logits_ptr + row * V + vocab_idx
    logits = tl.load(logit_ptrs, mask=mask, other=-float("inf")).to(tl.float32)

    max_val = tl.load(max_logit_ptr + row)
    sum_val = tl.load(sum_exp_ptr + row)
    softmax = tl.exp(logits - max_val) / sum_val

    # CE gradient: (softmax - one_hot) * grad_loss / N
    g_loss = tl.load(grad_loss_ptr)
    is_label = (vocab_idx == label)
    grad = (softmax - is_label.to(tl.float32)) * g_loss * inv_N

    # LSE gradient: softmax * grad_lse[row]  (for z-loss backprop)
    g_lse = tl.load(grad_lse_ptr + row)
    grad = grad + softmax * g_lse

    # Store gradient (Triton handles F32 -> BF16 conversion if needed)
    grad_ptrs = grad_logits_ptr + row * V + vocab_idx
    tl.store(grad_ptrs, grad, mask=mask)


class _FusedCEFunction(torch.autograd.Function):
    """Autograd wrapper for chunked fused cross-entropy Triton kernel.

    Makes loss and logsumexp differentiable w.r.t. logits so backward() works.
    The backward recomputes softmax per-chunk from saved max/sum_exp — no extra
    [N, V] memory beyond the required grad_logits output.
    """

    @staticmethod
    def forward(ctx, logits, labels):
        N, V = logits.shape
        device = logits.device

        loss = torch.zeros(N, device=device, dtype=torch.float32)
        max_logit = torch.full((N,), -float("inf"), device=device, dtype=torch.float32)
        sum_exp = torch.zeros(N, device=device, dtype=torch.float32)
        lse = torch.zeros(N, device=device, dtype=torch.float32)

        for chunk_start in range(0, V, VOCAB_CHUNK):
            chunk_size = min(VOCAB_CHUNK, V - chunk_start)
            padded_chunk = triton.next_power_of_2(chunk_size)
            _fused_ce_forward_kernel[(N,)](
                logits, labels, loss, max_logit, sum_exp,
                V, chunk_start,
                CHUNK_SIZE=padded_chunk,
            )

        BLOCK = 1024
        _fused_ce_finalize_kernel[(triton.cdiv(N, BLOCK),)](
            loss, lse, max_logit, sum_exp, N, BLOCK=BLOCK,
        )

        loss_mean = loss.mean()

        # Save labels/stats via save_for_backward; logits stored separately
        # to allow in-place gradient write (avoids 8GB grad_logits allocation)
        ctx.save_for_backward(labels, max_logit, sum_exp)
        ctx.logits = logits
        ctx.N = N
        ctx.V = V

        return loss_mean, lse

    @staticmethod
    def backward(ctx, grad_loss, grad_lse):
        labels, max_logit, sum_exp = ctx.saved_tensors
        logits = ctx.logits
        N, V = ctx.N, ctx.V

        # In-place backward: overwrite logits with gradients (saves 8GB allocation).
        # Safe because the backward kernel reads each chunk THEN writes the gradient
        # to the same location — no cross-chunk or cross-row dependencies.

        # If z-loss not used, grad_lse is None — use zeros
        if grad_lse is None:
            grad_lse = torch.zeros(N, device=logits.device, dtype=torch.float32)

        # Ensure grad_loss is a contiguous F32 1-element tensor for Triton pointer load
        grad_loss_t = grad_loss.detach().float().contiguous().view(1)

        inv_N = 1.0 / N

        for chunk_start in range(0, V, VOCAB_CHUNK):
            chunk_size = min(VOCAB_CHUNK, V - chunk_start)
            padded_chunk = triton.next_power_of_2(chunk_size)
            _fused_ce_backward_kernel[(N,)](
                logits, logits, labels,  # read AND write to same tensor
                max_logit, sum_exp,
                grad_loss_t, grad_lse,
                V, inv_N, chunk_start,
                CHUNK_SIZE=padded_chunk,
            )

        return logits, None  # logits now contains gradients


def fused_cross_entropy(
    logits: torch.Tensor,   # [B, S, V] or [N, V]
    labels: torch.Tensor,   # [B, S] or [N]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked fused cross-entropy — O(B*S*chunk) memory instead of O(B*S*V).

    For V=128K, this saves ~17GB vs standard F.cross_entropy.
    Returns (loss, logsumexp) — logsumexp is per-token for z-loss computation.
    Both outputs are differentiable w.r.t. logits.
    Falls back to PyTorch if logits are small enough.
    """
    orig_shape = logits.shape
    if logits.dim() == 3:
        B, S, V = orig_shape
        logits = logits.reshape(B * S, V)
        labels = labels.reshape(B * S)
    else:
        N, V = logits.shape

    N = logits.shape[0]

    # For vocab <= 65536, just use PyTorch (faster, avoids Triton kernel issues)
    if V <= 65536:
        loss = torch.nn.functional.cross_entropy(
            logits.float(), labels, reduction="mean"
        )
        # Compute logsumexp for z-loss (small vocab = cheap)
        lse = torch.logsumexp(logits.float(), dim=-1)
        return loss, lse

    # Chunked Triton kernel with autograd support for large vocab
    return _FusedCEFunction.apply(logits, labels)
