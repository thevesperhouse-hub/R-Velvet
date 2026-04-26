"""Velvet Optimizer — Triton fused kernel (PGM + LVS).

Single kernel: weight decay + PGM momentum + moment update + bias correction + LVS LR + param update.
Avoids 8+ separate PyTorch kernel launches per parameter per step.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _velvet_update_kernel(
    # Pointers
    param_ptr, grad_ptr, m_ptr, v_ptr,
    # Scalars
    lr, beta1, beta2, eps, wd,
    bias_correction1, bias_correction2,
    entropy_adaptive: tl.constexpr, entropy_lr_scale,
    perplexity_guided: tl.constexpr, ppl_momentum_scale,
    sparse_aware: tl.constexpr,
    n,
    # Block
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load
    p = tl.load(param_ptr + offsets, mask=mask)
    g = tl.load(grad_ptr + offsets, mask=mask)
    m_val = tl.load(m_ptr + offsets, mask=mask)
    v_val = tl.load(v_ptr + offsets, mask=mask)

    # Sparse: skip near-zero weights
    if sparse_aware:
        active = tl.abs(p) >= 1e-9
        g = tl.where(active, g, 0.0)

    # Decoupled weight decay (AdamW)
    p = p * (1.0 - lr * wd)

    # PGM: adaptive momentum (beta1)
    if perplexity_guided:
        eff_beta1 = beta1 * ppl_momentum_scale
        eff_beta1 = tl.minimum(tl.maximum(eff_beta1, 0.5), 0.999)
    else:
        eff_beta1 = beta1

    # Update moments (PGM-adjusted beta1)
    m_val = eff_beta1 * m_val + (1.0 - eff_beta1) * g
    v_val = beta2 * v_val + (1.0 - beta2) * g * g

    # Bias-corrected (base beta1)
    m_hat = m_val / bias_correction1
    v_hat = v_val / bias_correction2

    # LVS: adaptive LR
    if entropy_adaptive:
        eff_lr = lr * entropy_lr_scale
    else:
        eff_lr = lr

    # Update
    p = p - eff_lr * m_hat / (tl.sqrt(v_hat) + eps)

    # Store
    tl.store(param_ptr + offsets, p, mask=mask)
    tl.store(m_ptr + offsets, m_val, mask=mask)
    tl.store(v_ptr + offsets, v_val, mask=mask)


def velvet_update_triton(
    param: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    bias_correction1: float,
    bias_correction2: float,
    entropy_adaptive: bool = False,
    entropy_lr_scale: float = 1.0,
    perplexity_guided: bool = False,
    ppl_momentum_scale: float = 1.0,
    sparse_aware: bool = False,
):
    """Fused Velvet optimizer update — single Triton kernel per parameter."""
    n = param.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    # Flatten for contiguous access
    p_flat = param.view(-1)
    g_flat = grad.view(-1).to(torch.float32)
    m_flat = m.view(-1)
    v_flat = v.view(-1)

    # Kernel operates in F32
    orig_dtype = p_flat.dtype
    if orig_dtype != torch.float32:
        p_flat_f32 = p_flat.float()
    else:
        p_flat_f32 = p_flat

    _velvet_update_kernel[grid](
        p_flat_f32, g_flat, m_flat, v_flat,
        lr, beta1, beta2, eps, wd,
        bias_correction1, bias_correction2,
        entropy_adaptive, entropy_lr_scale,
        perplexity_guided, ppl_momentum_scale,
        sparse_aware,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Cast back
    if orig_dtype != torch.float32:
        p_flat.copy_(p_flat_f32.to(orig_dtype))
