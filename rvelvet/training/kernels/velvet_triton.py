"""Velvet Optimizer — Triton fused kernel (PGM + LVS).

Single kernel: weight decay + classical Adam moments + bias correction +
PGM post-correction scale + LVS LR + param update.

Optimizations:
- In-kernel dtype upcast (no host-side .float() copy of param/grad)
- @triton.autotune over BLOCK_SIZE × num_warps
- Cache eviction hints (grad evict_first, param/m/v evict_last)
"""

import torch
import triton
import triton.language as tl


def _velvet_autotune_configs():
    return [
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ]


@triton.autotune(
    configs=_velvet_autotune_configs(),
    key=['n'],
    # In-place kernel: backup/restore these tensors between benchmark runs
    # so timing doesn't accumulate optimizer steps onto the user's state.
    restore_value=['param_ptr', 'm_ptr', 'v_ptr'],
)
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
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load. Param/grad upcast to f32 in-kernel — avoids host-side .float() copies
    # for bf16/fp16 model weights. m and v are always stored in f32.
    p = tl.load(param_ptr + offsets, mask=mask, eviction_policy='evict_last').to(tl.float32)
    g = tl.load(grad_ptr + offsets, mask=mask, eviction_policy='evict_first').to(tl.float32)
    m_val = tl.load(m_ptr + offsets, mask=mask)
    v_val = tl.load(v_ptr + offsets, mask=mask)

    # Sparse: skip near-zero weights
    if sparse_aware:
        active = tl.abs(p) >= 1e-9
        g = tl.where(active, g, 0.0)

    # Decoupled weight decay (AdamW)
    p = p * (1.0 - lr * wd)

    # Classical Adam moments (constant beta1) — bias-correction math preserved
    m_val = beta1 * m_val + (1.0 - beta1) * g
    v_val = beta2 * v_val + (1.0 - beta2) * g * g

    # Bias-corrected
    m_hat = m_val / bias_correction1
    v_hat = v_val / bias_correction2

    # PGM: post-correction momentum scaling (no EMA drift)
    if perplexity_guided:
        m_hat = m_hat * ppl_momentum_scale

    # LVS: adaptive LR
    if entropy_adaptive:
        eff_lr = lr * entropy_lr_scale
    else:
        eff_lr = lr

    # Update
    p = p - eff_lr * m_hat / (tl.sqrt(v_hat) + eps)

    # Store. Triton auto-converts f32 -> param's native dtype on store.
    tl.store(param_ptr + offsets, p, mask=mask, eviction_policy='evict_last')
    tl.store(m_ptr + offsets, m_val, mask=mask, eviction_policy='evict_last')
    tl.store(v_ptr + offsets, v_val, mask=mask, eviction_policy='evict_last')


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

    # Triton autotune picks BLOCK_SIZE; grid is computed from chosen meta.
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    _velvet_update_kernel[grid](
        param.view(-1), grad.view(-1), m.view(-1), v.view(-1),
        lr, beta1, beta2, eps, wd,
        bias_correction1, bias_correction2,
        entropy_adaptive, entropy_lr_scale,
        perplexity_guided, ppl_momentum_scale,
        sparse_aware,
        n,
    )
