"""ERA Activation — Triton fused kernel.

ERA(x) = GELU(x) + gamma * softplus(x)

Fuses GELU + softplus + add into a single kernel pass (forward and backward).
Standard PyTorch would launch 5+ separate kernels.

Optimizations:
- libdevice.tanh hardware intrinsic (~2-3× faster than manual exp-based tanh)
- tl.sigmoid intrinsic for the softplus derivative
- @triton.autotune over BLOCK_SIZE × num_warps
- Cache eviction hints
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


def _era_autotune_configs():
    return [
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ]


@triton.autotune(
    configs=_era_autotune_configs(),
    key=['n'],
    # ERA forward writes only to out_ptr (independent of input). Restore it so
    # benchmark iterations don't see partially-written buffers.
    restore_value=['out_ptr'],
)
@triton.jit
def _era_forward_kernel(
    x_ptr, out_ptr,
    gamma,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    x = tl.load(x_ptr + offs, mask=mask, eviction_policy='evict_first').to(tl.float32)

    # GELU (tanh approximation):
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)  # sqrt(2/pi)
    tanh_inner = libdevice.tanh(inner)
    gelu = 0.5 * x * (1.0 + tanh_inner)

    # softplus(x) = log(1 + exp(x)), numerically stable for large x
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))

    # ERA = GELU + gamma * softplus
    result = gelu + gamma * sp

    tl.store(out_ptr + offs, result, mask=mask, eviction_policy='evict_last')


@triton.autotune(
    configs=_era_autotune_configs(),
    key=['n'],
    restore_value=['grad_in_ptr'],
)
@triton.jit
def _era_backward_kernel(
    x_ptr, grad_out_ptr, grad_in_ptr,
    gamma,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward pass for ERA activation."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    x = tl.load(x_ptr + offs, mask=mask, eviction_policy='evict_first').to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offs, mask=mask, eviction_policy='evict_first').to(tl.float32)

    # d(GELU)/dx (tanh approximation derivative)
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    tanh_inner = libdevice.tanh(inner)
    sech2 = 1.0 - tanh_inner * tanh_inner
    d_inner = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
    d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner

    # d(softplus)/dx = sigmoid(x)
    d_sp = tl.sigmoid(x)

    grad_in = grad_out * (d_gelu + gamma * d_sp)
    tl.store(grad_in_ptr + offs, grad_in, mask=mask, eviction_policy='evict_last')


class ERAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma):
        x = x.contiguous()
        n = x.numel()
        out = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        _era_forward_kernel[grid](x.view(-1), out.view(-1), gamma, n)
        ctx.save_for_backward(x)
        ctx.gamma = gamma
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        n = x.numel()
        grad_in = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        _era_backward_kernel[grid](
            x.view(-1), grad_output.view(-1), grad_in.view(-1),
            ctx.gamma, n,
        )
        return grad_in, None


def era_forward_triton(x: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """ERA activation with fused Triton kernel (forward + backward)."""
    return ERAFunction.apply(x, gamma)
