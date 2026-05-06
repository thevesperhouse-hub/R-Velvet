"""ERA Activation — Triton fused kernel.

ERA(x) = GELU(x) + gamma * softplus(x)

Fuses GELU + softplus + add into a single kernel pass.
Standard PyTorch would launch 5+ separate kernels.
"""

import torch
import triton
import triton.language as tl
import math


SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


@triton.jit
def _tanh(x):
    """Manual tanh — portable across all Triton versions, numerically stable."""
    # Clamp to [-20, 20] to prevent exp overflow (tanh saturates to +-1 beyond ~10)
    x_clamped = tl.minimum(tl.maximum(x, -20.0), 20.0)
    exp2x = tl.exp(2.0 * x_clamped)
    return (exp2x - 1.0) / (exp2x + 1.0)


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

    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)

    # GELU (tanh approximation):
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)  # sqrt(2/pi)
    tanh_inner = _tanh(inner)
    gelu = 0.5 * x * (1.0 + tanh_inner)

    # softplus(x) = log(1 + exp(x)), numerically stable
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))

    # ERA = GELU + gamma * softplus
    result = gelu + gamma * sp

    tl.store(out_ptr + offs, result, mask=mask)


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

    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offs, mask=mask).to(tl.float32)

    # d(GELU)/dx (tanh approximation derivative)
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    tanh_inner = _tanh(inner)
    sech2 = 1.0 - tanh_inner * tanh_inner
    d_inner = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
    d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner

    # d(softplus)/dx = sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    d_sp = sigmoid_x

    grad_in = grad_out * (d_gelu + gamma * d_sp)
    tl.store(grad_in_ptr + offs, grad_in, mask=mask)


class ERAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma):
        n = x.numel()
        out = torch.empty_like(x)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _era_forward_kernel[grid](
            x.contiguous().view(-1), out.view(-1),
            gamma, n, BLOCK_SIZE=BLOCK,
        )
        ctx.save_for_backward(x)
        ctx.gamma = gamma
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        n = x.numel()
        grad_in = torch.empty_like(x)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _era_backward_kernel[grid](
            x.contiguous().view(-1),
            grad_output.contiguous().view(-1),
            grad_in.view(-1),
            ctx.gamma, n, BLOCK_SIZE=BLOCK,
        )
        return grad_in, None


def era_forward_triton(x: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """ERA activation with fused Triton kernel (forward + backward)."""
    return ERAFunction.apply(x, gamma)
