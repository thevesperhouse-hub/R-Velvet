"""Velvet Optimizer — CUDA kernel via torch.utils.cpp_extension (PGM + LVS).

Compiles the CUDA kernel at first import. Subsequent imports use cached .so.
Triton path is preferred (no compilation step), but this serves as fallback.
"""

import os
import torch

_cuda_module = None


VELVET_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void velvet_complete_kernel(
    float* params, float* m, float* v, const float* grads,
    float base_lr, float beta1, float beta2, float eps, float wd,
    float bias_correction1, float bias_correction2, long long n,
    bool entropy_adaptive, float entropy_lr_scale,
    bool perplexity_guided, float ppl_momentum_scale,
    bool sparse_aware
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float p = params[idx];

    // Sparse: skip near-zero weights
    if (sparse_aware && fabsf(p) < 1e-9f) return;

    float g = grads[idx];
    float m_val = m[idx];
    float v_val = v[idx];

    // Decoupled weight decay (AdamW)
    p *= (1.0f - base_lr * wd);

    // Classical Adam moment update (constant beta1) — bias-correction math preserved
    m_val = beta1 * m_val + (1.0f - beta1) * g;
    v_val = beta2 * v_val + (1.0f - beta2) * g * g;

    // Bias-corrected
    float m_hat = m_val / bias_correction1;
    float v_hat = v_val / bias_correction2;

    // PGM: post-correction momentum scaling (no EMA drift)
    if (perplexity_guided) {
        m_hat *= ppl_momentum_scale;
    }

    // LVS: adaptive LR
    float effective_lr = base_lr;
    if (entropy_adaptive) effective_lr *= entropy_lr_scale;

    // Update
    p -= effective_lr * m_hat / (sqrtf(v_hat) + eps);

    params[idx] = p;
    m[idx] = m_val;
    v[idx] = v_val;
}

void velvet_update_cuda(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor m, torch::Tensor v,
    float lr, float beta1, float beta2, float eps, float wd,
    float bias_correction1, float bias_correction2,
    bool entropy_adaptive, float entropy_lr_scale,
    bool perplexity_guided, float ppl_momentum_scale,
    bool sparse_aware
) {
    long long n = param.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    velvet_complete_kernel<<<blocks, threads>>>(
        param.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(),
        grad.data_ptr<float>(),
        lr, beta1, beta2, eps, wd,
        bias_correction1, bias_correction2, n,
        entropy_adaptive, entropy_lr_scale,
        perplexity_guided, ppl_momentum_scale,
        sparse_aware
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("velvet_update_cuda", &velvet_update_cuda, "Velvet optimizer CUDA kernel");
}
"""


def _load_cuda_extension():
    """Compile and load the CUDA extension (cached after first call)."""
    global _cuda_module
    if _cuda_module is not None:
        return _cuda_module

    from torch.utils.cpp_extension import load_inline

    _cuda_module = load_inline(
        name="velvet_cuda_ext",
        cpp_sources="",
        cuda_sources=VELVET_CUDA_SOURCE,
        functions=["velvet_update_cuda"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return _cuda_module


def velvet_update_cuda(
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
    """Call the native CUDA kernel for Velvet optimizer update."""
    ext = _load_cuda_extension()

    # Kernel requires F32 contiguous
    orig_dtype = param.dtype
    p_f32 = param.float().contiguous()
    g_f32 = grad.float().contiguous()

    ext.velvet_update_cuda(
        p_f32, g_f32, m, v,
        lr, beta1, beta2, eps, wd,
        bias_correction1, bias_correction2,
        entropy_adaptive, entropy_lr_scale,
        perplexity_guided, ppl_momentum_scale,
        sparse_aware,
    )

    if orig_dtype != torch.float32:
        param.copy_(p_f32.to(orig_dtype))
    else:
        param.copy_(p_f32)
