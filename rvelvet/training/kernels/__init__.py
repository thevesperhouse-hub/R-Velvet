"""Kernel dispatch — auto-selects Triton > CUDA > PyTorch fallback."""

import torch
import warnings

HAS_TRITON = False
HAS_CUDA_EXT = False

# --- Triton (preferred — fused kernels, no compilation step) ---
try:
    import triton
    HAS_TRITON = True
except ImportError:
    pass

# --- Native CUDA (fallback — compiled via cpp_extension at first import) ---
if not HAS_TRITON and torch.cuda.is_available():
    try:
        from .velvet_cuda import velvet_update_cuda
        HAS_CUDA_EXT = True
    except (ImportError, RuntimeError, OSError) as e:
        warnings.warn(f"CUDA kernel compilation failed, falling back to PyTorch: {e}")

# --- Transformer Engine (FP8 — Ada/Hopper/Blackwell GPUs) ---
HAS_TE = False
try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    pass

# --- GPU architecture detection ---
GPU_ARCH = "unknown"
GPU_SM = 0

def get_gpu_arch() -> tuple:
    """Detect GPU arch. Returns (name, sm_version). SM89=Ada, SM90=Hopper, SM100=Blackwell."""
    if not torch.cuda.is_available():
        return ("cpu", 0)
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm >= 100:
        return ("Blackwell", sm)
    elif sm >= 90:
        return ("Hopper", sm)
    elif sm >= 89:
        return ("Ada Lovelace", sm)
    elif sm >= 80:
        return ("Ampere", sm)
    elif sm >= 75:
        return ("Turing", sm)
    elif sm >= 70:
        return ("Volta", sm)
    return (f"SM{sm}", sm)

GPU_ARCH, GPU_SM = get_gpu_arch()

def fp8_available() -> bool:
    """True if TE installed + GPU SM89+."""
    return HAS_TE and GPU_SM >= 89

# --- Dispatch kernel references ---
velvet_update_kernel = None
cross_entropy_kernel = None
era_kernel = None

if HAS_TRITON:
    from .velvet_triton import velvet_update_triton
    from .fused_ce import fused_cross_entropy
    from .era_triton import era_forward_triton
    velvet_update_kernel = velvet_update_triton
    cross_entropy_kernel = fused_cross_entropy
    era_kernel = era_forward_triton
elif HAS_CUDA_EXT:
    from .velvet_cuda import velvet_update_cuda
    velvet_update_kernel = velvet_update_cuda
    # No fused CE or ERA in CUDA path — those fall back to PyTorch


def get_backend() -> str:
    if HAS_TRITON:
        return "triton"
    elif HAS_CUDA_EXT:
        return "cuda"
    else:
        return "pytorch"
