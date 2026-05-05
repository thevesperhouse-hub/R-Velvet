"""Quick benchmark to quantify kernel optimization wins (not a regression test).

Run: python tests/_bench_kernels.py
"""
import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvelvet.training.kernels.velvet_triton import velvet_update_triton
from rvelvet.training.kernels.fused_ce import fused_cross_entropy
from rvelvet.training.kernels.era_triton import era_forward_triton


def bench(name, fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / iters
    print(f"  {name:<40} {ms:8.3f} ms")


def main():
    if not torch.cuda.is_available():
        print("No CUDA; skipping benches")
        return

    torch.manual_seed(0)

    print("=== velvet_update — bf16 param (16M elements) ===")
    n = 16 * 1024 * 1024
    p = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    m = torch.zeros(n, device="cuda", dtype=torch.float32)
    v = torch.zeros(n, device="cuda", dtype=torch.float32)
    bench("velvet_triton (autotune+in-kernel cast)", lambda: velvet_update_triton(
        p, g, m, v, 1e-3, 0.9, 0.999, 1e-8, 1e-2, 0.1, 0.001,
        entropy_adaptive=True, entropy_lr_scale=1.05,
        perplexity_guided=True, ppl_momentum_scale=0.9,
        sparse_aware=True,
    ))

    print("\n=== era forward (8M elements, bf16) ===")
    x = torch.randn(8 * 1024 * 1024, device="cuda", dtype=torch.bfloat16)
    bench("era_triton (libdevice.tanh+autotune)", lambda: era_forward_triton(x, 0.1))

    print("\n=== fused_cross_entropy ===")
    for V in [32_000, 65_537, 128_000]:
        N = 1024
        logits = torch.randn(N, V, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, V, (N,), device="cuda")
        bench(f"fused_ce  V={V:>7}", lambda: fused_cross_entropy(logits.clone(), labels))
        bench(f"F.cross_entropy V={V:>7}", lambda: F.cross_entropy(logits.float().reshape(-1, V), labels))


if __name__ == "__main__":
    main()
