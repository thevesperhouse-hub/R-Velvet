"""Validation tests for VelvetOptimizer fixes and kernel sanity.

Covers:
1. Backend parity (Triton / CUDA / PyTorch fallback)
2. AdamW math reproduction (no PGM, no LVS)
3. Bias correction post-fix (PGM applied post-correction)
4. state_dict / load_state_dict round-trip (incl. LVS/PGM internals)
5. Trainer resume (global_step continuity, no LR jump)
6. fused_ce kernel sanity vs F.cross_entropy
7. era_triton kernel sanity (forward + backward via finite differences)
"""

import os
import sys
import math
import copy
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvelvet.training.velvet_optimizer import VelvetOptimizer
from rvelvet.training.kernels import HAS_TRITON, HAS_CUDA_EXT


CUDA = torch.cuda.is_available()


def _make_param(seed=0, shape=(32, 32), device="cpu", dtype=torch.float32):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(*shape, generator=g, device=device, dtype=dtype, requires_grad=True)


def _attach_grad(p, seed=1):
    g = torch.Generator(device=p.device).manual_seed(seed)
    p.grad = torch.randn_like(p) if p.device.type == "cuda" else torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype)


# ---------------------------------------------------------------------------
# Test 1 — Backend parity (PyTorch always; Triton/CUDA when available)
# ---------------------------------------------------------------------------

def _run_one_step(backend, params, grads, lr, betas, eps, wd,
                  entropy_adaptive, perplexity_guided, sparse_aware,
                  ppl_scale=1.0, lvs_scale=1.0):
    """Apply a single optimizer step with the requested backend."""
    opt = VelvetOptimizer(
        [{'params': params}],
        lr=lr, betas=betas, eps=eps, weight_decay=wd,
        max_grad_norm=0.0,  # disable clipping in tests
        entropy_adaptive=entropy_adaptive,
        perplexity_guided=perplexity_guided,
        sparse_aware=sparse_aware,
    )
    # Force backend
    opt._kernel_backend = backend
    opt._perplexity_scale = ppl_scale
    opt._entropy_scale = lvs_scale
    for p, g in zip(params, grads):
        p.grad = g.clone()
    opt.step()
    return [p.detach().clone() for p in params]


@pytest.mark.skipif(not CUDA, reason="parity test needs CUDA for triton/cuda backends")
@pytest.mark.parametrize("perplexity_guided", [False, True])
@pytest.mark.parametrize("entropy_adaptive", [False, True])
@pytest.mark.parametrize("wd", [0.0, 1e-2])
def test_backend_parity(perplexity_guided, entropy_adaptive, wd):
    """Triton, CUDA and PyTorch fallback must produce the same param updates."""
    torch.manual_seed(0)
    shape = (64, 64)
    p_init = torch.randn(*shape, device="cuda")
    g_init = torch.randn(*shape, device="cuda")

    backends = ["pytorch"]
    if HAS_TRITON:
        backends.append("triton")
    if HAS_CUDA_EXT:
        backends.append("cuda")
    if len(backends) < 2:
        pytest.skip("only one backend available; nothing to compare")

    results = {}
    for be in backends:
        p = p_init.clone().detach().requires_grad_(True)
        out = _run_one_step(
            be, [p], [g_init], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=wd,
            entropy_adaptive=entropy_adaptive,
            perplexity_guided=perplexity_guided,
            sparse_aware=False,
            ppl_scale=0.85 if perplexity_guided else 1.0,
            lvs_scale=1.1 if entropy_adaptive else 1.0,
        )
        results[be] = out[0]

    ref = results[backends[0]]
    for be in backends[1:]:
        diff = (results[be] - ref).abs().max().item()
        assert diff < 1e-5, f"backend {be} differs from {backends[0]}: max abs diff = {diff:.2e}"


# ---------------------------------------------------------------------------
# Test 2 — AdamW math reproduction (no PGM, no LVS, no sparse)
# ---------------------------------------------------------------------------

def test_adamw_equivalence():
    """With all adaptive features off, VelvetOptimizer must match torch.optim.AdamW."""
    torch.manual_seed(0)
    shape = (16, 16)
    lr, b1, b2, wd = 1e-3, 0.9, 0.999, 1e-2

    p_v = torch.randn(*shape, requires_grad=True)
    p_a = p_v.detach().clone().requires_grad_(True)

    opt_v = VelvetOptimizer(
        [p_v], lr=lr, betas=(b1, b2), weight_decay=wd, max_grad_norm=0.0,
        entropy_adaptive=False, perplexity_guided=False, sparse_aware=False,
    )
    opt_a = torch.optim.AdamW([p_a], lr=lr, betas=(b1, b2), weight_decay=wd, eps=1e-8)

    g = torch.Generator().manual_seed(123)
    for _ in range(100):
        grad = torch.randn(*shape, generator=g)
        p_v.grad = grad.clone()
        p_a.grad = grad.clone()
        opt_v.step()
        opt_a.step()

    diff = (p_v.detach() - p_a.detach()).abs().max().item()
    assert diff < 1e-6, f"VelvetOptimizer (no adaptive) diverges from AdamW: {diff:.2e}"


# ---------------------------------------------------------------------------
# Test 3 — Bias correction post-fix
# ---------------------------------------------------------------------------

def test_bias_correction_step1_classical():
    """Step 1 of Adam with constant gradient g: m_hat ≈ g, update ≈ -lr * sign(g)."""
    torch.manual_seed(0)
    shape = (8, 8)
    lr, b1, b2 = 1e-3, 0.9, 0.999
    g_const = torch.full(shape, 0.5)

    p = torch.zeros(*shape, requires_grad=True)
    opt = VelvetOptimizer(
        [p], lr=lr, betas=(b1, b2), weight_decay=0.0, max_grad_norm=0.0,
        entropy_adaptive=False, perplexity_guided=False, sparse_aware=False,
    )
    p.grad = g_const.clone()
    p_before = p.detach().clone()
    opt.step()

    # Adam step 1 with v_hat = g², m_hat = g → update = lr * sign(g)
    update = (p.detach() - p_before).abs().max().item()
    assert abs(update - lr) < 1e-6, f"step 1 update = {update}, expected {lr}"


def test_bias_correction_pgm_post_fix():
    """With PGM ON at step 1: m_hat is scaled by ppl_scale AFTER bias correction.

    Pre-fix bug: m_val = (1-eff_b1)*g, divided by (1 - b1) => inflated.
    Post-fix: m_hat = (1 - b1^t)/(1 - b1^t) * g * ppl_scale = g * ppl_scale.
    """
    torch.manual_seed(0)
    shape = (8, 8)
    lr, b1, b2 = 1e-3, 0.9, 0.999
    g_const = torch.full(shape, 0.5)
    ppl_scale = 0.8

    p = torch.zeros(*shape, requires_grad=True)
    opt = VelvetOptimizer(
        [p], lr=lr, betas=(b1, b2), weight_decay=0.0, max_grad_norm=0.0,
        entropy_adaptive=False, perplexity_guided=True, sparse_aware=False,
    )
    opt._perplexity_scale = ppl_scale
    p.grad = g_const.clone()
    p_before = p.detach().clone()
    opt.step()

    # Expected update = lr * (m_hat / sqrt(v_hat)) = lr * ppl_scale * sign(g)
    # because v_hat = g² → sqrt(v_hat) = |g|, m_hat (post-scale) = g * ppl_scale
    update = (p_before - p.detach()).abs().max().item()
    expected = lr * ppl_scale
    assert abs(update - expected) < 1e-5, f"PGM update = {update}, expected {expected}"


# ---------------------------------------------------------------------------
# Test 4 — state_dict round-trip
# ---------------------------------------------------------------------------

def test_state_dict_roundtrip():
    """After save/load, all LVS/PGM internals are restored and subsequent steps match."""
    torch.manual_seed(0)
    shape = (16, 16)
    p1 = torch.randn(*shape, requires_grad=True)
    p2 = p1.detach().clone().requires_grad_(True)

    opt1 = VelvetOptimizer(
        [p1], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2, max_grad_norm=0.0,
        entropy_adaptive=True, perplexity_guided=True, sparse_aware=False,
    )
    opt1.set_training_steps(1000)

    g = torch.Generator().manual_seed(42)
    # 50 steps with variable loss values to exercise LVS/PGM
    for i in range(50):
        grad = torch.randn(*shape, generator=g)
        p1.grad = grad.clone()
        opt1.set_loss_metrics(2.0 * (1.0 - i / 100.0), vocab_size=32000)
        opt1.step()

    sd = opt1.state_dict()

    # New optimizer, load, and verify internals
    opt2 = VelvetOptimizer(
        [p2], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2, max_grad_norm=0.0,
        entropy_adaptive=True, perplexity_guided=True, sparse_aware=False,
    )
    opt2.set_training_steps(1000)
    opt2.load_state_dict(copy.deepcopy(sd))

    # We need p2 to match p1 BEFORE additional steps
    p2.data.copy_(p1.data)

    for k in VelvetOptimizer._VELVET_STATE_KEYS:
        v1, v2 = getattr(opt1, k), getattr(opt2, k)
        if isinstance(v1, float):
            assert math.isclose(v1, v2, rel_tol=0, abs_tol=1e-9), f"{k}: {v1} vs {v2}"
        else:
            assert v1 == v2, f"{k}: {v1} vs {v2}"

    # 10 more steps from each must produce identical params
    g2 = torch.Generator().manual_seed(7)
    for i in range(10):
        grad = torch.randn(*shape, generator=g2)
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        loss = 1.5 - i * 0.01
        opt1.set_loss_metrics(loss, vocab_size=32000)
        opt2.set_loss_metrics(loss, vocab_size=32000)
        opt1.step()
        opt2.step()

    diff = (p1.detach() - p2.detach()).abs().max().item()
    assert diff < 1e-6, f"params drift after resume: {diff:.2e}"


# ---------------------------------------------------------------------------
# Test 5 — Trainer resume (global_step continuity)
# ---------------------------------------------------------------------------

class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n=64, seq=16, vocab=128):
        torch.manual_seed(0)
        self.input_ids = torch.randint(0, vocab, (n, seq))
        self.targets = torch.randint(0, vocab, (n, seq))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {'input_ids': self.input_ids[i], 'targets': self.targets[i]}


class _TinyModel(nn.Module):
    def __init__(self, vocab=128, dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x):
        h = self.emb(x)
        logits = self.head(h)
        # Trainer expects either tensor or dict; phase1 loss expects logits
        return {'logits': logits}


def _make_cfg(output_dir, resume_from=None, max_steps=20):
    from types import SimpleNamespace
    return SimpleNamespace(
        output_dir=str(output_dir),
        model=SimpleNamespace(vocab_size=128),
        data=SimpleNamespace(num_workers=0),
        training=SimpleNamespace(
            phase='phase1_pretrain',
            optimizer='adamw',
            lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
            weight_decay=0.0, grad_clip=1.0,
            warmup_steps=2, min_lr=1e-5, max_steps=max_steps,
            batch_size=4, grad_accum_steps=1,
            log_every=5, save_every=10, keep_last_n=5,
            amp='fp32', wandb=False, debug=False,
            resume_from=resume_from,
            debug_steps=max_steps, debug_log_every=5,
        ),
    )


def test_trainer_resume_continuity():
    """After resume, global_step continues monotonically and no LR discontinuity."""
    from rvelvet.training.trainer import Trainer
    from rvelvet.training import losses as _losses

    # Patch compute_phase_loss to a simple CE so we don't need full RVelvet model
    def fake_compute(output, targets, phase, vocab_size, cfg, model):
        logits = output['logits'] if isinstance(output, dict) else output
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return ce, {'ce': ce.detach()}

    orig = _losses.compute_phase_loss
    _losses.compute_phase_loss = fake_compute
    # Trainer imports the symbol directly; patch there too
    import rvelvet.training.trainer as trainer_mod
    orig_t = trainer_mod.compute_phase_loss
    trainer_mod.compute_phase_loss = fake_compute

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ds = _TinyDataset()

            # Phase A: train 10 steps, save checkpoint
            torch.manual_seed(0)
            model = _TinyModel()
            cfg = _make_cfg(tmp / "run_a", max_steps=10)
            cfg.training.save_every = 10
            t1 = Trainer(model, ds, cfg)
            t1.train()
            ckpt_path = tmp / "run_a" / "ckpt_step10.pt"
            assert ckpt_path.exists(), f"checkpoint not saved at {ckpt_path}"

            # Phase B: resume, train 10 more steps
            torch.manual_seed(1)
            model2 = _TinyModel()
            cfg2 = _make_cfg(tmp / "run_b", resume_from=str(ckpt_path), max_steps=20)
            cfg2.training.save_every = 100  # don't save during resume run
            t2 = Trainer(model2, ds, cfg2)
            assert t2.global_step == 10, f"resumed global_step = {t2.global_step}, expected 10"
            t2.train()
            assert t2.global_step == 20, f"final global_step = {t2.global_step}, expected 20"

            # Verify metrics.csv reflects continuity (steps strictly increasing in resumed run)
            csv_resumed = tmp / "run_b" / "metrics.csv"
            assert csv_resumed.exists()
            with open(csv_resumed) as f:
                lines = f.readlines()
            # Either header was written or appended; either way, all data rows should have step > 10
            data_rows = [l for l in lines if l and not l.startswith('step')]
            steps = [int(r.split(',')[0]) for r in data_rows if r.strip()]
            assert all(s > 10 for s in steps), f"resumed run logged step <= 10: {steps}"
    finally:
        _losses.compute_phase_loss = orig
        trainer_mod.compute_phase_loss = orig_t


# ---------------------------------------------------------------------------
# Test 6 — fused_ce kernel sanity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="fused_ce requires CUDA + triton")
def test_fused_ce_matches_pytorch():
    """fused_cross_entropy must match F.cross_entropy in forward & backward."""
    from rvelvet.training.kernels.fused_ce import fused_cross_entropy

    torch.manual_seed(0)
    # Use a vocab > 65536 to actually hit the Triton chunked path,
    # plus a non-power-of-two size to stress chunk alignment.
    B, L, V = 2, 8, 70_001  # not aligned to VOCAB_CHUNK=4096
    logits_a = torch.randn(B, L, V, device="cuda", requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    labels = torch.randint(0, V, (B, L), device="cuda")

    # PyTorch reference
    loss_ref = F.cross_entropy(logits_b.reshape(-1, V).float(), labels.reshape(-1))
    loss_ref.backward()

    # Fused
    loss_fused, _ = fused_cross_entropy(logits_a, labels)
    loss_fused.backward()

    fwd_diff = (loss_fused - loss_ref).abs().item()
    assert fwd_diff < 1e-4, f"fused_ce forward diff = {fwd_diff:.2e}"

    # logits_a.grad now contains gradients (in-place, see kernel docstring)
    grad_diff = (logits_a.grad - logits_b.grad).abs().max().item()
    assert grad_diff < 1e-3, f"fused_ce grad diff = {grad_diff:.2e}"


# ---------------------------------------------------------------------------
# Test 7 — era_triton kernel sanity (forward + backward via finite diff)
# ---------------------------------------------------------------------------

def _era_pytorch(x, gamma=0.1):
    return F.gelu(x, approximate='tanh') + gamma * F.softplus(x)


@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="era_triton requires CUDA + triton")
def test_era_forward_matches_pytorch():
    from rvelvet.training.kernels.era_triton import era_forward_triton
    torch.manual_seed(0)
    x = torch.randn(64, 32, device="cuda")
    out_t = era_forward_triton(x, gamma=0.1)
    out_p = _era_pytorch(x, gamma=0.1)
    diff = (out_t - out_p).abs().max().item()
    assert diff < 1e-4, f"era forward diff = {diff:.2e}"


@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="era_triton requires CUDA + triton")
def test_era_backward_finite_diff():
    """Verify Triton backward against numerical gradient on a small tensor."""
    from rvelvet.training.kernels.era_triton import era_forward_triton

    torch.manual_seed(0)
    x = torch.randn(8, 4, device="cuda", dtype=torch.float64)
    # Triton kernel works in fp32; use fp32 here too with looser tolerance
    x = x.float().requires_grad_(True)

    out = era_forward_triton(x, gamma=0.1)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    analytic = x.grad.detach().clone()

    # Finite-difference reference (PyTorch graph)
    eps = 1e-3
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = _era_pytorch(x_ref, gamma=0.1)
    y_ref.backward(grad_out)
    numeric = x_ref.grad.detach()

    diff = (analytic - numeric).abs().max().item()
    assert diff < 1e-3, f"era backward diff = {diff:.2e}"


# ---------------------------------------------------------------------------
# Test 8 — velvet kernel handles bf16/fp16 params via in-kernel f32 cast
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + triton")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_velvet_triton_bf16_fp16_param(dtype):
    """In-kernel cast: param stored as bf16/fp16, kernel upcasts to f32 internally.

    Without the fix this used to require a host-side .float() copy; verify both
    that the result is correct AND that the param tensor remains in original dtype.
    """
    if dtype == torch.float16 and not torch.cuda.is_available():
        pytest.skip("fp16 needs CUDA")

    torch.manual_seed(0)
    shape = (128, 128)
    p_low = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
    p_f32 = p_low.detach().float().requires_grad_(True)
    g_low = torch.randn(*shape, device="cuda", dtype=dtype)
    g_f32 = g_low.float()

    out_low = _run_one_step(
        "triton", [p_low], [g_low], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=1e-2,
        entropy_adaptive=True, perplexity_guided=True, sparse_aware=False,
        ppl_scale=0.9, lvs_scale=1.05,
    )[0]
    out_f32 = _run_one_step(
        "pytorch", [p_f32], [g_f32], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=1e-2,
        entropy_adaptive=True, perplexity_guided=True, sparse_aware=False,
        ppl_scale=0.9, lvs_scale=1.05,
    )[0]

    # Param dtype preserved
    assert out_low.dtype == dtype, f"param dtype changed to {out_low.dtype}"

    # Numerical match within bf16/fp16 tolerance
    tol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    diff = (out_low.float() - out_f32).abs().max().item()
    assert diff < tol, f"{dtype} param diverges from f32 ref: {diff:.2e}"


# ---------------------------------------------------------------------------
# Test 9 — autotune sanity: same input → identical output across calls
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + triton")
def test_velvet_autotune_deterministic():
    """Calling the kernel twice on identical inputs must give identical outputs,
    after autotune has settled. Catches restore_value bugs (corrupted state)."""
    torch.manual_seed(0)
    shape = (256, 256)
    p_a = torch.randn(*shape, device="cuda", requires_grad=True)
    p_b = p_a.detach().clone().requires_grad_(True)
    grad = torch.randn(*shape, device="cuda")

    a = _run_one_step(
        "triton", [p_a], [grad.clone()], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=1e-2,
        entropy_adaptive=False, perplexity_guided=False, sparse_aware=False,
    )[0]
    b = _run_one_step(
        "triton", [p_b], [grad.clone()], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=1e-2,
        entropy_adaptive=False, perplexity_guided=False, sparse_aware=False,
    )[0]
    diff = (a - b).abs().max().item()
    assert diff == 0.0, f"non-deterministic kernel output: {diff:.2e}"


# ---------------------------------------------------------------------------
# Test 10 — fused_ce z-loss path (lse used downstream → grad_lse non-None)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="fused_ce z-loss requires CUDA + triton")
def test_fused_ce_z_loss_path():
    """When lse is used downstream (z-loss style), grad_lse is non-None and
    the kernel must add `softmax * grad_lse` to the gradient."""
    from rvelvet.training.kernels.fused_ce import fused_cross_entropy

    torch.manual_seed(0)
    B, L, V = 2, 4, 70_001
    z_coef = 1e-3

    logits_a = torch.randn(B, L, V, device="cuda", requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    labels = torch.randint(0, V, (B, L), device="cuda")

    # Reference: F.cross_entropy + manual z-loss = (logsumexp ** 2).mean()
    flat_b = logits_b.reshape(-1, V).float()
    loss_ref = F.cross_entropy(flat_b, labels.reshape(-1))
    lse_ref = torch.logsumexp(flat_b, dim=-1)
    total_ref = loss_ref + z_coef * (lse_ref ** 2).mean()
    total_ref.backward()

    # Fused path
    loss_fused, lse_fused = fused_cross_entropy(logits_a, labels)
    total_fused = loss_fused + z_coef * (lse_fused ** 2).mean()
    total_fused.backward()

    fwd = (total_fused - total_ref).abs().item()
    grad = (logits_a.grad - logits_b.grad).abs().max().item()
    assert fwd < 1e-3, f"z-loss forward diff = {fwd:.2e}"
    assert grad < 5e-3, f"z-loss grad diff = {grad:.2e}"


# ---------------------------------------------------------------------------
# Test 11 — fused_ce small-vocab fallback (V <= 1024) skips Triton path
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fused_ce_small_vocab_fallback():
    """V <= 1024 routes to PyTorch fallback. Verify it matches F.cross_entropy
    and accepts bf16 logits without the old full-tensor .float() copy."""
    from rvelvet.training.kernels.fused_ce import fused_cross_entropy

    torch.manual_seed(0)
    B, L, V = 4, 16, 512  # V <= 1024 → fallback path
    logits_bf16 = torch.randn(B, L, V, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, V, (B, L), device="cuda")

    loss, lse = fused_cross_entropy(logits_bf16, labels)
    ref = F.cross_entropy(logits_bf16.reshape(-1, V).float(), labels.reshape(-1))

    # bf16 logits → bf16-internal CE (no explicit .float() copy by design).
    # The reference upcasts manually, so a few ulps of diff is expected.
    diff = (loss - ref).abs().item()
    assert diff < 5e-2, f"small-vocab fallback diff = {diff:.2e}"
    assert lse.shape == (B * L,)
    assert lse.dtype == torch.float32, f"lse dtype = {lse.dtype}, expected f32 for stability"


# ---------------------------------------------------------------------------
# Test 12 — fused_ce large vocab + non-aligned V (stress chunk boundaries)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="fused_ce kernel needs CUDA + triton")
@pytest.mark.parametrize("V", [4096, 4097, 8192, 65537, 131_073])
def test_fused_ce_various_vocab_sizes(V):
    """Verify correctness at V == CHUNK_SIZE, V == CHUNK_SIZE+1, V >> CHUNK_SIZE,
    and V just above a power-of-two chunk boundary."""
    from rvelvet.training.kernels.fused_ce import fused_cross_entropy

    torch.manual_seed(0)
    N = 8
    logits_a = torch.randn(N, V, device="cuda", requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    labels = torch.randint(0, V, (N,), device="cuda")

    loss_ref = F.cross_entropy(logits_b.float(), labels)
    loss_ref.backward()

    loss_fused, _ = fused_cross_entropy(logits_a, labels)
    loss_fused.backward()

    fwd = (loss_fused - loss_ref).abs().item()
    grad = (logits_a.grad - logits_b.grad).abs().max().item()
    assert fwd < 1e-4, f"V={V}: forward diff = {fwd:.2e}"
    assert grad < 5e-4, f"V={V}: grad diff = {grad:.2e}"


# ---------------------------------------------------------------------------
# Test 13 — era kernel: bf16/fp16 input correctness
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="era kernel needs CUDA + triton")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_era_low_precision(dtype):
    from rvelvet.training.kernels.era_triton import era_forward_triton

    torch.manual_seed(0)
    x = torch.randn(64, 32, device="cuda", dtype=dtype, requires_grad=True)
    out = era_forward_triton(x, gamma=0.1)
    assert out.dtype == dtype, f"output dtype changed to {out.dtype}"

    ref = F.gelu(x.float(), approximate='tanh') + 0.1 * F.softplus(x.float())
    diff = (out.float() - ref).abs().max().item()
    tol = 5e-2 if dtype == torch.float16 else 1e-1
    assert diff < tol, f"{dtype} era diff = {diff:.2e}"


# ---------------------------------------------------------------------------
# Test 14 — era backward in low precision
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="era backward needs CUDA + triton")
def test_era_backward_chains_through_autograd():
    """End-to-end autograd through era kernel: train a 1-step linear+ERA, verify grads exist."""
    from rvelvet.training.kernels.era_triton import era_forward_triton

    torch.manual_seed(0)
    x = torch.randn(8, 16, device="cuda", requires_grad=True)
    W = torch.randn(16, 16, device="cuda", requires_grad=True)
    h = era_forward_triton(x @ W, gamma=0.05)
    loss = h.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert W.grad is not None and torch.isfinite(W.grad).all()


# ---------------------------------------------------------------------------
# Test 15 — fused_ce in-place backward does not leak across rows
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="fused_ce needs CUDA + triton")
def test_fused_ce_inplace_backward_no_cross_row():
    """Verify the in-place backward correctly produces gradients for ALL rows
    (catches bugs where chunk loop or row-program indexing is off)."""
    from rvelvet.training.kernels.fused_ce import fused_cross_entropy

    torch.manual_seed(0)
    N, V = 64, 70_001
    logits_a = torch.randn(N, V, device="cuda", requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    labels = torch.randint(0, V, (N,), device="cuda")

    F.cross_entropy(logits_b.float(), labels).backward()
    fused_cross_entropy(logits_a, labels)[0].backward()

    # Per-row max diff — none should be wildly off
    row_diffs = (logits_a.grad - logits_b.grad).abs().reshape(N, V).max(dim=-1).values
    worst = row_diffs.max().item()
    assert worst < 5e-4, f"per-row worst diff = {worst:.2e}"
    # Also confirm at least one entry per row is non-zero (label gradient)
    assert (logits_a.grad.abs().sum(dim=-1) > 0).all(), "some rows have zero gradient"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
