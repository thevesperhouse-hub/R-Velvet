"""Invariant tests for VelvetOptimizer hardening.

Covers behaviors that must hold regardless of phase / config:
1. AdamW equivalence when all Velvet adaptive flags are off.
2. NaN/Inf gradients are skipped without mutating state (m, v, EMAs, step, scales).
3. State-dict version round-trip + forward-compat warning on unknown version.
4. Configurable thresholds reach the optimizer (e.g. burst_warmup_steps).
"""

import os
import sys
import copy
import warnings

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvelvet.training.velvet_optimizer import VelvetOptimizer


def _seed_everything(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_pair(shape=(64, 64), device="cpu"):
    """Return two parameter tensors with identical init, on the same device."""
    _seed_everything(0)
    a = torch.randn(*shape, device=device, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    return a, b


# ----------------------------------------------------------------------
# 1. AdamW equivalence with all Velvet adaptivity disabled
# ----------------------------------------------------------------------
def test_adamw_equivalence_no_adaptivity():
    """With entropy/PGM/sparse all off and no set_loss_metrics calls,
    Velvet must match torch.optim.AdamW exactly (cpu fallback path)."""
    a, b = _make_pair()
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01

    velvet = VelvetOptimizer(
        [a], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd,
        entropy_adaptive=False, perplexity_guided=False, sparse_aware=False,
        max_grad_norm=1e9,  # effectively disable clipping
    )
    adamw = torch.optim.AdamW([b], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)

    _seed_everything(42)
    grads = [torch.randn_like(a) * 0.01 for _ in range(40)]

    for g in grads:
        a.grad = g.clone()
        b.grad = g.clone()
        velvet.step()
        adamw.step()
        velvet.zero_grad()
        adamw.zero_grad()

    # Tight tolerance: same math, same fp32 path.
    assert torch.allclose(a.data, b.data, atol=1e-5, rtol=1e-5), \
        f"max diff = {(a - b).abs().max().item():.3e}"


# ----------------------------------------------------------------------
# 2. NaN/Inf grad skip preserves all state
# ----------------------------------------------------------------------
def test_nonfinite_grad_skip_preserves_state():
    """A step with NaN/Inf grads must early-return without touching m, v, step,
    or any LVS/PGM scale. The very next clean step should look identical to
    a step taken without the bad one."""
    a = torch.randn(16, 16, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)

    opt = VelvetOptimizer([a], lr=1e-3, max_grad_norm=1e9, skip_nonfinite=True)
    opt_ref = VelvetOptimizer([a_ref], lr=1e-3, max_grad_norm=1e9, skip_nonfinite=True)

    _seed_everything(7)
    g_clean = torch.randn_like(a) * 0.01

    # opt: bad step then clean step
    a.grad = torch.full_like(a, float("nan"))
    skipped_before = opt.skipped_steps
    opt.step()
    assert opt.skipped_steps == skipped_before + 1, "skipped counter must increment"
    opt.zero_grad()

    # State should still be empty (no entries added).
    state = opt.state.get(a, {})
    assert state.get("step", 0) == 0, "step counter must not advance on bad grad"
    assert "m" not in state or state["m"].abs().max().item() == 0.0, \
        "moment m must not be updated on bad grad"

    # Now a clean step on opt
    a.grad = g_clean.clone()
    opt.step()
    opt.zero_grad()

    # opt_ref: just one clean step
    a_ref.grad = g_clean.clone()
    opt_ref.step()
    opt_ref.zero_grad()

    assert torch.allclose(a.data, a_ref.data, atol=1e-6), \
        "params after [skip, clean] must match a single clean step"


def test_nonfinite_grad_skip_off_does_not_skip():
    """When skip_nonfinite=False, the optimizer should not increment the skip
    counter (it tries to step; the kernel/fallback either NaN-poisons params or
    works with whatever it gets — we only assert the counter doesn't move)."""
    a = torch.randn(8, 8, requires_grad=True)
    opt = VelvetOptimizer([a], lr=1e-3, skip_nonfinite=False)
    a.grad = torch.full_like(a, float("inf"))
    before = opt.skipped_steps
    try:
        opt.step()
    except Exception:
        # Some backends may error on inf; that's fine — counter still must not increment.
        pass
    assert opt.skipped_steps == before


# ----------------------------------------------------------------------
# 3. state_dict version round-trip
# ----------------------------------------------------------------------
def test_state_dict_round_trip_preserves_velvet_state():
    p = torch.randn(32, 32, requires_grad=True)
    opt = VelvetOptimizer([p], lr=1e-3)

    _seed_everything(1)
    for _ in range(20):
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        opt.zero_grad()
    # Drive LVS/PGM signal so internal scales differ from defaults.
    for loss in [4.5, 4.2, 4.1, 4.05, 4.04]:
        opt.set_loss_metrics(loss, vocab_size=32000)

    sd = opt.state_dict()
    assert "velvet_state" in sd, "state_dict must embed velvet_state"
    assert sd.get("velvet_version") == VelvetOptimizer.STATE_DICT_VERSION, \
        "version field must accompany velvet_state for forward-compat checks"

    # Fresh optimizer, identical params, restore.
    p2 = p.detach().clone().requires_grad_(True)
    opt2 = VelvetOptimizer([p2], lr=1e-3)
    opt2.load_state_dict(sd)

    for attr in ("_ema_current", "_ema_anchor", "_global_step",
                 "_perplexity_scale", "_entropy_scale"):
        v1 = getattr(opt, attr)
        v2 = getattr(opt2, attr)
        assert v1 == v2 or (v1 is None and v2 is None), \
            f"{attr} mismatch after round-trip: {v1} vs {v2}"


def test_state_dict_unknown_version_is_handled():
    """Loading a state_dict with a future/unknown version must not crash;
    it should either accept (forward-compat) or warn and fall back gracefully."""
    p = torch.randn(8, 8, requires_grad=True)
    opt = VelvetOptimizer([p], lr=1e-3)
    p.grad = torch.randn_like(p) * 0.01
    opt.step()
    opt.zero_grad()

    sd = opt.state_dict()
    sd["velvet_version"] = 9999  # pretend future version

    p2 = p.detach().clone().requires_grad_(True)
    opt2 = VelvetOptimizer([p2], lr=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        # Must not raise. Behaviour: either load best-effort or ignore the velvet block.
        try:
            opt2.load_state_dict(sd)
        except Exception as e:
            pytest.fail(f"load_state_dict raised on unknown version: {e}")


# ----------------------------------------------------------------------
# 4. Configurable thresholds are actually applied
# ----------------------------------------------------------------------
def test_burst_warmup_blocks_early_bursts():
    """With a high burst_warmup_steps, no burst should fire before that step
    even if loss plateaus. Validates the configurable plumbing."""
    p = torch.randn(8, 8, requires_grad=True)
    opt = VelvetOptimizer(
        [p], lr=1e-3,
        burst_warmup_steps=10_000,
        plateau_patience=5,
        plateau_threshold=10.0,  # any small loss change triggers "plateau"
    )

    # Drive 50 steps with flat loss to try to provoke a burst.
    for i in range(50):
        p.grad = torch.randn_like(p) * 1e-3
        opt.step()
        opt.zero_grad()
        opt.set_loss_metrics(4.0, vocab_size=32000)

    assert not opt.is_bursting, \
        "Burst must not fire before burst_warmup_steps even on plateau"


def test_pgm_bounds_respected():
    """perplexity_scale must always lie within [pgm_min_scale, pgm_max_scale]."""
    p = torch.randn(8, 8, requires_grad=True)
    opt = VelvetOptimizer([p], lr=1e-3, pgm_min_scale=0.85, pgm_max_scale=1.15)

    # Hammer set_loss_metrics with extreme losses on both ends.
    for loss in [100.0, 0.001, 50.0, 0.01, 1e6, 1e-6]:
        opt.set_loss_metrics(loss, vocab_size=32000)
        s = opt.perplexity_scale
        assert 0.85 - 1e-6 <= s <= 1.15 + 1e-6, f"PGM scale {s} out of bounds"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
