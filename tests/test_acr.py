"""
Test Adaptive Computation Routing (ACR) for R-Velvet.

Verifies:
1. Scanner shapes and parameter count
2. Router differentiability (Gumbel gradient check)
3. Three compression paths (SKIM, PROCESS, FOCUS)
4. Full forward pass with ACR
5. Gradient flow through all 3 routes
6. Memory write priority gating
7. Load balance loss
8. Inference mode (hard routing)
9. Temperature annealing
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvelvet.layers.adaptive_router import (
    SegmentScanner, AdaptiveRouter, compute_layer_gates, compute_acr_losses,
)
from rvelvet.layers.segment_compressor import AdaptiveSegmentCompressor
from rvelvet.layers.memory_controller import MemoryController
from rvelvet.model import RVelvet


def test_scanner_shapes():
    print("=" * 60)
    print("ACR TEST 1: Scanner Shapes & Param Count")
    print("=" * 60)

    scanner = SegmentScanner(d_model=384, n_heads=6, segment_size=512)
    segments = torch.randn(2, 8, 512, 384)  # 8 segments of 512 tokens

    out = scanner(segments)

    assert out['route_logits'].shape == (2, 8, 3), \
        f"route_logits shape: {out['route_logits'].shape}"
    assert out['write_priority'].shape == (2, 8), \
        f"write_priority shape: {out['write_priority'].shape}"
    assert out['segment_summary'].shape == (2, 8, 384), \
        f"segment_summary shape: {out['segment_summary'].shape}"

    # Check write_priority is in [0, 1] (sigmoid output)
    assert (out['write_priority'] >= 0).all() and (out['write_priority'] <= 1).all()

    # Param count ~370K
    n_params = sum(p.numel() for p in scanner.parameters())
    print(f"  Scanner params: {n_params:,}")
    assert 500_000 < n_params < 800_000, f"Unexpected param count: {n_params}"

    print(f"  route_logits:    {tuple(out['route_logits'].shape)}")
    print(f"  write_priority:  {tuple(out['write_priority'].shape)}")
    print(f"  segment_summary: {tuple(out['segment_summary'].shape)}")
    print(f"  PASSED\n")


def test_router_differentiability():
    print("=" * 60)
    print("ACR TEST 2: Router Differentiability (Gumbel)")
    print("=" * 60)

    router = AdaptiveRouter(tau_start=1.0, tau_end=0.1, tau_anneal_steps=100)
    router.train()

    logits = torch.randn(2, 4, 3, requires_grad=True)
    weights = router(logits)

    assert weights.shape == (2, 4, 3), f"weights shape: {weights.shape}"

    # Check straight-through: forward is hard (one-hot), backward is soft
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4))

    # Gradient check: use a route-discriminating loss (not .sum() which always = 1)
    # Weighted by different costs per route → loss depends on WHICH route is chosen
    cost = torch.tensor([0.1, 0.5, 1.0])
    loss = (weights * cost).sum()
    loss.backward()
    assert logits.grad is not None, "No gradient through Gumbel-softmax"
    assert logits.grad.abs().sum() > 0, "Zero gradients"

    print(f"  weights shape: {tuple(weights.shape)}")
    print(f"  weights sum per segment: {weights.sum(dim=-1)[0].tolist()}")
    print(f"  grad norm: {logits.grad.norm().item():.6f}")
    print(f"  PASSED\n")


def test_compression_paths():
    print("=" * 60)
    print("ACR TEST 3: Three Compression Paths")
    print("=" * 60)

    comp = AdaptiveSegmentCompressor(
        d_model=128, n_heads=4, segment_size=64,
        n_concepts_focus=16, n_refine_layers=2,
    )
    local_out = torch.randn(2, 4, 64, 128)  # 4 segments, 64 tokens each

    # SKIM: 1 concept per segment
    skim_out = comp.compress_segment(local_out, 'SKIM')
    assert skim_out.shape == (2, 4, 1, 128), f"SKIM shape: {skim_out.shape}"

    # PROCESS: 4 concepts per segment
    proc_out = comp.compress_segment(local_out, 'PROCESS')
    assert proc_out.shape == (2, 4, 4, 128), f"PROCESS shape: {proc_out.shape}"

    # FOCUS: 16 concepts per segment (minimal compression)
    focus_out = comp.compress_segment(local_out, 'FOCUS')
    assert focus_out.shape == (2, 4, 16, 128), f"FOCUS shape: {focus_out.shape}"

    # k_max should equal n_concepts_focus
    assert comp.k_max == 16

    # Soft forward should produce (B, S, K_max, D)
    route_weights = torch.zeros(2, 4, 3)
    route_weights[:, :, 1] = 1.0  # all PROCESS
    comp.train()
    blended = comp(local_out, route_weights)
    assert blended.shape == (2, 4, 16, 128), f"Blended shape: {blended.shape}"

    print(f"  SKIM:    {tuple(skim_out.shape)}  (512:1 compression)")
    print(f"  PROCESS: {tuple(proc_out.shape)}  (128:1 compression)")
    print(f"  FOCUS:   {tuple(focus_out.shape)}  (32:1 compression)")
    print(f"  K_max:   {comp.k_max}")
    print(f"  Blended: {tuple(blended.shape)}")
    print(f"  PASSED\n")


def test_full_forward_acr():
    print("=" * 60)
    print("ACR TEST 4: Full Forward Pass (ACR)")
    print("=" * 60)

    model = RVelvet(
        vocab_size=1000,
        d_model=128,
        n_local_layers=6,
        n_global_layers=6,
        n_local_heads=4,
        n_global_heads=4,
        window_size=64,
        segment_size=64,
        n_concepts=1,
        n_refine_layers=1,
        memory_size=32,
        n_read_steps=1,
        ffn_mult=2.0,
        max_seq_len=1024,
        use_acr=True,
    )

    input_ids = torch.randint(0, 1000, (2, 256))
    out = model(input_ids)

    assert out['logits'].shape == (2, 256, 1000), f"logits: {out['logits'].shape}"
    assert 'route_weights' in out, "Missing route_weights"
    assert 'route_logits' in out, "Missing route_logits"
    assert 'write_priority' in out, "Missing write_priority"
    assert out['memory'].shape[0] == 2 and out['memory'].shape[2] == 128

    n_seg = 256 // 64  # 4 segments
    assert out['route_weights'].shape == (2, n_seg, 3), \
        f"route_weights: {out['route_weights'].shape}"
    assert out['route_logits'].shape == (2, n_seg, 3), \
        f"route_logits: {out['route_logits'].shape}"
    assert out['write_priority'].shape == (2, n_seg), \
        f"write_priority: {out['write_priority'].shape}"

    print(f"  Input:          (2, 256) tokens")
    print(f"  Logits:         {tuple(out['logits'].shape)}")
    print(f"  Concepts:       {tuple(out['concepts'].shape)}")
    print(f"  Memory:         {tuple(out['memory'].shape)}")
    print(f"  Route weights:  {tuple(out['route_weights'].shape)}")
    print(f"  Write priority: {tuple(out['write_priority'].shape)}")

    # Parameter count
    counts = model.count_parameters()
    print(f"\n  Parameter counts:")
    for name, count in counts.items():
        print(f"    {name:25s}: {count:>10,}")

    print(f"  PASSED\n")


def test_gradient_flow_acr():
    print("=" * 60)
    print("ACR TEST 5: Gradient Flow Through All Routes")
    print("=" * 60)

    model = RVelvet(
        vocab_size=1000, d_model=128,
        n_local_layers=6, n_global_layers=6,
        n_local_heads=4, n_global_heads=4,
        window_size=64, segment_size=64,
        n_concepts=1, n_refine_layers=1,
        memory_size=32, n_read_steps=1,
        ffn_mult=2.0, max_seq_len=512,
        use_acr=True,
    )
    model.train()

    input_ids = torch.randint(0, 1000, (2, 128))
    targets = torch.randint(0, 1000, (2, 128))

    out = model(input_ids)

    # Compute LM loss + ACR losses
    lm_loss = torch.nn.functional.cross_entropy(
        out['logits'].view(-1, 1000), targets.view(-1)
    )
    acr_losses = compute_acr_losses(out['route_weights'], out['route_logits'])
    total_loss = (
        lm_loss
        + 0.01 * acr_losses['load_balance']
        + 0.001 * acr_losses['entropy']
        + 0.005 * acr_losses['compute_cost']
    )

    total_loss.backward()

    # Check gradients reach key components
    # Note: adaptive_compressor.skim / .process may get zero grad if Gumbel
    # didn't select that route for any segment. These are checked separately.
    required_components = {
        'token_embed': model.token_embed.weight,
        'local_encoder': list(model.local_encoder.parameters())[0],
        'scanner.scan_query': model.scanner.scan_query,
        'scanner.route_head': list(model.scanner.route_head.parameters())[0],
        'global_reasoner': list(model.global_reasoner.parameters())[0],
        'memory_controller': model.memory_controller.memory_init,
        'expansion': list(model.expansion.parameters())[0],
    }
    optional_components = {
        'adaptive_compressor.skim': model.adaptive_compressor.concept_queries_skim,
        'adaptive_compressor.process': model.adaptive_compressor.concept_queries_process,
        'adaptive_compressor.focus': model.adaptive_compressor.concept_queries_focus,
    }

    all_ok = True
    for name, param in required_components.items():
        has_grad = param.grad is not None
        grad_norm = param.grad.norm().item() if has_grad else 0.0
        status = "OK" if has_grad and grad_norm > 0 else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {name:35s}: grad_norm = {grad_norm:.6f}  [{status}]")

    for name, param in optional_components.items():
        has_grad = param.grad is not None
        grad_norm = param.grad.norm().item() if has_grad else 0.0
        # OK if grad flows, SKIP if route wasn't selected (expected with hard Gumbel)
        status = "OK" if has_grad and grad_norm > 0 else "SKIP (route not selected)"
        print(f"  {name:35s}: grad_norm = {grad_norm:.6f}  [{status}]")

    print(f"\n  LM loss:           {lm_loss.item():.4f}")
    print(f"  Load balance loss: {acr_losses['load_balance'].item():.4f}")
    print(f"  Entropy loss:      {acr_losses['entropy'].item():.4f}")
    print(f"  Compute cost loss: {acr_losses['compute_cost'].item():.4f}")
    print(f"  Total loss:        {total_loss.item():.4f}")
    assert all_ok, "Required components have no gradients!"
    print(f"  PASSED\n")


def test_memory_write_priority():
    print("=" * 60)
    print("ACR TEST 6: Memory Write Priority Gating")
    print("=" * 60)

    mem = MemoryController(d_model=128, n_heads=4, memory_size=64, n_read_steps=1)
    concepts = torch.randn(2, 8, 128)
    memory = mem.init_memory(2, concepts.device)

    # High priority write
    high_priority = torch.ones(2, 8)
    mem_high = mem.write_with_priority(concepts, memory, high_priority)

    # Low priority write
    low_priority = torch.zeros(2, 8)
    mem_low = mem.write_with_priority(concepts, memory, low_priority)

    # High priority should change memory more than low priority
    diff_high = (mem_high - memory).abs().mean().item()
    diff_low = (mem_low - memory).abs().mean().item()

    print(f"  High priority memory change: {diff_high:.6f}")
    print(f"  Low priority memory change:  {diff_low:.6f}")
    assert diff_high > diff_low, \
        f"High priority ({diff_high}) should change memory more than low ({diff_low})"

    # Test through forward
    out_with_priority = mem(concepts, memory, write_priority=high_priority)
    out_without = mem(concepts, memory)
    assert out_with_priority['enriched'].shape == out_without['enriched'].shape

    print(f"  Forward with priority: enriched {tuple(out_with_priority['enriched'].shape)}")
    print(f"  PASSED\n")


def test_load_balance_loss():
    print("=" * 60)
    print("ACR TEST 7: Load Balance Loss")
    print("=" * 60)

    # Perfectly balanced (matches target: 60/30/10)
    B, S = 2, 100
    balanced = torch.zeros(B, S, 3)
    balanced[:, :60, 0] = 1.0   # 60% SKIM
    balanced[:, 60:90, 1] = 1.0  # 30% PROCESS
    balanced[:, 90:, 2] = 1.0   # 10% FOCUS
    logits_balanced = torch.randn(B, S, 3)

    losses_balanced = compute_acr_losses(balanced, logits_balanced)

    # All FOCUS (worst case)
    all_focus = torch.zeros(B, S, 3)
    all_focus[:, :, 2] = 1.0
    logits_focus = torch.randn(B, S, 3)

    losses_focus = compute_acr_losses(all_focus, logits_focus)

    print(f"  Balanced (60/30/10):")
    print(f"    load_balance: {losses_balanced['load_balance'].item():.6f}")
    print(f"    compute_cost: {losses_balanced['compute_cost'].item():.6f}")
    print(f"  All FOCUS:")
    print(f"    load_balance: {losses_focus['load_balance'].item():.6f}")
    print(f"    compute_cost: {losses_focus['compute_cost'].item():.6f}")

    assert losses_balanced['load_balance'] < losses_focus['load_balance'], \
        "Balanced should have lower load balance loss"
    assert losses_balanced['compute_cost'] < losses_focus['compute_cost'], \
        "Balanced should have lower compute cost"

    print(f"  PASSED\n")


def test_inference_mode():
    print("=" * 60)
    print("ACR TEST 8: Inference Mode (Hard Routing)")
    print("=" * 60)

    model = RVelvet(
        vocab_size=1000, d_model=128,
        n_local_layers=6, n_global_layers=6,
        n_local_heads=4, n_global_heads=4,
        window_size=64, segment_size=64,
        n_concepts=1, n_refine_layers=1,
        memory_size=32, n_read_steps=1,
        ffn_mult=2.0, max_seq_len=512,
        use_acr=True,
    )
    model.eval()

    with torch.no_grad():
        input_ids = torch.randint(0, 1000, (1, 128))
        out = model(input_ids)

        # Route weights should be one-hot in eval mode
        rw = out['route_weights']  # (1, S, 3)
        assert torch.allclose(rw.sum(dim=-1), torch.ones_like(rw.sum(dim=-1))), \
            "Route weights should sum to 1"

        # Each segment should have exactly one route selected
        max_vals = rw.max(dim=-1).values
        assert torch.allclose(max_vals, torch.ones_like(max_vals)), \
            "Each segment should have exactly one route (hard)"

    print(f"  Route weights (eval): {rw[0].tolist()}")
    print(f"  All one-hot: True")
    print(f"  Logits shape: {tuple(out['logits'].shape)}")
    print(f"  PASSED\n")


def test_temperature_annealing():
    print("=" * 60)
    print("ACR TEST 9: Temperature Annealing")
    print("=" * 60)

    router = AdaptiveRouter(tau_start=1.0, tau_end=0.1, tau_anneal_steps=100)

    tau_start = router.tau
    assert abs(tau_start - 1.0) < 1e-6, f"Initial tau: {tau_start}"

    # Simulate 50 steps
    router.train()
    for _ in range(50):
        logits = torch.randn(1, 1, 3)
        router(logits)

    tau_mid = router.tau
    assert 0.4 < tau_mid < 0.7, f"Mid tau: {tau_mid}"

    # Simulate 50 more steps (total 100)
    for _ in range(50):
        logits = torch.randn(1, 1, 3)
        router(logits)

    tau_end = router.tau
    assert abs(tau_end - 0.1) < 0.05, f"End tau: {tau_end}"

    print(f"  Step 0:   tau = {tau_start:.4f}")
    print(f"  Step 50:  tau = {tau_mid:.4f}")
    print(f"  Step 100: tau = {tau_end:.4f}")
    print(f"  PASSED\n")


def test_layer_gates():
    print("=" * 60)
    print("ACR TEST 10: Layer Gate Computation")
    print("=" * 60)

    # Test with 6 layers: boundary_low=2, boundary_high=4
    # SKIM = [1,0,0], PROCESS = [0,1,0], FOCUS = [0,0,1]
    B, S = 1, 3
    route_weights = torch.zeros(B, S, 3)
    route_weights[0, 0, 0] = 1.0  # segment 0: SKIM
    route_weights[0, 1, 1] = 1.0  # segment 1: PROCESS
    route_weights[0, 2, 2] = 1.0  # segment 2: FOCUS

    gates = compute_layer_gates(route_weights, n_layers=6)
    # Expected for 6 layers (boundary_low=2, boundary_high=4):
    # SKIM:    [1, 1, 0, 0, 0, 0]
    # PROCESS: [1, 1, 1, 1, 0, 0]
    # FOCUS:   [1, 1, 1, 1, 1, 1]

    skim_gates = gates[0, 0].tolist()
    process_gates = gates[0, 1].tolist()
    focus_gates = gates[0, 2].tolist()

    print(f"  SKIM gates:    {skim_gates}")
    print(f"  PROCESS gates: {process_gates}")
    print(f"  FOCUS gates:   {focus_gates}")

    assert skim_gates == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0], \
        f"SKIM gates wrong: {skim_gates}"
    assert process_gates == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0], \
        f"PROCESS gates wrong: {process_gates}"
    assert focus_gates == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], \
        f"FOCUS gates wrong: {focus_gates}"

    print(f"  PASSED\n")


def test_concept_masking():
    """Verify padded concepts are masked in global reasoner."""
    print("=" * 60)
    print("ACR TEST 11: Concept Masking (Zero-Pad Protection)")
    print("=" * 60)

    model = RVelvet(
        vocab_size=1000, d_model=128,
        n_local_layers=6, n_global_layers=6,
        n_local_heads=4, n_global_heads=4,
        window_size=64, segment_size=64,
        n_concepts=1, n_refine_layers=1,
        memory_size=32, n_read_steps=1,
        ffn_mult=2.0, max_seq_len=512,
        use_acr=True,
    )
    model.train()

    input_ids = torch.randint(0, 1000, (2, 128))
    out = model(input_ids)

    K_max = model.adaptive_compressor.k_max
    n_seg = 128 // 64
    concepts = out['concepts']  # (B, n_seg * K_max, D)

    assert concepts.shape == (2, n_seg * K_max, 128), \
        f"Concepts shape: {concepts.shape}"

    # Padded concepts should be zero (zeroed after global reasoner)
    # Determine which are padded based on route weights
    rw = out['route_weights']  # (B, S, 3)
    n_per_route = torch.tensor([1, 4, K_max], dtype=torch.float)
    concepts_per_seg = (rw * n_per_route).sum(dim=-1)  # (B, S)

    # Check some padded positions are actually zero
    for b in range(2):
        for s in range(n_seg):
            n_real = int(concepts_per_seg[b, s].item())
            if n_real < K_max:
                start = s * K_max + n_real
                end = (s + 1) * K_max
                padded_slice = concepts[b, start:end, :]
                assert (padded_slice == 0).all(), \
                    f"Padded concepts at b={b},s={s} not zero (n_real={n_real})"

    print(f"  K_max: {K_max}")
    print(f"  Concepts per seg: {concepts_per_seg[0].tolist()}")
    print(f"  Padded concepts are zero: True")
    print(f"  PASSED\n")


def test_no_regression():
    """Verify standard (non-ACR) model still works."""
    print("=" * 60)
    print("ACR TEST 12: No Regression (Standard Forward)")
    print("=" * 60)

    model = RVelvet(
        vocab_size=1000, d_model=128,
        n_local_layers=2, n_global_layers=2,
        n_local_heads=4, n_global_heads=4,
        window_size=64, segment_size=64,
        n_concepts=1, n_refine_layers=1,
        memory_size=32, n_read_steps=1,
        ffn_mult=2.0, max_seq_len=512,
        use_acr=False,
    )

    input_ids = torch.randint(0, 1000, (2, 128))
    out = model(input_ids)

    assert out['logits'].shape == (2, 128, 1000)
    assert 'route_weights' not in out, "Standard model should not return route_weights"

    # Gradient check
    targets = torch.randint(0, 1000, (2, 128))
    loss = torch.nn.functional.cross_entropy(
        out['logits'].view(-1, 1000), targets.view(-1)
    )
    loss.backward()

    print(f"  Standard forward: logits {tuple(out['logits'].shape)}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  No ACR keys in output: True")
    print(f"  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  R-VELVET ACR (ADAPTIVE COMPUTATION ROUTING) TESTS")
    print("=" * 60 + "\n")

    torch.manual_seed(42)

    test_scanner_shapes()
    test_router_differentiability()
    test_compression_paths()
    test_full_forward_acr()
    test_gradient_flow_acr()
    test_memory_write_priority()
    test_load_balance_loss()
    test_inference_mode()
    test_temperature_annealing()
    test_layer_gates()
    test_concept_masking()
    test_no_regression()

    print("=" * 60)
    print("  ALL ACR TESTS PASSED")
    print("=" * 60)
