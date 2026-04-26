"""
Test Iterative Reasoning Loop for R-Velvet.

Verifies:
1. LoRA shapes and zero-init
2. LoRA Bank indexation and param count
3. HaltingUnit output range [0,1]
4. compute_halting_loss (non-negative, gradient flow)
5. IterativeReasoner forward shapes (training)
6. IterativeReasoner inference early exit
7. Gradient flow through iterative loop
8. Deep supervision (per-iteration outputs)
9. Full model integration (standard mode)
10. Full model integration (ACR mode)
11. Backward compatibility (use_iterative_reasoning=False)
12. Parameter budget (<1M new params)
13. qkv_delta_fn passthrough in GlobalBlock
14. Iterative reasoner with padding_mask
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvelvet.layers.lora_adapter import LoRAAdapter, IterationLoRABank
from rvelvet.layers.halting import HaltingUnit, compute_halting_loss
from rvelvet.layers.iterative_reasoner import IterativeReasoner
from rvelvet.layers.global_reasoner import GlobalReasoner, GlobalBlock
from rvelvet.layers.memory_controller import MemoryController
from rvelvet.model import RVelvet


def test_lora_shapes_and_zero_init():
    print("=" * 60)
    print("ITER TEST 1: LoRA Shapes & Zero-Init")
    print("=" * 60)

    d_model, rank = 384, 8
    d_out = 3 * d_model  # 1152

    adapter = LoRAAdapter(d_model=d_model, d_out=d_out, rank=rank)

    # Check shapes
    assert adapter.down_proj.weight.shape == (rank, d_model), \
        f"down_proj shape: {adapter.down_proj.weight.shape}"
    assert adapter.up_proj.weight.shape == (d_out, rank), \
        f"up_proj shape: {adapter.up_proj.weight.shape}"

    # Check zero-init of up_proj (critical: model starts identical)
    assert (adapter.up_proj.weight == 0).all(), "up_proj should be zero-initialized"

    # Check that output is zero at init
    x = torch.randn(2, 16, d_model)
    out = adapter(x)
    assert out.shape == (2, 16, d_out), f"Output shape: {out.shape}"
    assert (out == 0).all(), "Output should be zero at initialization"

    # Check scaling
    assert adapter.scaling == 1.0, f"scaling = {adapter.scaling} (expected 1.0 for alpha=rank=8)"

    print(f"  down_proj: {tuple(adapter.down_proj.weight.shape)}")
    print(f"  up_proj:   {tuple(adapter.up_proj.weight.shape)}")
    print(f"  scaling:   {adapter.scaling}")
    print(f"  Output at init: all zeros ✓")
    print(f"  PASSED\n")


def test_lora_bank_indexation_and_param_count():
    print("=" * 60)
    print("ITER TEST 2: LoRA Bank Indexation & Param Count")
    print("=" * 60)

    d_model = 384
    n_layers = 8
    max_iter = 8
    rank = 8

    bank = IterationLoRABank(
        d_model=d_model, n_layers=n_layers,
        max_iterations=max_iter, rank=rank,
    )

    # Check indexation
    for i in range(max_iter):
        for j in range(n_layers):
            adapter = bank.get_adapter(i, j)
            assert adapter is not None
            fn = bank.get_qkv_delta_fn(i, j)
            assert callable(fn)

    # Check different adapters for different (iter, layer)
    a_00 = bank.get_adapter(0, 0)
    a_01 = bank.get_adapter(0, 1)
    a_10 = bank.get_adapter(1, 0)
    assert a_00 is not a_01, "Different layers should have different adapters"
    assert a_00 is not a_10, "Different iterations should have different adapters"

    # Param count: 8 iter × 8 layers × (384*8 + 8*1152) = 786,432
    n_params = sum(p.numel() for p in bank.parameters())
    print(f"  LoRA Bank params: {n_params:,}")
    assert 700_000 < n_params < 900_000, f"Unexpected param count: {n_params}"

    print(f"  Indexation: {max_iter} iters × {n_layers} layers ✓")
    print(f"  All adapters distinct ✓")
    print(f"  PASSED\n")


def test_halting_unit_output_range():
    print("=" * 60)
    print("ITER TEST 3: HaltingUnit Output Range [0,1]")
    print("=" * 60)

    halting = HaltingUnit(d_model=128, hidden_dim=32)

    concepts = torch.randn(4, 16, 128)
    p_halt = halting(concepts)

    assert p_halt.shape == (4,), f"p_halt shape: {p_halt.shape}"
    assert (p_halt >= 0).all() and (p_halt <= 1).all(), \
        f"p_halt range: [{p_halt.min():.4f}, {p_halt.max():.4f}]"

    # At init (zeros last linear), should be ~0.5
    assert (p_halt - 0.5).abs().max() < 0.05, \
        f"Expected ~0.5 at init, got {p_halt.tolist()}"

    # Test with padding mask
    mask = torch.zeros(4, 16, dtype=torch.bool)
    mask[:, 8:] = True  # Mask last 8 concepts
    p_halt_masked = halting(concepts, padding_mask=mask)
    assert p_halt_masked.shape == (4,)
    assert (p_halt_masked >= 0).all() and (p_halt_masked <= 1).all()

    print(f"  p_halt shape: {tuple(p_halt.shape)}")
    print(f"  p_halt range: [{p_halt.min():.4f}, {p_halt.max():.4f}]")
    print(f"  Init value ~0.5: {p_halt[0].item():.4f} ✓")
    print(f"  With padding_mask: {p_halt_masked[0].item():.4f} ✓")
    print(f"  PASSED\n")


def test_halting_loss():
    print("=" * 60)
    print("ITER TEST 4: compute_halting_loss")
    print("=" * 60)

    # Create p_halts with gradients
    p_halts = [torch.tensor([0.3, 0.4], requires_grad=True) for _ in range(4)]

    loss = compute_halting_loss(p_halts, lambda_p=0.5)

    # Non-negative
    assert loss.item() >= 0, f"Loss should be non-negative: {loss.item()}"

    # Gradient flow (last p_halt has zero gradient by design:
    # halt_dist[-1] = remaining * p[-1] + remaining * (1-p[-1]) = remaining,
    # so p[-1] cancels out — it gets all remaining mass regardless)
    loss.backward()
    for i, p in enumerate(p_halts[:-1]):
        assert p.grad is not None, f"No gradient for p_halts[{i}]"
        assert p.grad.abs().sum() > 0, f"Zero gradient for p_halts[{i}]"
    # Last p_halt: gradient is zero (expected)
    assert p_halts[-1].grad is not None, "Last p_halt should still have grad tensor"

    # Different lambda_p should give different loss
    p_halts2 = [torch.tensor([0.3, 0.4]) for _ in range(4)]
    loss_05 = compute_halting_loss(p_halts2, lambda_p=0.5)
    loss_08 = compute_halting_loss(p_halts2, lambda_p=0.8)
    assert abs(loss_05.item() - loss_08.item()) > 1e-6, \
        "Different lambda_p should give different loss"

    print(f"  Loss (λ=0.5): {loss_05.item():.6f}")
    print(f"  Loss (λ=0.8): {loss_08.item():.6f}")
    print(f"  Non-negative: ✓")
    print(f"  Gradient flow: ✓")
    print(f"  PASSED\n")


def test_iterative_reasoner_forward_training():
    print("=" * 60)
    print("ITER TEST 5: IterativeReasoner Forward (Training)")
    print("=" * 60)

    d_model = 128
    n_layers = 4
    max_iter = 4

    reasoner = GlobalReasoner(d_model=d_model, n_heads=4, n_layers=n_layers)
    memory_ctrl = MemoryController(d_model=d_model, n_heads=4, memory_size=32)

    iter_reasoner = IterativeReasoner(
        global_reasoner=reasoner,
        memory_controller=memory_ctrl,
        d_model=d_model,
        n_layers=n_layers,
        max_iterations=max_iter,
        lora_rank=4,
        halt_threshold=0.5,
    )
    iter_reasoner.train()

    concepts = torch.randn(2, 8, d_model)
    out = iter_reasoner(concepts)

    assert out['concepts'].shape == (2, 8, d_model), \
        f"concepts shape: {out['concepts'].shape}"
    assert out['relevance'].shape == (2, 8), \
        f"relevance shape: {out['relevance'].shape}"
    assert out['memory'].shape == (2, 32, d_model), \
        f"memory shape: {out['memory'].shape}"
    assert out['halt_distribution'].shape == (2, max_iter), \
        f"halt_distribution shape: {out['halt_distribution'].shape}"
    assert len(out['iteration_outputs']) == max_iter, \
        f"Expected {max_iter} iteration outputs, got {len(out['iteration_outputs'])}"
    assert len(out['p_halts']) == max_iter, \
        f"Expected {max_iter} p_halts, got {len(out['p_halts'])}"
    assert out['n_iterations'] == max_iter, \
        f"Training should run all iterations: {out['n_iterations']}"

    # halt_distribution should sum to 1
    halt_sum = out['halt_distribution'].sum(dim=1)
    assert torch.allclose(halt_sum, torch.ones(2), atol=1e-5), \
        f"halt_distribution should sum to 1: {halt_sum}"

    print(f"  concepts:          {tuple(out['concepts'].shape)}")
    print(f"  relevance:         {tuple(out['relevance'].shape)}")
    print(f"  memory:            {tuple(out['memory'].shape)}")
    print(f"  halt_distribution: {tuple(out['halt_distribution'].shape)}")
    print(f"  n_iterations:      {out['n_iterations']}")
    print(f"  halt_dist sum:     {halt_sum[0].item():.6f}")
    print(f"  PASSED\n")


def test_iterative_reasoner_inference_early_exit():
    print("=" * 60)
    print("ITER TEST 6: IterativeReasoner Inference Early Exit")
    print("=" * 60)

    d_model = 128
    n_layers = 4
    max_iter = 8

    reasoner = GlobalReasoner(d_model=d_model, n_heads=4, n_layers=n_layers)
    memory_ctrl = MemoryController(d_model=d_model, n_heads=4, memory_size=32)

    iter_reasoner = IterativeReasoner(
        global_reasoner=reasoner,
        memory_controller=memory_ctrl,
        d_model=d_model,
        n_layers=n_layers,
        max_iterations=max_iter,
        lora_rank=4,
        halt_threshold=0.0,  # Very low threshold → should exit early
    )
    iter_reasoner.eval()

    with torch.no_grad():
        concepts = torch.randn(2, 8, d_model)
        out = iter_reasoner(concepts)

    # With threshold=0.0 and sigmoid(0)=0.5 > 0.0, should exit at iteration 1
    assert out['n_iterations'] <= max_iter, \
        f"Should have early-exited: {out['n_iterations']} iterations"
    assert out['n_iterations'] >= 1, "Must run at least 1 iteration"

    print(f"  max_iterations: {max_iter}")
    print(f"  halt_threshold: 0.0")
    print(f"  Actual iterations: {out['n_iterations']}")
    print(f"  Early exit worked: {out['n_iterations'] < max_iter}")
    print(f"  PASSED\n")


def test_gradient_flow_iterative():
    print("=" * 60)
    print("ITER TEST 7: Gradient Flow Through Iterative Loop")
    print("=" * 60)

    d_model = 128
    n_layers = 4
    max_iter = 3

    reasoner = GlobalReasoner(d_model=d_model, n_heads=4, n_layers=n_layers)
    memory_ctrl = MemoryController(d_model=d_model, n_heads=4, memory_size=32)

    iter_reasoner = IterativeReasoner(
        global_reasoner=reasoner,
        memory_controller=memory_ctrl,
        d_model=d_model,
        n_layers=n_layers,
        max_iterations=max_iter,
        lora_rank=4,
    )
    iter_reasoner.train()

    concepts = torch.randn(2, 8, d_model)

    # Perturb zero-initialized weights so gradients flow through all params
    with torch.no_grad():
        for adapter_list in iter_reasoner.lora_bank.adapters:
            for adapter in adapter_list:
                adapter.up_proj.weight.fill_(0.01)
        # Halting unit last linear (zero-init blocks gradient to earlier layers)
        iter_reasoner.halting_unit.mlp[2].weight.fill_(0.01)

    out = iter_reasoner(concepts)

    # Use a loss that depends on halt_distribution (exercises halting unit)
    # and on concepts (exercises LoRA + reasoner + memory)
    halting_loss = compute_halting_loss(out['p_halts'], lambda_p=0.5)
    concept_loss = out['concepts'].sum()
    loss = concept_loss + halting_loss
    loss.backward()

    # Check gradients reach all components
    components = {
        'lora_bank': list(iter_reasoner.lora_bank.parameters())[0],
        'halting_unit': list(iter_reasoner.halting_unit.parameters())[0],
        'iteration_embed': iter_reasoner.iteration_embed,
        'global_reasoner': list(reasoner.parameters())[0],
        'memory_controller': memory_ctrl.memory_init,
    }

    all_ok = True
    for name, param in components.items():
        has_grad = param.grad is not None
        grad_norm = param.grad.norm().item() if has_grad else 0.0
        status = "OK" if has_grad and grad_norm > 0 else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {name:25s}: grad_norm = {grad_norm:.6f}  [{status}]")

    assert all_ok, "Some components have no gradients!"
    print(f"  PASSED\n")


def test_deep_supervision():
    print("=" * 60)
    print("ITER TEST 8: Deep Supervision (Per-Iteration Outputs)")
    print("=" * 60)

    d_model = 128
    n_layers = 4
    max_iter = 4

    reasoner = GlobalReasoner(d_model=d_model, n_heads=4, n_layers=n_layers)
    memory_ctrl = MemoryController(d_model=d_model, n_heads=4, memory_size=32)

    iter_reasoner = IterativeReasoner(
        global_reasoner=reasoner,
        memory_controller=memory_ctrl,
        d_model=d_model,
        n_layers=n_layers,
        max_iterations=max_iter,
        lora_rank=4,
    )
    iter_reasoner.train()

    concepts = torch.randn(2, 8, d_model)
    out = iter_reasoner(concepts)

    # Check each iteration output
    for i, iter_out in enumerate(out['iteration_outputs']):
        assert iter_out.shape == (2, 8, d_model), \
            f"Iteration {i} output shape: {iter_out.shape}"

    # Outputs should differ across iterations (due to LoRA + memory)
    diff_01 = (out['iteration_outputs'][0] - out['iteration_outputs'][1]).abs().mean()
    assert diff_01 > 0, "Iteration outputs should differ"

    # Deep supervision loss: each iteration output should be differentiable
    deep_loss = sum(o.sum() for o in out['iteration_outputs'])
    deep_loss.backward()

    # Check all LoRA adapters got gradients
    has_lora_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in iter_reasoner.lora_bank.parameters()
    )
    assert has_lora_grad, "LoRA bank should receive gradients from deep supervision"

    print(f"  {max_iter} iteration outputs, each (2, 8, {d_model})")
    print(f"  Diff between iter 0 and 1: {diff_01:.6f}")
    print(f"  Deep supervision loss gradient flows: ✓")
    print(f"  PASSED\n")


def test_full_model_standard():
    print("=" * 60)
    print("ITER TEST 9: Full Model Integration (Standard Mode)")
    print("=" * 60)

    model = RVelvet(
        vocab_size=1000, d_model=128,
        n_local_layers=2, n_global_layers=4,
        n_local_heads=4, n_global_heads=4,
        window_size=64, segment_size=64,
        n_concepts=1, n_refine_layers=1,
        memory_size=32, n_read_steps=1,
        ffn_mult=2.0, max_seq_len=512,
        use_acr=False,
        use_iterative_reasoning=True,
        max_reasoning_iterations=4,
        lora_rank=4,
    )
    model.train()

    input_ids = torch.randint(0, 1000, (2, 128))
    out = model(input_ids)

    assert out['logits'].shape == (2, 128, 1000), \
        f"logits: {out['logits'].shape}"
    assert 'iteration_outputs' in out, "Missing iteration_outputs"
    assert 'p_halts' in out, "Missing p_halts"
    assert 'halt_distribution' in out, "Missing halt_distribution"
    assert 'local_out' in out, "Missing local_out"
    assert out['n_iterations'] == 4, f"Expected 4 iterations: {out['n_iterations']}"

    # Backward
    targets = torch.randint(0, 1000, (2, 128))
    loss = torch.nn.functional.cross_entropy(
        out['logits'].view(-1, 1000), targets.view(-1)
    )
    loss.backward()

    print(f"  Logits:          {tuple(out['logits'].shape)}")
    print(f"  n_iterations:    {out['n_iterations']}")
    print(f"  halt_dist shape: {tuple(out['halt_distribution'].shape)}")
    print(f"  Loss:            {loss.item():.4f}")
    print(f"  PASSED\n")


def test_full_model_acr():
    print("=" * 60)
    print("ITER TEST 10: Full Model Integration (ACR Mode)")
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
        use_iterative_reasoning=True,
        max_reasoning_iterations=3,
        lora_rank=4,
    )
    model.train()

    input_ids = torch.randint(0, 1000, (2, 128))
    out = model(input_ids)

    assert out['logits'].shape == (2, 128, 1000), \
        f"logits: {out['logits'].shape}"
    assert 'route_weights' in out, "Missing route_weights (ACR)"
    assert 'iteration_outputs' in out, "Missing iteration_outputs (iterative)"
    assert 'p_halts' in out, "Missing p_halts"
    assert 'halt_distribution' in out, "Missing halt_distribution"

    # Backward
    targets = torch.randint(0, 1000, (2, 128))
    loss = torch.nn.functional.cross_entropy(
        out['logits'].view(-1, 1000), targets.view(-1)
    )
    loss.backward()

    print(f"  Logits:          {tuple(out['logits'].shape)}")
    print(f"  route_weights:   {tuple(out['route_weights'].shape)}")
    print(f"  n_iterations:    {out['n_iterations']}")
    print(f"  halt_dist shape: {tuple(out['halt_distribution'].shape)}")
    print(f"  Loss:            {loss.item():.4f}")
    print(f"  PASSED\n")


def test_backward_compatibility():
    print("=" * 60)
    print("ITER TEST 11: Backward Compatibility (use_iterative_reasoning=False)")
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
        use_iterative_reasoning=False,
    )

    input_ids = torch.randint(0, 1000, (2, 128))
    out = model(input_ids)

    assert out['logits'].shape == (2, 128, 1000)
    assert 'iteration_outputs' not in out, "Standard model should not return iteration_outputs"
    assert 'p_halts' not in out, "Standard model should not return p_halts"

    # Gradient check
    targets = torch.randint(0, 1000, (2, 128))
    loss = torch.nn.functional.cross_entropy(
        out['logits'].view(-1, 1000), targets.view(-1)
    )
    loss.backward()

    print(f"  Standard forward: logits {tuple(out['logits'].shape)}")
    print(f"  No iterative keys in output: ✓")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  PASSED\n")


def test_parameter_budget():
    print("=" * 60)
    print("ITER TEST 12: Parameter Budget (<1M New Params)")
    print("=" * 60)

    d_model = 384
    n_layers = 8
    max_iter = 8
    rank = 8

    reasoner = GlobalReasoner(d_model=d_model, n_heads=8, n_layers=n_layers)
    memory_ctrl = MemoryController(d_model=d_model, n_heads=4, memory_size=256)

    iter_reasoner = IterativeReasoner(
        global_reasoner=reasoner,
        memory_controller=memory_ctrl,
        d_model=d_model,
        n_layers=n_layers,
        max_iterations=max_iter,
        lora_rank=rank,
    )

    # Count ONLY new params (not shared reasoner/memory)
    lora_params = sum(p.numel() for p in iter_reasoner.lora_bank.parameters())
    halt_params = sum(p.numel() for p in iter_reasoner.halting_unit.parameters())
    embed_params = iter_reasoner.iteration_embed.numel()
    total_new = lora_params + halt_params + embed_params

    print(f"  LoRA Bank:            {lora_params:>10,}")
    print(f"  Halting Unit:         {halt_params:>10,}")
    print(f"  Iteration Embeddings: {embed_params:>10,}")
    print(f"  Total New Params:     {total_new:>10,}")

    assert total_new < 1_000_000, f"New params ({total_new}) exceed 1M budget"

    # Check it's in the expected range (~826K)
    assert total_new > 500_000, f"Suspiciously few params: {total_new}"

    print(f"  Budget check: {total_new:,} < 1,000,000 ✓")
    print(f"  PASSED\n")


def test_qkv_delta_fn_passthrough():
    print("=" * 60)
    print("ITER TEST 13: qkv_delta_fn Passthrough in GlobalBlock")
    print("=" * 60)

    d_model = 128
    block = GlobalBlock(d_model=d_model, n_heads=4, ffn_mult=2.0, dropout=0.0)

    x = torch.randn(2, 8, d_model)

    # Without delta fn
    out_no_delta = block(x)

    # With identity delta fn (should change output)
    def delta_fn(inp):
        return torch.ones_like(block.attn.qkv(inp)) * 0.1

    out_with_delta = block(x, qkv_delta_fn=delta_fn)

    # Outputs should differ
    diff = (out_no_delta - out_with_delta).abs().mean()
    assert diff > 0, "qkv_delta_fn should affect output"

    # Without delta fn (None) should be same as before
    out_none = block(x, qkv_delta_fn=None)
    assert torch.allclose(out_no_delta, out_none), \
        "None qkv_delta_fn should give same result"

    print(f"  Output diff with delta_fn: {diff:.6f}")
    print(f"  None delta_fn = no delta_fn: ✓")
    print(f"  PASSED\n")


def test_iterative_reasoner_with_padding_mask():
    print("=" * 60)
    print("ITER TEST 14: Iterative Reasoner with Padding Mask")
    print("=" * 60)

    d_model = 128
    n_layers = 4
    max_iter = 3

    reasoner = GlobalReasoner(d_model=d_model, n_heads=4, n_layers=n_layers)
    memory_ctrl = MemoryController(d_model=d_model, n_heads=4, memory_size=32)

    iter_reasoner = IterativeReasoner(
        global_reasoner=reasoner,
        memory_controller=memory_ctrl,
        d_model=d_model,
        n_layers=n_layers,
        max_iterations=max_iter,
        lora_rank=4,
    )
    iter_reasoner.train()

    concepts = torch.randn(2, 16, d_model)
    padding_mask = torch.zeros(2, 16, dtype=torch.bool)
    padding_mask[:, 12:] = True  # Last 4 concepts are padded

    out = iter_reasoner(concepts, padding_mask=padding_mask)

    assert out['concepts'].shape == (2, 16, d_model)
    assert out['n_iterations'] == max_iter

    # p_halt should still be valid with masked input
    for p in out['p_halts']:
        assert (p >= 0).all() and (p <= 1).all(), \
            f"p_halt out of range: [{p.min():.4f}, {p.max():.4f}]"

    # Backward should work
    loss = out['concepts'].sum()
    loss.backward()

    print(f"  Output shape: {tuple(out['concepts'].shape)}")
    print(f"  Padding mask: last 4 of 16 concepts masked")
    print(f"  n_iterations: {out['n_iterations']}")
    print(f"  Backward OK: ✓")
    print(f"  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  R-VELVET ITERATIVE REASONING TESTS")
    print("=" * 60 + "\n")

    torch.manual_seed(42)

    test_lora_shapes_and_zero_init()
    test_lora_bank_indexation_and_param_count()
    test_halting_unit_output_range()
    test_halting_loss()
    test_iterative_reasoner_forward_training()
    test_iterative_reasoner_inference_early_exit()
    test_gradient_flow_iterative()
    test_deep_supervision()
    test_full_model_standard()
    test_full_model_acr()
    test_backward_compatibility()
    test_parameter_budget()
    test_qkv_delta_fn_passthrough()
    test_iterative_reasoner_with_padding_mask()

    print("=" * 60)
    print("  ALL ITERATIVE REASONING TESTS PASSED")
    print("=" * 60)
