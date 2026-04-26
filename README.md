# R-Velvet

Multi-scale transformer for unlimited context. Processes 1M+ tokens by compressing intelligently, reasoning globally, and remembering selectively.

## Architecture Overview

```
tokens (1M+)
    │
    ▼
┌──────────────────┐
│  Token Embedding  │  vocab → d_model
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Local Encoder    │  Windowed attention O(n·w²)
│  (6 layers)       │  Captures syntax, local semantics
└────────┬─────────┘
         │ ← Optional: ACR residual gating (skip layers for SKIM segments)
         ▼
┌──────────────────┐
│ Segment Compressor│  1M tokens → ~500 concepts
│ (cross-attention) │  Learned compression, not pooling
└────────┬─────────┘
         │ ← Optional: Adaptive compression (1/4/16 concepts per segment)
         ▼
┌──────────────────┐
│  Global Reasoner  │  Full attention O(N²) where N ≈ 500
│  (8 layers)       │  "Paragraph 3 contradicts paragraph 47"
└────────┬─────────┘
         │ ← Optional: Iterative reasoning loop (LoRA + halting)
         ▼
┌──────────────────┐
│ Memory Controller │  Semantic read/write to external memory
│ (multi-hop)       │  Persists across chunks
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    Expansion      │  Concepts → token-level (cross-attention)
│    + LM Head      │  → next-token logits
└──────────────────┘
```

**Complexity story**: 1M tokens with window=512, 1 concept/segment:
- Local attention: `1M × 512² = 262B ops` (parallelizable per window)
- Global attention: `2000² = 4M ops` (trivial)
- Full quadratic would be: `1T ops` — R-Velvet is **~3800x cheaper**

## Components

| Module | Path | Description |
|---|---|---|
| `RVelvet` | `rvelvet/model.py` | Top-level model, orchestrates all stages |
| `LocalEncoder` | `rvelvet/layers/local_attention.py` | Windowed self-attention stack |
| `SegmentCompressor` | `rvelvet/layers/segment_compressor.py` | Cross-attention compression (tokens → concepts) |
| `GlobalReasoner` | `rvelvet/layers/global_reasoner.py` | Full attention over concepts |
| `MemoryController` | `rvelvet/layers/memory_controller.py` | Semantic read/write external memory |
| `ExpansionLayer` | `rvelvet/model.py` | Cross-attention expansion (concepts → tokens) |
| `SegmentScanner` | `rvelvet/layers/adaptive_router.py` | ACR: lightweight segment scanner |
| `AdaptiveRouter` | `rvelvet/layers/adaptive_router.py` | ACR: Gumbel-softmax routing |
| `AdaptiveSegmentCompressor` | `rvelvet/layers/segment_compressor.py` | ACR: variable-rate compression |
| `IterativeReasoner` | `rvelvet/layers/iterative_reasoner.py` | Multi-pass reasoning with LoRA + halting |
| `IterationLoRABank` | `rvelvet/layers/lora_adapter.py` | Per-iteration LoRA adapters |
| `HaltingUnit` | `rvelvet/layers/halting.py` | PonderNet-style halting prediction |

## Optional Modes

### ACR (Adaptive Computation Routing) — `use_acr=True`

Routes each segment to one of three computation paths:

| Route | Local Layers | Global Layers | Concepts/Segment | Cost |
|---|---|---|---|---|
| **SKIM** | 2 | 2 | 1 | 10% |
| **PROCESS** | 4 | 6 | 4 | 50% |
| **FOCUS** | 6 | 8 | 16 | 100% |

Target distribution: 60% SKIM, 30% PROCESS, 10% FOCUS. Training uses Gumbel-softmax with temperature annealing; inference uses hard argmax. Overhead: ~0.7% params (scanner).

### Iterative Reasoning — `use_iterative_reasoning=True`

Multi-pass reasoning through the global reasoner with:
- **Shared weights**: no duplication of the global reasoner
- **Per-iteration LoRA**: different behavior each pass (~786K params)
- **PonderNet halting**: learned early exit at inference
- **COCONUT**: output of iteration i → input of iteration i+1

Overhead: ~826K params (1.6% of a 50M model).

## Model Sizes

| Config | d_model | Local | Global | Heads | ~Params (base) | ~Params (+ACR+Iter) |
|---|---|---|---|---|---|---|
| `small` | 256 | 4 layers | 4 layers | 4/4 | ~5M | ~6M |
| `base` | 384 | 6 layers | 8 layers | 6/8 | ~50M | ~52M |

## Installation

```bash
cd R-Velvet
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

## Quick Start

### 1. Tokenize data

```bash
python scripts/tokenize_data.py \
    --input corpus.txt \
    --output data/train.bin \
    --tokenizer gpt2
```

### 2. Phase 1 — Base pretraining

```bash
python scripts/train.py training=phase1_pretrain model=base data=text
```

### 3. Phase 2 — ACR finetuning

```bash
python scripts/train.py \
    training=phase2_acr model=base data=text \
    training.resume_from=outputs/phase1_pretrain/ckpt_final.pt
```

### 4. Phase 3 — Iterative reasoning

```bash
python scripts/train.py \
    training=phase3_iterative model=base data=text \
    training.resume_from=outputs/phase2_acr/ckpt_final.pt
```

### Debug mode (any phase)

```bash
python scripts/train.py training=phase1_pretrain model=small training.debug=true
```

Runs 10 steps on synthetic data. No checkpoint, no wandb.

## Configuration

Hydra configs in `configs/`:

```
configs/
├── config.yaml                  # Entry point (defaults)
├── model/
│   ├── small.yaml               # ~5M params (debug)
│   └── base.yaml                # ~50M params
├── training/
│   ├── phase1_pretrain.yaml     # LR=3e-4, 100K steps
│   ├── phase2_acr.yaml          # LR=1e-4, 50K steps
│   └── phase3_iterative.yaml   # LR=5e-4, 30K steps
└── data/
    └── text.yaml                # Dataset paths, seq_len
```

Override any value from CLI:

```bash
# Change learning rate and batch size
python scripts/train.py training=phase1_pretrain training.lr=1e-4 training.batch_size=16

# Use small model
python scripts/train.py training=phase1_pretrain model=small

# Enable wandb
python scripts/train.py training=phase1_pretrain training.wandb=true
```

## Training Phases

### Phase 1 — Base Pretraining

Standard language model pretraining. All parameters trained with cross-entropy loss.

| Param | Value |
|---|---|
| Mode | `use_acr=false`, `use_iterative_reasoning=false` |
| LR | 3e-4 (cosine decay, 2K warmup) |
| Effective batch | 128 (32 x 4 accum) |
| Steps | 100K |
| Loss | Cross-entropy |

### Phase 2 — ACR Finetuning

Enable adaptive computation routing. Loads Phase 1 checkpoint with `strict=False` (new scanner/router/compressor modules).

| Param | Value |
|---|---|
| Mode | `use_acr=true`, `use_iterative_reasoning=false` |
| LR | 1e-4 (base params at x0.1, ACR params at full) |
| Effective batch | 128 (16 x 8 accum) |
| Steps | 50K |
| Loss | CE + 0.01\*load_balance + 0.001\*entropy + 0.005\*compute_cost |

### Phase 3 — Iterative Reasoning

Enable multi-pass reasoning. Freezes all base parameters, only trains LoRA bank + halting unit + iteration embeddings.

| Param | Value |
|---|---|
| Mode | `use_acr=true`, `use_iterative_reasoning=true` |
| LR | 5e-4 (only unfrozen params) |
| Effective batch | 128 (16 x 8 accum) |
| Steps | 30K |
| Loss | CE + 0.1\*halting + 0.1\*deep_supervision |

Deep supervision: each iteration's concepts are expanded → lm_head → CE, averaged across iterations.

## VelvetOptimizer

Custom optimizer built on AdamW with two adaptive mechanisms: **PGM** (Perplexity-Guided Momentum) and **LVS** (Loss-Velocity Scaling). Converges faster than AdamW and maintains advantage through late training.

`rvelvet/training/velvet_optimizer.py`

### PGM — Perplexity-Guided Momentum

Adapts beta1 dynamically based on normalized loss ratio (loss / max_entropy). Uses a U-curve mapping:

| Training Zone | Loss Ratio | beta1 Effect |
|---|---|---|
| Random (high loss) | > 0.6 | beta1 x 1.0 (normal momentum) |
| Active learning | 0.2 — 0.6 | beta1 x 0.8-1.0 (less momentum, faster reaction) |
| Converging (low loss) | < 0.2 | beta1 x 0.8-1.1 (more momentum, stability) |

Clamped to [0.7, 1.3] range. Zero overhead — pure scalar math.

### LVS v5.2 — Loss-Velocity Scaling

Scales the learning rate based on whether the model is actively improving. Uses dual-EMA crossover in **log-space** with adaptive windows and asymmetric momentum.

**Core mechanism:**
1. Two EMAs track `log(loss)` — a fast "current" EMA and a slow "anchor" EMA
2. If current < anchor (loss improving) → **full LR boost** (max_boost)
3. If current > anchor (loss worsening) → proportional LR dampening
4. If current ≈ anchor (plateau) → neutral (1.0)

**Why log-space?** In linear space, as absolute loss decreases, percentage changes naturally shrink even if relative improvement rate is constant. This caused the signal to decay after ~300 steps. In log-space, exponential decay becomes linear → constant EMA gap → stable signal.

**Adaptive EMA windows (Chinchilla-inspired):** Windows scale with total training steps instead of being hardcoded:
- Current EMA: 3% of max_steps (clamped 50-300)
- Anchor EMA: starts at 10%, grows to 30% over training (clamped 100-500)

| Run Length | Current Window | Anchor Window |
|---|---|---|
| 500 steps (debug) | 50 | 100 → 200 |
| 3,500 steps (bench) | 105 | 300 → 500 |
| 100K steps (prod) | 300 | 300 → 500 |

**Asymmetric momentum:** LVS ramps up fast (momentum 0.8, ~5 steps) but decays slowly (momentum 0.995, half-life ~140 steps). This holds the LR boost much longer after peak signal.

**Binary boost:** When the model is improving at all (gap < -0.005 in log-space), LVS applies the full phase-adaptive max_boost — no proportional decay. This prevents the boost from eroding as improvement rate naturally slows.

**Phase-adaptive range:**

| Phase | Boost Range | Dampen Range |
|---|---|---|
| Early (step 0) | up to 1.3x | down to 0.7x |
| Late (final step) | up to 1.1x | down to 0.9x |

**Plateau burst:** If the EMA gap stays below 0.5% for 200 consecutive steps (after warmup), triggers a cosine-shaped LR spike (2x for 50 steps) to escape local minima.

### Kernel Backends

Three backends, auto-detected by priority:

| Backend | Requirement | Speed |
|---|---|---|
| Triton | `triton` package + CUDA GPU | Fastest (fused kernel) |
| CUDA | CUDA GPU + cpp_extension | ~80% of Triton |
| PyTorch | CPU or no GPU extensions | Baseline |

### Usage

```python
from rvelvet.training.velvet_optimizer import VelvetOptimizer

optimizer = VelvetOptimizer(
    model.parameters(),
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-3,
    max_grad_norm=1.0,
    entropy_adaptive=True,      # enable LVS
    perplexity_guided=True,     # enable PGM
)

# Tell LVS the total run length (required for adaptive windows)
optimizer.set_training_steps(max_steps=100000)

for step in range(max_steps):
    loss = model(batch)
    loss.backward()
    optimizer.clip_grad_norm_()
    optimizer.step()
    optimizer.zero_grad()

    # Feed loss to LVS + PGM (after optimizer.step)
    optimizer.set_loss_metrics(loss.item(), vocab_size=50257)

    # Logging
    print(f"lr={optimizer.effective_lr:.2e} "
          f"beta1={optimizer.effective_beta1:.3f} "
          f"lvs={optimizer.lr_scale:.3f} "
          f"sig={optimizer.lvs_confidence:.2f}")
```

To use AdamW instead, set `optimizer: adamw` in the training config.

### Plotting Training Runs

```bash
# Single run (generates 8-panel plot for Velvet, 4-panel for AdamW)
python scripts/plot_run.py outputs/phase1_pretrain/metrics.csv

# Compare Velvet vs AdamW
python scripts/plot_run.py outputs/velvet/metrics.csv outputs/adamw/metrics.csv \
    --labels Velvet AdamW
```

## Testing

```bash
# Architecture tests (shapes, gradients, memory persistence)
python tests/test_architecture.py

# ACR tests (routing, gating, load balance)
python tests/test_acr.py

# Iterative reasoning tests (LoRA, halting, deep supervision)
python tests/test_iterative_reasoning.py
```

## Project Structure

```
R-Velvet/
├── rvelvet/
│   ├── __init__.py
│   ├── model.py                     # RVelvet main model
│   ├── layers/
│   │   ├── local_attention.py       # Windowed self-attention
│   │   ├── segment_compressor.py    # Learned compression
│   │   ├── global_reasoner.py       # Full attention on concepts
│   │   ├── memory_controller.py     # External semantic memory
│   │   ├── adaptive_router.py       # ACR scanner + router
│   │   ├── lora_adapter.py          # Per-iteration LoRA
│   │   ├── halting.py               # PonderNet halting
│   │   └── iterative_reasoner.py    # Multi-pass orchestrator
│   ├── data/
│   │   └── text_dataset.py          # Memmap text dataset
│   └── training/
│       ├── losses.py                # Per-phase loss computation
│       ├── trainer.py               # Shared training loop
│       ├── velvet_optimizer.py      # VelvetOptimizer (AdamW + PGM + LVS)
│       └── kernels/
│           ├── velvet_triton.py     # Triton fused kernel
│           ├── velvet_cuda.py       # CUDA fallback kernel
│           ├── fused_ce.py          # Fused cross-entropy
│           └── era_triton.py        # ERA Triton kernel
├── configs/
│   ├── config.yaml
│   ├── model/
│   ├── training/
│   └── data/
├── scripts/
│   ├── train.py                     # Hydra entry point
│   ├── tokenize_data.py             # Text → .bin tokenizer
│   ├── plot_run.py                  # Training metrics visualization
│   ├── train_tokenizer.py           # BPE tokenizer training
│   └── download_culturax.py         # CulturaX dataset downloader
├── tests/
│   ├── test_architecture.py
│   ├── test_acr.py
│   └── test_iterative_reasoning.py
└── requirements.txt
```
