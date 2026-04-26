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
│       └── trainer.py               # Shared training loop
├── configs/
│   ├── config.yaml
│   ├── model/
│   ├── training/
│   └── data/
├── scripts/
│   ├── train.py                     # Hydra entry point
│   └── tokenize_data.py             # Text → .bin tokenizer
├── tests/
│   ├── test_architecture.py
│   ├── test_acr.py
│   └── test_iterative_reasoning.py
└── requirements.txt
```
