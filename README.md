# R-Velvet

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1+-76B900.svg)
![Triton](https://img.shields.io/badge/Triton-3.0+-blueviolet.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Multi-scale transformer that handles 1M+ token contexts by compressing intelligently, reasoning globally, and remembering selectively. Instead of brute-forcing quadratic attention, R-Velvet processes local windows cheaply, compresses them into abstract "concepts", and runs full attention only over these concepts. Theoretical estimates suggest this could be ~3800x cheaper than full quadratic for million-token sequences (currently being tested).

Comes with **VelvetOptimizer**, a custom optimizer that beats AdamW by adapting momentum (PGM) and learning rate (LVS) based on training dynamics. Converges faster and maintains the advantage through late training.

## What makes this different

Most long-context models either use sparse attention (which breaks global reasoning) or hierarchical processing (which loses fine-grained information). R-Velvet does both without the tradeoffs:

**Local-to-global pipeline:** Windowed attention captures syntax and local semantics (cheap, parallelizable). Cross-attention compressor learns to extract the important bits into a small set of concept vectors. Global reasoner runs full attention over concepts, enabling true long-range reasoning like "paragraph 3 contradicts paragraph 47". Finally, concepts get expanded back to token-level for next-token prediction.

**Adaptive computation routing (ACR):** Not all text segments need the same compute. News articles need deep processing, filler text can be skimmed. ACR routes each segment to SKIM (10% cost), PROCESS (50% cost), or FOCUS (100% cost) based on learned heuristics. Target distribution is 60/30/10, reducing average compute by ~65%.

**Iterative reasoning:** For hard prompts, the model can loop through the global reasoner multiple times with per-iteration LoRA adapters. Each pass refines the concepts further. A PonderNet-style halting unit learns when to stop (between 1-8 iterations). Adds minimal overhead (1.6% params) but enables multi-step reasoning chains.

## Architecture

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

## Quick start

```bash
# Clone and install
git clone https://github.com/thevesperhouse-hub/R-Velvet.git
cd R-Velvet
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train a FR-tuned BPE tokenizer (default 64K vocab, FR-aware pre-tokenizer)
python scripts/train_tokenizer.py --input corpus.txt --output data/velvet_tok_64k

# Tokenize your corpus (adaptive uint16/uint32 — works for any vocab size)
python scripts/tokenize_data.py --input corpus.txt --output data/train.bin --tokenizer data/velvet_tok_64k

# Launch the interactive wizard (asks size, data, ratio, steps, batch, wandb…)
python scripts/train.py --phase phase1_pretrain

# Or skip the wizard with explicit flags
python scripts/train.py --phase phase1_pretrain --target-size 1.3B
```

For an end-to-end cloud launch guide (GPU choice, tokenizer prep, monitoring,
phase chaining, common pitfalls), see [`RUNBOOK.md`](RUNBOOK.md).

Training happens in three phases. Phase 1 pretrains the base model, Phase 2 enables ACR, Phase 3 adds iterative reasoning. Each phase resumes from the previous checkpoint with `--resume path/to/ckpt.pt`.

## Auto-sizing & interactive wizard

Pass a target parameter budget and R-Velvet derives the model dims for you. The sizer searches `(d_model, n_local_layers, n_global_layers)` on the meta device (no allocation), scoring candidates by both param-count error and aspect-ratio deviation from the LLaMA-2/3/Mistral family (`d_model / n_total_layers ≈ 110`). Lands within ~1.5% of the target.

```bash
# Auto-size to a target
python scripts/train.py --phase phase1_pretrain --target-size 1.3B

# Persist the auto-sized config for reuse as a preset
python scripts/train.py --phase phase1_pretrain --target-size 2.5B --save-auto-config my_2_5b.yaml

# Wizard: no flags → walks size, vocab, data, seq_len, batch, ratio, steps,
# wandb, resume in one guided pass. Recommends batch defaults from model
# size and computes max_steps from token budget / (batch × seq × grad_accum).
python scripts/train.py --phase phase1_pretrain
```

The wizard explains token-to-param ratios inline (Chinchilla 20:1, modern 500:1, TinyLlama 1000:1) and shows a final summary with effective batch + total tokens before launching. See [`RUNBOOK.md`](RUNBOOK.md) for the full cloud-launch playbook.

Sample output for `--target-size 1.3B --vocab-size 64000`:

```
Auto-sized config (target 1.30B):
  d_model:          1,792
  n_local_layers:   7
  n_global_layers:  13   (total 20 transformer layers)
  n_local_heads:    14   (head_dim=128)
  Total params:     1.28B (target 1.30B, -1.3%)

  Static memory (bf16+AdamW, no activations):
    weights:        2.55 GB
    grads:          2.55 GB
    optimizer:      15.31 GB
    total:          20.41 GB
```

Or use a YAML preset (`small`, `base`) with `--model base`. Override anything with `--set training.lr=1e-4 training.batch_size=16`.

## French data pipeline

For FR pretraining, two streaming corpora are wired up out of the box (no full download required — HF `interleave_datasets` with weighted sampling, EOS-separated cross-document packing, worker sharding):

```bash
# Phase 1/2: FineWeb-2 FR + Wikipedia FR + CulturaX + HAL + OSCAR + small code
python scripts/train.py --phase phase1_pretrain --target-size 1.3B --data fineweb2_fr

# Phase 3: reasoning mix (translated GSM8K/MATH + Wikipedia + anti-forgetting)
python scripts/build_reasoning_fr.py --output data/reasoning_fr   # NLLB translation
python scripts/train.py --phase phase3_iterative --target-size 1.3B --data reasoning_fr
```

The tokenizer trainer is FR-tuned: NFC normalization, Split layer that isolates apostrophe-contractions (`l'`, `d'`, `qu'`, `n'`) and individual digits, then byte-level BPE. Default vocab is 64k (sweet spot for 1.3B-2.5B FR). Larger vocabs (`--vocab-size 100000`) are supported transparently — `tokenize_data.py` switches to uint32 with a sidecar `<bin>.meta.json` so the dataset loader can pick the right dtype.

```bash
# Stream FineWeb-2 FR for tokenizer training (no disk corpus needed)
python scripts/train_tokenizer.py \
    --hf-source HuggingFaceFW/fineweb-2:fra_Latn:train \
    --max-examples 5_000_000 \
    --output data/velvet_tok_64k
```

## VelvetOptimizer

AdamW with two adaptive mechanisms that adjust to training dynamics:

**PGM (Perplexity-Guided Momentum)** adapts beta1 based on where the model is in training. Early on when the model is random, momentum stays normal. During active learning (mid-training), momentum drops to react faster to gradients. Near convergence, momentum increases for stability. Uses a simple U-curve based on loss/max_entropy ratio. Zero overhead.

**LVS (Loss-Velocity Scaling)** boosts the learning rate when loss is actively dropping, reduces it when loss stagnates or worsens. Tracks two EMAs of log(loss): a fast "current" EMA and a slow "anchor" EMA. When current < anchor, the model is improving → boost LR. When current > anchor, the model is worsening → reduce LR.

Log-space is critical. In linear space, as loss decreases (e.g., 8.0 → 4.0), the absolute gap between EMAs shrinks even if the relative improvement rate stays constant. This caused the signal to decay after ~300 steps. In log-space, a constant improvement rate produces a constant EMA gap.

EMA windows scale with total training steps (Chinchilla-inspired). Current window is 2% of max_steps, anchor starts at 10% and grows to 25%. For a 16K step run, current=150, anchor grows from 1200→4000. This keeps the signal stable across different run lengths.

Asymmetric momentum: when the signal says "boost", LR ramps up fast (momentum 0.8). When the signal says "reduce", LR decays slowly (momentum 0.995, half-life ~140 steps). This holds the boost longer.

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

optimizer.set_training_steps(max_steps=16000)

for step in range(max_steps):
    loss = model(batch)
    loss.backward()
    optimizer.clip_grad_norm_()
    optimizer.step()
    optimizer.zero_grad()

    optimizer.set_loss_metrics(loss.item(), vocab_size=32000)

    # Monitor adaptive mechanisms
    print(f"lr={optimizer.effective_lr:.2e} "
          f"beta1={optimizer.effective_beta1:.3f} "
          f"lvs={optimizer.lr_scale:.3f}")
```

To use AdamW instead, set `training.optimizer=adamw` in configs.

### Velvet vs AdamW on 50M params, 16K steps

![Comparison](assets/comparison.png)

Velvet converges faster and maintains a consistent advantage throughout training. Final loss: **3.59** (Velvet) vs **3.61** (AdamW). The LR plot shows Velvet maintaining a smooth 15-20% boost above the base cosine schedule.

Visualize your own runs:

```bash
python scripts/plot_run.py outputs/phase1/metrics.csv outputs/adamw/metrics.csv --labels Velvet AdamW
```

## Model configs

| Config | d_model | Layers (local/global) | Heads | Params | Use case |
|---|---|---|---|---|---|
| `small` | 256 | 4 / 4 | 4 / 4 | ~5M | Debug, fast iteration |
| `base` | 384 | 6 / 8 | 6 / 8 | ~50M | Production |

Or skip presets entirely with `--target-size` (see [Auto-sizing](#auto-sizing) above). The sizer reproduces LLaMA-family shapes:

| Target | d_model | Layers | Heads | Aspect | Sized |
|---|---|---|---|---|---|
| 350M  | 1024 | 14 | 8  | 73  | -1.4% |
| 1.3B  | 1792 | 20 | 14 | 90  | -1.3% |
| 2.5B  | 2304 | 25 | 18 | 92  | -0.1% |
| 7B    | 3584 | 30 | 28 | 119 | -1.0% |

Optional modes add minimal overhead: ACR adds ~0.7% params (scanner), iterative reasoning adds ~1.6% params (LoRA bank + halting unit).

## Training phases

**Phase 1 — Base pretraining.** Standard LM training. All params trained with cross-entropy. LR 3e-4, cosine decay, 2K warmup. Modern token-to-param ratios (TinyLlama 2700:1, LLaMA-3.2-1B 9000:1) prefer one pass over a large unique-token corpus rather than the older Chinchilla 20:1 with multiple epochs.

**Phase 2 — ACR finetuning.** Enables adaptive routing. Loads Phase 1 checkpoint with `strict=False` to add new modules. Base params get LR×0.1, ACR params get full LR. Loss is CE + load_balance + entropy + compute_cost.

**Phase 3 — Iterative reasoning.** Freezes base model, only trains LoRA bank + halting unit + iteration embeddings. Loss is CE + halting + deep_supervision (each iteration's output gets supervised).

Each phase resumes from the previous: `--resume outputs/phase1_pretrain/ckpt_final.pt`

## Configuration

YAML configs live in `configs/{model,training,data}/`. Override anything from the CLI:

```bash
# Smaller batch, longer run
python scripts/train.py --phase phase1_pretrain --model base \
    --set training.batch_size=16 training.max_steps=20000

# Use small model for quick tests
python scripts/train.py --phase phase1_pretrain --model small

# Enable wandb logging
python scripts/train.py --phase phase1_pretrain --model base \
    --set training.wandb=true training.wandb_project=my-project
```

Debug mode runs a short loop on synthetic data:

```bash
python scripts/train.py --phase phase1_pretrain --target-size 350M --debug
```

## Kernel backends

VelvetOptimizer has three backends, auto-detected by priority:

**Triton** (fastest): Fused kernel, single kernel launch per parameter update. Requires `triton` package + CUDA GPU.

**CUDA** (fallback): Native CUDA kernel via cpp_extension. ~80% of Triton speed.

**PyTorch** (baseline): Pure PyTorch fallback for CPU or when extensions aren't available.

The optimizer prints which backend is active at training start.

## Project structure

```
R-Velvet/
├── rvelvet/
│   ├── model.py                     # Main model
│   ├── layers/                      # All architecture components
│   ├── data/
│   │   ├── text_dataset.py          # Memory-mapped .bin loader (adaptive dtype)
│   │   └── streaming_dataset.py     # HF streaming + interleave + packing
│   ├── utils/
│   │   ├── sizing.py                # Auto-sizing engine
│   │   └── dtypes.py                # uint16/uint32 + sidecar metadata
│   └── training/
│       ├── trainer.py               # Training loop
│       ├── velvet_optimizer.py      # Custom optimizer
│       └── kernels/                 # Triton/CUDA kernels
├── configs/                         # YAML configs (model/training/data)
├── scripts/
│   ├── train.py                     # Main entry point (--target-size aware)
│   ├── train_tokenizer.py           # BPE training (FR-tuned, 64k default)
│   ├── tokenize_data.py             # Corpus → .bin (adaptive dtype + sidecar)
│   ├── build_reasoning_fr.py        # NLLB translation for GSM8K/MATH → FR
│   └── plot_run.py                  # Visualize training
└── tests/                           # 119 unit tests
```

## Tests

```bash
pytest tests/                        # full suite (119 tests)
pytest tests/test_sizing.py -v       # auto-sizing + adaptive dtype
pytest tests/test_optimizer.py -v    # 3-backend optimizer parity
```

## License

MIT
