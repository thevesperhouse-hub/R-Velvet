# R-Velvet Cloud Runbook

End-to-end guide for launching an R-Velvet training run on a fresh cloud GPU
(Lambda Cloud, RunPod, vast.ai, etc.). Tested for 1.3B–7B models on a single
GPU with bf16 + AdamW (or VelvetOptimizer).

> **Prerequisites:** SSH access to a Linux GPU instance with CUDA 12.x and
> Python 3.10+. Hugging Face account if you intend to stream FineWeb-2.

---

## 1. Choose your hardware

For a **single-GPU** run (current code path — DDP/FSDP not yet wired):

| Target size | Min VRAM     | Reco GPU         | Headroom for activations |
|------------:|:-------------|:-----------------|:-------------------------|
| 350M        | 12 GB        | A10 24 GB        | comfortable              |
| 1.3B        | 24 GB        | A100 40 GB       | tight but OK             |
| 1.3B        | -            | **H100 80 GB**   | recommended              |
| 2.5B        | 40 GB        | H100 80 GB       | comfortable              |
| 7B          | 80 GB        | H100 80 GB       | tight, may need grad-ckpt|
| 7B+         | multi-GPU    | (FSDP TBD)       | not supported yet        |

On Lambda Cloud, **1× H100 SXM 80 GB** is the best credit/throughput trade
for the 1.3B–2.5B range.

Persistent storage: 200 GB is enough (HF dataset cache + checkpoints).

---

## 2. Provision and connect

```bash
# After SSH'ing in:
nvidia-smi                       # confirm GPU + driver
python --version                 # >= 3.10
git --version
```

---

## 3. Clone & install

```bash
git clone git@github.com:thevesperhouse-hub/R-Velvet.git
cd R-Velvet

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install datasets transformers tokenizers   # if not already pulled

# HF login (required for FineWeb-2 — accept the dataset card on huggingface.co first)
huggingface-cli login
```

Sanity check the install:

```bash
pytest tests/ -q --ignore=tests/_bench_kernels.py
# expect: 119 passed
```

---

## 4. (Optional) Train a 64k FR tokenizer

Skip if you're reusing `data/velvet_tok` (32k existing). The 64k version
gives ~10–15% better compression on French — worth the extra setup time
for a long pretraining run.

```bash
python scripts/train_tokenizer.py \
    --hf-source HuggingFaceFW/fineweb-2:fra_Latn:train \
    --max-examples 5_000_000 \
    --vocab-size 64000 \
    --output data/velvet_tok_64k

# Verify chars/token at the end of the run:
#   ≥ 4.0 = OK for FR
#   ≥ 4.5 = excellent
```

Update `configs/data/fineweb2_fr.yaml` to point at `data/velvet_tok_64k`
if you trained a new one.

---

## 5. Launch the wizard

The simplest path. Just run:

```bash
python scripts/train.py --phase phase1_pretrain
```

The wizard walks you through:

1. **Target parameter budget** (`350M`, `1.3B`, `2.5B`, `7B`, …)
2. **Vocab size** (default 64000 — match your tokenizer)
3. **Data config** — picks one of `configs/data/*.yaml`
4. **Sequence length** — overrides the yaml default
5. **Auto-sized config preview** + memory estimate (asks to confirm)
6. **Batch size / gradient accumulation** — smart defaults by model size
7. **Token-to-param ratio** — explained inline (Chinchilla 20:1, modern 500:1, TinyLlama 1000:1)
8. **Max steps** — auto-computed from `target_tokens / (batch × seq × grad_accum)`, override at will
9. **Wandb on/off + project/run name**
10. **Resume from checkpoint** (optional)
11. **Final summary + confirm before launch**

### Token ratio cheat sheet

| Ratio  | Style                | Example models               | When to use |
|-------:|:---------------------|:-----------------------------|:------------|
| 20:1   | Chinchilla optimal   | Chinchilla 70B               | Compute-bound, no inference cost concern |
| 100:1  | LLaMA-1 era          | LLaMA-1 7B                   | Balanced |
| 500:1  | Modern conservative  | LLaMA-2 7B (~285:1)          | Good final quality |
| 1000:1 | Mildly over-trained  | LLaMA-3 8B (~1875:1)         | Cheaper inference, small extra cost |
| 3000:1 | Heavily over-trained | TinyLlama 1.1B (~2700:1)     | Maximum quality, big budget |

For a 1.3B model:

- 20:1   = 26 B tokens
- 500:1  = 650 B tokens
- 1000:1 = 1.3 T tokens

With `bs=8, seq=4096, grad_accum=4` (≈ 131 K tokens/step):

- 26 B  → ~200 K steps
- 650 B → ~5 M steps (overkill for a first run)
- 1.3 T → ~10 M steps

A reasonable **first** 1.3B run: 50–100 K steps with `seq_len=4096`, batch=8, grad_accum=4 — checks the curve before committing to the long haul.

---

## 6. Run inside `tmux`

SSH disconnects kill foreground processes. Always launch inside `tmux`:

```bash
tmux new -s velvet
source venv/bin/activate

python scripts/train.py --phase phase1_pretrain   # then walk the wizard
# OR fully non-interactive:
python scripts/train.py \
    --phase phase1_pretrain \
    --target-size 1.3B \
    --vocab-size 64000 \
    --data fineweb2_fr \
    --set training.batch_size=8 \
          training.grad_accum_steps=4 \
          training.max_steps=50000 \
          training.wandb=true \
          training.wandb_project=rvelvet-1.3b \
    2>&1 | tee outputs/phase1_pretrain/run.log
```

Detach: `Ctrl+B` then `D`. Reattach: `tmux attach -t velvet`.

---

## 7. Monitoring during the run

**Inside the tmux pane:**
- `tokens/sec` printed every `log_every` steps (target ≥ 60% of theoretical peak)
- `grad_norm` — alarming if it spikes > 10 on a stable run
- `lr` (with LVS scale visible if VelvetOptimizer is on)

**From a second SSH session:**
```bash
nvidia-smi dmon                # GPU util / memory / power
tail -f outputs/phase1_pretrain/metrics.csv
tmux attach -t velvet          # peek at the live training log
```

**Wandb** (if enabled): live loss / lr / grad_norm / token rate dashboards.

### Red flags

| Symptom                                  | Likely cause                       | Action |
|:-----------------------------------------|:-----------------------------------|:-------|
| GPU util < 60%                           | Data loader bottleneck             | `--set data.num_workers=8` |
| Loss = NaN/Inf after a few steps         | LR too high / bad init             | Halve `training.lr`, restart |
| Grad-norm > 10 spikes                    | Loss spike, will likely recover    | Watch; if persists, lower LR |
| OOM after warmup                         | Batch + seq_len too large          | Halve `batch_size`, double `grad_accum_steps` |
| `tokens/sec` collapses mid-run           | Disk I/O for HF cache              | Check `df -h`; HF cache may be filling up |

---

## 8. Three-phase strategy

R-Velvet trains in three phases. Each loads the previous checkpoint with
`strict=False` so newly-introduced modules (ACR scanner, LoRA bank, halting)
get random-init while the rest is preserved.

```bash
# Phase 1 — base pretraining
python scripts/train.py --phase phase1_pretrain --target-size 1.3B --data fineweb2_fr

# Phase 2 — ACR finetuning (adaptive compute routing)
python scripts/train.py \
    --phase phase2_acr --target-size 1.3B --data fineweb2_fr \
    --resume outputs/phase1_pretrain/ckpt_final.pt

# Phase 3 — iterative reasoning (LoRA + halting)
python scripts/build_reasoning_fr.py --output data/reasoning_fr   # NLLB FR translation
python scripts/train.py \
    --phase phase3_iterative --target-size 1.3B --data reasoning_fr \
    --resume outputs/phase2_acr/ckpt_final.pt
```

> **Caveat:** the current `_load_checkpoint` resumes model+optim+scheduler+scaler
> but doesn't replay the streaming dataset position. Plan to keep each phase
> in a single uninterrupted session for now (DDP/FSDP and skip-steps are on the backlog).

---

## 9. Cost & timing

H100 80 GB SXM on Lambda: ~$2.50–3.00/hour.

You won't know your real `tokens/sec` until the smoke run — it depends on
streaming overhead, num_workers, ACR/iter mode, and seq_len. Plan to spend
~$20–50 on a Phase 1 50K-step shakedown to calibrate before committing to
the long run.

The repo logs `tokens/sec` and timestamps on every interval — read the first
few hundred steps to extrapolate the full run cost before you walk away.

---

## 10. Saving & evaluating

Checkpoints land in `outputs/<phase>/ckpt_<step>.pt`. The trainer keeps
the last `keep_last_n` (default 5) and writes `ckpt_final.pt` at the end.

To pull a checkpoint back to your laptop:

```bash
# from the cloud instance
cd outputs/phase1_pretrain
ls -lh ckpt_*.pt

# from your laptop:
scp -i ~/.ssh/lambda.pem ubuntu@<ip>:/path/to/R-Velvet/outputs/phase1_pretrain/ckpt_final.pt .
```

Evaluation harness is not yet wired into this repo. For now, load the
checkpoint manually with `torch.load(..., weights_only=True)` and feed
sequences through `model.forward`.

---

## 11. Known gaps / backlog

For full transparency — these are not implemented yet:

- **DDP / FSDP** for multi-GPU
- **`generate()` + KV cache** for sampling
- **Validation loop** during training
- **Token packing in IterableDataset** with skip-steps for clean resume
- **Loss spike auto-recovery** (rewind + LR ÷10)
- **OOM auto-recover** (catch + retry with smaller batch)
- **MFU logging** alongside tokens/sec
- **WSD / cyclic schedulers**
- **EMA model weights** for evaluation
- **`torch.profiler` hook**

Anything new you'd want here, open an issue.

---

## Quick reference

```bash
# Wizard mode
python scripts/train.py --phase phase1_pretrain

# Non-interactive
python scripts/train.py --phase phase1_pretrain \
    --target-size 1.3B --vocab-size 64000 --data fineweb2_fr \
    --set training.batch_size=8 training.grad_accum_steps=4 training.max_steps=50000

# Tokenizer
python scripts/train_tokenizer.py --hf-source HuggingFaceFW/fineweb-2:fra_Latn:train \
    --max-examples 5_000_000 --vocab-size 64000 --output data/velvet_tok_64k

# Tokenize a local file (adaptive uint16/uint32 with sidecar)
python scripts/tokenize_data.py --input corpus.txt --output data/train.bin --tokenizer data/velvet_tok_64k

# Run tests
pytest tests/ -q --ignore=tests/_bench_kernels.py
```
