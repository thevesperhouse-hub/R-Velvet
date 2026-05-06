"""
R-Velvet unified training entry point.

Usage:
    # Phase 1: Base pretraining (preset model)
    python scripts/train.py --phase phase1_pretrain --model base

    # Auto-size to a target parameter budget (no preset needed)
    python scripts/train.py --phase phase1_pretrain --target-size 1.3B

    # Interactive: if neither --model nor --target-size is given,
    # the script prompts for a target size at startup.
    python scripts/train.py --phase phase1_pretrain

    # Phase 2: ACR finetuning (loads Phase 1 checkpoint)
    python scripts/train.py --phase phase2_acr --model base --resume outputs/phase1_pretrain/ckpt_final.pt

    # Phase 3: Iterative reasoning (loads Phase 2 checkpoint)
    python scripts/train.py --phase phase3_iterative --model base --resume outputs/phase2_acr/ckpt_final.pt

    # Debug (any phase)
    python scripts/train.py --phase phase1_pretrain --model small --debug

    # Override any config value
    python scripts/train.py --phase phase1_pretrain --model base --set training.lr=1e-4 training.batch_size=16
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy

from rvelvet.model import RVelvet
from rvelvet.data.text_dataset import TextDataset
from rvelvet.data.streaming_dataset import StreamingTextDataset
from rvelvet.training.trainer import Trainer
from rvelvet.utils.sizing import auto_size, parse_size, format_size


class Config(dict):
    """Dict that supports attribute access (cfg.model.d_model)."""
    def __getattr__(self, key):
        try:
            val = self[key]
            return val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key, val):
        self[key] = val

    @staticmethod
    def from_dict(d):
        cfg = Config()
        for k, v in d.items():
            if isinstance(v, dict):
                cfg[k] = Config.from_dict(v)
            else:
                cfg[k] = v
        return cfg


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return _coerce_numerics(data)


def _coerce_numerics(obj):
    if isinstance(obj, dict):
        return {k: _coerce_numerics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numerics(v) for v in obj]
    if isinstance(obj, str):
        try:
            return int(obj)
        except ValueError:
            try:
                return float(obj)
            except ValueError:
                return obj
    return obj


def load_config(phase: str, model=None, data: str = "text",
                config_dir: str = "configs",
                model_dict: dict = None) -> Config:
    """Load and merge YAML configs into a single Config object.

    If `model_dict` is provided (e.g. from auto_size), it's used directly
    instead of loading a YAML preset.
    """
    if model_dict is not None:
        model_cfg = model_dict
    else:
        model_cfg = load_yaml(os.path.join(config_dir, "model", f"{model}.yaml"))
    training_cfg = load_yaml(os.path.join(config_dir, "training", f"{phase}.yaml"))
    data_cfg = load_yaml(os.path.join(config_dir, "data", f"{data}.yaml"))

    merged = {
        'model': model_cfg,
        'training': training_cfg,
        'data': data_cfg,
        'seed': 42,
        'output_dir': f"outputs/{phase}",
    }
    return Config.from_dict(merged)


def estimate_memory(params: int, *, optimizer: str = "adamw",
                    precision: str = "bf16") -> dict:
    """Rough static memory footprint (excludes activations, which depend on
    batch/seq_len). Returns bytes for weights / grads / optim_states / total.

    bf16 mixed precision with AdamW: typically ~16 bytes/param at peak
    (2 weights + 2 grads + 4 m + 4 v + 4 master copy).
    """
    bpp_weights = 2 if precision == "bf16" else 4
    bpp_grads = 2 if precision == "bf16" else 4
    # AdamW: m + v in fp32 (8 bytes), plus fp32 master copy under mixed precision
    bpp_optim = 8 + (4 if precision == "bf16" else 0)

    weights = params * bpp_weights
    grads = params * bpp_grads
    optim = params * bpp_optim
    total = weights + grads + optim
    return {"weights": weights, "grads": grads, "optim": optim, "total": total}


def _fmt_gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def auto_size_from_target(target, *, vocab_size: int, max_seq_len: int):
    """Run auto_size and produce a model dict ready to feed into RVelvet."""
    result = auto_size(target, vocab_size=vocab_size, max_seq_len=max_seq_len)
    print()
    print(f"Auto-sized config (target {format_size(result.target)}):")
    print(result.pretty())
    mem = estimate_memory(result.params)
    print()
    print(f"  Static memory (bf16+AdamW, no activations):")
    print(f"    weights:        {_fmt_gb(mem['weights'])}")
    print(f"    grads:          {_fmt_gb(mem['grads'])}")
    print(f"    optimizer:      {_fmt_gb(mem['optim'])}")
    print(f"    total:          {_fmt_gb(mem['total'])}")
    print()
    return result.config


def prompt_target_size(default: str = "1.3B") -> str:
    """Interactively ask the user for a target model size."""
    print()
    print("=" * 60)
    print("  No --model or --target-size provided.")
    print("  Enter a target parameter budget for auto-sizing.")
    print("  Examples: 350M, 1.3B, 2.5B, 7B")
    print("=" * 60)
    raw = input(f"Target size [{default}]: ").strip() or default
    try:
        parse_size(raw)
    except ValueError as e:
        print(f"Invalid size: {e}")
        sys.exit(2)
    return raw


# ----------------------------------------------------------------------
# Interactive wizard
# ----------------------------------------------------------------------
def _ask(prompt: str, default=None, parser=None, validate=None):
    """Prompt with a default. Returns the parsed/validated value."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw:
            if default is None:
                print("  (required)")
                continue
            raw = str(default)
        try:
            val = parser(raw) if parser else raw
            if validate:
                validate(val)
            return val
        except (ValueError, AssertionError) as e:
            print(f"  invalid: {e}")


def _ask_yn(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{d}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes", "o", "oui"):
            return True
        if raw in ("n", "no", "non"):
            return False
        print("  answer y or n")


def _ask_choice(prompt: str, choices, default=None):
    """Numbered choice selector. Returns the chosen value."""
    print(f"\n{prompt}")
    default_idx = None
    for i, c in enumerate(choices, 1):
        marker = ""
        if c == default:
            marker = " (default)"
            default_idx = i
        print(f"  [{i}] {c}{marker}")
    while True:
        raw = input(f"Choice [{default_idx}]: ").strip()
        if not raw and default_idx is not None:
            return choices[default_idx - 1]
        try:
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
        except ValueError:
            pass
        print(f"  enter a number between 1 and {len(choices)}")


def _smart_batch_defaults(params: int) -> dict:
    """Per-step batch defaults based on model size, calibrated for ~80GB GPU.

    Conservative — user can scale up on bigger hardware. Effective batch
    targets ~256-512 sequences via gradient accumulation.
    """
    if params <= 500_000_000:
        return {"batch_size": 16, "seq_len": 2048, "grad_accum": 2}
    if params <= 2_000_000_000:
        return {"batch_size": 8, "seq_len": 4096, "grad_accum": 4}
    if params <= 5_000_000_000:
        return {"batch_size": 4, "seq_len": 4096, "grad_accum": 8}
    return {"batch_size": 2, "seq_len": 4096, "grad_accum": 16}


# Token-to-param ratios with explanatory labels.
_RATIO_PRESETS = [
    (20,   "Chinchilla 20:1 (compute-optimal, under-trained for inference)"),
    (100,  "LLaMA-1 era ~100:1 (balanced)"),
    (500,  "Modern conservative ~500:1 (good final quality)"),
    (1000, "TinyLlama-style ~1000:1 (over-trained, cheaper inference)"),
    (3000, "Heavy over-training ~3000:1 (maximum quality, big budget)"),
]


def _list_data_configs(config_dir: str = "configs") -> list:
    return sorted([p.stem for p in Path(config_dir, "data").glob("*.yaml")])


def run_wizard(phase: str, debug: bool, config_dir: str = "configs") -> dict:
    """Full interactive setup. Returns a dict with everything needed to
    configure the run:
        {
            "model_dict": dict,           # auto-sized model config
            "data_name": str,             # e.g. 'fineweb2_fr'
            "training_overrides": dict,   # to merge into training cfg
            "resume_path": str | None,
        }
    """
    print()
    print("=" * 64)
    print("  R-VELVET TRAINING WIZARD")
    print(f"  Phase: {phase}")
    print("=" * 64)
    print("Press Enter to accept the default in [brackets].")

    # 1. Target size + vocab
    print("\n--- Model size ---")
    target = _ask("Target parameter budget (e.g. 350M, 1.3B, 2.5B, 7B)",
                  default="1.3B", parser=lambda s: s,
                  validate=lambda s: parse_size(s))
    target_params = parse_size(target)
    vocab = _ask("Vocab size", default=64000, parser=int,
                 validate=lambda v: (_ for _ in ()).throw(
                     ValueError("vocab must be > 1000")) if v < 1000 else None)

    # 2. Data config (need it before sizing for max_seq_len)
    print("\n--- Data ---")
    data_choices = _list_data_configs(config_dir)
    if not data_choices:
        raise SystemExit("No configs/data/*.yaml found")
    default_data = "fineweb2_fr" if "fineweb2_fr" in data_choices else data_choices[0]
    data_name = _ask_choice("Pick a data config:", data_choices, default=default_data)
    data_yaml = load_yaml(os.path.join(config_dir, "data", f"{data_name}.yaml"))
    yaml_seq_len = data_yaml.get("seq_len", 2048)

    # 3. Sequence length (model.max_seq_len follows this)
    print("\n--- Sequence length ---")
    seq_len = _ask(f"Training sequence length (default from {data_name}.yaml)",
                   default=yaml_seq_len, parser=int,
                   validate=lambda v: (_ for _ in ()).throw(
                       ValueError("seq_len must be > 64")) if v < 64 else None)

    # 4. Auto-size and show the result
    print("\n--- Auto-sizing ---")
    sized = auto_size(target, vocab_size=vocab, max_seq_len=seq_len)
    print(sized.pretty())
    mem = estimate_memory(sized.params)
    print()
    print(f"  Static memory (bf16+AdamW, no activations):")
    print(f"    weights:        {_fmt_gb(mem['weights'])}")
    print(f"    grads:          {_fmt_gb(mem['grads'])}")
    print(f"    optimizer:      {_fmt_gb(mem['optim'])}")
    print(f"    total:          {_fmt_gb(mem['total'])}")
    if not _ask_yn("\nProceed with this config?", default=True):
        sys.exit(0)

    # 5. Batch / grad_accum (smart defaults from model size)
    print("\n--- Batch & gradient accumulation ---")
    defaults = _smart_batch_defaults(sized.params)
    print(f"  Recommended for {format_size(sized.params)}: "
          f"bs={defaults['batch_size']}, grad_accum={defaults['grad_accum']} "
          f"(effective batch = {defaults['batch_size'] * defaults['grad_accum']})")
    batch_size = _ask("Per-step batch size", default=defaults["batch_size"],
                      parser=int,
                      validate=lambda v: (_ for _ in ()).throw(
                          ValueError("batch_size must be > 0")) if v <= 0 else None)
    grad_accum = _ask("Gradient accumulation steps (1 = no accumulation)",
                      default=defaults["grad_accum"], parser=int)
    if grad_accum < 1:
        # Treat 0 / negative as "no accumulation" — that's grad_accum=1.
        grad_accum = 1

    # 6. Optimizer choice
    print("\n--- Optimizer ---")
    print("  velvet : AdamW + PGM (Perplexity-Guided Momentum) + LVS (Loss-Velocity")
    print("           Scaling). Auto-adapts beta1 and LR from loss dynamics. Best on")
    print("           long pretraining runs (>10k steps).")
    print("  adamw  : Vanilla PyTorch AdamW. Predictable, no adaptive heuristics. Use")
    print("           for short debug runs or as a baseline.")
    optimizer = _ask_choice("Optimizer:", ["velvet", "adamw"], default="velvet")

    # 7. Token ratio + max_steps
    print("\n--- Token budget ---")
    print("  Token-to-param ratio drives how long you train:")
    for r, label in _RATIO_PRESETS:
        print(f"    {r:>5}:1  {label}")
    print(f"    custom    Enter a custom ratio")
    while True:
        raw = input(f"Ratio [500]: ").strip() or "500"
        if raw.lower() == "custom":
            ratio = _ask("  Custom ratio (tokens per param)", default=500, parser=int)
            break
        try:
            ratio = int(raw)
            break
        except ValueError:
            print("  enter a number or 'custom'")

    target_tokens = sized.params * ratio
    tokens_per_step = batch_size * seq_len * grad_accum
    suggested_steps = max(100, target_tokens // tokens_per_step)
    print()
    print(f"  Target tokens:    {target_tokens:>15,}  ({ratio}:1 ratio)")
    print(f"  Tokens per step:  {tokens_per_step:>15,}  "
          f"({batch_size} x {seq_len} x {grad_accum})")
    print(f"  Suggested steps:  {suggested_steps:>15,}")
    max_steps = _ask("Max training steps (override or accept)",
                     default=int(suggested_steps), parser=int,
                     validate=lambda v: (_ for _ in ()).throw(
                         ValueError("steps must be > 0")) if v <= 0 else None)

    # 8. Wandb
    print("\n--- Logging ---")
    use_wandb = _ask_yn("Enable wandb logging?", default=False)
    wandb_project = "r-velvet"
    wandb_run = phase
    if use_wandb:
        wandb_project = _ask("Wandb project name", default="rvelvet-runs",
                             parser=lambda s: s)
        wandb_run = _ask("Wandb run name",
                         default=f"{phase}_{format_size(sized.params)}",
                         parser=lambda s: s)

    # 9. Resume
    print("\n--- Resume ---")
    resume_path = None
    if _ask_yn("Resume from a checkpoint?", default=False):
        resume_path = _ask("Checkpoint path",
                           default="outputs/phase1_pretrain/ckpt_final.pt",
                           parser=lambda s: s)

    # 10. Build training overrides + summary
    overrides = {
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "max_steps": max_steps,
        "optimizer": optimizer,
        "wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_run": wandb_run,
    }
    if resume_path:
        overrides["resume_from"] = resume_path

    # Also override seq_len in data config if user changed it
    data_overrides = {}
    if seq_len != yaml_seq_len:
        data_overrides["seq_len"] = seq_len

    print()
    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  Phase:              {phase}")
    print(f"  Target / actual:    {target} / {format_size(sized.params)} "
          f"({sized.error_pct:+.1f}%)")
    print(f"  d_model:            {sized.config['d_model']:,}")
    print(f"  Layers (loc/glob):  {sized.config['n_local_layers']} / "
          f"{sized.config['n_global_layers']}")
    print(f"  Vocab:              {vocab:,}")
    print(f"  Seq len:            {seq_len:,}")
    print(f"  Data config:        {data_name}")
    print(f"  Batch / accum:      {batch_size} x {grad_accum} "
          f"(effective {batch_size * grad_accum})")
    print(f"  Optimizer:          {optimizer}")
    print(f"  Max steps:          {max_steps:,}")
    print(f"  Total tokens:       {max_steps * tokens_per_step:,} "
          f"(~{(max_steps * tokens_per_step) / sized.params:.0f}:1 ratio)")
    print(f"  Wandb:              {'on' if use_wandb else 'off'}"
          + (f" ({wandb_project}/{wandb_run})" if use_wandb else ""))
    print(f"  Resume:             {resume_path or 'no'}")
    print(f"  Debug mode:         {debug}")
    print("=" * 64)
    if not _ask_yn("Launch training?", default=True):
        sys.exit(0)
    print()

    return {
        "model_dict": sized.config,
        "data_name": data_name,
        "training_overrides": overrides,
        "data_overrides": data_overrides,
        "resume_path": resume_path,
    }


def apply_overrides(cfg: Config, overrides: list):
    """Apply dotted key=value overrides like 'training.lr=1e-4'."""
    for override in overrides:
        if '=' not in override:
            print(f"Warning: skipping invalid override '{override}' (expected key=value)")
            continue
        key, val = override.split('=', 1)
        parts = key.split('.')

        obj = cfg
        for part in parts[:-1]:
            obj = obj[part]

        final_key = parts[-1]
        old_val = obj.get(final_key)
        if old_val is not None:
            if isinstance(old_val, bool):
                val = val.lower() in ('true', '1', 'yes')
            elif isinstance(old_val, int):
                val = int(val)
            elif isinstance(old_val, float):
                val = float(val)
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    if val.lower() in ('true', 'false'):
                        val = val.lower() == 'true'

        obj[final_key] = val


def build_model(cfg: Config) -> RVelvet:
    """Build RVelvet model from config."""
    mcfg = cfg.model
    tcfg = cfg.training

    model = RVelvet(
        vocab_size=mcfg.vocab_size,
        d_model=mcfg.d_model,
        n_local_layers=mcfg.n_local_layers,
        n_global_layers=mcfg.n_global_layers,
        n_local_heads=mcfg.n_local_heads,
        n_global_heads=mcfg.n_global_heads,
        window_size=mcfg.window_size,
        segment_size=mcfg.segment_size,
        n_concepts=mcfg.n_concepts,
        n_refine_layers=mcfg.n_refine_layers,
        memory_size=mcfg.memory_size,
        n_read_steps=mcfg.n_read_steps,
        ffn_mult=mcfg.ffn_mult,
        dropout=mcfg.dropout,
        max_seq_len=mcfg.max_seq_len,
        use_acr=tcfg.use_acr,
        use_iterative_reasoning=tcfg.use_iterative_reasoning,
        max_reasoning_iterations=mcfg.max_reasoning_iterations,
        lora_rank=mcfg.lora_rank,
        halt_threshold=mcfg.halt_threshold,
        lambda_p=mcfg.lambda_p,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="R-Velvet training")
    parser.add_argument("--phase", type=str, required=True,
                        choices=["phase1_pretrain", "phase2_acr", "phase3_iterative"],
                        help="Training phase")
    parser.add_argument("--model", type=str, default=None,
                        help="Model config name under configs/model/. "
                             "Mutually exclusive with --target-size.")
    parser.add_argument("--target-size", type=str, default=None,
                        help="Auto-size to a parameter budget (e.g. '1.3B'). "
                             "Bypasses --model and computes dims on the fly.")
    parser.add_argument("--vocab-size", type=int, default=64000,
                        help="Vocab size used for auto-sizing (default 64000). "
                             "Ignored when --model is given.")
    parser.add_argument("--save-auto-config", type=str, default=None,
                        help="If set, write the auto-sized model dict to this "
                             "YAML path (under configs/model/) for reuse.")
    parser.add_argument("--data", type=str, default="text",
                        help="Data config name (under configs/data/<name>.yaml). "
                             "Default 'text' uses the local .bin file; pick "
                             "'fineweb2_fr' or 'reasoning_fr' for streaming.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 10 steps, synthetic data")
    parser.add_argument("--set", nargs="*", default=[],
                        help="Override config values: key=value")
    args = parser.parse_args()

    if args.model and args.target_size:
        print("Error: --model and --target-size are mutually exclusive.")
        sys.exit(2)

    # Resolve model spec: explicit preset, target-size flag, or wizard.
    auto_dict = None
    wizard_result = None
    data_name = args.data

    if args.target_size:
        data_yaml = load_yaml(os.path.join("configs", "data", f"{data_name}.yaml"))
        max_seq_len = data_yaml.get("seq_len", 8192)
        auto_dict = auto_size_from_target(
            args.target_size, vocab_size=args.vocab_size, max_seq_len=max_seq_len,
        )
    elif args.model is None:
        # Full interactive wizard — only fires on a TTY without an explicit
        # model spec. Walks the user through size, data, batch, ratio, steps,
        # wandb, resume in one shot.
        if not sys.stdin.isatty():
            print("Error: no --model or --target-size given and stdin is not a TTY.")
            sys.exit(2)
        wizard_result = run_wizard(args.phase, debug=args.debug)
        auto_dict = wizard_result["model_dict"]
        data_name = wizard_result["data_name"]

    cfg = load_config(args.phase, args.model, data=data_name, model_dict=auto_dict)

    # Apply wizard overrides (after YAML load, before --set / explicit flags).
    if wizard_result:
        for k, v in wizard_result["training_overrides"].items():
            cfg.training[k] = v
        for k, v in wizard_result["data_overrides"].items():
            cfg.data[k] = v

    # Optionally persist the auto-sized config so it can be re-used as a preset.
    if args.save_auto_config and auto_dict is not None:
        target_path = Path("configs/model") / args.save_auto_config
        if not target_path.suffix:
            target_path = target_path.with_suffix(".yaml")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w") as f:
            yaml.safe_dump(auto_dict, f, sort_keys=False)
        print(f"Saved auto-sized config to: {target_path}")

    if args.resume:
        cfg.training.resume_from = args.resume
    if args.debug:
        cfg.training.debug = True
    apply_overrides(cfg, args.set)

    print("=" * 60)
    print(f"  R-VELVET TRAINING — {args.phase}")
    print("=" * 60)
    model_label = args.model if args.model else f"auto-sized -> {args.target_size or 'interactive'}"
    print(f"Model:    {model_label}")
    print(f"Phase:    {args.phase}")
    print(f"Debug:    {cfg.training.debug}")
    print(f"Resume:   {cfg.training.get('resume_from', None)}")
    print()

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    model = build_model(cfg)
    counts = model.count_parameters()
    print(f"Model: {cfg.model.name}")
    for name, count in counts.items():
        print(f"  {name:25s}: {count:>10,}")

    resume = cfg.training.get('resume_from', None)
    if resume:
        print(f"\nLoading checkpoint: {resume}")
        ckpt = torch.load(resume, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        if missing:
            print(f"  Missing keys (new modules): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    if cfg.training.debug:
        debug_path = Path("data/_debug.bin")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        n_tokens = cfg.data.seq_len * 100
        np.random.randint(0, cfg.model.vocab_size, size=n_tokens).astype(np.uint16).tofile(str(debug_path))
        dataset = TextDataset(str(debug_path), seq_len=cfg.data.seq_len)
        print(f"\nDebug dataset: {len(dataset)} samples ({n_tokens:,} random tokens)")
    elif cfg.data.get('sources', None):
        # Streaming multi-source mix (e.g. fineweb2_fr.yaml). Builds an
        # IterableDataset that interleaves HF streaming datasets and tokenizes
        # on-the-fly — required for >50B-token corpora that won't fit a .bin.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
        dataset = StreamingTextDataset.from_config(cfg.data, tok, seed=cfg.seed)
        print(f"\nStreaming dataset: {len(cfg.data.sources)} sources, "
              f"seq_len={cfg.data.seq_len}, "
              f"shuffle_buffer={getattr(cfg.data, 'shuffle_buffer', 10000)}")
    else:
        dataset = TextDataset(
            cfg.data.train_path,
            seq_len=cfg.data.seq_len,
            stride=cfg.data.stride,
        )
        print(f"\nDataset: {cfg.data.train_path} ({len(dataset)} samples)")

    trainer = Trainer(model, dataset, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
