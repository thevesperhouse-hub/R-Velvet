"""
R-Velvet unified training entry point.

Usage:
    # Phase 1: Base pretraining
    python scripts/train.py --phase phase1_pretrain --model base

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


def load_config(phase: str, model: str, data: str = "text", config_dir: str = "configs") -> Config:
    """Load and merge YAML configs into a single Config object."""
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
    parser.add_argument("--model", type=str, default="base", choices=["small", "base", "1_3b"],
                        help="Model config name")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--data", type=str, default="text",
                        help="Data config name (under configs/data/)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 10 steps, synthetic data")
    parser.add_argument("--set", nargs="*", default=[],
                        help="Override config values: key=value")
    args = parser.parse_args()

    cfg = load_config(args.phase, args.model, data=args.data)

    if args.resume:
        cfg.training.resume_from = args.resume
    if args.debug:
        cfg.training.debug = True
    apply_overrides(cfg, args.set)

    print("=" * 60)
    print(f"  R-VELVET TRAINING — {args.phase}")
    print("=" * 60)
    print(f"Model:    {args.model}")
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
    resume_ckpt = None
    if resume:
        print(f"\nLoading checkpoint: {resume}")
        resume_ckpt = torch.load(resume, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(resume_ckpt['model'], strict=False)
        if missing:
            print(f"  Missing keys (new modules): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print(f"  Resuming from step {resume_ckpt.get('step', '?')}")

    if cfg.training.debug:
        debug_path = Path("data/_debug.bin")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        n_tokens = cfg.data.seq_len * 100
        np.random.randint(0, cfg.model.vocab_size, size=n_tokens).astype(np.uint16).tofile(str(debug_path))
        dataset = TextDataset(str(debug_path), seq_len=cfg.data.seq_len)
        print(f"\nDebug dataset: {len(dataset)} samples ({n_tokens:,} random tokens)")
    elif hasattr(cfg.data, 'sources'):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
        dataset = StreamingTextDataset(
            sources=cfg.data.sources,
            tokenizer=tokenizer,
            seq_len=cfg.data.seq_len,
            shuffle_buffer=getattr(cfg.data, 'shuffle_buffer', 1000),
            max_doc_tokens=getattr(cfg.data, 'max_doc_tokens', 65536),
            stopping_strategy=getattr(cfg.data, 'stopping_strategy', 'all_exhausted'),
            eos_token_id=getattr(cfg.data, 'eos_token_id', None),
        )
        print(f"\nStreaming dataset: {len(cfg.data.sources)} source(s), tokenizer vocab={len(tokenizer)}")
    else:
        dataset = TextDataset(
            cfg.data.train_path,
            seq_len=cfg.data.seq_len,
            stride=cfg.data.stride,
        )
        print(f"\nDataset: {cfg.data.train_path} ({len(dataset)} samples)")

    trainer = Trainer(model, dataset, cfg)

    if resume_ckpt:
        trainer.resume_from_checkpoint(resume_ckpt)

    trainer.train()


if __name__ == "__main__":
    main()
