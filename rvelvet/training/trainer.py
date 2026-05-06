"""Shared trainer for all three training phases with per-phase parameter groups,
mixed precision, gradient accumulation, LR scheduling, checkpointing, and logging."""

import os
# expandable_segments reduces fragmentation on large/variable-shape allocations
# (e.g. KV-cache style buffers, dynamic seq lens). Set BEFORE any CUDA init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import math
import random
import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from .losses import compute_phase_loss
from .velvet_optimizer import VelvetOptimizer


def set_global_seed(seed: int, *, deterministic: bool = False):
    """Reproducibility helper. Seeds torch, cuda, numpy, random.

    deterministic=True forces deterministic CUDA kernels (slower) — only useful
    for diff-checks against a baseline. Default leaves cudnn.benchmark on.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _safe_torch_load(path, map_location):
    """Prefer weights_only=True (security + future-proof). Fall back to legacy
    load when the checkpoint contains non-tensor objects we explicitly trust
    (e.g. older R-Velvet checkpoints with Python state)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        warnings.warn(
            f"weights_only=True load failed for {path}; retrying with weights_only=False. "
            "Only do this for checkpoints you trust.",
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


class Trainer:
    """Multi-phase trainer for R-Velvet. Phase differences handled via parameter groups,
    loss computation, and YAML configs."""

    def __init__(self, model, train_dataset, cfg):
        self.model = model
        self.train_dataset = train_dataset
        self.cfg = cfg
        self.tcfg = cfg.training  # training sub-config
        self.phase = self.tcfg.phase

        # ------- Performance / determinism toggles -------
        seed = int(getattr(self.tcfg, 'seed', 0))
        deterministic = bool(getattr(self.tcfg, 'deterministic', False))
        set_global_seed(seed, deterministic=deterministic)

        if torch.cuda.is_available():
            # Stable shapes (typical training step) → cudnn benchmark wins.
            torch.backends.cudnn.benchmark = not deterministic
            # Allow TF32 on the residual fp32 matmuls (Ampere+); free ~10%
            # speed with negligible accuracy cost.
            torch.set_float32_matmul_precision(
                getattr(self.tcfg, 'matmul_precision', 'high')
            )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.amp_dtype = self._resolve_amp_dtype()
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.amp_dtype == torch.float16))

        # Activation checkpointing: trade compute for VRAM on the deep stacks.
        # Toggled via cfg.training.gradient_checkpointing. use_reentrant=False
        # is set in the encoder/reasoner forwards.
        if getattr(self.tcfg, 'gradient_checkpointing', False):
            self._enable_gradient_checkpointing()

        if self.phase == 'phase3_iterative':
            self._freeze_for_phase3()

        # Optional torch.compile — opt-in because it can conflict with the
        # custom Triton kernels under some configs. Tries reduce-overhead by
        # default; falls back gracefully if compilation raises.
        if getattr(self.tcfg, 'compile', False):
            self._try_compile_model()

        self.optimizer = self._build_optimizer()
        self.scheduler = None
        self.global_step = 0
        self.skipped_steps = 0
        self.best_loss = float('inf')
        # Throughput tracking (tokens/sec) — populated in train().
        self._tokens_window = 0
        self._tokens_t0 = None

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        resume_from = getattr(self.tcfg, 'resume_from', None)
        if resume_from:
            self._load_checkpoint(resume_from)

    def _resolve_amp_dtype(self):
        amp_setting = getattr(self.tcfg, 'amp', 'bf16')
        if amp_setting == 'bf16' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif amp_setting in ('fp16', 'bf16'):
            return torch.float16
        return torch.float32

    def _try_compile_model(self):
        """Wrap self.model with torch.compile. Best-effort: failures are warned
        about but never abort training (compile pipelines can be flaky on new
        torch/triton combos)."""
        mode = getattr(self.tcfg, 'compile_mode', 'reduce-overhead')
        try:
            self.model = torch.compile(self.model, mode=mode)
            print(f"  torch.compile enabled (mode={mode})")
        except Exception as e:
            warnings.warn(f"torch.compile failed ({e}); running eager.", stacklevel=2)

    def _enable_gradient_checkpointing(self):
        """Enable activation checkpointing on the deepest stacks (LocalEncoder,
        GlobalReasoner). Each block's forward becomes a checkpointed call —
        activations are recomputed during backward, saving ~30-50% VRAM at the
        cost of one extra forward pass per checkpointed block.
        """
        for module in self.model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True

    def _freeze_for_phase3(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        if hasattr(self.model, 'iterative_reasoner'):
            ir = self.model.iterative_reasoner
            for p in ir.lora_bank.parameters():
                p.requires_grad_(True)
            for p in ir.halting_unit.parameters():
                p.requires_grad_(True)
            ir.iteration_embed.requires_grad_(True)

    def _build_optimizer(self):
        tcfg = self.tcfg

        if self.phase == 'phase1_pretrain':
            params = [p for p in self.model.parameters() if p.requires_grad]
            param_groups = [{'params': params, 'lr': tcfg.lr}]

        elif self.phase == 'phase2_acr':
            acr_names = {'scanner', 'router', 'adaptive_compressor'}
            acr_params = []
            base_params = []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if any(n in name for n in acr_names):
                    acr_params.append(p)
                else:
                    base_params.append(p)

            base_lr = tcfg.lr * getattr(tcfg, 'lr_base_factor', 0.1)
            param_groups = [
                {'params': base_params, 'lr': base_lr},
                {'params': acr_params, 'lr': tcfg.lr},
            ]

        elif self.phase == 'phase3_iterative':
            params = [p for p in self.model.parameters() if p.requires_grad]
            param_groups = [{'params': params, 'lr': tcfg.lr}]

        else:
            raise ValueError(f"Unknown phase: {self.phase}")

        optimizer_type = getattr(tcfg, 'optimizer', 'adamw')
        self.use_velvet = (optimizer_type == 'velvet')

        if self.use_velvet:
            # All Velvet thresholds are individually overridable from YAML.
            # We read them via getattr so older configs keep working with
            # the optimizer's own defaults.
            velvet_kwargs = dict(
                betas=(tcfg.beta1, tcfg.beta2),
                eps=getattr(tcfg, 'eps', 1e-8),
                weight_decay=tcfg.weight_decay,
                max_grad_norm=tcfg.grad_clip,
                entropy_adaptive=getattr(tcfg, 'entropy_adaptive', True),
                sparse_aware=getattr(tcfg, 'sparse_aware', True),
                perplexity_guided=getattr(tcfg, 'perplexity_guided', True),
            )
            # LVS / plateau / burst / PGM tunables — only forwarded if set,
            # so we don't override the optimizer's own defaults silently.
            for key in (
                'lvs_min_scale', 'lvs_max_scale', 'lvs_gap_clamp',
                'lvs_gap_dead_zone', 'lvs_gap_strength_full',
                'lvs_momentum_up', 'lvs_momentum_down', 'lvs_phase_decay',
                'plateau_threshold', 'plateau_patience',
                'burst_duration', 'burst_multiplier', 'burst_warmup_steps',
                'pgm_min_scale', 'pgm_max_scale',
                'sparse_threshold', 'skip_nonfinite',
            ):
                if hasattr(tcfg, key):
                    velvet_kwargs[key] = getattr(tcfg, key)
            optimizer = VelvetOptimizer(param_groups, **velvet_kwargs)
        else:
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=(tcfg.beta1, tcfg.beta2),
                weight_decay=tcfg.weight_decay,
            )
        return optimizer

    def _build_scheduler(self, max_steps: int):
        warmup = self.tcfg.warmup_steps
        min_lr = self.tcfg.min_lr

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(max_steps - warmup, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr_ratio = min_lr / self.tcfg.lr
            return lr_ratio + (1.0 - lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        tcfg = self.tcfg
        max_steps = getattr(tcfg, 'debug_steps', 500) if tcfg.debug else tcfg.max_steps
        log_every = getattr(tcfg, 'debug_log_every', 10) if tcfg.debug else tcfg.log_every
        accum_steps = tcfg.grad_accum_steps

        self.scheduler = self._build_scheduler(max_steps)
        # Restore scheduler state captured during _load_checkpoint (before scheduler existed).
        resume_sched = getattr(self, '_resume_scheduler_state', None)
        if resume_sched is not None:
            try:
                self.scheduler.load_state_dict(resume_sched)
            except Exception:
                pass
            self._resume_scheduler_state = None

        if self.use_velvet:
            self.optimizer.set_training_steps(max_steps)
        csv_path = self.output_dir / "metrics.csv"
        csv_mode = 'a' if (self.global_step > 0 and csv_path.exists()) else 'w'
        csv_file = open(csv_path, csv_mode, newline='')
        csv_writer = csv.writer(csv_file)
        csv_headers = ['step', 'loss', 'ce', 'ppl', 'lr', 'elapsed', 'tok_per_s']
        if self.use_velvet:
            csv_headers += ['beta1', 'lvs_scale', 'signal', 'pgm_scale', 'grad_norm', 'lvs_phase', 'skipped']
        if csv_mode == 'w':
            csv_writer.writerow(csv_headers)

        # Default to a small worker pool when not specified — keeps CPU pre-processing
        # off the main thread so the GPU isn't waiting on tokenization/IO.
        num_workers = int(getattr(self.cfg.data, 'num_workers', 2))
        # IterableDataset (streaming) cannot use shuffle / drop_last on the
        # DataLoader side — shuffling lives inside the dataset's own buffer,
        # and we must not silently drop the tail.
        is_iterable = isinstance(self.train_dataset, torch.utils.data.IterableDataset)
        loader_kwargs = dict(
            batch_size=tcfg.batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if not is_iterable:
            loader_kwargs['shuffle'] = True
            loader_kwargs['drop_last'] = True
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = int(
                getattr(self.cfg.data, 'prefetch_factor', 2)
            )
        loader = DataLoader(self.train_dataset, **loader_kwargs)

        self._wandb_active = False
        if tcfg.wandb and not tcfg.debug:
            import wandb
            wandb.init(project=tcfg.wandb_project, name=tcfg.wandb_run)
            self._wandb_active = True

        self.model.train()
        data_iter = iter(loader)
        self.optimizer.zero_grad()

        accum_loss = 0.0
        loss_dict_accum = {}
        t0 = time.time()
        # Reset throughput window to "now" so the first log isn't biased by
        # model-build / first-batch warmup time.
        self._tokens_window = 0
        self._tokens_t0 = t0
        # Snapshot optimizer skip count so per-window deltas are accurate
        # even after a resume (where self.skipped_steps was loaded from ckpt).
        skipped_at_window_start = (
            self.optimizer.skipped_steps if self.use_velvet else 0
        )

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        optim_name = 'VelvetOptimizer' if self.use_velvet else 'AdamW'
        print(f"Phase: {self.phase}")
        print(f"Trainable params: {n_trainable:,}")
        print(f"Device: {self.device} | AMP: {self.amp_dtype}")
        print(f"Optimizer: {optim_name}", end="")
        if self.use_velvet:
            print(f" (backend={self.optimizer.kernel_backend}, PGM+LVS)")
            print(f"  EMA windows: current={self.optimizer._current_window}, "
                  f"anchor={self.optimizer._anchor_window_min}→{self.optimizer._anchor_window_max}")
        else:
            print()
        print(f"Effective batch: {tcfg.batch_size * accum_steps}")
        print(f"Max steps: {max_steps}")
        print("-" * 60)

        try:
            while self.global_step < max_steps:
                step_loss = 0.0

                for micro_step in range(accum_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        batch = next(data_iter)

                    input_ids = batch['input_ids'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    # Count actual tokens fed this micro-batch for tokens/sec.
                    self._tokens_window += int(input_ids.numel())

                    with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=(self.amp_dtype != torch.float32)):
                        output = self.model(input_ids)
                        loss, ld = compute_phase_loss(
                            output, targets, self.phase,
                            vocab_size=self.cfg.model.vocab_size,
                            cfg=tcfg,
                            model=self.model,
                        )
                        loss = loss / accum_steps

                    self.scaler.scale(loss).backward()
                    step_loss += loss.item()
                    accum_loss += loss.item()
                    for k, v in ld.items():
                        if k not in loss_dict_accum:
                            loss_dict_accum[k] = 0.0
                        loss_dict_accum[k] += v.item() / accum_steps

                self.scaler.unscale_(self.optimizer)
                if self.use_velvet:
                    self.optimizer.clip_grad_norm_()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), tcfg.grad_clip,
                        foreach=True,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

                if self.use_velvet:
                    self.optimizer.set_loss_metrics(step_loss, self.cfg.model.vocab_size)

                self.global_step += 1

                if self.global_step % log_every == 0:
                    now = time.time()
                    dt = now - t0
                    window_dt = max(now - (self._tokens_t0 or now), 1e-6)
                    tok_per_s = self._tokens_window / window_dt

                    if self.use_velvet:
                        cur_skipped = self.optimizer.skipped_steps
                        skipped_window = cur_skipped - skipped_at_window_start
                        skipped_at_window_start = cur_skipped
                        # Mirror to Trainer-level counter for ckpt resume continuity.
                        self.skipped_steps = cur_skipped
                    else:
                        skipped_window = 0

                    n = log_every
                    avg_loss = accum_loss / n
                    avg_dict = {k: v / n for k, v in loss_dict_accum.items()}

                    if self.use_velvet:
                        lr_now = self.optimizer.effective_lr
                    else:
                        lr_now = self.optimizer.param_groups[0]['lr']

                    ce_val = avg_dict.get('ce', 0)
                    ppl_val = math.exp(min(ce_val, 20))

                    log_parts = [
                        f"step {self.global_step}/{max_steps}",
                        f"loss={avg_loss:.4f}",
                        f"ce={ce_val:.4f}",
                        f"ppl={ppl_val:.1f}",
                        f"lr={lr_now:.2e}",
                        f"{dt:.1f}s",
                        f"{tok_per_s:,.0f} tok/s",
                    ]
                    if skipped_window > 0:
                        log_parts.append(f"skip={skipped_window}")
                    if 'load_balance' in avg_dict:
                        log_parts.append(f"lb={avg_dict['load_balance']:.4f}")
                    if 'halting' in avg_dict:
                        log_parts.append(f"halt={avg_dict['halting']:.4f}")
                    if 'deep_supervision' in avg_dict:
                        log_parts.append(f"deep={avg_dict['deep_supervision']:.4f}")
                    if self.use_velvet:
                        log_parts.append(f"b1={self.optimizer.effective_beta1:.3f}")
                        log_parts.append(f"lvs={self.optimizer.lr_scale:.3f}")
                        log_parts.append(f"sig={self.optimizer.lvs_confidence:.2f}")
                        if self.optimizer.is_bursting:
                            log_parts.append("BURST")

                    print(" | ".join(log_parts))

                    csv_row = [self.global_step, f"{avg_loss:.6f}", f"{ce_val:.6f}",
                               f"{ppl_val:.2f}", f"{lr_now:.6e}", f"{dt:.2f}",
                               f"{tok_per_s:.2f}"]
                    if self.use_velvet:
                        csv_row += [
                            f"{self.optimizer.effective_beta1:.4f}",
                            f"{self.optimizer.lr_scale:.4f}",
                            f"{self.optimizer.lvs_confidence:.4f}",
                            f"{self.optimizer.perplexity_scale:.4f}",
                            f"{self.optimizer.last_grad_norm:.4f}",
                            f"{self.optimizer.lvs_phase:.4f}",
                            f"{skipped_window}",
                        ]
                    csv_writer.writerow(csv_row)
                    csv_file.flush()

                    if self._wandb_active:
                        import wandb
                        log_dict = {
                            **avg_dict,
                            'lr': lr_now,
                            'ppl': ppl_val,
                            'step': self.global_step,
                            'tokens_per_sec': tok_per_s,
                            'vram_gb': torch.cuda.memory_allocated() / (1024 ** 3),
                            'vram_peak_gb': torch.cuda.max_memory_allocated() / (1024 ** 3),
                        }
                        if self.use_velvet:
                            log_dict['effective_beta1'] = self.optimizer.effective_beta1
                            log_dict['lvs_scale'] = self.optimizer.lr_scale
                            log_dict['lvs_confidence'] = self.optimizer.lvs_confidence
                            log_dict['grad_norm'] = self.optimizer.last_grad_norm
                            log_dict['skipped_window'] = skipped_window
                        wandb.log(log_dict)

                    accum_loss = 0.0
                    loss_dict_accum = {}
                    t0 = time.time()
                    self._tokens_window = 0
                    self._tokens_t0 = t0

                if not tcfg.debug and self.global_step % tcfg.save_every == 0:
                    self._save_checkpoint()

            if not tcfg.debug:
                self._save_checkpoint(tag='final')
        finally:
            csv_file.close()

        print(f"\nTraining complete: {self.global_step} steps")
        print(f"Metrics saved: {csv_path}")

    def _save_checkpoint(self, tag=None):
        if tag:
            path = self.output_dir / f"ckpt_{tag}.pt"
        else:
            path = self.output_dir / f"ckpt_step{self.global_step}.pt"

        # Atomic save: write to .tmp then rename. Avoids leaving corrupted
        # checkpoints on disk if the process is killed mid-save (the previous
        # ckpt_step* file remains intact and resumable).
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        payload = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict(),
            'step': self.global_step,
            'phase': self.phase,
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        print(f"  Saved: {path}")

        self._cleanup_checkpoints()

    def _load_checkpoint(self, path):
        """Restore training state from a checkpoint produced by `_save_checkpoint`.

        Uses strict=False on the model to allow new modules (e.g. ACR/iterative)
        to be added between phases without failing the load.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"resume_from path not found: {path}")
        ckpt = _safe_torch_load(path, map_location=self.device)

        # Model: non-strict so phase transitions can add new submodules.
        missing, unexpected = self.model.load_state_dict(ckpt['model'], strict=False)
        if missing:
            print(f"  resume: {len(missing)} missing keys (new modules) — kept random init")
        if unexpected:
            print(f"  resume: {len(unexpected)} unexpected keys — ignored")

        if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except (ValueError, KeyError) as e:
                print(f"  resume: optimizer state could not be restored ({e}) — fresh state")

        # Scheduler is rebuilt inside train(); restore here is best-effort for inspection.
        self._resume_scheduler_state = ckpt.get('scheduler')

        if 'scaler' in ckpt and ckpt['scaler'] is not None:
            try:
                self.scaler.load_state_dict(ckpt['scaler'])
            except Exception:
                pass

        self.global_step = int(ckpt.get('step', 0))
        print(f"  Resumed from {path} at step {self.global_step}")

    def _cleanup_checkpoints(self):
        keep_n = self.tcfg.keep_last_n
        ckpts = sorted(
            self.output_dir.glob("ckpt_step*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        for old in ckpts[:-keep_n]:
            old.unlink()
