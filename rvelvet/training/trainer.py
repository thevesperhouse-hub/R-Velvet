"""Shared trainer for all three training phases with per-phase parameter groups,
mixed precision, gradient accumulation, LR scheduling, checkpointing, and logging."""

import os
import csv
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from .losses import compute_phase_loss
from .velvet_optimizer import VelvetOptimizer


class Trainer:
    """Multi-phase trainer for R-Velvet. Phase differences handled via parameter groups,
    loss computation, and YAML configs."""

    def __init__(self, model, train_dataset, cfg):
        self.model = model
        self.train_dataset = train_dataset
        self.cfg = cfg
        self.tcfg = cfg.training  # training sub-config
        self.phase = self.tcfg.phase

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        if getattr(self.tcfg, 'compile', False) and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        self.amp_dtype = self._resolve_amp_dtype()
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.amp_dtype == torch.float16))

        if self.phase == 'phase3_iterative':
            self._freeze_for_phase3()

        self.optimizer = self._build_optimizer()
        self.scheduler = None
        self.global_step = 0
        self.best_loss = float('inf')

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def resume_from_checkpoint(self, ckpt: dict):
        """Restore optimizer, scheduler, and global_step from a checkpoint dict."""
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            print(f"  Restored optimizer state")
        if 'step' in ckpt:
            self.global_step = ckpt['step']
            print(f"  Restored global_step = {self.global_step}")

    def _resolve_amp_dtype(self):
        amp_setting = getattr(self.tcfg, 'amp', 'bf16')
        if amp_setting == 'bf16' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif amp_setting in ('fp16', 'bf16'):
            return torch.float16
        return torch.float32

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
            optimizer = VelvetOptimizer(
                param_groups,
                betas=(tcfg.beta1, tcfg.beta2),
                eps=getattr(tcfg, 'eps', 1e-8),
                weight_decay=tcfg.weight_decay,
                max_grad_norm=tcfg.grad_clip,
                entropy_adaptive=getattr(tcfg, 'velvet_lvs', False),
                perplexity_guided=getattr(tcfg, 'velvet_pgm', False),
                sparse_aware=getattr(tcfg, 'velvet_sparse', False),
                force_backend=getattr(tcfg, 'velvet_backend', None),
            )
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

        # If resuming, fast-forward scheduler to current step
        if self.global_step > 0:
            for _ in range(self.global_step):
                self.scheduler.step()
            print(f"  Scheduler fast-forwarded to step {self.global_step}")

        if self.use_velvet:
            self.optimizer.set_training_steps(max_steps)
        csv_path = self.output_dir / "metrics.csv"
        resuming = self.global_step > 0
        csv_file = open(csv_path, 'a' if resuming else 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_headers = ['step', 'loss', 'ce', 'ppl', 'lr', 'elapsed']
        if self.use_velvet:
            csv_headers += ['beta1', 'lvs_scale', 'signal', 'pgm_scale', 'grad_norm', 'lvs_phase']
        if not resuming:
            csv_writer.writerow(csv_headers)

        is_iterable = isinstance(self.train_dataset, torch.utils.data.IterableDataset)
        loader = DataLoader(
            self.train_dataset,
            batch_size=tcfg.batch_size,
            shuffle=not is_iterable,
            num_workers=getattr(self.cfg.data, 'num_workers', 0),
            pin_memory=True,
            drop_last=True,
        )

        if tcfg.wandb and not tcfg.debug:
            try:
                import wandb
                wandb.init(project=tcfg.wandb_project, name=tcfg.wandb_run)
            except Exception:
                pass

        self.model.train()
        data_iter = iter(loader)
        self.optimizer.zero_grad()

        accum_loss = 0.0
        loss_dict_accum = {}
        t0 = time.time()

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        optim_name = 'VelvetOptimizer' if self.use_velvet else 'AdamW'
        print(f"Phase: {self.phase}")
        print(f"Trainable params: {n_trainable:,}")
        print(f"Device: {self.device} | AMP: {self.amp_dtype}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            from .kernels import GPU_ARCH, GPU_SM, get_backend
            print(f"GPU: {gpu_name} ({gpu_mem:.0f}GB) | Arch: {GPU_ARCH} (SM{GPU_SM}) | Kernels: {get_backend()}")
        print(f"Optimizer: {optim_name}", end="")
        if self.use_velvet:
            print(f" (backend={self.optimizer.kernel_backend}, PGM+LVS)")
            print(f"  EMA windows: current={self.optimizer._current_window}, "
                  f"anchor={self.optimizer._anchor_window_min}→{self.optimizer._anchor_window_max}")
        else:
            print()
        eff_batch = tcfg.batch_size * accum_steps
        seq_len = self.cfg.data.seq_len
        tokens_per_step = eff_batch * seq_len
        print(f"Effective batch: {eff_batch}")
        print(f"Tokens/step: {tokens_per_step:,}")
        print(f"Max steps: {max_steps} (~{max_steps * tokens_per_step / 1e9:.1f}B tokens)")
        print("-" * 60)

        pbar = tqdm(total=max_steps, unit="step", desc="Training",
                    initial=self.global_step)

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
                grad_norm = self.optimizer.clip_grad_norm_()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), tcfg.grad_clip,
                ).item()

            # Stability guard: skip step if loss or grads are bad
            skip_step = False
            if not math.isfinite(step_loss):
                tqdm.write(f"[step {self.global_step+1}] WARNING: non-finite loss ({step_loss}), skipping step")
                skip_step = True
            elif grad_norm > 100:
                tqdm.write(f"[step {self.global_step+1}] WARNING: grad_norm={grad_norm:.2e}, skipping step")
                skip_step = True
            elif hasattr(self, '_prev_loss') and self._prev_loss is not None:
                if step_loss > self._prev_loss * 1.5 and self.global_step > 100:
                    tqdm.write(f"[step {self.global_step+1}] WARNING: loss spike {self._prev_loss:.2f} -> {step_loss:.2f}, skipping step")
                    skip_step = True

            if skip_step:
                self.optimizer.zero_grad()
                self._skip_count = getattr(self, '_skip_count', 0) + 1
                # If too many skips in a row, rollback to last checkpoint
                if self._skip_count >= 10:
                    tqdm.write(f"[step {self.global_step+1}] 10 consecutive skips — rolling back to last checkpoint")
                    self._rollback_checkpoint()
                    self._skip_count = 0
                continue

            self._skip_count = 0
            self._prev_loss = step_loss

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

            if self.use_velvet:
                self.optimizer.set_loss_metrics(step_loss, self.cfg.model.vocab_size)

            self.global_step += 1
            pbar.update(1)

            # Update tqdm postfix every step
            if self.use_velvet:
                lr_now = self.optimizer.effective_lr
            else:
                lr_now = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix(
                loss=f"{step_loss:.3f}",
                ppl=f"{math.exp(min(step_loss, 20)):.0f}",
                lr=f"{lr_now:.1e}",
            )

            if self.global_step % log_every == 0:
                dt = time.time() - t0

                n = log_every
                avg_loss = accum_loss / n
                avg_dict = {k: v / n for k, v in loss_dict_accum.items()}

                ce_val = avg_dict.get('ce', 0)
                ppl_val = math.exp(min(ce_val, 20))
                tok_per_sec = n * tokens_per_step / max(dt, 0.01)

                log_parts = [
                    f"loss={avg_loss:.4f}",
                    f"ppl={ppl_val:.1f}",
                    f"lr={lr_now:.2e}",
                    f"{tok_per_sec/1000:.0f}k tok/s",
                ]
                if 'load_balance' in avg_dict:
                    log_parts.append(f"lb={avg_dict['load_balance']:.4f}")
                if 'halting' in avg_dict:
                    log_parts.append(f"halt={avg_dict['halting']:.4f}")
                if 'deep_supervision' in avg_dict:
                    log_parts.append(f"deep={avg_dict['deep_supervision']:.4f}")
                if 'z_loss' in avg_dict:
                    log_parts.append(f"z={avg_dict['z_loss']:.4f}")
                # Log expansion gate value
                gate_val = None
                _m = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
                if hasattr(_m, 'expansion') and hasattr(_m.expansion, 'cross_gate'):
                    gate_val = torch.sigmoid(_m.expansion.cross_gate).item()
                    log_parts.append(f"gate={gate_val:.3f}")
                if self.use_velvet:
                    log_parts.append(f"b1={self.optimizer.effective_beta1:.3f}")
                    log_parts.append(f"lvs={self.optimizer.lr_scale:.3f}")
                    if self.optimizer.is_bursting:
                        log_parts.append("BURST")

                tqdm.write(f"[step {self.global_step}] " + " | ".join(log_parts))

                csv_row = [self.global_step, f"{avg_loss:.6f}", f"{ce_val:.6f}",
                           f"{ppl_val:.2f}", f"{lr_now:.6e}", f"{dt:.2f}"]
                if self.use_velvet:
                    csv_row += [
                        f"{self.optimizer.effective_beta1:.4f}",
                        f"{self.optimizer.lr_scale:.4f}",
                        f"{self.optimizer.lvs_confidence:.4f}",
                        f"{self.optimizer.perplexity_scale:.4f}",
                        f"{self.optimizer.last_grad_norm:.4f}",
                        f"{self.optimizer.lvs_phase:.4f}",
                    ]
                csv_writer.writerow(csv_row)
                csv_file.flush()

                if tcfg.wandb and not tcfg.debug:
                    try:
                        import wandb
                        log_dict = {**avg_dict, 'lr': lr_now, 'step': self.global_step,
                                    'tokens_per_sec': tok_per_sec}
                        if self.use_velvet:
                            log_dict['effective_beta1'] = self.optimizer.effective_beta1
                            log_dict['lvs_scale'] = self.optimizer.lr_scale
                            log_dict['lvs_confidence'] = self.optimizer.lvs_confidence
                            log_dict['grad_norm'] = self.optimizer.last_grad_norm
                        if gate_val is not None:
                            log_dict['expansion_gate'] = gate_val
                        wandb.log(log_dict)
                    except Exception:
                        pass

                accum_loss = 0.0
                loss_dict_accum = {}
                t0 = time.time()

            if not tcfg.debug and self.global_step % tcfg.save_every == 0:
                self._save_checkpoint()

        pbar.close()

        if not tcfg.debug:
            self._save_checkpoint(tag='final')

        csv_file.close()
        print(f"\nTraining complete: {self.global_step} steps ({self.global_step * tokens_per_step / 1e9:.1f}B tokens)")
        print(f"Metrics saved: {csv_path}")

    def _save_checkpoint(self, tag=None):
        if tag:
            path = self.output_dir / f"ckpt_{tag}.pt"
        else:
            path = self.output_dir / f"ckpt_step{self.global_step}.pt"

        # Save model without _orig_mod. prefix from torch.compile
        model_state = self.model.state_dict()
        if any(k.startswith('_orig_mod.') for k in model_state):
            model_state = {k.replace('_orig_mod.', '', 1): v for k, v in model_state.items()}

        torch.save({
            'model': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'step': self.global_step,
            'phase': self.phase,
        }, path)
        print(f"  Saved: {path}")

        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        keep_n = self.tcfg.keep_last_n
        ckpts = sorted(
            self.output_dir.glob("ckpt_step*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        for old in ckpts[:-keep_n]:
            old.unlink()

    def _rollback_checkpoint(self):
        """Load the most recent checkpoint after repeated unstable steps."""
        ckpts = sorted(
            self.output_dir.glob("ckpt_step*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if not ckpts:
            tqdm.write("  No checkpoint found for rollback, continuing anyway")
            return

        path = ckpts[-1]
        tqdm.write(f"  Rolling back to {path.name}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Unwrap compiled model if needed
        model = self.model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        model.load_state_dict(ckpt['model'])

        self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.scheduler and ckpt.get('scheduler'):
            self.scheduler.load_state_dict(ckpt['scheduler'])
        self.global_step = ckpt['step']
        self._prev_loss = None
        tqdm.write(f"  Rolled back to step {self.global_step}")
