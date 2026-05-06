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
                entropy_adaptive=True,
                sparse_aware=True,
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

        if self.use_velvet:
            self.optimizer.set_training_steps(max_steps)
        csv_path = self.output_dir / "metrics.csv"
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_headers = ['step', 'loss', 'ce', 'ppl', 'lr', 'elapsed']
        if self.use_velvet:
            csv_headers += ['beta1', 'lvs_scale', 'signal', 'pgm_scale', 'grad_norm', 'lvs_phase']
        csv_writer.writerow(csv_headers)

        is_iterable = isinstance(self.train_dataset, torch.utils.data.IterableDataset)
        loader = DataLoader(
            self.train_dataset,
            batch_size=tcfg.batch_size if not is_iterable else None,
            shuffle=not is_iterable,
            num_workers=getattr(self.cfg.data, 'num_workers', 0),
            pin_memory=True,
            drop_last=not is_iterable,
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
                self.optimizer.clip_grad_norm_()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), tcfg.grad_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

            if self.use_velvet:
                self.optimizer.set_loss_metrics(step_loss, self.cfg.model.vocab_size)

            self.global_step += 1

            if self.global_step % log_every == 0:
                dt = time.time() - t0

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
                ]
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
                        log_dict = {**avg_dict, 'lr': lr_now, 'step': self.global_step}
                        if self.use_velvet:
                            log_dict['effective_beta1'] = self.optimizer.effective_beta1
                            log_dict['lvs_scale'] = self.optimizer.lr_scale
                            log_dict['lvs_confidence'] = self.optimizer.lvs_confidence
                            log_dict['grad_norm'] = self.optimizer.last_grad_norm
                        wandb.log(log_dict)
                    except Exception:
                        pass

                accum_loss = 0.0
                loss_dict_accum = {}
                t0 = time.time()

            if not tcfg.debug and self.global_step % tcfg.save_every == 0:
                self._save_checkpoint()

        if not tcfg.debug:
            self._save_checkpoint(tag='final')

        csv_file.close()
        print(f"\nTraining complete: {self.global_step} steps")
        print(f"Metrics saved: {csv_path}")

    def _save_checkpoint(self, tag=None):
        if tag:
            path = self.output_dir / f"ckpt_{tag}.pt"
        else:
            path = self.output_dir / f"ckpt_step{self.global_step}.pt"

        torch.save({
            'model': self.model.state_dict(),
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
