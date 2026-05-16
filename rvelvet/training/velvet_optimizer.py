"""Velvet Optimizer: AdamW with PGM (Perplexity-Guided Momentum) and LVS v5.2 (Loss-Velocity Scaling).

PGM adapts beta1 using a U-curve based on normalized loss ratio. Lower momentum during
active learning enables faster reactions; higher momentum during convergence provides stability.

LVS v5.2 uses dual slow EMAs in log-space (current vs anchor) to scale learning rate.
Log-space computation maintains constant gap under constant relative improvement rates,
preventing signal decay as absolute loss shrinks. Windows scale with run length following
Chinchilla principles. Phase-adaptive ranges shift from wide [0.7, 1.3] early to narrow
[0.9, 1.1] late. Plateau detection triggers cyclical LR bursts to escape local minima.

Three backends auto-detected in priority order: Triton fused kernel, native CUDA kernel,
or pure PyTorch fallback.
"""

import math
import torch
from torch.optim import Optimizer

from .kernels import (
    velvet_update_kernel, HAS_TRITON, HAS_CUDA_EXT,
)


class VelvetOptimizer(Optimizer):
    """AdamW with PGM and LVS adaptive mechanisms."""

    def __init__(
        self,
        params,
        lr: float = 5e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-3,
        max_grad_norm: float = 1.0,
        entropy_adaptive: bool = True,
        perplexity_guided: bool = True,
        sparse_aware: bool = True,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            entropy_adaptive=entropy_adaptive,
            perplexity_guided=perplexity_guided,
            sparse_aware=sparse_aware,
        )
        super().__init__(params, defaults)

        self._ema_current = None
        self._ema_anchor = None
        self._current_window = 100
        self._anchor_window_min = 200
        self._anchor_window_max = 500
        self._ema_current_alpha = 2.0 / (100 + 1)
        self._ema_anchor_alpha = 2.0 / (200 + 1)
        self._lvs_phase_steps = 20000
        self._lvs_confidence = 0.0
        self._entropy_scale = 1.0
        self._lvs_momentum_up = 0.995
        self._lvs_momentum_down = 0.9
        self._plateau_counter = 0
        self._plateau_threshold = 0.005
        self._plateau_patience = 200
        self._burst_duration = 50
        self._burst_multiplier = 1.3
        self._burst_step = -1
        self._perplexity_scale = 1.0
        self._last_grad_norm = 0.0
        self._global_step = 0
        if HAS_TRITON and torch.cuda.is_available():
            self._kernel_backend = "triton"
        elif HAS_CUDA_EXT and torch.cuda.is_available():
            self._kernel_backend = "cuda"
        else:
            self._kernel_backend = "pytorch"

    def set_training_steps(self, max_steps: int):
        """Compute adaptive EMA windows scaled to run length. Current window is 2% of max_steps
        (capped 50-150). Anchor window is always ≥ 3x current and scales with max_steps."""
        self._lvs_phase_steps = max(max_steps, 1)
        self._current_window = max(50, min(150, int(max_steps * 0.02)))
        self._anchor_window_min = max(self._current_window * 3, min(2000, int(max_steps * 0.10)))
        self._anchor_window_max = max(self._anchor_window_min, min(5000, int(max_steps * 0.25)))
        self._ema_current_alpha = 2.0 / (self._current_window + 1)
        self._ema_anchor_alpha = 2.0 / (self._anchor_window_min + 1)

    def set_loss_metrics(self, loss_val: float, vocab_size: int):
        """Update LVS (LR scaling) and PGM (momentum scaling). LVS v5.2 computes EMAs on log(loss)
        to maintain constant gap under constant relative improvement, preventing signal decay as
        absolute loss shrinks. PGM uses U-curve based on normalized loss ratio."""
        if not math.isfinite(loss_val):
            return
        max_entropy = math.log(max(vocab_size, 2))
        ratio = min(loss_val, max_entropy) / max_entropy

        if ratio > 0.6:
            pgm_scale = 1.0
        elif ratio > 0.2:
            pgm_scale = 0.8 + 0.5 * (ratio - 0.2)
        else:
            pgm_scale = 0.8 + 2.0 * (0.2 - ratio)
        target = max(0.7, min(1.3, pgm_scale))
        # Dampen PGM: max ±0.02 change per step to prevent momentum whiplash
        delta = target - self._perplexity_scale
        delta = max(-0.02, min(0.02, delta))
        self._perplexity_scale += delta

        log_loss = math.log(max(loss_val, 1e-8))
        if self._ema_current is None:
            self._ema_current = log_loss
            self._ema_anchor = log_loss
            self._entropy_scale = 1.0
            self._lvs_confidence = 0.0
            return

        progress = min(1.0, self._global_step / max(self._lvs_phase_steps, 1))
        anchor_window = self._anchor_window_min + (self._anchor_window_max - self._anchor_window_min) * progress
        self._ema_anchor_alpha = 2.0 / (anchor_window + 1)

        self._ema_current += self._ema_current_alpha * (log_loss - self._ema_current)
        self._ema_anchor += self._ema_anchor_alpha * (log_loss - self._ema_anchor)

        gap = self._ema_current - self._ema_anchor
        gap = max(-1.0, min(1.0, gap))
        self._lvs_confidence = min(1.0, abs(gap) * 10.0)
        phase = min(1.0, self._global_step / max(self._lvs_phase_steps, 1))
        max_boost = 1.3 - 0.2 * phase
        max_damp = 0.7 + 0.2 * phase
        if gap < -0.005:
            raw_scale = max_boost
        elif gap > 0.005:
            strength = min(1.0, gap / 0.05)
            raw_scale = 1.0 - (1.0 - max_damp) * strength
        else:
            raw_scale = 1.0

        if raw_scale >= self._entropy_scale:
            momentum = self._lvs_momentum_up
        else:
            momentum = self._lvs_momentum_down
        self._entropy_scale = momentum * self._entropy_scale + (1.0 - momentum) * raw_scale
        if abs(gap) < self._plateau_threshold:
            self._plateau_counter += 1
        else:
            self._plateau_counter = max(0, self._plateau_counter - 5)

        if self._burst_step >= 0:
            burst_progress = (self._global_step - self._burst_step) / self._burst_duration
            if burst_progress < 1.0:
                spike = 0.5 * (1.0 - math.cos(2.0 * math.pi * burst_progress))
                burst_scale = 1.0 + (self._burst_multiplier - 1.0) * spike
                self._entropy_scale *= burst_scale
            else:
                self._burst_step = -1
                self._plateau_counter = 0
        elif self._plateau_counter >= self._plateau_patience and self._global_step > 500:
            self._burst_step = self._global_step
            self._plateau_counter = 0
        burst_max = max_boost * self._burst_multiplier if self._burst_step >= 0 else max_boost
        self._entropy_scale = max(max_damp, min(burst_max, self._entropy_scale))

    @property
    def effective_lr(self) -> float:
        lr = self.param_groups[0]["lr"]  # current scheduled LR (not initial)
        if self.defaults["entropy_adaptive"]:
            lr *= self._entropy_scale
        return lr

    @property
    def lr_scale(self) -> float:
        return self._entropy_scale

    @property
    def effective_beta1(self) -> float:
        beta1 = self.defaults["betas"][0]
        if self.defaults["perplexity_guided"]:
            beta1 = max(0.5, min(0.999, beta1 * self._perplexity_scale))
        return beta1

    @property
    def perplexity_scale(self) -> float:
        return self._perplexity_scale

    @property
    def lvs_confidence(self) -> float:
        return self._lvs_confidence

    @property
    def lvs_phase(self) -> float:
        return min(1.0, self._global_step / max(self._lvs_phase_steps, 1))

    @property
    def is_bursting(self) -> bool:
        return self._burst_step >= 0

    @property
    def plateau_counter(self) -> int:
        return self._plateau_counter

    @property
    def anchor_window(self) -> float:
        progress = min(1.0, self._global_step / max(self._lvs_phase_steps, 1))
        return self._anchor_window_min + (self._anchor_window_max - self._anchor_window_min) * progress

    @property
    def last_grad_norm(self) -> float:
        return self._last_grad_norm

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def kernel_backend(self) -> str:
        return self._kernel_backend

    def clip_grad_norm_(self) -> float:
        max_norm = self.defaults["max_grad_norm"]
        if max_norm <= 0:
            return 0.0

        all_params = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    all_params.append(p)

        if not all_params:
            return 0.0

        total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm)
        self._last_grad_norm = total_norm.item()
        return self._last_grad_norm

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            base_lr = group["lr"]
            wd = group["weight_decay"]

            if group["perplexity_guided"]:
                eff_beta1 = max(0.5, min(0.999, beta1 * self._perplexity_scale))
            else:
                eff_beta1 = beta1
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Velvet does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1

                bc1 = 1.0 - beta1 ** state["step"]
                bc2 = 1.0 - beta2 ** state["step"]

                if self._kernel_backend in ("triton", "cuda") and p.is_cuda and velvet_update_kernel is not None:
                    velvet_update_kernel(
                        param=p.data,
                        grad=grad,
                        m=state["m"],
                        v=state["v"],
                        lr=base_lr,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        wd=wd,
                        bias_correction1=bc1,
                        bias_correction2=bc2,
                        entropy_adaptive=group["entropy_adaptive"],
                        entropy_lr_scale=self._entropy_scale,
                        perplexity_guided=group["perplexity_guided"],
                        ppl_momentum_scale=self._perplexity_scale,
                        sparse_aware=group["sparse_aware"],
                    )
                else:
                    p_f32 = p.data.float()
                    g_f32 = grad.float()
                    p_f32.mul_(1.0 - base_lr * wd)
                    state["m"].mul_(eff_beta1).add_(g_f32, alpha=1.0 - eff_beta1)
                    state["v"].mul_(beta2).addcmul_(g_f32, g_f32, value=1.0 - beta2)
                    m_hat = state["m"] / bc1
                    v_hat = state["v"] / bc2
                    eff_lr = base_lr
                    if group["entropy_adaptive"]:
                        eff_lr *= self._entropy_scale
                    update = m_hat / (v_hat.sqrt() + eps)
                    p_f32.add_(update, alpha=-eff_lr)
                    p.data.copy_(p_f32.to(p.dtype))

        return loss
