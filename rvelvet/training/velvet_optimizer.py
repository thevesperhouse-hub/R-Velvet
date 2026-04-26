"""Velvet Optimizer — AdamW + PGM + LVS v2.

Combines two adaptive mechanisms on top of AdamW:

  PGM (Perplexity-Guided Momentum):
    - Adapts beta1 based on perplexity: high ppl → more momentum, low ppl → less
    - Fast convergence: pushes through plateaus when model is struggling
    - Formula: eff_beta1 = clamp(beta1 * (40 / ppl), 0.5, 0.999)

  LVS v2 (Loss-Velocity Scaling):
    - Window-based trend detection (linear regression over 64 losses + R²)
    - Adapts LR: stalled → boost, diverging → dampen, noisy → stay neutral
    - Phase-adaptive range: early [0.7, 1.3], late [0.9, 1.1]

Three backends (auto-detected, priority order):
  1. Triton fused kernel — single kernel per param
  2. Native CUDA kernel (fallback) — compiled via cpp_extension
  3. Pure PyTorch (CPU or no Triton/CUDA)
"""

import math
import torch
from torch.optim import Optimizer

from .kernels import (
    velvet_update_kernel, HAS_TRITON, HAS_CUDA_EXT,
)


class VelvetOptimizer(Optimizer):
    """Velvet: AdamW with PGM (Perplexity-Guided Momentum) + LVS (Loss-Velocity Scaling).

    Args:
        params: model parameters
        lr: learning rate (default: 5e-4)
        betas: coefficients for moment estimation (default: (0.9, 0.999))
        eps: term for numerical stability (default: 1e-8)
        weight_decay: decoupled weight decay (default: 1e-3)
        max_grad_norm: global gradient clipping (0 = disabled, default: 1.0)
        entropy_adaptive: enable LVS for LR scaling (default: True)
        perplexity_guided: enable PGM for momentum scaling (default: True)
        sparse_aware: skip near-zero weights (default: True)
    """

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

        # LVS v2 state: window-based trend detection (for LR)
        self._loss_window = []
        self._lvs_window_size = 64
        self._lvs_min_window = 16
        self._lvs_phase_steps = 20000  # overridden by set_training_steps()
        self._lvs_confidence = 0.0
        self._entropy_scale = 1.0
        # Stall detection: EMA baseline for plateau damping
        self._loss_ema = None
        self._loss_ema_alpha = 0.02  # slow-moving baseline
        # PGM state: perplexity-guided momentum (for beta1)
        self._perplexity_scale = 1.0
        self._last_grad_norm = 0.0
        self._global_step = 0
        # Auto-detect best kernel backend
        if HAS_TRITON and torch.cuda.is_available():
            self._kernel_backend = "triton"
        elif HAS_CUDA_EXT and torch.cuda.is_available():
            self._kernel_backend = "cuda"
        else:
            self._kernel_backend = "pytorch"

    # ---- Adaptive signals (set by training loop) ----

    def set_training_steps(self, max_steps: int):
        """Set total training steps so LVS phase tracks actual run length."""
        self._lvs_phase_steps = max(max_steps, 1)

    def set_loss_metrics(self, loss_val: float, vocab_size: int):
        """Update both LVS (LR scaling) and PGM (momentum scaling).

        LVS: Window-based trend detection via linear regression + R².
        PGM: Perplexity-guided momentum — high ppl → more momentum.
        """
        if not math.isfinite(loss_val):
            return

        # ---- PGM v2: Loss-Normalized Momentum ----
        # Normalize loss relative to max entropy (= ln(vocab_size))
        max_entropy = math.log(max(vocab_size, 2))
        ratio = min(loss_val, max_entropy) / max_entropy  # 1.0 at random init, → 0

        # U-curve: neutral at extremes, reactive in the middle
        #   ratio > 0.6 (high loss, early training): neutral (1.0)
        #   ratio 0.2-0.6 (active learning): dip to 0.8 (less momentum, faster reaction)
        #   ratio < 0.2 (converging): rise to 1.2 (more momentum, stability)
        if ratio > 0.6:
            pgm_scale = 1.0
        elif ratio > 0.2:
            pgm_scale = 0.8 + 0.5 * (ratio - 0.2)  # 1.0→0.8
        else:
            pgm_scale = 0.8 + 2.0 * (0.2 - ratio)  # 0.8→1.2
        self._perplexity_scale = max(0.7, min(1.3, pgm_scale))

        # ---- LVS v2: Window-based trend detection ----
        # Ring buffer: append + trim
        self._loss_window.append(loss_val)
        if len(self._loss_window) > self._lvs_window_size:
            self._loss_window.pop(0)

        n = len(self._loss_window)
        if n < self._lvs_min_window:
            self._entropy_scale = 1.0
            self._lvs_confidence = 0.0
            return

        # Linear regression (OLS) over window: loss = a + b*t
        sum_t = n * (n - 1) / 2.0
        sum_t2 = n * (n - 1) * (2 * n - 1) / 6.0
        sum_y = 0.0
        sum_ty = 0.0
        for i, y in enumerate(self._loss_window):
            sum_y += y
            sum_ty += i * y

        denom = n * sum_t2 - sum_t * sum_t
        if denom < 1e-12:
            self._entropy_scale = 1.0
            self._lvs_confidence = 0.0
            return

        slope = (n * sum_ty - sum_t * sum_y) / denom
        intercept = (sum_y - slope * sum_t) / n

        # R² = 1 - SS_res / SS_tot
        mean_y = sum_y / n
        ss_tot = 0.0
        ss_res = 0.0
        for i, y in enumerate(self._loss_window):
            ss_tot += (y - mean_y) ** 2
            predicted = intercept + slope * i
            ss_res += (y - predicted) ** 2

        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))
        self._lvs_confidence = r_squared

        # Phase-adaptive range: early=wide [0.7, 1.3], late=narrow [0.9, 1.1]
        phase = min(1.0, self._global_step / max(self._lvs_phase_steps, 1))
        max_boost = 1.3 - 0.2 * phase   # 1.3 early -> 1.1 late
        max_damp = 0.7 + 0.2 * phase     # 0.7 early -> 0.9 late

        # Normalize slope relative to mean loss magnitude
        baseline = abs(mean_y) * 0.01 + 1e-8
        norm_slope = max(-1.0, min(1.0, slope / baseline))

        # Map slope to raw scale:
        #   slope < 0 (improving)  → between 1.0 and max_boost (mild boost)
        #   slope > 0 (diverging)  → between max_damp and 1.0
        #   slope ~ 0 (stalled)    → max_boost (boost to escape plateau)
        if norm_slope <= 0:
            raw_scale = 1.0 + (max_boost - 1.0) * (1.0 + norm_slope)
        else:
            raw_scale = max_boost - (max_boost - max_damp) * norm_slope

        # Blend with confidence: low R² → scale stays near 1.0
        self._entropy_scale = 1.0 + r_squared * (raw_scale - 1.0)

        # ---- Stall damping: when loss stops improving, dampen to settle ----
        # EMA tracks a slow-moving loss baseline
        if self._loss_ema is None:
            self._loss_ema = mean_y
        else:
            self._loss_ema += self._loss_ema_alpha * (mean_y - self._loss_ema)

        # If current window mean is above EMA (stalling/worsening), dampen
        if self._loss_ema > 1e-8:
            stall_gap = (mean_y - self._loss_ema) / self._loss_ema
            if stall_gap > 0.005:  # loss is >0.5% above EMA baseline
                # Progressive damping: harder stall → more damping, capped at max_damp
                damp = max(max_damp, 1.0 - min(stall_gap * 2, 0.15))
                self._entropy_scale = min(self._entropy_scale, damp)

    @property
    def effective_lr(self) -> float:
        lr = self.param_groups[0]["lr"]  # current scheduled LR (not initial)
        if self.defaults["entropy_adaptive"]:
            lr *= self._entropy_scale
        return lr

    @property
    def lr_scale(self) -> float:
        """Current LVS scale factor (1.0=neutral, >1.0=boosting, <1.0=dampening)."""
        return self._entropy_scale

    @property
    def effective_beta1(self) -> float:
        """Current momentum coefficient after PGM scaling."""
        beta1 = self.defaults["betas"][0]
        if self.defaults["perplexity_guided"]:
            beta1 = max(0.5, min(0.999, beta1 * self._perplexity_scale))
        return beta1

    @property
    def perplexity_scale(self) -> float:
        """PGM scale factor for beta1 (>1.0=more momentum, <1.0=less)."""
        return self._perplexity_scale

    @property
    def lvs_confidence(self) -> float:
        """R-squared of the loss trend regression (0=noise, 1=strong trend)."""
        return self._lvs_confidence

    @property
    def lvs_phase(self) -> float:
        """Training phase (0.0=early/reactive, 1.0=late/stable)."""
        return min(1.0, self._global_step / max(self._lvs_phase_steps, 1))

    @property
    def last_grad_norm(self) -> float:
        return self._last_grad_norm

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def kernel_backend(self) -> str:
        return self._kernel_backend

    # ---- Gradient clipping ----

    def clip_grad_norm_(self) -> float:
        """Global gradient clipping. Returns the global norm."""
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

    # ---- Step ----

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            base_lr = group["lr"]  # scheduled LR — NOT pre-scaled
            wd = group["weight_decay"]

            # PGM: adaptive momentum (computed here for PyTorch fallback)
            if group["perplexity_guided"]:
                eff_beta1 = max(0.5, min(0.999, beta1 * self._perplexity_scale))
            else:
                eff_beta1 = beta1

            # --- Per-tensor path (Triton, CUDA, or PyTorch fallback) ---
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

                # GPU kernel path — kernel handles LVS + PGM internally
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
                    # Pure PyTorch fallback — apply LVS + PGM here
                    p_f32 = p.data.float()
                    g_f32 = grad.float()

                    # Decoupled weight decay (base LR, not LVS-scaled)
                    p_f32.mul_(1.0 - base_lr * wd)

                    # Moments (with PGM-adjusted beta1)
                    state["m"].mul_(eff_beta1).add_(g_f32, alpha=1.0 - eff_beta1)
                    state["v"].mul_(beta2).addcmul_(g_f32, g_f32, value=1.0 - beta2)

                    # Bias-corrected
                    m_hat = state["m"] / bc1
                    v_hat = state["v"] / bc2

                    # LVS-scaled LR for update only
                    eff_lr = base_lr
                    if group["entropy_adaptive"]:
                        eff_lr *= self._entropy_scale

                    # Update
                    update = m_hat / (v_hat.sqrt() + eps)
                    p_f32.add_(update, alpha=-eff_lr)
                    p.data.copy_(p_f32.to(p.dtype))

        return loss
