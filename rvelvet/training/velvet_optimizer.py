"""Velvet Optimizer — AdamW + PGM + LVS v5.2.

Combines two adaptive mechanisms on top of AdamW:

  PGM (Perplexity-Guided Momentum):
    - Adapts beta1 based on normalized loss ratio (U-curve)
    - Active learning zone: less momentum (faster reaction)
    - Converging zone: more momentum (stability)

  LVS v5.2 (Loss-Velocity Scaling):
    - Two slow EMAs in LOG-SPACE: current vs anchor
    - Log-space: constant gap as long as relative improvement rate is constant
      (fixes signal decay where raw-loss EMAs converge as absolute loss shrinks)
    - Windows scaled to run length (Chinchilla-inspired)
    - Current < anchor = loss improved → boost LR (floor 50%)
    - Current > anchor = loss worsening → dampen LR
    - Phase-adaptive range: early [0.7, 1.3], late [0.9, 1.1]
    - Plateau burst: cyclical 2x LR spike to escape local minima

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

        # LVS v5 state: dual slow-EMA crossover (windows scaled to run length)
        self._ema_current = None
        self._ema_anchor = None
        # Default windows (overridden by set_training_steps)
        self._current_window = 100
        self._anchor_window_min = 200   # start of run
        self._anchor_window_max = 500   # end of run (grows linearly)
        self._ema_current_alpha = 2.0 / (100 + 1)
        self._ema_anchor_alpha = 2.0 / (200 + 1)
        self._lvs_phase_steps = 20000  # overridden by set_training_steps()
        self._lvs_confidence = 0.0  # signal strength for logging
        self._entropy_scale = 1.0
        self._lvs_momentum_up = 0.8    # fast ramp-up (responds in ~5 steps)
        self._lvs_momentum_down = 0.995  # slow decay (half-life ~140 steps)
        # Plateau burst: cyclical LR spike to escape local minima
        self._plateau_counter = 0       # steps with |gap| < threshold
        self._plateau_threshold = 0.005 # log-space gap below this = plateau
        self._plateau_patience = 200    # steps before triggering burst
        self._burst_duration = 50       # burst lasts this many steps
        self._burst_multiplier = 2.0    # LR multiplied by this during burst
        self._burst_step = -1           # step when burst started (-1 = inactive)
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
        """Set total training steps and compute adaptive EMA windows.

        Windows scale with run length (Chinchilla-inspired):
          - Current EMA: 3% of max_steps (min 50, max 300)
          - Anchor EMA: starts at 10% → grows to 30% over training
        Short runs (500 steps): current=50, anchor=50→150
        Long runs (100K steps): current=300, anchor=300→300 (capped)
        """
        self._lvs_phase_steps = max(max_steps, 1)

        # Adaptive window sizing
        self._current_window = max(50, min(300, int(max_steps * 0.03)))
        self._anchor_window_min = max(100, min(300, int(max_steps * 0.10)))
        self._anchor_window_max = max(200, min(500, int(max_steps * 0.30)))

        # Ensure anchor_min <= anchor_max
        if self._anchor_window_min > self._anchor_window_max:
            self._anchor_window_min = self._anchor_window_max

        # Set initial alphas
        self._ema_current_alpha = 2.0 / (self._current_window + 1)
        self._ema_anchor_alpha = 2.0 / (self._anchor_window_min + 1)

    def set_loss_metrics(self, loss_val: float, vocab_size: int):
        """Update both LVS (LR scaling) and PGM (momentum scaling).

        LVS v5.2: EMAs computed on log(loss) instead of raw loss.
        In log-space, exponential decay → linear decrease → constant EMA gap.
        This prevents sig from decaying just because absolute loss values shrink.

        PGM: Perplexity-guided momentum — U-curve based on loss ratio.
        """
        if not math.isfinite(loss_val):
            return

        # ---- PGM v2: Loss-Normalized Momentum ----
        max_entropy = math.log(max(vocab_size, 2))
        ratio = min(loss_val, max_entropy) / max_entropy

        if ratio > 0.6:
            pgm_scale = 1.0
        elif ratio > 0.2:
            pgm_scale = 0.8 + 0.5 * (ratio - 0.2)
        else:
            pgm_scale = 0.8 + 2.0 * (0.2 - ratio)
        self._perplexity_scale = max(0.7, min(1.3, pgm_scale))

        # ---- LVS v5.2: Dual slow-EMA crossover in LOG-SPACE ----
        # Log-space EMAs: constant gap as long as relative improvement is constant
        # Raw loss 9.5→9.0 (-5.3%) and 6.5→6.3 (-3.1%) look different in linear space
        # but in log space both produce proportional gaps → sig stays stable
        log_loss = math.log(max(loss_val, 1e-8))

        # Initialize EMAs on first call
        if self._ema_current is None:
            self._ema_current = log_loss
            self._ema_anchor = log_loss
            self._entropy_scale = 1.0
            self._lvs_confidence = 0.0
            return

        # Adaptive anchor window: grows linearly from min → max over training
        progress = min(1.0, self._global_step / max(self._lvs_phase_steps, 1))
        anchor_window = self._anchor_window_min + (self._anchor_window_max - self._anchor_window_min) * progress
        self._ema_anchor_alpha = 2.0 / (anchor_window + 1)

        # Update both EMAs in log-space
        self._ema_current += self._ema_current_alpha * (log_loss - self._ema_current)
        self._ema_anchor += self._ema_anchor_alpha * (log_loss - self._ema_anchor)

        # Gap: difference in log-space (= log ratio of smoothed losses)
        # Negative = current < anchor = loss has improved → boost
        # Positive = current > anchor = loss worsening → dampen
        gap = self._ema_current - self._ema_anchor
        gap = max(-1.0, min(1.0, gap))

        # Signal strength for logging (absolute gap in log-space, scaled)
        # |gap|=0.01 → 1% relative improvement → sig=0.10
        # |gap|=0.05 → 5% relative improvement → sig=0.50
        # |gap|=0.10+ → 10%+ relative improvement → sig=1.00
        self._lvs_confidence = min(1.0, abs(gap) * 10.0)

        # Phase-adaptive range: early=wide [0.7, 1.3], late=narrow [0.9, 1.1]
        phase = min(1.0, self._global_step / max(self._lvs_phase_steps, 1))
        max_boost = 1.3 - 0.2 * phase
        max_damp = 0.7 + 0.2 * phase

        # Map gap to scale (thresholds calibrated for log-space):
        #   gap < -0.005 (improving ≥0.5%)  → boost, floor at 50%
        #   gap > +0.005 (worsening ≥0.5%)  → dampen proportionally
        #   |gap| < 0.005 (stable)           → neutral
        if gap < -0.005:
            # Improving: proportional + floor guarantee
            # 0.5% → strength 0.5 (floor), 5%+ → strength 1.0 (max)
            strength = min(1.0, abs(gap) / 0.05)
            strength = max(0.5, strength)  # FLOOR: at least 50% of max boost
            raw_scale = 1.0 + (max_boost - 1.0) * strength
        elif gap > 0.005:
            # Worsening: proportional damping
            strength = min(1.0, gap / 0.05)
            raw_scale = 1.0 - (1.0 - max_damp) * strength
        else:
            raw_scale = 1.0

        # Asymmetric momentum: fast up, slow down (holds boost longer)
        if raw_scale >= self._entropy_scale:
            momentum = self._lvs_momentum_up    # rising: react fast
        else:
            momentum = self._lvs_momentum_down  # falling: hold the boost
        self._entropy_scale = momentum * self._entropy_scale + (1.0 - momentum) * raw_scale

        # ---- Plateau burst: cyclical LR spike to escape local minima ----
        # Detect plateau: gap near zero for too long
        if abs(gap) < self._plateau_threshold:
            self._plateau_counter += 1
        else:
            self._plateau_counter = max(0, self._plateau_counter - 5)  # fast reset

        # Check if we're currently in a burst
        if self._burst_step >= 0:
            # Active burst: cosine-shaped spike (ramp up, peak, ramp down)
            burst_progress = (self._global_step - self._burst_step) / self._burst_duration
            if burst_progress < 1.0:
                # Cosine spike: 1.0 → burst_multiplier → 1.0
                spike = 0.5 * (1.0 - math.cos(2.0 * math.pi * burst_progress))
                burst_scale = 1.0 + (self._burst_multiplier - 1.0) * spike
                self._entropy_scale *= burst_scale
            else:
                # Burst finished, reset
                self._burst_step = -1
                self._plateau_counter = 0
        elif self._plateau_counter >= self._plateau_patience and self._global_step > 500:
            # Trigger new burst (only after warmup)
            self._burst_step = self._global_step
            self._plateau_counter = 0

        # Clamp (allow burst to exceed normal max_boost temporarily)
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
        """Signal strength of the EMA crossover (0=flat, 1=strong trend)."""
        return self._lvs_confidence

    @property
    def lvs_phase(self) -> float:
        """Training phase (0.0=early/reactive, 1.0=late/stable)."""
        return min(1.0, self._global_step / max(self._lvs_phase_steps, 1))

    @property
    def is_bursting(self) -> bool:
        """True if a plateau burst is currently active."""
        return self._burst_step >= 0

    @property
    def plateau_counter(self) -> int:
        """Steps spent on current plateau (triggers burst at patience)."""
        return self._plateau_counter

    @property
    def anchor_window(self) -> float:
        """Current anchor EMA window size (grows over training)."""
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
