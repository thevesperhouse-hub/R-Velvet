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
    """AdamW with PGM (Perplexity-Guided Momentum) and LVS (Loss-Velocity Scaling).

    All adaptive thresholds and bounds are exposed as constructor args so the
    optimizer can be tuned per-task without source edits.
    """

    # Bumped on incompatible state_dict layout changes.
    STATE_DICT_VERSION = 1

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
        # --- LVS configurables ---
        lvs_min_scale: float = 0.7,
        lvs_max_scale: float = 1.3,
        lvs_gap_clamp: float = 1.0,
        lvs_gap_dead_zone: float = 0.005,
        lvs_gap_strength_full: float = 0.05,
        lvs_momentum_up: float = 0.8,
        lvs_momentum_down: float = 0.995,
        lvs_phase_decay: float = 0.2,    # max_boost shrinks by this over phase
        # --- Plateau / burst ---
        plateau_threshold: float = 0.005,
        plateau_patience: int = 200,
        burst_duration: int = 50,
        burst_multiplier: float = 2.0,
        burst_warmup_steps: int = 500,
        # --- PGM configurables ---
        pgm_min_scale: float = 0.7,
        pgm_max_scale: float = 1.3,
        # --- Sparsity ---
        sparse_threshold: float = 1e-9,
        # --- Robustness ---
        skip_nonfinite: bool = True,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            entropy_adaptive=entropy_adaptive,
            perplexity_guided=perplexity_guided,
            sparse_aware=sparse_aware,
            sparse_threshold=sparse_threshold,
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
        # Configurable LVS knobs
        self._lvs_min_scale = lvs_min_scale
        self._lvs_max_scale = lvs_max_scale
        self._lvs_gap_clamp = lvs_gap_clamp
        self._lvs_gap_dead_zone = lvs_gap_dead_zone
        self._lvs_gap_strength_full = lvs_gap_strength_full
        self._lvs_momentum_up = lvs_momentum_up
        self._lvs_momentum_down = lvs_momentum_down
        self._lvs_phase_decay = lvs_phase_decay
        # Plateau / burst
        self._plateau_counter = 0
        self._plateau_threshold = plateau_threshold
        self._plateau_patience = plateau_patience
        self._burst_duration = burst_duration
        self._burst_multiplier = burst_multiplier
        self._burst_warmup_steps = burst_warmup_steps
        self._burst_step = -1
        # PGM
        self._pgm_min_scale = pgm_min_scale
        self._pgm_max_scale = pgm_max_scale
        self._perplexity_scale = 1.0
        # Robustness / counters
        self._skip_nonfinite = skip_nonfinite
        self._last_grad_norm = 0.0
        self._global_step = 0
        self._skipped_steps = 0
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
        """Update LVS (LR scaling) and PGM (momentum scaling).

        LVS computes EMAs on ``log(loss)`` so the gap stays meaningful as the
        absolute loss shrinks (constant relative improvement → constant gap).
        PGM uses a U-curve over the normalised loss ratio so momentum is
        relaxed during fast learning and re-tightened during convergence.

        Non-finite loss values are silently ignored — the EMAs are protected
        because a single inf/NaN would corrupt the running average forever.
        """
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
        self._perplexity_scale = max(self._pgm_min_scale, min(self._pgm_max_scale, pgm_scale))

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
        clamp = self._lvs_gap_clamp
        gap = max(-clamp, min(clamp, gap))
        self._lvs_confidence = min(1.0, abs(gap) * 10.0)
        phase = progress
        max_boost = self._lvs_max_scale - self._lvs_phase_decay * phase
        max_damp = self._lvs_min_scale + self._lvs_phase_decay * phase
        dead = self._lvs_gap_dead_zone
        if gap < -dead:
            raw_scale = max_boost
        elif gap > dead:
            strength = min(1.0, gap / self._lvs_gap_strength_full)
            raw_scale = 1.0 - (1.0 - max_damp) * strength
        else:
            raw_scale = 1.0

        momentum = self._lvs_momentum_up if raw_scale >= self._entropy_scale else self._lvs_momentum_down
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
        elif (self._plateau_counter >= self._plateau_patience
              and self._global_step > self._burst_warmup_steps):
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

    @property
    def skipped_steps(self) -> int:
        """How many step() calls have been skipped because of non-finite grads."""
        return self._skipped_steps

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

        # foreach=True dispatches a single fused multi-tensor kernel — much faster
        # than per-parameter calls when there are hundreds of params.
        total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm, foreach=True)
        self._last_grad_norm = total_norm.item()
        return self._last_grad_norm

    # Keys of internal LVS/PGM state that must be persisted across save/load.
    # The base Optimizer.state_dict() only handles per-param `state` and `param_groups`;
    # without this override, all adaptive scalars below would reset on resume.
    _VELVET_STATE_KEYS = (
        "_ema_current", "_ema_anchor",
        "_current_window", "_anchor_window_min", "_anchor_window_max",
        "_ema_current_alpha", "_ema_anchor_alpha",
        "_lvs_phase_steps", "_lvs_confidence",
        "_entropy_scale",
        "_lvs_min_scale", "_lvs_max_scale", "_lvs_gap_clamp",
        "_lvs_gap_dead_zone", "_lvs_gap_strength_full",
        "_lvs_momentum_up", "_lvs_momentum_down", "_lvs_phase_decay",
        "_plateau_counter", "_plateau_threshold", "_plateau_patience",
        "_burst_duration", "_burst_multiplier", "_burst_warmup_steps", "_burst_step",
        "_pgm_min_scale", "_pgm_max_scale", "_perplexity_scale",
        "_last_grad_norm", "_global_step", "_skipped_steps",
        "_skip_nonfinite",
    )

    def state_dict(self):
        sd = super().state_dict()
        sd["velvet_state"] = {k: getattr(self, k) for k in self._VELVET_STATE_KEYS}
        sd["velvet_version"] = self.STATE_DICT_VERSION
        return sd

    def load_state_dict(self, state_dict):
        velvet_state = state_dict.pop("velvet_state", None)
        version = state_dict.pop("velvet_version", 0)
        super().load_state_dict(state_dict)
        if velvet_state is not None:
            # Only assign keys that exist on this version of the optimizer —
            # silently drop unknown keys instead of failing the load.
            for k, v in velvet_state.items():
                if k in self._VELVET_STATE_KEYS:
                    setattr(self, k, v)
            if version < self.STATE_DICT_VERSION:
                # Older checkpoints don't have the new configurable thresholds;
                # the constructor defaults are already in place — nothing to do.
                pass

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # NaN/Inf guard: scan all gradients ONCE before touching any state.
        # Without this, a single inf gradient would corrupt:
        #   - the param tensor (irreversible)
        #   - the m/v Adam moments (sticky for 1/(1-β2) ≈ 1000 steps)
        #   - the LVS log-loss EMA via downstream metric updates
        if self._skip_nonfinite:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all():
                        self._skipped_steps += 1
                        # Do NOT increment _global_step or per-param state["step"].
                        # That keeps bias correction aligned with the number of
                        # successful updates, not the number of attempts.
                        return loss

        self._global_step += 1

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            base_lr = group["lr"]
            wd = group["weight_decay"]
            sparse_th = group.get("sparse_threshold", 1e-9)

            # Split params into kernel-eligible vs fallback for foreach grouping.
            kernel_params = []
            fallback_params = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Velvet does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1

                if (self._kernel_backend in ("triton", "cuda")
                        and p.is_cuda and velvet_update_kernel is not None):
                    kernel_params.append(p)
                else:
                    fallback_params.append(p)

            # --- Kernel path: per-param fused launch ---
            for p in kernel_params:
                state = self.state[p]
                bc1 = 1.0 - beta1 ** state["step"]
                bc2 = 1.0 - beta2 ** state["step"]
                velvet_update_kernel(
                    param=p.data,
                    grad=p.grad,
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

            # --- Fallback path: multi-tensor ops (_foreach_*) ---
            if fallback_params:
                self._foreach_fallback_step(
                    fallback_params, group, beta1, beta2, eps, base_lr, wd,
                    sparse_th,
                )

        return loss

    def _foreach_fallback_step(self, params, group, beta1, beta2, eps, base_lr, wd, sparse_th):
        """Multi-tensor (foreach) fallback step.

        Uses the public torch._foreach_* ops, which dispatch to a single fused
        kernel each (vs a Python loop calling .mul_/.add_ N times). Bias-
        correction is applied per-param because state["step"] differs across
        params — done as a list of scalars passed to _foreach_div_scalar_list.
        """
        # Group sparse-aware vs not by per-group flag (uniform for the group).
        sparse_aware = group["sparse_aware"]
        entropy_adaptive = group["entropy_adaptive"]
        perplexity_guided = group["perplexity_guided"]

        # Materialise param/grad in f32 for math; native dtype copied back at end.
        p_f32 = [p.data.float() for p in params]
        g_f32 = [p.grad.float() for p in params]

        if sparse_aware:
            # Mask out grads where param is essentially zero — preserves sparsity.
            g_f32 = [
                torch.where(p.abs() >= sparse_th, g, torch.zeros_like(g))
                for p, g in zip(p_f32, g_f32)
            ]

        # Decoupled weight decay (AdamW): p ← p * (1 - lr*wd)
        if wd != 0.0:
            torch._foreach_mul_(p_f32, 1.0 - base_lr * wd)

        m_list = [self.state[p]["m"] for p in params]
        v_list = [self.state[p]["v"] for p in params]

        # m ← β1·m + (1-β1)·g    (single fused dispatch)
        torch._foreach_mul_(m_list, beta1)
        torch._foreach_add_(m_list, g_f32, alpha=1.0 - beta1)
        # v ← β2·v + (1-β2)·g²
        torch._foreach_mul_(v_list, beta2)
        torch._foreach_addcmul_(v_list, g_f32, g_f32, value=1.0 - beta2)

        # Per-param bias correction — list of scalars.
        bc1 = [1.0 - beta1 ** self.state[p]["step"] for p in params]
        bc2 = [1.0 - beta2 ** self.state[p]["step"] for p in params]
        m_hat = torch._foreach_div(m_list, bc1)
        v_hat = torch._foreach_div(v_list, bc2)

        # PGM: post-correction scale on m_hat
        if perplexity_guided and self._perplexity_scale != 1.0:
            torch._foreach_mul_(m_hat, self._perplexity_scale)

        # update = m_hat / (sqrt(v_hat) + eps)
        denom = torch._foreach_sqrt(v_hat)
        torch._foreach_add_(denom, eps)
        torch._foreach_div_(m_hat, denom)

        eff_lr = base_lr * self._entropy_scale if entropy_adaptive else base_lr
        torch._foreach_add_(p_f32, m_hat, alpha=-eff_lr)

        # Copy back into the original param tensors (handles bf16/fp16 cast).
        for p, p_new in zip(params, p_f32):
            p.data.copy_(p_new.to(p.dtype))
