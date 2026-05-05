"""Auto-sizing utilities for R-Velvet.

Given a target parameter budget (e.g., 1.3B) and a vocabulary size, search the
(d_model, n_local_layers, n_global_layers) space and return a complete model
config that lands within tolerance of the target.

Param counting uses PyTorch's `meta` device, which constructs tensors without
allocating memory — fast enough to evaluate dozens of candidate configs in
under a second even at 7B scale.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# ----------------------------------------------------------------------
# Size parsing / formatting
# ----------------------------------------------------------------------
_SIZE_RE = re.compile(r"""
    ^\s*
    ([0-9]+(?:[._][0-9]+)?)        # number  (1, 1.3, 1_300)
    \s*
    ([KMBT]?)                       # optional unit
    \s*$
""", re.IGNORECASE | re.VERBOSE)

_UNIT_MULTIPLIERS = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def parse_size(s) -> int:
    """Accept '1.3B', '350M', '1_300_000_000', or an int. Return int params count."""
    if isinstance(s, (int, float)):
        return int(s)
    if not isinstance(s, str):
        raise TypeError(f"parse_size expects str/int/float, got {type(s).__name__}")
    s_clean = s.strip().replace("_", "")
    m = _SIZE_RE.match(s_clean)
    if not m:
        # Fallback: maybe a raw integer with underscores.
        try:
            return int(s.replace("_", "").strip())
        except ValueError:
            raise ValueError(f"Cannot parse size: {s!r}") from None
    num = float(m.group(1).replace("_", ""))
    unit = m.group(2).upper()
    return int(num * _UNIT_MULTIPLIERS[unit])


def format_size(n: int) -> str:
    """Inverse of parse_size, with sensible precision."""
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


# ----------------------------------------------------------------------
# Param counting via meta device
# ----------------------------------------------------------------------
def estimate_params(cfg: Dict, *, use_acr: bool = False,
                    use_iterative_reasoning: bool = False) -> int:
    """Build RVelvet on the meta device and count parameters.

    The meta device skips memory allocation, so even a 7B config builds in
    well under a second. Returns the exact param count for the given config.
    """
    # Local import to avoid a circular import at package load time.
    from ..model import RVelvet

    kwargs = dict(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_local_layers=cfg["n_local_layers"],
        n_global_layers=cfg["n_global_layers"],
        n_local_heads=cfg["n_local_heads"],
        n_global_heads=cfg["n_global_heads"],
        window_size=cfg.get("window_size", 512),
        segment_size=cfg.get("segment_size", 512),
        n_concepts=cfg.get("n_concepts", 1),
        n_refine_layers=cfg.get("n_refine_layers", 2),
        memory_size=cfg.get("memory_size", 256),
        n_read_steps=cfg.get("n_read_steps", 2),
        ffn_mult=cfg.get("ffn_mult", 4.0),
        dropout=0.0,
        max_seq_len=cfg.get("max_seq_len", 8192),
        use_acr=use_acr,
        use_iterative_reasoning=use_iterative_reasoning,
        max_reasoning_iterations=cfg.get("max_reasoning_iterations", 8),
        lora_rank=cfg.get("lora_rank", 8),
        halt_threshold=cfg.get("halt_threshold", 0.5),
        lambda_p=cfg.get("lambda_p", 0.5),
    )

    with torch.device("meta"):
        model = RVelvet(**kwargs)
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------
# Auto-sizing search
# ----------------------------------------------------------------------
# Search space — d_model values that produce well-aligned dimensions
# (multiples of head_dim) and stable training widths.
_DEFAULT_D_MODELS = [
    256, 384, 512, 640, 768, 896, 1024, 1152, 1280,
    1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328,
    3584, 3840, 4096, 4608, 5120, 5632, 6144, 7168, 8192,
]


@dataclass
class SizingResult:
    config: Dict
    params: int
    target: int
    error_pct: float

    def pretty(self) -> str:
        cfg = self.config
        head_dim = cfg["d_model"] // cfg["n_global_heads"]
        n_total = cfg["n_local_layers"] + cfg["n_global_layers"]
        lines = [
            f"  vocab_size:       {cfg['vocab_size']:,}",
            f"  d_model:          {cfg['d_model']:,}",
            f"  n_local_layers:   {cfg['n_local_layers']}",
            f"  n_global_layers:  {cfg['n_global_layers']}",
            f"                    (total {n_total} transformer layers)",
            f"  n_local_heads:    {cfg['n_local_heads']}  (head_dim={cfg['d_model']//cfg['n_local_heads']})",
            f"  n_global_heads:   {cfg['n_global_heads']}  (head_dim={head_dim})",
            f"  ffn_mult:         {cfg['ffn_mult']}",
            f"  max_seq_len:      {cfg['max_seq_len']:,}",
            f"",
            f"  Total params:     {format_size(self.params)} "
            f"(target {format_size(self.target)}, {self.error_pct:+.1f}%)",
        ]
        return "\n".join(lines)


def _choose_heads(d_model: int, head_dim: int) -> int:
    """Pick the largest divisor of d_model that yields a head_dim close to target."""
    candidate = max(1, d_model // head_dim)
    # Walk down to the nearest divisor of d_model.
    while candidate > 1 and (d_model % candidate) != 0:
        candidate -= 1
    return candidate


def _split_layers(n_total: int, ratio_local: float = 1 / 3) -> Tuple[int, int]:
    """Split total transformer depth between local encoder and global reasoner.

    Default: ~1/3 local, 2/3 global — concept-level reasoning matters more for
    the iterative reasoner / ACR downstream modules.
    """
    n_total = max(4, int(n_total))
    n_local = max(2, int(round(n_total * ratio_local)))
    n_global = max(2, n_total - n_local)
    return n_local, n_global


def _build_candidate(d_model: int, n_total: int, *,
                     vocab_size: int, head_dim: int, ffn_mult: float,
                     ratio_local: float, max_seq_len: int) -> Optional[Dict]:
    """Construct a full model config for given (d_model, n_total). Returns None
    if dimensions are invalid (head_dim incompatible, etc.)."""
    if d_model < 128 or d_model % 8 != 0:
        return None
    n_heads = _choose_heads(d_model, head_dim)
    if n_heads < 1 or d_model % n_heads != 0:
        return None
    n_local, n_global = _split_layers(n_total, ratio_local)

    cfg = {
        "name": "auto",
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "n_local_layers": int(n_local),
        "n_global_layers": int(n_global),
        "n_local_heads": int(n_heads),
        "n_global_heads": int(n_heads),
        "window_size": 512,
        "segment_size": 512,
        "n_concepts": 1,
        "n_refine_layers": 2,
        "memory_size": 256,
        "n_read_steps": 2,
        "ffn_mult": float(ffn_mult),
        "dropout": 0.0,
        "max_seq_len": int(max_seq_len),
        "max_reasoning_iterations": 8,
        "lora_rank": max(8, int(d_model // 256)),
        "halt_threshold": 0.5,
        "lambda_p": 0.5,
    }
    return cfg


def auto_size(
    target,
    *,
    vocab_size: int = 64000,
    head_dim: int = 128,
    ffn_mult: float = 4.0,
    ratio_local: float = 1 / 3,
    max_seq_len: int = 8192,
    d_model_choices: Optional[List[int]] = None,
    target_aspect: float = 110.0,
    min_layers: int = 12,
    max_layers: int = 80,
) -> SizingResult:
    """Search for the model config closest to a target parameter budget.

    The heuristic follows the modern shape of LLaMA-2/3/Mistral/Qwen: aim for
    `d_model / n_total_layers ≈ 100-130`. This avoids degenerate
    "ultra-wide-shallow" or "narrow-deep" configs that the param count alone
    is happy to pick.

    Args:
        target: target params count, accepts '1.3B', int, etc.
        vocab_size: tokenizer vocab size (default 64k for FR).
        head_dim: target head dimension (default 128, modern standard).
        ffn_mult: FFN expansion ratio (default 4.0).
        ratio_local: fraction of total depth allocated to LocalEncoder.
        max_seq_len: positional embedding range.
        d_model_choices: override the default search list.
        target_aspect: preferred d_model / total_layers ratio. 110 matches
                       the LLaMA-2/3/Mistral/Qwen2.5 family. Lower values
                       (~80) bias toward deeper, higher (~150) toward wider.
        min_layers: minimum total transformer depth — guards against the
                    sizer picking absurdly shallow configs to hit the target.
        max_layers: hard upper bound on depth.

    Returns:
        SizingResult with the chosen config + actual param count + error.
    """
    target = int(parse_size(target))
    if target < 1_000_000:
        raise ValueError(f"Target {target} suspiciously small; expected ≥ 1M params.")

    d_models = d_model_choices or _DEFAULT_D_MODELS
    candidates: List[Tuple[int, Dict]] = []

    # For each d_model, find the n_total that gets closest to the target via
    # binary search on the meta-device param count. R-Velvet's overhead
    # (compressor, memory controller, refine layers) makes a closed-form
    # estimator unreliable, so we search directly.
    def best_n_total_for_width(d: int) -> Optional[Tuple[int, int, Dict]]:
        lo, hi = min_layers, max_layers
        # Quick check: if even max_layers can't reach the target, skip.
        cfg_hi = _build_candidate(d, hi, vocab_size=vocab_size, head_dim=head_dim,
                                  ffn_mult=ffn_mult, ratio_local=ratio_local,
                                  max_seq_len=max_seq_len)
        cfg_lo = _build_candidate(d, lo, vocab_size=vocab_size, head_dim=head_dim,
                                  ffn_mult=ffn_mult, ratio_local=ratio_local,
                                  max_seq_len=max_seq_len)
        if cfg_hi is None or cfg_lo is None:
            return None

        # Bisect: param count is monotonic in n_total for fixed d.
        while hi - lo > 1:
            mid = (lo + hi) // 2
            cfg_mid = _build_candidate(d, mid, vocab_size=vocab_size, head_dim=head_dim,
                                       ffn_mult=ffn_mult, ratio_local=ratio_local,
                                       max_seq_len=max_seq_len)
            if cfg_mid is None:
                lo = mid
                continue
            p_mid = estimate_params(cfg_mid)
            if p_mid < target:
                lo = mid
            else:
                hi = mid

        # Pick whichever of {lo, hi} is closer in absolute params.
        best = None
        for nt in (lo, hi):
            cfg = _build_candidate(d, nt, vocab_size=vocab_size, head_dim=head_dim,
                                   ffn_mult=ffn_mult, ratio_local=ratio_local,
                                   max_seq_len=max_seq_len)
            if cfg is None:
                continue
            p = estimate_params(cfg)
            if best is None or abs(p - target) < abs(best[1] - target):
                best = (nt, p, cfg)
        return best

    for d in d_models:
        if d % head_dim != 0:
            continue
        # Skip widths whose embedding alone already overshoots the budget.
        if vocab_size * d * 2 > target * 1.2:
            continue
        result = best_n_total_for_width(d)
        if result is None:
            continue
        nt, params, cfg = result
        candidates.append((params, cfg))

    if not candidates:
        raise RuntimeError(
            f"No valid candidate found for target={format_size(target)} "
            f"with vocab={vocab_size}, head_dim={head_dim}. "
            "Try a smaller target or expand d_model_choices."
        )

    # Score: combined (param error %) + (aspect deviation %). Aspect weight
    # is 0.5 — strong enough to reject narrow-deep configs that match params
    # by saturating max_layers, but weak enough that a 5%-better param fit
    # still wins over a 10%-better aspect fit. Also penalize hitting
    # min_layers / max_layers boundary (signals the search ran out of room).
    def score(item):
        actual, cfg = item
        param_err_pct = abs(actual - target) / target * 100.0
        n_total = cfg["n_local_layers"] + cfg["n_global_layers"]
        aspect = cfg["d_model"] / max(1, n_total)
        aspect_err_pct = abs(aspect - target_aspect) / target_aspect * 100.0
        boundary_penalty = 50.0 if (n_total >= max_layers or n_total <= min_layers) else 0.0
        return param_err_pct + 0.5 * aspect_err_pct + boundary_penalty

    candidates.sort(key=score)
    best_actual, best_cfg = candidates[0]
    err_pct = (best_actual - target) / target * 100.0

    return SizingResult(
        config=best_cfg, params=best_actual, target=target, error_pct=err_pct,
    )
