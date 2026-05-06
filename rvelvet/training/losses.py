"""Loss computation per training phase. Phase 1: CE only. Phase 2: CE + ACR auxiliary losses.
Phase 3: CE + halting loss + deep supervision."""

import torch

from ..layers.adaptive_router import compute_acr_losses
from ..layers.halting import compute_halting_loss
from .kernels.fused_ce import fused_cross_entropy


def _ce_loss(logits_2d: torch.Tensor, targets_1d: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with -1 ignore semantics, routed through fused_cross_entropy.

    fused_cross_entropy avoids materializing the [N, V] softmax buffer
    (~4.9 GB for our config). It does not natively support ignore_index, so we
    filter -1 rows ourselves. In pretraining there are no -1 targets, so the
    fast path runs without copies.
    """
    if (targets_1d == -1).any():
        valid = targets_1d != -1
        loss, _ = fused_cross_entropy(logits_2d[valid], targets_1d[valid])
    else:
        loss, _ = fused_cross_entropy(logits_2d, targets_1d)
    return loss


def compute_phase_loss(
    model_output: dict,
    targets: torch.Tensor,
    phase: str,
    vocab_size: int,
    cfg,
    model=None,
) -> tuple:
    """Compute total loss and components for the given phase."""
    logits = model_output['logits']
    B, L, V = logits.shape
    ce_loss = _ce_loss(logits.reshape(B * L, V), targets.reshape(B * L))
    loss_dict = {'ce': ce_loss}
    total_loss = ce_loss

    if phase == 'phase2_acr':
        acr_losses = compute_acr_losses(
            model_output['route_weights'],
            model_output['route_logits'],
        )
        lb = cfg.lambda_load_balance * acr_losses['load_balance']
        ent = cfg.lambda_entropy * acr_losses['entropy']
        cc = cfg.lambda_compute_cost * acr_losses['compute_cost']

        total_loss = total_loss + lb + ent + cc
        loss_dict['load_balance'] = acr_losses['load_balance']
        loss_dict['entropy'] = acr_losses['entropy']
        loss_dict['compute_cost'] = acr_losses['compute_cost']

    elif phase == 'phase3_iterative':
        halting_loss = compute_halting_loss(
            model_output['p_halts'],
            lambda_p=getattr(cfg, 'lambda_p', 0.5),
        )
        total_loss = total_loss + cfg.lambda_halting * halting_loss
        loss_dict['halting'] = halting_loss

        if model is not None and 'iteration_outputs' in model_output:
            local_out = model_output.get('local_out')
            iter_outputs = model_output['iteration_outputs']

            if local_out is not None and len(iter_outputs) > 0:
                deep_ce = _compute_deep_supervision(
                    model, iter_outputs, local_out, targets, vocab_size,
                )
                total_loss = total_loss + cfg.lambda_deep_supervision * deep_ce
                loss_dict['deep_supervision'] = deep_ce

    loss_dict['total'] = total_loss
    return total_loss, loss_dict


def _compute_deep_supervision(
    model,
    iter_outputs: list,
    local_out: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Deep supervision: expand each iteration's concepts to token-level, project through lm_head,
    compute CE, and return mean across iterations."""
    B, L = targets.shape
    losses = []

    for iter_concepts in iter_outputs:
        expanded = model.expansion(local_out, iter_concepts)
        normed = model.out_norm(expanded)
        logits = model.lm_head(normed)  # (B, L, V)
        ce = _ce_loss(logits.reshape(B * L, vocab_size), targets.reshape(B * L))
        losses.append(ce)

    return torch.stack(losses).mean()
