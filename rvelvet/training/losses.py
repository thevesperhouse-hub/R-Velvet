"""
Loss computation per training phase.

Phase 1: Cross-entropy only
Phase 2: CE + ACR auxiliary losses (load_balance, entropy, compute_cost)
Phase 3: CE + halting loss + deep supervision
"""

import torch
import torch.nn.functional as F

from ..layers.adaptive_router import compute_acr_losses
from ..layers.halting import compute_halting_loss


def compute_phase_loss(
    model_output: dict,
    targets: torch.Tensor,
    phase: str,
    vocab_size: int,
    cfg,
    model=None,
) -> tuple:
    """
    Compute total loss and individual components for a given training phase.

    Args:
        model_output: dict from model.forward()
        targets: (B, L) target token IDs
        phase: one of 'phase1_pretrain', 'phase2_acr', 'phase3_iterative'
        vocab_size: vocabulary size for CE
        cfg: training config (OmegaConf or similar) with loss weights
        model: the RVelvet model (needed for Phase 3 deep supervision)

    Returns:
        (total_loss, loss_dict) where loss_dict has all individual loss terms
    """
    logits = model_output['logits']  # (B, L, V)
    B, L, V = logits.shape

    # --- Cross-entropy (all phases) ---
    ce_loss = F.cross_entropy(
        logits.reshape(B * L, V),
        targets.reshape(B * L),
        ignore_index=-1,
    )
    loss_dict = {'ce': ce_loss}
    total_loss = ce_loss

    # --- Phase 2: ACR losses ---
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

    # --- Phase 3: Halting + Deep supervision ---
    elif phase == 'phase3_iterative':
        # Halting loss
        halting_loss = compute_halting_loss(
            model_output['p_halts'],
            lambda_p=getattr(cfg, 'lambda_p', 0.5),
        )
        total_loss = total_loss + cfg.lambda_halting * halting_loss
        loss_dict['halting'] = halting_loss

        # Deep supervision: CE on each iteration's output
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
    """
    Deep supervision: expand each iteration's concepts back to token-level,
    project through lm_head, compute CE.

    Args:
        model: RVelvet model (needs .expansion, .out_norm, .lm_head)
        iter_outputs: list of (B, N, D) concept tensors per iteration
        local_out: (B, L, D) local encoder output
        targets: (B, L) target token IDs
        vocab_size: vocabulary size

    Returns:
        mean CE loss across iterations
    """
    B, L = targets.shape
    losses = []

    for iter_concepts in iter_outputs:
        expanded = model.expansion(local_out, iter_concepts)
        normed = model.out_norm(expanded)
        logits = model.lm_head(normed)  # (B, L, V)
        ce = F.cross_entropy(
            logits.reshape(B * L, vocab_size),
            targets.reshape(B * L),
            ignore_index=-1,
        )
        losses.append(ce)

    return torch.stack(losses).mean()
