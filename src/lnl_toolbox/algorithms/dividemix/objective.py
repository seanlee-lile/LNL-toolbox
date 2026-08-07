from __future__ import annotations

"""Paper-oriented DivideMix MixMatch objective."""

import torch
from torch import Tensor


def unsupervised_weight(maximum: float, progress: float, warmup_epochs: int, rampup_epochs: int) -> float:
    fraction = min(1.0, max(0.0, (float(progress) - warmup_epochs) / rampup_epochs))
    return float(maximum) * fraction


def dividemix_objective(logits_labeled: Tensor, targets_labeled: Tensor, logits_unlabeled: Tensor, targets_unlabeled: Tensor, logits_all: Tensor, *, lambda_u: float, lambda_r: float) -> tuple[Tensor, dict[str, float]]:
    if logits_labeled.shape != targets_labeled.shape or logits_unlabeled.shape != targets_unlabeled.shape:
        raise ValueError("DivideMix logits and soft targets must have matching [B,C] shapes")
    log_prob = torch.log_softmax(logits_labeled, dim=1)
    loss_x = -(targets_labeled * log_prob).sum(1).mean()
    probabilities_u = torch.softmax(logits_unlabeled, dim=1)
    loss_u = torch.mean((probabilities_u - targets_unlabeled) ** 2)
    prediction_mean = torch.softmax(logits_all, dim=1).mean(0)
    prior = torch.full_like(prediction_mean, 1.0 / prediction_mean.numel())
    loss_regularizer = torch.sum(prior * torch.log(prior / prediction_mean.clamp_min(torch.finfo(prediction_mean.dtype).tiny)))
    objective = loss_x + float(lambda_u) * loss_u + float(lambda_r) * loss_regularizer
    if not bool(torch.isfinite(objective).item()):
        raise FloatingPointError("DivideMix objective is not finite")
    return objective, {
        "loss_x": float(loss_x.detach().item()),
        "loss_u": float(loss_u.detach().item()),
        "loss_regularizer": float(loss_regularizer.detach().item()),
        "lambda_u": float(lambda_u),
        "lambda_r": float(lambda_r),
        "objective": float(objective.detach().item()),
    }
