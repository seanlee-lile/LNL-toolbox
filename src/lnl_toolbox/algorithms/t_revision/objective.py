from __future__ import annotations

"""Corrected vectorized implementation of T-Revision Equation (3)."""

from dataclasses import dataclass, field
import math

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class TRevisionObjectiveResult:
    objective: Tensor
    base_ce_mean: float
    weight_mean: float
    weight_min: float
    weight_max: float
    numerator_min: float
    denominator_min: float
    clean_probability_entropy: float
    sample_weights: Tensor = field(repr=False)
    sample_denominators: Tensor = field(repr=False)

    @property
    def metrics(self) -> dict[str, float]:
        return {
            "weighted_objective": float(self.objective.detach().item()),
            "base_ce": self.base_ce_mean,
            "weight_mean": self.weight_mean,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "numerator_min": self.numerator_min,
            "denominator_min": self.denominator_min,
            "clean_posterior_entropy": self.clean_probability_entropy,
        }


def t_revision_reweight_objective(
    logits: Tensor,
    noisy_targets: Tensor,
    transition: Tensor,
    *,
    denominator_floor: float,
    detach_ratio: bool = False,
) -> TRevisionObjectiveResult:
    """Return paper Eq. (3) with row-vector clean-to-noisy convention.

    ``detach_ratio`` controls only autograd ownership, not Equation (3)'s
    numerical value.  Classifier initialization uses a detached fixed-T ratio;
    transition revision keeps the ratio differentiable so both the classifier
    and additive transition correction receive gradients.  No inverse,
    clipping, normalization, or projection is applied.
    """

    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError("logits must have shape [B, C]")
    batch_size, num_classes = logits.shape
    if batch_size == 0 or num_classes < 2 or not torch.is_floating_point(logits):
        raise ValueError("logits must be a non-empty floating [B, C] tensor")
    if not torch.is_tensor(noisy_targets) or noisy_targets.shape != (batch_size,):
        raise ValueError(f"noisy_targets must have shape [{batch_size}]")
    if noisy_targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("noisy_targets must use an integer dtype")
    if noisy_targets.device != logits.device:
        raise ValueError("logits and noisy_targets must share a device")
    if noisy_targets.numel() and (
        int(noisy_targets.min().item()) < 0
        or int(noisy_targets.max().item()) >= num_classes
    ):
        raise ValueError("noisy_targets contain an out-of-range class")
    if not torch.is_tensor(transition) or transition.shape != (num_classes, num_classes):
        raise ValueError(f"transition must have shape [{num_classes}, {num_classes}]")
    if not torch.is_floating_point(transition):
        raise ValueError("transition must use a floating-point dtype")
    if transition.device != logits.device:
        raise ValueError("logits and transition must share a device")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must be finite")
    if not bool(torch.isfinite(transition).all().item()):
        raise ValueError("transition must be finite")
    floor = float(denominator_floor)
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("denominator_floor must be finite and non-negative")

    clean_prob = torch.softmax(logits, dim=1)
    noisy_prob = clean_prob @ transition
    gather_index = noisy_targets.to(dtype=torch.long).unsqueeze(1)
    numerator = clean_prob.gather(1, gather_index).squeeze(1)
    denominator = noisy_prob.gather(1, gather_index).squeeze(1)
    if not bool(torch.isfinite(denominator).all().item()):
        raise ValueError("T-Revision denominator must be finite")
    if bool((denominator <= floor).any().item()):
        raise ValueError(
            "T-Revision denominator must be strictly greater than "
            "denominator_floor"
        )
    weights = numerator / denominator
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("T-Revision importance weights must be finite")
    base_ce = F.cross_entropy(logits, noisy_targets.long(), reduction="none")
    loss_weights = weights.detach() if detach_ratio else weights
    objective = (loss_weights * base_ce).mean()
    if not bool(torch.isfinite(objective).item()):
        raise ValueError("T-Revision objective must be finite")
    entropy = -(clean_prob * clean_prob.clamp_min(torch.finfo(clean_prob.dtype).tiny).log()).sum(dim=1).mean()
    return TRevisionObjectiveResult(
        objective=objective,
        base_ce_mean=float(base_ce.detach().mean().item()),
        weight_mean=float(weights.detach().mean().item()),
        weight_min=float(weights.detach().min().item()),
        weight_max=float(weights.detach().max().item()),
        numerator_min=float(numerator.detach().min().item()),
        denominator_min=float(denominator.detach().min().item()),
        clean_probability_entropy=float(entropy.detach().item()),
        sample_weights=weights.detach(),
        sample_denominators=denominator.detach(),
    )
