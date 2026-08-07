from __future__ import annotations

"""DivideMix-private minibatch MixUp assembly."""

from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class MixedBatch:
    inputs: Tensor
    targets: Tensor
    labeled_count: int
    mix_lambda: float
    permutation: Tensor


def mixmatch_mixup(labeled_views: tuple[Tensor, ...], unlabeled_views: tuple[Tensor, ...], labeled_targets: Tensor, unlabeled_targets: Tensor, *, alpha: float, rng: np.random.Generator | None = None, permutation: Tensor | None = None) -> MixedBatch:
    if not labeled_views or len(labeled_views) != len(unlabeled_views):
        raise ValueError("MixMatch requires equal non-empty labeled/unlabeled view sets")
    if alpha <= 0.0:
        raise ValueError("MixUp alpha must be positive")
    labeled = torch.cat(labeled_views, dim=0)
    unlabeled = torch.cat(unlabeled_views, dim=0)
    repeated_labeled = labeled_targets.repeat(len(labeled_views), 1)
    repeated_unlabeled = unlabeled_targets.repeat(len(unlabeled_views), 1)
    inputs = torch.cat((labeled, unlabeled), dim=0)
    targets = torch.cat((repeated_labeled, repeated_unlabeled), dim=0)
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("MixMatch inputs and targets must align")
    generator = rng or np.random.default_rng()
    sampled = float(generator.beta(alpha, alpha))
    coefficient = max(sampled, 1.0 - sampled)
    order = torch.randperm(inputs.shape[0], device=inputs.device) if permutation is None else permutation.to(inputs.device)
    if order.shape != (inputs.shape[0],) or torch.unique(order).numel() != inputs.shape[0]:
        raise ValueError("MixMatch permutation must contain every batch position once")
    mixed_inputs = coefficient * inputs + (1.0 - coefficient) * inputs[order]
    mixed_targets = coefficient * targets + (1.0 - coefficient) * targets[order]
    return MixedBatch(mixed_inputs, mixed_targets.detach(), labeled.shape[0], coefficient, order.detach())
