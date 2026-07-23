"""Reduction of differentiable per-sample losses with explicit contributions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .base import ContributionResult, validate_contribution_result


@dataclass(frozen=True)
class ReductionSpec:
    """Declare how a weighted per-sample loss becomes a scalar objective."""

    normalization: str = "weight_sum_mean"

    def __post_init__(self) -> None:
        valid = {"weight_sum_mean", "batch_mean", "sum"}
        if self.normalization not in valid:
            raise ValueError(
                "reduction normalization must be weight_sum_mean, batch_mean, or sum"
            )


def reduce_per_sample_loss(
    per_sample_loss: Tensor,
    contribution: ContributionResult,
    spec: ReductionSpec | None = None,
) -> Tensor:
    """Apply selection and weights without detaching the loss graph."""

    if not isinstance(per_sample_loss, Tensor) or per_sample_loss.ndim != 1:
        raise ValueError("per_sample_loss must be a one-dimensional tensor")
    if per_sample_loss.numel() == 0:
        raise ValueError("per_sample_loss must not be empty")
    if not torch.is_floating_point(per_sample_loss):
        raise ValueError("per_sample_loss must use a floating-point dtype")
    if not bool(torch.isfinite(per_sample_loss).all().item()):
        raise ValueError("per_sample_loss must be finite")

    mask, weights = validate_contribution_result(
        contribution,
        batch_size=int(per_sample_loss.numel()),
        device=per_sample_loss.device,
    )
    spec = spec or ReductionSpec()
    effective_weights = weights.to(dtype=per_sample_loss.dtype) * mask.to(
        dtype=per_sample_loss.dtype
    )
    numerator = (per_sample_loss * effective_weights).sum()

    if spec.normalization == "sum":
        return numerator
    if spec.normalization == "batch_mean":
        return numerator / per_sample_loss.numel()
    return numerator / effective_weights.sum()
