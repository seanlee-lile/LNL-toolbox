"""Task-neutral contracts for per-sample training contributions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class ContributionResult:
    """Hard inclusion, continuous weight, and scalar treatment statistics."""

    selected_mask: Tensor
    sample_weights: Tensor
    metrics: Mapping[str, float] = field(default_factory=dict)
    selection_mask: Tensor | None = None


def validate_contribution_result(
    result: ContributionResult,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Validate an aligned, non-empty contribution result."""

    if not isinstance(result, ContributionResult):
        raise TypeError("sample treatment must return a ContributionResult")

    mask = result.selected_mask
    if not isinstance(mask, Tensor) or mask.shape != (batch_size,):
        raise ValueError(
            f"selected_mask must have shape [{batch_size}]"
        )
    if mask.dtype != torch.bool:
        raise ValueError("selected_mask must use torch.bool")
    if mask.device != device:
        raise ValueError("selected_mask must be on the loss tensor device")

    weights = result.sample_weights
    if not isinstance(weights, Tensor) or weights.shape != (batch_size,):
        raise ValueError(
            f"sample_weights must have shape [{batch_size}]"
        )
    if not torch.is_floating_point(weights):
        raise ValueError("sample_weights must use a floating-point dtype")
    if weights.device != device:
        raise ValueError("sample_weights must be on the loss tensor device")
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("sample_weights must be finite")
    if bool((weights < 0).any().item()):
        raise ValueError("sample_weights must be non-negative")
    if not bool((mask & (weights > 0)).any().item()):
        raise ValueError("sample treatment must keep positive contribution")

    if not isinstance(result.metrics, Mapping):
        raise TypeError("sample treatment metrics must be a mapping")
    for name, value in result.metrics.items():
        if not isinstance(name, str):
            raise TypeError("sample treatment metric names must be strings")
        if not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(
                f"sample treatment metric {name!r} must be a finite scalar"
            )
    if result.selection_mask is not None:
        selection_mask = result.selection_mask
        if not isinstance(selection_mask, Tensor) or selection_mask.shape != (batch_size,):
            raise ValueError("selection_mask must have shape [batch_size]")
        if selection_mask.dtype != torch.bool or selection_mask.device != device:
            raise ValueError("selection_mask must be a boolean mask on the loss device")
    return mask, weights
