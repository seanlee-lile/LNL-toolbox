"""Task-neutral contracts for selecting samples within one batch."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
from torch import Tensor


@dataclass(frozen=True)
class SelectionInput:
    """Detached per-sample scores and stable sample identities for one batch."""

    scores: Tensor
    sample_indices: Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionResult:
    """A hard selection mask plus selector-owned scalar statistics."""

    selected_mask: Tensor
    metrics: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class Selector(Protocol):
    """Stateless interface for selecting samples from one score vector."""

    def select(self, selection_input: SelectionInput) -> SelectionResult:
        """Return a hard mask aligned with ``selection_input.scores``."""


def validate_selection_input(selection_input: SelectionInput) -> int:
    """Validate the generic selector boundary and return the batch size."""

    if not isinstance(selection_input, SelectionInput):
        raise TypeError("selection_input must be a SelectionInput")

    scores = selection_input.scores
    indices = selection_input.sample_indices
    if not isinstance(scores, Tensor) or scores.ndim != 1:
        raise ValueError("selector scores must be a one-dimensional tensor")
    if scores.numel() == 0:
        raise ValueError("selector scores must not be empty")
    if scores.requires_grad:
        raise ValueError("selector scores must be detached from autograd")
    if not torch.is_floating_point(scores):
        raise ValueError("selector scores must use a floating-point dtype")
    if not bool(torch.isfinite(scores).all().item()):
        raise ValueError("selector scores must be finite")

    if not isinstance(indices, Tensor) or indices.ndim != 1:
        raise ValueError("sample_indices must be a one-dimensional tensor")
    if indices.numel() != scores.numel():
        raise ValueError("sample_indices must align one-to-one with scores")
    if indices.device != scores.device:
        raise ValueError("sample_indices and scores must be on the same device")
    if indices.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("sample_indices must use an integer dtype")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("sample_indices must be unique within a batch")
    if not isinstance(selection_input.metadata, Mapping):
        raise TypeError("selector metadata must be a mapping")
    return int(scores.numel())


def validate_selection_result(
    result: SelectionResult,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Validate a selector result and return its hard mask."""

    if not isinstance(result, SelectionResult):
        raise TypeError("selector must return a SelectionResult")
    mask = result.selected_mask
    if not isinstance(mask, Tensor) or mask.ndim != 1:
        raise ValueError("selected_mask must be a one-dimensional tensor")
    if mask.dtype != torch.bool:
        raise ValueError("selected_mask must use torch.bool")
    if mask.device != device:
        raise ValueError("selected_mask must be on the score tensor device")
    if mask.numel() != batch_size:
        raise ValueError("selected_mask must align one-to-one with scores")
    if not bool(mask.any().item()):
        raise ValueError("selector must keep at least one sample")
    if not isinstance(result.metrics, Mapping):
        raise TypeError("selector metrics must be a mapping")
    for name, value in result.metrics.items():
        if not isinstance(name, str):
            raise TypeError("selector metric names must be strings")
        if not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"selector metric {name!r} must be a finite scalar")
    return mask
