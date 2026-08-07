"""Typed result contracts for reliability and statistic estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import (
    Generic,
    Mapping,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import torch
from torch import Tensor


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass(frozen=True)
class ReliabilityResult:
    """Sample-aligned evidence where a larger score means higher reliability."""

    sample_indices: Tensor
    scores: Tensor
    metrics: Mapping[str, float] = field(default_factory=dict)


InputT_contra = TypeVar("InputT_contra", contravariant=True)


@runtime_checkable
class ReliabilityEstimator(Protocol[InputT_contra]):
    """Compute sample-aligned reliability from a method-specific input."""

    def estimate(self, estimator_input: InputT_contra) -> ReliabilityResult:
        """Return evidence aligned with the input's stable sample identities."""
        ...


StatisticT = TypeVar("StatisticT")


@dataclass(frozen=True)
class StatisticResult(Generic[StatisticT]):
    """A method-specific statistic payload plus finite scalar metrics."""

    statistics: StatisticT
    metrics: Mapping[str, float] = field(default_factory=dict)


def _validate_metrics(metrics: Mapping[str, float], *, owner: str) -> None:
    if not isinstance(metrics, Mapping):
        raise TypeError(f"{owner} metrics must be a mapping")
    for name, value in metrics.items():
        if not isinstance(name, str):
            raise TypeError(f"{owner} metric names must be strings")
        if type(value) is not float:
            raise TypeError(
                f"{owner} metric {name!r} must be a Python float"
            )
        if not math.isfinite(value):
            raise ValueError(f"{owner} metric {name!r} must be finite")


def validate_reliability_result(
    result: ReliabilityResult,
    *,
    expected_sample_indices: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Validate detached reliability evidence and its stable-index alignment."""

    if not isinstance(result, ReliabilityResult):
        raise TypeError(
            "reliability estimator must return a ReliabilityResult"
        )

    indices = result.sample_indices
    scores = result.scores
    if not isinstance(indices, Tensor) or indices.ndim != 1:
        raise ValueError("reliability sample_indices must be one-dimensional")
    if indices.numel() == 0:
        raise ValueError("reliability sample_indices must not be empty")
    if indices.dtype not in _INTEGER_DTYPES:
        raise ValueError("reliability sample_indices must use an integer dtype")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("reliability sample_indices must be unique")

    if not isinstance(scores, Tensor) or scores.ndim != 1:
        raise ValueError("reliability scores must be one-dimensional")
    if scores.numel() != indices.numel():
        raise ValueError(
            "reliability scores must align one-to-one with sample_indices"
        )
    if not torch.is_floating_point(scores):
        raise ValueError("reliability scores must use a floating-point dtype")
    if scores.device != indices.device:
        raise ValueError(
            "reliability scores and sample_indices must be on the same device"
        )
    if scores.requires_grad:
        raise ValueError("reliability scores must be detached from autograd")
    if not bool(torch.isfinite(scores).all().item()):
        raise ValueError("reliability scores must be finite")

    if expected_sample_indices is not None:
        if (
            not isinstance(expected_sample_indices, Tensor)
            or expected_sample_indices.ndim != 1
        ):
            raise ValueError(
                "expected_sample_indices must be a one-dimensional tensor"
            )
        if expected_sample_indices.device != indices.device:
            raise ValueError(
                "expected_sample_indices must be on the result index device"
            )
        if expected_sample_indices.dtype not in _INTEGER_DTYPES:
            raise ValueError(
                "expected_sample_indices must use an integer dtype"
            )
        if not torch.equal(indices, expected_sample_indices):
            raise ValueError(
                "reliability sample_indices do not match the expected order"
            )

    _validate_metrics(result.metrics, owner="reliability")
    return indices, scores


def validate_statistic_result(
    result: StatisticResult[StatisticT],
) -> StatisticT:
    """Validate only the generic result container and its scalar metrics."""

    if not isinstance(result, StatisticResult):
        raise TypeError(
            "statistic estimator must return a StatisticResult"
        )
    _validate_metrics(result.metrics, owner="statistic")
    return result.statistics
