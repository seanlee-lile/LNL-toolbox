from __future__ import annotations

"""Shared target-construction contracts with no clean-label dependency."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TYPE_CHECKING, runtime_checkable

import torch

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

from .result import (
    CandidateLabelResult,
    PseudoLabelResult,
    SoftTargetResult,
    _validate_sample_indices,
)


@dataclass(frozen=True, slots=True)
class TargetInput:
    logits: Tensor
    noisy_targets: Tensor
    sample_indices: Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class LabelProvider(Protocol):
    """Construct an alternative supervision object from noisy-only inputs."""

    def resolve(self, target_input: TargetInput) -> Any: ...


@runtime_checkable
class SoftTargetProvider(LabelProvider, Protocol):
    def resolve(self, target_input: TargetInput) -> SoftTargetResult: ...


@runtime_checkable
class PseudoLabelProvider(LabelProvider, Protocol):
    def resolve(self, target_input: TargetInput) -> PseudoLabelResult: ...


@runtime_checkable
class CandidateSetProvider(LabelProvider, Protocol):
    def resolve(self, target_input: TargetInput) -> CandidateLabelResult: ...


@dataclass(frozen=True, slots=True)
class ComplementaryLabelResult:
    """Negative/complementary labels represented as a boolean [B, C] mask."""

    negatives: Tensor
    sample_indices: Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.negatives, torch.Tensor)
            or self.negatives.ndim != 2
            or self.negatives.dtype != torch.bool
        ):
            raise ValueError("complementary labels must be boolean shape [B, C]")
        batch_size, classes = self.negatives.shape
        if batch_size == 0 or classes == 0:
            raise ValueError("complementary labels must not be empty")
        _validate_sample_indices(
            self.sample_indices,
            batch_size=batch_size,
            device=self.negatives.device,
            owner="complementary label",
        )
        if not bool(self.negatives.any(dim=1).all().item()):
            raise ValueError(
                "every sample must have at least one complementary label"
            )


__all__ = [
    "CandidateLabelResult",
    "CandidateSetProvider",
    "ComplementaryLabelResult",
    "LabelProvider",
    "PseudoLabelProvider",
    "PseudoLabelResult",
    "SoftTargetProvider",
    "SoftTargetResult",
    "TargetInput",
]
