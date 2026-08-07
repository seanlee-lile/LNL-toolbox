from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}
_PROBABILITY_ATOL = 1e-6
_PROBABILITY_RTOL = 1e-5


def _validate_sample_indices(
    sample_indices: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    owner: str,
) -> None:
    if not isinstance(sample_indices, torch.Tensor):
        raise TypeError(f"{owner} sample_indices must be a tensor")
    if sample_indices.ndim != 1 or sample_indices.shape != (batch_size,):
        raise ValueError(f"{owner} sample_indices must have shape [B]")
    if sample_indices.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{owner} sample_indices must use an integer dtype")
    if sample_indices.device != device:
        raise ValueError(
            f"{owner} sample_indices must be on the result tensor device"
        )
    if not bool(torch.isfinite(sample_indices).all().item()):
        raise ValueError(f"{owner} sample_indices must be finite")
    if torch.unique(sample_indices).numel() != sample_indices.numel():
        raise ValueError(f"{owner} sample_indices must be unique")


def _validate_selected_mask(
    selected_mask: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    owner: str,
) -> None:
    if not isinstance(selected_mask, torch.Tensor):
        raise TypeError(f"{owner} selected_mask must be a tensor")
    if selected_mask.shape != (batch_size,) or selected_mask.dtype != torch.bool:
        raise ValueError(f"{owner} selected_mask must be boolean shape [B]")
    if selected_mask.device != device:
        raise ValueError(
            f"{owner} selected_mask must be on the result tensor device"
        )


def _validate_confidence(
    confidence: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    owner: str,
) -> None:
    if not isinstance(confidence, torch.Tensor):
        raise TypeError(f"{owner} confidence must be a tensor")
    if confidence.shape != (batch_size,):
        raise ValueError(f"{owner} confidence must have shape [B]")
    if not torch.is_floating_point(confidence):
        raise ValueError(f"{owner} confidence must use a floating-point dtype")
    if confidence.device != device:
        raise ValueError(
            f"{owner} confidence must be on the result tensor device"
        )
    if confidence.requires_grad:
        raise ValueError(f"{owner} confidence must be detached")
    if not bool(torch.isfinite(confidence).all().item()):
        raise ValueError(f"{owner} confidence must be finite")
    if bool((confidence < 0).any().item()):
        raise ValueError(f"{owner} confidence must be non-negative")


@dataclass(frozen=True, slots=True)
class Artifact:
    name: str
    value: Any
    media_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepResult:
    """Common result envelope; every field is optional."""

    outputs: Any = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SoftTargetResult:
    """Detached replacement targets aligned by stable sample index."""

    targets: Tensor
    sample_indices: Tensor
    confidence: Tensor | None = None
    selected_mask: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.targets, torch.Tensor) or self.targets.ndim != 2:
            raise ValueError("soft targets must have floating shape [B, C]")
        batch_size, classes = self.targets.shape
        if batch_size == 0 or classes == 0:
            raise ValueError("soft targets must not be empty")
        if not torch.is_floating_point(self.targets):
            raise ValueError("soft targets must use a floating-point dtype")
        if self.targets.requires_grad:
            raise ValueError("soft targets must be detached")
        if not bool(torch.isfinite(self.targets).all().item()):
            raise ValueError("soft targets must be finite")
        if bool((self.targets < 0).any().item()):
            raise ValueError("soft targets must be non-negative")
        if not torch.allclose(
            self.targets.sum(dim=1),
            torch.ones(
                batch_size,
                dtype=self.targets.dtype,
                device=self.targets.device,
            ),
            atol=_PROBABILITY_ATOL,
            rtol=_PROBABILITY_RTOL,
        ):
            raise ValueError("each soft target row must sum to one")
        _validate_sample_indices(
            self.sample_indices,
            batch_size=batch_size,
            device=self.targets.device,
            owner="soft target",
        )
        if self.confidence is not None:
            _validate_confidence(
                self.confidence,
                batch_size=batch_size,
                device=self.targets.device,
                owner="soft target",
            )
        if self.selected_mask is not None:
            _validate_selected_mask(
                self.selected_mask,
                batch_size=batch_size,
                device=self.targets.device,
                owner="soft target",
            )


@dataclass(frozen=True, slots=True)
class PseudoLabelResult:
    """Hard pseudo-labels with confidence and an explicit participation mask."""

    labels: Tensor
    confidence: Tensor
    selected_mask: Tensor
    sample_indices: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.labels, torch.Tensor) or self.labels.ndim != 1:
            raise ValueError("pseudo labels must have shape [B]")
        size = self.labels.numel()
        if size == 0:
            raise ValueError("pseudo labels must not be empty")
        if self.labels.dtype not in _INTEGER_DTYPES:
            raise ValueError("pseudo labels must use an integer dtype")
        if bool((self.labels < 0).any().item()):
            raise ValueError("pseudo labels must be non-negative")
        _validate_sample_indices(
            self.sample_indices,
            batch_size=size,
            device=self.labels.device,
            owner="pseudo label",
        )
        _validate_confidence(
            self.confidence,
            batch_size=size,
            device=self.labels.device,
            owner="pseudo label",
        )
        _validate_selected_mask(
            self.selected_mask,
            batch_size=size,
            device=self.labels.device,
            owner="pseudo label",
        )


@dataclass(frozen=True, slots=True)
class CandidateLabelResult:
    """Candidate-set supervision represented as a boolean [B, C] mask."""

    candidates: Tensor
    sample_indices: Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidates, torch.Tensor)
            or self.candidates.ndim != 2
            or self.candidates.dtype != torch.bool
        ):
            raise ValueError("candidate labels must be boolean shape [B, C]")
        batch_size, classes = self.candidates.shape
        if batch_size == 0 or classes == 0:
            raise ValueError("candidate labels must not be empty")
        _validate_sample_indices(
            self.sample_indices,
            batch_size=batch_size,
            device=self.candidates.device,
            owner="candidate label",
        )
        if not bool(self.candidates.any(dim=1).all().item()):
            raise ValueError("every sample must have at least one candidate label")

