from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


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
        if self.targets.ndim != 2:
            raise ValueError("soft targets must have floating shape [B, C]")
        if self.sample_indices.shape != (self.targets.shape[0],):
            raise ValueError("soft target indices must have shape [B]")
        if self.targets.requires_grad:
            raise ValueError("soft targets must be detached")
        if self.confidence is not None and self.confidence.shape != (self.targets.shape[0],):
            raise ValueError("soft target confidence must have shape [B]")
        if self.selected_mask is not None:
            if self.selected_mask.shape != (self.targets.shape[0],) or self.selected_mask.dtype != torch.bool:
                raise ValueError("soft target selected_mask must be boolean shape [B]")


@dataclass(frozen=True, slots=True)
class PseudoLabelResult:
    """Hard pseudo-labels with confidence and an explicit participation mask."""

    labels: Tensor
    confidence: Tensor
    selected_mask: Tensor
    sample_indices: Tensor

    def __post_init__(self) -> None:
        size = self.labels.numel()
        if self.labels.ndim != 1 or self.confidence.shape != (size,):
            raise ValueError("pseudo-label fields must align with shape [B]")
        if self.selected_mask.shape != (size,) or self.selected_mask.dtype != torch.bool:
            raise ValueError("pseudo-label selected_mask must be boolean shape [B]")
        if self.sample_indices.shape != (size,):
            raise ValueError("pseudo-label indices must have shape [B]")


@dataclass(frozen=True, slots=True)
class CandidateLabelResult:
    """Candidate-set supervision represented as a boolean [B, C] mask."""

    candidates: Tensor
    sample_indices: Tensor

    def __post_init__(self) -> None:
        if self.candidates.ndim != 2 or self.candidates.dtype != torch.bool:
            raise ValueError("candidate labels must be boolean shape [B, C]")
        if self.sample_indices.shape != (self.candidates.shape[0],):
            raise ValueError("candidate label indices must have shape [B]")
        if not bool(self.candidates.any(dim=1).all().item()):
            raise ValueError("every sample must have at least one candidate label")

