"""Framework-neutral contracts for components that own an objective."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ObjectiveResult:
    """Optimization objective with separate reporting and sample accounting."""

    objective: Any
    selected_mask: Any | None = None
    reporting_loss: Any | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class ObjectiveConsumer(Protocol):
    """Own the scalar objective for one batch."""

    def compute(
        self,
        *,
        logits: Any,
        noisy_targets: Any,
        sample_indices: Any,
        base_loss: Any,
        metadata: Mapping[str, Any],
    ) -> ObjectiveResult:
        ...


__all__ = ["ObjectiveConsumer", "ObjectiveResult"]
