from __future__ import annotations

"""Framework-neutral contracts for composable training objectives."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ObjectiveResult:
    """Structured objective plus optional sample accounting and diagnostics."""

    objective: Any
    selected_mask: Any | None = None
    reporting_loss: Any | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class ObjectiveConsumer(Protocol):
    """Build a scalar or structured objective from shared training inputs."""

    def compute(
        self,
        *,
        model: Any,
        logits: Any,
        features: Any,
        noisy_targets: Any,
        sample_indices: Any,
        base_loss: Any,
        metadata: Mapping[str, Any],
    ) -> Any | ObjectiveResult:
        ...


__all__ = ["ObjectiveConsumer", "ObjectiveResult"]
