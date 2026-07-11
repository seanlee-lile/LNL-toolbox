from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RunState:
    """Framework state only; method-specific state belongs to the algorithm."""

    cycle: int = 0
    step: int = 0
    phase: str = "default"
    stopped: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

