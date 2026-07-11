from __future__ import annotations

from typing import Protocol

from .result import StepResult
from .state import RunState


class Evaluator(Protocol):
    """Consumes results without assuming classification or a particular metric set."""

    def update(self, result: StepResult, state: RunState) -> None: ...
    def compute(self, state: RunState) -> dict[str, float]: ...
    def reset(self) -> None: ...

