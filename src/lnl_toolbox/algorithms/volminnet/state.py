from __future__ import annotations

"""Method-private progress state for single-stage VolMinNet."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass
class VolMinNetState:
    completed_epochs: int = 0
    global_step: int = 0
    best_epoch: int = -1
    best_validation_loss: float = float("inf")
    best_validation_accuracy: float = float("-inf")
    classifier_optimizer_steps: int = 0
    transition_optimizer_steps: int = 0
    completed: bool = False

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> "VolMinNetState":
        if not isinstance(value, Mapping):
            raise TypeError("VolMinNet state must be a mapping")
        state = cls(**dict(value))
        if state.completed_epochs < 0 or state.global_step < 0:
            raise ValueError("VolMinNet progress must be non-negative")
        if state.classifier_optimizer_steps != state.global_step:
            raise ValueError("VolMinNet classifier optimizer-step count mismatch")
        if state.transition_optimizer_steps != state.global_step:
            raise ValueError("VolMinNet transition optimizer-step count mismatch")
        return state
