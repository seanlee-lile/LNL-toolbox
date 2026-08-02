from __future__ import annotations

"""Method-owned runtime state for Co-teaching."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class CoTeachingState:
    optimizer_steps_a: int = 0
    optimizer_steps_b: int = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "optimizer_steps_a": int(self.optimizer_steps_a),
            "optimizer_steps_b": int(self.optimizer_steps_b),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("Co-teaching private state must be a mapping")
        if set(state) != {"optimizer_steps_a", "optimizer_steps_b"}:
            raise ValueError("Co-teaching private state keys do not match")
        steps_a = int(state["optimizer_steps_a"])
        steps_b = int(state["optimizer_steps_b"])
        if steps_a < 0 or steps_b < 0 or steps_a != steps_b:
            raise ValueError("Co-teaching optimizer step counts must match and be non-negative")
        self.optimizer_steps_a = steps_a
        self.optimizer_steps_b = steps_b
