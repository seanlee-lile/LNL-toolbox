from __future__ import annotations

"""Method-owned mutable state for the single-stage LEND lifecycle."""

from typing import Any, Mapping

from .history import LENDLabelHistory


class LENDState:
    def __init__(self, history: LENDLabelHistory) -> None:
        self.history = history
        self.optimizer_steps = 0
        self.empty_selection_batches = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "history": self.history.state_dict(),
            "optimizer_steps": self.optimizer_steps,
            "empty_selection_batches": self.empty_selection_batches,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != set(self.state_dict()):
            raise ValueError("LEND private state keys do not match")
        optimizer_steps = int(state["optimizer_steps"])
        empty_batches = int(state["empty_selection_batches"])
        if optimizer_steps < 0 or empty_batches < 0:
            raise ValueError("LEND step counters must be non-negative")
        self.history.load_state_dict(state["history"])
        self.optimizer_steps = optimizer_steps
        self.empty_selection_batches = empty_batches


__all__ = ["LENDState"]
