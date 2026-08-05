from __future__ import annotations

"""Method-owned state for CNLCU-S."""

from typing import Any, Mapping

from .history import PeerLossHistory


class CNLCUState:
    def __init__(self, history_a: PeerLossHistory, history_b: PeerLossHistory) -> None:
        if history_a is history_b:
            raise ValueError("CNLCU peer histories must be distinct")
        self.history_a = history_a
        self.history_b = history_b
        self.optimizer_steps_a = 0
        self.optimizer_steps_b = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "history_a": self.history_a.state_dict(),
            "history_b": self.history_b.state_dict(),
            "optimizer_steps_a": self.optimizer_steps_a,
            "optimizer_steps_b": self.optimizer_steps_b,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != set(self.state_dict()):
            raise ValueError("CNLCU private state keys do not match")
        steps_a, steps_b = int(state["optimizer_steps_a"]), int(state["optimizer_steps_b"])
        if steps_a < 0 or steps_a != steps_b:
            raise ValueError("CNLCU optimizer step counts must match and be non-negative")
        self.history_a.load_state_dict(state["history_a"])
        self.history_b.load_state_dict(state["history_b"])
        self.optimizer_steps_a, self.optimizer_steps_b = steps_a, steps_b


__all__ = ["CNLCUState"]
