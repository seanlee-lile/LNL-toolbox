from __future__ import annotations

"""Phase-aware state for the DivideMix two-network lifecycle."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class DivideMixPhase(str, Enum):
    WARMUP = "warmup"
    CO_DIVIDE_FITTING = "co_divide_fitting"
    CO_DIVIDE_READY = "co_divide_ready"
    TRAIN_NETWORK_A = "train_network_a"
    NETWORK_A_READY = "network_a_ready"
    TRAIN_NETWORK_B = "train_network_b"
    EPOCH_READY = "epoch_ready"
    COMPLETED = "completed"


@dataclass
class DivideMixState:
    phase: DivideMixPhase = DivideMixPhase.WARMUP
    warmup_completed_epochs: int = 0
    main_completed_epochs: int = 0
    optimizer_steps_a: int = 0
    optimizer_steps_b: int = 0
    current_artifact: str | None = None
    current_artifact_hash: str | None = None
    loss_history_a: list[dict[str, Any]] = field(default_factory=list)
    loss_history_b: list[dict[str, Any]] = field(default_factory=list)

    def transition(self, target: DivideMixPhase) -> None:
        allowed = {
            DivideMixPhase.WARMUP: {DivideMixPhase.WARMUP, DivideMixPhase.CO_DIVIDE_FITTING},
            DivideMixPhase.CO_DIVIDE_FITTING: {DivideMixPhase.CO_DIVIDE_READY},
            DivideMixPhase.CO_DIVIDE_READY: {DivideMixPhase.TRAIN_NETWORK_A},
            DivideMixPhase.TRAIN_NETWORK_A: {DivideMixPhase.NETWORK_A_READY},
            DivideMixPhase.NETWORK_A_READY: {DivideMixPhase.TRAIN_NETWORK_B},
            DivideMixPhase.TRAIN_NETWORK_B: {DivideMixPhase.EPOCH_READY},
            DivideMixPhase.EPOCH_READY: {DivideMixPhase.CO_DIVIDE_FITTING, DivideMixPhase.COMPLETED},
            DivideMixPhase.COMPLETED: {DivideMixPhase.COMPLETED},
        }
        if target not in allowed[self.phase]:
            raise ValueError(f"illegal DivideMix phase transition {self.phase.value} -> {target.value}")
        self.phase = target

    def state_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "warmup_completed_epochs": self.warmup_completed_epochs,
            "main_completed_epochs": self.main_completed_epochs,
            "optimizer_steps_a": self.optimizer_steps_a,
            "optimizer_steps_b": self.optimizer_steps_b,
            "current_artifact": self.current_artifact,
            "current_artifact_hash": self.current_artifact_hash,
            "loss_history_a": self.loss_history_a,
            "loss_history_b": self.loss_history_b,
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> "DivideMixState":
        if not isinstance(value, Mapping):
            raise TypeError("DivideMix state must be a mapping")
        state = cls(
            phase=DivideMixPhase(str(value["phase"])),
            warmup_completed_epochs=int(value["warmup_completed_epochs"]),
            main_completed_epochs=int(value["main_completed_epochs"]),
            optimizer_steps_a=int(value["optimizer_steps_a"]),
            optimizer_steps_b=int(value["optimizer_steps_b"]),
            current_artifact=value.get("current_artifact"),
            current_artifact_hash=value.get("current_artifact_hash"),
            loss_history_a=list(value.get("loss_history_a", [])),
            loss_history_b=list(value.get("loss_history_b", [])),
        )
        if min(state.warmup_completed_epochs, state.main_completed_epochs, state.optimizer_steps_a, state.optimizer_steps_b) < 0:
            raise ValueError("DivideMix state counters must be non-negative")
        return state
