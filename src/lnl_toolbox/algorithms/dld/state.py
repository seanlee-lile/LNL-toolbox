from __future__ import annotations

"""Phase state owned by the DLD workflow."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class DLDPhase(str, Enum):
    FEATURE_EXTRACTION = "feature_extraction"
    PRECORRECTION_READY = "precorrection_ready"
    DIFFUSION_TRAINING = "diffusion_training"
    COMPLETED = "completed"


_NEXT = {
    DLDPhase.FEATURE_EXTRACTION: DLDPhase.PRECORRECTION_READY,
    DLDPhase.PRECORRECTION_READY: DLDPhase.DIFFUSION_TRAINING,
    DLDPhase.DIFFUSION_TRAINING: DLDPhase.COMPLETED,
}


@dataclass
class DLDState:
    phase: DLDPhase = DLDPhase.FEATURE_EXTRACTION
    completed_epochs: int = 0
    global_step: int = 0
    best_epoch: int = -1
    best_validation_accuracy: float = float("-inf")
    precorrection_artifact_hash: str = ""
    precorrection_file_sha256: str = ""

    def advance(self, target: DLDPhase) -> None:
        if _NEXT.get(self.phase) is not target:
            raise ValueError(f"illegal DLD phase transition: {self.phase.value} -> {target.value}")
        self.phase = target

    def state_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "completed_epochs": self.completed_epochs,
            "global_step": self.global_step,
            "best_epoch": self.best_epoch,
            "best_validation_accuracy": self.best_validation_accuracy,
            "precorrection_artifact_hash": self.precorrection_artifact_hash,
            "precorrection_file_sha256": self.precorrection_file_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DLDState":
        state = cls(
            phase=DLDPhase(str(value["phase"])),
            completed_epochs=int(value.get("completed_epochs", 0)),
            global_step=int(value.get("global_step", 0)),
            best_epoch=int(value.get("best_epoch", -1)),
            best_validation_accuracy=float(
                value.get("best_validation_accuracy", float("-inf"))
            ),
            precorrection_artifact_hash=str(
                value.get("precorrection_artifact_hash", "")
            ),
            precorrection_file_sha256=str(
                value.get("precorrection_file_sha256", "")
            ),
        )
        if state.completed_epochs < 0 or state.global_step < 0:
            raise ValueError("DLD progress counters must be non-negative")
        if state.phase is not DLDPhase.FEATURE_EXTRACTION and not state.precorrection_artifact_hash:
            raise ValueError("DLD ready state requires a pre-correction artifact hash")
        return state


__all__ = ["DLDPhase", "DLDState"]
