from __future__ import annotations

"""Serializable phase state for UPM."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping


class UPMPhase(str, Enum):
    STAGE1_TRAINING = "stage1_training"
    STAGE1_READY = "stage1_ready"
    PSI_READY = "psi_ready"
    MAIN_TRAINING = "main_training"
    COMPLETED = "completed"


_NEXT = {
    UPMPhase.STAGE1_TRAINING: UPMPhase.STAGE1_READY,
    UPMPhase.STAGE1_READY: UPMPhase.PSI_READY,
    UPMPhase.PSI_READY: UPMPhase.MAIN_TRAINING,
    UPMPhase.MAIN_TRAINING: UPMPhase.COMPLETED,
}


@dataclass
class UPMState:
    phase: UPMPhase = UPMPhase.STAGE1_TRAINING
    stage1_completed_epochs: int = 0
    stage1_global_step: int = 0
    stage1_best_epoch: int = -1
    stage1_best_validation_accuracy: float = float("-inf")
    stage1_best_checkpoint_sha256: str = ""
    psi_snapshot_hash: str = ""
    psi_file_sha256: str = ""
    main_completed_epochs: int = 0
    main_global_step: int = 0
    main_best_epoch: int = -1
    main_best_validation_accuracy: float = float("-inf")

    def advance(self, phase: UPMPhase) -> None:
        expected = _NEXT.get(self.phase)
        if phase is not expected:
            raise ValueError(
                f"illegal UPM phase transition: {self.phase.value} -> {phase.value}"
            )
        self.phase = phase

    def state_dict(self) -> dict[str, Any]:
        result = dict(vars(self))
        result["phase"] = self.phase.value
        return result

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> "UPMState":
        if not isinstance(value, Mapping):
            raise TypeError("UPM state must be a mapping")
        result = cls(
            phase=UPMPhase(str(value["phase"])),
            stage1_completed_epochs=int(value.get("stage1_completed_epochs", 0)),
            stage1_global_step=int(value.get("stage1_global_step", 0)),
            stage1_best_epoch=int(value.get("stage1_best_epoch", -1)),
            stage1_best_validation_accuracy=float(value.get("stage1_best_validation_accuracy", float("-inf"))),
            stage1_best_checkpoint_sha256=str(value.get("stage1_best_checkpoint_sha256", "")),
            psi_snapshot_hash=str(value.get("psi_snapshot_hash", "")),
            psi_file_sha256=str(value.get("psi_file_sha256", "")),
            main_completed_epochs=int(value.get("main_completed_epochs", 0)),
            main_global_step=int(value.get("main_global_step", 0)),
            main_best_epoch=int(value.get("main_best_epoch", -1)),
            main_best_validation_accuracy=float(value.get("main_best_validation_accuracy", float("-inf"))),
        )
        result.validate()
        return result

    def validate(self) -> None:
        counts = (
            self.stage1_completed_epochs, self.stage1_global_step,
            self.main_completed_epochs, self.main_global_step,
        )
        if any(value < 0 for value in counts):
            raise ValueError("UPM progress counters must be non-negative")
        if self.stage1_best_epoch >= self.stage1_completed_epochs:
            raise ValueError("UPM Stage 1 best epoch is inconsistent")
        if self.main_best_epoch >= self.main_completed_epochs:
            raise ValueError("UPM main best epoch is inconsistent")
        if self.phase is not UPMPhase.STAGE1_TRAINING:
            if self.stage1_best_epoch < 0 or len(self.stage1_best_checkpoint_sha256) != 64:
                raise ValueError("UPM Stage 1 ready state requires a best checkpoint")
            if not math.isfinite(self.stage1_best_validation_accuracy):
                raise ValueError("UPM Stage 1 best metric must be finite")
        if self.phase in {UPMPhase.PSI_READY, UPMPhase.MAIN_TRAINING, UPMPhase.COMPLETED}:
            if len(self.psi_snapshot_hash) != 64 or len(self.psi_file_sha256) != 64:
                raise ValueError("UPM psi-ready state requires snapshot identities")
        if self.phase is UPMPhase.COMPLETED and self.main_best_epoch < 0:
            raise ValueError("completed UPM state requires a main best checkpoint")


__all__ = ["UPMPhase", "UPMState"]
