from __future__ import annotations

"""Serializable phase state for binary importance reweighting."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class ImportanceReweightingPhase(str, Enum):
    POSTERIOR_FITTING = "posterior_fitting"
    POSTERIOR_READY = "posterior_ready"
    RATE_READY = "rate_ready"
    FINAL_TRAINING = "final_training"
    COMPLETED = "completed"


_NEXT = {
    ImportanceReweightingPhase.POSTERIOR_FITTING:
        ImportanceReweightingPhase.POSTERIOR_READY,
    ImportanceReweightingPhase.POSTERIOR_READY:
        ImportanceReweightingPhase.RATE_READY,
    ImportanceReweightingPhase.RATE_READY:
        ImportanceReweightingPhase.FINAL_TRAINING,
    ImportanceReweightingPhase.FINAL_TRAINING:
        ImportanceReweightingPhase.COMPLETED,
}


@dataclass
class ImportanceReweightingState:
    phase: ImportanceReweightingPhase = (
        ImportanceReweightingPhase.POSTERIOR_FITTING
    )
    posterior_snapshot_hash: str = ""
    posterior_backend_hash: str = ""
    noise_rate_artifact_hash: str = ""
    final_completed_epochs: int = 0
    final_global_step: int = 0
    best_final_epoch: int = -1
    best_final_validation_accuracy: float = float("-inf")

    def advance(self, phase: ImportanceReweightingPhase) -> None:
        expected = _NEXT.get(self.phase)
        if phase is not expected:
            raise ValueError(
                "illegal importance reweighting phase transition: "
                f"{self.phase.value} -> {phase.value}; expected "
                f"{None if expected is None else expected.value}"
            )
        self.phase = phase

    def reopen_final_training(self, target_epochs: int) -> None:
        """Reopen a completed run only to extend its final training budget."""

        if self.phase is not ImportanceReweightingPhase.COMPLETED:
            raise ValueError("only a completed run can reopen final training")
        if int(target_epochs) <= self.final_completed_epochs:
            raise ValueError("extended epoch budget must exceed completed epochs")
        self.phase = ImportanceReweightingPhase.FINAL_TRAINING

    def state_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "posterior_snapshot_hash": self.posterior_snapshot_hash,
            "posterior_backend_hash": self.posterior_backend_hash,
            "noise_rate_artifact_hash": self.noise_rate_artifact_hash,
            "final_completed_epochs": self.final_completed_epochs,
            "final_global_step": self.final_global_step,
            "best_final_epoch": self.best_final_epoch,
            "best_final_validation_accuracy": (
                self.best_final_validation_accuracy
            ),
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any]
    ) -> "ImportanceReweightingState":
        if not isinstance(state, Mapping):
            raise TypeError("importance reweighting state must be a mapping")
        result = cls(
            phase=ImportanceReweightingPhase(str(state["phase"])),
            posterior_snapshot_hash=str(
                state.get("posterior_snapshot_hash", "")
            ),
            posterior_backend_hash=str(
                state.get("posterior_backend_hash", "")
            ),
            noise_rate_artifact_hash=str(
                state.get("noise_rate_artifact_hash", "")
            ),
            final_completed_epochs=int(state.get("final_completed_epochs", 0)),
            final_global_step=int(state.get("final_global_step", 0)),
            best_final_epoch=int(state.get("best_final_epoch", -1)),
            best_final_validation_accuracy=float(
                state.get("best_final_validation_accuracy", float("-inf"))
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.final_completed_epochs < 0 or self.final_global_step < 0:
            raise ValueError("importance reweighting progress must be non-negative")
        if (
            self.posterior_backend_hash
            and len(self.posterior_backend_hash) != 64
        ):
            raise ValueError("posterior backend hash must contain 64 hex characters")
        if self.best_final_epoch >= self.final_completed_epochs:
            raise ValueError("best final epoch must precede completed final epochs")
        if self.phase is not ImportanceReweightingPhase.POSTERIOR_FITTING:
            if len(self.posterior_snapshot_hash) != 64:
                raise ValueError(
                    "posterior-ready state requires a posterior snapshot hash"
                )
        if self.phase in {
            ImportanceReweightingPhase.RATE_READY,
            ImportanceReweightingPhase.FINAL_TRAINING,
            ImportanceReweightingPhase.COMPLETED,
        } and len(self.noise_rate_artifact_hash) != 64:
            raise ValueError("rate-ready state requires a noise-rate artifact hash")
        if (
            self.phase is ImportanceReweightingPhase.COMPLETED
            and self.best_final_epoch < 0
        ):
            raise ValueError(
                "completed importance reweighting requires a final best model"
            )
