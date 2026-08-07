from __future__ import annotations

"""Phase state for the paper-specific Dual-T + Forward lifecycle."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class DualTPhase(str, Enum):
    POSTERIOR_TRAINING = "posterior_training"
    POSTERIOR_READY = "posterior_ready"
    TRANSITION_READY = "transition_ready"
    FINAL_TRAINING = "final_training"
    COMPLETED = "completed"


_NEXT_PHASE = {
    DualTPhase.POSTERIOR_TRAINING: DualTPhase.POSTERIOR_READY,
    DualTPhase.POSTERIOR_READY: DualTPhase.TRANSITION_READY,
    DualTPhase.TRANSITION_READY: DualTPhase.FINAL_TRAINING,
    DualTPhase.FINAL_TRAINING: DualTPhase.COMPLETED,
}


@dataclass
class DualTState:
    """Serializable progress and artifact identities for both training stages."""

    phase: DualTPhase = DualTPhase.POSTERIOR_TRAINING
    posterior_completed_epochs: int = 0
    posterior_global_step: int = 0
    best_posterior_epoch: int = -1
    best_posterior_validation_accuracy: float = float("-inf")
    best_posterior_checkpoint_sha256: str = ""
    posterior_snapshot_hash: str = ""
    transition_artifact_hash: str = ""
    final_completed_epochs: int = 0
    final_global_step: int = 0
    best_final_epoch: int = -1
    best_final_validation_accuracy: float = float("-inf")

    def advance(self, phase: DualTPhase) -> None:
        expected = _NEXT_PHASE.get(self.phase)
        if phase is not expected:
            raise ValueError(
                f"illegal Dual-T phase transition: {self.phase.value} -> "
                f"{phase.value}; expected "
                f"{None if expected is None else expected.value}"
            )
        self.phase = phase

    def state_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "posterior_completed_epochs": self.posterior_completed_epochs,
            "posterior_global_step": self.posterior_global_step,
            "best_posterior_epoch": self.best_posterior_epoch,
            "best_posterior_validation_accuracy": (
                self.best_posterior_validation_accuracy
            ),
            "best_posterior_checkpoint_sha256": (
                self.best_posterior_checkpoint_sha256
            ),
            "posterior_snapshot_hash": self.posterior_snapshot_hash,
            "transition_artifact_hash": self.transition_artifact_hash,
            "final_completed_epochs": self.final_completed_epochs,
            "final_global_step": self.final_global_step,
            "best_final_epoch": self.best_final_epoch,
            "best_final_validation_accuracy": (
                self.best_final_validation_accuracy
            ),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "DualTState":
        if not isinstance(state, Mapping):
            raise TypeError("Dual-T state must be a mapping")
        result = cls(
            phase=DualTPhase(str(state["phase"])),
            posterior_completed_epochs=int(
                state.get("posterior_completed_epochs", 0)
            ),
            posterior_global_step=int(state.get("posterior_global_step", 0)),
            best_posterior_epoch=int(state.get("best_posterior_epoch", -1)),
            best_posterior_validation_accuracy=float(
                state.get("best_posterior_validation_accuracy", float("-inf"))
            ),
            best_posterior_checkpoint_sha256=str(
                state.get("best_posterior_checkpoint_sha256", "")
            ),
            posterior_snapshot_hash=str(
                state.get("posterior_snapshot_hash", "")
            ),
            transition_artifact_hash=str(
                state.get("transition_artifact_hash", "")
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
        counters = (
            self.posterior_completed_epochs,
            self.posterior_global_step,
            self.final_completed_epochs,
            self.final_global_step,
        )
        if any(value < 0 for value in counters):
            raise ValueError("Dual-T progress counters must be non-negative")
        if self.best_posterior_epoch >= self.posterior_completed_epochs:
            raise ValueError(
                "best posterior epoch must precede completed posterior epochs"
            )
        if self.best_final_epoch >= self.final_completed_epochs:
            raise ValueError("best final epoch must precede completed final epochs")
        if self.phase is not DualTPhase.POSTERIOR_TRAINING:
            if self.best_posterior_epoch < 0:
                raise ValueError(
                    "completed posterior training requires a best checkpoint"
                )
            if len(self.best_posterior_checkpoint_sha256) != 64:
                raise ValueError(
                    "completed posterior training requires a checkpoint SHA-256"
                )
        if self.phase in {
            DualTPhase.TRANSITION_READY,
            DualTPhase.FINAL_TRAINING,
            DualTPhase.COMPLETED,
        }:
            if len(self.posterior_snapshot_hash) != 64:
                raise ValueError(
                    "transition-ready state requires a posterior snapshot hash"
                )
            if len(self.transition_artifact_hash) != 64:
                raise ValueError(
                    "transition-ready state requires a transition artifact hash"
                )
        if self.phase is DualTPhase.COMPLETED and self.best_final_epoch < 0:
            raise ValueError("completed Dual-T training requires a final best model")
