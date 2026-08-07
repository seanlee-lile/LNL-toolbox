from __future__ import annotations

"""Serializable phase state for T-Revision Reweight-R."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


class TRevisionPhase(str, Enum):
    STAGE1_TRAINING = "stage1_training"
    STAGE1_READY = "stage1_ready"
    TRANSITION_INITIALIZED = "transition_initialized"
    CLASSIFIER_INITIALIZATION = "classifier_initialization"
    CLASSIFIER_READY = "classifier_ready"
    REVISION_TRAINING = "revision_training"
    COMPLETED = "completed"


_NEXT = {
    TRevisionPhase.STAGE1_TRAINING: TRevisionPhase.STAGE1_READY,
    TRevisionPhase.STAGE1_READY: TRevisionPhase.TRANSITION_INITIALIZED,
    TRevisionPhase.TRANSITION_INITIALIZED: TRevisionPhase.CLASSIFIER_INITIALIZATION,
    TRevisionPhase.CLASSIFIER_INITIALIZATION: TRevisionPhase.CLASSIFIER_READY,
    TRevisionPhase.CLASSIFIER_READY: TRevisionPhase.REVISION_TRAINING,
    TRevisionPhase.REVISION_TRAINING: TRevisionPhase.COMPLETED,
}


@dataclass
class TRevisionState:
    phase: TRevisionPhase = TRevisionPhase.STAGE1_TRAINING
    stage1_completed_epochs: int = 0
    stage1_global_step: int = 0
    stage1_best_epoch: int = -1
    stage1_best_metric: float = float("-inf")
    stage1_best_hash: str = ""
    snapshot_hash: str = ""
    initial_transition_hash: str = ""
    stage2a_completed_epochs: int = 0
    stage2a_global_step: int = 0
    stage2a_best_epoch: int = -1
    stage2a_best_metric: float = float("-inf")
    stage2a_best_hash: str = ""
    revision_completed_epochs: int = 0
    revision_global_step: int = 0
    revision_best_epoch: int = -1
    revision_best_metric: float = float("-inf")
    revision_best_checkpoint_hash: str = ""
    revised_transition_hash: str = ""

    def advance(self, phase: TRevisionPhase) -> None:
        expected = _NEXT.get(self.phase)
        if phase is not expected:
            raise ValueError(
                f"illegal T-Revision phase transition: {self.phase.value} -> "
                f"{phase.value}"
            )
        self.phase = phase

    def state_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["phase"] = self.phase.value
        return result

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> "TRevisionState":
        if not isinstance(value, Mapping):
            raise TypeError("T-Revision state must be a mapping")
        fields = dict(value)
        fields["phase"] = TRevisionPhase(str(fields["phase"]))
        result = cls(**fields)
        result.validate()
        return result

    def validate(self) -> None:
        counters = (
            self.stage1_completed_epochs,
            self.stage1_global_step,
            self.stage2a_completed_epochs,
            self.stage2a_global_step,
            self.revision_completed_epochs,
            self.revision_global_step,
        )
        if any(value < 0 for value in counters):
            raise ValueError("T-Revision progress counters must be non-negative")
        for epoch, completed, owner in (
            (self.stage1_best_epoch, self.stage1_completed_epochs, "stage1"),
            (self.stage2a_best_epoch, self.stage2a_completed_epochs, "stage2a"),
            (self.revision_best_epoch, self.revision_completed_epochs, "revision"),
        ):
            if epoch >= completed:
                raise ValueError(f"{owner} best epoch must precede completed epochs")
        if self.phase is not TRevisionPhase.STAGE1_TRAINING:
            if self.stage1_best_epoch < 0 or len(self.stage1_best_hash) != 64:
                raise ValueError("stage1-ready state requires a valid best checkpoint")
        if self.phase in set(TRevisionPhase) - {
            TRevisionPhase.STAGE1_TRAINING,
            TRevisionPhase.STAGE1_READY,
        }:
            if len(self.snapshot_hash) != 64 or len(self.initial_transition_hash) != 64:
                raise ValueError("transition-initialized state requires artifact hashes")
        if self.phase in {
            TRevisionPhase.CLASSIFIER_READY,
            TRevisionPhase.REVISION_TRAINING,
            TRevisionPhase.COMPLETED,
        }:
            if self.stage2a_best_epoch < 0 or len(self.stage2a_best_hash) != 64:
                raise ValueError("classifier-ready state requires stage2a best")
        if self.phase is TRevisionPhase.COMPLETED:
            if (
                self.revision_best_epoch < 0
                or len(self.revision_best_checkpoint_hash) != 64
                or len(self.revised_transition_hash) != 64
            ):
                raise ValueError("completed state requires revision best artifacts")
