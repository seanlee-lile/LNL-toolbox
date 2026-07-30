from __future__ import annotations

"""Phase-aware serializable state for PCSE."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping


class PCSEPhase(str, Enum):
    PRETRAINING = "pretraining"
    PRETRAINED = "pretrained"
    TRANSITION_TRAINING = "transition_training"
    TRANSITION_READY = "transition_ready"
    STATISTICS_READY = "statistics_ready"
    GDA_READY = "gda_ready"
    ENSEMBLE_TRAINING = "ensemble_training"
    COMPLETED = "completed"


_ALLOWED = {
    PCSEPhase.PRETRAINING: {PCSEPhase.PRETRAINED},
    PCSEPhase.PRETRAINED: {
        PCSEPhase.TRANSITION_TRAINING,
        PCSEPhase.TRANSITION_READY,
    },
    PCSEPhase.TRANSITION_TRAINING: {PCSEPhase.TRANSITION_READY},
    PCSEPhase.TRANSITION_READY: {PCSEPhase.STATISTICS_READY},
    PCSEPhase.STATISTICS_READY: {PCSEPhase.GDA_READY},
    PCSEPhase.GDA_READY: {PCSEPhase.ENSEMBLE_TRAINING},
    PCSEPhase.ENSEMBLE_TRAINING: {PCSEPhase.COMPLETED},
}


@dataclass
class PCSEState:
    phase: PCSEPhase = PCSEPhase.PRETRAINING
    pretraining_completed_epochs: int = 0
    pretraining_global_step: int = 0
    best_pretraining_epoch: int = -1
    best_pretraining_validation_accuracy: float = float("-inf")
    best_pretraining_validation_loss: float = float("inf")
    pretrained_checkpoint_sha256: str = ""
    posterior_snapshot_hash: str = ""
    transition_completed_epochs: int = 0
    transition_global_step: int = 0
    volmin_final_checkpoint_sha256: str = ""
    feature_model_checkpoint_sha256: str = ""
    transition_artifact_hash: str = ""
    statistics_artifact_hash: str = ""
    gda_artifact_hash: str = ""
    ensemble_completed_epochs: int = 0
    ensemble_global_step: int = 0
    ensemble_artifact_hash: str = ""

    def advance(self, phase: PCSEPhase) -> None:
        allowed = _ALLOWED.get(self.phase, set())
        if phase not in allowed:
            raise ValueError(
                f"illegal PCSE phase transition: {self.phase.value} -> "
                f"{phase.value}"
            )
        self.phase = phase

    def state_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "pretraining_completed_epochs": self.pretraining_completed_epochs,
            "pretraining_global_step": self.pretraining_global_step,
            "best_pretraining_epoch": self.best_pretraining_epoch,
            "best_pretraining_validation_accuracy": (
                self.best_pretraining_validation_accuracy
            ),
            "best_pretraining_validation_loss": (
                self.best_pretraining_validation_loss
            ),
            "pretrained_checkpoint_sha256": self.pretrained_checkpoint_sha256,
            "posterior_snapshot_hash": self.posterior_snapshot_hash,
            "transition_completed_epochs": self.transition_completed_epochs,
            "transition_global_step": self.transition_global_step,
            "volmin_final_checkpoint_sha256": (
                self.volmin_final_checkpoint_sha256
            ),
            "feature_model_checkpoint_sha256": (
                self.feature_model_checkpoint_sha256
            ),
            "transition_artifact_hash": self.transition_artifact_hash,
            "statistics_artifact_hash": self.statistics_artifact_hash,
            "gda_artifact_hash": self.gda_artifact_hash,
            "ensemble_completed_epochs": self.ensemble_completed_epochs,
            "ensemble_global_step": self.ensemble_global_step,
            "ensemble_artifact_hash": self.ensemble_artifact_hash,
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> "PCSEState":
        if not isinstance(value, Mapping):
            raise TypeError("PCSE state must be a mapping")
        result = cls(
            phase=PCSEPhase(str(value["phase"])),
            pretraining_completed_epochs=int(
                value.get("pretraining_completed_epochs", 0)
            ),
            pretraining_global_step=int(value.get("pretraining_global_step", 0)),
            best_pretraining_epoch=int(value.get("best_pretraining_epoch", -1)),
            best_pretraining_validation_accuracy=float(
                value.get(
                    "best_pretraining_validation_accuracy", float("-inf")
                )
            ),
            best_pretraining_validation_loss=float(
                value.get("best_pretraining_validation_loss", float("inf"))
            ),
            pretrained_checkpoint_sha256=str(
                value.get("pretrained_checkpoint_sha256", "")
            ),
            posterior_snapshot_hash=str(
                value.get("posterior_snapshot_hash", "")
            ),
            transition_completed_epochs=int(
                value.get("transition_completed_epochs", 0)
            ),
            transition_global_step=int(
                value.get("transition_global_step", 0)
            ),
            volmin_final_checkpoint_sha256=str(
                value.get("volmin_final_checkpoint_sha256", "")
            ),
            feature_model_checkpoint_sha256=str(
                value.get(
                    "feature_model_checkpoint_sha256",
                    value.get("pretrained_checkpoint_sha256", ""),
                )
            ),
            transition_artifact_hash=str(
                value.get("transition_artifact_hash", "")
            ),
            statistics_artifact_hash=str(
                value.get("statistics_artifact_hash", "")
            ),
            gda_artifact_hash=str(value.get("gda_artifact_hash", "")),
            ensemble_completed_epochs=int(
                value.get("ensemble_completed_epochs", 0)
            ),
            ensemble_global_step=int(value.get("ensemble_global_step", 0)),
            ensemble_artifact_hash=str(
                value.get("ensemble_artifact_hash", "")
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if min(
            self.pretraining_completed_epochs,
            self.pretraining_global_step,
            self.transition_completed_epochs,
            self.transition_global_step,
            self.ensemble_completed_epochs,
            self.ensemble_global_step,
        ) < 0:
            raise ValueError("PCSE progress counters must be non-negative")
        if self.best_pretraining_epoch >= self.pretraining_completed_epochs:
            raise ValueError(
                "best PCSE pretraining epoch must precede completed epochs"
            )
        required: list[tuple[PCSEPhase, str, str]] = [
            (PCSEPhase.PRETRAINED, "pretrained checkpoint", self.pretrained_checkpoint_sha256),
            (PCSEPhase.TRANSITION_READY, "feature-model checkpoint", self.feature_model_checkpoint_sha256),
            (PCSEPhase.TRANSITION_READY, "transition artifact", self.transition_artifact_hash),
            (PCSEPhase.STATISTICS_READY, "statistics artifact", self.statistics_artifact_hash),
            (PCSEPhase.GDA_READY, "GDA artifact", self.gda_artifact_hash),
        ]
        order = list(PCSEPhase)
        current = order.index(self.phase)
        if current >= order.index(PCSEPhase.PRETRAINED):
            if (
                self.pretraining_completed_epochs <= 0
                or self.best_pretraining_epoch < 0
                or not math.isfinite(
                    self.best_pretraining_validation_accuracy
                )
                or not math.isfinite(self.best_pretraining_validation_loss)
            ):
                raise ValueError(
                    "pretrained PCSE state requires finite best-checkpoint "
                    "metrics and completed progress"
                )
        for phase, owner, digest in required:
            if current >= order.index(phase) and len(digest) != 64:
                raise ValueError(f"PCSE {phase.value} requires {owner} hash")
        if (
            self.phase is PCSEPhase.COMPLETED
            and len(self.ensemble_artifact_hash) != 64
        ):
            raise ValueError("completed PCSE requires ensemble artifact hash")
