from __future__ import annotations

"""Validated configuration for the two-stage UPM workflow."""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping


def _mapping(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class UPMStageConfig:
    epochs: int
    model: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any, *, owner: str) -> "UPMStageConfig":
        config = _mapping(value, owner=owner)
        epochs = int(config.get("epochs", 0))
        if epochs <= 0:
            raise ValueError(f"{owner}.epochs must be positive")
        return cls(
            epochs=epochs,
            model=_mapping(config.get("model"), owner=f"{owner}.model"),
            optimizer=_mapping(
                config.get("optimizer"), owner=f"{owner}.optimizer"
            ),
            scheduler=_mapping(
                config.get("scheduler", {"name": "none"}),
                owner=f"{owner}.scheduler",
            ),
        )


@dataclass(frozen=True)
class UPMConfusingConfig:
    initial_value: float
    learning_rate: float
    epsilon: float
    update_start_epoch: int
    update_interval_epochs: int

    @classmethod
    def from_mapping(cls, value: Any) -> "UPMConfusingConfig":
        config = _mapping(value, owner="upm.confusing_probability")
        initial = float(config.get("initial_value", 0.01))
        learning_rate = float(config.get("learning_rate", 0.0))
        epsilon = float(config.get("epsilon", 1e-4))
        start = int(config.get("update_start_epoch", 0))
        interval = int(config.get("update_interval_epochs", 1))
        if not math.isfinite(initial) or not 0.0 <= initial <= 1.0:
            raise ValueError("UPM eta initial_value must be finite and in [0, 1]")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("UPM eta learning_rate must be finite and positive")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("UPM eta epsilon must be finite and positive")
        if start < 0:
            raise ValueError("UPM eta update_start_epoch must be non-negative")
        if interval <= 0:
            raise ValueError("UPM eta update_interval_epochs must be positive")
        return cls(initial, learning_rate, epsilon, start, interval)

    def updates_at(self, epoch: int) -> bool:
        return epoch >= self.update_start_epoch and (
            epoch - self.update_start_epoch
        ) % self.update_interval_epochs == 0


@dataclass(frozen=True)
class UPMConfig:
    stage1: UPMStageConfig
    main: UPMStageConfig
    confusing_probability: UPMConfusingConfig
    psi: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UPMConfig":
        method = value.get("method", "")
        if isinstance(method, Mapping):
            method = method.get("name", "")
        if str(method).strip().lower() != "upm":
            raise ValueError("UPM runner requires method: upm")
        execution = _mapping(value.get("execution", {}), owner="execution")
        if str(execution.get("runner", "upm")).strip().lower() != "upm":
            raise ValueError("UPM requires execution.runner: upm")
        upm = _mapping(value.get("upm"), owner="upm")
        stage1_values = _mapping(upm.get("stage1"), owner="upm.stage1")
        if str(stage1_values.get("best_metric", "noisy_validation_accuracy")).lower() != "noisy_validation_accuracy":
            raise ValueError("UPM Stage 1 must select noisy_validation_accuracy")
        psi = _mapping(upm.get("psi"), owner="upm.psi")
        checks = {
            "source": str(psi.get("source", "")).lower() == "stage1_best",
            "split": str(psi.get("split", "")).lower() == "train",
            "augmentation": psi.get("augmentation", True) is False,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError("UPM psi contract is invalid: " + ", ".join(failed))
        main_values = _mapping(upm.get("main"), owner="upm.main")
        if str(main_values.get("initialization", "fresh")).lower() != "fresh":
            raise ValueError("UPM Stage 2 main.initialization must be fresh")
        noise = _mapping(value.get("noise"), owner="noise")
        if str(noise.get("validation_targets", "")).lower() != "noisy":
            raise ValueError("UPM best selection requires noise.validation_targets: noisy")
        evaluation = _mapping(value.get("evaluation", {}), owner="evaluation")
        if str(evaluation.get("selection_split", "validation")).lower() != "validation":
            raise ValueError("UPM checkpoint selection must use validation, not test")
        return cls(
            UPMStageConfig.from_mapping(stage1_values, owner="upm.stage1"),
            UPMStageConfig.from_mapping(main_values, owner="upm.main"),
            UPMConfusingConfig.from_mapping(upm.get("confusing_probability")),
            psi,
        )

    @property
    def identity_hash(self) -> str:
        payload = {
            "stage1": {
                "model": dict(self.stage1.model),
                "optimizer": dict(self.stage1.optimizer),
                "scheduler": dict(self.stage1.scheduler),
            },
            "main": {
                "model": dict(self.main.model),
                "optimizer": dict(self.main.optimizer),
                "scheduler": dict(self.main.scheduler),
            },
            "psi": dict(self.psi),
            "confusing_probability": {
                "initial_value": self.confusing_probability.initial_value,
                "learning_rate": self.confusing_probability.learning_rate,
                "epsilon": self.confusing_probability.epsilon,
                "update_start_epoch": self.confusing_probability.update_start_epoch,
                "update_interval_epochs": self.confusing_probability.update_interval_epochs,
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


__all__ = ["UPMConfig", "UPMConfusingConfig", "UPMStageConfig"]
