from __future__ import annotations

"""Validated configuration for the T-Revision Reweight-R workflow."""

from dataclasses import dataclass
import math
from typing import Any, Mapping


def _mapping(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    return dict(value)


def _named(value: Mapping[str, Any], expected: str, *, owner: str) -> None:
    actual = str(value.get("name", "")).strip().lower()
    if actual != expected:
        raise ValueError(f"{owner}.name must be {expected!r}")


@dataclass(frozen=True)
class TRevisionTrainStageConfig:
    epochs: int
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls, value: Any, *, owner: str
    ) -> "TRevisionTrainStageConfig":
        config = _mapping(value, owner=owner)
        epochs = int(config.get("epochs", 0))
        if epochs <= 0:
            raise ValueError(f"{owner}.epochs must be positive")
        return cls(
            epochs=epochs,
            optimizer=_mapping(config.get("optimizer"), owner=f"{owner}.optimizer"),
            scheduler=_mapping(
                config.get("scheduler", {"name": "none"}),
                owner=f"{owner}.scheduler",
            ),
        )


@dataclass(frozen=True)
class TRevisionConfig:
    model: Mapping[str, Any]
    stage1: TRevisionTrainStageConfig
    transition_initialization: Mapping[str, Any]
    classifier_initialization: TRevisionTrainStageConfig
    revision: TRevisionTrainStageConfig
    denominator_floor: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TRevisionConfig":
        method = value.get("method")
        method_name = (
            str(method.get("name", "")).strip().lower()
            if isinstance(method, Mapping)
            else str(method or "").strip().lower()
        )
        if method_name != "t_revision":
            raise ValueError("T-Revision runner requires method: t_revision")
        config = _mapping(value.get("t_revision"), owner="t_revision")
        if str(config.get("objective", "")).strip().lower() != "reweight":
            raise NotImplementedError(
                "T-Revision first version only supports objective: reweight"
            )
        fidelity = str(config.get("fidelity", "")).strip().lower()
        if fidelity != "paper_experiment_raw_additive":
            raise NotImplementedError(
                "T-Revision first version requires fidelity: "
                "paper_experiment_raw_additive"
            )

        stage1_values = _mapping(config.get("stage1"), owner="t_revision.stage1")
        model = _mapping(stage1_values.get("model"), owner="t_revision.stage1.model")
        if str(stage1_values.get("best_metric", "")).strip().lower() != (
            "noisy_validation_accuracy"
        ):
            raise ValueError(
                "t_revision.stage1.best_metric must be noisy_validation_accuracy"
            )

        initializer = _mapping(
            config.get("transition_initialization"),
            owner="t_revision.transition_initialization",
        )
        expected_initializer = {
            "method": "pseudo_anchor_max_posterior",
            "posterior_split": "train",
            "extraction_augmentation": False,
            "tie_break": "stable_sample_index",
        }
        for name, expected in expected_initializer.items():
            if initializer.get(name) != expected:
                raise ValueError(
                    f"t_revision.transition_initialization.{name} must be "
                    f"{expected!r}"
                )

        classifier_values = _mapping(
            config.get("classifier_initialization"),
            owner="t_revision.classifier_initialization",
        )
        if classifier_values.get("start_from") != "stage1_best":
            raise ValueError(
                "classifier_initialization.start_from must be stage1_best"
            )
        if str(classifier_values.get("best_metric", "")).strip().lower() != (
            "revised_noisy_validation_accuracy"
        ):
            raise ValueError(
                "classifier_initialization.best_metric must be "
                "revised_noisy_validation_accuracy"
            )
        _named(
            _mapping(
                classifier_values.get("optimizer"),
                owner="classifier_initialization.optimizer",
            ),
            "sgd",
            owner="classifier_initialization.optimizer",
        )

        revision_values = _mapping(
            config.get("revision"), owner="t_revision.revision"
        )
        required_revision = {
            "transition_mode": "paper_experiment_raw_additive",
            "delta_initialization": "zeros",
            "start_from": "classifier_initialization_best",
            "best_metric": "revised_noisy_validation_accuracy",
        }
        for name, expected in required_revision.items():
            if revision_values.get(name) != expected:
                raise ValueError(f"t_revision.revision.{name} must be {expected!r}")
        _named(
            _mapping(revision_values.get("optimizer"), owner="revision.optimizer"),
            "adam",
            owner="revision.optimizer",
        )
        ratio = _mapping(revision_values.get("ratio"), owner="revision.ratio")
        if ratio.get("detach") is not False:
            raise ValueError("revision.ratio.detach must be false")
        if ratio.get("clamp") != "none":
            raise ValueError("revision.ratio.clamp must be none")
        denominator_floor = float(ratio.get("denominator_floor", 0.0))
        if not math.isfinite(denominator_floor) or denominator_floor < 0.0:
            raise ValueError(
                "revision.ratio.denominator_floor must be finite and non-negative"
            )

        noise = _mapping(value.get("noise"), owner="noise")
        noise_name = str(noise.get("name", "")).strip().lower()
        has_manifest = bool(str(noise.get("manifest", "")).strip())
        if not has_manifest and noise_name not in {"symmetric", "pairflip"}:
            raise ValueError(
                "T-Revision requires class-dependent, instance-independent "
                "noise (symmetric, pairflip, or an external manifest)"
            )
        if str(noise.get("validation_targets", "")).strip().lower() != "noisy":
            raise ValueError(
                "T-Revision checkpoint selection requires "
                "noise.validation_targets: noisy"
            )
        evaluation = _mapping(
            value.get("evaluation", {"selection_split": "validation"}),
            owner="evaluation",
        )
        if str(evaluation.get("selection_split", "")).strip().lower() != (
            "validation"
        ):
            raise ValueError("T-Revision best selection must use validation")

        data = _mapping(value.get("data"), owner="data")
        validation_size = int(data.get("validation_size", 0))
        if validation_size <= 0:
            raise ValueError("T-Revision requires data.validation_size > 0")
        _mapping(value.get("loader"), owner="loader")
        trainer = _mapping(value.get("trainer"), owner="trainer")
        if not str(trainer.get("device", "")).strip():
            raise ValueError("T-Revision requires trainer.device")

        return cls(
            model=model,
            stage1=TRevisionTrainStageConfig.from_mapping(
                stage1_values, owner="t_revision.stage1"
            ),
            transition_initialization=initializer,
            classifier_initialization=TRevisionTrainStageConfig.from_mapping(
                classifier_values, owner="t_revision.classifier_initialization"
            ),
            revision=TRevisionTrainStageConfig.from_mapping(
                revision_values, owner="t_revision.revision"
            ),
            denominator_floor=denominator_floor,
        )
