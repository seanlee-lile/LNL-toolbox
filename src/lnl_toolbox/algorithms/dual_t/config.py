from __future__ import annotations

"""Validated configuration for the first Dual-T + Forward implementation."""

from dataclasses import dataclass
from typing import Any, Mapping


def _mapping(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class DualTStageConfig:
    model: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    epochs: int

    @classmethod
    def from_mapping(
        cls, value: Any, *, owner: str
    ) -> "DualTStageConfig":
        config = _mapping(value, owner=owner)
        epochs = int(config.get("epochs", 0))
        if epochs <= 0:
            raise ValueError(f"{owner}.epochs must be positive")
        return cls(
            model=_mapping(config.get("model"), owner=f"{owner}.model"),
            optimizer=_mapping(
                config.get("optimizer"), owner=f"{owner}.optimizer"
            ),
            scheduler=_mapping(
                config.get("scheduler", {"name": "none"}),
                owner=f"{owner}.scheduler",
            ),
            epochs=epochs,
        )


@dataclass(frozen=True)
class DualTConfig:
    posterior_stage: DualTStageConfig
    transition_stage: Mapping[str, Any]
    final_stage: DualTStageConfig
    classifier_backend: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DualTConfig":
        method_value = value.get("method")
        if isinstance(method_value, Mapping):
            method_name = str(method_value.get("name", "")).strip().lower()
        else:
            method_name = str(method_value or "").strip().lower()
        if method_name == "dual_t_forward":
            raise ValueError("method 'dual_t_forward' was renamed to 'dual_t'")
        if method_name != "dual_t":
            raise ValueError("Dual-T runner requires method: dual_t")

        posterior_values = _mapping(
            value.get("posterior_stage"), owner="posterior_stage"
        )
        selection = str(
            posterior_values.get(
                "checkpoint_selection", "noisy_validation_accuracy"
            )
        ).strip().lower()
        if selection != "noisy_validation_accuracy":
            raise ValueError(
                "posterior_stage.checkpoint_selection must be "
                "'noisy_validation_accuracy'"
            )
        posterior_loss = _mapping(
            posterior_values.get("loss", {"name": "ce"}),
            owner="posterior_stage.loss",
        )
        if str(posterior_loss.get("name", "")).strip().lower() != "ce":
            raise ValueError("posterior_stage.loss must be cross entropy")

        transition = _mapping(
            value.get("transition_stage", {}) or {}, owner="transition_stage"
        )
        forbidden_transition_fields = sorted(
            set(transition) & {"estimator", "correction", "classifier"}
        )
        if forbidden_transition_fields:
            raise ValueError(
                "transition_stage contains internal method fields that are "
                f"fixed by method: dual_t: {forbidden_transition_fields}"
            )

        final_values = _mapping(value.get("final_stage"), owner="final_stage")
        if final_values.get("fresh_model", True) is not True:
            raise ValueError("final_stage.fresh_model must be true")
        classifier_backend = str(
            final_values.get("classifier", "forward")
        ).strip().lower()
        if classifier_backend != "forward":
            raise NotImplementedError(
                "Dual-T first version only supports classifier backend 'forward'"
            )
        final_loss = _mapping(
            final_values.get("loss", {"name": "ce"}),
            owner="final_stage.loss",
        )
        if str(final_loss.get("name", "")).strip().lower() != "ce":
            raise ValueError(
                "Dual-T Forward first version requires final_stage.loss: ce"
            )

        noise = _mapping(value.get("noise"), owner="noise")
        validation_targets = str(
            noise.get("validation_targets", "")
        ).strip().lower()
        if validation_targets != "noisy":
            raise ValueError(
                "Dual-T posterior selection requires "
                "noise.validation_targets: noisy"
            )
        evaluation = _mapping(
            value.get("evaluation", {"selection_split": "validation"}),
            owner="evaluation",
        )
        if str(
            evaluation.get("selection_split", "validation")
        ).strip().lower() != "validation":
            raise ValueError(
                "Dual-T checkpoint selection must use validation, not test"
            )

        return cls(
            posterior_stage=DualTStageConfig.from_mapping(
                posterior_values, owner="posterior_stage"
            ),
            transition_stage=transition,
            final_stage=DualTStageConfig.from_mapping(
                final_values, owner="final_stage"
            ),
            classifier_backend=classifier_backend,
        )
