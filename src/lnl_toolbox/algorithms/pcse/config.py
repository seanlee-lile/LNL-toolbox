from __future__ import annotations

"""Validated configuration for the first PCSE method-level workflow."""

from dataclasses import dataclass
import math
from typing import Any, Mapping


def _mapping(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    return dict(value)


def _positive_float(value: Any, *, owner: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{owner} must be finite and positive")
    return result


def _exact_keys(
    value: Mapping[str, Any],
    *,
    owner: str,
    allowed: set[str],
) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(f"{owner} has unsupported fields: {unexpected}")


@dataclass(frozen=True)
class PCSEFeatureLayerConfig:
    name: str
    pooling: str = "global_average"

    @classmethod
    def from_mapping(cls, value: Any) -> "PCSEFeatureLayerConfig":
        config = _mapping(value, owner="feature layer")
        name = str(config.get("name", "")).strip()
        if not name:
            raise ValueError("feature layer name must not be empty")
        pooling = str(config.get("pooling", "global_average")).strip().lower()
        if pooling not in {"global_average", "flatten"}:
            raise ValueError(
                "feature layer pooling must be 'global_average' or 'flatten'"
            )
        return cls(name=name, pooling=pooling)


@dataclass(frozen=True)
class PCSEPretrainingConfig:
    model: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    epochs: int


@dataclass(frozen=True)
class PCSEEnsembleConfig:
    epochs: int
    learning_rate: float


@dataclass(frozen=True)
class PCSEConfig:
    pretraining: PCSEPretrainingConfig
    transition_backend: str
    transition_backend_config: Mapping[str, Any]
    feature_layers: tuple[PCSEFeatureLayerConfig, ...]
    condition_limit: float
    covariance_ridge: float
    ensemble: PCSEEnsembleConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PCSEConfig":
        method = value.get("method")
        method_name = (
            str(method.get("name", "")).strip().lower()
            if isinstance(method, Mapping)
            else str(method or "").strip().lower()
        )
        if method_name != "pcse":
            raise ValueError("PCSE runner requires method: pcse")
        data = _mapping(value.get("data"), owner="data")
        num_classes = int(data.get("num_classes", 0))
        if num_classes < 3:
            raise ValueError("PCSE first version requires data.num_classes >= 3")

        pretraining = _mapping(
            value.get("pretraining_stage"), owner="pretraining_stage"
        )
        epochs = int(pretraining.get("epochs", 0))
        if epochs <= 0:
            raise ValueError("pretraining_stage.epochs must be positive")
        loss = _mapping(
            pretraining.get("loss", {"name": "ce"}),
            owner="pretraining_stage.loss",
        )
        if str(loss.get("name", "")).strip().lower() != "ce":
            raise ValueError("PCSE first version requires CE pretraining")
        checkpoint_selection = str(
            pretraining.get(
                "checkpoint_selection", "noisy_validation_accuracy"
            )
        ).strip().lower()
        if checkpoint_selection != "noisy_validation_accuracy":
            raise ValueError(
                "PCSE pretraining checkpoint selection must use noisy "
                "validation accuracy"
            )
        pretraining_config = PCSEPretrainingConfig(
            model=_mapping(pretraining.get("model"), owner="pretraining_stage.model"),
            optimizer=_mapping(
                pretraining.get("optimizer"), owner="pretraining_stage.optimizer"
            ),
            scheduler=_mapping(
                pretraining.get("scheduler", {"name": "none"}),
                owner="pretraining_stage.scheduler",
            ),
            epochs=epochs,
        )

        transition = _mapping(
            value.get("transition_stage", {"name": "dual_t"}),
            owner="transition_stage",
        )
        backend = str(transition.get("name", "dual_t")).strip().lower()
        if backend not in {"dual_t", "paper_volmin"}:
            raise ValueError(
                "PCSE transition_stage.name must be dual_t or paper_volmin"
            )
        backend_config = {
            key: transition[key] for key in sorted(transition) if key != "name"
        }
        if backend == "dual_t" and backend_config:
            raise ValueError(
                "PCSE dual_t transition backend has no configurable fields"
            )
        if backend == "paper_volmin":
            _exact_keys(
                transition,
                owner="transition_stage",
                allowed={
                    "name",
                    "epochs",
                    "lambda_volume",
                    "optimizer",
                    "scheduler",
                    "parameterization",
                    "determinant_tolerance",
                    "condition_limit",
                },
            )
            volmin_epochs = int(transition.get("epochs", 0))
            if volmin_epochs <= 0:
                raise ValueError(
                    "paper_volmin transition_stage.epochs must be positive"
                )
            lambda_volume = _positive_float(
                transition.get("lambda_volume"),
                owner="transition_stage.lambda_volume",
            )
            optimizer = _mapping(
                transition.get("optimizer"),
                owner="transition_stage.optimizer",
            )
            _exact_keys(
                optimizer,
                owner="transition_stage.optimizer",
                allowed={
                    "name",
                    "model_lr",
                    "transition_lr",
                    "weight_decay",
                },
            )
            if str(optimizer.get("name", "")).strip().lower() != "adamw":
                raise ValueError("paper_volmin optimizer.name must be adamw")
            model_lr = _positive_float(
                optimizer.get("model_lr"),
                owner="transition_stage.optimizer.model_lr",
            )
            transition_lr = _positive_float(
                optimizer.get("transition_lr"),
                owner="transition_stage.optimizer.transition_lr",
            )
            weight_decay = float(optimizer.get("weight_decay", 0.0))
            if not math.isfinite(weight_decay) or weight_decay < 0.0:
                raise ValueError(
                    "transition_stage.optimizer.weight_decay must be finite "
                    "and non-negative"
                )
            scheduler = _mapping(
                transition.get("scheduler", {"name": "none"}),
                owner="transition_stage.scheduler",
            )
            _exact_keys(
                scheduler,
                owner="transition_stage.scheduler",
                allowed={"name"},
            )
            if str(scheduler.get("name", "")).strip().lower() != "none":
                raise ValueError(
                    "paper_volmin first version requires scheduler.name: none"
                )
            parameterization = _mapping(
                transition.get("parameterization"),
                owner="transition_stage.parameterization",
            )
            _exact_keys(
                parameterization,
                owner="transition_stage.parameterization",
                allowed={
                    "name",
                    "initial_flip_mass",
                    "max_flip_mass",
                    "temperature",
                    "seed",
                },
            )
            if (
                str(parameterization.get("name", "")).strip().lower()
                != "diagonal_dominant"
            ):
                raise ValueError(
                    "paper_volmin parameterization.name must be "
                    "diagonal_dominant"
                )
            initial_flip_mass = _positive_float(
                parameterization.get("initial_flip_mass"),
                owner=(
                    "transition_stage.parameterization.initial_flip_mass"
                ),
            )
            max_flip_mass = _positive_float(
                parameterization.get("max_flip_mass"),
                owner="transition_stage.parameterization.max_flip_mass",
            )
            if max_flip_mass >= 0.5:
                raise ValueError(
                    "paper_volmin max_flip_mass must be less than 0.5"
                )
            if initial_flip_mass >= max_flip_mass:
                raise ValueError(
                    "paper_volmin initial_flip_mass must be below "
                    "max_flip_mass"
                )
            temperature = _positive_float(
                parameterization.get("temperature"),
                owner="transition_stage.parameterization.temperature",
            )
            seed = int(parameterization.get("seed"))
            determinant_tolerance = _positive_float(
                transition.get("determinant_tolerance"),
                owner="transition_stage.determinant_tolerance",
            )
            transition_condition_limit = _positive_float(
                transition.get("condition_limit"),
                owner="transition_stage.condition_limit",
            )
            backend_config = {
                "epochs": volmin_epochs,
                "lambda_volume": lambda_volume,
                "optimizer": {
                    "name": "adamw",
                    "model_lr": model_lr,
                    "transition_lr": transition_lr,
                    "weight_decay": weight_decay,
                },
                "scheduler": {"name": "none"},
                "parameterization": {
                    "name": "diagonal_dominant",
                    "initial_flip_mass": initial_flip_mass,
                    "max_flip_mass": max_flip_mass,
                    "temperature": temperature,
                    "seed": seed,
                },
                "determinant_tolerance": determinant_tolerance,
                "condition_limit": transition_condition_limit,
            }

        feature = _mapping(value.get("feature_stage"), owner="feature_stage")
        raw_layers = feature.get("layers")
        if not isinstance(raw_layers, list):
            raise TypeError("feature_stage.layers must be a list")
        layers = tuple(
            PCSEFeatureLayerConfig.from_mapping(item) for item in raw_layers
        )
        if len(layers) < 2:
            raise ValueError("PCSE requires at least two hidden feature layers")
        if len({layer.name for layer in layers}) != len(layers):
            raise ValueError("PCSE feature layer names must be unique")

        statistics = _mapping(
            value.get("statistics_stage", {}), owner="statistics_stage"
        )
        condition_limit = _positive_float(
            statistics.get("condition_limit", 1e8),
            owner="statistics_stage.condition_limit",
        )

        gda = _mapping(value.get("gda_stage", {}), owner="gda_stage")
        covariance_ridge = float(gda.get("covariance_ridge", 1e-5))
        if not math.isfinite(covariance_ridge) or covariance_ridge < 0.0:
            raise ValueError(
                "gda_stage.covariance_ridge must be finite and non-negative"
            )

        ensemble = _mapping(
            value.get("ensemble_stage"), owner="ensemble_stage"
        )
        ensemble_epochs = int(ensemble.get("epochs", 0))
        if ensemble_epochs <= 0:
            raise ValueError("ensemble_stage.epochs must be positive")
        ensemble_config = PCSEEnsembleConfig(
            epochs=ensemble_epochs,
            learning_rate=_positive_float(
                ensemble.get("learning_rate", 0.05),
                owner="ensemble_stage.learning_rate",
            ),
        )

        noise = _mapping(value.get("noise"), owner="noise")
        if str(noise.get("validation_targets", "")).strip().lower() != "noisy":
            raise ValueError(
                "PCSE requires noise.validation_targets: noisy"
            )
        evaluation = _mapping(
            value.get("evaluation", {"selection_split": "validation"}),
            owner="evaluation",
        )
        if str(
            evaluation.get("selection_split", "validation")
        ).strip().lower() != "validation":
            raise ValueError("PCSE may not use the test set for model selection")

        return cls(
            pretraining=pretraining_config,
            transition_backend=backend,
            transition_backend_config=backend_config,
            feature_layers=layers,
            condition_limit=condition_limit,
            covariance_ridge=covariance_ridge,
            ensemble=ensemble_config,
        )
