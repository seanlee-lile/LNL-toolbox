from __future__ import annotations

"""Versioned, task-neutral normalization for experiment configuration files."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


CONFIG_SCHEMA_VERSION = 1

_KINDS = {"experiment", "fragment", "mentor_artifact"}
_LEGACY_TOP_LEVEL = {
    "batch_size", "data_root", "device", "epochs", "hidden_width",
    "learning_rate", "noise_rate", "student_epochs", "student_learning_rate",
    "student_width", "trusted_size", "augment",
}
_EXPERIMENT_KEYS = {
    "schema_version", "kind", "method", "execution", "seed", "output_root",
    "configuration_fidelity", "parameter_record", "parameter_sampling",
    "hyperparameters", "data", "noise", "loader", "model", "models", "loss",
    "selector", "parameter_update", "optimizer", "scheduler", "trainer",
    "evaluation", "algorithm", "pipeline", "risk", "trusted_validation", "local_dataset",
    "transition_estimator",
    "early_stopping", "diagnostics", "label_convention", "phases", "warmup",
    "posterior_stage", "transition_stage", "final_stage", "pretraining_stage",
    "feature_stage", "statistics_stage", "gda_stage", "ensemble_stage",
    "feature_model", "transition", "statistics", "evidence", "revision",
    "instance_transition", "coteaching", "cnlcu", "cwd", "fine", "mc_ldce",
    "ca2c", "cal", "sieve", "meta", "t_revision", "upm", "volminnet", "dld",
    "dividemix", "lend", "num_classes",
}
_MENTOR_KEYS = {
    "schema_version", "kind", "seed", "feature_data", "data", "noise",
    "loader", "student_model", "student_optimizer", "student_trainer", "model",
    "optimizer", "trainer", "execution",
}
def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return deepcopy(dict(value))


def _positive_numbers(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if key == "batch_size" and (isinstance(item, bool) or int(item) <= 0):
                raise ValueError(f"{child} must be positive")
            if key in {"epochs", "max_steps", "num_workers"} and int(item) < 0:
                raise ValueError(f"{child} must be non-negative")
            _positive_numbers(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _positive_numbers(item, f"{path}[{index}]")


def _infer_kind(config: Mapping[str, Any]) -> str:
    if "feature_data" in config and "data" not in config:
        return "mentor_artifact"
    if "data" not in config and "execution" not in config:
        return "fragment"
    return "experiment"


def normalize_experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical public representation without runtime-only aliases."""

    value = deepcopy(dict(config))
    version = int(value.get("schema_version", CONFIG_SCHEMA_VERSION))
    if version != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported config schema_version {version}; expected {CONFIG_SCHEMA_VERSION}"
        )
    kind = str(value.get("kind", _infer_kind(value))).strip().lower()
    if kind not in _KINDS:
        raise ValueError(f"unsupported configuration kind: {kind!r}")
    value["schema_version"] = version
    value["kind"] = kind
    if kind == "fragment":
        return value

    noise = value.get("noise")
    if isinstance(noise, Mapping):
        normalized_noise = deepcopy(dict(noise))
        legacy_name = normalized_noise.pop("type", None)
        if legacy_name is not None:
            if "name" in normalized_noise and normalized_noise["name"] != legacy_name:
                raise ValueError("noise.name conflicts with legacy noise.type")
            normalized_noise["name"] = legacy_name
        if "name" not in normalized_noise and {
            "rho_positive", "rho_negative"
        } <= set(normalized_noise):
            normalized_noise["name"] = "binary_asymmetric_rcn"
        value["noise"] = normalized_noise

    runner = str((_mapping(value.get("execution", {}), "execution")).get("runner", ""))
    if runner == "binary" or "risk" in value:
        loader = _mapping(value.get("loader", {}), "loader")
        if "batch_size" in value:
            loader.setdefault("batch_size", value.pop("batch_size"))
        value["loader"] = loader
        model = _mapping(value.get("model", {}), "model")
        if "hidden_width" in value:
            model.setdefault("hidden_width", value.pop("hidden_width"))
        model.setdefault("name", "mlp")
        value["model"] = model
        optimizer = _mapping(value.get("optimizer", {}), "optimizer")
        if "learning_rate" in value:
            optimizer.setdefault("lr", value.pop("learning_rate"))
        optimizer.setdefault("name", "sgd")
        optimizer.setdefault("momentum", 0.9)
        value["optimizer"] = optimizer
        trainer = _mapping(value.get("trainer", {}), "trainer")
        if "epochs" in value:
            trainer.setdefault("epochs", value.pop("epochs"))
        value["trainer"] = trainer

    validate_experiment_config(value)
    return value


def runtime_experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Add temporary compatibility aliases only after canonical validation."""

    value = normalize_experiment_config(config)
    noise = value.get("noise")
    if isinstance(noise, Mapping) and "name" in noise:
        runtime_noise = deepcopy(dict(noise))
        runtime_noise.setdefault("type", runtime_noise["name"])
        value["noise"] = runtime_noise
    return value


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    version = int(config.get("schema_version", CONFIG_SCHEMA_VERSION))
    if version != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported config schema_version: {version}")
    kind = str(config.get("kind", _infer_kind(config))).strip().lower()
    if kind == "fragment":
        return
    allowed = _MENTOR_KEYS if kind == "mentor_artifact" else _EXPERIMENT_KEYS
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError("unknown top-level configuration fields: " + ", ".join(unknown))
    if kind == "experiment":
        data = _mapping(config.get("data"), "data")
        if not str(data.get("name", "")).strip():
            raise ValueError("data.name is required")
        if "execution" in config:
            execution = _mapping(config["execution"], "execution")
            if not str(execution.get("runner", "")).strip():
                raise ValueError("execution.runner is required")
    for key in ("loader", "model", "optimizer", "scheduler", "trainer", "evaluation"):
        if key in config and key != "model":
            _mapping(config[key], key)
        elif key == "model" and key in config and not isinstance(config[key], Mapping):
            raise ValueError("model must be a mapping")
    noise = config.get("noise")
    if isinstance(noise, Mapping):
        if "type" in noise:
            raise ValueError("noise.type is legacy; use noise.name")
        if "rate" in noise and not 0.0 <= float(noise["rate"]) <= 1.0:
            raise ValueError("noise.rate must be in [0, 1]")
    stale = sorted(set(config) & _LEGACY_TOP_LEVEL)
    if stale:
        raise ValueError("legacy top-level fields remain: " + ", ".join(stale))
    trainer = config.get("trainer")
    if isinstance(trainer, Mapping) and "epochs" in trainer and int(trainer["epochs"]) <= 0:
        raise ValueError("trainer.epochs must be positive")
    _positive_numbers(config, "")


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "normalize_experiment_config",
    "runtime_experiment_config",
    "validate_experiment_config",
]
