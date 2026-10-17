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
def training_stage_controls(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Public boolean controls for iterative training, not artifact computation."""
    execution = config.get("execution") or {}
    method = config.get("method") or ""
    if isinstance(method, Mapping):
        method = method.get("name", "")
    runner = str(execution.get("runner") or method or "supervised")
    if not execution.get("runner"):
        for section in ("cwd", "fine", "instance_transition"):
            if section in config:
                runner = section
    definitions = {
        "t_revision": [("t_revision.stage1", "后验模型训练"), ("t_revision.classifier_initialization", "分类器初始化"), ("t_revision.revision", "转移矩阵修正训练")],
        "dual_t": [("posterior_stage", "后验模型训练"), ("final_stage", "最终分类器训练")],
        "upm": [("upm.stage1", "后验模型预训练"), ("upm.main", "交替优化训练")],
        "pcse": [("pretraining_stage", "特征模型预训练"), ("transition_stage", "转移矩阵训练"), ("ensemble_stage", "集成权重训练")],
        "dld": [("dld.diffusion", "标签扩散模型训练")],
        "lend": [("lend.training", "标签稀释训练")],
        "dividemix": [("dividemix.warmup", "预热训练"), ("dividemix.training", "双网络训练")],
        "instance_transition": [("warmup", "模型预训练"), ("trainer", "校正训练"), ("revision", "修正训练")],
        "cal": [("warmup", "辅助标签模型训练"), ("trainer", "CAL训练")],
        "mc_ldce": [("warmup", "模型预训练"), ("transition", "转移矩阵估计训练"), ("trainer", "分类器训练")],
    }
    entries = definitions.get(runner, [("trainer", "模型训练")])
    if config.get("kind") == "mentor_artifact" or "student_trainer" in config:
        entries = [("student_trainer", "反馈样本生成训练"), ("trainer", "Mentor网络训练")]
    result = []
    for prefix, label in entries:
        current: Any = config
        for part in prefix.split("."):
            current = current.get(part) if isinstance(current, Mapping) else None
        if isinstance(current, Mapping):
            if prefix == "pretraining_stage" and current.get("mode") == "external_checkpoint":
                continue
            if prefix == "transition_stage" and runner == "pcse" and current.get("name", "dual_t") != "paper_volmin":
                continue
            result.append({"path": prefix + ".early_stopping", "stage": prefix,
                           "label": label + "早停", "value": current.get("early_stopping", False)})
    extra = {
        "ca2c": ("ca2c.warmup_early_stopping", "预热阶段早停"),
        "fine": ("fine.warmup_early_stopping", "预热阶段早停"),
    }
    if runner in extra:
        path, label = extra[runner]
        parent, leaf = path.rsplit(".", 1)
        result.insert(0, {"path": path, "stage": parent + ".warmup", "label": label,
                          "value": (config.get(parent) or {}).get(leaf, False)})
    pipeline = config.get("pipeline") or {}
    if runner == "supervised" and int(pipeline.get("warmup_epochs") or 0) > 0:
        result.insert(0, {"path": "pipeline.early_stopping", "stage": "pipeline",
                          "label": "预训练早停", "value": pipeline.get("early_stopping", False)})
    consumer = pipeline.get("objective_consumer") or {}
    if runner == "supervised" and int(consumer.get("warmup_epochs") or 0) > 0:
        result.insert(0, {"path": "pipeline.objective_consumer.warmup_early_stopping",
                          "stage": "pipeline.objective_consumer.warmup", "label": "预热阶段早停",
                          "value": consumer.get("warmup_early_stopping", False)})
    errors = [
        f"invalid config for {field['path']}, expected a boolean"
        for field in result if not isinstance(field["value"], bool)
    ]
    if errors:
        raise ValueError("\n".join(errors))
    return result


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


def synchronize_experiment_seed(config: Mapping[str, Any]) -> dict[str, Any]:
    """Use the experiment seed for every active stochastic component.

    Historical parameter-search records and peer offsets describe provenance
    or a deterministic offset, not independently configurable random seeds.
    """

    value = deepcopy(dict(config))
    if "seed" not in value:
        return value
    seed = int(value["seed"])
    value["seed"] = seed

    def synchronize(section: dict[str, Any]) -> None:
        for key, item in section.items():
            if key in {"parameter_record", "parameter_sampling", "hyperparameters"}:
                continue
            if isinstance(key, str) and (key == "seed" or (key.endswith("_seed") and key != "peer_seed_offset")):
                section[key] = seed
            elif isinstance(item, dict):
                synchronize(item)
            elif isinstance(item, list):
                for child in item:
                    if isinstance(child, dict):
                        synchronize(child)

    synchronize(value)
    return value


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

    algorithm = value.get("algorithm", {})
    if isinstance(algorithm, Mapping) and algorithm.get("name") == "jocor":
        algorithm = dict(algorithm)
        selector = deepcopy(dict(value.get("selector", {})))
        keep_rate = dict(selector.get("keep_rate", {}))
        # Legacy recipes already encode the paper's prior as the final
        # retained fraction. Preserve that fraction when adding the new input.
        if "estimated_noise_rate" in algorithm:
            estimate = algorithm["estimated_noise_rate"]
        else:
            try:
                estimate = 1.0 - float(keep_rate.get("end", 1.0))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "invalid config for selector.keep_rate.end, expected number in [0, 1]"
                ) from error
        try:
            if isinstance(estimate, bool):
                raise ValueError("boolean is not a noise rate")
            estimate = float(estimate)
            if not 0.0 <= estimate <= 1.0:
                raise ValueError("noise rate is outside [0, 1]")
        except (TypeError, ValueError) as error:
            raise ValueError(
                "invalid config for algorithm.estimated_noise_rate, expected number in [0, 1]"
            ) from error
        algorithm["estimated_noise_rate"] = estimate
        keep_rate["end"] = 1.0 - estimate
        selector["keep_rate"] = keep_rate
        value["algorithm"] = algorithm
        value["selector"] = selector
    validate_experiment_config(value)
    return synchronize_experiment_seed(value)


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
        training_stage_controls(config)
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
