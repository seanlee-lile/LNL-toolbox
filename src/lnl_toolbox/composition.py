from __future__ import annotations

"""Compatibility-aware composition helpers for the user-facing CLI."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from lnl_toolbox.catalog import validate_config
from lnl_toolbox.plugins.builtin import (
    build_builtin_loss,
    build_builtin_parameter_update_policy,
    build_builtin_pipeline,
    build_builtin_selector,
)


@dataclass(frozen=True, slots=True)
class CompositionSummary:
    runner: str
    loss: str
    selector: str
    parameter_update: str
    pipeline_features: tuple[str, ...]


def _name(config: Mapping[str, Any], key: str, default: str) -> str:
    value = config.get(key, {}) or {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} configuration must be a mapping")
    return str(value.get("name", default)).strip().lower()


def validate_composition(config: Mapping[str, Any]) -> CompositionSummary:
    """Validate one public composition more deeply than name-only static checks."""

    candidate = deepcopy(dict(config))
    runner = validate_config(candidate)
    if runner.name != "supervised":
        raise ValueError(
            f"runner {runner.name!r} has a dedicated lifecycle; "
            "compose create currently supports runner 'supervised' only"
        )

    # Reuse the runtime's cross-field checks, then instantiate every selected slot.
    from lnl_toolbox.training.experiment import (
        _resolve_dss_epoch_contract,
        _validate_supervised_config,
    )

    candidate.setdefault("loss", {"name": "ce"})
    candidate.setdefault("selector", {"name": "all"})
    candidate.setdefault("parameter_update", {"name": "standard"})
    _validate_supervised_config(candidate)
    epochs = int((candidate.get("trainer", {}) or {}).get("epochs", 0))
    _resolve_dss_epoch_contract(candidate, epochs)

    build_builtin_loss(candidate["loss"])
    build_builtin_selector(candidate["selector"])
    build_builtin_parameter_update_policy(candidate["parameter_update"])
    pipeline = build_builtin_pipeline(candidate.get("pipeline"))

    loss = _name(candidate, "loss", "ce")
    selector = _name(candidate, "selector", "all")
    parameter_update = _name(candidate, "parameter_update", "standard")
    pipeline_config = candidate.get("pipeline", {}) or {}
    if not isinstance(pipeline_config, Mapping):
        raise TypeError("pipeline configuration must be a mapping")

    features = tuple(
        name
        for name in (
            "transition_estimator",
            "risk_corrector",
            "weight_provider",
            "statistic_estimator",
            "objective_consumer",
            "regularizer",
        )
        if pipeline_config.get(name) not in (None, False)
    )
    if pipeline.transition_estimator is not None and pipeline.risk_corrector is None:
        raise ValueError(
            "pipeline.transition_estimator alone does not change training; "
            "add pipeline.risk_corrector"
        )
    if pipeline.regularizer is not None:
        raise ValueError(
            "pipeline.regularizer is registered but not connected to the supervised runner"
        )
    if pipeline.objective_consumer is not None:
        if selector != "all":
            raise ValueError("objective_consumer requires selector.name='all'")
        if parameter_update != "standard":
            raise ValueError(
                "objective_consumer requires parameter_update.name='standard'"
            )
        objective = pipeline_config.get("objective_consumer", {}) or {}
        objective_name = str(objective.get("name", "")).strip().lower()
        if objective_name == "dss" and loss != "ce":
            raise ValueError("DSS requires loss.name='ce'")

    return CompositionSummary(
        runner=runner.name,
        loss=loss,
        selector=selector,
        parameter_update=parameter_update,
        pipeline_features=features,
    )


def apply_overrides(
    config: Mapping[str, Any],
    *,
    loss: str | None = None,
    selector: str | None = None,
    keep_rate: float | None = None,
    parameter_update: str | None = None,
    milestones: list[int] | None = None,
    gamma: float | None = None,
    cdr_noise_rate: float | None = None,
    l1_decay: float | None = None,
) -> dict[str, Any]:
    """Apply explicit CLI choices without mutating the source recipe."""

    result = deepcopy(dict(config))
    if loss is not None:
        result["loss"] = {"name": loss}

    if selector is not None:
        if selector == "small_loss":
            if keep_rate is None:
                raise ValueError("--selector small_loss requires --keep-rate")
            if not 0.0 < keep_rate <= 1.0:
                raise ValueError("--keep-rate must satisfy 0 < value <= 1")
            result["selector"] = {"name": "small_loss", "keep_rate": keep_rate}
        else:
            if keep_rate is not None:
                raise ValueError("--keep-rate is only valid with --selector small_loss")
            result["selector"] = {"name": selector}
    elif keep_rate is not None:
        raise ValueError("--keep-rate requires --selector small_loss")

    if parameter_update is not None:
        if parameter_update == "step_milestone":
            if not milestones:
                raise ValueError(
                    "--parameter-update step_milestone requires --milestones"
                )
            update: dict[str, Any] = {
                "name": "step_milestone",
                "milestones": milestones,
            }
            if gamma is not None:
                update["gamma"] = gamma
            result["parameter_update"] = update
        elif parameter_update == "cdr":
            if cdr_noise_rate is None:
                raise ValueError(
                    "--parameter-update cdr requires --cdr-noise-rate"
                )
            update = {
                "name": "cdr",
                "noise_rate": cdr_noise_rate,
                "l1_decay": 0.001 if l1_decay is None else l1_decay,
                "critical_scope": "all_trainable",
                "compatibility_mode": "paper",
            }
            result["parameter_update"] = update
        else:
            if any(value is not None for value in (milestones, gamma, cdr_noise_rate, l1_decay)):
                raise ValueError(
                    "update-specific options do not apply to parameter_update=standard"
                )
            result["parameter_update"] = {"name": "standard"}
    elif any(value is not None for value in (milestones, gamma, cdr_noise_rate, l1_decay)):
        raise ValueError("update-specific options require --parameter-update")

    result["execution"] = {"runner": "supervised"}
    return result


def write_composed_config(config: Mapping[str, Any], output: Path) -> Path:
    """Write a validated portable YAML file without overwriting user data."""

    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")
    if destination.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("compose output must use a .yaml or .yml extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    destination.write_text(
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


__all__ = [
    "CompositionSummary",
    "apply_overrides",
    "validate_composition",
    "write_composed_config",
]
