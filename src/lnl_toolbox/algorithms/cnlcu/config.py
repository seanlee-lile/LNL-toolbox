from __future__ import annotations

"""Validated configuration for the CNLCU-S paper method."""

from dataclasses import dataclass
import math
from typing import Any, Mapping

from lnl_toolbox.selectors import LinearKeepRateSchedule


def _mapping(value: Any, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class CNLCUConfig:
    noise_rate: float
    peer_seed_offset: int
    remember_schedule: LinearKeepRateSchedule
    window_size: int
    sigma_squared: float
    variant: str = "soft"
    model_count: int = 2
    storage_dtype: str = "float32"
    count_rule: str = "floor"
    tie_break: str = "stable_sample_index"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "CNLCUConfig":
        method = config.get("method")
        name = (
            str(method.get("name", "")).strip().lower()
            if isinstance(method, Mapping)
            else str(method or "").strip().lower()
        )
        if name != "cnlcu":
            raise ValueError("CNLCU requires method: cnlcu")
        values = _mapping(config.get("cnlcu"), "cnlcu configuration")
        variant = str(values.get("variant", "")).strip().lower()
        if variant != "soft":
            raise ValueError("CNLCU first version supports only variant: soft")
        model_count = int(values.get("model_count", 2))
        if model_count != 2:
            raise ValueError("CNLCU requires exactly two models")
        rate = float(values.get("noise_rate", float("nan")))
        if not math.isfinite(rate) or not 0.0 <= rate < 1.0:
            raise ValueError("cnlcu.noise_rate must be finite and in [0,1)")
        noise = _mapping(config.get("noise"), "noise configuration")
        if "rate" in noise and not math.isclose(float(noise["rate"]), rate, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("noise.rate and cnlcu.noise_rate must match")
        initialization = _mapping(values.get("initialization", {}), "cnlcu.initialization")
        peer_seed_offset = int(initialization.get("peer_seed_offset", 1))
        if peer_seed_offset == 0:
            raise ValueError("cnlcu peer_seed_offset must be non-zero")
        schedule = _mapping(values.get("remember_schedule"), "cnlcu.remember_schedule")
        if set(schedule) != {"name", "start", "end", "gradual_epochs"} or str(schedule["name"]).lower() != "linear":
            raise ValueError("CNLCU remember_schedule requires linear name/start/end/gradual_epochs")
        start, end = float(schedule["start"]), float(schedule["end"])
        if not math.isclose(start, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("CNLCU remember schedule must start at 1.0")
        if not math.isclose(end, 1.0 - rate, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("CNLCU remember schedule end must equal 1-noise_rate")
        remember_schedule = LinearKeepRateSchedule(start, end, int(schedule["gradual_epochs"]))
        history = _mapping(values.get("history"), "cnlcu.history")
        window_size = int(history.get("window_size", 0))
        storage_dtype = str(history.get("storage_dtype", "")).lower()
        if window_size <= 0:
            raise ValueError("CNLCU history window_size must be positive")
        if storage_dtype != "float32":
            raise ValueError("CNLCU history storage_dtype must be float32")
        uncertainty = _mapping(values.get("uncertainty"), "cnlcu.uncertainty")
        sigma = float(uncertainty.get("sigma_squared", float("nan")))
        if not math.isfinite(sigma) or not 0.0 < sigma < 1.0:
            raise ValueError("CNLCU sigma_squared must be finite and in (0,1)")
        selection = _mapping(values.get("selection", {}), "cnlcu.selection")
        count_rule = str(selection.get("count_rule", "floor")).lower()
        tie_break = str(selection.get("tie_break", "stable_sample_index")).lower()
        if count_rule != "floor" or tie_break != "stable_sample_index":
            raise ValueError("CNLCU requires floor count and stable_sample_index tie-break")
        evaluation = _mapping(config.get("evaluation", {}), "evaluation configuration")
        if str(evaluation.get("selection_split", "validation")).lower() != "validation":
            raise ValueError("CNLCU best-checkpoint selection requires validation")
        if str(evaluation.get("primary", "mean_peer_accuracy")).lower() != "mean_peer_accuracy":
            raise ValueError("CNLCU primary metric must be mean_peer_accuracy")
        if str(evaluation.get("ensemble", "mean_probabilities")).lower() != "mean_probabilities":
            raise ValueError("CNLCU ensemble must use mean_probabilities")
        incompatible = {
            "selector", "parameter_update", "pipeline", "weight_provider",
            "target_provider", "objective_consumer", "dss",
        }
        present = sorted(key for key in incompatible if key in config)
        if present:
            raise ValueError("CNLCU cannot combine with single-model components: " + ", ".join(present))
        return cls(rate, peer_seed_offset, remember_schedule, window_size, sigma,
                   variant, model_count, storage_dtype, count_rule, tie_break)

    def rate_at(self, epoch: int) -> float:
        return self.remember_schedule.rate_at(epoch)


__all__ = ["CNLCUConfig"]
