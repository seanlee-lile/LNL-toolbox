from __future__ import annotations

"""Validated configuration for the CNLCU-S/H method variants."""

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
    sigma_squared: float | None
    variant: str = "soft"
    model_count: int = 2
    storage_dtype: str = "float32"
    count_rule: str = "floor"
    tie_break: str = "stable_sample_index"
    hard_fidelity: str | None = None
    tau_min: float | None = None
    loss_upper_bound_mode: str | None = None
    loss_upper_bound: float | None = None
    truncation_method: str | None = None
    n_neighbors: int | None = None
    contamination: float | None = None
    minimum_observations: int | None = None

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
        if variant not in {"soft", "hard"}:
            raise ValueError("CNLCU variant must be soft or hard")
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
        sigma: float | None = None
        hard_fidelity: str | None = None
        tau_min: float | None = None
        loss_upper_bound_mode: str | None = None
        loss_upper_bound: float | None = None
        truncation_method: str | None = None
        n_neighbors: int | None = None
        contamination: float | None = None
        minimum_observations: int | None = None
        if variant == "soft":
            if set(uncertainty) != {"sigma_squared"}:
                raise ValueError("CNLCU-S uncertainty accepts only sigma_squared")
            if "hard_fidelity" in values or "truncation" in values:
                raise ValueError("CNLCU-S cannot include hard-only configuration")
            sigma = float(uncertainty.get("sigma_squared", float("nan")))
            if not math.isfinite(sigma) or not 0.0 < sigma < 1.0:
                raise ValueError("CNLCU sigma_squared must be finite and in (0,1)")
        else:
            hard_fidelity = str(values.get("hard_fidelity", "")).strip().lower()
            if hard_fidelity != "paper_formula_corrected_lof":
                raise ValueError(
                    "CNLCU-H requires hard_fidelity: paper_formula_corrected_lof"
                )
            if set(uncertainty) != {"tau_min", "loss_upper_bound"}:
                raise ValueError(
                    "CNLCU-H uncertainty requires tau_min and loss_upper_bound"
                )
            tau_min = float(uncertainty.get("tau_min", float("nan")))
            if not math.isfinite(tau_min) or tau_min <= 0.0:
                raise ValueError("CNLCU-H tau_min must be finite and positive")
            bound = _mapping(
                uncertainty.get("loss_upper_bound"),
                "cnlcu.uncertainty.loss_upper_bound",
            )
            if set(bound) != {"mode", "value"}:
                raise ValueError("CNLCU-H loss_upper_bound requires mode and value")
            loss_upper_bound_mode = str(bound["mode"]).strip().lower()
            loss_upper_bound = float(bound["value"])
            if loss_upper_bound_mode != "fixed":
                raise ValueError("CNLCU-H supports only fixed loss_upper_bound")
            if not math.isfinite(loss_upper_bound) or loss_upper_bound <= 0.0:
                raise ValueError("CNLCU-H loss_upper_bound must be finite and positive")
            truncation = _mapping(values.get("truncation"), "cnlcu.truncation")
            required_truncation = {
                "method", "n_neighbors", "contamination", "minimum_observations"
            }
            if set(truncation) != required_truncation:
                raise ValueError(
                    "CNLCU-H truncation requires method/n_neighbors/contamination/"
                    "minimum_observations"
                )
            truncation_method = str(truncation["method"]).strip().lower()
            n_neighbors = int(truncation["n_neighbors"])
            contamination = float(truncation["contamination"])
            minimum_observations = int(truncation["minimum_observations"])
            if truncation_method != "lof":
                raise ValueError("CNLCU-H supports only corrected LOF truncation")
            if n_neighbors < 1:
                raise ValueError("CNLCU-H n_neighbors must be at least one")
            if (
                not math.isfinite(contamination)
                or not 0.0 < contamination < 0.5
            ):
                raise ValueError("CNLCU-H contamination must be finite and in (0,0.5)")
            if minimum_observations < n_neighbors + 1:
                raise ValueError(
                    "CNLCU-H minimum_observations must be at least n_neighbors + 1"
                )
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
        return cls(
            noise_rate=rate,
            peer_seed_offset=peer_seed_offset,
            remember_schedule=remember_schedule,
            window_size=window_size,
            sigma_squared=sigma,
            variant=variant,
            model_count=model_count,
            storage_dtype=storage_dtype,
            count_rule=count_rule,
            tie_break=tie_break,
            hard_fidelity=hard_fidelity,
            tau_min=tau_min,
            loss_upper_bound_mode=loss_upper_bound_mode,
            loss_upper_bound=loss_upper_bound,
            truncation_method=truncation_method,
            n_neighbors=n_neighbors,
            contamination=contamination,
            minimum_observations=minimum_observations,
        )

    def rate_at(self, epoch: int) -> float:
        return self.remember_schedule.rate_at(epoch)


__all__ = ["CNLCUConfig"]
