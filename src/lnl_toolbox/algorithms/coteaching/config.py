from __future__ import annotations

"""Validated configuration contract for the complete Co-teaching method."""

from dataclasses import dataclass
import math
from typing import Any, Mapping

from lnl_toolbox.selectors import LinearKeepRateSchedule


def _mapping(value: Any, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class CoTeachingConfig:
    noise_rate: float
    peer_seed_offset: int
    remember_schedule: LinearKeepRateSchedule
    model_count: int = 2
    count_rule: str = "floor"
    tie_break: str = "stable_sample_index"
    primary_metric: str = "mean_peer_accuracy"
    ensemble: str = "mean_probabilities"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "CoTeachingConfig":
        method = config.get("method")
        name = (
            str(method.get("name", "")).strip().lower()
            if isinstance(method, Mapping)
            else str(method or "").strip().lower()
        )
        if name != "coteaching":
            raise ValueError("Co-teaching requires method: coteaching")

        values = _mapping(config.get("coteaching"), "coteaching configuration")
        model_count = int(values.get("model_count", 2))
        if model_count != 2:
            raise ValueError("Co-teaching requires exactly two models")
        initialization = _mapping(
            values.get("initialization", {}),
            "coteaching.initialization",
        )
        peer_seed_offset = int(initialization.get("peer_seed_offset", 1))
        if peer_seed_offset == 0:
            raise ValueError("coteaching peer_seed_offset must be non-zero")

        rate = float(values.get("noise_rate", float("nan")))
        if not math.isfinite(rate) or not 0.0 <= rate < 1.0:
            raise ValueError("coteaching.noise_rate must be finite and in [0, 1)")
        noise = _mapping(config.get("noise"), "noise configuration")
        if "rate" in noise and not math.isclose(
            float(noise["rate"]), rate, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("noise.rate and coteaching.noise_rate must match")

        schedule = _mapping(
            values.get("remember_schedule"),
            "coteaching.remember_schedule",
        )
        if str(schedule.get("name", "")).strip().lower() != "linear":
            raise ValueError("Co-teaching remember_schedule must be linear")
        if set(schedule) != {"name", "start", "end", "gradual_epochs"}:
            raise ValueError(
                "remember_schedule requires only name, start, end, and gradual_epochs"
            )
        start = float(schedule["start"])
        end = float(schedule["end"])
        if not math.isclose(start, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Co-teaching remember schedule must start at 1.0")
        if not math.isclose(end, 1.0 - rate, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("remember schedule end must equal 1 - noise_rate")
        gradual_epochs = int(schedule["gradual_epochs"])
        remember_schedule = LinearKeepRateSchedule(start, end, gradual_epochs)

        selection = _mapping(values.get("selection", {}), "coteaching.selection")
        count_rule = str(selection.get("count_rule", "floor")).strip().lower()
        tie_break = str(
            selection.get("tie_break", "stable_sample_index")
        ).strip().lower()
        if count_rule != "floor":
            raise ValueError("Co-teaching selection count_rule must be floor")
        if tie_break != "stable_sample_index":
            raise ValueError(
                "Co-teaching selection tie_break must be stable_sample_index"
            )

        evaluation = _mapping(config.get("evaluation", {}), "evaluation configuration")
        if str(evaluation.get("selection_split", "validation")).lower() != "validation":
            raise ValueError("Co-teaching best-checkpoint selection requires validation")
        primary = str(evaluation.get("primary", "mean_peer_accuracy")).lower()
        ensemble = str(evaluation.get("ensemble", "mean_probabilities")).lower()
        if primary != "mean_peer_accuracy":
            raise ValueError("Co-teaching evaluation primary must be mean_peer_accuracy")
        if ensemble != "mean_probabilities":
            raise ValueError("Co-teaching ensemble must use mean_probabilities")

        incompatible = {
            "selector",
            "parameter_update",
            "pipeline",
            "weight_provider",
            "target_provider",
            "objective_consumer",
        }
        present = sorted(key for key in incompatible if key in config)
        if present:
            raise ValueError(
                "Co-teaching cannot be combined with single-model components: "
                + ", ".join(present)
            )
        return cls(
            noise_rate=rate,
            peer_seed_offset=peer_seed_offset,
            remember_schedule=remember_schedule,
            model_count=model_count,
            count_rule=count_rule,
            tie_break=tie_break,
            primary_metric=primary,
            ensemble=ensemble,
        )

    def rate_at(self, epoch: int) -> float:
        return self.remember_schedule.rate_at(epoch)
