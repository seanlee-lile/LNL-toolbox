from __future__ import annotations

"""Strongly validated configuration for the paper-oriented DLD workflow."""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping


_FIDELITY = {
    "name": "paper_oriented_v2_cosine_similarity",
    "hard_y0": "averaged_views",
    "direction_endpoint": "estimated_yn",
    "direction": "yn_minus_y0",
    "neighbor_metric": "cosine_similarity",
    "neighbor_weighting": "inverse_neighbor_value",
    "self_neighbor": "include",
    "divergence": "kl_ps_to_pw",
    "divergence_softmax": False,
    "hard_yn_zero_denominator": "fail",
    "schedule": "average",
    "inference_initialization": "zero",
    "inference_steps": 5,
}


def _mapping(value: Any, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    return dict(value)


def _positive_int(value: Any, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{owner} must be a positive integer")
    return int(value)


def _finite_positive(value: Any, owner: str, *, allow_zero: bool = False) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or (parsed < 0 if allow_zero else parsed <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{owner} must be finite and {qualifier}")
    return parsed


@dataclass(frozen=True)
class DLDConfig:
    fidelity: Mapping[str, Any]
    feature_extractor: Mapping[str, Any]
    precorrection: Mapping[str, Any]
    diffusion: Mapping[str, Any]
    inference: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "DLDConfig":
        if str(config.get("method", "")).strip().lower() != "dld":
            raise ValueError("DLD configuration requires method: dld")
        section = _mapping(config.get("dld"), "dld configuration")
        fidelity = _mapping(section.get("fidelity"), "dld.fidelity")
        for name, expected in _FIDELITY.items():
            if fidelity.get(name) != expected:
                raise ValueError(
                    f"dld.fidelity.{name} must be {expected!r} for "
                    "paper_oriented_v2_cosine_similarity"
                )

        extractor = _mapping(
            section.get("feature_extractor"), "dld.feature_extractor"
        )
        source = str(extractor.get("source", "")).strip().lower()
        if source not in {"repository_frozen_model", "external_checkpoint"}:
            raise ValueError(
                "DLD feature_extractor.source must be repository_frozen_model "
                "or external_checkpoint"
            )
        _mapping(extractor.get("model"), "dld.feature_extractor.model")
        if source == "external_checkpoint":
            external = _mapping(
                extractor.get("external"), "dld.feature_extractor.external"
            )
            if str(external.get("adapter", "")).strip().lower() != "upm_main_best":
                raise ValueError("DLD external feature adapter must be upm_main_best")
            for field in (
                "run_directory_env",
                "checkpoint_sha256",
                "manifest_sha256",
                "mapping_hash",
                "dataset_fingerprint",
            ):
                if not str(external.get(field, "")).strip():
                    raise ValueError(f"dld.feature_extractor.external.{field} is required")

        precorrection = _mapping(section.get("precorrection"), "dld.precorrection")
        _positive_int(precorrection.get("k_neighbors"), "dld.precorrection.k_neighbors")
        _finite_positive(precorrection.get("delta"), "dld.precorrection.delta")
        chunk_size = precorrection.get("query_chunk_size")
        if chunk_size is not None:
            _positive_int(chunk_size, "dld.precorrection.query_chunk_size")
        if _positive_int(
            precorrection.get("gmm_components"),
            "dld.precorrection.gmm_components",
        ) != 2:
            raise ValueError("DLD requires exactly two GMM components")
        if isinstance(precorrection.get("gmm_seed"), bool) or not isinstance(
            precorrection.get("gmm_seed"), int
        ):
            raise ValueError("dld.precorrection.gmm_seed must be an integer")
        separation = precorrection.get("minimum_mean_separation", 1e-6)
        _finite_positive(
            separation,
            "dld.precorrection.minimum_mean_separation",
            allow_zero=True,
        )

        diffusion = _mapping(section.get("diffusion"), "dld.diffusion")
        timesteps = _positive_int(diffusion.get("timesteps"), "dld.diffusion.timesteps")
        _positive_int(diffusion.get("epochs"), "dld.diffusion.epochs")
        model = _mapping(diffusion.get("model"), "dld.diffusion.model")
        if model.get("independent_predictors") is not True:
            raise ValueError("DLD requires independent_predictors: true")
        _positive_int(model.get("hidden_dim", 64), "dld.diffusion.model.hidden_dim")
        _positive_int(model.get("time_dim", 16), "dld.diffusion.model.time_dim")
        optimizers = _mapping(diffusion.get("optimizer"), "dld.diffusion.optimizer")
        for peer in ("direction", "noise"):
            optimizer = _mapping(optimizers.get(peer), f"dld.diffusion.optimizer.{peer}")
            if str(optimizer.get("name", "")).lower() not in {"adam", "adamw", "sgd"}:
                raise ValueError(f"unsupported DLD {peer} optimizer")
            _finite_positive(optimizer.get("lr"), f"DLD {peer} learning rate")
        schedulers = _mapping(diffusion.get("scheduler"), "dld.diffusion.scheduler")
        for peer in ("direction", "noise"):
            scheduler = _mapping(schedulers.get(peer), f"dld.diffusion.scheduler.{peer}")
            if str(scheduler.get("name", "none")).lower() not in {"none", "cosine", "multistep"}:
                raise ValueError(f"unsupported DLD {peer} scheduler")
        ema = _mapping(diffusion.get("ema", {}), "dld.diffusion.ema")
        if ema.get("enabled", False):
            decay = float(ema.get("decay", 0.999))
            if not math.isfinite(decay) or not 0 <= decay < 1:
                raise ValueError("dld.diffusion.ema.decay must be in [0, 1)")

        inference = _mapping(section.get("inference"), "dld.inference")
        steps = _positive_int(inference.get("steps"), "dld.inference.steps")
        if steps != 5 or steps > timesteps:
            raise ValueError(
                "paper_oriented_v2_cosine_similarity requires five inference "
                "steps <= timesteps"
            )
        if inference.get("deterministic") is not True:
            raise ValueError(
                "paper_oriented_v2_cosine_similarity inference must be deterministic"
            )
        if inference.get("initialization") != "zero":
            raise ValueError(
                "paper_oriented_v2_cosine_similarity inference initialization "
                "must be zero"
            )
        return cls(fidelity, extractor, precorrection, diffusion, inference)

    @property
    def epochs(self) -> int:
        return int(self.diffusion["epochs"])

    @property
    def identity_payload(self) -> dict[str, Any]:
        diffusion = dict(self.diffusion)
        diffusion.pop("epochs", None)
        return {
            "fidelity": dict(self.fidelity),
            "feature_extractor": dict(self.feature_extractor),
            "precorrection": dict(self.precorrection),
            "diffusion": diffusion,
            "inference": dict(self.inference),
        }

    @property
    def identity_hash(self) -> str:
        encoded = json.dumps(
            self.identity_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = ["DLDConfig"]
