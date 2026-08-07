from __future__ import annotations

"""Validated configuration for the paper/official-oriented DivideMix workflow."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Mapping


def _mapping(value: Any, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    return dict(value)


def _positive_int(value: Any, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{owner} must be an integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{owner} must be a positive integer")
    return parsed


def _finite(value: Any, owner: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or (parsed <= minimum if strict else parsed < minimum):
        qualifier = "greater than" if strict else "at least"
        raise ValueError(f"{owner} must be finite and {qualifier} {minimum}")
    return parsed


@dataclass(frozen=True)
class DivideMixGMMConfig:
    threshold: float
    random_seed: int
    max_iterations: int
    tolerance: float
    covariance_regularization: float
    minimum_mean_separation: float
    history_name: str
    high_noise_rate: float
    history_window_epochs: int


@dataclass(frozen=True)
class DivideMixConfig:
    fidelity: str
    peer_seed_offset: int
    warmup_epochs: int
    confidence_penalty_weight: float
    gmm: DivideMixGMMConfig
    augmentations: int
    temperature: float
    mixup_alpha: float
    lambda_u: float
    lambda_r: float
    rampup_epochs: int
    training_epochs: int
    ensemble: str

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "DivideMixConfig":
        method = config.get("method")
        name = str(method.get("name", "") if isinstance(method, Mapping) else method or "").strip().lower()
        if name != "dividemix":
            raise ValueError("DivideMix requires method: dividemix")
        execution = _mapping(config.get("execution"), "execution")
        if str(execution.get("runner", "")).strip().lower() != "dividemix":
            raise ValueError("DivideMix requires execution.runner: dividemix")
        values = _mapping(config.get("dividemix"), "dividemix")
        fidelity = str(values.get("fidelity", "official_cifar_v1")).strip().lower()
        if fidelity != "official_cifar_v1":
            raise ValueError("DivideMix fidelity must be official_cifar_v1")
        if int(values.get("model_count", 2)) != 2:
            raise ValueError("DivideMix requires exactly two models")
        initialization = _mapping(values.get("initialization", {}), "dividemix.initialization")
        offset = int(initialization.get("peer_seed_offset", 1))
        if offset == 0:
            raise ValueError("DivideMix peer_seed_offset must be non-zero")
        warmup = _mapping(values.get("warmup"), "dividemix.warmup")
        warmup_epochs = _positive_int(warmup.get("epochs"), "dividemix.warmup.epochs")
        confidence = _finite(warmup.get("confidence_penalty_weight", 1.0), "confidence penalty weight")
        gmm = _mapping(values.get("gmm"), "dividemix.gmm")
        threshold = _finite(gmm.get("threshold", 0.5), "GMM threshold")
        if not 0.0 < threshold < 1.0:
            raise ValueError("DivideMix GMM threshold must be in (0, 1)")
        history = _mapping(gmm.get("loss_history", {}), "dividemix.gmm.loss_history")
        history_name = str(history.get("name", "official_auto")).strip().lower()
        if history_name not in {"official_auto", "current_epoch"}:
            raise ValueError("DivideMix loss history must be official_auto or current_epoch")
        gmm_config = DivideMixGMMConfig(
            threshold=threshold,
            random_seed=int(gmm.get("random_seed", 0)),
            max_iterations=_positive_int(gmm.get("max_iterations", 10), "GMM max_iterations"),
            tolerance=_finite(gmm.get("tolerance", 1e-2), "GMM tolerance", strict=True),
            covariance_regularization=_finite(gmm.get("covariance_regularization", 5e-4), "GMM covariance regularization"),
            minimum_mean_separation=_finite(gmm.get("minimum_mean_separation", 1e-6), "GMM minimum mean separation"),
            history_name=history_name,
            high_noise_rate=_finite(history.get("high_noise_rate", 0.9), "high noise rate"),
            history_window_epochs=_positive_int(history.get("window_epochs", 5), "loss history window"),
        )
        mixmatch = _mapping(values.get("mixmatch"), "dividemix.mixmatch")
        if str(mixmatch.get("mixup_lambda_scope", "minibatch")).lower() != "minibatch":
            raise ValueError("DivideMix MixUp lambda scope must be minibatch")
        objective = _mapping(values.get("objective"), "dividemix.objective")
        training = _mapping(values.get("training"), "dividemix.training")
        inference = _mapping(values.get("inference", {}), "dividemix.inference")
        ensemble = str(inference.get("ensemble", "official_logits_sum")).strip().lower()
        if ensemble != "official_logits_sum":
            raise ValueError("DivideMix first version requires official_logits_sum ensemble")
        evaluation = _mapping(config.get("evaluation"), "evaluation")
        if str(evaluation.get("selection_split", "validation")).lower() != "validation":
            raise ValueError("DivideMix selects paired best checkpoints on validation")
        if str(config.get("noise", {}).get("validation_targets", "")).lower() != "noisy":
            raise ValueError("DivideMix requires noisy validation targets")
        incompatible = {"selector", "weight_provider", "target_provider", "parameter_update", "pipeline"}
        present = sorted(incompatible.intersection(config))
        if present:
            raise ValueError("DivideMix owns its complete pipeline and cannot combine: " + ", ".join(present))
        return cls(
            fidelity=fidelity,
            peer_seed_offset=offset,
            warmup_epochs=warmup_epochs,
            confidence_penalty_weight=confidence,
            gmm=gmm_config,
            augmentations=_positive_int(mixmatch.get("augmentations", 2), "MixMatch augmentations"),
            temperature=_finite(mixmatch.get("temperature", 0.5), "MixMatch temperature", strict=True),
            mixup_alpha=_finite(mixmatch.get("mixup_alpha", 4.0), "MixUp alpha", strict=True),
            lambda_u=_finite(objective.get("lambda_u", 25.0), "lambda_u"),
            lambda_r=_finite(objective.get("lambda_r", 1.0), "lambda_r"),
            rampup_epochs=_positive_int(objective.get("rampup_epochs", 16), "rampup epochs"),
            training_epochs=_positive_int(training.get("epochs"), "dividemix.training.epochs"),
            ensemble=ensemble,
        )

    @property
    def identity_hash(self) -> str:
        values = asdict(self)
        # The main-stage target may be extended explicitly on resume.
        values.pop("training_epochs")
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
