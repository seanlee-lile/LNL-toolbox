from __future__ import annotations

"""Posterior backend contracts and paper-exact raw-min rate estimation."""

from hashlib import sha256
import json
from typing import Any, Mapping, Protocol

import numpy as np

from lnl_toolbox.data.binary_synthetic import validate_zero_one_labels
from lnl_toolbox.noise.estimators import PosteriorSnapshot

from .artifacts import NoiseRateArtifact


def _kernel_density_class() -> Any:
    try:
        from sklearn.neighbors import KernelDensity
    except ImportError as exc:
        raise ImportError(
            "KDE posterior fitting requires the optional training dependency; "
            'install it with: python -m pip install -e ".[train]"'
        ) from exc
    return KernelDensity


def validate_binary_posterior_snapshot(
    snapshot: PosteriorSnapshot,
    *,
    require_both_noisy_classes: bool = True,
) -> PosteriorSnapshot:
    """Apply binary method constraints to the generic snapshot contract."""

    if not isinstance(snapshot, PosteriorSnapshot):
        raise TypeError("importance reweighting requires a PosteriorSnapshot")
    if snapshot.num_classes != 2:
        raise ValueError(
            "importance reweighting posterior snapshot must have shape [N, 2]"
        )
    validate_zero_one_labels(
        snapshot.noisy_targets,
        owner="importance reweighting posterior snapshot",
        require_both_classes=require_both_noisy_classes,
    )
    if np.unique(snapshot.global_indices).size != snapshot.num_samples:
        raise ValueError(
            "importance reweighting posterior snapshot indices must be unique"
        )
    return snapshot


class KDEBinaryNoisyPosteriorEstimator:
    """Estimate ``P(noisy label | x)`` using two class-conditional KDEs."""

    def __init__(self, bandwidth: float) -> None:
        self.bandwidth = float(bandwidth)
        if not np.isfinite(self.bandwidth) or self.bandwidth <= 0.0:
            raise ValueError("KDE bandwidth must be finite and positive")

    def identity(self, feature_dimension: int) -> dict[str, Any]:
        return {
            "name": "kde",
            "implementation_version": 1,
            "feature_dimension": int(feature_dimension),
            "bandwidth": self.bandwidth,
        }

    def fit_predict(
        self,
        features: np.ndarray,
        noisy_targets: np.ndarray,
        global_indices: np.ndarray,
        *,
        dataset: str,
        split: str,
    ) -> PosteriorSnapshot:
        values = np.asarray(features, dtype=np.float64)
        targets = validate_zero_one_labels(
            noisy_targets,
            owner="KDE posterior",
            require_both_classes=True,
        )
        indices = np.asarray(global_indices)
        if (
            values.ndim != 2
            or values.shape[0] != targets.size
            or values.shape[1] <= 0
        ):
            raise ValueError("KDE posterior features must have shape [N, D]")
        if not np.isfinite(values).all():
            raise ValueError("KDE posterior features must be finite")
        if indices.shape != targets.shape or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError("KDE posterior indices must be integer with shape [N]")
        indices = indices.astype(np.int64, copy=True)
        if indices.min() < 0 or np.unique(indices).size != indices.size:
            raise ValueError("KDE posterior indices must be non-negative and unique")

        KernelDensity = _kernel_density_class()
        log_joint = np.empty((targets.size, 2), dtype=np.float64)
        for class_index in (0, 1):
            class_features = values[targets == class_index]
            model = KernelDensity(
                kernel="gaussian",
                bandwidth=self.bandwidth,
            ).fit(class_features)
            prior = class_features.shape[0] / targets.size
            log_joint[:, class_index] = model.score_samples(values) + np.log(prior)
        maximum = log_joint.max(axis=1, keepdims=True)
        unnormalized = np.exp(log_joint - maximum)
        probabilities = unnormalized / unnormalized.sum(axis=1, keepdims=True)
        if not np.isfinite(probabilities).all():
            raise ValueError("KDE posterior produced non-finite probabilities")

        order = np.argsort(indices, kind="stable")
        snapshot = PosteriorSnapshot(
            noisy_probabilities=probabilities[order],
            noisy_targets=targets[order],
            global_indices=indices[order],
            dataset=dataset,
            split=split,
        )
        return validate_binary_posterior_snapshot(snapshot)


class BinaryNoisyPosteriorBackend(Protocol):
    """Method-local producer of binary noisy-label posterior snapshots."""

    def identity(self, feature_dimension: int) -> Mapping[str, Any]:
        ...

    def fit_predict(
        self,
        features: np.ndarray,
        noisy_targets: np.ndarray,
        global_indices: np.ndarray,
        *,
        dataset: str,
        split: str,
    ) -> PosteriorSnapshot:
        ...


def posterior_backend_identity_hash(identity: Mapping[str, Any]) -> str:
    """Hash the effective posterior producer configuration."""

    if not isinstance(identity, Mapping):
        raise TypeError("posterior backend identity must be a mapping")
    encoded = json.dumps(
        dict(identity),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_binary_noisy_posterior_backend(
    config: Mapping[str, Any],
) -> BinaryNoisyPosteriorBackend:
    """Build the configured binary posterior producer."""

    name = str(config.get("name", "")).strip().lower()
    if name == "kde":
        return KDEBinaryNoisyPosteriorEstimator(float(config["bandwidth"]))
    if name == "kliep":
        from .kliep import KLIEPBinaryNoisyPosteriorEstimator

        return KLIEPBinaryNoisyPosteriorEstimator(
            bandwidth=float(config["bandwidth"]),
            max_centers=int(config["max_centers"]),
            max_iterations=int(config["max_iterations"]),
            learning_rate=float(config["learning_rate"]),
            tolerance=float(config["tolerance"]),
            epsilon=float(config["epsilon"]),
            seed=int(config["seed"]),
        )
    raise ValueError(f"unsupported binary posterior backend: {name!r}")


class PaperRawMinNoiseRateEstimator:
    """Estimate binary asymmetric RCN rates using the paper's raw minima."""

    def estimate(self, snapshot: PosteriorSnapshot) -> NoiseRateArtifact:
        snapshot = validate_binary_posterior_snapshot(snapshot)
        probabilities = snapshot.noisy_probabilities
        positive_minimum = probabilities[:, 0].min()
        negative_minimum = probabilities[:, 1].min()
        positive_positions = np.flatnonzero(
            probabilities[:, 0] == positive_minimum
        )
        negative_positions = np.flatnonzero(
            probabilities[:, 1] == negative_minimum
        )
        positive_position = int(positive_positions[
            np.argmin(snapshot.global_indices[positive_positions])
        ])
        negative_position = int(negative_positions[
            np.argmin(snapshot.global_indices[negative_positions])
        ])
        return NoiseRateArtifact(
            rho_positive=float(positive_minimum),
            rho_negative=float(negative_minimum),
            positive_extreme_global_index=int(
                snapshot.global_indices[positive_position]
            ),
            negative_extreme_global_index=int(
                snapshot.global_indices[negative_position]
            ),
            source_snapshot_hash=snapshot.snapshot_hash,
            dataset=snapshot.dataset,
            split=snapshot.split,
        )
