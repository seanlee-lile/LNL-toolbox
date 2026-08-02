from __future__ import annotations

"""Part-dependent transition estimation and compact instance artifacts."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lnl_toolbox.training.snapshots import FeatureSnapshot
from .estimators import PosteriorSnapshot, select_anchor_candidates


def project_probability_simplex(values: np.ndarray) -> np.ndarray:
    """Project the last axis onto the probability simplex."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] == 0:
        raise ValueError("values must have a non-empty final axis")
    flat = array.reshape(-1, array.shape[-1])
    ordered = np.sort(flat, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    ranks = np.arange(1, flat.shape[1] + 1, dtype=np.float64)
    valid = ordered - cumulative / ranks > 0.0
    rho = valid.sum(axis=1) - 1
    theta = cumulative[np.arange(flat.shape[0]), rho] / (rho + 1.0)
    projected = np.maximum(flat - theta[:, None], 0.0)
    return projected.reshape(array.shape)


def fit_part_representation(
    features: np.ndarray,
    num_parts: int,
    *,
    seed: int = 0,
    iterations: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Eq. (1) with non-negative simplex coefficients.

    Returns ``parts[D, R]`` and ``coefficients[N, R]``.  Alternating projected
    least squares keeps the implementation dependency-free and deterministic.
    """

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("features must have shape [N, D]")
    if not np.isfinite(values).all():
        raise ValueError("features must be finite")
    if not 1 <= int(num_parts) <= min(values.shape):
        raise ValueError("num_parts must be within [1, min(N, D)]")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    random = np.random.default_rng(seed)
    coefficients = random.dirichlet(np.ones(int(num_parts)), size=values.shape[0])
    parts = np.zeros((values.shape[1], int(num_parts)), dtype=np.float64)
    for _ in range(int(iterations)):
        parts = np.linalg.lstsq(coefficients, values, rcond=None)[0].T
        gram = parts.T @ parts
        lipschitz = max(float(np.linalg.norm(gram, ord=2)), 1e-12)
        coefficients = project_probability_simplex(
            coefficients - ((coefficients @ parts.T - values) @ parts) / lipschitz
        )
    return parts, coefficients


def fit_part_transition_matrices(
    anchor_coefficients: np.ndarray,
    anchor_posteriors: np.ndarray,
) -> np.ndarray:
    """Fit Eq. (4) and return row-stochastic part matrices ``[R,C,C]``."""

    coefficients = np.asarray(anchor_coefficients, dtype=np.float64)
    posteriors = np.asarray(anchor_posteriors, dtype=np.float64)
    if coefficients.ndim != 3:
        raise ValueError("anchor_coefficients must have shape [C, K, R]")
    classes, candidates, parts = coefficients.shape
    if posteriors.shape != (classes, candidates, classes):
        raise ValueError("anchor_posteriors must have shape [C, K, C]")
    if candidates < parts:
        raise ValueError("at least num_parts anchor candidates are required per class")
    if not np.isfinite(coefficients).all() or not np.isfinite(posteriors).all():
        raise ValueError("anchor values must be finite")

    matrices = np.empty((parts, classes, classes), dtype=np.float64)
    for clean_class in range(classes):
        design = coefficients[clean_class]
        if np.linalg.matrix_rank(design) < parts:
            raise ValueError(
                f"anchor coefficients for class {clean_class} are rank deficient"
            )
        fitted = np.linalg.lstsq(design, posteriors[clean_class], rcond=None)[0]
        matrices[:, clean_class, :] = project_probability_simplex(fitted)
    return matrices


@dataclass(frozen=True, slots=True)
class PartTransitionArtifact:
    """Compact ``T(x)=sum_r h_r(x) P_r`` artifact aligned by global index."""

    parts: np.ndarray
    coefficients: np.ndarray
    part_matrices: np.ndarray
    global_indices: np.ndarray
    feature_snapshot_hash: str
    posterior_snapshot_hash: str
    anchor_indices: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    convention: str = "clean_to_noisy_row"

    def __post_init__(self) -> None:
        parts = np.asarray(self.parts, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        matrices = np.asarray(self.part_matrices, dtype=np.float64)
        indices = np.asarray(self.global_indices, dtype=np.int64)
        anchors = np.asarray(self.anchor_indices, dtype=np.int64)
        if self.version != "1.0":
            raise ValueError(f"unsupported part transition artifact version: {self.version!r}")
        if self.convention != "clean_to_noisy_row":
            raise ValueError("part transition convention must be 'clean_to_noisy_row'")
        if parts.ndim != 2:
            raise ValueError("parts must have shape [D, R]")
        samples, num_parts = coefficients.shape if coefficients.ndim == 2 else (-1, -1)
        if parts.shape[1] != num_parts or samples <= 0:
            raise ValueError("coefficients must have shape [N, R] matching parts")
        if matrices.ndim != 3 or matrices.shape[0] != num_parts or matrices.shape[1] != matrices.shape[2]:
            raise ValueError("part_matrices must have shape [R, C, C]")
        if indices.shape != (samples,) or np.unique(indices).size != samples:
            raise ValueError("global_indices must be unique shape [N]")
        if anchors.ndim != 2 or anchors.shape[0] != matrices.shape[1]:
            raise ValueError("anchor_indices must have shape [C, K]")
        for name, value in (("feature", self.feature_snapshot_hash), ("posterior", self.posterior_snapshot_hash)):
            if len(value) != 64:
                raise ValueError(f"{name}_snapshot_hash must be an SHA-256 digest")
            try:
                bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(f"{name}_snapshot_hash must be an SHA-256 digest") from exc
        if not np.isfinite(parts).all() or not np.isfinite(coefficients).all() or not np.isfinite(matrices).all():
            raise ValueError("artifact arrays must be finite")
        if (coefficients < -1e-10).any() or not np.allclose(coefficients.sum(axis=1), 1.0):
            raise ValueError("coefficient rows must be probabilities")
        if (matrices < -1e-10).any() or not np.allclose(matrices.sum(axis=2), 1.0):
            raise ValueError("every part transition row must be a probability")
        order = np.argsort(indices, kind="stable")
        metadata = MappingProxyType(json.loads(json.dumps(dict(self.metadata), sort_keys=True)))
        for name, value in (
            ("parts", parts), ("coefficients", coefficients[order]),
            ("part_matrices", matrices), ("global_indices", indices[order]),
            ("anchor_indices", anchors),
        ):
            copy = value.copy()
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)
        object.__setattr__(self, "metadata", metadata)

    @property
    def num_classes(self) -> int:
        return int(self.part_matrices.shape[1])

    @property
    def artifact_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps({
            "version": self.version, "convention": self.convention,
            "feature_snapshot_hash": self.feature_snapshot_hash,
            "posterior_snapshot_hash": self.posterior_snapshot_hash,
            "metadata": dict(self.metadata),
        }, sort_keys=True, separators=(",", ":")).encode())
        for value in (self.parts, self.coefficients, self.part_matrices, self.global_indices, self.anchor_indices):
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    def transitions_for(self, sample_indices: Any, *, device: Any = None, dtype: Any = None):
        import torch

        requested = torch.as_tensor(sample_indices).detach().cpu().numpy().astype(np.int64)
        if requested.ndim != 1:
            raise ValueError("sample_indices must have shape [B]")
        positions = np.searchsorted(self.global_indices, requested)
        valid = positions < self.global_indices.size
        if not np.all(valid) or not np.array_equal(self.global_indices[positions[valid]], requested[valid]):
            raise KeyError("instance transition artifact does not cover every sample index")
        transitions = np.einsum("br,rcd->bcd", self.coefficients[positions], self.part_matrices)
        return torch.as_tensor(transitions.copy(), device=device, dtype=dtype or torch.float32)

    def transition_for(self, inputs: Any, sample_indices: Any, *, device: Any = None, dtype: Any = None):
        return self.transitions_for(sample_indices, device=device, dtype=dtype)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version, "convention": self.convention,
            "feature_snapshot_hash": self.feature_snapshot_hash,
            "posterior_snapshot_hash": self.posterior_snapshot_hash,
            "metadata": dict(self.metadata), "artifact_hash": self.artifact_hash,
        }
        np.savez_compressed(destination, parts=self.parts, coefficients=self.coefficients,
            part_matrices=self.part_matrices, global_indices=self.global_indices,
            anchor_indices=self.anchor_indices, metadata_json=np.array(json.dumps(payload, sort_keys=True)))

    @classmethod
    def load(cls, path: str | Path) -> "PartTransitionArtifact":
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["metadata_json"].item()))
            artifact = cls(data["parts"], data["coefficients"], data["part_matrices"],
                data["global_indices"], payload["feature_snapshot_hash"],
                payload["posterior_snapshot_hash"], data["anchor_indices"],
                payload.get("metadata", {}), payload["version"], payload["convention"])
            if artifact.artifact_hash != payload.get("artifact_hash"):
                raise ValueError("part transition artifact hash does not match contents")
            return artifact


@dataclass(frozen=True, slots=True)
class PartTransitionEstimator:
    num_parts: int
    anchor_candidates: int
    representation_seed: int = 0
    representation_iterations: int = 200

    def estimate(self, features: FeatureSnapshot, posteriors: PosteriorSnapshot) -> PartTransitionArtifact:
        if features.dataset != posteriors.dataset or features.split != posteriors.split:
            raise ValueError("feature and posterior snapshots must describe the same dataset split")
        if not np.array_equal(features.global_indices, posteriors.global_indices):
            raise ValueError("feature and posterior snapshots must use identical global indices")
        parts, coefficients = fit_part_representation(features.features, self.num_parts,
            seed=self.representation_seed, iterations=self.representation_iterations)
        anchor_positions, anchor_indices = select_anchor_candidates(posteriors, self.anchor_candidates)
        anchor_coefficients = coefficients[anchor_positions]
        anchor_posteriors = posteriors.noisy_probabilities[anchor_positions]
        matrices = fit_part_transition_matrices(anchor_coefficients, anchor_posteriors)
        return PartTransitionArtifact(parts, coefficients, matrices, features.global_indices,
            features.snapshot_hash, posteriors.snapshot_hash, anchor_indices,
            metadata={"estimator": "part_dependent", "num_parts": self.num_parts,
                "anchor_candidates": self.anchor_candidates,
                "representation_seed": self.representation_seed,
                "representation_iterations": self.representation_iterations})
