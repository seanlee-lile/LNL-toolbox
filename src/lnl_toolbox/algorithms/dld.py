from __future__ import annotations

"""Stateless pre-correction utilities for Directional Label Diffusion.

The official DLD workflow first builds label distributions from two feature
views, separates samples with a two-component GMM, and then trains a
conditional diffusion model on hard or soft targets.  This module owns the
first, artifact-producing part of that workflow; it intentionally does not
own a model or a training loop.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def _as_features(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape [N,D] and be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _normalize_distribution(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.maximum(values, 0.0)
    denominator = values.sum(axis=1, keepdims=True)
    return values / np.maximum(denominator, 1e-12)


def build_knn_label_distribution(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 50,
    use_cosine: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a class distribution and neighbor indices for every sample.

    Neighbor rows are returned in the same order as ``features``.  The
    implementation uses only observed labels and feature representations;
    clean labels never enter this interface.
    """

    features = _as_features(features, "features")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (features.shape[0],):
        raise ValueError("labels must align with features")
    if labels.min(initial=0) < 0:
        raise ValueError("labels must be non-negative")
    classes = int(labels.max(initial=0)) + 1
    if classes == 0:
        raise ValueError("labels must contain at least one class")
    if isinstance(k, bool) or int(k) != k or int(k) <= 0:
        raise ValueError("k must be a positive integer")
    k = min(int(k), max(1, features.shape[0] - 1))

    if use_cosine:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        normalized = features / np.maximum(norms, 1e-12)
        scores = normalized @ normalized.T
        distances = 1.0 - np.clip(scores, -1.0, 1.0)
    else:
        squared = np.sum(features * features, axis=1, keepdims=True)
        distances = squared + squared.T - 2.0 * (features @ features.T)
        distances = np.sqrt(np.maximum(distances, 0.0))
    np.fill_diagonal(distances, np.inf)

    if features.shape[0] == 1:
        neighbors = np.zeros((1, 1), dtype=np.int64)
        distribution = np.zeros((1, classes), dtype=np.float32)
        distribution[0, labels[0]] = 1.0
        return distribution, neighbors

    candidates = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    row = np.arange(features.shape[0])[:, None]
    order = np.argsort(distances[row, candidates], axis=1)
    neighbors = np.take_along_axis(candidates, order, axis=1).astype(np.int64)
    selected_distances = np.take_along_axis(distances, neighbors, axis=1)
    weights = 1.0 / np.maximum(selected_distances, 1e-6)
    weights[~np.isfinite(weights)] = 0.0
    distribution = np.zeros((features.shape[0], classes), dtype=np.float32)
    np.add.at(
        distribution,
        (np.repeat(np.arange(features.shape[0]), k), labels[neighbors].reshape(-1)),
        weights.reshape(-1).astype(np.float32),
    )
    distribution = _normalize_distribution(distribution)
    return distribution, neighbors


def _gmm_clean_partition(values: np.ndarray, *, iterations: int = 32) -> np.ndarray:
    """Return the lower-loss component of a deterministic two-Gaussian fit."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("GMM values must be a finite non-empty vector")
    if np.ptp(values) <= 1e-12:
        return np.ones(values.shape, dtype=bool)
    means = np.array([values.min(), values.max()], dtype=np.float64)
    variance = max(float(np.var(values)), 1e-6)
    variances = np.array([variance, variance], dtype=np.float64)
    priors = np.array([0.5, 0.5], dtype=np.float64)
    for _ in range(iterations):
        log_probability = np.log(np.maximum(priors, 1e-12))[None, :] - 0.5 * (
            np.log(2.0 * np.pi * np.maximum(variances, 1e-12))[None, :]
            + (values[:, None] - means[None, :]) ** 2
            / np.maximum(variances, 1e-12)[None, :]
        )
        log_probability -= log_probability.max(axis=1, keepdims=True)
        responsibility = np.exp(log_probability)
        responsibility /= np.maximum(responsibility.sum(axis=1, keepdims=True), 1e-12)
        counts = responsibility.sum(axis=0)
        priors = counts / values.size
        means = (responsibility * values[:, None]).sum(axis=0) / np.maximum(counts, 1e-12)
        variances = (
            responsibility * (values[:, None] - means[None, :]) ** 2
        ).sum(axis=0) / np.maximum(counts, 1e-12)
        variances = np.maximum(variances, 1e-6)
    clean_component = int(np.argmin(means))
    return responsibility.argmax(axis=1) == clean_component


@dataclass(frozen=True, slots=True)
class DLDPrecorrectionArtifact:
    """Immutable, hashable targets consumed by the DLD training runner."""

    global_indices: np.ndarray
    partition: np.ndarray
    weak_targets: np.ndarray
    strong_targets: np.ndarray
    loss_weights: np.ndarray
    weak_graph_hash: str
    strong_graph_hash: str
    seed: int = 0
    artifact_hash: str = ""

    def __post_init__(self) -> None:
        indices = np.asarray(self.global_indices, dtype=np.int64)
        partition = np.asarray(self.partition, dtype=np.int64)
        weak = np.asarray(self.weak_targets, dtype=np.float32)
        strong = np.asarray(self.strong_targets, dtype=np.float32)
        weights = np.asarray(self.loss_weights, dtype=np.float32)
        if indices.ndim != 1 or indices.size == 0 or np.unique(indices).size != indices.size:
            raise ValueError("global_indices must be a non-empty unique vector")
        if partition.shape != indices.shape or weights.shape != indices.shape:
            raise ValueError("partition and loss_weights must align with global_indices")
        if weak.ndim != 2 or strong.shape != weak.shape or weak.shape[0] != indices.size:
            raise ValueError("DLD targets must have shape [N,C]")
        if np.any((partition < 0) | (partition > 2)):
            raise ValueError("DLD partition values must be 0, 1, or 2")
        if not np.isfinite(weak).all() or not np.isfinite(strong).all() or not np.isfinite(weights).all():
            raise ValueError("DLD artifact values must be finite")
        if np.any(weights <= 0.0):
            raise ValueError("DLD loss weights must be positive")
        for target in (weak, strong):
            if np.any(target < -1e-6) or not np.allclose(target.sum(axis=1), 1.0, atol=1e-4):
                raise ValueError("DLD targets must be probability distributions")
        expected = _array_hash(indices, partition, weak, strong, weights)
        if self.artifact_hash and self.artifact_hash != expected:
            raise ValueError("DLD artifact hash does not match its arrays")
        object.__setattr__(self, "global_indices", indices.copy())
        object.__setattr__(self, "partition", partition.copy())
        object.__setattr__(self, "weak_targets", weak.copy())
        object.__setattr__(self, "strong_targets", strong.copy())
        object.__setattr__(self, "loss_weights", weights.copy())
        object.__setattr__(self, "artifact_hash", expected)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            global_indices=self.global_indices,
            partition=self.partition,
            weak_targets=self.weak_targets,
            strong_targets=self.strong_targets,
            loss_weights=self.loss_weights,
            weak_graph_hash=np.asarray(self.weak_graph_hash),
            strong_graph_hash=np.asarray(self.strong_graph_hash),
            seed=np.asarray(self.seed, dtype=np.int64),
            artifact_hash=np.asarray(self.artifact_hash),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "DLDPrecorrectionArtifact":
        with np.load(Path(path), allow_pickle=False) as payload:
            artifact = cls(
                payload["global_indices"],
                payload["partition"],
                payload["weak_targets"],
                payload["strong_targets"],
                payload["loss_weights"],
                str(payload["weak_graph_hash"].item()),
                str(payload["strong_graph_hash"].item()),
                int(payload["seed"].item()),
                str(payload["artifact_hash"].item()),
            )
        return artifact


def precorrect_two_views(
    weak_features: np.ndarray,
    strong_features: np.ndarray,
    noisy_labels: np.ndarray,
    global_indices: np.ndarray,
    *,
    k: int = 50,
    use_cosine: bool = True,
    seed: int = 0,
) -> DLDPrecorrectionArtifact:
    """Build official-style hard/soft targets from two noisy-label views."""

    weak_features = _as_features(weak_features, "weak_features")
    strong_features = _as_features(strong_features, "strong_features")
    if strong_features.shape != weak_features.shape:
        raise ValueError("weak and strong features must have the same shape")
    labels = np.asarray(noisy_labels, dtype=np.int64)
    indices = np.asarray(global_indices, dtype=np.int64)
    if labels.shape != (weak_features.shape[0],) or indices.shape != labels.shape:
        raise ValueError("noisy_labels and global_indices must align with features")
    if np.unique(indices).size != indices.size:
        raise ValueError("global_indices must be unique")

    weak_distribution, weak_neighbors = build_knn_label_distribution(
        weak_features, labels, k=k, use_cosine=use_cosine
    )
    strong_distribution, strong_neighbors = build_knn_label_distribution(
        strong_features, labels, k=k, use_cosine=use_cosine
    )
    epsilon = 1e-8
    weak_log = np.log(np.maximum(weak_distribution, epsilon))
    strong_log = np.log(np.maximum(strong_distribution, epsilon))
    kl = 0.5 * (
        np.sum(weak_distribution * (weak_log - strong_log), axis=1)
        + np.sum(strong_distribution * (strong_log - weak_log), axis=1)
    )
    clean = _gmm_clean_partition(kl)
    pseudo = ((weak_distribution + strong_distribution) * 0.5).argmax(axis=1)
    weak_targets = np.empty_like(weak_distribution)
    strong_targets = np.empty_like(strong_distribution)
    partition = np.full(labels.shape, 2, dtype=np.int64)
    for row in range(labels.size):
        if clean[row]:
            label = int(labels[row]) if int(labels[row]) == int(pseudo[row]) else int(pseudo[row])
            weak_targets[row] = 0.0
            strong_targets[row] = 0.0
            weak_targets[row, label] = 1.0
            strong_targets[row, label] = 1.0
            partition[row] = 0 if label == int(labels[row]) else 1
        else:
            weak_targets[row] = weak_distribution[row]
            strong_targets[row] = strong_distribution[row]
    confidence = 0.5 * (weak_targets.max(axis=1) + strong_targets.max(axis=1))
    weights = np.maximum(confidence, 1e-3).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-12)
    return DLDPrecorrectionArtifact(
        indices,
        partition,
        weak_targets,
        strong_targets,
        weights,
        _array_hash(weak_features, weak_neighbors),
        _array_hash(strong_features, strong_neighbors),
        int(seed),
    )


def weighted_mse(
    prediction: Any,
    target: Any,
    weights: Any | None = None,
) -> Any:
    """Return a per-sample weighted MSE objective for diffusion outputs."""

    import torch

    prediction = torch.as_tensor(prediction) if not torch.is_tensor(prediction) else prediction
    target = torch.as_tensor(target, device=prediction.device) if not torch.is_tensor(target) else target
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have matching shape [B,C]")
    loss = (prediction - target).square().mean(dim=1)
    if weights is None:
        return loss
    weights = torch.as_tensor(weights, dtype=loss.dtype, device=loss.device)
    if weights.shape != (prediction.shape[0],):
        raise ValueError("weights must have shape [B]")
    return loss * weights


__all__ = [
    "DLDPrecorrectionArtifact",
    "build_knn_label_distribution",
    "precorrect_two_views",
    "weighted_mse",
]
