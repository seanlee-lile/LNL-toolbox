from __future__ import annotations

"""Class-wise virtual auxiliary statistics for CWD."""

from typing import Any

import numpy as np

from lnl_toolbox.noise.statistics import CWDStatisticArtifact
from lnl_toolbox.training.snapshots import FeatureSnapshot


def _matrix(value: Any, classes: int) -> np.ndarray:
    if value is None:
        return np.eye(classes, dtype=np.float64)
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (classes, classes) or not np.isfinite(result).all():
        raise ValueError("label-flip matrix must have shape [C, C]")
    if (result < 0).any() or not np.allclose(result.sum(axis=1), 1.0):
        raise ValueError("label-flip matrix must be row-stochastic")
    if np.linalg.matrix_rank(result) < classes:
        raise ValueError("label-flip matrix must be identifiable and invertible")
    return result


class CWDEstimator:
    """Estimate class centroids after constructing a virtual clean set."""

    name = "cwd"

    def __init__(self, label_flip_matrix: np.ndarray | None = None, ridge: float = 1e-8) -> None:
        if ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        self.label_flip_matrix = None if label_flip_matrix is None else np.asarray(label_flip_matrix, dtype=np.float64)
        self.ridge = float(ridge)

    def estimate(
        self,
        snapshot: FeatureSnapshot,
        *,
        label_flip_matrix: np.ndarray | None = None,
        class_prior: np.ndarray | None = None,
    ) -> CWDStatisticArtifact:
        features = snapshot.features
        labels = snapshot.noisy_targets
        classes = int(max(labels.max(), 1) + 1)
        transition = _matrix(label_flip_matrix if label_flip_matrix is not None else self.label_flip_matrix, classes)
        observed_counts = np.bincount(labels, minlength=classes).astype(np.float64)
        observed_sums = np.zeros((classes, features.shape[1]), dtype=np.float64)
        for label in range(classes):
            observed_sums[label] = features[labels == label].sum(axis=0)
        inverse_transpose = np.linalg.inv(transition.T)
        corrected_sums = inverse_transpose @ observed_sums
        corrected_counts = inverse_transpose @ observed_counts
        if class_prior is not None:
            prior = np.asarray(class_prior, dtype=np.float64)
            if prior.shape != (classes,) or (prior < 0).any() or not np.isfinite(prior).all():
                raise ValueError("class_prior must be a finite non-negative [C] vector")
            corrected_counts = prior / max(prior.sum(), 1e-12) * len(labels)
        corrected_counts = np.maximum(corrected_counts, 1e-12)
        centroids = corrected_sums / corrected_counts[:, None]
        if not np.isfinite(centroids).all():
            raise ValueError("CWD produced non-finite class centroids")
        return CWDStatisticArtifact(
            values=centroids,
            estimator=self.name,
            metadata={
                "dataset": snapshot.dataset,
                "split": snapshot.split,
                "class_prior": (corrected_counts / corrected_counts.sum()).tolist(),
                "label_flip_matrix": transition.tolist(),
                "pseudo_inverse": np.linalg.inv(transition).tolist(),
                "transition_rank": int(np.linalg.matrix_rank(transition)),
                "virtual_samples": int(len(labels)),
                "source_snapshot_hash": snapshot.snapshot_hash,
            },
        )

    def virtual_auxiliary_set(self, snapshot: FeatureSnapshot, *, label_flip_matrix: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        artifact = self.estimate(snapshot, label_flip_matrix=label_flip_matrix)
        labels = np.arange(artifact.values.shape[0], dtype=np.int64)
        return artifact.values, labels


ClassWiseVirtualSetEstimator = CWDEstimator

__all__ = ["CWDEstimator", "ClassWiseVirtualSetEstimator"]
