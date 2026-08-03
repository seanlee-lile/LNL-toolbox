from __future__ import annotations

"""Paper-exact class-wise virtual auxiliary statistics for CWD."""

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


def _swap_rows(classes: int, first: int, second: int) -> np.ndarray:
    """Return the elementary matrix that swaps two one-hot rows."""

    result = np.eye(classes, dtype=np.float64)
    result[[first, second]] = result[[second, first]]
    return result


def _clean_prior(observed_prior: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Solve Eq. (19), ``observed_prior = transition.T @ clean_prior``."""

    prior = np.linalg.solve(transition.T, observed_prior)
    tolerance = 1e-8
    if not np.isfinite(prior).all() or (prior < -tolerance).any():
        raise ValueError("label-flip matrix and observed labels imply an invalid clean prior")
    prior = np.maximum(prior, 0.0)
    total = prior.sum()
    if total <= 0.0:
        raise ValueError("estimated clean class prior has zero mass")
    return prior / total


def _virtual_system(
    transition: np.ndarray,
    clean_prior: np.ndarray,
    clean_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Eqs. (21)-(29) for one virtual auxiliary set."""

    classes = transition.shape[0]
    virtual_prior = clean_prior @ transition
    virtual_prior -= clean_prior[clean_class] * transition[clean_class]
    virtual_prior[clean_class] += clean_prior[clean_class]
    denominator = virtual_prior[clean_class]
    if denominator <= 0.0:
        raise ValueError("CWD virtual class prior must be positive")

    virtual_flip = np.eye(classes, dtype=np.float64)
    for target in range(classes):
        if target == clean_class:
            continue
        virtual_flip[clean_class, target] = (
            clean_prior[clean_class] * transition[clean_class, target]
            / denominator
        )
        virtual_flip[target, clean_class] = 0.0
    virtual_flip[clean_class, clean_class] = (
        1.0 - virtual_flip[clean_class].sum() + virtual_flip[clean_class, clean_class]
    )
    if virtual_flip[clean_class, clean_class] < -1e-8:
        raise ValueError("CWD virtual label-flip matrix is invalid")
    virtual_flip[clean_class, clean_class] = max(
        0.0, virtual_flip[clean_class, clean_class]
    )

    coefficient = np.zeros((classes, classes), dtype=np.float64)
    for source in range(classes):
        for target in range(classes):
            coefficient += (
                virtual_prior[source]
                * virtual_flip[source, target]
                * _swap_rows(classes, source, target).T
            )
    return virtual_prior, virtual_flip, coefficient


class CWDEstimator:
    """Estimate the clean centroid through CWD's virtual auxiliary sets."""

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
        features = np.asarray(snapshot.features, dtype=np.float64)
        labels = snapshot.noisy_targets
        configured = label_flip_matrix if label_flip_matrix is not None else self.label_flip_matrix
        classes = int(np.asarray(configured).shape[0]) if configured is not None else int(labels.max() + 1)
        if classes < 2 or labels.min() < 0 or labels.max() >= classes:
            raise ValueError("CWD requires aligned labels for at least two classes")
        transition = _matrix(label_flip_matrix if label_flip_matrix is not None else self.label_flip_matrix, classes)
        observed_counts = np.bincount(labels, minlength=classes).astype(np.float64)
        observed_prior = observed_counts / len(labels)
        prior = (
            _clean_prior(observed_prior, transition)
            if class_prior is None
            else np.asarray(class_prior, dtype=np.float64)
        )
        if prior.shape != (classes,) or (prior < 0).any() or not np.isfinite(prior).all():
            raise ValueError("class_prior must be a finite non-negative [C] vector")
        prior_total = prior.sum()
        if prior_total <= 0.0:
            raise ValueError("class_prior must have positive mass")
        prior = prior / prior_total

        observed_centroid = np.zeros((features.shape[1], classes), dtype=np.float64)
        for label in range(classes):
            observed_centroid[:, label] = features[labels == label].sum(axis=0) / len(labels)

        virtual_centroids: list[np.ndarray] = []
        virtual_priors: list[list[float]] = []
        virtual_flip_matrices: list[list[list[float]]] = []
        coefficient_matrices: list[list[list[float]]] = []
        coefficient_pseudoinverses: list[list[list[float]]] = []
        for clean_class in range(classes):
            virtual_prior, virtual_flip, coefficient = _virtual_system(
                transition, prior, clean_class
            )
            coefficient_pinv = np.linalg.pinv(coefficient, rcond=max(self.ridge, 1e-15))
            virtual_centroids.append(observed_centroid @ coefficient_pinv)
            virtual_priors.append(virtual_prior.tolist())
            virtual_flip_matrices.append(virtual_flip.tolist())
            coefficient_matrices.append(coefficient.tolist())
            coefficient_pseudoinverses.append(coefficient_pinv.tolist())
        clean_centroid = (
            np.sum(virtual_centroids, axis=0) - (classes - 1) * observed_centroid
        )
        centroids = clean_centroid.T
        if not np.isfinite(centroids).all():
            raise ValueError("CWD produced non-finite class centroids")
        return CWDStatisticArtifact(
            values=centroids,
            estimator=self.name,
            metadata={
                "dataset": snapshot.dataset,
                "split": snapshot.split,
                "class_prior": prior.tolist(),
                "observed_class_prior": observed_prior.tolist(),
                "label_flip_matrix": transition.tolist(),
                "virtual_class_priors": virtual_priors,
                "virtual_flip_matrices": virtual_flip_matrices,
                "coefficient_matrices": coefficient_matrices,
                "coefficient_pseudoinverses": coefficient_pseudoinverses,
                "transition_rank": int(np.linalg.matrix_rank(transition)),
                "virtual_samples": int(len(labels)),
                "cwd_equations": "19,21-30",
                "source_snapshot_hash": snapshot.snapshot_hash,
            },
        )

    def virtual_auxiliary_set(self, snapshot: FeatureSnapshot, *, label_flip_matrix: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        artifact = self.estimate(snapshot, label_flip_matrix=label_flip_matrix)
        labels = np.arange(artifact.values.shape[0], dtype=np.int64)
        return artifact.values, labels


ClassWiseVirtualSetEstimator = CWDEstimator

__all__ = ["CWDEstimator", "ClassWiseVirtualSetEstimator"]
