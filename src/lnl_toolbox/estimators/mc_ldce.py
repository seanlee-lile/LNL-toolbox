from __future__ import annotations

"""Centroid reconstruction for MC-LDCE (SDM 2022)."""

import math

import numpy as np

from lnl_toolbox.noise.statistics import StatisticArtifact
from lnl_toolbox.noise.transition import TransitionArtifact
from lnl_toolbox.training.snapshots import FeatureSnapshot


def _swap(classes: int, source: int, target: int) -> np.ndarray:
    matrix = np.eye(classes, dtype=np.float64)
    matrix[[source, target]] = matrix[[target, source]]
    return matrix


def estimate_clean_prior(observed_prior: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Solve ``observed_prior = clean_prior @ transition``."""

    observed = np.asarray(observed_prior, dtype=np.float64)
    matrix = np.asarray(transition, dtype=np.float64)
    if matrix.shape != (observed.size, observed.size):
        raise ValueError("transition and observed prior dimensions differ")
    prior = np.linalg.solve(matrix.T, observed)
    if not np.isfinite(prior).all() or (prior < -1e-8).any():
        raise ValueError("transition implies an invalid clean class prior")
    prior = np.maximum(prior, 0.0)
    total = float(prior.sum())
    if total <= 0.0:
        raise ValueError("clean class prior has zero mass")
    return prior / total


def build_label_imputation_matrix(
    clean_prior: np.ndarray, transition: np.ndarray
) -> np.ndarray:
    """Build the paper's coefficient matrix ``M``."""

    prior = np.asarray(clean_prior, dtype=np.float64)
    matrix = np.asarray(transition, dtype=np.float64)
    if matrix.shape != (prior.size, prior.size):
        raise ValueError("transition and clean prior dimensions differ")
    result = np.zeros_like(matrix)
    for source in range(prior.size):
        for target in range(prior.size):
            result += prior[source] * matrix[source, target] * _swap(
                prior.size, source, target
            ).T
    return result


class MCLDCEEstimator:
    """Recover the clean joint feature-label centroid without clean labels."""

    name = "mc_ldce"

    def __init__(self, *, rcond: float = 1e-8, condition_limit: float = 1e8) -> None:
        rcond = float(rcond)
        condition_limit = float(condition_limit)
        if not math.isfinite(rcond) or rcond <= 0.0:
            raise ValueError("rcond must be finite and positive")
        if not math.isfinite(condition_limit) or condition_limit <= 1.0:
            raise ValueError("condition_limit must be finite and greater than one")
        self.rcond = rcond
        self.condition_limit = condition_limit

    def estimate(
        self, snapshot: FeatureSnapshot, transition: TransitionArtifact
    ) -> StatisticArtifact:
        labels = np.asarray(snapshot.noisy_targets, dtype=np.int64)
        classes = transition.num_classes
        if labels.min() < 0 or labels.max() >= classes:
            raise ValueError("snapshot labels do not match transition classes")
        matrix = np.asarray(transition.matrix, dtype=np.float64)
        rank = int(np.linalg.matrix_rank(matrix))
        condition = float(np.linalg.cond(matrix))
        if rank != classes or not math.isfinite(condition):
            raise ValueError("transition is not identifiable")
        observed_prior = np.bincount(labels, minlength=classes) / labels.size
        clean_prior = estimate_clean_prior(observed_prior, matrix)
        coefficient = build_label_imputation_matrix(clean_prior, matrix)
        coefficient_rank = int(np.linalg.matrix_rank(coefficient))
        coefficient_condition = float(np.linalg.cond(coefficient))
        if coefficient_rank != classes:
            raise ValueError("MC-LDCE coefficient matrix is not identifiable")
        if not math.isfinite(coefficient_condition) or coefficient_condition > self.condition_limit:
            raise ValueError("MC-LDCE coefficient matrix exceeds condition limit")
        one_hot = np.eye(classes, dtype=np.float64)[labels]
        noisy_centroid = snapshot.features.T @ one_hot / labels.size
        clean_centroid = noisy_centroid @ np.linalg.pinv(coefficient, rcond=self.rcond)
        return StatisticArtifact(
            values=clean_centroid.T,
            estimator=self.name,
            metadata={
                "dataset": snapshot.dataset,
                "split": snapshot.split,
                "source_snapshot_hash": snapshot.snapshot_hash,
                "transition_artifact_hash": transition.artifact_hash,
                "clean_class_prior": clean_prior.tolist(),
                "observed_class_prior": observed_prior.tolist(),
                "coefficient_matrix": coefficient.tolist(),
                "coefficient_rank": coefficient_rank,
                "coefficient_condition": coefficient_condition,
                "transition_rank": rank,
                "transition_condition": condition,
                "feature_dimension": int(snapshot.features.shape[1]),
                "num_classes": int(classes),
                "coefficient_convention": "M=sum_i pi_i sum_j T_ij K_i_to_j^T",
                "centroid_recovery": "mu_clean=mu_noisy@pinv(M)",
                "rcond": self.rcond,
            },
        )


__all__ = [
    "MCLDCEEstimator",
    "build_label_imputation_matrix",
    "estimate_clean_prior",
]
