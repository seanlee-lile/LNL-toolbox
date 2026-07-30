from __future__ import annotations

"""Paper equations (17)–(23) for multi-class PCSE statistics."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lnl_toolbox.noise.transition import validate_transition_matrix
from lnl_toolbox.training.snapshots import FeatureSnapshot


def _condition_number(matrix: np.ndarray, *, owner: str, limit: float) -> float:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] <= 0.0:
        raise ValueError(f"{owner} is singular")
    condition = float(singular_values[0] / singular_values[-1])
    if not np.isfinite(condition) or condition > limit:
        raise ValueError(
            f"{owner} is ill-conditioned: condition={condition:.6g}, "
            f"limit={limit:.6g}"
        )
    return condition


def recover_clean_priors(
    noisy_priors: np.ndarray,
    transition: np.ndarray,
    *,
    condition_limit: float = 1e8,
) -> np.ndarray:
    """Solve ``T.T @ clean_prior = noisy_prior`` without a pseudo-inverse."""

    matrix = validate_transition_matrix(transition)
    noisy = np.asarray(noisy_priors, dtype=np.float64)
    if noisy.shape != (matrix.shape[0],) or not np.isfinite(noisy).all():
        raise ValueError("noisy priors must have finite shape [C]")
    if (noisy <= 0.0).any() or not np.isclose(noisy.sum(), 1.0):
        raise ValueError("noisy priors must be strictly positive and sum to one")
    _condition_number(matrix, owner="PCSE transition matrix", limit=condition_limit)
    clean = np.linalg.solve(matrix.T, noisy)
    if not np.isfinite(clean).all():
        raise ValueError("PCSE clean-prior solve produced non-finite values")
    if (clean <= 0.0).any():
        raise ValueError("PCSE estimated clean priors must be strictly positive")
    if not np.isclose(clean.sum(), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("PCSE estimated clean priors must sum to one")
    return clean


def build_coefficient_matrix(
    clean_priors: np.ndarray,
    transition: np.ndarray,
) -> np.ndarray:
    """Build paper Eq. (19).

    ``K[i->j]`` is the identity with rows ``i`` and ``j`` exchanged, and
    ``M = sum_ij prior[i] * T[i,j] * K[i->j].T``.
    """

    matrix = validate_transition_matrix(transition)
    priors = np.asarray(clean_priors, dtype=np.float64)
    classes = matrix.shape[0]
    if priors.shape != (classes,) or not np.isfinite(priors).all():
        raise ValueError("clean priors must have finite shape [C]")
    if (priors <= 0.0).any() or not np.isclose(priors.sum(), 1.0):
        raise ValueError("clean priors must be strictly positive and sum to one")
    result = np.zeros_like(matrix)
    identity = np.eye(classes, dtype=np.float64)
    for clean_class in range(classes):
        for noisy_class in range(classes):
            permutation = identity.copy()
            permutation[[clean_class, noisy_class]] = permutation[
                [noisy_class, clean_class]
            ]
            result += (
                priors[clean_class]
                * matrix[clean_class, noisy_class]
                * permutation.T
            )
    return result


@dataclass(frozen=True)
class PCSELayerStatistics:
    name: str
    noisy_means: np.ndarray  # [C,D]
    noisy_second_moments: np.ndarray  # [C,D,D], not covariance
    clean_means: np.ndarray  # [C,D]
    clean_second_moments: np.ndarray  # [C,D,D], not covariance
    clean_covariances: np.ndarray  # [C,D,D]

    def __post_init__(self) -> None:
        noisy_means = np.asarray(self.noisy_means, dtype=np.float64)
        clean_means = np.asarray(self.clean_means, dtype=np.float64)
        noisy_second = np.asarray(self.noisy_second_moments, dtype=np.float64)
        clean_second = np.asarray(self.clean_second_moments, dtype=np.float64)
        covariances = np.asarray(self.clean_covariances, dtype=np.float64)
        if noisy_means.ndim != 2 or clean_means.shape != noisy_means.shape:
            raise ValueError("PCSE means must have matching shape [C,D]")
        classes, dimension = noisy_means.shape
        shape = (classes, dimension, dimension)
        if (
            noisy_second.shape != shape
            or clean_second.shape != shape
            or covariances.shape != shape
        ):
            raise ValueError(
                "PCSE second moments and covariances must have shape [C,D,D]"
            )
        values = (
            noisy_means,
            clean_means,
            noisy_second,
            clean_second,
            covariances,
        )
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError("PCSE layer statistics must be finite")
        if not np.allclose(
            covariances,
            covariances.transpose(0, 2, 1),
            rtol=1e-7,
            atol=1e-9,
        ):
            raise ValueError("PCSE clean covariances must be symmetric")
        name = str(self.name).strip()
        if not name:
            raise ValueError("PCSE layer name must not be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "noisy_means", noisy_means.copy())
        object.__setattr__(self, "noisy_second_moments", noisy_second.copy())
        object.__setattr__(self, "clean_means", clean_means.copy())
        object.__setattr__(self, "clean_second_moments", clean_second.copy())
        object.__setattr__(self, "clean_covariances", covariances.copy())


@dataclass(frozen=True)
class PCSEStatistics:
    noisy_priors: np.ndarray
    clean_priors: np.ndarray
    coefficient_matrix: np.ndarray
    transition_condition: float
    coefficient_condition: float
    layers: tuple[PCSELayerStatistics, ...]

    def __post_init__(self) -> None:
        noisy = np.asarray(self.noisy_priors, dtype=np.float64)
        clean = np.asarray(self.clean_priors, dtype=np.float64)
        coefficient = np.asarray(self.coefficient_matrix, dtype=np.float64)
        if noisy.ndim != 1 or clean.shape != noisy.shape:
            raise ValueError("PCSE priors must have matching shape [C]")
        if coefficient.shape != (noisy.size, noisy.size):
            raise ValueError("PCSE coefficient matrix must have shape [C,C]")
        if not (
            np.isfinite(noisy).all()
            and np.isfinite(clean).all()
            and np.isfinite(coefficient).all()
        ):
            raise ValueError("PCSE priors and coefficient matrix must be finite")
        if (
            (noisy <= 0.0).any()
            or (clean <= 0.0).any()
            or not np.isclose(noisy.sum(), 1.0)
            or not np.isclose(clean.sum(), 1.0)
        ):
            raise ValueError("PCSE priors must be positive and sum to one")
        if (
            not np.isfinite(self.transition_condition)
            or self.transition_condition <= 0.0
            or not np.isfinite(self.coefficient_condition)
            or self.coefficient_condition <= 0.0
        ):
            raise ValueError("PCSE condition metrics must be finite and positive")
        if len(self.layers) < 2:
            raise ValueError("PCSE statistics require at least two layers")
        if any(layer.clean_means.shape[0] != noisy.size for layer in self.layers):
            raise ValueError("PCSE layer class counts must match priors")
        if len({layer.name for layer in self.layers}) != len(self.layers):
            raise ValueError("PCSE statistic layer names must be unique")
        object.__setattr__(self, "noisy_priors", noisy.copy())
        object.__setattr__(self, "clean_priors", clean.copy())
        object.__setattr__(self, "coefficient_matrix", coefficient.copy())
        object.__setattr__(self, "layers", tuple(self.layers))


def estimate_pcse_statistics(
    snapshots: Sequence[FeatureSnapshot],
    layer_names: Sequence[str],
    transition: np.ndarray,
    *,
    condition_limit: float = 1e8,
) -> PCSEStatistics:
    """Recover clean per-class statistics using paper Eqs. (20)–(23)."""

    if len(snapshots) < 2 or len(snapshots) != len(layer_names):
        raise ValueError("PCSE requires matching snapshots for at least two layers")
    matrix = validate_transition_matrix(transition)
    classes = matrix.shape[0]
    reference = snapshots[0]
    for snapshot in snapshots[1:]:
        if not np.array_equal(snapshot.global_indices, reference.global_indices):
            raise ValueError("PCSE feature snapshots have misaligned stable indices")
        if not np.array_equal(snapshot.noisy_targets, reference.noisy_targets):
            raise ValueError("PCSE feature snapshots have misaligned noisy targets")
    targets = reference.noisy_targets
    if targets.size == 0 or targets.min() < 0 or targets.max() >= classes:
        raise ValueError("PCSE noisy targets are outside transition classes")
    counts = np.bincount(targets, minlength=classes)
    missing = np.flatnonzero(counts == 0)
    if missing.size:
        raise ValueError(
            "PCSE cannot estimate statistics for missing observed classes: "
            + ", ".join(map(str, missing.tolist()))
        )
    noisy_priors = counts.astype(np.float64) / float(targets.size)
    transition_condition = _condition_number(
        matrix, owner="PCSE transition matrix", limit=condition_limit
    )
    clean_priors = recover_clean_priors(
        noisy_priors, matrix, condition_limit=condition_limit
    )
    coefficient = build_coefficient_matrix(clean_priors, matrix)
    coefficient_condition = _condition_number(
        coefficient, owner="PCSE coefficient matrix M", limit=condition_limit
    )

    # A[j,i] = noisy_prior[j] * M^{-1}[j,i] / clean_prior[i].
    # Linear solve is used instead of forming a pseudo-inverse.
    weighted = np.diag(noisy_priors)
    right_factor = np.linalg.solve(coefficient.T, weighted.T).T
    recovery = right_factor / clean_priors[None, :]

    layers: list[PCSELayerStatistics] = []
    for name, snapshot in zip(layer_names, snapshots):
        features = snapshot.features
        dimension = features.shape[1]
        noisy_means = np.empty((classes, dimension), dtype=np.float64)
        noisy_second = np.empty(
            (classes, dimension, dimension), dtype=np.float64
        )
        for class_index in range(classes):
            values = features[targets == class_index]
            noisy_means[class_index] = values.mean(axis=0)
            noisy_second[class_index] = np.einsum(
                "ni,nj->ij", values, values
            ) / values.shape[0]
        clean_means = recovery.T @ noisy_means
        clean_second = np.einsum("ji,jab->iab", recovery, noisy_second)
        clean_covariances = clean_second - np.einsum(
            "ci,cj->cij", clean_means, clean_means
        )
        asymmetry = np.max(
            np.abs(clean_covariances - clean_covariances.transpose(0, 2, 1))
        )
        if not np.isfinite(asymmetry) or asymmetry > 1e-7:
            raise ValueError("PCSE covariance recovery produced asymmetry")
        clean_covariances = 0.5 * (
            clean_covariances + clean_covariances.transpose(0, 2, 1)
        )
        layers.append(
            PCSELayerStatistics(
                name=str(name),
                noisy_means=noisy_means,
                noisy_second_moments=noisy_second,
                clean_means=clean_means,
                clean_second_moments=clean_second,
                clean_covariances=clean_covariances,
            )
        )
    return PCSEStatistics(
        noisy_priors=noisy_priors,
        clean_priors=clean_priors,
        coefficient_matrix=coefficient,
        transition_condition=transition_condition,
        coefficient_condition=coefficient_condition,
        layers=tuple(layers),
    )
