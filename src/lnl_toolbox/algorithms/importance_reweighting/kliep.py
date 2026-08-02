from __future__ import annotations

"""KLIEP density-ratio posterior backend for high-dimensional binary data."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from lnl_toolbox.data.binary_synthetic import validate_zero_one_labels
from lnl_toolbox.noise.estimators import PosteriorSnapshot


@dataclass(frozen=True, slots=True)
class _DensityRatioFit:
    coefficients: np.ndarray
    centers: np.ndarray


class KLIEPBinaryNoisyPosteriorEstimator:
    """Estimate ``P(noisy label | x)`` through two KLIEP density ratios.

    Each class-specific ratio estimates ``p(x | noisy_y) / p(x)``. Multiplying
    by the empirical noisy-class prior follows Bayes' rule. Independently
    estimated finite-sample ratios need not sum to one, so the two resulting
    scores are normalized row-wise as an explicit implementation choice.
    """

    def __init__(
        self,
        *,
        bandwidth: float,
        max_centers: int,
        max_iterations: int,
        learning_rate: float,
        tolerance: float,
        epsilon: float,
        seed: int,
    ) -> None:
        self.bandwidth = float(bandwidth)
        self.max_centers = int(max_centers)
        self.max_iterations = int(max_iterations)
        self.learning_rate = float(learning_rate)
        self.tolerance = float(tolerance)
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        for name, value in (
            ("bandwidth", self.bandwidth),
            ("learning_rate", self.learning_rate),
            ("tolerance", self.tolerance),
            ("epsilon", self.epsilon),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"KLIEP {name} must be finite and positive")
        if self.max_centers <= 0 or self.max_iterations <= 0:
            raise ValueError(
                "KLIEP max_centers and max_iterations must be positive"
            )

    def identity(self, feature_dimension: int) -> dict[str, Any]:
        dimension = int(feature_dimension)
        if dimension <= 0:
            raise ValueError("KLIEP feature dimension must be positive")
        return {
            "name": "kliep",
            "implementation_version": 1,
            "feature_dimension": dimension,
            "bandwidth": self.bandwidth,
            "max_centers": self.max_centers,
            "max_iterations": self.max_iterations,
            "learning_rate": self.learning_rate,
            "tolerance": self.tolerance,
            "epsilon": self.epsilon,
            "seed": self.seed,
        }

    def _kernel(self, values: np.ndarray, centers: np.ndarray) -> np.ndarray:
        squared = (
            np.square(values).sum(axis=1, keepdims=True)
            + np.square(centers).sum(axis=1)[None, :]
            - 2.0 * values @ centers.T
        )
        np.maximum(squared, 0.0, out=squared)
        result = np.exp(-squared / (2.0 * self.bandwidth**2))
        if not np.isfinite(result).all():
            raise ValueError("KLIEP kernel matrix contains non-finite values")
        return result

    def _choose_centers(
        self,
        numerator: np.ndarray,
        *,
        class_index: int,
    ) -> np.ndarray:
        count = min(self.max_centers, numerator.shape[0])
        rng = np.random.default_rng(self.seed + 104729 * class_index)
        positions = rng.choice(numerator.shape[0], size=count, replace=False)
        return numerator[np.sort(positions)].copy()

    def _fit_ratio(
        self,
        numerator: np.ndarray,
        denominator: np.ndarray,
        *,
        class_index: int,
    ) -> _DensityRatioFit:
        centers = self._choose_centers(
            numerator,
            class_index=class_index,
        )
        numerator_basis = self._kernel(numerator, centers)
        denominator_basis = self._kernel(denominator, centers)
        denominator_mean = denominator_basis.mean(axis=0)
        if not np.isfinite(denominator_mean).all() or not bool(
            (denominator_mean > self.epsilon).any()
        ):
            raise ValueError("KLIEP denominator kernel support is degenerate")

        coefficients = np.ones(centers.shape[0], dtype=np.float64)
        normalizer = float(denominator_mean @ coefficients)
        if not np.isfinite(normalizer) or normalizer <= self.epsilon:
            raise ValueError("KLIEP coefficient normalization is degenerate")
        coefficients /= normalizer
        previous_objective: float | None = None
        for _ in range(self.max_iterations):
            ratios = numerator_basis @ coefficients
            safe_ratios = np.maximum(ratios, self.epsilon)
            objective = float(np.log(safe_ratios).mean())
            if not np.isfinite(objective):
                raise ValueError("KLIEP objective became non-finite")
            gradient = (
                numerator_basis.T @ (1.0 / safe_ratios)
            ) / numerator_basis.shape[0]
            candidate = np.maximum(
                coefficients + self.learning_rate * gradient,
                0.0,
            )
            normalizer = float(denominator_mean @ candidate)
            if not np.isfinite(normalizer) or normalizer <= self.epsilon:
                raise ValueError("KLIEP coefficient update is degenerate")
            candidate /= normalizer
            change = float(np.max(np.abs(candidate - coefficients)))
            coefficients = candidate
            if (
                previous_objective is not None
                and abs(objective - previous_objective) <= self.tolerance
                and change <= np.sqrt(self.tolerance)
            ):
                break
            previous_objective = objective

        if not np.isfinite(coefficients).all() or bool((coefficients < 0).any()):
            raise ValueError("KLIEP produced invalid density-ratio coefficients")
        empirical_mean = float(
            (denominator_basis @ coefficients).mean()
        )
        if not np.isfinite(empirical_mean) or not np.isclose(
            empirical_mean,
            1.0,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError("KLIEP density ratio violates normalization")
        return _DensityRatioFit(coefficients, centers)

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
            owner="KLIEP posterior",
            require_both_classes=True,
        )
        indices = np.asarray(global_indices)
        if (
            values.ndim != 2
            or values.shape[0] != targets.size
            or values.shape[1] <= 2
        ):
            raise ValueError(
                "KLIEP posterior features must have shape [N, D] with D > 2"
            )
        if not np.isfinite(values).all():
            raise ValueError("KLIEP posterior features must be finite")
        if indices.shape != targets.shape or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError(
                "KLIEP posterior indices must be integer with shape [N]"
            )
        indices = indices.astype(np.int64, copy=True)
        if indices.min() < 0 or np.unique(indices).size != indices.size:
            raise ValueError(
                "KLIEP posterior indices must be non-negative and unique"
            )

        order = np.argsort(indices, kind="stable")
        ordered_values = values[order]
        ordered_targets = targets[order]
        ordered_indices = indices[order]
        scores = np.empty((targets.size, 2), dtype=np.float64)
        for class_index in (0, 1):
            numerator = ordered_values[ordered_targets == class_index]
            fit = self._fit_ratio(
                numerator,
                ordered_values,
                class_index=class_index,
            )
            ratios = self._kernel(ordered_values, fit.centers) @ fit.coefficients
            if not np.isfinite(ratios).all() or bool((ratios < 0).any()):
                raise ValueError("KLIEP produced invalid density ratios")
            prior = numerator.shape[0] / ordered_targets.size
            scores[:, class_index] = prior * ratios

        row_sums = scores.sum(axis=1, keepdims=True)
        if not np.isfinite(row_sums).all() or bool(
            (row_sums <= self.epsilon).any()
        ):
            raise ValueError("KLIEP posterior normalization is degenerate")
        probabilities = scores / row_sums
        if (
            not np.isfinite(probabilities).all()
            or bool((probabilities < 0).any())
            or not np.allclose(
                probabilities.sum(axis=1),
                1.0,
                rtol=1e-8,
                atol=1e-10,
            )
        ):
            raise ValueError("KLIEP posterior probabilities are invalid")

        from .estimation import validate_binary_posterior_snapshot

        return validate_binary_posterior_snapshot(PosteriorSnapshot(
            noisy_probabilities=probabilities,
            noisy_targets=ordered_targets,
            global_indices=ordered_indices,
            dataset=dataset,
            split=split,
        ))
