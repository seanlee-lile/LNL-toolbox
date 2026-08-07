"""DivideMix's epoch-level GMM clean-probability subcomponent."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
import warnings

import numpy as np
import torch
from torch import Tensor

from .base import ReliabilityResult, validate_reliability_result


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}
_PROBABILITY_TOLERANCE = 1e-12


def _load_sklearn_gmm():
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.mixture import GaussianMixture
    except ImportError as error:
        raise ImportError(
            "DivideMix GMM is missing its optional training dependency. "
            'Install it with: python -m pip install -e ".[train]"'
        ) from error
    return GaussianMixture, ConvergenceWarning


@dataclass(frozen=True)
class DivideMixGMMLossInput:
    """A full-dataset loss vector and its stable sample identities."""

    per_sample_losses: Tensor
    sample_indices: Tensor


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _finite_real(
    name: str,
    value: float,
    *,
    minimum: float,
    strict: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if strict and parsed <= minimum:
        raise ValueError(f"{name} must be greater than {minimum}")
    if not strict and parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _validate_input(
    estimator_input: DivideMixGMMLossInput,
) -> tuple[Tensor, Tensor]:
    if not isinstance(estimator_input, DivideMixGMMLossInput):
        raise TypeError(
            "estimator_input must be a DivideMixGMMLossInput"
        )

    losses = estimator_input.per_sample_losses
    indices = estimator_input.sample_indices
    if not isinstance(losses, Tensor) or losses.ndim != 1:
        raise ValueError("per_sample_losses must be one-dimensional")
    if losses.numel() < 2:
        raise ValueError("per_sample_losses must contain at least two samples")
    if not torch.is_floating_point(losses):
        raise ValueError("per_sample_losses must use a floating-point dtype")
    if not bool(torch.isfinite(losses).all().item()):
        raise ValueError("per_sample_losses must be finite")

    if not isinstance(indices, Tensor) or indices.shape != losses.shape:
        raise ValueError(
            "sample_indices must have the same one-dimensional shape as losses"
        )
    if indices.dtype not in _INTEGER_DTYPES:
        raise ValueError("sample_indices must use an integer dtype")
    if indices.device != losses.device:
        raise ValueError(
            "sample_indices and per_sample_losses must be on the same device"
        )
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("sample_indices must be unique")
    return losses, indices


class DivideMixGMMCleanProbabilityEstimator:
    """Fit the DivideMix two-Gaussian loss model and return clean posteriors.

    Every call performs a fresh CPU float64 fit. Sorting by stable sample index
    before fitting is a deterministic engineering enhancement, not a step from
    the DivideMix paper. Results are restored to the caller's input order.
    """

    def __init__(
        self,
        *,
        random_seed: int = 0,
        max_iter: int = 10,
        tolerance: float = 1e-2,
        covariance_regularization: float = 5e-4,
        minimum_mean_separation: float = 1e-6,
    ) -> None:
        if isinstance(random_seed, bool) or not isinstance(
            random_seed, Integral
        ):
            raise TypeError("random_seed must be an integer")
        self.random_seed = int(random_seed)
        self.max_iter = _positive_integer("max_iter", max_iter)
        self.tolerance = _finite_real(
            "tolerance", tolerance, minimum=0.0, strict=True
        )
        self.covariance_regularization = _finite_real(
            "covariance_regularization",
            covariance_regularization,
            minimum=0.0,
            strict=False,
        )
        self.minimum_mean_separation = _finite_real(
            "minimum_mean_separation",
            minimum_mean_separation,
            minimum=0.0,
            strict=False,
        )

    def estimate(
        self,
        estimator_input: DivideMixGMMLossInput,
    ) -> ReliabilityResult:
        """Fit one loss GMM and return lower-mean-component probabilities."""

        losses, indices = _validate_input(estimator_input)
        cpu_losses = losses.detach().to(
            device="cpu", dtype=torch.float64
        ).numpy()
        cpu_indices = indices.detach().to(
            device="cpu", dtype=torch.int64
        ).numpy()

        canonical_order = np.argsort(cpu_indices, kind="stable")
        canonical_losses = cpu_losses[canonical_order]
        loss_min = float(canonical_losses.min())
        loss_max = float(canonical_losses.max())
        loss_range = loss_max - loss_min
        if loss_range <= 0.0:
            raise ValueError(
                "per_sample_losses must have a positive range for "
                "min-max normalization"
            )

        normalized_losses = (
            (canonical_losses - loss_min) / loss_range
        ).reshape(-1, 1)
        GaussianMixture, ConvergenceWarning = _load_sklearn_gmm()
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            max_iter=self.max_iter,
            tol=self.tolerance,
            reg_covar=self.covariance_regularization,
            n_init=1,
            init_params="kmeans",
            random_state=self.random_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gmm.fit(normalized_losses)
        if not bool(gmm.converged_):
            raise RuntimeError(
                "DivideMix loss GMM did not converge within max_iter"
            )

        means = np.asarray(gmm.means_, dtype=np.float64).reshape(-1)
        if means.shape != (2,) or not np.isfinite(means).all():
            raise ValueError("DivideMix loss GMM produced invalid means")
        mean_separation = float(abs(means[0] - means[1]))
        if mean_separation <= self.minimum_mean_separation:
            raise ValueError(
                "DivideMix loss GMM component means are not sufficiently "
                "separated"
            )

        clean_component = int(np.argmin(means))
        noisy_component = 1 - clean_component
        posterior = np.asarray(
            gmm.predict_proba(normalized_losses), dtype=np.float64
        )
        if posterior.shape != (losses.numel(), 2):
            raise ValueError(
                "DivideMix loss GMM returned an invalid posterior shape"
            )
        if not np.isfinite(posterior).all():
            raise ValueError(
                "DivideMix loss GMM returned non-finite probabilities"
            )
        if (
            np.any(posterior < -_PROBABILITY_TOLERANCE)
            or np.any(posterior > 1.0 + _PROBABILITY_TOLERANCE)
        ):
            raise ValueError(
                "DivideMix loss GMM returned probabilities outside [0, 1]"
            )

        canonical_clean_probability = np.clip(
            posterior[:, clean_component], 0.0, 1.0
        )
        clean_probability = np.empty_like(
            canonical_clean_probability, dtype=np.float64
        )
        clean_probability[canonical_order] = canonical_clean_probability
        scores = torch.from_numpy(clean_probability).to(
            device=losses.device, dtype=torch.float64
        ).detach()

        result = ReliabilityResult(
            sample_indices=indices.detach(),
            scores=scores,
            metrics={
                "loss_min": loss_min,
                "loss_max": loss_max,
                "clean_component_mean": float(means[clean_component]),
                "noisy_component_mean": float(means[noisy_component]),
                "mean_separation": mean_separation,
                "clean_probability_mean": float(clean_probability.mean()),
                "gmm_iterations": float(gmm.n_iter_),
            },
        )
        validate_reliability_result(
            result, expected_sample_indices=indices
        )
        return result
