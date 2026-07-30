from __future__ import annotations

"""Binary asymmetric random-classification-noise generation and validation."""

import math
from numbers import Real

import numpy as np

from lnl_toolbox.data.binary_synthetic import validate_zero_one_labels

from .manifest import NoiseManifest


NOISE_TYPE = "binary_asymmetric_rcn"
LABEL_CONVENTION = "zero_one"


def _rate(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1")
    return result


def generate_binary_asymmetric_rcn(
    clean_targets: np.ndarray,
    global_indices: np.ndarray,
    *,
    rho_positive: float,
    rho_negative: float,
    seed: int,
    dataset: str = "synthetic_binary_2d",
) -> NoiseManifest:
    """Independently flip binary labels under the paper's rate convention."""

    clean = validate_zero_one_labels(
        clean_targets, owner="binary RCN clean", require_both_classes=True
    )
    indices = np.asarray(global_indices)
    if indices.shape != clean.shape or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("binary RCN global_indices must be integer and have shape [N]")
    indices = indices.astype(np.int64, copy=True)
    if indices.min() < 0 or np.unique(indices).size != indices.size:
        raise ValueError("binary RCN global_indices must be non-negative and unique")
    positive = _rate("rho_positive", rho_positive)
    negative = _rate("rho_negative", rho_negative)
    if positive + negative >= 1.0:
        raise ValueError("rho_positive + rho_negative must be less than 1")

    noisy = clean.copy()
    rng = np.random.default_rng(int(seed))
    flip_probabilities = np.where(clean == 1, positive, negative)
    flip_mask = rng.random(clean.size) < flip_probabilities
    noisy[flip_mask] = 1 - noisy[flip_mask]
    transition = np.array([
        [1.0 - negative, negative],
        [positive, 1.0 - positive],
    ], dtype=np.float64)
    requested_rate = float(flip_probabilities.mean())
    return NoiseManifest(
        dataset=dataset,
        noise_type=NOISE_TYPE,
        seed=int(seed),
        requested_rate=requested_rate,
        clean_targets=clean,
        noisy_targets=noisy,
        transition_matrix=transition,
        metadata={
            "rho_positive": positive,
            "rho_negative": negative,
            "label_convention": LABEL_CONVENTION,
            "sampling": "independent_bernoulli_by_clean_class",
        },
        version="2.0",
        split="train",
        num_classes=2,
        global_indices=indices,
    )


def validate_binary_rcn_manifest(
    manifest: NoiseManifest,
    *,
    expected_indices: np.ndarray | None = None,
    rho_positive: float | None = None,
    rho_negative: float | None = None,
) -> NoiseManifest:
    """Apply method-specific binary checks without narrowing NoiseManifest."""

    if not isinstance(manifest, NoiseManifest):
        raise TypeError("importance reweighting requires a NoiseManifest")
    if manifest.num_classes != 2:
        raise ValueError("importance reweighting manifest num_classes must be 2")
    if manifest.noise_type != NOISE_TYPE:
        raise ValueError(
            f"importance reweighting requires noise_type {NOISE_TYPE!r}"
        )
    validate_zero_one_labels(
        manifest.clean_targets,
        owner="importance reweighting manifest clean",
        require_both_classes=True,
    )
    validate_zero_one_labels(
        manifest.noisy_targets,
        owner="importance reweighting manifest noisy",
    )
    if manifest.transition_matrix is None or manifest.transition_matrix.shape != (2, 2):
        raise ValueError(
            "importance reweighting manifest transition_matrix must have shape [2, 2]"
        )
    if manifest.metadata.get("label_convention") != LABEL_CONVENTION:
        raise ValueError(
            "importance reweighting manifest label_convention must be zero_one"
        )
    stored_positive = _rate(
        "manifest rho_positive", manifest.metadata.get("rho_positive")
    )
    stored_negative = _rate(
        "manifest rho_negative", manifest.metadata.get("rho_negative")
    )
    if stored_positive + stored_negative >= 1.0:
        raise ValueError("manifest noise rates must sum to less than 1")
    expected_transition = np.array([
        [1.0 - stored_negative, stored_negative],
        [stored_positive, 1.0 - stored_positive],
    ])
    if not np.allclose(
        manifest.transition_matrix, expected_transition, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "importance reweighting manifest transition matrix does not match "
            "its binary noise rates"
        )
    if rho_positive is not None and not np.isclose(
        stored_positive, _rate("rho_positive", rho_positive), rtol=0.0, atol=1e-12
    ):
        raise ValueError("manifest rho_positive does not match the configuration")
    if rho_negative is not None and not np.isclose(
        stored_negative, _rate("rho_negative", rho_negative), rtol=0.0, atol=1e-12
    ):
        raise ValueError("manifest rho_negative does not match the configuration")
    if expected_indices is not None:
        expected = np.asarray(expected_indices, dtype=np.int64)
        if expected.ndim != 1 or np.unique(expected).size != expected.size:
            raise ValueError("expected manifest indices must be one-dimensional and unique")
        if not np.array_equal(
            np.sort(manifest.global_indices), np.sort(expected)
        ):
            raise ValueError(
                "importance reweighting manifest global indices do not match "
                "the configured training/validation population"
            )
    return manifest
