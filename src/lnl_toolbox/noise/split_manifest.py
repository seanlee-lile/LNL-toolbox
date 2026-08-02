from __future__ import annotations

"""Split-aware noise manifests with deterministic per-split RNG scopes."""

from collections.abc import Sequence

import numpy as np

from .generators import generate_symmetric
from .manifest import NoiseManifest


def _validate_symmetric_rate(
    transition: np.ndarray,
    requested_rate: float,
) -> None:
    implied_rates = 1.0 - np.diag(transition)
    if not np.allclose(
        implied_rates,
        float(requested_rate),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "requested symmetric noise rate does not match the "
            "transition matrix implied rate"
        )


def generate_split_symmetric_manifest(
    clean_targets: np.ndarray,
    split_indices: Sequence[np.ndarray],
    *,
    num_classes: int,
    rate: float,
    seed: int,
    dataset: str,
    rng: str = "numpy_legacy",
    split_names: Sequence[str] | None = None,
) -> NoiseManifest:
    """Corrupt disjoint splits by restarting the configured RNG per split."""

    targets = np.asarray(clean_targets, dtype=np.int64)
    if targets.ndim != 1:
        raise ValueError("clean_targets must be one-dimensional")
    normalized: list[np.ndarray] = []
    for part in split_indices:
        raw_indices = np.asarray(part)
        if not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("split indices must use an integer dtype")
        indices = raw_indices.astype(np.int64, copy=False)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(
                "every split must be a non-empty index vector"
            )
        if indices.min() < 0 or indices.max() >= targets.size:
            raise ValueError(
                "split indices are outside the target array"
            )
        if np.unique(indices).size != indices.size:
            raise ValueError("indices within each split must be unique")
        normalized.append(indices)
    if not normalized:
        raise ValueError("at least one split is required")
    combined = np.concatenate(normalized)
    if np.unique(combined).size != combined.size:
        raise ValueError("split indices must be disjoint")

    if split_names is None:
        names = tuple(
            f"split_{index}" for index in range(len(normalized))
        )
    else:
        names = tuple(str(name).strip() for name in split_names)
        if (
            len(names) != len(normalized)
            or any(not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "split_names must be unique, non-empty, and align "
                "with split_indices"
            )

    corrupted_parts = [
        generate_symmetric(
            targets[indices],
            num_classes,
            rate,
            seed,
            dataset,
            sampling="transition",
            rng=rng,
        ).noisy_targets
        for indices in normalized
    ]
    transition = np.full(
        (num_classes, num_classes),
        rate / (num_classes - 1),
        dtype=np.float64,
    )
    np.fill_diagonal(transition, 1.0 - rate)
    _validate_symmetric_rate(transition, rate)
    return NoiseManifest(
        dataset=dataset,
        noise_type="symmetric",
        seed=seed,
        requested_rate=rate,
        clean_targets=targets[combined],
        noisy_targets=np.concatenate(corrupted_parts),
        transition_matrix=transition,
        metadata={
            "source": "split_generated",
            "sampling": "transition",
            "rng": rng,
            "rng_scope": "per_split",
            "split_names": list(names),
            "split_sizes": [
                int(indices.size) for indices in normalized
            ],
        },
        split="train",
        num_classes=num_classes,
        global_indices=combined,
    )


__all__ = ["generate_split_symmetric_manifest"]
