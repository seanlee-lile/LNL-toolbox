from __future__ import annotations

"""Split-aware noise manifests for papers that restart RNG per split."""

from collections.abc import Sequence

import numpy as np

from .generators import generate_symmetric
from .manifest import NoiseManifest


def generate_split_symmetric_manifest(
    clean_targets: np.ndarray,
    split_indices: Sequence[np.ndarray],
    *,
    num_classes: int,
    rate: float,
    seed: int,
    dataset: str,
    rng: str = "numpy_legacy",
) -> NoiseManifest:
    """Corrupt each disjoint split with a freshly seeded symmetric generator.

    Some reference implementations call their corruption routine separately
    for train and validation data. This function preserves that RNG boundary
    while returning one global-index manifest consumable by the standard
    external-manifest path.
    """

    targets = np.asarray(clean_targets, dtype=np.int64)
    if targets.ndim != 1:
        raise ValueError("clean_targets must be one-dimensional")
    normalized: list[np.ndarray] = []
    for part in split_indices:
        indices = np.asarray(part, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("every split must be a non-empty index vector")
        if indices.min() < 0 or indices.max() >= targets.size:
            raise ValueError("split indices are outside the target array")
        normalized.append(indices)
    combined = np.concatenate(normalized)
    if np.unique(combined).size != combined.size:
        raise ValueError("split indices must be disjoint")

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
            "split_sizes": [int(indices.size) for indices in normalized],
        },
        split="train",
        num_classes=num_classes,
        global_indices=combined,
    )


__all__ = ["generate_split_symmetric_manifest"]
