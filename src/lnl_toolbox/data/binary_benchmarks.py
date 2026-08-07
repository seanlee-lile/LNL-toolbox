from __future__ import annotations

"""Stable-index binary benchmark readers and corruption manifests."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BinaryBenchmark:
    features: np.ndarray
    targets: np.ndarray
    dataset: str
    split: str = "train"
    global_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        targets = np.asarray(self.targets, dtype=np.int64)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("binary features must have shape [N, D]")
        if not np.isfinite(features).all():
            raise ValueError("binary features must be finite")
        if (
            targets.shape != (features.shape[0],)
            or set(np.unique(targets)) - {0, 1}
        ):
            raise ValueError(
                "binary targets must be labels 0 and 1 aligned with features"
            )
        if self.global_indices is None:
            indices = np.arange(len(targets), dtype=np.int64)
        else:
            raw_indices = np.asarray(self.global_indices)
            if not np.issubdtype(raw_indices.dtype, np.integer):
                raise ValueError(
                    "binary global_indices must use an integer dtype"
                )
            indices = raw_indices.astype(np.int64, copy=True)
        if (
            indices.shape != targets.shape
            or (indices.size and indices.min() < 0)
            or np.unique(indices).size != len(indices)
        ):
            raise ValueError(
                "binary global_indices must be non-negative, unique, "
                "and aligned"
            )
        dataset = str(self.dataset).strip()
        split = str(self.split).strip()
        if not dataset or not split:
            raise ValueError("binary dataset and split must not be empty")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "global_indices", indices)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "split", split)

    def __len__(self) -> int:
        return int(self.targets.size)


def load_uci_binary(
    path: str | Path,
    *,
    target_column: int = -1,
    delimiter: str = ",",
    name: str | None = None,
) -> BinaryBenchmark:
    """Fit preprocessing and read one training-only UCI-style file.

    Validation and test splits must instead reuse an explicit fitted
    :class:`BinaryPreprocessor`.
    """

    from .preprocessing import (
        BinaryPreprocessingConfig,
        BinaryPreprocessor,
    )

    processor = BinaryPreprocessor(BinaryPreprocessingConfig(
        file_format="delimited",
        delimiter=delimiter,
        target_column=target_column,
    ))
    return processor.fit_transform(
        path,
        dataset=name or Path(path).stem,
    )


def load_binary_npz(
    path: str | Path,
    *,
    name: str | None = None,
) -> BinaryBenchmark:
    with np.load(path, allow_pickle=False) as data:
        if "features" not in data or "targets" not in data:
            raise ValueError(
                "binary npz must contain features and targets"
            )
        return BinaryBenchmark(
            data["features"],
            data["targets"],
            name or Path(path).stem,
        )


def stratified_binary_splits(
    targets: Sequence[int],
    folds: int = 3,
    seed: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic train/validation folds stratified by binary label."""

    labels = np.asarray(targets, dtype=np.int64)
    if labels.ndim != 1 or set(np.unique(labels)) - {0, 1}:
        raise ValueError("targets must be a one-dimensional binary label array")
    if folds < 2 or folds > len(labels):
        raise ValueError("folds must be between 2 and the sample count")
    random = np.random.default_rng(seed)
    buckets = [np.asarray([], dtype=np.int64) for _ in range(folds)]
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        random.shuffle(indices)
        for part, values in enumerate(np.array_split(indices, folds)):
            buckets[part] = np.concatenate((buckets[part], values))
    return [
        (
            np.sort(np.concatenate([
                buckets[part] for part in range(folds) if part != held_out
            ])),
            np.sort(buckets[held_out]),
        )
        for held_out in range(folds)
    ]


def corrupt_binary_labels(
    targets: Sequence[int],
    rho_positive: float,
    rho_negative: float,
    seed: int,
) -> "NoiseManifest":
    """Apply class-dependent binary label noise and retain its manifest."""

    from lnl_toolbox.noise.manifest import NoiseManifest

    clean = np.asarray(targets, dtype=np.int64)
    if clean.ndim != 1 or set(np.unique(clean)) - {0, 1}:
        raise ValueError("targets must be a one-dimensional binary label array")
    if not 0.0 <= rho_positive < 1.0 or not 0.0 <= rho_negative < 1.0:
        raise ValueError("binary noise rates must be in [0, 1)")
    random = np.random.default_rng(seed)
    noisy = clean.copy()
    noisy[(clean == 1) & (random.random(len(clean)) < rho_positive)] = 0
    noisy[(clean == 0) & (random.random(len(clean)) < rho_negative)] = 1
    return NoiseManifest(
        dataset="binary",
        split="train",
        noise_type="class_dependent",
        seed=int(seed),
        requested_rate=float((rho_positive + rho_negative) / 2.0),
        clean_targets=clean,
        noisy_targets=noisy,
        transition_matrix=np.asarray([
            [1.0 - rho_negative, rho_negative],
            [rho_positive, 1.0 - rho_positive],
        ]),
        num_classes=2,
    )


def cifar_airplane_automobile_view(
    data: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the CIFAR airplane/automobile binary view with stable indices."""

    labels = np.asarray(getattr(data, "labels"), dtype=np.int64)
    images = np.asarray(getattr(data, "images"))
    indices = np.flatnonzero(np.isin(labels, (0, 1)))
    return images[indices], labels[indices], indices


__all__ = [
    "BinaryBenchmark",
    "cifar_airplane_automobile_view",
    "corrupt_binary_labels",
    "load_binary_npz",
    "load_uci_binary",
    "stratified_binary_splits",
]
