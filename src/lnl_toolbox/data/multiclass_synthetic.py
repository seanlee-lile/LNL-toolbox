from __future__ import annotations

"""Deterministic multiclass data used by the PCSE method-level smoke test."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class SyntheticMulticlassData:
    """Feature vectors, clean targets and stable global sample identities."""

    features: np.ndarray
    labels: np.ndarray
    global_indices: np.ndarray
    split: str
    dataset: str = "synthetic_multiclass"

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        labels = np.asarray(self.labels)
        indices = np.asarray(self.global_indices)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("synthetic multiclass features must have shape [N,D]")
        if not np.isfinite(features).all():
            raise ValueError("synthetic multiclass features must be finite")
        if labels.shape != (features.shape[0],) or not np.issubdtype(
            labels.dtype, np.integer
        ):
            raise ValueError("synthetic multiclass labels must be integer shape [N]")
        labels = labels.astype(np.int64, copy=True)
        if labels.min() < 0:
            raise ValueError("synthetic multiclass labels must be non-negative")
        classes = np.unique(labels)
        if classes.size < 3 or not np.array_equal(
            classes, np.arange(classes.size, dtype=np.int64)
        ):
            raise ValueError(
                "synthetic multiclass labels must contain contiguous classes "
                "0..C-1 with C >= 3"
            )
        if indices.shape != labels.shape or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError(
                "synthetic multiclass indices must be integer shape [N]"
            )
        indices = indices.astype(np.int64, copy=True)
        if indices.min() < 0 or np.unique(indices).size != indices.size:
            raise ValueError(
                "synthetic multiclass indices must be non-negative and unique"
            )
        split = str(self.split).strip().lower()
        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "synthetic multiclass split must be train, validation, or test"
            )
        object.__setattr__(self, "features", features.copy())
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "global_indices", indices)
        object.__setattr__(self, "split", split)

    @property
    def num_classes(self) -> int:
        return int(np.unique(self.labels).size)


def generate_synthetic_multiclass(
    size: int,
    dimension: int,
    num_classes: int,
    seed: int,
    *,
    start_index: int,
    split: str,
    class_separation: float = 4.0,
    class_scale: float = 0.65,
) -> SyntheticMulticlassData:
    """Generate balanced Gaussian classes for deterministic pipeline checks.

    This distribution is an engineering smoke fixture, not a PCSE paper
    benchmark or a numerical reproduction of a paper dataset.
    """

    if isinstance(size, bool) or int(size) != size or int(size) <= 0:
        raise ValueError("synthetic multiclass size must be a positive integer")
    if isinstance(dimension, bool) or int(dimension) != dimension:
        raise ValueError("synthetic multiclass dimension must be an integer")
    if isinstance(num_classes, bool) or int(num_classes) != num_classes:
        raise ValueError("synthetic multiclass num_classes must be an integer")
    size = int(size)
    dimension = int(dimension)
    num_classes = int(num_classes)
    if num_classes < 3 or dimension < num_classes:
        raise ValueError(
            "synthetic multiclass requires C >= 3 and feature dimension >= C"
        )
    if size < num_classes or size % num_classes:
        raise ValueError(
            "synthetic multiclass size must be divisible by num_classes"
        )
    if start_index < 0:
        raise ValueError("synthetic multiclass start_index must be non-negative")
    if (
        not np.isfinite(class_separation)
        or class_separation <= 0.0
        or not np.isfinite(class_scale)
        or class_scale <= 0.0
    ):
        raise ValueError("synthetic multiclass distribution parameters are invalid")

    rng = np.random.default_rng(int(seed))
    samples_per_class = size // num_classes
    means = np.zeros((num_classes, dimension), dtype=np.float64)
    means[np.arange(num_classes), np.arange(num_classes)] = float(
        class_separation
    )
    features = np.concatenate(
        [
            rng.normal(
                loc=means[class_index],
                scale=float(class_scale),
                size=(samples_per_class, dimension),
            )
            for class_index in range(num_classes)
        ],
        axis=0,
    )
    labels = np.repeat(
        np.arange(num_classes, dtype=np.int64), samples_per_class
    )
    indices = np.arange(start_index, start_index + size, dtype=np.int64)
    order = rng.permutation(size)
    return SyntheticMulticlassData(
        features=features[order],
        labels=labels[order],
        global_indices=indices[order],
        split=split,
    )


class MulticlassTensorDataset(Dataset[dict[str, Any]]):
    """Training-safe view exposing only inputs, observed targets and indices."""

    def __init__(
        self,
        data: SyntheticMulticlassData,
        targets: np.ndarray | None = None,
    ) -> None:
        values = data.labels if targets is None else np.asarray(targets)
        if values.shape != data.labels.shape or not np.issubdtype(
            values.dtype, np.integer
        ):
            raise ValueError("multiclass targets must be integer shape [N]")
        values = values.astype(np.int64, copy=True)
        if values.min() < 0 or values.max() >= data.num_classes:
            raise ValueError("multiclass targets are outside [0,C)")
        self.features = torch.as_tensor(data.features, dtype=torch.float32)
        self.targets = torch.as_tensor(values, dtype=torch.long)
        self.indices = torch.as_tensor(data.global_indices, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, item: int) -> dict[str, Any]:
        return {
            "input": self.features[item],
            "target": self.targets[item],
            "index": self.indices[item],
        }
