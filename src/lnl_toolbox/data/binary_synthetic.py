from __future__ import annotations

"""Deterministic binary data for Importance Reweighting smoke tests."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def validate_zero_one_labels(
    labels: np.ndarray,
    *,
    owner: str,
    require_both_classes: bool = False,
) -> np.ndarray:
    """Return validated integer labels under the method's fixed convention."""

    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{owner} labels must be a non-empty one-dimensional array")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{owner} labels must use an integer dtype")
    values = values.astype(np.int64, copy=True)
    unique = np.unique(values)
    if not np.isin(unique, np.array([0, 1], dtype=np.int64)).all():
        raise ValueError(f"{owner} labels must contain only 0 and 1")
    if require_both_classes and not np.array_equal(
        unique, np.array([0, 1], dtype=np.int64)
    ):
        raise ValueError(f"{owner} labels must contain both binary classes 0 and 1")
    return values


@dataclass(frozen=True, slots=True)
class SyntheticBinaryData:
    """Feature vectors, clean labels and stable global sample identities."""

    features: np.ndarray
    labels: np.ndarray
    global_indices: np.ndarray
    split: str
    dataset: str = "synthetic_binary_2d"

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        labels = validate_zero_one_labels(
            self.labels, owner=f"{self.split} dataset"
        )
        indices = np.asarray(self.global_indices)
        dataset = str(self.dataset).strip().lower()
        if dataset == "synthetic_binary_2d":
            valid_shape = features.ndim == 2 and features.shape[1] == 2
            shape_message = "synthetic binary features must have shape [N, 2]"
        elif dataset == "synthetic_binary_high_dim":
            valid_shape = features.ndim == 2 and features.shape[1] > 2
            shape_message = (
                "high-dimensional synthetic binary features must have shape "
                "[N, D] with D > 2"
            )
        else:
            raise ValueError(f"unsupported synthetic binary dataset: {dataset!r}")
        if not valid_shape:
            raise ValueError(shape_message)
        if features.shape[0] != labels.size:
            raise ValueError("synthetic binary features and labels must align")
        if not np.isfinite(features).all():
            raise ValueError("synthetic binary features must be finite")
        if indices.shape != labels.shape or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError(
                "synthetic binary global_indices must be integer and have shape [N]"
            )
        indices = indices.astype(np.int64, copy=True)
        if indices.min() < 0 or np.unique(indices).size != indices.size:
            raise ValueError(
                "synthetic binary global_indices must be non-negative and unique"
            )
        split = str(self.split).strip().lower()
        if split not in {"train", "validation", "test"}:
            raise ValueError("synthetic binary split must be train, validation, or test")
        object.__setattr__(self, "features", features.copy())
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "global_indices", indices)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "dataset", dataset)


def generate_synthetic_binary_2d(
    size: int,
    seed: int,
    *,
    start_index: int = 0,
    split: str,
) -> SyntheticBinaryData:
    """Generate balanced uniform points separated by ``x0 + x1 = 1``."""

    if isinstance(size, bool) or int(size) != size or size < 2 or size % 2:
        raise ValueError("synthetic binary size must be a positive even integer")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    target_per_class = int(size) // 2
    rng = np.random.default_rng(int(seed))
    parts: list[np.ndarray] = []
    for class_index in (0, 1):
        accepted: list[np.ndarray] = []
        count = 0
        while count < target_per_class:
            candidates = rng.random((max(32, 2 * (target_per_class - count)), 2))
            candidate_labels = (candidates.sum(axis=1) >= 1.0).astype(np.int64)
            selected = candidates[candidate_labels == class_index]
            if selected.size:
                take = selected[: target_per_class - count]
                accepted.append(take)
                count += take.shape[0]
        parts.append(np.concatenate(accepted, axis=0))
    features = np.concatenate(parts, axis=0)
    labels = np.concatenate((
        np.zeros(target_per_class, dtype=np.int64),
        np.ones(target_per_class, dtype=np.int64),
    ))
    order = rng.permutation(int(size))
    return SyntheticBinaryData(
        features=features[order],
        labels=labels[order],
        global_indices=np.arange(
            int(start_index), int(start_index) + int(size), dtype=np.int64
        )[order],
        split=split,
    )


def generate_synthetic_binary_high_dim(
    size: int,
    dimension: int,
    seed: int,
    *,
    start_index: int = 0,
    split: str,
) -> SyntheticBinaryData:
    """Generate a balanced Gaussian binary problem for KLIEP smoke tests.

    This controlled high-dimensional distribution is an implementation choice
    for pipeline verification; it is not a reproduction of the paper's
    numerical synthetic experiment.
    """

    if isinstance(size, bool) or int(size) != size or size < 2 or size % 2:
        raise ValueError("synthetic binary size must be a positive even integer")
    if isinstance(dimension, bool) or int(dimension) != dimension or dimension <= 2:
        raise ValueError("high-dimensional binary dimension must exceed 2")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    size = int(size)
    dimension = int(dimension)
    target_per_class = size // 2
    feature_rng = np.random.default_rng(int(seed))
    order_rng = np.random.default_rng(int(seed) + 7919)
    mean = np.zeros(dimension, dtype=np.float64)
    mean[: min(5, dimension)] = 1.25
    class_zero = feature_rng.normal(
        loc=-mean,
        scale=0.75,
        size=(target_per_class, dimension),
    )
    class_one = feature_rng.normal(
        loc=mean,
        scale=0.75,
        size=(target_per_class, dimension),
    )
    features = np.concatenate((class_zero, class_one), axis=0)
    labels = np.concatenate((
        np.zeros(target_per_class, dtype=np.int64),
        np.ones(target_per_class, dtype=np.int64),
    ))
    order = order_rng.permutation(size)
    return SyntheticBinaryData(
        features=features[order],
        labels=labels[order],
        global_indices=np.arange(
            int(start_index), int(start_index) + size, dtype=np.int64
        )[order],
        split=split,
        dataset="synthetic_binary_high_dim",
    )


class BinaryTensorDataset(Dataset[dict[str, Any]]):
    """Training-safe tensor dataset exposing only input, target and index."""

    def __init__(
        self,
        data: SyntheticBinaryData,
        targets: np.ndarray | None = None,
    ) -> None:
        values = data.labels if targets is None else targets
        self.targets = validate_zero_one_labels(
            values, owner=f"{data.split} training view"
        )
        if self.targets.shape != data.labels.shape:
            raise ValueError("binary training targets must align with the dataset")
        self.features = torch.as_tensor(data.features, dtype=torch.float32)
        self.indices = torch.as_tensor(data.global_indices, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.targets.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        return {
            "input": self.features[item],
            "target": int(self.targets[item]),
            "index": int(self.indices[item]),
        }
