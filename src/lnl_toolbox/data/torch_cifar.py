from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .cifar import CifarData


CIFAR_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR_STD = (0.24703233, 0.24348505, 0.26158768)


def stratified_split(labels: np.ndarray, validation_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/validation indices while preserving class ratios."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not 0 < validation_size < labels.size:
        raise ValueError("validation_size must be between zero and the dataset size")

    classes, counts = np.unique(labels, return_counts=True)
    exact = validation_size * counts / labels.size
    quotas = np.floor(exact).astype(np.int64)
    remaining = validation_size - int(quotas.sum())
    order = np.lexsort((classes, -(exact - quotas)))
    quotas[order[:remaining]] += 1

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for class_id, quota in zip(classes, quotas):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        validation_parts.append(indices[:quota])
        train_parts.append(indices[quota:])

    train_indices = np.sort(np.concatenate(train_parts))
    validation_indices = np.sort(np.concatenate(validation_parts))
    return train_indices, validation_indices


def train_validation_split(
    labels: np.ndarray,
    validation_size: int,
    seed: int,
    *,
    strategy: str = "stratified",
    rng: str = "default_rng",
) -> tuple[np.ndarray, np.ndarray]:
    """Split stable global indices using a configured, reproducible strategy."""

    labels = np.asarray(labels, dtype=np.int64)
    strategy = str(strategy).strip().lower()
    rng = str(rng).strip().lower()
    if strategy == "classwise_legacy":
        if rng != "numpy_legacy":
            raise ValueError(
                "classwise_legacy split requires rng='numpy_legacy'"
            )
        if labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if not 0 < validation_size < labels.size:
            raise ValueError(
                "validation_size must be between zero and the dataset size"
            )
        classes = np.unique(labels)
        train_per_class = int(
            (labels.size - validation_size) / classes.size
        )
        random = np.random.RandomState(seed)
        train_parts: list[np.ndarray] = []
        validation_parts: list[np.ndarray] = []
        for class_index in classes:
            indices = np.flatnonzero(labels == class_index)
            random.shuffle(indices)
            train_parts.append(indices[:train_per_class])
            validation_parts.append(indices[train_per_class:])
        train_indices = np.concatenate(train_parts).astype(
            np.int64, copy=False
        )
        validation_indices = np.concatenate(validation_parts).astype(
            np.int64, copy=False
        )
        if validation_indices.size != validation_size:
            raise ValueError(
                "classwise_legacy split requires class counts compatible "
                "with validation_size"
            )
        random.shuffle(train_indices)
        random.shuffle(validation_indices)
        return train_indices, validation_indices
    if strategy == "stratified":
        if rng != "default_rng":
            raise ValueError("stratified split requires rng='default_rng'")
        return stratified_split(labels, validation_size, seed)
    if strategy != "random":
        raise ValueError(
            "split strategy must be 'stratified', 'classwise_legacy', "
            "or 'random'"
        )
    if rng not in {"default_rng", "numpy_legacy"}:
        raise ValueError("split rng must be 'default_rng' or 'numpy_legacy'")
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not 0 < validation_size < labels.size:
        raise ValueError("validation_size must be between zero and the dataset size")
    random = (
        np.random.RandomState(seed)
        if rng == "numpy_legacy"
        else np.random.default_rng(seed)
    )
    train_size = labels.size - validation_size
    train_indices = np.asarray(
        random.choice(labels.size, size=train_size, replace=False),
        dtype=np.int64,
    )
    validation_indices = np.delete(np.arange(labels.size), train_indices)
    return train_indices, validation_indices.astype(np.int64, copy=False)


def cifar_pixel_mean(images: np.ndarray) -> torch.Tensor:
    """Return the training-set per-pixel RGB mean as a ``[3, 32, 32]`` tensor."""

    values = np.asarray(images)
    if values.ndim != 4 or values.shape[1:] != (32, 32, 3):
        raise ValueError("CIFAR images must have shape [N, 32, 32, 3]")
    if values.shape[0] == 0:
        raise ValueError("CIFAR images must not be empty")
    mean = values.astype(np.float64).mean(axis=0) / 255.0
    return torch.from_numpy(mean.transpose(2, 0, 1).astype(np.float32))


def build_cifar_transform(
    training: bool,
    augment: bool = True,
    *,
    preprocessing: str = "standard",
    pixel_mean: torch.Tensor | None = None,
    normalization_mean: Sequence[float] | None = None,
    normalization_std: Sequence[float] | None = None,
) -> Callable[[Image.Image], torch.Tensor]:
    preprocessing = str(preprocessing).strip().lower()
    if preprocessing == "gce2018":
        if pixel_mean is None or tuple(pixel_mean.shape) != (3, 32, 32):
            raise ValueError(
                "gce2018 preprocessing requires pixel_mean with shape [3, 32, 32]"
            )
        mean = pixel_mean.detach().clone().to(dtype=torch.float32)
        operations: list[Any] = [
            transforms.ToTensor(),
            transforms.Lambda(lambda value: value - mean),
        ]
        if training and augment:
            operations.extend((
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
            ))
        return transforms.Compose(operations)
    if preprocessing != "standard":
        raise ValueError(f"Unsupported CIFAR preprocessing: {preprocessing}")
    if (normalization_mean is None) != (normalization_std is None):
        raise ValueError("normalization_mean and normalization_std must be provided together")
    mean = CIFAR_MEAN if normalization_mean is None else tuple(
        float(value) for value in normalization_mean
    )
    std = CIFAR_STD if normalization_std is None else tuple(
        float(value) for value in normalization_std
    )
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("CIFAR normalization mean and std must contain three values")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or any(
        value <= 0.0 for value in std
    ):
        raise ValueError("CIFAR normalization values must be finite and std must be positive")
    operations: list[Any] = []
    if training and augment:
        operations.extend((transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()))
    operations.extend((transforms.ToTensor(), transforms.Normalize(mean, std)))
    return transforms.Compose(operations)


@dataclass(frozen=True, slots=True)
class DatasetView:
    dataset: "TorchCifarDataset"
    indices: np.ndarray


class TorchCifarDataset(Dataset[dict[str, Any]]):
    """PyTorch view over decoded CIFAR arrays with stable global indices."""

    def __init__(
        self,
        data: CifarData,
        indices: Sequence[int] | np.ndarray | None = None,
        *,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.data = data
        self.indices = (
            np.arange(len(data), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if self.indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if self.indices.size and (self.indices.min() < 0 or self.indices.max() >= len(data)):
            raise IndexError("dataset indices are out of range")
        self.transform = transform or build_cifar_transform(training=False)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        image = Image.fromarray(self.data.images[index], mode="RGB")
        return {
            "input": self.transform(image),
            "target": int(self.data.labels[index]),
            "index": index,
        }

