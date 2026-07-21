from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .cifar import CifarData


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


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


def build_cifar_transform(training: bool, augment: bool = True) -> Callable[[Image.Image], torch.Tensor]:
    operations: list[Any] = []
    if training and augment:
        operations.extend((transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()))
    operations.extend((transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)))
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
        targets: Sequence[int] | np.ndarray | None = None,
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
        self.targets = None if targets is None else np.asarray(targets, dtype=np.int64).copy()
        if self.targets is not None:
            if self.targets.shape != (len(data),):
                raise ValueError("targets must contain one label for every global dataset index")
            if self.targets.size and (
                self.targets.min() < 0 or self.targets.max() >= len(data.class_names)
            ):
                raise ValueError("targets contain labels outside the dataset class range")
        self.transform = transform or build_cifar_transform(training=False)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        image = Image.fromarray(self.data.images[index], mode="RGB")
        return {
            "input": self.transform(image),
            "target": int(self.data.labels[index] if self.targets is None else self.targets[index]),
            "index": index,
        }

