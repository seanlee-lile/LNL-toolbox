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

