from __future__ import annotations

"""Reusable indexed datasets that expose weak and strong image views."""

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .cifar import CifarData
from .torch_cifar import CIFAR_MEAN, CIFAR_STD


def build_strong_cifar_transform(
    *,
    mean: Sequence[float] = CIFAR_MEAN,
    std: Sequence[float] = CIFAR_STD,
    magnitude: int = 10,
) -> Callable[[Image.Image], torch.Tensor]:
    """Return the RandAugment strong view used by robust SSL pipelines."""

    return transforms.Compose((
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=int(magnitude)),
        transforms.ToTensor(),
        transforms.Normalize(tuple(mean), tuple(std)),
    ))


class IndexedMultiViewCifarDataset(Dataset[dict[str, Any]]):
    """CIFAR view with stable indices and independently sampled transforms."""

    def __init__(
        self,
        data: CifarData,
        indices: Sequence[int] | np.ndarray,
        *,
        weak_transform: Callable[[Image.Image], torch.Tensor],
        strong_transform: Callable[[Image.Image], torch.Tensor],
        targets_by_index: Mapping[int, int] | None = None,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or self.indices.size == 0:
            raise ValueError("multi-view indices must be a non-empty vector")
        if self.indices.min() < 0 or self.indices.max() >= len(data):
            raise IndexError("multi-view indices are out of range")
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.targets_by_index = None if targets_by_index is None else {
            int(index): int(target) for index, target in targets_by_index.items()
        }

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        image = Image.fromarray(self.data.images[index], mode="RGB")
        target = int(self.data.labels[index])
        if self.targets_by_index is not None:
            try:
                target = self.targets_by_index[index]
            except KeyError as exc:
                raise KeyError(f"missing target for global index {index}") from exc
        return {
            "input": self.weak_transform(image),
            "strong_input": self.strong_transform(image),
            "target": target,
            "index": index,
        }


__all__ = ["IndexedMultiViewCifarDataset", "build_strong_cifar_transform"]
