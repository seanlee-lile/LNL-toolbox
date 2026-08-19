from __future__ import annotations

"""MNIST-family adapters using local torchvision data only."""

import numpy as np

from .contracts import DataSpec, RawDatasetSplit
from .registry import DatasetRegistry


class MnistAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.aliases = ("fashion_mnist", "fashionmnist") if name == "fashion_mnist" else ()

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None:
            raise ValueError(f"{self.name} requires data.root")
        if not spec.root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {spec.root}")

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del seed
        if split not in {"train", "test"}:
            raise ValueError("MNIST-family split must be train or test")
        from torchvision.datasets import FashionMNIST, MNIST

        cls = FashionMNIST if self.name == "fashion_mnist" else MNIST
        try:
            data = cls(spec.root, train=split == "train", download=False)
        except RuntimeError as exc:
            raise FileNotFoundError(
                f"{self.name} is not prepared under {spec.root}; automatic download is disabled"
            ) from exc
        images = data.data.detach().cpu().numpy().astype(np.uint8, copy=False)
        labels = np.asarray(data.targets, dtype=np.int64)
        return RawDatasetSplit(
            images,
            labels,
            np.arange(labels.size, dtype=np.int64),
            self.name,
            split,
            10,
            clean_targets=labels,
            class_names=tuple(map(str, getattr(data, "classes", range(10)))),
            source="torchvision_local",
        )


def add_mnist_sources(registry: DatasetRegistry) -> None:
    registry.add(MnistAdapter("mnist"))
    registry.add(MnistAdapter("fashion_mnist"))


__all__ = ["add_mnist_sources"]
