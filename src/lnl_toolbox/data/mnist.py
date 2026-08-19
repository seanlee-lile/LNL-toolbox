from __future__ import annotations

"""Local-only MNIST-family adapters for official IDX and IDX.GZ files."""

import gzip
from pathlib import Path
import struct

import numpy as np

from .contracts import DataSpec, RawDatasetSplit
from .registry import DatasetRegistry


class MnistAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.aliases = ("fashion_mnist", "fashionmnist") if name == "fashion_mnist" else ()

    @property
    def _directory_name(self) -> str:
        return "FashionMNIST" if self.name == "fashion_mnist" else "MNIST"

    @staticmethod
    def _filenames(split: str) -> tuple[str, str]:
        prefix = "train" if split == "train" else "t10k"
        return f"{prefix}-images-idx3-ubyte", f"{prefix}-labels-idx1-ubyte"

    def _idx_files(self, root: Path, split: str) -> tuple[Path, Path] | None:
        names = self._filenames(split)
        directories = (root, root / "raw", root / self._directory_name / "raw")
        for directory in directories:
            for suffix in ("", ".gz"):
                pair = tuple(directory / f"{name}{suffix}" for name in names)
                if all(path.is_file() for path in pair):
                    return pair  # type: ignore[return-value]
        return None

    @staticmethod
    def _read_bytes(path: Path) -> bytes:
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rb") as handle:
                return handle.read()
        return path.read_bytes()

    def _read_idx(self, root: Path, split: str) -> tuple[np.ndarray, np.ndarray, str]:
        pair = self._idx_files(root, split)
        if pair is None:
            raise FileNotFoundError(f"{self.name} {split} IDX files are missing under {root}")
        image_path, label_path = pair
        image_payload = self._read_bytes(image_path)
        label_payload = self._read_bytes(label_path)
        if len(image_payload) < 16 or len(label_payload) < 8:
            raise ValueError(f"{self.name} IDX files are truncated")
        image_magic, count, rows, columns = struct.unpack(">IIII", image_payload[:16])
        label_magic, label_count = struct.unpack(">II", label_payload[:8])
        if image_magic != 2051 or label_magic != 2049:
            raise ValueError(f"{self.name} IDX magic numbers are invalid")
        if count != label_count or rows != 28 or columns != 28:
            raise ValueError(f"{self.name} IDX dimensions or sample counts are invalid")
        if len(image_payload) != 16 + count * rows * columns:
            raise ValueError(f"{self.name} image IDX payload length is invalid")
        if len(label_payload) != 8 + count:
            raise ValueError(f"{self.name} label IDX payload length is invalid")
        images = np.frombuffer(image_payload, dtype=np.uint8, offset=16).reshape(count, rows, columns).copy()
        labels = np.frombuffer(label_payload, dtype=np.uint8, offset=8).astype(np.int64, copy=True)
        if labels.size and labels.max() >= 10:
            raise ValueError(f"{self.name} IDX labels are outside the class range")
        source = "official_idx_gzip" if image_path.suffix.lower() == ".gz" else "official_idx"
        return images, labels, source

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None:
            raise ValueError(f"{self.name} requires data.root")
        if not spec.root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {spec.root}")
        for split in ("train", "test"):
            if self._idx_files(spec.root, split) is None:
                raise FileNotFoundError(
                    f"{self.name} {split} data is missing; expected official IDX/IDX.GZ files "
                    f"directly under {spec.root}, under raw/, or under {self._directory_name}/raw/"
                )

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del seed
        if split not in {"train", "test"}:
            raise ValueError("MNIST-family split must be train or test")
        if spec.root is None:
            raise ValueError(f"{self.name} requires data.root")
        images, labels, source = self._read_idx(spec.root, split)
        if self.name == "fashion_mnist":
            class_names = (
                "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
            )
        else:
            class_names = tuple(map(str, range(10)))
        return RawDatasetSplit(
            images,
            labels,
            np.arange(labels.size, dtype=np.int64),
            self.name,
            split,
            10,
            clean_targets=labels,
            class_names=class_names,
            source=source,
        )


def add_mnist_sources(registry: DatasetRegistry) -> None:
    registry.add(MnistAdapter("mnist"))
    registry.add(MnistAdapter("fashion_mnist"))


__all__ = ["add_mnist_sources"]
