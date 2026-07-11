from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CifarData:
    images: np.ndarray
    labels: np.ndarray
    class_names: tuple[str, ...]
    split: str
    dataset: str

    def __post_init__(self) -> None:
        if self.images.ndim != 4 or self.images.shape[1:] != (32, 32, 3):
            raise ValueError(f"Expected images with shape [N, 32, 32, 3], got {self.images.shape}")
        if self.labels.shape != (self.images.shape[0],):
            raise ValueError("Image and label counts differ")
        if self.images.dtype != np.uint8:
            raise ValueError("CIFAR images must use uint8 storage")

    def __len__(self) -> int:
        return int(self.labels.size)


def default_data_root() -> Path:
    """Return the repository-level data directory."""

    return Path(__file__).resolve().parents[3] / "data"


def _unpickle(path: Path) -> dict[Any, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CIFAR file: {path}")
    with path.open("rb") as handle:
        value = pickle.load(handle, encoding="bytes")
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    return value


def _get(record: dict[Any, Any], key: str) -> Any:
    if key in record:
        return record[key]
    byte_key = key.encode()
    if byte_key in record:
        return record[byte_key]
    raise KeyError(f"Missing key {key!r}; available keys: {list(record)[:8]}")


def _decode_names(values: Any) -> tuple[str, ...]:
    return tuple(value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values)


def _decode_images(flat: Any, source: Path) -> np.ndarray:
    array = np.asarray(flat, dtype=np.uint8)
    if array.ndim != 2 or array.shape[1] != 3072:
        raise ValueError(f"Expected [N, 3072] image data in {source}, got {array.shape}")
    return array.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()


def load_cifar10(root: str | Path | None = None, split: str = "train") -> CifarData:
    root = Path(root) if root is not None else default_data_root() / "cifar10"
    if split == "train":
        files = [root / f"data_batch_{index}" for index in range(1, 6)]
    elif split == "test":
        files = [root / "test_batch"]
    else:
        raise ValueError("CIFAR-10 split must be 'train' or 'test'")

    records = [_unpickle(path) for path in files]
    images = np.concatenate([_decode_images(_get(record, "data"), path) for record, path in zip(records, files)])
    labels = np.concatenate([np.asarray(_get(record, "labels"), dtype=np.int64) for record in records])
    names = _decode_names(_get(_unpickle(root / "batches.meta"), "label_names"))
    return CifarData(images, labels, names, split, "cifar10")


def load_cifar100(root: str | Path | None = None, split: str = "train") -> CifarData:
    root = Path(root) if root is not None else default_data_root() / "cifar100"
    if split not in {"train", "test"}:
        raise ValueError("CIFAR-100 split must be 'train' or 'test'")
    source = root / split
    record = _unpickle(source)
    images = _decode_images(_get(record, "data"), source)
    labels = np.asarray(_get(record, "fine_labels"), dtype=np.int64)
    names = _decode_names(_get(_unpickle(root / "meta"), "fine_label_names"))
    return CifarData(images, labels, names, split, "cifar100")


def summarize_cifar(data: CifarData) -> dict[str, Any]:
    counts = np.bincount(data.labels, minlength=len(data.class_names))
    return {
        "dataset": data.dataset,
        "split": data.split,
        "samples": len(data),
        "image_shape": list(data.images.shape[1:]),
        "dtype": str(data.images.dtype),
        "classes": len(data.class_names),
        "label_min": int(data.labels.min()) if len(data) else None,
        "label_max": int(data.labels.max()) if len(data) else None,
        "class_count_min": int(counts.min()) if counts.size else 0,
        "class_count_max": int(counts.max()) if counts.size else 0,
    }
