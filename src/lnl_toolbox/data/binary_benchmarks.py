from __future__ import annotations

"""Stable-index binary benchmark readers and corruption manifests."""

from dataclasses import dataclass
from pathlib import Path

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


__all__ = [
    "BinaryBenchmark",
    "load_binary_npz",
    "load_uci_binary",
]
