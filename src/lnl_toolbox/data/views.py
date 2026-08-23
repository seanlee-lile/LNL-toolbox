from __future__ import annotations

"""Generic stable-index dataset views shared by every experiment runner."""

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .contracts import RawDatasetSplit


Transform = Callable[[Any], torch.Tensor]


def _decode_input(value: Any) -> Any:
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert("RGB")
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.dtype == np.uint8:
            return Image.fromarray(value, mode="L")
        if value.ndim == 3 and value.dtype == np.uint8:
            return Image.fromarray(value)
        return torch.as_tensor(value, dtype=torch.float32)
    if torch.is_tensor(value):
        return value
    return value


class IndexedDatasetView(Dataset[dict[str, Any]]):
    def __init__(
        self,
        split: RawDatasetSplit,
        global_indices: Sequence[int] | np.ndarray | None = None,
        *,
        targets_by_index: Mapping[int, int] | None = None,
        transforms: Mapping[str, Transform | None] | None = None,
        overlays: Mapping[str, Mapping[int, Any]] | None = None,
    ) -> None:
        self.split = split
        lookup = {int(index): position for position, index in enumerate(split.global_indices)}
        requested = split.global_indices if global_indices is None else np.asarray(global_indices, dtype=np.int64)
        if requested.ndim != 1 or np.unique(requested).size != requested.size:
            raise ValueError("dataset view indices must be a unique vector")
        try:
            self.positions = np.asarray([lookup[int(index)] for index in requested], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"dataset view requests unknown global index {exc.args[0]}") from exc
        self.indices = requested.astype(np.int64, copy=True)
        self.targets = {
            int(index): int(target)
            for index, target in zip(split.global_indices, split.observed_targets)
        }
        if targets_by_index is not None:
            missing = set(map(int, requested)) - set(map(int, targets_by_index))
            if missing:
                raise KeyError(f"target overlay is missing {len(missing)} requested indices")
            self.targets.update({int(k): int(v) for k, v in targets_by_index.items()})
        self.transforms = dict(transforms or {"weak": None})
        if not self.transforms:
            raise ValueError("dataset view requires at least one transform")
        self.overlays = {
            str(name): {int(k): value for k, value in values.items()}
            for name, values in (overlays or {}).items()
        }
        for name, values in self.overlays.items():
            missing = set(map(int, requested)) - set(values)
            if missing:
                raise KeyError(f"overlay {name!r} is missing {len(missing)} requested indices")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        position = int(self.positions[item])
        index = int(self.indices[item])
        source = _decode_input(self.split.inputs[position])
        views: dict[str, Any] = {}
        for name, transform in self.transforms.items():
            value = source.copy() if hasattr(source, "copy") else source
            views[name] = value if transform is None else transform(value)
        primary_name = "weak" if "weak" in views else next(iter(views))
        result: dict[str, Any] = {
            "input": views[primary_name],
            "target": self.targets[index],
            "index": index,
        }
        if len(views) > 1 or primary_name != "weak":
            result["views"] = views
        if "strong" in views:
            result["strong_input"] = views["strong"]
        for name, values in self.overlays.items():
            result[name] = values[index]
        return result


__all__ = ["IndexedDatasetView", "Transform"]
