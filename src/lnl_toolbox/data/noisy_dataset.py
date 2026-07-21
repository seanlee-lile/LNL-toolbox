from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import Dataset


class NoisyTargetDataset(Dataset[dict[str, Any]]):
    """Replace targets by explicit global-index mappings without exposing clean labels."""

    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        global_indices: np.ndarray,
        noisy_targets: np.ndarray,
    ) -> None:
        indices = np.asarray(global_indices, dtype=np.int64)
        targets = np.asarray(noisy_targets, dtype=np.int64)
        if indices.ndim != 1 or targets.ndim != 1 or indices.shape != targets.shape:
            raise ValueError("global_indices and noisy_targets must be matching one-dimensional arrays")
        if np.unique(indices).size != indices.size:
            raise ValueError("global_indices must be unique")

        self.dataset = dataset
        self._target_by_index = {int(index): int(target) for index, target in zip(indices, targets)}
        dataset_indices = getattr(dataset, "indices", None)
        if dataset_indices is not None:
            missing = set(map(int, np.asarray(dataset_indices))) - self._target_by_index.keys()
            if missing:
                raise ValueError(f"no noisy target exists for {len(missing)} dataset indices")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = self.dataset[item]
        index = int(sample["index"])
        try:
            target = self._target_by_index[index]
        except KeyError as error:
            raise KeyError(f"no noisy target exists for global index {index}") from error
        return {"input": sample["input"], "target": target, "index": index}
