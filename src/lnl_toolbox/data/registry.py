from __future__ import annotations

"""Task-neutral registry for dataset source adapters."""

from difflib import get_close_matches
from typing import Iterable

import numpy as np

from .contracts import DataSpec, DatasetAdapter, RawDatasetSplit, UnsupportedDatasetSplitError


def normalize_dataset_name(value: object) -> str:
    name = str(value).strip().lower().replace("-", "_")
    if not name:
        raise ValueError("dataset name must not be empty")
    return name


class DatasetRegistry:
    def __init__(self, adapters: Iterable[DatasetAdapter] = ()) -> None:
        self._adapters: dict[str, DatasetAdapter] = {}
        for adapter in adapters:
            self.add(adapter)

    def add(self, adapter: DatasetAdapter) -> None:
        if not isinstance(adapter, DatasetAdapter):
            raise TypeError("dataset adapter does not satisfy DatasetAdapter")
        keys = (adapter.name, *adapter.aliases)
        normalized = tuple(normalize_dataset_name(key) for key in keys)
        conflicts = [key for key in normalized if key in self._adapters]
        if conflicts:
            raise KeyError(f"dataset aliases are already registered: {conflicts}")
        for key in normalized:
            self._adapters[key] = adapter

    def get(self, name: object) -> DatasetAdapter:
        key = normalize_dataset_name(name)
        try:
            return self._adapters[key]
        except KeyError as exc:
            names = self.names()
            suggestion = get_close_matches(key, names, n=1)
            hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
            raise ValueError(
                f"unknown dataset {key!r}{hint}; registered datasets: "
                + ", ".join(names)
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted({normalize_dataset_name(value.name) for value in self._adapters.values()}))

    def validate(self, spec: DataSpec) -> None:
        self.get(spec.name).validate(spec)

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        adapter = self.get(spec.name)
        adapter.validate(spec)
        return adapter.load(spec, split, seed=seed)

    def training_pool(
        self, spec: DataSpec, *, seed: int, train: RawDatasetSplit | None = None,
    ) -> RawDatasetSplit:
        """Restore the non-test source pool before an experiment partitions it.

        Do not renumber native samples: external labels depend on their IDs.
        Independently generated validation data is not part of a source pool.
        """
        adapter = self.get(spec.name)
        train = self.load(spec, "train", seed=seed) if train is None else train
        if getattr(adapter, "independent_validation", False):
            return train
        try:
            validation = self.load(spec, "validation", seed=seed)
        except (UnsupportedDatasetSplitError, FileNotFoundError, KeyError):
            return train
        if not len(validation):
            return train
        if (train.dataset, train.version, train.num_classes, train.class_names) != (
            validation.dataset, validation.version, validation.num_classes, validation.class_names
        ):
            raise ValueError("training and validation sources cannot form one training pool")
        if np.intersect1d(train.global_indices, validation.global_indices).size:
            raise ValueError("training pool requires disjoint source sample IDs; native train/validation IDs overlap")
        inputs = (
            np.concatenate((train.inputs, validation.inputs))
            if isinstance(train.inputs, np.ndarray) and isinstance(validation.inputs, np.ndarray)
            else [*train.inputs, *validation.inputs]
        )
        clean = (
            np.concatenate((train.clean_targets, validation.clean_targets))
            if train.clean_targets is not None and validation.clean_targets is not None else None
        )
        return RawDatasetSplit(
            inputs, np.concatenate((train.observed_targets, validation.observed_targets)),
            np.concatenate((train.global_indices, validation.global_indices)),
            train.dataset, "train", train.num_classes, train.version, clean,
            train.class_names, train.source + ":training_pool",
        )


def training_pool_size(adapter: str, counts: dict) -> int:
    """Count the unpartitioned non-test pool using accepted dataset facts."""
    train = int(counts.get("train", 0) or 0)
    return train if adapter.startswith("synthetic_") else train + int(counts.get("validation", 0) or 0)


__all__ = ["DatasetRegistry", "normalize_dataset_name", "training_pool_size"]
