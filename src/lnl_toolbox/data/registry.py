from __future__ import annotations

"""Task-neutral registry for dataset source adapters."""

from difflib import get_close_matches
from typing import Iterable

from .contracts import DataSpec, DatasetAdapter, RawDatasetSplit


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


__all__ = ["DatasetRegistry", "normalize_dataset_name"]
