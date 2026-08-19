from __future__ import annotations

"""Adapters for CIFAR-N human annotation files without implicit downloads."""

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .cifar import load_cifar10, load_cifar100
from .contracts import DataSpec, RawDatasetSplit
from .registry import DatasetRegistry


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"CIFAR-N annotation file does not exist: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise TypeError("CIFAR-N annotation file must contain a mapping")
    return value


class CifarNAdapter:
    def __init__(self, name: str, classes: int) -> None:
        self.name = name
        self.classes = classes
        self.aliases = (name.replace("cifar", "cifar_"),)

    def _annotation_path(self, spec: DataSpec) -> Path:
        configured = spec.options.get("noise_path") or spec.options.get("labels_path")
        if configured:
            return Path(configured)
        if spec.root is None:
            raise ValueError(f"{self.name} requires data.root")
        filename = "CIFAR-10_human.pt" if self.classes == 10 else "CIFAR-100_human.pt"
        return spec.root / filename

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None or not spec.root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {spec.root}")
        _load_mapping(self._annotation_path(spec))

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del seed
        loader = load_cifar10 if self.classes == 10 else load_cifar100
        corpus = loader(spec.root, split)
        clean = corpus.labels
        if split == "test":
            observed = clean
            source = "official_clean_test"
        else:
            mapping = _load_mapping(self._annotation_path(spec))
            default = "aggre_label" if self.classes == 10 else "noisy_label"
            variant = str(spec.options.get("noise_variant", default))
            if variant not in mapping:
                raise KeyError(
                    f"CIFAR-N variant {variant!r} is unavailable; keys: {sorted(map(str, mapping))}"
                )
            observed = np.asarray(mapping[variant], dtype=np.int64)
            if observed.shape != clean.shape:
                raise ValueError("CIFAR-N labels do not align with the CIFAR train split")
            clean_key = "clean_label"
            if clean_key in mapping and not np.array_equal(
                np.asarray(mapping[clean_key], dtype=np.int64), clean
            ):
                raise ValueError("CIFAR-N clean labels do not match the base CIFAR dataset")
            source = f"human_annotation:{variant}"
        return RawDatasetSplit(
            corpus.images,
            observed,
            np.arange(len(corpus), dtype=np.int64),
            self.name,
            split,
            self.classes,
            clean_targets=clean,
            class_names=corpus.class_names,
            source=source,
        )


def add_cifar_n_sources(registry: DatasetRegistry) -> None:
    registry.add(CifarNAdapter("cifar10n", 10))
    registry.add(CifarNAdapter("cifar100n", 100))


__all__ = ["add_cifar_n_sources"]
