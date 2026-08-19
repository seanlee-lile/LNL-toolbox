from __future__ import annotations

"""Lazy local-file adapters for common real-world noisy-label datasets."""

from pathlib import Path

import numpy as np

from .contracts import DataSpec, RawDatasetSplit
from .registry import DatasetRegistry


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_label_manifest(root: Path, path: Path) -> tuple[tuple[Path, ...], np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"label manifest does not exist: {path}")
    images: list[Path] = []
    labels: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        try:
            relative, raw_label = value.rsplit(maxsplit=1)
            label = int(raw_label)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid label manifest row {path}:{line_number}") from exc
        image = Path(relative)
        image = image if image.is_absolute() else root / image
        if not image.is_file():
            raise FileNotFoundError(f"manifest image does not exist: {image}")
        images.append(image)
        labels.append(label)
    if not images:
        raise ValueError(f"label manifest is empty: {path}")
    return tuple(images), np.asarray(labels, dtype=np.int64)


class Clothing1MAdapter:
    name = "clothing1m"
    aliases = ("clothing_1m",)
    _defaults = {
        "train": "noisy_train_key_list.txt",
        "validation": "clean_val_key_list.txt",
        "test": "clean_test_key_list.txt",
    }

    def _manifest(self, spec: DataSpec, split: str) -> Path:
        if spec.root is None:
            raise ValueError("Clothing1M requires data.root")
        configured = spec.options.get(f"{split}_manifest")
        return Path(configured) if configured else spec.root / self._defaults[split]

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None or not spec.root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {spec.root}")
        for split in self._defaults:
            if split == "validation" and bool(spec.options.get("validation_optional", False)):
                continue
            if not self._manifest(spec, split).is_file():
                raise FileNotFoundError(
                    f"Clothing1M {split} manifest does not exist: {self._manifest(spec, split)}"
                )

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del seed
        if split not in self._defaults:
            raise ValueError("Clothing1M split must be train, validation, or test")
        images, labels = _read_label_manifest(spec.root, self._manifest(spec, split))
        clean = None if split == "train" else labels
        return RawDatasetSplit(
            images,
            labels,
            np.arange(labels.size, dtype=np.int64),
            self.name,
            split,
            14,
            clean_targets=clean,
            source=f"manifest:{self._manifest(spec, split).resolve()}",
        )


class Animal10NAdapter:
    name = "animal10n"
    aliases = ("animal_10n",)

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None or not spec.root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {spec.root}")
        for split in ("train", "test"):
            if not (spec.root / split).is_dir():
                raise FileNotFoundError(f"Animal-10N split directory is missing: {spec.root / split}")

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del seed
        if split == "validation":
            split = "test"
        if split not in {"train", "test"}:
            raise ValueError("Animal-10N split must be train or test")
        root = spec.root / split
        class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        if len(class_dirs) != 10:
            raise ValueError(f"Animal-10N requires 10 class directories under {root}")
        paths: list[Path] = []
        labels: list[int] = []
        for label, directory in enumerate(class_dirs):
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                    paths.append(path)
                    labels.append(label)
        if not paths:
            raise ValueError(f"Animal-10N split contains no images: {root}")
        targets = np.asarray(labels, dtype=np.int64)
        clean = None if split == "train" else targets
        return RawDatasetSplit(
            tuple(paths),
            targets,
            np.arange(targets.size, dtype=np.int64),
            self.name,
            split,
            10,
            clean_targets=clean,
            class_names=tuple(path.name for path in class_dirs),
            source=f"folder:{root.resolve()}",
        )


def add_real_noise_sources(registry: DatasetRegistry) -> None:
    registry.add(Clothing1MAdapter())
    registry.add(Animal10NAdapter())


__all__ = ["add_real_noise_sources"]
