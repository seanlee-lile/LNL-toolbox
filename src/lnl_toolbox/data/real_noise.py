from __future__ import annotations

"""Lazy local-file adapters for common real-world noisy-label datasets."""

from pathlib import Path
import re

import numpy as np

from .contracts import DataSpec, RawDatasetSplit
from .registry import DatasetRegistry


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _manifest_key(value: str) -> str:
    return value.strip().replace("\\", "/").removeprefix("./")


def _configured_path(root: Path, value: object | None, default: str) -> Path:
    path = Path(default if value is None else str(value))
    return path if path.is_absolute() else root / path


def _read_key_list(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset key list does not exist: {path}")
    keys: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if len(value.split()) != 1:
            raise ValueError(f"key-list row must contain one image path: {path}:{line_number}")
        keys.append(_manifest_key(value))
    if not keys:
        raise ValueError(f"dataset key list is empty: {path}")
    if len(set(keys)) != len(keys):
        raise ValueError(f"dataset key list contains duplicate image paths: {path}")
    return tuple(keys)


def _read_label_kv(path: Path, *, num_classes: int) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset label mapping does not exist: {path}")
    labels: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        try:
            raw_key, raw_label = value.rsplit(maxsplit=1)
            key, label = _manifest_key(raw_key), int(raw_label)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid label mapping row {path}:{line_number}") from exc
        if not 0 <= label < num_classes:
            raise ValueError(f"label outside [0, {num_classes}) at {path}:{line_number}")
        if key in labels:
            raise ValueError(f"duplicate image in label mapping at {path}:{line_number}")
        labels[key] = label
    if not labels:
        raise ValueError(f"dataset label mapping is empty: {path}")
    return labels


def _resolve_images(root: Path, keys: tuple[str, ...]) -> tuple[Path, ...]:
    images: list[Path] = []
    for key in keys:
        path = Path(key)
        image = path if path.is_absolute() else root / path
        if not image.is_file():
            raise FileNotFoundError(f"dataset image does not exist: {image}")
        images.append(image)
    return tuple(images)


class Clothing1MAdapter:
    name = "clothing1m"
    aliases = ("clothing_1m",)
    _defaults = {
        "train": "noisy_train_key_list.txt",
        "validation": "clean_val_key_list.txt",
        "test": "clean_test_key_list.txt",
    }
    _label_defaults = {
        "train": "noisy_label_kv.txt",
        "validation": "clean_label_kv.txt",
        "test": "clean_label_kv.txt",
    }

    def _manifest(self, spec: DataSpec, split: str) -> Path:
        if spec.root is None:
            raise ValueError("Clothing1M requires data.root")
        configured = spec.options.get(f"{split}_manifest")
        return _configured_path(spec.root, configured, self._defaults[split])

    def _labels(self, spec: DataSpec, split: str) -> Path:
        if spec.root is None:
            raise ValueError("Clothing1M requires data.root")
        key = "noisy_labels" if split == "train" else "clean_labels"
        configured = spec.options.get(f"{split}_labels", spec.options.get(key))
        return _configured_path(spec.root, configured, self._label_defaults[split])

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None or not spec.root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {spec.root}")
        for split in self._defaults:
            if split == "validation" and bool(spec.options.get("validation_optional", False)):
                continue
            for kind, path in (("key list", self._manifest(spec, split)), ("label mapping", self._labels(spec, split))):
                if not path.is_file():
                    raise FileNotFoundError(f"Clothing1M {split} {kind} does not exist: {path}")

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del seed
        if split not in self._defaults:
            raise ValueError("Clothing1M split must be train, validation, or test")
        keys = _read_key_list(self._manifest(spec, split))
        mapping = _read_label_kv(self._labels(spec, split), num_classes=14)
        missing = [key for key in keys if key not in mapping]
        if missing:
            raise KeyError(f"Clothing1M label mapping does not cover key-list image: {missing[0]}")
        images = _resolve_images(spec.root, keys)
        labels = np.asarray([mapping[key] for key in keys], dtype=np.int64)
        clean = None if split == "train" else labels
        return RawDatasetSplit(
            images,
            labels,
            np.arange(labels.size, dtype=np.int64),
            self.name,
            split,
            14,
            clean_targets=clean,
            source=(
                f"official_key_list:{self._manifest(spec, split).resolve()};"
                f"labels:{self._labels(spec, split).resolve()}"
            ),
        )


class Animal10NAdapter:
    name = "animal10n"
    aliases = ("animal_10n",)

    class_names = (
        "cat", "lynx", "wolf", "coyote", "cheetah", "jaguar",
        "chimpanzee", "orangutan", "hamster", "guinea pig",
    )
    _record_size = 4 + 4 + 3 * 64 * 64

    @staticmethod
    def _binary_files(root: Path, split: str) -> tuple[Path, ...]:
        if split == "test":
            path = root / "test_batch.bin"
            return (path,) if path.is_file() else ()
        files = sorted(
            root.glob("data_batch_*.bin"),
            key=lambda path: int(re.search(r"(\d+)$", path.stem).group(1)),
        )
        return tuple(files)

    @staticmethod
    def _folder(root: Path, split: str) -> Path | None:
        names = ("training", "train") if split == "train" else ("testing", "test")
        return next((root / name for name in names if (root / name).is_dir()), None)

    def _layout(self, root: Path) -> str:
        if self._binary_files(root, "train") and self._binary_files(root, "test"):
            return "official_binary"
        train, test = self._folder(root, "train"), self._folder(root, "test")
        if train is not None and test is not None:
            return "image_folder"
        raise FileNotFoundError(
            "Animal-10N requires official data_batch_*.bin/test_batch.bin files, "
            "or training/testing (train/test) image directories"
        )

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None or not spec.root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {spec.root}")
        self._layout(spec.root)

    @staticmethod
    def _decode_labels(raw: np.ndarray) -> np.ndarray:
        packed = np.ascontiguousarray(raw)
        little = packed.view("<u4").reshape(-1).astype(np.int64)
        big = packed.view(">u4").reshape(-1).astype(np.int64)
        for candidate in (little, big):
            if candidate.size and candidate.min() >= 0 and candidate.max() < 10:
                return candidate
        raise ValueError("Animal-10N binary labels are not valid uint32 class IDs")

    def _load_binary(self, root: Path, split: str) -> tuple[np.ndarray, np.ndarray, str]:
        files = self._binary_files(root, split)
        if not files:
            raise FileNotFoundError(f"Animal-10N {split} binary files are missing")
        images: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for path in files:
            payload = np.fromfile(path, dtype=np.uint8)
            if payload.size == 0 or payload.size % self._record_size:
                raise ValueError(f"Animal-10N binary file has invalid size: {path}")
            records = payload.reshape(-1, self._record_size)
            labels.append(self._decode_labels(records[:, 4:8]))
            pixels = records[:, 8:].reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1).copy()
            images.append(pixels)
        return np.concatenate(images), np.concatenate(labels), "official_binary"

    @classmethod
    def _directory_label(cls, name: str) -> int | None:
        normalized = re.sub(r"[\s_-]+", " ", name.strip().lower())
        numeric = re.fullmatch(r"(?:class\s*)?(\d+)", normalized)
        if numeric and 0 <= int(numeric.group(1)) < 10:
            return int(numeric.group(1))
        aliases = {value: index for index, value in enumerate(cls.class_names)}
        return aliases.get(normalized)

    def _load_images(self, root: Path, split: str) -> tuple[tuple[Path, ...], np.ndarray, str]:
        directory = self._folder(root, split)
        if directory is None:
            raise FileNotFoundError(f"Animal-10N {split} image directory is missing")
        class_directories = [path for path in directory.iterdir() if path.is_dir()]
        paths: list[Path] = []
        labels: list[int] = []
        if class_directories:
            mapped = [(self._directory_label(path.name), path) for path in class_directories]
            if len(mapped) != 10 or any(label is None for label, _ in mapped) or len({label for label, _ in mapped}) != 10:
                raise ValueError(f"Animal-10N requires ten recognized class directories under {directory}")
            for label, class_directory in sorted(mapped):
                for path in sorted(class_directory.rglob("*")):
                    if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                        paths.append(path)
                        labels.append(int(label))
        else:
            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                if not path.name or not path.name[0].isdigit() or int(path.name[0]) >= 10:
                    raise ValueError(f"Animal-10N flat-layout filename must begin with label 0-9: {path}")
                paths.append(path)
                labels.append(int(path.name[0]))
        if not paths:
            raise ValueError(f"Animal-10N split contains no images: {directory}")
        return tuple(paths), np.asarray(labels, dtype=np.int64), f"image_folder:{directory.name}"

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del seed
        if split == "validation":
            split = "test"
        if split not in {"train", "test"}:
            raise ValueError("Animal-10N split must be train or test")
        if spec.root is None:
            raise ValueError("Animal-10N requires data.root")
        layout = self._layout(spec.root)
        if layout == "official_binary":
            inputs, targets, source = self._load_binary(spec.root, split)
        else:
            inputs, targets, source = self._load_images(spec.root, split)
        clean = None if split == "train" else targets
        return RawDatasetSplit(
            inputs,
            targets,
            np.arange(targets.size, dtype=np.int64),
            self.name,
            split,
            10,
            clean_targets=clean,
            class_names=self.class_names,
            source=source,
        )


def add_real_noise_sources(registry: DatasetRegistry) -> None:
    registry.add(Clothing1MAdapter())
    registry.add(Animal10NAdapter())


__all__ = ["add_real_noise_sources"]
