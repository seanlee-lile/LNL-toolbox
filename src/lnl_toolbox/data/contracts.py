from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class Sample:
    """Stable dataset protocol; clean fields are evaluator-only."""

    image: Any
    target: int
    index: int
    clean_target: int | None = None
    is_clean: bool | None = None

    def training_view(self) -> dict[str, Any]:
        return {"input": self.image, "target": self.target, "index": self.index}


class DataRole(str, Enum):
    TRAIN = "train"
    TRAIN_EVAL = "train_eval"
    NOISY_VALIDATION = "noisy_validation"
    CLEAN_VALIDATION = "clean_validation"
    TRUSTED_VALIDATION = "trusted_validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class DataSpec:
    """Canonical, backward-compatible dataset configuration."""

    name: str
    root: Path | None = None
    path: Path | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.strip().lower().replace("-", "_")
        if not name:
            raise ValueError("data.name must not be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "root", None if self.root is None else Path(self.root))
        object.__setattr__(self, "path", None if self.path is None else Path(self.path))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DataSpec":
        if not isinstance(value, Mapping):
            raise TypeError("data configuration must be a mapping")
        if "name" not in value:
            raise ValueError("data.name is required")
        options = dict(value)
        name = str(options.pop("name"))
        root = options.pop("root", None)
        path = options.pop("path", None)
        return cls(name, root, path, options)


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    dataset: str
    split: str
    version: str
    sample_count: int
    inputs_hash: str
    global_indices_hash: str
    observed_targets_hash: str
    clean_targets_hash: str | None = None
    source: str = "configured"

    def __post_init__(self) -> None:
        if not self.dataset.strip() or not self.split.strip() or not self.version.strip():
            raise ValueError("dataset identity strings must not be empty")
        if self.sample_count < 0:
            raise ValueError("dataset sample_count must be non-negative")

    @property
    def fingerprint(self) -> str:
        payload = {
            "dataset": self.dataset,
            "split": self.split,
            "version": self.version,
            "sample_count": self.sample_count,
            "inputs_hash": self.inputs_hash,
            "global_indices_hash": self.global_indices_hash,
            "observed_targets_hash": self.observed_targets_hash,
            "clean_targets_hash": self.clean_targets_hash,
            "source": self.source,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "version": self.version,
            "sample_count": self.sample_count,
            "inputs_hash": self.inputs_hash,
            "global_indices_hash": self.global_indices_hash,
            "observed_targets_hash": self.observed_targets_hash,
            "clean_targets_hash": self.clean_targets_hash,
            "source": self.source,
            "fingerprint": self.fingerprint,
        }


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def inputs_sha256(values: Any) -> str:
    if isinstance(values, np.ndarray):
        return array_sha256(values)
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, (str, Path)):
            path = Path(value).resolve()
            stat = path.stat()
            digest.update(str(path).encode("utf-8"))
            digest.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RawDatasetSplit:
    """Adapter output before transforms, noise overlays, and loader creation."""

    inputs: Any
    observed_targets: np.ndarray
    global_indices: np.ndarray
    dataset: str
    split: str
    num_classes: int
    version: str = "1"
    clean_targets: np.ndarray | None = None
    class_names: tuple[str, ...] = ()
    source: str = "configured"

    def __post_init__(self) -> None:
        observed = np.asarray(self.observed_targets, dtype=np.int64)
        indices = np.asarray(self.global_indices, dtype=np.int64)
        clean = None if self.clean_targets is None else np.asarray(self.clean_targets, dtype=np.int64)
        if observed.ndim != 1 or indices.shape != observed.shape:
            raise ValueError("dataset targets and global indices must be aligned vectors")
        if np.unique(indices).size != indices.size or (indices.size and indices.min() < 0):
            raise ValueError("dataset global indices must be unique and non-negative")
        if clean is not None and clean.shape != observed.shape:
            raise ValueError("clean targets must align with observed targets")
        if self.num_classes <= 1:
            raise ValueError("dataset must contain at least two classes")
        for name, targets in (("observed", observed), ("clean", clean)):
            if targets is not None and targets.size and (
                targets.min() < 0 or targets.max() >= self.num_classes
            ):
                raise ValueError(f"{name} targets are outside the class range")
        if hasattr(self.inputs, "__len__") and len(self.inputs) != len(observed):
            raise ValueError("dataset inputs and targets must have equal length")
        object.__setattr__(self, "observed_targets", observed.copy())
        object.__setattr__(self, "global_indices", indices.copy())
        object.__setattr__(self, "clean_targets", None if clean is None else clean.copy())

    def __len__(self) -> int:
        return int(self.observed_targets.size)

    @property
    def identity(self) -> DatasetIdentity:
        return DatasetIdentity(
            self.dataset,
            self.split,
            self.version,
            len(self),
            inputs_sha256(self.inputs),
            array_sha256(self.global_indices),
            array_sha256(self.observed_targets),
            None if self.clean_targets is None else array_sha256(self.clean_targets),
            self.source,
        )


@dataclass(frozen=True, slots=True)
class DataRequirements:
    roles: frozenset[DataRole] = frozenset({DataRole.TRAIN, DataRole.CLEAN_VALIDATION, DataRole.TEST})
    views: tuple[str, ...] = ("weak",)
    validation_targets: str = "clean"
    needs_noise_manifest: bool = True
    class_subset: tuple[int, ...] | None = None
    manifest_scope: str = "train_split"
    train_drop_last: bool | None = None
    validation_size: int | None = None
    split_strategy: str | None = None
    subset_before_split: bool = False

    def __post_init__(self) -> None:
        if not self.roles:
            raise ValueError("at least one data role is required")
        if not self.views or len(set(self.views)) != len(self.views):
            raise ValueError("data views must be non-empty and unique")
        if self.validation_targets not in {"clean", "noisy"}:
            raise ValueError("validation_targets must be clean or noisy")
        if self.manifest_scope not in {"train_split", "effective_train"}:
            raise ValueError("manifest_scope must be train_split or effective_train")
        if self.class_subset is not None and len(set(self.class_subset)) < 2:
            raise ValueError("class_subset must contain at least two unique classes")


@runtime_checkable
class DatasetAdapter(Protocol):
    name: str
    aliases: tuple[str, ...]

    def validate(self, spec: DataSpec) -> None: ...

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit: ...


__all__ = [
    "DataRequirements",
    "DataRole",
    "DataSpec",
    "DatasetAdapter",
    "DatasetIdentity",
    "RawDatasetSplit",
    "Sample",
    "array_sha256",
    "inputs_sha256",
]

