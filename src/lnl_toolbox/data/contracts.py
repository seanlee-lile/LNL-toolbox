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


@dataclass(frozen=True, slots=True)
class SampleKey:
    """Cross-split sample identity; ``index`` alone is split-scoped."""

    dataset: str
    version: str
    split: str
    index: int

    def __post_init__(self) -> None:
        if not self.dataset.strip() or not self.version.strip() or not self.split.strip():
            raise ValueError("sample key namespace strings must not be empty")
        if self.index < 0:
            raise ValueError("sample key index must be non-negative")


class DataRole(str, Enum):
    TRAIN = "train"
    TRAIN_EVAL = "train_eval"
    NOISY_VALIDATION = "noisy_validation"
    CLEAN_VALIDATION = "clean_validation"
    TRUSTED_VALIDATION = "trusted_validation"
    UNLABELED = "unlabeled"
    CURRICULUM = "curriculum"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class InputSpec:
    """Dataset input facts exposed to algorithms without dataset-name checks."""

    modality: str
    shape: tuple[int, ...] | None = None
    channels: int | None = None
    feature_dim: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        modality = str(getattr(self.modality, "value", self.modality)).strip().lower()
        if not modality:
            raise ValueError("input modality must not be empty")
        shape = None if self.shape is None else tuple(int(value) for value in self.shape)
        if shape is not None and (not shape or any(value <= 0 for value in shape)):
            raise ValueError("input shape dimensions must be positive")
        if self.channels is not None and int(self.channels) <= 0:
            raise ValueError("input channels must be positive")
        if self.feature_dim is not None and int(self.feature_dim) <= 0:
            raise ValueError("input feature_dim must be positive")
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "channels", None if self.channels is None else int(self.channels))
        object.__setattr__(self, "feature_dim", None if self.feature_dim is None else int(self.feature_dim))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class NoiseDescriptor:
    """Read-only noise metadata; it never exposes clean training labels."""

    noise_type: str = "unknown"
    nominal_rate: float | None = None
    realized_rate: float | None = None
    rho_positive: float | None = None
    rho_negative: float | None = None
    transition_matrix: np.ndarray | None = None
    instance_transition: np.ndarray | None = None
    provenance: str | None = None
    mapping_hash: str | None = None

    def __post_init__(self) -> None:
        noise_type = str(self.noise_type).strip().lower()
        if not noise_type:
            raise ValueError("noise_type must not be empty")
        object.__setattr__(self, "noise_type", noise_type)
        for name in ("nominal_rate", "realized_rate", "rho_positive", "rho_negative"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be finite and in [0, 1]")
            if value is not None:
                object.__setattr__(self, name, float(value))
        for name in ("transition_matrix", "instance_transition"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=np.float64).copy()
                if not np.isfinite(array).all() or (array < 0.0).any():
                    raise ValueError(f"{name} must contain finite non-negative values")
                if name == "transition_matrix" and (
                    array.ndim != 2 or array.shape[0] != array.shape[1]
                ):
                    raise ValueError("transition_matrix must be square [C, C]")
                if name == "instance_transition" and array.ndim not in {2, 3}:
                    raise ValueError("instance_transition must have shape [N, C] or [N, C, C]")
                if not np.allclose(array.sum(axis=-1), 1.0, rtol=1e-6, atol=1e-8):
                    raise ValueError(f"every {name} probability row must sum to one")
                array.setflags(write=False)
                object.__setattr__(self, name, array)


@dataclass(frozen=True, slots=True)
class DataProtocol:
    """Reproduction or generic data policy, separate from method requirements."""

    name: str = "generic"
    transform_identity: str = "generic"
    validation_size: int | None = None
    split_strategy: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.transform_identity.strip():
            raise ValueError("data protocol name and transform_identity must not be empty")
        if self.validation_size is not None and int(self.validation_size) < 0:
            raise ValueError("data protocol validation_size must be non-negative")
        object.__setattr__(self, "validation_size", None if self.validation_size is None else int(self.validation_size))
        object.__setattr__(self, "split_strategy", None if self.split_strategy is None else str(self.split_strategy).strip() or None)
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transform_identity": self.transform_identity,
            "validation_size": self.validation_size,
            "split_strategy": self.split_strategy,
            "options": dict(self.options),
        }


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
    """Adapter output before transforms, noise overlays, and loader creation.

    ``global_indices`` is a historical name: values are unique only within this
    split's ``(dataset, version, split)`` namespace, not across dataset splits.
    """

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
    def sample_namespace(self) -> tuple[str, str, str]:
        return (self.dataset, self.version, self.split)

    def sample_key(self, index: int) -> SampleKey:
        value = int(index)
        if not bool(np.any(self.global_indices == value)):
            raise KeyError(f"sample index {value} is outside split {self.split!r}")
        return SampleKey(self.dataset, self.version, self.split, value)

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
    "DataProtocol",
    "DataRequirements",
    "DataRole",
    "DataSpec",
    "DatasetAdapter",
    "DatasetIdentity",
    "InputSpec",
    "NoiseDescriptor",
    "RawDatasetSplit",
    "Sample",
    "SampleKey",
    "array_sha256",
    "inputs_sha256",
]
