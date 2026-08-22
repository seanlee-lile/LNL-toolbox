from __future__ import annotations

"""Stable metadata contracts for dataset inspection and compatibility."""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping


class KnowledgeState(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class Modality(str, Enum):
    IMAGE = "image"
    TABULAR = "tabular"
    UNKNOWN = "unknown"


class NoiseStatus(str, Enum):
    CLEAN = "clean"
    NOISY = "noisy"
    UNKNOWN = "unknown"


class NoiseOrigin(str, Enum):
    SYNTHETIC = "synthetic"
    NATIVE = "native"
    UNKNOWN = "unknown"


class NoiseRateStatus(str, Enum):
    KNOWN = "known"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class NoiseRateInfo:
    status: NoiseRateStatus = NoiseRateStatus.UNKNOWN
    value: float | None = None
    provenance: str | None = None

    def __post_init__(self) -> None:
        status = NoiseRateStatus(self.status)
        object.__setattr__(self, "status", status)
        if status in {NoiseRateStatus.KNOWN, NoiseRateStatus.ESTIMATED}:
            if self.value is None or not math.isfinite(float(self.value)):
                raise ValueError(f"{status.value} noise rate requires a finite value")
            value = float(self.value)
            if not 0.0 <= value <= 1.0:
                raise ValueError("noise rate must be in [0, 1]")
            object.__setattr__(self, "value", value)
            if status is NoiseRateStatus.ESTIMATED and not str(self.provenance or "").strip():
                raise ValueError("estimated noise rate requires provenance")
        elif self.value is not None:
            raise ValueError(f"{status.value} noise rate must not carry a value")
        if self.provenance is not None:
            object.__setattr__(self, "provenance", str(self.provenance).strip() or None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "value": self.value,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "NoiseRateInfo":
        if value is None:
            return cls()
        return cls(
            NoiseRateStatus(str(value.get("status", "unknown"))),
            value.get("value"),
            value.get("provenance"),
        )


@dataclass(frozen=True, slots=True)
class NoiseKnowledge:
    status: NoiseStatus = NoiseStatus.UNKNOWN
    origin: NoiseOrigin = NoiseOrigin.UNKNOWN
    rate: NoiseRateInfo = field(default_factory=NoiseRateInfo)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", NoiseStatus(self.status))
        object.__setattr__(self, "origin", NoiseOrigin(self.origin))
        if not isinstance(self.rate, NoiseRateInfo):
            object.__setattr__(self, "rate", NoiseRateInfo.from_dict(self.rate))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "origin": self.origin.value,
            "rate": self.rate.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "NoiseKnowledge":
        if value is None:
            return cls()
        return cls(
            NoiseStatus(str(value.get("status", "unknown"))),
            NoiseOrigin(str(value.get("origin", "unknown"))),
            NoiseRateInfo.from_dict(value.get("rate")),
        )


def _pairs(value: Mapping[str, Any] | tuple[tuple[str, Any], ...]) -> tuple[tuple[str, Any], ...]:
    source = value.items() if isinstance(value, Mapping) else value
    return tuple(sorted(((str(key), item) for key, item in source), key=lambda pair: pair[0]))


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    dataset: str
    adapter: str
    source: str | None
    task: str
    modality: Modality
    num_classes: int
    input_shape: tuple[int, ...] | None
    channels: int | None
    sample_counts_by_split: tuple[tuple[str, int], ...]
    available_splits: tuple[str, ...]
    class_names: tuple[str, ...]
    class_distribution_by_split: tuple[tuple[str, tuple[int, ...]], ...]
    observed_train_labels: KnowledgeState
    clean_train_labels: KnowledgeState
    clean_validation_labels: KnowledgeState
    stable_indices: KnowledgeState
    dataset_fingerprint: str
    split_fingerprints: tuple[tuple[str, str], ...]
    noise: NoiseKnowledge = field(default_factory=NoiseKnowledge)
    profile_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality", Modality(self.modality))
        for name in (
            "observed_train_labels", "clean_train_labels",
            "clean_validation_labels", "stable_indices",
        ):
            object.__setattr__(self, name, KnowledgeState(getattr(self, name)))
        if self.num_classes <= 1:
            raise ValueError("dataset profile num_classes must exceed one")
        if self.input_shape is not None and any(int(value) <= 0 for value in self.input_shape):
            raise ValueError("dataset profile input_shape must be positive")
        if self.channels is not None and self.channels <= 0:
            raise ValueError("dataset profile channels must be positive")
        counts = tuple((key, int(value)) for key, value in _pairs(self.sample_counts_by_split))
        if any(value < 0 for _, value in counts):
            raise ValueError("dataset profile split counts must be non-negative")
        distributions = tuple(
            (key, tuple(int(count) for count in values))
            for key, values in _pairs(self.class_distribution_by_split)
        )
        if any(len(values) != self.num_classes for _, values in distributions):
            raise ValueError("class distributions must match num_classes")
        object.__setattr__(self, "sample_counts_by_split", counts)
        object.__setattr__(self, "available_splits", tuple(sorted(set(self.available_splits))))
        object.__setattr__(self, "class_distribution_by_split", distributions)
        object.__setattr__(self, "split_fingerprints", _pairs(self.split_fingerprints))
        if not isinstance(self.noise, NoiseKnowledge):
            object.__setattr__(self, "noise", NoiseKnowledge.from_dict(self.noise))

    @property
    def fingerprint(self) -> str:
        payload = self.to_dict(include_fingerprint=False)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "profile_version": self.profile_version,
            "dataset": self.dataset,
            "adapter": self.adapter,
            "source": self.source,
            "task": self.task,
            "modality": self.modality.value,
            "num_classes": self.num_classes,
            "input_shape": None if self.input_shape is None else list(self.input_shape),
            "channels": self.channels,
            "sample_counts_by_split": dict(self.sample_counts_by_split),
            "available_splits": list(self.available_splits),
            "class_names": list(self.class_names),
            "class_distribution_by_split": {
                key: list(counts) for key, counts in self.class_distribution_by_split
            },
            "observed_train_labels": self.observed_train_labels.value,
            "clean_train_labels": self.clean_train_labels.value,
            "clean_validation_labels": self.clean_validation_labels.value,
            "stable_indices": self.stable_indices.value,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_fingerprints": dict(self.split_fingerprints),
            "noise": self.noise.to_dict(),
        }
        if include_fingerprint:
            value["fingerprint"] = self.fingerprint
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetProfile":
        return cls(
            dataset=str(value["dataset"]), adapter=str(value["adapter"]),
            source=value.get("source"), task=str(value.get("task", "classification")),
            modality=Modality(str(value.get("modality", "unknown"))),
            num_classes=int(value["num_classes"]),
            input_shape=None if value.get("input_shape") is None else tuple(value["input_shape"]),
            channels=None if value.get("channels") is None else int(value["channels"]),
            sample_counts_by_split=_pairs(value.get("sample_counts_by_split", {})),
            available_splits=tuple(value.get("available_splits", ())),
            class_names=tuple(value.get("class_names", ())),
            class_distribution_by_split=_pairs(value.get("class_distribution_by_split", {})),
            observed_train_labels=KnowledgeState(value.get("observed_train_labels", "unknown")),
            clean_train_labels=KnowledgeState(value.get("clean_train_labels", "unknown")),
            clean_validation_labels=KnowledgeState(value.get("clean_validation_labels", "unknown")),
            stable_indices=KnowledgeState(value.get("stable_indices", "unknown")),
            dataset_fingerprint=str(value["dataset_fingerprint"]),
            split_fingerprints=_pairs(value.get("split_fingerprints", {})),
            noise=NoiseKnowledge.from_dict(value.get("noise")),
            profile_version=int(value.get("profile_version", 1)),
        )


@dataclass(frozen=True, slots=True)
class DatasetDeclarations:
    clean_train_labels: KnowledgeState = KnowledgeState.UNKNOWN
    noise_status: NoiseStatus = NoiseStatus.UNKNOWN
    noise_origin: NoiseOrigin = NoiseOrigin.UNKNOWN
    noise_rate: NoiseRateInfo = field(default_factory=NoiseRateInfo)
    noise_manifest: KnowledgeState = KnowledgeState.UNKNOWN
    method_noise_rate_prior: NoiseRateInfo = field(default_factory=NoiseRateInfo)
    pretrained_roles: tuple[str, ...] = ()
    clean_labels_location: str | None = None
    clean_labels_provenance: str | None = None
    semantic_notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "clean_train_labels", KnowledgeState(self.clean_train_labels))
        object.__setattr__(self, "noise_status", NoiseStatus(self.noise_status))
        object.__setattr__(self, "noise_origin", NoiseOrigin(self.noise_origin))
        if not isinstance(self.noise_rate, NoiseRateInfo):
            object.__setattr__(self, "noise_rate", NoiseRateInfo.from_dict(self.noise_rate))
        if not isinstance(self.method_noise_rate_prior, NoiseRateInfo):
            object.__setattr__(
                self,
                "method_noise_rate_prior",
                NoiseRateInfo.from_dict(self.method_noise_rate_prior),
            )
        object.__setattr__(self, "noise_manifest", KnowledgeState(self.noise_manifest))
        object.__setattr__(self, "pretrained_roles", tuple(sorted(set(self.pretrained_roles))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean_train_labels": self.clean_train_labels.value,
            "noise_status": self.noise_status.value,
            "noise_origin": self.noise_origin.value,
            "noise_rate": self.noise_rate.to_dict(),
            "noise_manifest": self.noise_manifest.value,
            "method_noise_rate_prior": self.method_noise_rate_prior.to_dict(),
            "pretrained_roles": list(self.pretrained_roles),
            "clean_labels_location": self.clean_labels_location,
            "clean_labels_provenance": self.clean_labels_provenance,
            "semantic_notes": self.semantic_notes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "DatasetDeclarations":
        if value is None:
            return cls()
        return cls(
            clean_train_labels=KnowledgeState(value.get("clean_train_labels", "unknown")),
            noise_status=NoiseStatus(value.get("noise_status", "unknown")),
            noise_origin=NoiseOrigin(value.get("noise_origin", "unknown")),
            noise_rate=NoiseRateInfo.from_dict(value.get("noise_rate")),
            noise_manifest=KnowledgeState(value.get("noise_manifest", "unknown")),
            method_noise_rate_prior=NoiseRateInfo.from_dict(value.get("method_noise_rate_prior")),
            pretrained_roles=tuple(value.get("pretrained_roles", ())),
            clean_labels_location=value.get("clean_labels_location"),
            clean_labels_provenance=value.get("clean_labels_provenance"),
            semantic_notes=value.get("semantic_notes"),
        )


@dataclass(frozen=True, slots=True)
class DatasetCapabilities:
    dataset: str
    task: str
    modality: Modality
    num_classes: int
    input_shape: tuple[int, ...] | None
    channels: int | None
    available_splits: tuple[str, ...]
    observed_train_labels: KnowledgeState
    clean_train_labels: KnowledgeState
    clean_validation_labels: KnowledgeState
    noise_status: NoiseStatus
    noise_origin: NoiseOrigin
    noise_rate: NoiseRateInfo
    noise_manifest: KnowledgeState
    method_noise_rate_prior: NoiseRateInfo
    supports_synthetic_corruption: bool
    aligned_clean_noisy_targets: KnowledgeState
    stable_indices: KnowledgeState
    pretrained_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality", Modality(self.modality))
        for name in (
            "observed_train_labels", "clean_train_labels", "clean_validation_labels",
            "noise_manifest", "aligned_clean_noisy_targets", "stable_indices",
        ):
            object.__setattr__(self, name, KnowledgeState(getattr(self, name)))
        object.__setattr__(self, "noise_status", NoiseStatus(self.noise_status))
        object.__setattr__(self, "noise_origin", NoiseOrigin(self.noise_origin))
        if not isinstance(self.noise_rate, NoiseRateInfo):
            object.__setattr__(self, "noise_rate", NoiseRateInfo.from_dict(self.noise_rate))
        if not isinstance(self.method_noise_rate_prior, NoiseRateInfo):
            object.__setattr__(
                self,
                "method_noise_rate_prior",
                NoiseRateInfo.from_dict(self.method_noise_rate_prior),
            )
        if self.num_classes <= 1:
            raise ValueError("dataset capabilities num_classes must exceed one")
        object.__setattr__(self, "available_splits", tuple(sorted(set(self.available_splits))))
        object.__setattr__(self, "pretrained_roles", tuple(sorted(set(self.pretrained_roles))))


class DatasetDeclarationConflict(ValueError):
    """Raised when a user declaration contradicts an inspected hard fact."""


def _merge_state(detected: KnowledgeState, declared: KnowledgeState, field_name: str) -> KnowledgeState:
    if declared is KnowledgeState.UNKNOWN:
        return detected
    if detected is not KnowledgeState.UNKNOWN and detected is not declared:
        raise DatasetDeclarationConflict(f"{field_name} declaration conflicts with inspected data")
    return declared


def resolve_dataset_capabilities(
    profile: DatasetProfile,
    declarations: DatasetDeclarations | None = None,
) -> DatasetCapabilities:
    declared = declarations or DatasetDeclarations()
    clean = _merge_state(
        profile.clean_train_labels, declared.clean_train_labels, "clean_train_labels"
    )
    status = profile.noise.status
    if declared.noise_status is not NoiseStatus.UNKNOWN:
        if status is not NoiseStatus.UNKNOWN and status is not declared.noise_status:
            raise DatasetDeclarationConflict("noise_status declaration conflicts with inspected data")
        status = declared.noise_status
    origin = profile.noise.origin
    if declared.noise_origin is not NoiseOrigin.UNKNOWN:
        if origin is not NoiseOrigin.UNKNOWN and origin is not declared.noise_origin:
            raise DatasetDeclarationConflict("noise_origin declaration conflicts with inspected data")
        origin = declared.noise_origin
    rate = profile.noise.rate
    if declared.noise_rate.status is not NoiseRateStatus.UNKNOWN:
        if rate.status not in {NoiseRateStatus.UNKNOWN, NoiseRateStatus.NOT_APPLICABLE} and rate != declared.noise_rate:
            raise DatasetDeclarationConflict("noise_rate declaration conflicts with inspected data")
        if rate.status is NoiseRateStatus.NOT_APPLICABLE and status is NoiseStatus.CLEAN:
            raise DatasetDeclarationConflict("clean dataset cannot declare a noise rate")
        rate = declared.noise_rate
    aligned = (
        KnowledgeState.AVAILABLE
        if clean is KnowledgeState.AVAILABLE
        else clean
    )
    return DatasetCapabilities(
        dataset=profile.dataset, task=profile.task, modality=profile.modality,
        num_classes=profile.num_classes, input_shape=profile.input_shape,
        channels=profile.channels, available_splits=profile.available_splits,
        observed_train_labels=profile.observed_train_labels,
        clean_train_labels=clean,
        clean_validation_labels=profile.clean_validation_labels,
        noise_status=status, noise_origin=origin, noise_rate=rate,
        noise_manifest=declared.noise_manifest,
        method_noise_rate_prior=declared.method_noise_rate_prior,
        supports_synthetic_corruption=clean is KnowledgeState.AVAILABLE,
        aligned_clean_noisy_targets=aligned,
        stable_indices=profile.stable_indices,
        pretrained_roles=declared.pretrained_roles,
    )


__all__ = [
    "DatasetCapabilities", "DatasetDeclarationConflict", "DatasetDeclarations",
    "DatasetProfile", "KnowledgeState", "Modality", "NoiseKnowledge",
    "NoiseOrigin", "NoiseRateInfo", "NoiseRateStatus", "NoiseStatus",
    "resolve_dataset_capabilities",
]
