from __future__ import annotations

"""Validated class-conditional transition providers and artifacts."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor

    from .manifest import NoiseManifest


TRANSITION_CONVENTION = "clean_to_noisy_row"
TRANSITION_ARTIFACT_VERSION = "1.0"


def _json_copy(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached JSON-compatible copy used for stable hashing."""

    try:
        payload = json.dumps(
            dict(values), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("transition metadata must be JSON-compatible") from exc
    return json.loads(payload)


def validate_transition_matrix(
    matrix: np.ndarray,
    num_classes: int | None = None,
) -> np.ndarray:
    """Return a validated copy of ``T[i, j] = P(noisy=j | clean=i)``."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("transition_matrix must have square shape [C, C]")
    if num_classes is not None and values.shape != (num_classes, num_classes):
        raise ValueError(
            f"transition_matrix must have shape [{num_classes}, {num_classes}], "
            f"got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("transition_matrix must contain only finite values")
    if (values < 0.0).any():
        raise ValueError("transition_matrix must be non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("every transition_matrix row must sum to one")
    return values.copy()


@runtime_checkable
class TransitionProvider(Protocol):
    """Provide a validated class-conditional transition matrix."""

    @property
    def num_classes(self) -> int:
        ...

    def as_tensor(self, *, device: Any = None, dtype: Any = None) -> "Tensor":
        ...


@dataclass(frozen=True, slots=True)
class TransitionArtifact:
    """Versioned output of a class-conditional transition estimator.

    The canonical convention is ``matrix[i, j] = P(noisy=j | clean=i)``.
    Artifacts are independent from loss correction so the same estimate can be
    consumed by Forward, Backward, reweighting, or diagnostics.
    """

    matrix: np.ndarray
    estimator: str
    source_snapshot_hash: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = TRANSITION_ARTIFACT_VERSION
    convention: str = TRANSITION_CONVENTION

    def __post_init__(self) -> None:
        matrix = validate_transition_matrix(self.matrix)
        matrix.setflags(write=False)
        estimator = self.estimator.strip().lower()
        if not estimator:
            raise ValueError("transition artifact estimator must not be empty")
        if self.version != TRANSITION_ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported transition artifact version: {self.version!r}"
            )
        if self.convention != TRANSITION_CONVENTION:
            raise ValueError(
                "transition artifact convention must be "
                f"{TRANSITION_CONVENTION!r}"
            )
        if self.source_snapshot_hash and len(self.source_snapshot_hash) != 64:
            raise ValueError("source_snapshot_hash must be an SHA-256 hex digest")
        if self.source_snapshot_hash:
            try:
                bytes.fromhex(self.source_snapshot_hash)
            except ValueError as exc:
                raise ValueError(
                    "source_snapshot_hash must be an SHA-256 hex digest"
                ) from exc
        metadata = MappingProxyType(_json_copy(self.metadata))
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "estimator", estimator)
        object.__setattr__(self, "metadata", metadata)

    @property
    def num_classes(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def artifact_hash(self) -> str:
        digest = hashlib.sha256()
        context = {
            "version": self.version,
            "convention": self.convention,
            "estimator": self.estimator,
            "source_snapshot_hash": self.source_snapshot_hash,
            "metadata": dict(self.metadata),
        }
        digest.update(
            json.dumps(
                context, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(str(self.matrix.shape).encode("ascii"))
        digest.update(self.matrix.astype("<f8", copy=False).tobytes(order="C"))
        return digest.hexdigest()

    def as_tensor(self, *, device: Any = None, dtype: Any = None) -> "Tensor":
        import torch

        return torch.as_tensor(
            self.matrix.copy(),
            device=device,
            dtype=torch.float32 if dtype is None else dtype,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "convention": self.convention,
            "estimator": self.estimator,
            "source_snapshot_hash": self.source_snapshot_hash,
            "metadata": dict(self.metadata),
            "artifact_hash": self.artifact_hash,
        }
        np.savez_compressed(
            destination,
            matrix=self.matrix,
            metadata_json=np.array(json.dumps(payload, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TransitionArtifact":
        with np.load(path, allow_pickle=False) as data:
            if "matrix" not in data.files or "metadata_json" not in data.files:
                raise ValueError("transition artifact is missing required fields")
            try:
                payload = json.loads(str(data["metadata_json"].item()))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError("transition artifact metadata is invalid") from exc
            required = {
                "version",
                "convention",
                "estimator",
                "source_snapshot_hash",
                "metadata",
                "artifact_hash",
            }
            if not isinstance(payload, dict) or not required.issubset(payload):
                raise ValueError("transition artifact metadata is missing required fields")
            artifact = cls(
                matrix=data["matrix"],
                estimator=payload["estimator"],
                source_snapshot_hash=payload["source_snapshot_hash"],
                metadata=payload["metadata"],
                version=payload["version"],
                convention=payload["convention"],
            )
            if payload["artifact_hash"] != artifact.artifact_hash:
                raise ValueError("transition artifact hash does not match its contents")
            return artifact


@dataclass(frozen=True, slots=True)
class KnownTransition:
    """Known/oracle transition matrix for controlled experiments."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", validate_transition_matrix(self.matrix))

    @property
    def num_classes(self) -> int:
        return int(self.matrix.shape[0])

    def as_tensor(self, *, device: Any = None, dtype: Any = None) -> "Tensor":
        import torch

        return torch.as_tensor(
            self.matrix.copy(),
            device=device,
            dtype=torch.float32 if dtype is None else dtype,
        )

    @classmethod
    def from_manifest(cls, manifest: "NoiseManifest") -> "KnownTransition":
        if manifest.transition_matrix is None:
            raise ValueError("Noise manifest does not contain a class transition matrix")
        return cls(manifest.transition_matrix)
