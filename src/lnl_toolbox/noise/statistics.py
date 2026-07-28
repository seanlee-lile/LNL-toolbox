from __future__ import annotations

"""Versioned global/class-wise statistics artifacts."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class StatisticArtifact:
    values: np.ndarray
    estimator: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).copy()
        if values.ndim == 0 or not np.isfinite(values).all():
            raise ValueError("statistic values must be finite and non-scalar")
        values.setflags(write=False)
        if not str(self.estimator).strip():
            raise ValueError("statistic estimator must not be empty")
        object.__setattr__(self, "values", values)

    @property
    def artifact_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.estimator).encode())
        digest.update(json.dumps(dict(self.metadata), sort_keys=True).encode())
        digest.update(str(self.values.shape).encode())
        digest.update(self.values.astype("<f8").tobytes(order="C"))
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        payload = {"estimator": self.estimator, "metadata": dict(self.metadata), "artifact_hash": self.artifact_hash}
        np.savez_compressed(path, values=self.values, metadata_json=np.array(json.dumps(payload, sort_keys=True)))

    @classmethod
    def load(cls, path: str | Path) -> "StatisticArtifact":
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["metadata_json"].item()))
            artifact = cls(data["values"], payload["estimator"], payload["metadata"])
            if payload.get("artifact_hash") != artifact.artifact_hash:
                raise ValueError("statistic artifact hash does not match contents")
            return artifact


@dataclass(frozen=True)
class CWDStatisticArtifact(StatisticArtifact):
    """Named class-wise artifact used by the CWD risk consumer."""

    @property
    def class_centroids(self) -> np.ndarray:
        return self.values

    @property
    def class_prior(self) -> np.ndarray:
        value = self.metadata.get("class_prior")
        if value is None:
            return np.full(self.values.shape[0], 1.0 / self.values.shape[0])
        return np.asarray(value, dtype=np.float64)

    @property
    def label_flip_matrix(self) -> np.ndarray:
        value = self.metadata.get("label_flip_matrix")
        if value is None:
            return np.eye(self.values.shape[0], dtype=np.float64)
        return np.asarray(value, dtype=np.float64)


__all__ = ["CWDStatisticArtifact", "StatisticArtifact"]
