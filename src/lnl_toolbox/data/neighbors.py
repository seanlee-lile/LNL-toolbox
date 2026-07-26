from __future__ import annotations

"""Stable feature-neighbor graph artifacts."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class NeighborGraphArtifact:
    neighbors: np.ndarray
    distances: np.ndarray
    global_indices: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        neighbors = np.asarray(self.neighbors, dtype=np.int64)
        distances = np.asarray(self.distances, dtype=np.float64)
        indices = np.asarray(self.global_indices, dtype=np.int64)
        if neighbors.ndim != 2 or distances.shape != neighbors.shape:
            raise ValueError("neighbors and distances must have matching shape [N, K]")
        if indices.shape != (neighbors.shape[0],):
            raise ValueError("global_indices must have shape [N]")
        if (neighbors < 0).any() or not np.isfinite(distances).all():
            raise ValueError("neighbor graph contains invalid values")
        for value in (neighbors, distances, indices):
            value.setflags(write=False)
        object.__setattr__(self, "neighbors", neighbors)
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "global_indices", indices)

    @property
    def artifact_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(dict(self.metadata), sort_keys=True).encode())
        for value in (self.neighbors, self.distances, self.global_indices):
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        payload = {"metadata": dict(self.metadata), "artifact_hash": self.artifact_hash}
        np.savez_compressed(path, neighbors=self.neighbors, distances=self.distances, global_indices=self.global_indices, metadata_json=np.array(json.dumps(payload, sort_keys=True)))

    @classmethod
    def load(cls, path: str | Path) -> "NeighborGraphArtifact":
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["metadata_json"].item()))
            artifact = cls(data["neighbors"], data["distances"], data["global_indices"], payload["metadata"])
            if payload.get("artifact_hash") != artifact.artifact_hash:
                raise ValueError("neighbor graph artifact hash does not match contents")
            return artifact
