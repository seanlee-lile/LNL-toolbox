from __future__ import annotations

"""Hashed, atomically persisted VolMinNet transition artifacts."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lnl_toolbox.noise.transition import validate_transition_matrix


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class VolMinTransitionArtifact:
    raw_off_diagonal: np.ndarray
    matrix: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_off_diagonal, dtype=np.float64)
        matrix = validate_transition_matrix(self.matrix)
        if raw.ndim != 1 or raw.size != matrix.shape[0] * (matrix.shape[0] - 1):
            raise ValueError("VolMinNet raw transition state has invalid shape")
        if not np.isfinite(raw).all():
            raise ValueError("VolMinNet raw transition state must be finite")
        metadata = _json_copy(self.metadata)
        required = {
            "method", "role", "convention", "parameterization",
            "normalization_axis", "initialization", "epoch", "global_step",
        }
        if metadata.get("method") != "volminnet" or not required.issubset(metadata):
            raise ValueError("VolMinNet transition provenance is incomplete")
        object.__setattr__(self, "raw_off_diagonal", raw.copy())
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "metadata", metadata)

    @property
    def artifact_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(dict(self.metadata), sort_keys=True, separators=(",", ":")).encode())
        digest.update(self.raw_off_diagonal.astype("<f8", copy=False).tobytes())
        digest.update(self.matrix.astype("<f8", copy=False).tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            raw_off_diagonal=self.raw_off_diagonal,
            matrix=self.matrix,
            metadata_json=np.array(json.dumps({
                "metadata": dict(self.metadata), "artifact_hash": self.artifact_hash
            }, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "VolMinTransitionArtifact":
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"raw_off_diagonal", "matrix", "metadata_json"}:
                raise ValueError("VolMinNet transition artifact fields are invalid")
            payload = json.loads(str(data["metadata_json"].item()))
            artifact = cls(data["raw_off_diagonal"], data["matrix"], payload["metadata"])
            if payload.get("artifact_hash") != artifact.artifact_hash:
                raise ValueError("VolMinNet transition artifact hash mismatch")
            return artifact


def persist_transition_atomically(
    artifact: VolMinTransitionArtifact, destination: str | Path
) -> VolMinTransitionArtifact:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    if temporary.exists():
        raise FileExistsError(f"VolMinNet temporary artifact exists: {temporary}")
    try:
        artifact.save(temporary)
        loaded = VolMinTransitionArtifact.load(temporary)
        if loaded.artifact_hash != artifact.artifact_hash:
            raise ValueError("VolMinNet temporary artifact validation failed")
        temporary.replace(destination)
        return loaded
    finally:
        if temporary.exists():
            temporary.unlink()
