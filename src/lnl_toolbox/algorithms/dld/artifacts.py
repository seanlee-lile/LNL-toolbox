from __future__ import annotations

"""Versioned, sample-aligned DLD pre-correction artifact."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class DLDPreCorrectionArtifact:
    global_indices: np.ndarray
    noisy_targets: np.ndarray
    p_w: np.ndarray
    p_s: np.ndarray
    p_ws: np.ndarray
    divergence: np.ndarray
    partition: np.ndarray
    y0: np.ndarray
    yn: np.ndarray
    yd: np.ndarray
    condition_features: np.ndarray
    metadata: Mapping[str, Any]
    format_version: int = 1

    def __post_init__(self) -> None:
        indices = np.asarray(self.global_indices)
        targets = np.asarray(self.noisy_targets)
        partition = np.asarray(self.partition)
        divergence = np.asarray(self.divergence, dtype=np.float64)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("DLD artifact global_indices must be integer [N]")
        if indices.size == 0 or np.unique(indices).size != indices.size or (indices < 0).any():
            raise ValueError("DLD artifact global_indices must be non-empty, unique and non-negative")
        n = indices.size
        if targets.shape != (n,) or not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("DLD artifact noisy_targets must be integer [N]")
        if partition.shape != (n,) or not np.isin(partition, [0, 1, 2]).all():
            raise ValueError("DLD artifact partition must contain clean/noisy/hard values")
        if divergence.shape != (n,) or not np.isfinite(divergence).all():
            raise ValueError("DLD artifact divergence must be finite [N]")
        distributions = [np.asarray(value, dtype=np.float64) for value in (self.p_w, self.p_s, self.p_ws, self.y0, self.yn, self.yd)]
        shape = distributions[0].shape
        if len(shape) != 2 or shape[0] != n or shape[1] < 2:
            raise ValueError("DLD artifact label arrays must have shape [N, C]")
        if any(value.shape != shape or not np.isfinite(value).all() for value in distributions):
            raise ValueError("DLD artifact label arrays are invalid")
        p_w, p_s, p_ws, y0, yn, yd = distributions
        for value in (p_w, p_s, p_ws):
            if (value < 0).any() or not np.allclose(value.sum(1), 1.0, atol=1e-7, rtol=0):
                raise ValueError("DLD artifact distributions must be row-stochastic")
        if not np.allclose(p_ws, (p_w + p_s) / 2, atol=1e-10, rtol=0):
            raise ValueError("DLD artifact p_ws does not average its views")
        if not np.allclose(yd, yn - y0, atol=1e-10, rtol=0):
            raise ValueError("DLD artifact direction is not yn-y0")
        features = np.asarray(self.condition_features, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] != n or not np.isfinite(features).all():
            raise ValueError("DLD artifact condition_features must be finite [N, D]")
        order = np.argsort(indices, kind="stable")
        object.__setattr__(self, "global_indices", indices[order].astype(np.int64, copy=True))
        object.__setattr__(self, "noisy_targets", targets[order].astype(np.int64, copy=True))
        object.__setattr__(self, "partition", partition[order].astype(np.int64, copy=True))
        object.__setattr__(self, "divergence", divergence[order].copy())
        for name, value in zip(("p_w", "p_s", "p_ws", "y0", "yn", "yd"), distributions):
            object.__setattr__(self, name, value[order].copy())
        object.__setattr__(self, "condition_features", features[order].copy())
        object.__setattr__(self, "metadata", dict(self.metadata))
        if int(self.format_version) != 1:
            raise ValueError("unsupported DLD pre-correction artifact version")

    @property
    def artifact_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(dict(self.metadata), sort_keys=True, separators=(",", ":")).encode())
        digest.update(str(self.format_version).encode())
        for value in (
            self.global_indices, self.noisy_targets, self.p_w, self.p_s,
            self.p_ws, self.divergence, self.partition, self.y0, self.yn,
            self.yd, self.condition_features,
        ):
            array = np.ascontiguousarray(value)
            digest.update(str(array.shape).encode())
            digest.update(array.dtype.str.encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        payload = {
            "format_version": self.format_version,
            "metadata": dict(self.metadata),
            "artifact_hash": self.artifact_hash,
        }
        np.savez_compressed(
            path,
            global_indices=self.global_indices,
            noisy_targets=self.noisy_targets,
            p_w=self.p_w,
            p_s=self.p_s,
            p_ws=self.p_ws,
            divergence=self.divergence,
            partition=self.partition,
            y0=self.y0,
            yn=self.yn,
            yd=self.yd,
            condition_features=self.condition_features,
            metadata_json=np.array(json.dumps(payload, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DLDPreCorrectionArtifact":
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["metadata_json"].item()))
            artifact = cls(
                data["global_indices"], data["noisy_targets"], data["p_w"],
                data["p_s"], data["p_ws"], data["divergence"],
                data["partition"], data["y0"], data["yn"], data["yd"],
                data["condition_features"], payload["metadata"],
                int(payload["format_version"]),
            )
        if payload.get("artifact_hash") != artifact.artifact_hash:
            raise ValueError("DLD pre-correction artifact hash mismatch")
        return artifact


def persist_precorrection_atomically(
    artifact: DLDPreCorrectionArtifact, path: str | Path
) -> DLDPreCorrectionArtifact:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    try:
        artifact.save(temporary)
        loaded = DLDPreCorrectionArtifact.load(temporary)
        if loaded.artifact_hash != artifact.artifact_hash:
            raise ValueError("temporary DLD artifact content hash mismatch")
        temporary.replace(destination)
        return DLDPreCorrectionArtifact.load(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = ["DLDPreCorrectionArtifact", "persist_precorrection_atomically"]
