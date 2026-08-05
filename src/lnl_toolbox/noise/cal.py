from __future__ import annotations

"""Versioned proxy-label artifact for CAL."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lnl_toolbox.noise.estimators import PosteriorSnapshot


KEEP, RELABEL, DROP = 0, 1, 2


@dataclass(frozen=True, slots=True)
class CALProxyArtifact:
    global_indices: np.ndarray
    proxy_targets: np.ndarray
    sample_status: np.ndarray
    warmup_artifact_hash: str
    lower_threshold: float
    upper_threshold: float

    def __post_init__(self) -> None:
        indices = np.asarray(self.global_indices, dtype=np.int64)
        targets = np.asarray(self.proxy_targets, dtype=np.int64)
        status = np.asarray(self.sample_status, dtype=np.int8)
        if indices.ndim != 1 or targets.shape != indices.shape or status.shape != indices.shape:
            raise ValueError("CAL proxy arrays must be aligned one-dimensional arrays")
        if indices.size == 0 or np.unique(indices).size != indices.size:
            raise ValueError("CAL proxy indices must be non-empty and unique")
        if targets.min() < 0 or not np.isin(status, (KEEP, RELABEL, DROP)).all():
            raise ValueError("CAL proxy targets or status are invalid")
        if not np.isfinite([self.lower_threshold, self.upper_threshold]).all():
            raise ValueError("CAL thresholds must be finite")
        if self.lower_threshold > self.upper_threshold:
            raise ValueError("CAL lower threshold exceeds upper threshold")
        order = np.argsort(indices, kind="stable")
        object.__setattr__(self, "global_indices", indices[order].copy())
        object.__setattr__(self, "proxy_targets", targets[order].copy())
        object.__setattr__(self, "sample_status", status[order].copy())

    @property
    def artifact_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.warmup_artifact_hash.encode())
        digest.update(np.asarray([self.lower_threshold, self.upper_threshold], dtype="<f8").tobytes())
        for value in (self.global_indices, self.proxy_targets, self.sample_status):
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    def lookup(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        requested = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        positions = np.searchsorted(self.global_indices, requested)
        valid = positions < self.global_indices.size
        valid &= self.global_indices[np.minimum(positions, self.global_indices.size - 1)] == requested
        if not valid.all():
            raise KeyError("CAL proxy artifact is missing a global sample index")
        targets = torch.as_tensor(self.proxy_targets[positions], device=indices.device, dtype=torch.long)
        status = torch.as_tensor(self.sample_status[positions], device=indices.device, dtype=torch.long)
        return targets, status.ne(DROP), status

    def save(self, path: str | Path) -> None:
        metadata = {
            "warmup_artifact_hash": self.warmup_artifact_hash,
            "lower_threshold": self.lower_threshold,
            "upper_threshold": self.upper_threshold,
            "artifact_hash": self.artifact_hash,
        }
        np.savez_compressed(
            path,
            global_indices=self.global_indices,
            proxy_targets=self.proxy_targets,
            sample_status=self.sample_status,
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CALProxyArtifact":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            artifact = cls(
                data["global_indices"], data["proxy_targets"], data["sample_status"],
                metadata["warmup_artifact_hash"], metadata["lower_threshold"],
                metadata["upper_threshold"],
            )
        if metadata.get("artifact_hash") != artifact.artifact_hash:
            raise ValueError("CAL proxy artifact hash mismatch")
        return artifact


def build_cal_proxy_artifact(
    snapshot: PosteriorSnapshot,
    adjusted_losses: np.ndarray,
    *,
    lower_threshold: float,
    upper_threshold: float,
) -> CALProxyArtifact:
    scores = np.asarray(adjusted_losses, dtype=np.float64)
    if scores.shape != snapshot.noisy_targets.shape or not np.isfinite(scores).all():
        raise ValueError("CAL adjusted losses must be finite and aligned with snapshot")
    status = np.full(scores.shape, DROP, dtype=np.int8)
    status[scores < lower_threshold] = KEEP
    status[scores > upper_threshold] = RELABEL
    predicted = snapshot.noisy_probabilities.argmax(axis=1)
    targets = snapshot.noisy_targets.copy()
    targets[status == RELABEL] = predicted[status == RELABEL]
    return CALProxyArtifact(
        snapshot.global_indices,
        targets,
        status,
        snapshot.snapshot_hash,
        lower_threshold,
        upper_threshold,
    )


__all__ = [
    "CALProxyArtifact", "DROP", "KEEP", "RELABEL", "build_cal_proxy_artifact"
]
