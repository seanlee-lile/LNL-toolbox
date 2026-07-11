from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def fingerprint_labels(labels: np.ndarray) -> str:
    values = np.asarray(labels, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(slots=True)
class NoiseManifest:
    dataset: str
    noise_type: str
    seed: int
    requested_rate: float
    clean_targets: np.ndarray
    noisy_targets: np.ndarray
    transition_matrix: np.ndarray | None = None
    per_sample_transition: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    dataset_fingerprint: str = ""

    def __post_init__(self) -> None:
        self.clean_targets = np.asarray(self.clean_targets, dtype=np.int64)
        self.noisy_targets = np.asarray(self.noisy_targets, dtype=np.int64)
        if self.clean_targets.shape != self.noisy_targets.shape:
            raise ValueError("clean_targets and noisy_targets must have the same shape")
        if not 0.0 <= self.requested_rate <= 1.0:
            raise ValueError("requested_rate must be in [0, 1]")
        if not self.dataset_fingerprint:
            self.dataset_fingerprint = fingerprint_labels(self.clean_targets)

    @property
    def flip_mask(self) -> np.ndarray:
        return self.clean_targets != self.noisy_targets

    @property
    def realized_rate(self) -> float:
        return float(self.flip_mask.mean()) if self.clean_targets.size else 0.0

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "dataset": self.dataset,
            "dataset_fingerprint": self.dataset_fingerprint,
            "noise_type": self.noise_type,
            "seed": self.seed,
            "requested_rate": self.requested_rate,
            "realized_rate": self.realized_rate,
            "metadata": self.metadata,
        }
        np.savez_compressed(
            destination,
            clean_targets=self.clean_targets,
            noisy_targets=self.noisy_targets,
            flip_mask=self.flip_mask,
            transition_matrix=np.array([]) if self.transition_matrix is None else self.transition_matrix,
            per_sample_transition=np.array([]) if self.per_sample_transition is None else self.per_sample_transition,
            metadata_json=np.array(json.dumps(payload, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NoiseManifest":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"].item()))
            transition = data["transition_matrix"]
            per_sample = data["per_sample_transition"]
            return cls(
                version=meta["version"],
                dataset=meta["dataset"],
                dataset_fingerprint=meta["dataset_fingerprint"],
                noise_type=meta["noise_type"],
                seed=int(meta["seed"]),
                requested_rate=float(meta["requested_rate"]),
                clean_targets=data["clean_targets"],
                noisy_targets=data["noisy_targets"],
                transition_matrix=None if transition.size == 0 else transition,
                per_sample_transition=None if per_sample.size == 0 else per_sample,
                metadata=meta.get("metadata", {}),
            )

