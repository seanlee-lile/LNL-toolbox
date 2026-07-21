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
    version: str = "2.0"
    dataset_fingerprint: str = ""
    split: str = "train"
    num_classes: int | None = None
    global_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.clean_targets = np.asarray(self.clean_targets, dtype=np.int64)
        self.noisy_targets = np.asarray(self.noisy_targets, dtype=np.int64)
        if self.clean_targets.ndim != 1 or self.noisy_targets.ndim != 1:
            raise ValueError("clean_targets and noisy_targets must be one-dimensional")
        if self.clean_targets.shape != self.noisy_targets.shape:
            raise ValueError("clean_targets and noisy_targets must have the same shape")
        if self.global_indices is None:
            self.global_indices = np.arange(self.clean_targets.size, dtype=np.int64)
        else:
            self.global_indices = np.asarray(self.global_indices, dtype=np.int64)
        if self.global_indices.ndim != 1 or self.global_indices.shape != self.clean_targets.shape:
            raise ValueError("global_indices must be one-dimensional and match target shapes")
        if np.unique(self.global_indices).size != self.global_indices.size:
            raise ValueError("global_indices must be unique")
        if not 0.0 <= self.requested_rate <= 1.0:
            raise ValueError("requested_rate must be in [0, 1]")
        if not self.dataset_fingerprint:
            self.dataset_fingerprint = fingerprint_labels(self.clean_targets)
        if self.num_classes is None:
            if self.transition_matrix is not None:
                transition = np.asarray(self.transition_matrix)
                self.num_classes = int(transition.shape[0]) if transition.ndim == 2 else None
            elif self.clean_targets.size:
                self.num_classes = int(max(self.clean_targets.max(), self.noisy_targets.max()) + 1)
        if self.num_classes is not None and self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")

    @property
    def flip_mask(self) -> np.ndarray:
        return self.clean_targets != self.noisy_targets

    @property
    def realized_rate(self) -> float:
        return float(self.flip_mask.mean()) if self.clean_targets.size else 0.0

    @property
    def actual_rate(self) -> float:
        return self.realized_rate

    @property
    def corruption_mask(self) -> np.ndarray:
        return self.flip_mask

    @property
    def mapping_hash(self) -> str:
        """Identify an explicit global-index to noisy-target mapping and its context."""

        digest = hashlib.sha256()
        context = {
            "dataset": self.dataset,
            "split": self.split,
            "dataset_fingerprint": self.dataset_fingerprint,
            "noise_type": self.noise_type,
            "seed": self.seed,
            "requested_rate": self.requested_rate,
            "num_classes": self.num_classes,
        }
        digest.update(json.dumps(context, sort_keys=True, separators=(",", ":")).encode())
        digest.update(self.global_indices.astype("<i8", copy=False).tobytes(order="C"))
        digest.update(self.noisy_targets.astype("<i8", copy=False).tobytes(order="C"))
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "dataset": self.dataset,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split": self.split,
            "noise_type": self.noise_type,
            "seed": self.seed,
            "requested_rate": self.requested_rate,
            "realized_rate": self.realized_rate,
            "num_classes": self.num_classes,
            "mapping_hash": self.mapping_hash,
            "metadata": self.metadata,
        }
        np.savez_compressed(
            destination,
            clean_targets=self.clean_targets,
            noisy_targets=self.noisy_targets,
            global_indices=self.global_indices,
            flip_mask=self.flip_mask,
            corruption_mask=self.corruption_mask,
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
            manifest = cls(
                version=meta.get("version", "1.0"),
                dataset=meta["dataset"],
                dataset_fingerprint=meta.get("dataset_fingerprint", ""),
                split=meta.get("split", "train"),
                noise_type=meta["noise_type"],
                seed=int(meta["seed"]),
                requested_rate=float(meta["requested_rate"]),
                num_classes=meta.get("num_classes"),
                global_indices=data["global_indices"] if "global_indices" in data.files else None,
                clean_targets=data["clean_targets"],
                noisy_targets=data["noisy_targets"],
                transition_matrix=None if transition.size == 0 else transition,
                per_sample_transition=None if per_sample.size == 0 else per_sample,
                metadata=meta.get("metadata", {}),
            )
            stored_hash = meta.get("mapping_hash")
            if stored_hash is not None and stored_hash != manifest.mapping_hash:
                raise ValueError("noise manifest mapping hash does not match its contents")
            return manifest
