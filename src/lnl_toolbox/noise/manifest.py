from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .transition import validate_transition_matrix


def fingerprint_labels(labels: np.ndarray) -> str:
    values = np.asarray(labels, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_dataset_name(value: str) -> str:
    return "".join(character for character in value.strip().lower() if character.isalnum())


def _validate_per_sample_transition(values: np.ndarray, samples: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != samples:
        raise ValueError(f"per_sample_transition must have shape [{samples}, C]")
    if probabilities.shape[1] < 2:
        raise ValueError("per_sample_transition must contain at least two classes")
    if not np.isfinite(probabilities).all():
        raise ValueError("per_sample_transition must contain only finite values")
    if (probabilities < 0.0).any():
        raise ValueError("per_sample_transition must be non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("every per_sample_transition row must sum to one")
    return probabilities.copy()


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
        if self.clean_targets.ndim != 1 or self.noisy_targets.ndim != 1:
            raise ValueError("clean_targets and noisy_targets must be one-dimensional")
        if self.clean_targets.shape != self.noisy_targets.shape:
            raise ValueError("clean_targets and noisy_targets must have the same shape")
        if self.clean_targets.size and (
            self.clean_targets.min() < 0 or self.noisy_targets.min() < 0
        ):
            raise ValueError("clean_targets and noisy_targets must be non-negative")
        if not 0.0 <= self.requested_rate <= 1.0:
            raise ValueError("requested_rate must be in [0, 1]")
        expected_fingerprint = fingerprint_labels(self.clean_targets)
        if self.dataset_fingerprint and self.dataset_fingerprint != expected_fingerprint:
            raise ValueError("dataset_fingerprint does not match clean_targets")
        self.dataset_fingerprint = expected_fingerprint
        if self.transition_matrix is not None:
            self.transition_matrix = validate_transition_matrix(self.transition_matrix)
        if self.per_sample_transition is not None:
            self.per_sample_transition = _validate_per_sample_transition(
                self.per_sample_transition, self.clean_targets.size
            )

    def validate_for(
        self,
        clean_targets: np.ndarray,
        dataset: str,
        num_classes: int,
    ) -> "NoiseManifest":
        """Verify that this manifest is safe to apply by stable global index."""

        reference = np.asarray(clean_targets, dtype=np.int64)
        if reference.ndim != 1:
            raise ValueError("reference clean targets must be one-dimensional")
        if _canonical_dataset_name(self.dataset) != _canonical_dataset_name(dataset):
            raise ValueError(
                f"Noise manifest dataset {self.dataset!r} does not match {dataset!r}"
            )
        if reference.shape != self.clean_targets.shape:
            raise ValueError(
                "Noise manifest length does not match the current training dataset"
            )
        if fingerprint_labels(reference) != self.dataset_fingerprint:
            raise ValueError("Noise manifest fingerprint does not match the current dataset")
        if not np.array_equal(reference, self.clean_targets):
            raise ValueError("Noise manifest clean targets do not match the current dataset")
        for name, values in (
            ("clean_targets", self.clean_targets),
            ("noisy_targets", self.noisy_targets),
        ):
            if values.size and (values.min() < 0 or values.max() >= num_classes):
                raise ValueError(f"{name} must be within [0, {num_classes})")
        if self.transition_matrix is not None:
            self.transition_matrix = validate_transition_matrix(
                self.transition_matrix, num_classes
            )
        if self.per_sample_transition is not None:
            if self.per_sample_transition.shape != (reference.size, num_classes):
                raise ValueError(
                    "per_sample_transition must have shape "
                    f"[{reference.size}, {num_classes}]"
                )
        return self

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

