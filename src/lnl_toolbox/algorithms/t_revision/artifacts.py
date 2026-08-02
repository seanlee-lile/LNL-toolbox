from __future__ import annotations

"""Method-local artifact for an unconstrained revised transition matrix."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def _json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        serialized = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
        return MappingProxyType(json.loads(serialized))
    except (TypeError, ValueError) as exc:
        raise TypeError("revised transition metadata must be JSON-compatible") from exc


@dataclass(frozen=True, slots=True)
class RevisedTransitionArtifact:
    """Persist raw ``T_hat + delta`` without claiming stochastic validity."""

    initial_transition: np.ndarray
    delta: np.ndarray
    source_initial_artifact_hash: str
    stage2a_best_checkpoint_sha256: str
    best_noisy_validation_accuracy: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    transition_mode: str = "paper_experiment_raw_additive"

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_transition, dtype=np.float64)
        delta = np.asarray(self.delta, dtype=np.float64)
        if initial.ndim != 2 or initial.shape[0] != initial.shape[1] or initial.shape[0] < 2:
            raise ValueError("initial_transition must have square shape [C, C]")
        if delta.shape != initial.shape:
            raise ValueError("delta must match initial_transition shape")
        if not np.isfinite(initial).all() or not np.isfinite(delta).all():
            raise ValueError("revised transition arrays must be finite")
        if self.transition_mode != "paper_experiment_raw_additive":
            raise ValueError("unsupported revised transition mode")
        for value, owner in (
            (self.source_initial_artifact_hash, "source initial artifact"),
            (self.stage2a_best_checkpoint_sha256, "stage2a best checkpoint"),
        ):
            if len(value) != 64:
                raise ValueError(f"{owner} hash must be SHA-256")
            try:
                bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(f"{owner} hash must be SHA-256") from exc
        metric = float(self.best_noisy_validation_accuracy)
        if not np.isfinite(metric):
            raise ValueError("best noisy validation accuracy must be finite")
        initial = initial.copy()
        delta = delta.copy()
        initial.setflags(write=False)
        delta.setflags(write=False)
        object.__setattr__(self, "initial_transition", initial)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "best_noisy_validation_accuracy", metric)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata))

    @property
    def revised_transition(self) -> np.ndarray:
        result = self.initial_transition + self.delta
        result.setflags(write=False)
        return result

    @property
    def artifact_hash(self) -> str:
        digest = hashlib.sha256()
        context = {
            "version": self.version,
            "transition_mode": self.transition_mode,
            "source_initial_artifact_hash": self.source_initial_artifact_hash,
            "stage2a_best_checkpoint_sha256": self.stage2a_best_checkpoint_sha256,
            "best_noisy_validation_accuracy": self.best_noisy_validation_accuracy,
            "metadata": dict(self.metadata),
        }
        digest.update(json.dumps(context, sort_keys=True, separators=(",", ":")).encode())
        digest.update(self.initial_transition.astype("<f8", copy=False).tobytes())
        digest.update(self.delta.astype("<f8", copy=False).tobytes())
        return digest.hexdigest()

    @property
    def diagnostics(self) -> dict[str, Any]:
        revised = self.revised_transition
        rows = revised.sum(axis=1)
        return {
            "row_sums": rows.tolist(),
            "minimum": float(revised.min()),
            "maximum": float(revised.max()),
            "diagonal": np.diag(revised).tolist(),
            "finite": bool(np.isfinite(revised).all()),
            "non_negative": bool((revised >= 0).all()),
            "row_stochastic": bool(
                (revised >= 0).all()
                and np.allclose(rows, 1.0, rtol=1e-6, atol=1e-8)
            ),
            "delta_l1": float(np.abs(self.delta).sum()),
            "delta_frobenius": float(np.linalg.norm(self.delta)),
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "version": self.version,
            "transition_mode": self.transition_mode,
            "source_initial_artifact_hash": self.source_initial_artifact_hash,
            "stage2a_best_checkpoint_sha256": self.stage2a_best_checkpoint_sha256,
            "best_noisy_validation_accuracy": self.best_noisy_validation_accuracy,
            "metadata": dict(self.metadata),
            "diagnostics": self.diagnostics,
            "artifact_hash": self.artifact_hash,
        }
        np.savez_compressed(
            Path(path),
            initial_transition=self.initial_transition,
            delta=self.delta,
            metadata_json=np.array(json.dumps(payload, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RevisedTransitionArtifact":
        with np.load(path, allow_pickle=False) as values:
            required = {"initial_transition", "delta", "metadata_json"}
            if not required.issubset(values.files):
                raise ValueError("revised transition artifact is missing fields")
            try:
                payload = json.loads(str(values["metadata_json"].item()))
            except (json.JSONDecodeError, ValueError, AttributeError) as exc:
                raise ValueError("revised transition metadata is invalid") from exc
            artifact = cls(
                initial_transition=values["initial_transition"],
                delta=values["delta"],
                source_initial_artifact_hash=str(payload["source_initial_artifact_hash"]),
                stage2a_best_checkpoint_sha256=str(payload["stage2a_best_checkpoint_sha256"]),
                best_noisy_validation_accuracy=float(payload["best_noisy_validation_accuracy"]),
                metadata=payload.get("metadata", {}),
                version=str(payload["version"]),
                transition_mode=str(payload["transition_mode"]),
            )
            if payload.get("artifact_hash") != artifact.artifact_hash:
                raise ValueError("revised transition artifact hash mismatch")
            if payload.get("diagnostics") != artifact.diagnostics:
                raise ValueError("revised transition diagnostics mismatch")
            return artifact
