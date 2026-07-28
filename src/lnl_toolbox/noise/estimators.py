from __future__ import annotations

"""Transition estimators that consume noisy-posterior snapshots only."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .transition import TransitionArtifact


def _readonly_copy(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PosteriorSnapshot:
    """Stable estimator input containing no clean-label information."""

    noisy_probabilities: np.ndarray
    noisy_targets: np.ndarray
    global_indices: np.ndarray
    dataset: str
    split: str

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.noisy_probabilities, dtype=np.float64)
        targets = np.asarray(self.noisy_targets, dtype=np.int64)
        indices = np.asarray(self.global_indices, dtype=np.int64)
        if probabilities.ndim != 2:
            raise ValueError("noisy_probabilities must have shape [N, C]")
        samples, classes = probabilities.shape
        if samples == 0 or classes < 2:
            raise ValueError(
                "noisy_probabilities must contain samples and at least two classes"
            )
        if targets.ndim != 1 or targets.shape != (samples,):
            raise ValueError(f"noisy_targets must have shape [{samples}]")
        if indices.ndim != 1 or indices.shape != (samples,):
            raise ValueError(f"global_indices must have shape [{samples}]")
        if not np.isfinite(probabilities).all():
            raise ValueError("noisy_probabilities must contain only finite values")
        if (probabilities < 0.0).any():
            raise ValueError("noisy_probabilities must be non-negative")
        if not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8
        ):
            raise ValueError("every noisy_probabilities row must sum to one")
        if targets.size and (targets.min() < 0 or targets.max() >= classes):
            raise ValueError(f"noisy_targets must be within [0, {classes})")
        if indices.size and indices.min() < 0:
            raise ValueError("global_indices must be non-negative")
        if np.unique(indices).size != samples:
            raise ValueError("global_indices must be unique")
        dataset = self.dataset.strip()
        split = self.split.strip()
        if not dataset or not split:
            raise ValueError("dataset and split must not be empty")
        object.__setattr__(
            self, "noisy_probabilities", _readonly_copy(probabilities, np.float64)
        )
        object.__setattr__(self, "noisy_targets", _readonly_copy(targets, np.int64))
        object.__setattr__(self, "global_indices", _readonly_copy(indices, np.int64))
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "split", split)

    @property
    def num_samples(self) -> int:
        return int(self.noisy_probabilities.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.noisy_probabilities.shape[1])

    @property
    def snapshot_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {"dataset": self.dataset, "split": self.split},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(str(self.noisy_probabilities.shape).encode("ascii"))
        digest.update(
            self.noisy_probabilities.astype("<f8", copy=False).tobytes(order="C")
        )
        digest.update(self.noisy_targets.astype("<i8", copy=False).tobytes(order="C"))
        digest.update(self.global_indices.astype("<i8", copy=False).tobytes(order="C"))
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        """Persist the snapshot with its dataset identity and content hash."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "dataset": self.dataset,
            "split": self.split,
            "snapshot_hash": self.snapshot_hash,
        }
        np.savez_compressed(
            destination,
            noisy_probabilities=self.noisy_probabilities,
            noisy_targets=self.noisy_targets,
            global_indices=self.global_indices,
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PosteriorSnapshot":
        """Load and integrity-check a persisted posterior snapshot."""

        with np.load(path, allow_pickle=False) as data:
            required = {
                "noisy_probabilities",
                "noisy_targets",
                "global_indices",
                "metadata_json",
            }
            if not required.issubset(data.files):
                raise ValueError("posterior snapshot is missing required fields")
            try:
                metadata = json.loads(str(data["metadata_json"].item()))
            except (AttributeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError("posterior snapshot metadata is invalid") from exc
            if not isinstance(metadata, dict):
                raise ValueError("posterior snapshot metadata must be a mapping")
            snapshot = cls(
                noisy_probabilities=data["noisy_probabilities"],
                noisy_targets=data["noisy_targets"],
                global_indices=data["global_indices"],
                dataset=str(metadata.get("dataset", "")),
                split=str(metadata.get("split", "")),
            )
            if metadata.get("snapshot_hash") != snapshot.snapshot_hash:
                raise ValueError("posterior snapshot hash does not match its contents")
            return snapshot


@runtime_checkable
class TransitionEstimator(Protocol):
    """Estimate one class-conditional transition artifact from noisy data."""

    def estimate(self, snapshot: PosteriorSnapshot) -> TransitionArtifact:
        ...


@dataclass(frozen=True, slots=True)
class AnchorTransitionEstimator:
    """Patrini et al. (CVPR 2017), Equations (12)-(13)."""

    def estimate(self, snapshot: PosteriorSnapshot) -> TransitionArtifact:
        probabilities = snapshot.noisy_probabilities
        anchors: list[int] = []
        anchor_scores: list[float] = []
        positions: list[int] = []
        for class_index in range(snapshot.num_classes):
            scores = probabilities[:, class_index]
            maximum = scores.max()
            tied_positions = np.flatnonzero(scores == maximum)
            tied_indices = snapshot.global_indices[tied_positions]
            selected_position = int(tied_positions[np.argmin(tied_indices)])
            positions.append(selected_position)
            anchors.append(int(snapshot.global_indices[selected_position]))
            anchor_scores.append(float(maximum))

        matrix = probabilities[np.asarray(positions, dtype=np.int64)]
        return TransitionArtifact(
            matrix=matrix,
            estimator="anchor",
            source_snapshot_hash=snapshot.snapshot_hash,
            metadata={
                "dataset": snapshot.dataset,
                "split": snapshot.split,
                "num_samples": snapshot.num_samples,
                "anchor_global_indices": anchors,
                "anchor_scores": anchor_scores,
                "selection": "argmax_noisy_posterior_tie_min_global_index",
                "paper": "Patrini et al., CVPR 2017, Equations (12)-(13)",
            },
        )


@dataclass(frozen=True, slots=True)
class DualTransitionEstimator:
    """Yao et al. (NeurIPS 2020), Algorithm 1, in row-vector convention."""

    def estimate(self, snapshot: PosteriorSnapshot) -> TransitionArtifact:
        t_club_artifact = AnchorTransitionEstimator().estimate(snapshot)
        probabilities = snapshot.noisy_probabilities
        intermediate_targets = probabilities.argmax(axis=1)

        counts = np.zeros(
            (snapshot.num_classes, snapshot.num_classes), dtype=np.int64
        )
        np.add.at(counts, (intermediate_targets, snapshot.noisy_targets), 1)
        totals = counts.sum(axis=1)
        missing = np.flatnonzero(totals == 0)
        if missing.size:
            missing_values = ", ".join(str(int(value)) for value in missing)
            raise ValueError(
                "Dual-T cannot estimate intermediate-to-noisy rows for empty "
                f"intermediate classes: {missing_values}"
            )

        t_spade = counts / totals[:, None]
        matrix = t_club_artifact.matrix @ t_spade
        return TransitionArtifact(
            matrix=matrix,
            estimator="dual_t",
            source_snapshot_hash=snapshot.snapshot_hash,
            metadata={
                "dataset": snapshot.dataset,
                "split": snapshot.split,
                "num_samples": snapshot.num_samples,
                "composition": "t_club @ t_spade",
                "t_club": t_club_artifact.matrix.tolist(),
                "t_spade": t_spade.tolist(),
                "t_spade_counts": counts.tolist(),
                "t_club_artifact_hash": t_club_artifact.artifact_hash,
                "anchor_global_indices": t_club_artifact.metadata[
                    "anchor_global_indices"
                ],
                "intermediate_assignment": "argmax_tie_min_class_index",
                "paper": "Yao et al., NeurIPS 2020, Algorithm 1",
            },
        )
