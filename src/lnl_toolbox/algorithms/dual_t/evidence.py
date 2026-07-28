from __future__ import annotations

"""Offline evidence contracts for comparing ordinary T and Dual-T."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from lnl_toolbox.noise.estimators import (
    AnchorTransitionEstimator,
    DualTransitionEstimator,
    PosteriorSnapshot,
)
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.noise.transition import (
    TRANSITION_CONVENTION,
    TransitionArtifact,
    validate_transition_matrix,
)


@dataclass(frozen=True, slots=True)
class TransitionMatrixError:
    """Elementwise errors against one synthetic generating matrix."""

    l1_total: float
    l1_mean: float
    frobenius: float
    max_absolute_error: float
    row_l1: tuple[float, ...]
    diagonal_absolute_error_mean: float
    off_diagonal_absolute_error_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "l1_total": self.l1_total,
            "l1_mean": self.l1_mean,
            "frobenius": self.frobenius,
            "max_absolute_error": self.max_absolute_error,
            "row_l1": list(self.row_l1),
            "diagonal_absolute_error_mean": (
                self.diagonal_absolute_error_mean
            ),
            "off_diagonal_absolute_error_mean": (
                self.off_diagonal_absolute_error_mean
            ),
        }


@dataclass(frozen=True, slots=True)
class TransitionEvidence:
    """Same-snapshot ordinary and Dual-T estimates plus offline errors."""

    snapshot_hash: str
    ground_truth_matrix: np.ndarray
    anchor_artifact: TransitionArtifact
    dual_t_artifact: TransitionArtifact
    anchor_error: TransitionMatrixError
    dual_t_error: TransitionMatrixError
    realized_empirical_matrix: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        improvement = self.anchor_error.l1_total - self.dual_t_error.l1_total
        denominator = self.anchor_error.l1_total
        relative = 0.0 if denominator == 0.0 else improvement / denominator
        return {
            "snapshot_hash": self.snapshot_hash,
            "transition_convention": TRANSITION_CONVENTION,
            "ground_truth_matrix": self.ground_truth_matrix.tolist(),
            "realized_empirical_matrix": (
                None
                if self.realized_empirical_matrix is None
                else self.realized_empirical_matrix.tolist()
            ),
            "anchor": {
                "artifact_hash": self.anchor_artifact.artifact_hash,
                "source_snapshot_hash": (
                    self.anchor_artifact.source_snapshot_hash
                ),
                "matrix": self.anchor_artifact.matrix.tolist(),
                "error": self.anchor_error.to_dict(),
            },
            "dual_t": {
                "artifact_hash": self.dual_t_artifact.artifact_hash,
                "source_snapshot_hash": (
                    self.dual_t_artifact.source_snapshot_hash
                ),
                "matrix": self.dual_t_artifact.matrix.tolist(),
                "error": self.dual_t_error.to_dict(),
            },
            "dual_t_l1_improvement": improvement,
            "dual_t_relative_l1_improvement": relative,
            "dual_t_has_lower_l1_error": (
                self.dual_t_error.l1_total
                < self.anchor_error.l1_total
            ),
        }


@dataclass(frozen=True, slots=True)
class FinalArmEvidence:
    """Classification evidence for one independently trained final arm."""

    name: str
    initial_state_hash: str
    sampler_seed: int
    completed_epochs: int
    global_step: int
    best_validation_epoch: int
    best_noisy_validation_accuracy: float
    best_checkpoint_clean_test_loss: float
    best_checkpoint_clean_test_accuracy: float
    final_epoch_clean_test_loss: float
    final_epoch_clean_test_accuracy: float
    batch_index_hashes: tuple[str, ...]
    input_tensor_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "initial_state_hash": self.initial_state_hash,
            "sampler_seed": self.sampler_seed,
            "completed_epochs": self.completed_epochs,
            "global_step": self.global_step,
            "best_validation_epoch": self.best_validation_epoch,
            "best_noisy_validation_accuracy": (
                self.best_noisy_validation_accuracy
            ),
            "best_checkpoint_clean_test_loss": (
                self.best_checkpoint_clean_test_loss
            ),
            "best_checkpoint_clean_test_accuracy": (
                self.best_checkpoint_clean_test_accuracy
            ),
            "final_epoch_clean_test_loss": self.final_epoch_clean_test_loss,
            "final_epoch_clean_test_accuracy": (
                self.final_epoch_clean_test_accuracy
            ),
            "batch_index_hashes": list(self.batch_index_hashes),
            "input_tensor_hashes": list(self.input_tensor_hashes),
        }


def transition_matrix_error(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> TransitionMatrixError:
    """Return the paper's elementwise L1 distance plus diagnostics."""

    estimate = validate_transition_matrix(estimated)
    truth = validate_transition_matrix(ground_truth, estimate.shape[0])
    absolute = np.abs(estimate - truth)
    classes = estimate.shape[0]
    diagonal = absolute[np.eye(classes, dtype=bool)]
    off_diagonal = absolute[~np.eye(classes, dtype=bool)]
    return TransitionMatrixError(
        l1_total=float(absolute.sum()),
        l1_mean=float(absolute.mean()),
        frobenius=float(np.linalg.norm(estimate - truth)),
        max_absolute_error=float(absolute.max()),
        row_l1=tuple(float(value) for value in absolute.sum(axis=1)),
        diagonal_absolute_error_mean=float(diagonal.mean()),
        off_diagonal_absolute_error_mean=float(off_diagonal.mean()),
    )


def realized_empirical_transition(
    manifest: NoiseManifest,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """Compute an offline realized matrix without exposing it to training."""

    requested = np.asarray(sample_indices, dtype=np.int64)
    if requested.ndim != 1 or requested.size == 0:
        raise ValueError("sample_indices must be a non-empty vector")
    if np.unique(requested).size != requested.size:
        raise ValueError("sample_indices must be unique")
    order = np.argsort(manifest.global_indices)
    sorted_indices = manifest.global_indices[order]
    positions = np.searchsorted(sorted_indices, requested)
    valid = positions < sorted_indices.size
    matched = np.zeros(requested.shape, dtype=bool)
    matched[valid] = sorted_indices[positions[valid]] == requested[valid]
    if not matched.all():
        missing = requested[~matched]
        raise ValueError(
            "sample_indices are missing from the noise manifest: "
            + ", ".join(str(int(value)) for value in missing)
        )
    manifest_positions = order[positions]
    clean = manifest.clean_targets[manifest_positions]
    noisy = manifest.noisy_targets[manifest_positions]
    classes = int(manifest.num_classes)
    counts = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(counts, (clean, noisy), 1)
    totals = counts.sum(axis=1)
    missing_classes = np.flatnonzero(totals == 0)
    if missing_classes.size:
        raise ValueError(
            "cannot compute realized transition for empty clean classes: "
            + ", ".join(str(int(value)) for value in missing_classes)
        )
    return counts / totals[:, None]


def build_transition_evidence(
    *,
    snapshot: PosteriorSnapshot,
    manifest: NoiseManifest,
    sample_indices: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TransitionEvidence:
    """Estimate ordinary and Dual-T from exactly one persisted snapshot."""

    if manifest.transition_matrix is None:
        raise ValueError(
            "synthetic evidence requires NoiseManifest.transition_matrix"
        )
    ground_truth = validate_transition_matrix(
        manifest.transition_matrix,
        snapshot.num_classes,
    )
    anchor_base = AnchorTransitionEstimator().estimate(snapshot)
    dual_base = DualTransitionEstimator().estimate(snapshot)
    shared_metadata = dict(metadata or {})
    shared_metadata["evidence_snapshot_hash"] = snapshot.snapshot_hash
    anchor = TransitionArtifact(
        matrix=anchor_base.matrix,
        estimator=anchor_base.estimator,
        source_snapshot_hash=anchor_base.source_snapshot_hash,
        metadata={**dict(anchor_base.metadata), **shared_metadata},
    )
    dual_t = TransitionArtifact(
        matrix=dual_base.matrix,
        estimator=dual_base.estimator,
        source_snapshot_hash=dual_base.source_snapshot_hash,
        metadata={**dict(dual_base.metadata), **shared_metadata},
    )
    if (
        anchor.source_snapshot_hash != snapshot.snapshot_hash
        or dual_t.source_snapshot_hash != snapshot.snapshot_hash
    ):
        raise ValueError(
            "transition estimates must share the persisted snapshot hash"
        )
    realized = (
        None
        if sample_indices is None
        else realized_empirical_transition(manifest, sample_indices)
    )
    return TransitionEvidence(
        snapshot_hash=snapshot.snapshot_hash,
        ground_truth_matrix=ground_truth,
        anchor_artifact=anchor,
        dual_t_artifact=dual_t,
        anchor_error=transition_matrix_error(anchor.matrix, ground_truth),
        dual_t_error=transition_matrix_error(dual_t.matrix, ground_truth),
        realized_empirical_matrix=realized,
    )
