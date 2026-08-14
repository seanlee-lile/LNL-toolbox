from __future__ import annotations

"""Part-dependent transition estimation and compact instance artifacts."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lnl_toolbox.training.snapshots import FeatureSnapshot
from .estimators import PosteriorSnapshot


def project_probability_simplex(values: np.ndarray) -> np.ndarray:
    """Project the last axis onto the probability simplex."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] == 0:
        raise ValueError("values must have a non-empty final axis")
    flat = array.reshape(-1, array.shape[-1])
    ordered = np.sort(flat, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    ranks = np.arange(1, flat.shape[1] + 1, dtype=np.float64)
    valid = ordered - cumulative / ranks > 0.0
    rho = valid.sum(axis=1) - 1
    theta = cumulative[np.arange(flat.shape[0]), rho] / (rho + 1.0)
    projected = np.maximum(flat - theta[:, None], 0.0)
    return projected.reshape(array.shape)


def fit_part_representation(
    features: np.ndarray,
    num_parts: int,
    *,
    seed: int | None = 0,
    iterations: int = 200,
    error_tolerance: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the official PDL multiplicative-update NMF (``train_m``).

    The source implementation uses ``W[N, R]`` and ``H[R, D]``.  The toolbox
    returns the equivalent ``parts[D, R] = H.T`` and normalized
    ``coefficients[N, R] = W``.
    """

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("features must have shape [N, D]")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("features must be finite and non-negative")
    if not 1 <= int(num_parts) <= min(values.shape):
        raise ValueError("num_parts must be within [1, min(N, D)]")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not np.isfinite(error_tolerance) or error_tolerance < 0.0:
        raise ValueError("error_tolerance must be finite and non-negative")

    # ``tools.train_m`` uses NumPy's module-level RNG.  ``seed=None`` is an
    # explicit official-source mode; the default keeps the reusable helper's
    # isolated deterministic behavior for non-official callers.
    random = np.random if seed is None else np.random.RandomState(seed)
    coefficients = random.random((values.shape[0], int(num_parts)))
    latent = random.random((int(num_parts), values.shape[1]))
    for _ in range(int(iterations)):
        error = values - coefficients @ latent
        if float(np.square(error).sum()) < float(error_tolerance):
            break
        numerator = coefficients.T @ values
        denominator = (coefficients.T @ coefficients) @ latent
        latent *= np.divide(
            numerator, denominator, out=np.zeros_like(numerator),
            where=denominator != 0.0,
        )
        numerator = values @ latent.T
        denominator = (coefficients @ latent) @ latent.T
        coefficients *= np.divide(
            numerator, denominator, out=np.zeros_like(numerator),
            where=denominator != 0.0,
        )
    # Official ``train_m`` applies ``tools.norm(W)`` once, after the complete
    # multiplicative-update loop, not after every iteration.
    coefficients /= np.maximum(coefficients.sum(axis=1, keepdims=True), 1e-12)
    return latent.T, coefficients


def select_pdl_anchor_candidates(
    probabilities: np.ndarray,
    percentages: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Replicate official ``tools.fit(..., filter_outlier=True)``."""

    values = np.asarray(probabilities, dtype=np.float64)
    levels = np.asarray(percentages, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("probabilities must have shape [N, C]")
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("percentages must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if np.any((levels < 0.0) | (levels > 100.0)):
        raise ValueError("anchor percentages must lie in [0, 100]")
    positions = np.empty((values.shape[1], levels.size), dtype=np.int64)
    for class_index in range(values.shape[1]):
        column = values[:, class_index]
        for basis_index, percentage in enumerate(levels):
            try:
                threshold = np.percentile(column, float(percentage), method="higher")
            except TypeError:
                threshold = np.percentile(column, float(percentage), interpolation="higher")
            robust = column.copy()
            robust[robust >= threshold] = 0.0
            positions[class_index, basis_index] = int(np.argmax(robust))
    return positions


def fit_pdl_basis_matrices(
    anchor_coefficients: np.ndarray,
    anchor_posteriors: np.ndarray,
    *,
    epochs: int = 1500,
    learning_rate: float = 0.001,
    loss_threshold: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """Fit one official ``basis_matrix_optimize`` group.

    The reference code constructs ``Matrix_optimize`` once, initializes its
    weights with standard deviation ``1e-1``, then calls ``init_params`` at
    the beginning of every clean-class group.  ``init_params`` resets each
    linear row to ``N(0, 1e-1)`` while the Adam state is kept.
    The optional pair helper below preserves the same optimizer lifetime
    across the official train and validation calls.
    """

    coefficients = np.asarray(anchor_coefficients, dtype=np.float64)
    targets = np.asarray(anchor_posteriors, dtype=np.float64)
    if coefficients.ndim != 3 or targets.ndim != 3:
        raise ValueError("PDL basis inputs must have shape [C, R, R] and [C, R, C]")
    classes, basis, parts = coefficients.shape
    if parts != basis or targets.shape != (classes, basis, classes):
        raise ValueError("PDL basis coefficient/target dimensions do not match")
    if epochs < 1 or learning_rate <= 0.0 or loss_threshold < 0.0:
        raise ValueError("invalid PDL basis optimization hyperparameters")
    if not np.isfinite(coefficients).all() or not np.isfinite(targets).all():
        raise ValueError("PDL basis inputs must be finite")

    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(int(seed))
    weights = nn.Parameter(torch.empty((basis, classes), dtype=torch.float32))
    nn.init.normal_(weights, mean=0.0, std=0.1)
    optimizer = torch.optim.Adam([weights], lr=float(learning_rate))
    return _fit_pdl_basis_group(
        coefficients,
        targets,
        weights,
        optimizer,
        epochs=int(epochs),
        loss_threshold=float(loss_threshold),
    )


def _fit_pdl_basis_group(
    coefficients: np.ndarray,
    targets: np.ndarray,
    weights: "torch.nn.Parameter",
    optimizer: "torch.optim.Optimizer",
    *,
    epochs: int,
    loss_threshold: float,
    normalize_output: bool = True,
) -> np.ndarray:
    """Run one class loop from the reference ``basis_matrix_optimize``."""

    import torch
    from torch.nn import functional as F

    classes, basis, parts = coefficients.shape
    if parts != basis or targets.shape != (classes, basis, classes):
        raise ValueError("PDL basis group dimensions do not match")
    result = np.empty((basis, classes, classes), dtype=np.float64)
    for class_index in range(classes):
        # Exact ``tools.init_params`` behavior for Matrix_optimize: reset
        # Linear rows to N(0, 1e-1), but do not reset Adam's state.
        with torch.no_grad():
            torch.nn.init.normal_(weights, mean=0.0, std=1e-1)
        class_coefficients = torch.as_tensor(
            coefficients[class_index], dtype=torch.float32
        )
        class_targets = torch.as_tensor(
            targets[class_index], dtype=torch.float32
        )
        for _ in range(int(epochs)):
            loss_total = torch.zeros((), dtype=torch.float32)
            for basis_index in range(basis):
                with torch.no_grad():
                    normalized = weights.abs() / weights.abs().sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-12)
                    weights.copy_(normalized)
                prediction = (
                    class_coefficients[basis_index, :, None] * weights
                ).sum(dim=0)
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(prediction, class_targets[basis_index])
                loss.backward()
                optimizer.step()
                loss_total = loss_total + loss.detach()
            if float(loss_total.item()) < float(loss_threshold):
                break
        with torch.no_grad():
            output = weights if not normalize_output else weights.abs() / weights.abs().sum(
                dim=1, keepdim=True
            ).clamp_min(1e-12)
        result[:, class_index, :] = output.detach().cpu().numpy()
    return result


def fit_pdl_basis_matrices_pair(
    train_coefficients: np.ndarray,
    train_targets: np.ndarray,
    validation_coefficients: np.ndarray,
    validation_targets: np.ndarray,
    *,
    epochs: int = 1500,
    learning_rate: float = 0.001,
    loss_threshold: float = 0.02,
    seed: int = 0,
    official_raw: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PDL train/validation groups with one reference optimizer state."""

    train_coefficients = np.asarray(train_coefficients, dtype=np.float64)
    train_targets = np.asarray(train_targets, dtype=np.float64)
    validation_coefficients = np.asarray(validation_coefficients, dtype=np.float64)
    validation_targets = np.asarray(validation_targets, dtype=np.float64)
    if train_coefficients.ndim != 3 or train_targets.ndim != 3:
        raise ValueError("train PDL basis inputs must be rank three")
    if validation_coefficients.ndim != 3 or validation_targets.ndim != 3:
        raise ValueError("validation PDL basis inputs must be rank three")
    if train_coefficients.shape[1:] != validation_coefficients.shape[1:]:
        raise ValueError("train and validation basis dimensions must match")
    classes, basis, parts = train_coefficients.shape
    if parts != basis or train_targets.shape != (classes, basis, classes):
        raise ValueError("train PDL basis dimensions do not match")
    if validation_targets.shape != (classes, basis, classes):
        raise ValueError("validation PDL basis dimensions do not match")
    if epochs < 1 or learning_rate <= 0.0 or loss_threshold < 0.0:
        raise ValueError("invalid PDL basis optimization hyperparameters")
    if not (
        np.isfinite(train_coefficients).all()
        and np.isfinite(train_targets).all()
        and np.isfinite(validation_coefficients).all()
        and np.isfinite(validation_targets).all()
    ):
        raise ValueError("PDL basis inputs must be finite")

    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    weights = nn.Parameter(torch.empty((basis, classes), dtype=torch.float32))
    nn.init.normal_(weights, mean=0.0, std=0.1)
    optimizer = torch.optim.Adam([weights], lr=float(learning_rate))
    train = _fit_pdl_basis_group(
        train_coefficients,
        train_targets,
        weights,
        optimizer,
        epochs=int(epochs),
        loss_threshold=float(loss_threshold),
        normalize_output=not official_raw,
    )
    validation = _fit_pdl_basis_group(
        validation_coefficients,
        validation_targets,
        weights,
        optimizer,
        epochs=int(epochs),
        loss_threshold=float(loss_threshold),
        normalize_output=not official_raw,
    )
    return train, validation


def fit_part_transition_matrices(
    anchor_coefficients: np.ndarray,
    anchor_posteriors: np.ndarray,
) -> np.ndarray:
    """Fit Eq. (4) and return row-stochastic part matrices ``[R,C,C]``."""

    coefficients = np.asarray(anchor_coefficients, dtype=np.float64)
    posteriors = np.asarray(anchor_posteriors, dtype=np.float64)
    if coefficients.ndim != 3:
        raise ValueError("anchor_coefficients must have shape [C, K, R]")
    classes, candidates, parts = coefficients.shape
    if posteriors.shape != (classes, candidates, classes):
        raise ValueError("anchor_posteriors must have shape [C, K, C]")
    if candidates < parts:
        raise ValueError("at least num_parts anchor candidates are required per class")
    if not np.isfinite(coefficients).all() or not np.isfinite(posteriors).all():
        raise ValueError("anchor values must be finite")

    matrices = np.empty((parts, classes, classes), dtype=np.float64)
    for clean_class in range(classes):
        design = coefficients[clean_class]
        if np.linalg.matrix_rank(design) < parts:
            raise ValueError(
                f"anchor coefficients for class {clean_class} are rank deficient"
            )
        fitted = np.linalg.lstsq(design, posteriors[clean_class], rcond=None)[0]
        matrices[:, clean_class, :] = project_probability_simplex(fitted)
    return matrices


@dataclass(frozen=True, slots=True)
class PartTransitionArtifact:
    """Compact ``T(x)=sum_r h_r(x) P_r`` artifact aligned by global index."""

    parts: np.ndarray
    coefficients: np.ndarray
    part_matrices: np.ndarray
    global_indices: np.ndarray
    feature_snapshot_hash: str
    posterior_snapshot_hash: str
    anchor_indices: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    convention: str = "clean_to_noisy_row"

    def __post_init__(self) -> None:
        parts = np.asarray(self.parts, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        matrices = np.asarray(self.part_matrices, dtype=np.float64)
        indices = np.asarray(self.global_indices, dtype=np.int64)
        anchors = np.asarray(self.anchor_indices, dtype=np.int64)
        if self.version != "1.0":
            raise ValueError(f"unsupported part transition artifact version: {self.version!r}")
        if self.convention != "clean_to_noisy_row":
            raise ValueError("part transition convention must be 'clean_to_noisy_row'")
        if parts.ndim != 2:
            raise ValueError("parts must have shape [D, R]")
        samples, num_parts = coefficients.shape if coefficients.ndim == 2 else (-1, -1)
        if parts.shape[1] != num_parts or samples <= 0:
            raise ValueError("coefficients must have shape [N, R] matching parts")
        if matrices.ndim != 3 or matrices.shape[0] != num_parts or matrices.shape[1] != matrices.shape[2]:
            raise ValueError("part_matrices must have shape [R, C, C]")
        if indices.shape != (samples,) or np.unique(indices).size != samples:
            raise ValueError("global_indices must be unique shape [N]")
        if anchors.ndim != 2 or anchors.shape[0] != matrices.shape[1]:
            raise ValueError("anchor_indices must have shape [C, K]")
        for name, value in (("feature", self.feature_snapshot_hash), ("posterior", self.posterior_snapshot_hash)):
            if len(value) != 64:
                raise ValueError(f"{name}_snapshot_hash must be an SHA-256 digest")
            try:
                bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(f"{name}_snapshot_hash must be an SHA-256 digest") from exc
        if not np.isfinite(parts).all() or not np.isfinite(coefficients).all() or not np.isfinite(matrices).all():
            raise ValueError("artifact arrays must be finite")
        if (coefficients < -1e-10).any() or not np.allclose(coefficients.sum(axis=1), 1.0):
            raise ValueError("coefficient rows must be probabilities")
        official_raw_basis = bool(dict(self.metadata).get("official_raw_basis", False))
        if not official_raw_basis and (
            (matrices < -1e-10).any() or not np.allclose(matrices.sum(axis=2), 1.0)
        ):
            raise ValueError("every part transition row must be a probability")
        order = np.argsort(indices, kind="stable")
        metadata = MappingProxyType(json.loads(json.dumps(dict(self.metadata), sort_keys=True)))
        for name, value in (
            ("parts", parts), ("coefficients", coefficients[order]),
            ("part_matrices", matrices), ("global_indices", indices[order]),
            ("anchor_indices", anchors),
        ):
            copy = value.copy()
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)
        object.__setattr__(self, "metadata", metadata)

    @property
    def num_classes(self) -> int:
        return int(self.part_matrices.shape[1])

    @property
    def artifact_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps({
            "version": self.version, "convention": self.convention,
            "feature_snapshot_hash": self.feature_snapshot_hash,
            "posterior_snapshot_hash": self.posterior_snapshot_hash,
            "metadata": dict(self.metadata),
        }, sort_keys=True, separators=(",", ":")).encode())
        for value in (self.parts, self.coefficients, self.part_matrices, self.global_indices, self.anchor_indices):
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    def transitions_for(self, sample_indices: Any, *, device: Any = None, dtype: Any = None):
        import torch

        requested = torch.as_tensor(sample_indices).detach().cpu().numpy().astype(np.int64)
        if requested.ndim != 1:
            raise ValueError("sample_indices must have shape [B]")
        positions = np.searchsorted(self.global_indices, requested)
        valid = positions < self.global_indices.size
        if not np.all(valid) or not np.array_equal(self.global_indices[positions[valid]], requested[valid]):
            raise KeyError("instance transition artifact does not cover every sample index")
        transitions = np.einsum("br,rcd->bcd", self.coefficients[positions], self.part_matrices)
        if bool(dict(self.metadata).get("official_raw_basis", False)):
            transitions[transitions < 1.0e-6] = 0.0
        return torch.as_tensor(transitions.copy(), device=device, dtype=dtype or torch.float32)

    def transition_for(self, inputs: Any, sample_indices: Any, *, device: Any = None, dtype: Any = None):
        return self.transitions_for(sample_indices, device=device, dtype=dtype)

    def with_part_matrices(
        self,
        part_matrices: np.ndarray,
        *,
        role: str,
        source_artifact_hash: str | None = None,
    ) -> "PartTransitionArtifact":
        """Return this index-aligned artifact with a different basis matrix set.

        PDL's official revision validation path keeps the validation NMF
        coefficients but reuses the matrices fitted from the training split.
        This operation makes that otherwise easy-to-miss lineage explicit
        without duplicating the artifact schema.
        """

        role = str(role).strip()
        if not role:
            raise ValueError("artifact role must not be empty")
        metadata = dict(self.metadata)
        metadata["artifact_role"] = role
        if source_artifact_hash is not None:
            source_artifact_hash = str(source_artifact_hash)
            if len(source_artifact_hash) != 64:
                raise ValueError("source_artifact_hash must be an SHA-256 digest")
            try:
                bytes.fromhex(source_artifact_hash)
            except ValueError as exc:
                raise ValueError("source_artifact_hash must be an SHA-256 digest") from exc
            metadata["source_artifact_hash"] = source_artifact_hash
        return PartTransitionArtifact(
            self.parts,
            self.coefficients,
            part_matrices,
            self.global_indices,
            self.feature_snapshot_hash,
            self.posterior_snapshot_hash,
            self.anchor_indices,
            metadata=metadata,
            version=self.version,
            convention=self.convention,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version, "convention": self.convention,
            "feature_snapshot_hash": self.feature_snapshot_hash,
            "posterior_snapshot_hash": self.posterior_snapshot_hash,
            "metadata": dict(self.metadata), "artifact_hash": self.artifact_hash,
        }
        np.savez_compressed(destination, parts=self.parts, coefficients=self.coefficients,
            part_matrices=self.part_matrices, global_indices=self.global_indices,
            anchor_indices=self.anchor_indices, metadata_json=np.array(json.dumps(payload, sort_keys=True)))

    @classmethod
    def load(cls, path: str | Path) -> "PartTransitionArtifact":
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["metadata_json"].item()))
            artifact = cls(data["parts"], data["coefficients"], data["part_matrices"],
                data["global_indices"], payload["feature_snapshot_hash"],
                payload["posterior_snapshot_hash"], data["anchor_indices"],
                payload.get("metadata", {}), payload["version"], payload["convention"])
            if artifact.artifact_hash != payload.get("artifact_hash"):
                raise ValueError("part transition artifact hash does not match contents")
            return artifact


@dataclass(frozen=True, slots=True)
class PartTransitionEstimator:
    num_parts: int
    anchor_candidates: int
    representation_seed: int = 0
    representation_iterations: int = 200
    representation_error_tolerance: float = 1e-5
    anchor_percentages: tuple[float, ...] | None = None
    basis_epochs: int = 1500
    basis_learning_rate: float = 0.001
    basis_loss_threshold: float = 0.02

    def _anchor_percentages(self) -> tuple[float, ...]:
        percentages = (
            tuple(float(value) for value in self.anchor_percentages)
            if self.anchor_percentages is not None
            else tuple(np.linspace(97.0, 99.0, self.num_parts))
        )
        if len(percentages) != self.num_parts:
            raise ValueError("PDL anchor_percentages must contain num_parts values")
        return percentages

    def anchor_percentages_for_paper(self) -> tuple[float, ...]:
        """Return the exact percentile grid used by the reference runner."""

        return self._anchor_percentages()

    def estimate_from_shared_representation(
        self,
        features: FeatureSnapshot,
        posteriors: PosteriorSnapshot,
        *,
        representation_parts: np.ndarray,
        representation_coefficients: np.ndarray,
        representation_indices: np.ndarray,
        part_matrices: np.ndarray | None = None,
        official_raw_basis: bool = False,
    ) -> PartTransitionArtifact:
        """Estimate one PDL artifact from a representation shared by splits.

        The official runner performs NMF once on the concatenated train and
        validation representation, then runs anchor selection and
        ``Matrix_optimize`` independently for each split.  The supplied
        representation arrays are therefore immutable shared inputs, while
        this method returns split-local coefficients, anchors, and matrices.
        """

        if features.dataset != posteriors.dataset or features.split != posteriors.split:
            raise ValueError("feature and posterior snapshots must describe the same dataset split")
        if not np.array_equal(features.global_indices, posteriors.global_indices):
            raise ValueError("feature and posterior snapshots must use identical global indices")
        parts = np.asarray(representation_parts, dtype=np.float64)
        coefficients = np.asarray(representation_coefficients, dtype=np.float64)
        source_indices = np.asarray(representation_indices, dtype=np.int64)
        if parts.ndim != 2 or parts.shape[1] != self.num_parts:
            raise ValueError("representation_parts must have shape [D, num_parts]")
        if coefficients.ndim != 2 or coefficients.shape[1] != self.num_parts:
            raise ValueError("representation_coefficients must have shape [N, num_parts]")
        if source_indices.shape != (coefficients.shape[0],):
            raise ValueError("representation_indices must align with coefficients")
        if np.unique(source_indices).size != source_indices.size:
            raise ValueError("representation_indices must be unique")
        if not np.isfinite(parts).all() or not np.isfinite(coefficients).all():
            raise ValueError("shared representation must be finite")
        if (coefficients < -1e-10).any() or not np.allclose(
            coefficients.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8
        ):
            raise ValueError("shared representation coefficients must be probabilities")

        order = np.argsort(source_indices, kind="stable")
        sorted_indices = source_indices[order]
        positions = np.searchsorted(sorted_indices, features.global_indices)
        valid = positions < sorted_indices.size
        if not np.all(valid) or not np.array_equal(
            sorted_indices[positions[valid]], features.global_indices[valid]
        ):
            raise KeyError("shared representation does not cover every snapshot index")
        split_coefficients = coefficients[order][positions]
        percentages = self._anchor_percentages()
        anchor_positions = select_pdl_anchor_candidates(
            posteriors.noisy_probabilities, percentages
        )
        anchor_indices = posteriors.global_indices[anchor_positions]
        matrices = (
            fit_pdl_basis_matrices(
                split_coefficients[anchor_positions],
                posteriors.noisy_probabilities[anchor_positions],
                epochs=self.basis_epochs,
                learning_rate=self.basis_learning_rate,
                loss_threshold=self.basis_loss_threshold,
                seed=self.representation_seed,
            )
            if part_matrices is None
            else np.asarray(part_matrices, dtype=np.float64)
        )
        return PartTransitionArtifact(
            parts,
            split_coefficients,
            matrices,
            features.global_indices,
            features.snapshot_hash,
            posteriors.snapshot_hash,
            anchor_indices,
            metadata={
                "estimator": "part_dependent",
                "num_parts": self.num_parts,
                "anchor_percentages": list(percentages),
                "representation_seed": self.representation_seed,
                "representation_iterations": self.representation_iterations,
                "representation_error_tolerance": self.representation_error_tolerance,
                "basis_epochs": self.basis_epochs,
                "basis_learning_rate": self.basis_learning_rate,
                "basis_loss_threshold": self.basis_loss_threshold,
                "paper_workflow": "PDL official main.py",
                "shared_representation": True,
                "basis_optimizer_state_reused": part_matrices is not None,
                "official_raw_basis": bool(official_raw_basis),
            },
        )

    def estimate(self, features: FeatureSnapshot, posteriors: PosteriorSnapshot) -> PartTransitionArtifact:
        parts, coefficients = fit_part_representation(
            features.features,
            self.num_parts,
            seed=self.representation_seed,
            iterations=self.representation_iterations,
            error_tolerance=self.representation_error_tolerance,
        )
        return self.estimate_from_shared_representation(
            features,
            posteriors,
            representation_parts=parts,
            representation_coefficients=coefficients,
            representation_indices=features.global_indices,
        )
