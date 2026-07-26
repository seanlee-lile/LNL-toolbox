from __future__ import annotations

"""Deterministic model snapshots consumed by offline noise estimators."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from lnl_toolbox.noise.estimators import PosteriorSnapshot


@dataclass(frozen=True)
class FeatureSnapshot:
    """Feature vectors and noisy targets aligned by stable global index."""

    features: np.ndarray
    noisy_targets: np.ndarray
    global_indices: np.ndarray
    dataset: str
    split: str

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        targets = np.asarray(self.noisy_targets, dtype=np.int64)
        indices = np.asarray(self.global_indices, dtype=np.int64)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must have shape [N, D]")
        if targets.shape != (features.shape[0],) or indices.shape != (features.shape[0],):
            raise ValueError("feature snapshot arrays must align by sample")
        if not np.isfinite(features).all() or np.unique(indices).size != len(indices):
            raise ValueError("feature snapshot contains invalid values or duplicate indices")
        order = np.argsort(indices, kind="stable")
        object.__setattr__(self, "features", features[order].copy())
        object.__setattr__(self, "noisy_targets", targets[order].copy())
        object.__setattr__(self, "global_indices", indices[order].copy())
        object.__setattr__(self, "dataset", str(self.dataset).strip())
        object.__setattr__(self, "split", str(self.split).strip())

    @property
    def snapshot_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps({"dataset": self.dataset, "split": self.split}, sort_keys=True).encode())
        for value in (self.features, self.noisy_targets, self.global_indices):
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        metadata = {"dataset": self.dataset, "split": self.split, "snapshot_hash": self.snapshot_hash}
        np.savez_compressed(path, features=self.features, noisy_targets=self.noisy_targets, global_indices=self.global_indices, metadata_json=np.array(json.dumps(metadata, sort_keys=True)))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSnapshot":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            snapshot = cls(data["features"], data["noisy_targets"], data["global_indices"], metadata["dataset"], metadata["split"])
            if metadata.get("snapshot_hash") != snapshot.snapshot_hash:
                raise ValueError("feature snapshot hash does not match contents")
            return snapshot


def _batch_tensor(batch: Mapping[str, Any], name: str) -> Tensor:
    value = batch.get(name)
    if not torch.is_tensor(value):
        raise TypeError(f"snapshot batch field {name!r} must be a torch.Tensor")
    return value


def collect_posterior_snapshot(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device | str,
    *,
    dataset: str,
    split: str,
) -> PosteriorSnapshot:
    """Collect ``P(noisy label | x)`` and noisy targets by stable global index."""

    resolved_device = torch.device(device)
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                if not isinstance(batch, Mapping):
                    raise TypeError("snapshot loader must yield mapping batches")
                inputs = _batch_tensor(batch, "input").to(resolved_device)
                batch_targets = _batch_tensor(batch, "target")
                batch_indices = _batch_tensor(batch, "index")
                if batch_targets.ndim != 1 or batch_indices.ndim != 1:
                    raise ValueError(
                        "snapshot target and index fields must have shape [B]"
                    )
                if batch_targets.shape != batch_indices.shape:
                    raise ValueError(
                        "snapshot target and index fields must have matching shapes"
                    )

                logits = model(inputs)
                if not torch.is_tensor(logits) or logits.ndim != 2:
                    raise ValueError("snapshot model must return logits with shape [B, C]")
                if logits.shape[0] != batch_targets.shape[0]:
                    raise ValueError(
                        "snapshot logits, targets and indices must share batch size"
                    )
                posterior = torch.softmax(logits, dim=1)
                if not torch.isfinite(posterior).all():
                    raise ValueError("snapshot model produced non-finite probabilities")

                probabilities.append(
                    posterior.to(device="cpu", dtype=torch.float64).numpy()
                )
                targets.append(
                    batch_targets.to(device="cpu", dtype=torch.int64).numpy()
                )
                indices.append(
                    batch_indices.to(device="cpu", dtype=torch.int64).numpy()
                )
    finally:
        model.train(was_training)

    if not probabilities:
        raise ValueError("snapshot loader must contain at least one batch")
    all_probabilities = np.concatenate(probabilities, axis=0)
    all_targets = np.concatenate(targets, axis=0)
    all_indices = np.concatenate(indices, axis=0)
    order = np.argsort(all_indices, kind="stable")
    return PosteriorSnapshot(
        noisy_probabilities=all_probabilities[order],
        noisy_targets=all_targets[order],
        global_indices=all_indices[order],
        dataset=dataset,
        split=split,
    )


def _extract_features(output: Any) -> torch.Tensor:
    if isinstance(output, Mapping) and "features" in output:
        output = output["features"]
    elif isinstance(output, (tuple, list)) and output:
        output = output[0]
    if not torch.is_tensor(output) or output.ndim < 2:
        raise ValueError("feature extractor must return tensor with batch dimension")
    return output.flatten(start_dim=1)


def collect_feature_snapshot(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device | str,
    *,
    dataset: str,
    split: str,
    feature_extractor=None,
) -> FeatureSnapshot:
    """Collect feature vectors without reading clean labels."""

    resolved_device = torch.device(device)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = _batch_tensor(batch, "input").to(resolved_device)
                batch_targets = _batch_tensor(batch, "target")
                batch_indices = _batch_tensor(batch, "index")
                output = model(inputs) if feature_extractor is None else feature_extractor(model, inputs)
                batch_features = _extract_features(output)
                if batch_features.shape[0] != batch_targets.shape[0] or batch_targets.shape != batch_indices.shape:
                    raise ValueError("feature snapshot batch fields are misaligned")
                features.append(batch_features.cpu().numpy())
                targets.append(batch_targets.cpu().numpy())
                indices.append(batch_indices.cpu().numpy())
    finally:
        model.train(was_training)
    if not features:
        raise ValueError("feature snapshot loader must contain at least one batch")
    return FeatureSnapshot(
        np.concatenate(features),
        np.concatenate(targets),
        np.concatenate(indices),
        dataset,
        split,
    )


def pretrain_noisy_classifier(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device | str,
    *,
    epochs: int,
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Shared noisy-label warm-up using only observed targets."""

    if epochs < 0:
        raise ValueError("warm-up epochs must be non-negative")
    criterion = criterion or nn.CrossEntropyLoss()
    resolved_device = torch.device(device)
    model.to(resolved_device)
    criterion.to(resolved_device)
    was_training = model.training
    model.train()
    steps = 0
    try:
        for _ in range(epochs):
            for batch in loader:
                inputs = _batch_tensor(batch, "input").to(resolved_device)
                targets = _batch_tensor(batch, "target").to(resolved_device)
                loss = criterion(model(inputs), targets)
                if loss.ndim != 0:
                    loss = loss.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                steps += 1
    finally:
        model.train(was_training)
    return {"epochs": int(epochs), "steps": int(steps)}
