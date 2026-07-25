from __future__ import annotations

"""Deterministic model snapshots consumed by offline noise estimators."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from lnl_toolbox.noise.estimators import PosteriorSnapshot


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
