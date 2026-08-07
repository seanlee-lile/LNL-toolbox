from __future__ import annotations

"""Stable-index lookup over a persisted noisy-posterior snapshot."""

import numpy as np
import torch
from torch import Tensor

from lnl_toolbox.noise.estimators import PosteriorSnapshot


class ObservedNoisyProbabilityLookup:
    """Gather scalar ``psi_i=P(noisy target_i|x_i)`` from a full snapshot."""

    def __init__(self, snapshot: PosteriorSnapshot, device: torch.device | str) -> None:
        order = np.argsort(snapshot.global_indices, kind="stable")
        self.indices = torch.as_tensor(
            snapshot.global_indices[order], dtype=torch.int64, device=device
        )
        self.targets = torch.as_tensor(
            snapshot.noisy_targets[order], dtype=torch.int64, device=device
        )
        self.probabilities = torch.as_tensor(
            snapshot.noisy_probabilities[order], dtype=torch.float64, device=device
        )
        self.snapshot_hash = snapshot.snapshot_hash

    def resolve(
        self,
        sample_indices: Tensor,
        noisy_targets: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        indices = torch.as_tensor(sample_indices, device=self.indices.device)
        targets = torch.as_tensor(noisy_targets, device=self.indices.device)
        if indices.ndim != 1 or targets.shape != indices.shape:
            raise ValueError("UPM psi lookup requires aligned index/target [B]")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("UPM psi lookup indices must be unique")
        positions = torch.searchsorted(self.indices, indices.to(torch.int64))
        bounded = positions.clamp_max(self.indices.numel() - 1)
        valid = (positions < self.indices.numel()) & (self.indices[bounded] == indices)
        if not bool(valid.all()):
            raise KeyError("UPM psi snapshot is missing requested stable indices")
        if not torch.equal(self.targets[positions], targets.to(torch.int64)):
            raise ValueError("UPM psi snapshot targets do not align with the batch")
        psi = self.probabilities[positions].gather(
            1, targets.to(torch.int64)[:, None]
        ).squeeze(1)
        result = psi.to(device=sample_indices.device, dtype=dtype).detach()
        if not bool(torch.isfinite(result).all()) or bool(((result < 0) | (result > 1)).any()):
            raise ValueError("UPM observed-label probabilities are invalid")
        return result


__all__ = ["ObservedNoisyProbabilityLookup"]
