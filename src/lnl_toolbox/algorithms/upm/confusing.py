from __future__ import annotations

"""Stable-index storage for UPM per-sample confusing probabilities."""

import hashlib
from typing import Any, Mapping

import torch
from torch import Tensor

from .objective import update_confusing_probability


_INTEGER_DTYPES = {
    torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
}


def _mapping_hash(indices: Tensor) -> str:
    canonical = indices.detach().cpu().to(torch.int64).contiguous()
    return hashlib.sha256(canonical.numpy().tobytes()).hexdigest()


class ConfusingProbabilityState:
    """Own ``eta[N]`` and update counts using stable global sample identities."""

    def __init__(
        self,
        canonical_sample_indices: Tensor,
        initial_value: float,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        indices = torch.as_tensor(canonical_sample_indices)
        if indices.ndim != 1 or indices.numel() == 0 or indices.dtype not in _INTEGER_DTYPES:
            raise ValueError("UPM canonical sample indices must be non-empty integer [N]")
        indices = indices.detach().to(device=device, dtype=torch.int64)
        if bool((indices < 0).any()) or torch.unique(indices).numel() != indices.numel():
            raise ValueError("UPM canonical sample indices must be unique and non-negative")
        if not 0.0 <= float(initial_value) <= 1.0:
            raise ValueError("UPM eta initial value must be in [0, 1]")
        self.canonical_sample_indices = torch.sort(indices).values
        self.initial_value = float(initial_value)
        self.mapping_hash = _mapping_hash(self.canonical_sample_indices)
        self.eta = torch.full(
            (indices.numel(),), self.initial_value,
            dtype=torch.float64, device=indices.device,
        )
        self.update_count = torch.zeros(
            indices.numel(), dtype=torch.int64, device=indices.device
        )

    @property
    def device(self) -> torch.device:
        return self.eta.device

    def resolve_rows(self, sample_indices: Tensor) -> Tensor:
        values = torch.as_tensor(sample_indices)
        if values.ndim != 1 or values.numel() == 0 or values.dtype not in _INTEGER_DTYPES:
            raise ValueError("UPM batch sample indices must be non-empty integer [B]")
        values = values.detach().to(device=self.device, dtype=torch.int64)
        if torch.unique(values).numel() != values.numel():
            raise ValueError("UPM batch sample indices must be unique")
        positions = torch.searchsorted(self.canonical_sample_indices, values)
        bounded = positions.clamp_max(self.canonical_sample_indices.numel() - 1)
        valid = (positions < self.canonical_sample_indices.numel()) & (
            self.canonical_sample_indices[bounded] == values
        )
        if not bool(valid.all()):
            raise KeyError(
                f"UPM sample indices are absent from eta mapping: "
                f"{values[~valid].detach().cpu().tolist()}"
            )
        return positions

    def gather(self, sample_indices: Tensor, *, dtype: torch.dtype) -> Tensor:
        rows = self.resolve_rows(sample_indices)
        return self.eta[rows].to(dtype=dtype)

    def update(
        self,
        sample_indices: Tensor,
        q: Tensor,
        noisy_targets: Tensor,
        psi: Tensor,
        *,
        learning_rate: float,
        epsilon: float,
    ) -> Tensor:
        rows = self.resolve_rows(sample_indices)
        current = self.eta[rows].to(device=q.device, dtype=q.dtype)
        updated = update_confusing_probability(
            current, q, noisy_targets, psi,
            learning_rate=learning_rate, epsilon=epsilon,
        )
        with torch.no_grad():
            self.eta[rows] = updated.to(device=self.device, dtype=self.eta.dtype)
            self.update_count[rows] += 1
        return updated

    def state_dict(self) -> dict[str, Any]:
        return {
            "canonical_sample_indices": self.canonical_sample_indices.detach().cpu().clone(),
            "sample_index_mapping_hash": self.mapping_hash,
            "initial_value": self.initial_value,
            "eta": self.eta.detach().cpu().clone(),
            "update_count": self.update_count.detach().cpu().clone(),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("UPM eta state must be a mapping")
        if set(value) != set(self.state_dict()):
            raise ValueError("UPM eta state keys changed")
        indices = torch.as_tensor(value["canonical_sample_indices"])
        if indices.dtype != torch.int64 or not torch.equal(
            indices.cpu(), self.canonical_sample_indices.cpu()
        ):
            raise ValueError("UPM eta sample mapping changed")
        if value["sample_index_mapping_hash"] != self.mapping_hash:
            raise ValueError("UPM eta sample-index mapping hash changed")
        if float(value["initial_value"]) != self.initial_value:
            raise ValueError("UPM eta initial value changed")
        eta = torch.as_tensor(value["eta"])
        count = torch.as_tensor(value["update_count"])
        if eta.shape != self.eta.shape or eta.dtype != torch.float64:
            raise ValueError("UPM eta checkpoint tensor is invalid")
        if not bool(torch.isfinite(eta).all()) or bool(((eta < 0) | (eta > 1)).any()):
            raise ValueError("UPM eta checkpoint values are invalid")
        if count.shape != self.update_count.shape or count.dtype != torch.int64 or bool((count < 0).any()):
            raise ValueError("UPM eta update counts are invalid")
        self.eta.copy_(eta.to(self.device))
        self.update_count.copy_(count.to(self.device))


__all__ = ["ConfusingProbabilityState"]
