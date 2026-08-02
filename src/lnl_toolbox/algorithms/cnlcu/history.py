from __future__ import annotations

"""Fixed-window, stable-index state for one CNLCU peer."""

import hashlib
from typing import Any, Mapping

import torch
from torch import Tensor


def sample_index_mapping_hash(indices: Tensor) -> str:
    canonical = torch.as_tensor(indices).detach().cpu().to(torch.int64).contiguous()
    return hashlib.sha256(canonical.numpy().tobytes()).hexdigest()


class PeerLossHistory:
    def __init__(
        self,
        canonical_global_indices: Tensor,
        window_size: int,
        peer_identity: str,
    ) -> None:
        indices = torch.as_tensor(canonical_global_indices).detach().cpu()
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("CNLCU canonical indices must be a non-empty vector")
        if indices.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise TypeError("CNLCU canonical indices must use an integer dtype")
        indices = indices.to(torch.int64)
        if bool((indices < 0).any()) or torch.unique(indices).numel() != indices.numel():
            raise ValueError("CNLCU canonical indices must be unique and non-negative")
        if int(window_size) <= 0:
            raise ValueError("CNLCU window_size must be positive")
        if peer_identity not in {"a", "b"}:
            raise ValueError("CNLCU history peer identity must be 'a' or 'b'")
        self.canonical_global_indices = torch.sort(indices).values
        self.mapping_hash = sample_index_mapping_hash(self.canonical_global_indices)
        self.window_size = int(window_size)
        self.peer_identity = peer_identity
        count = self.canonical_global_indices.numel()
        self.values = torch.zeros(count, self.window_size, dtype=torch.float32)
        self.observed = torch.zeros(count, self.window_size, dtype=torch.bool)
        self.selected_count = torch.zeros(count, dtype=torch.int64)
        self.window_start_epoch = -1
        self.window_epoch_count = 0
        self.active_epoch = -1

    def prepare_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("CNLCU epoch must be non-negative")
        window_start = epoch - epoch % self.window_size
        if self.window_start_epoch != window_start:
            if self.window_start_epoch > window_start:
                raise ValueError("CNLCU history cannot move backwards across windows")
            self.values.zero_()
            self.observed.zero_()
            self.window_start_epoch = window_start
            self.window_epoch_count = 0
        self.active_epoch = epoch
        self.window_epoch_count = max(self.window_epoch_count, epoch - window_start + 1)

    def resolve(self, indices: Tensor) -> Tensor:
        values = torch.as_tensor(indices).detach().cpu()
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError("CNLCU batch indices must be a non-empty vector")
        if values.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise TypeError("CNLCU batch indices must use an integer dtype")
        values = values.to(torch.int64)
        if torch.unique(values).numel() != values.numel():
            raise ValueError("CNLCU batch indices must be unique")
        positions = torch.searchsorted(self.canonical_global_indices, values)
        valid = positions < self.canonical_global_indices.numel()
        if bool(valid.any()):
            valid = valid & (self.canonical_global_indices[positions.clamp_max(
                self.canonical_global_indices.numel() - 1
            )] == values)
        if not bool(valid.all()):
            missing = values[~valid].tolist()
            raise KeyError(f"CNLCU batch indices are absent from the training mapping: {missing}")
        return positions

    def append(self, indices: Tensor, losses: Tensor) -> Tensor:
        if self.active_epoch < 0:
            raise RuntimeError("CNLCU history epoch was not prepared")
        rows = self.resolve(indices)
        detached = torch.as_tensor(losses).detach().cpu()
        if detached.shape != rows.shape or not detached.is_floating_point():
            raise ValueError("CNLCU appended losses must align with indices as floating [B]")
        detached = detached.to(torch.float32)
        if not bool(torch.isfinite(detached).all()) or bool((detached < 0).any()):
            raise ValueError("CNLCU appended losses must be finite and non-negative")
        slot = self.active_epoch - self.window_start_epoch
        if bool(self.observed[rows, slot].any()):
            raise ValueError("CNLCU sample was observed twice in one epoch")
        self.values[rows, slot] = detached
        self.observed[rows, slot] = True
        return rows

    def lookup_rows(self, rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        stop = self.active_epoch - self.window_start_epoch + 1
        return (
            self.values[rows, :stop].clone(),
            self.observed[rows, :stop].clone(),
            self.selected_count[rows].clone(),
        )

    def increment_selected(self, rows: Tensor, selected_mask: Tensor) -> None:
        mask = torch.as_tensor(selected_mask).detach().cpu()
        if mask.shape != rows.shape or mask.dtype != torch.bool:
            raise ValueError("CNLCU selected mask must align with history rows")
        self.selected_count[rows[mask]] += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "peer_identity": self.peer_identity,
            "canonical_global_indices": self.canonical_global_indices.clone(),
            "sample_index_mapping_hash": self.mapping_hash,
            "window_size": self.window_size,
            "values": self.values.clone(),
            "observed": self.observed.clone(),
            "selected_count": self.selected_count.clone(),
            "window_start_epoch": self.window_start_epoch,
            "window_epoch_count": self.window_epoch_count,
            "active_epoch": self.active_epoch,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("CNLCU history state must be a mapping")
        required = set(self.state_dict())
        if set(state) != required:
            raise ValueError("CNLCU history state keys do not match")
        if state["peer_identity"] != self.peer_identity:
            raise ValueError("CNLCU peer history identity changed")
        indices = torch.as_tensor(state["canonical_global_indices"])
        if indices.dtype != torch.int64 or not torch.equal(indices.cpu(), self.canonical_global_indices):
            raise ValueError("CNLCU sample mapping changed")
        if state["sample_index_mapping_hash"] != self.mapping_hash:
            raise ValueError("CNLCU sample-index mapping hash changed")
        if int(state["window_size"]) != self.window_size:
            raise ValueError("CNLCU history window size changed")
        values = torch.as_tensor(state["values"])
        observed = torch.as_tensor(state["observed"])
        counts = torch.as_tensor(state["selected_count"])
        if values.shape != self.values.shape or values.dtype != torch.float32 or not bool(torch.isfinite(values).all()):
            raise ValueError("CNLCU history values are invalid")
        if observed.shape != self.observed.shape or observed.dtype != torch.bool:
            raise ValueError("CNLCU observed history is invalid")
        if counts.shape != self.selected_count.shape or counts.dtype != torch.int64 or bool((counts < 0).any()):
            raise ValueError("CNLCU selected counts are invalid")
        start = int(state["window_start_epoch"])
        length = int(state["window_epoch_count"])
        active = int(state["active_epoch"])
        cursor_is_empty = start == -1 and length == 0 and active == -1
        cursor_is_active = (
            start >= 0
            and start % self.window_size == 0
            and 1 <= length <= self.window_size
            and start <= active < start + length
        )
        if not (cursor_is_empty or cursor_is_active):
            raise ValueError("CNLCU history cursor is invalid")
        self.values.copy_(values)
        self.observed.copy_(observed)
        self.selected_count.copy_(counts)
        self.window_start_epoch = start
        self.window_epoch_count = length
        self.active_epoch = active


__all__ = ["PeerLossHistory", "sample_index_mapping_hash"]
