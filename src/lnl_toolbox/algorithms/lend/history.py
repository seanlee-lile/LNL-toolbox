from __future__ import annotations

"""Stable-index persistent diluted-label history for LEND Eq. (5)."""

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class HistoryProposal:
    rows: Tensor
    values: Tensor
    epoch: int


class LENDLabelHistory:
    def __init__(self, canonical_sample_indices: Tensor, num_classes: int) -> None:
        indices = torch.as_tensor(canonical_sample_indices).detach().cpu()
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("LEND canonical indices must be a non-empty vector")
        if indices.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise ValueError("LEND canonical indices must use integer dtype")
        indices = indices.to(torch.int64)
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("LEND canonical indices must be unique")
        if num_classes < 2:
            raise ValueError("LEND requires at least two classes")
        self.canonical_sample_indices = torch.sort(indices).values
        self.values = torch.zeros((indices.numel(), num_classes), dtype=torch.float32)
        self.initialized = torch.zeros(indices.numel(), dtype=torch.bool)
        self.last_updated_epoch = torch.full((indices.numel(),), -1, dtype=torch.int64)

    @property
    def num_classes(self) -> int:
        return int(self.values.shape[1])

    def _rows(self, indices: Tensor) -> Tensor:
        query = torch.as_tensor(indices).detach().cpu()
        if query.ndim != 1 or query.numel() == 0:
            raise ValueError("LEND history query indices must be non-empty [B]")
        if query.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise ValueError("LEND history query indices must use integer dtype")
        query = query.to(torch.int64)
        if torch.unique(query).numel() != query.numel():
            raise ValueError("LEND history query indices must be unique")
        rows = torch.searchsorted(self.canonical_sample_indices, query)
        valid = rows < self.canonical_sample_indices.numel()
        matched = torch.zeros_like(valid)
        matched[valid] = self.canonical_sample_indices[rows[valid]] == query[valid]
        if not bool(matched.all()):
            missing = query[~matched].tolist()
            raise ValueError(f"LEND history is missing stable indices: {missing}")
        return rows

    def propose(self, indices: Tensor, current: Tensor, *, epoch: int,
                beta: float) -> HistoryProposal:
        if epoch < 0:
            raise ValueError("LEND history epoch must be non-negative")
        if not isinstance(current, Tensor) or current.ndim != 2 or current.shape[1] != self.num_classes:
            raise ValueError("LEND current diluted labels must have shape [B,C]")
        if current.requires_grad or not torch.is_floating_point(current) or not bool(torch.isfinite(current).all()):
            raise ValueError("LEND current diluted labels must be detached finite floats")
        if not isinstance(beta, (int, float)) or not 0.0 <= float(beta) <= 1.0:
            raise ValueError("LEND history beta must be in [0,1]")
        rows = self._rows(indices)
        if rows.numel() != current.shape[0]:
            raise ValueError("LEND history indices and values must align")
        if bool((self.last_updated_epoch[rows] == epoch).any()):
            raise ValueError("LEND sample history cannot update twice in one epoch")
        current_cpu = current.detach().to(device="cpu", dtype=torch.float32)
        previous = self.values[rows]
        blended = torch.where(
            self.initialized[rows, None],
            (1.0 - beta) * current_cpu + beta * previous,
            current_cpu,
        )
        return HistoryProposal(rows=rows, values=blended, epoch=epoch)

    def commit(self, proposal: HistoryProposal) -> None:
        if not isinstance(proposal, HistoryProposal):
            raise TypeError("LEND history commit requires a HistoryProposal")
        if bool((self.last_updated_epoch[proposal.rows] == proposal.epoch).any()):
            raise ValueError("LEND history proposal was already committed")
        self.values[proposal.rows] = proposal.values
        self.initialized[proposal.rows] = True
        self.last_updated_epoch[proposal.rows] = proposal.epoch

    def state_dict(self) -> dict[str, Any]:
        return {
            "canonical_sample_indices": self.canonical_sample_indices.clone(),
            "values": self.values.clone(),
            "initialized": self.initialized.clone(),
            "last_updated_epoch": self.last_updated_epoch.clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != set(self.state_dict()):
            raise ValueError("LEND history state keys do not match")
        raw_indices = torch.as_tensor(state["canonical_sample_indices"])
        raw_values = torch.as_tensor(state["values"])
        raw_initialized = torch.as_tensor(state["initialized"])
        raw_last = torch.as_tensor(state["last_updated_epoch"])
        if raw_indices.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise ValueError("LEND history state indices must use integer dtype")
        if raw_values.dtype != torch.float32:
            raise ValueError("LEND history state values must use float32")
        if raw_initialized.dtype != torch.bool or raw_last.dtype != torch.int64:
            raise ValueError("LEND history state mask or epoch dtype is invalid")
        indices = raw_indices.detach().cpu().to(torch.int64)
        values = raw_values.detach().cpu()
        initialized = raw_initialized.detach().cpu()
        last = raw_last.detach().cpu()
        if not torch.equal(indices, self.canonical_sample_indices):
            raise ValueError("LEND history stable-index mapping changed")
        if values.shape != self.values.shape or values.dtype != torch.float32 or not bool(torch.isfinite(values).all()):
            raise ValueError("LEND history values are invalid")
        if initialized.shape != self.initialized.shape or initialized.dtype != torch.bool:
            raise ValueError("LEND history initialized mask is invalid")
        if last.shape != self.last_updated_epoch.shape or last.dtype != torch.int64:
            raise ValueError("LEND history epoch state is invalid")
        if bool((last < -1).any()) or bool((initialized & (last < 0)).any()) or bool((~initialized & (last != -1)).any()):
            raise ValueError("LEND history initialized and epoch state disagree")
        self.values.copy_(values)
        self.initialized.copy_(initialized)
        self.last_updated_epoch.copy_(last)


__all__ = ["HistoryProposal", "LENDLabelHistory"]
