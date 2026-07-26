from __future__ import annotations

"""Stable global-index state stores for loss and selection history."""

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor


@dataclass
class IndexedHistory:
    default: float = 0.0
    values: dict[int, float] = field(default_factory=dict)

    def update(self, indices: Tensor, values: Tensor) -> None:
        if indices.ndim != 1 or values.ndim != 1 or indices.shape != values.shape:
            raise ValueError("history indices and values must be aligned shape [B]")
        if not torch.isfinite(values).all():
            raise ValueError("history values must be finite")
        for index, value in zip(indices.detach().cpu().tolist(), values.detach().cpu().tolist()):
            self.values[int(index)] = float(value)

    def lookup(self, indices: Tensor) -> Tensor:
        if indices.ndim != 1:
            raise ValueError("history lookup indices must have shape [B]")
        return torch.tensor(
            [self.values.get(int(index), self.default) for index in indices.detach().cpu().tolist()],
            dtype=torch.float32,
            device=indices.device,
        )

    def state_dict(self) -> dict[str, Any]:
        return {"default": self.default, "values": dict(self.values)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or "default" not in state or "values" not in state:
            raise ValueError("history state is incomplete")
        self.default = float(state["default"])
        self.values = {int(key): float(value) for key, value in dict(state["values"]).items()}


class LossHistory(IndexedHistory):
    """Per-sample loss history keyed by stable global index."""


class SelectionHistory(IndexedHistory):
    """Per-sample selection/participation history keyed by global index."""

