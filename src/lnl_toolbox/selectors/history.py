from __future__ import annotations

"""Stable global-index state stores for scalar and tensor histories."""

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


class IndexedTensorHistory:
    """Dense bounded tensor history addressed by stable global sample index."""

    def __init__(
        self,
        capacity: int,
        horizon: int,
        width: int,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if min(int(capacity), int(horizon), int(width)) <= 0:
            raise ValueError("history capacity, horizon, and width must be positive")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("tensor history dtype must be floating point")
        self.capacity = int(capacity)
        self.horizon = int(horizon)
        self.width = int(width)
        self.dtype = dtype
        self.values = torch.zeros(
            self.capacity, self.horizon, self.width, dtype=dtype
        )
        self.observed = torch.zeros(
            self.capacity, self.horizon, dtype=torch.bool
        )
        self.completed_steps = 0

    def _indices(self, indices: Tensor) -> Tensor:
        values = torch.as_tensor(indices, dtype=torch.long).detach().cpu()
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError("history indices must be a non-empty vector")
        if int(values.min()) < 0 or int(values.max()) >= self.capacity:
            raise IndexError("history index is outside configured capacity")
        if torch.unique(values).numel() != values.numel():
            raise ValueError("history indices must be unique")
        return values

    def previous(self, indices: Tensor, step: int) -> Tensor:
        resolved = self._indices(indices)
        step = int(step)
        if not 0 <= step < self.horizon:
            raise IndexError("history step is outside configured horizon")
        return self.values[resolved, :step]

    def update(self, indices: Tensor, step: int, values: Tensor) -> None:
        resolved = self._indices(indices)
        step = int(step)
        if not 0 <= step < self.horizon:
            raise IndexError("history step is outside configured horizon")
        detached = torch.as_tensor(values).detach().cpu().to(self.dtype)
        if detached.shape != (resolved.numel(), self.width):
            raise ValueError(
                "history values must have shape [batch_size, width]"
            )
        if not bool(torch.isfinite(detached).all()):
            raise ValueError("history values must be finite")
        if bool(self.observed[resolved, step].any()):
            raise ValueError("history sample was observed twice in one step")
        self.values[resolved, step] = detached
        self.observed[resolved, step] = True
        self.completed_steps = max(self.completed_steps, step + 1)

    def state_dict(self) -> dict[str, Any]:
        stop = self.completed_steps
        return {
            "capacity": self.capacity,
            "horizon": self.horizon,
            "width": self.width,
            "dtype": str(self.dtype),
            "completed_steps": stop,
            "values": self.values[:, :stop].clone(),
            "observed": self.observed[:, :stop].clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = (self.capacity, self.horizon, self.width, str(self.dtype))
        actual = (
            int(state.get("capacity", -1)),
            int(state.get("horizon", -1)),
            int(state.get("width", -1)),
            str(state.get("dtype", "")),
        )
        if actual != expected:
            raise ValueError("tensor history configuration mismatch")
        completed = int(state.get("completed_steps", -1))
        if not 0 <= completed <= self.horizon:
            raise ValueError("tensor history completed_steps is invalid")
        values = torch.as_tensor(state.get("values"))
        observed = torch.as_tensor(state.get("observed"))
        if values.shape != (self.capacity, completed, self.width):
            raise ValueError("tensor history values shape mismatch")
        if observed.shape != (self.capacity, completed):
            raise ValueError("tensor history observed shape mismatch")
        self.values.zero_()
        self.observed.zero_()
        self.values[:, :completed] = values.to(self.dtype)
        self.observed[:, :completed] = observed.to(torch.bool)
        self.completed_steps = completed
