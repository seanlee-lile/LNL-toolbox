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
            raise ValueError(
                "history capacity, horizon, and width must be positive"
            )
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
        if not torch.is_tensor(indices) or indices.ndim != 1:
            raise ValueError("history indices must be a non-empty vector")
        if indices.dtype not in {
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        }:
            raise ValueError("history indices must use an integer dtype")
        values = indices.detach().cpu().to(torch.long)
        if values.numel() == 0:
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
        detached = torch.as_tensor(values).detach().cpu()
        if detached.shape != (resolved.numel(), self.width):
            raise ValueError(
                "history values must have shape [batch_size, width]"
            )
        if not detached.is_floating_point():
            raise ValueError("history values must use a floating-point dtype")
        detached = detached.to(self.dtype)
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
        if values.dtype != self.dtype or not bool(torch.isfinite(values).all()):
            raise ValueError("tensor history values dtype or values are invalid")
        if observed.shape != (self.capacity, completed):
            raise ValueError("tensor history observed shape mismatch")
        if observed.dtype != torch.bool:
            raise ValueError("tensor history observed must use torch.bool")
        self.values.zero_()
        self.observed.zero_()
        self.values[:, :completed] = values
        self.observed[:, :completed] = observed
        self.completed_steps = completed


class IndexedSoftLabelState:
    """Global-indexed soft labels with an explicit momentum update.

    This is the persistent ``Z[N,C]`` state used by feature-diffusion methods
    such as LEND.  It intentionally stores no clean labels.
    """

    def __init__(self, capacity: int, width: int, *, dtype: torch.dtype = torch.float32) -> None:
        if int(capacity) <= 0 or int(width) <= 1:
            raise ValueError("soft-label capacity and width are invalid")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("soft-label state dtype must be floating point")
        self.capacity = int(capacity)
        self.width = int(width)
        self.dtype = dtype
        self.values = torch.full((self.capacity, self.width), 1.0 / self.width, dtype=dtype)
        self.observed = torch.zeros(self.capacity, dtype=torch.bool)

    def _indices(self, indices: Tensor) -> Tensor:
        resolved = torch.as_tensor(indices, dtype=torch.long).detach().cpu()
        if resolved.ndim != 1 or resolved.numel() == 0:
            raise ValueError("soft-label indices must be a non-empty vector")
        if int(resolved.min()) < 0 or int(resolved.max()) >= self.capacity:
            raise IndexError("soft-label index is outside configured capacity")
        if torch.unique(resolved).numel() != resolved.numel():
            raise ValueError("soft-label indices must be unique")
        return resolved

    def lookup(self, indices: Tensor) -> Tensor:
        return self.values[self._indices(indices)].to(device=indices.device)

    def update(self, indices: Tensor, values: Tensor, *, momentum: float) -> None:
        resolved = self._indices(indices)
        incoming = torch.as_tensor(values).detach().cpu().to(self.dtype)
        if incoming.shape != (resolved.numel(), self.width):
            raise ValueError("soft-label values must have shape [B, C]")
        if not 0.0 <= float(momentum) <= 1.0 or not bool(torch.isfinite(incoming).all()):
            raise ValueError("soft-label momentum or values are invalid")
        incoming = incoming / incoming.sum(dim=1, keepdim=True).clamp_min(1e-8)
        old = self.values[resolved]
        self.values[resolved] = float(momentum) * old + (1.0 - float(momentum)) * incoming
        self.values[resolved] /= self.values[resolved].sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.observed[resolved] = True

    def state_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "width": self.width, "dtype": str(self.dtype), "values": self.values.clone(), "observed": self.observed.clone()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (int(state.get("capacity", -1)), int(state.get("width", -1)), str(state.get("dtype", ""))) != (self.capacity, self.width, str(self.dtype)):
            raise ValueError("soft-label state configuration mismatch")
        values = torch.as_tensor(state.get("values"))
        observed = torch.as_tensor(state.get("observed"))
        if values.shape != self.values.shape or values.dtype != self.dtype or observed.shape != self.observed.shape or observed.dtype != torch.bool:
            raise ValueError("soft-label state shape or dtype mismatch")
        if not bool(torch.isfinite(values).all()) or bool((values < 0.0).any()):
            raise ValueError("soft-label state contains invalid values")
        self.values.copy_(values)
        self.observed.copy_(observed)
