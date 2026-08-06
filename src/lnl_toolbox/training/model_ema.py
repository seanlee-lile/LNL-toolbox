from __future__ import annotations

"""Reusable exponential moving average for teacher-style training."""

from copy import deepcopy
from typing import Mapping

import torch
from torch import nn


class ModelEMA:
    """Maintain a non-trainable EMA copy, including model buffers."""

    def __init__(
        self,
        model: nn.Module,
        momentum: float = 0.999,
        *,
        update_buffers: bool = True,
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("EMA momentum must be in [0, 1)")
        self.momentum = float(momentum)
        self.update_buffers = bool(update_buffers)
        self._parameter_names = tuple(name for name, _ in model.named_parameters())
        self.model = deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        target = self.model.state_dict()
        if source.keys() != target.keys():
            raise ValueError("EMA source and target state do not match")
        names = self._parameter_names if not self.update_buffers else tuple(target.keys())
        for name in names:
            value = target[name]
            incoming = source[name].detach().to(device=value.device)
            if torch.is_floating_point(value):
                value.mul_(self.momentum).add_(incoming, alpha=1.0 - self.momentum)
            else:
                value.copy_(incoming)

    def state_dict(self) -> dict[str, object]:
        return {
            "momentum": self.momentum,
            "update_buffers": self.update_buffers,
            "model": self.model.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if float(state.get("momentum", self.momentum)) != self.momentum:
            raise ValueError("EMA momentum mismatch")
        if bool(state.get("update_buffers", self.update_buffers)) != self.update_buffers:
            raise ValueError("EMA buffer-update configuration mismatch")
        model_state = state.get("model")
        if not isinstance(model_state, Mapping):
            raise ValueError("EMA checkpoint is missing model state")
        self.model.load_state_dict(dict(model_state))


__all__ = ["ModelEMA"]
