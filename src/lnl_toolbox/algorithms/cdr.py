from __future__ import annotations

"""Critical-parameter update rule from Xia et al., ICLR 2021."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn

from lnl_toolbox.algorithms.update_policy import (
    ParameterUpdateInput,
    ParameterUpdateResult,
)
from lnl_toolbox.core import ExperimentContext


@dataclass(frozen=True, slots=True)
class CriticalParameterMasks:
    """Deterministic per-parameter masks and their global scalar counts."""

    masks: Mapping[str, torch.Tensor]
    eligible_parameters: int
    critical_parameters: int


def _validate_noise_rate(noise_rate: float) -> float:
    value = float(noise_rate)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("CDR noise_rate must satisfy 0 <= noise_rate < 1")
    return value


def _eligible_named_parameters(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    critical_scope: str,
) -> list[tuple[str, nn.Parameter]]:
    if critical_scope != "all_trainable":
        raise ValueError("Paper-mode CDR only supports critical_scope='all_trainable'")
    values = sorted(
        (
            (str(name), parameter)
            for name, parameter in named_parameters
            if parameter.requires_grad and parameter.grad is not None
        ),
        key=lambda item: item[0],
    )
    if not values:
        raise ValueError("CDR requires at least one trainable parameter with a gradient")
    names = [name for name, _ in values]
    if len(names) != len(set(names)):
        raise ValueError("CDR parameter names must be unique")
    devices = {parameter.device for _, parameter in values}
    if len(devices) != 1:
        raise ValueError("CDR requires all eligible parameters on one device")
    for name, parameter in values:
        gradient = parameter.grad
        assert gradient is not None
        if gradient.is_sparse:
            raise ValueError(f"CDR does not support sparse gradients: {name}")
        if not torch.isfinite(parameter.detach()).all():
            raise ValueError(f"CDR parameter must be finite: {name}")
        if not torch.isfinite(gradient.detach()).all():
            raise ValueError(f"CDR gradient must be finite: {name}")
    return values


def critical_parameter_masks(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    noise_rate: float,
    *,
    critical_scope: str = "all_trainable",
) -> CriticalParameterMasks:
    """Select the exact top ``ceil((1-tau) * m)`` scalar criticalities.

    Scores follow Eq. (3), ``abs(gradient * parameter)``. Parameters are
    concatenated by name and flat offset so stable sorting gives deterministic
    tie-breaking independent of module registration order.
    """

    tau = _validate_noise_rate(noise_rate)
    values = _eligible_named_parameters(named_parameters, critical_scope)
    flattened = [
        (parameter.grad.detach() * parameter.detach()).abs().reshape(-1).to(torch.float64)
        for _, parameter in values
    ]
    scores = torch.cat(flattened)
    count = int(scores.numel())
    critical_count = int(math.ceil((1.0 - tau) * count))
    order = torch.argsort(scores, descending=True, stable=True)
    selected = torch.zeros(count, dtype=torch.bool, device=scores.device)
    selected[order[:critical_count]] = True

    masks: dict[str, torch.Tensor] = {}
    offset = 0
    for name, parameter in values:
        size = int(parameter.numel())
        masks[name] = selected[offset:offset + size].reshape(parameter.shape)
        offset += size
    return CriticalParameterMasks(masks, count, critical_count)


class CDRUpdatePolicy:
    """Paper-mode CDR update policy implementing Eq. (3)-(6)."""

    name = "cdr"

    def __init__(
        self,
        noise_rate: float,
        l1_decay: float,
        critical_scope: str = "all_trainable",
    ) -> None:
        self.noise_rate = _validate_noise_rate(noise_rate)
        self.l1_decay = float(l1_decay)
        if not math.isfinite(self.l1_decay) or self.l1_decay < 0.0:
            raise ValueError("CDR l1_decay must be finite and non-negative")
        if critical_scope != "all_trainable":
            raise ValueError(
                "Paper-mode CDR only supports critical_scope='all_trainable'"
            )
        self.critical_scope = critical_scope

    def setup(self, context: ExperimentContext) -> None:
        del context

    def _validate_optimizer(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if not isinstance(optimizer, torch.optim.SGD):
            raise TypeError("Paper-mode CDR requires torch.optim.SGD")
        for group in optimizer.param_groups:
            if float(group.get("weight_decay", 0.0)) != 0.0:
                raise ValueError(
                    "Paper-mode CDR requires optimizer weight_decay=0; "
                    "use l1_decay for Eq. (5)-(6)"
                )
        model_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        optimizer_id_list = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        optimizer_ids = set(optimizer_id_list)
        if model_ids != optimizer_ids or len(optimizer_id_list) != len(optimizer_ids):
            raise ValueError(
                "CDR requires the optimizer to contain every trainable model parameter "
                "exactly once"
            )

    def update(self, request: ParameterUpdateInput) -> ParameterUpdateResult:
        self._validate_optimizer(request.model, request.optimizer)
        request.optimizer.zero_grad(set_to_none=True)
        request.objective.backward()
        named_parameters = list(request.model.named_parameters())
        selected = critical_parameter_masks(
            named_parameters,
            self.noise_rate,
            critical_scope=self.critical_scope,
        )
        gradient_scale = 1.0 - self.noise_rate
        with torch.no_grad():
            for name, parameter in named_parameters:
                gradient = parameter.grad
                if not parameter.requires_grad or gradient is None:
                    continue
                mask = selected.masks[name]
                gradient.mul_(mask.to(dtype=gradient.dtype))
                gradient.mul_(gradient_scale)
                gradient.add_(parameter.sign(), alpha=self.l1_decay)
        request.optimizer.step()
        return ParameterUpdateResult({
            "update_eligible_parameters": float(selected.eligible_parameters),
            "update_critical_parameters": float(selected.critical_parameters),
            "update_critical_ratio": (
                selected.critical_parameters / selected.eligible_parameters
            ),
        })

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError("CDR update policy has no private state")
