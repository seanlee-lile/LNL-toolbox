from __future__ import annotations

"""Shared contract for the parameter-update stage of a training step."""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Protocol

import torch
from torch import nn

from lnl_toolbox.core import ExperimentContext, RunState


@dataclass(frozen=True, slots=True)
class ParameterUpdateInput:
    """Inputs owned by the update stage after an objective has been constructed."""

    objective: torch.Tensor
    model: nn.Module
    optimizer: torch.optim.Optimizer
    run_state: RunState

    def __post_init__(self) -> None:
        if self.objective.ndim != 0:
            raise ValueError("Parameter update objective must be a scalar tensor")
        if not self.objective.requires_grad:
            raise ValueError("Parameter update objective must require gradients")
        if not bool(torch.isfinite(self.objective.detach()).item()):
            raise ValueError("Parameter update objective must be finite")


@dataclass(frozen=True, slots=True)
class ParameterUpdateResult:
    """Method-neutral metrics produced by one completed parameter update."""

    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, float] = {}
        for key, value in self.metrics.items():
            name = str(key)
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Parameter update metric {name!r} must be finite")
            normalized[name] = number
        object.__setattr__(self, "metrics", normalized)


class ParameterUpdatePolicy(Protocol):
    """Own backward and optimizer stepping for one scalar objective."""

    name: str

    def setup(self, context: ExperimentContext) -> None:
        ...

    def update(self, request: ParameterUpdateInput) -> ParameterUpdateResult:
        ...

    def state_dict(self) -> Mapping[str, Any]:
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        ...


class StandardUpdatePolicy:
    """The ordinary zero-grad, backward, optimizer-step baseline."""

    name = "standard"

    def __init__(self, max_grad_norm: float | None = None) -> None:
        if max_grad_norm is not None:
            max_grad_norm = float(max_grad_norm)
            if not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
                raise ValueError("max_grad_norm must be finite and positive")
        self.max_grad_norm = max_grad_norm

    def setup(self, context: ExperimentContext) -> None:
        del context

    def update(self, request: ParameterUpdateInput) -> ParameterUpdateResult:
        request.optimizer.zero_grad(set_to_none=True)
        request.objective.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                request.model.parameters(), self.max_grad_norm
            )
        request.optimizer.step()
        return ParameterUpdateResult()

    def state_dict(self) -> Mapping[str, Any]:
        return {} if self.max_grad_norm is None else {"max_grad_norm": self.max_grad_norm}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {} if self.max_grad_norm is None else {"max_grad_norm": self.max_grad_norm}
        if dict(state) != expected:
            raise ValueError("Standard update policy max_grad_norm does not match")


class StepMilestoneUpdatePolicy(StandardUpdatePolicy):
    """Apply reusable optimizer learning-rate milestones by global step."""

    name = "step_milestone"

    def __init__(
        self,
        milestones: list[int] | tuple[int, ...],
        gamma: float = 0.1,
        max_grad_norm: float | None = None,
    ) -> None:
        super().__init__(max_grad_norm=max_grad_norm)
        resolved = tuple(int(value) for value in milestones)
        if not resolved or any(value <= 0 for value in resolved):
            raise ValueError("milestones must contain positive global steps")
        if tuple(sorted(set(resolved))) != resolved:
            raise ValueError("milestones must be strictly increasing")
        self.milestones = resolved
        self.gamma = float(gamma)
        if not math.isfinite(self.gamma) or not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must satisfy 0 < gamma < 1")
        self.applied = 0

    def update(self, request: ParameterUpdateInput) -> ParameterUpdateResult:
        while (
            self.applied < len(self.milestones)
            and request.run_state.step >= self.milestones[self.applied]
        ):
            for group in request.optimizer.param_groups:
                group["lr"] *= self.gamma
            self.applied += 1
        super().update(request)
        return ParameterUpdateResult(
            {"learning_rate": float(request.optimizer.param_groups[0]["lr"])}
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "milestones": list(self.milestones),
            "gamma": self.gamma,
            "max_grad_norm": self.max_grad_norm,
            "applied": self.applied,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "milestones": list(self.milestones),
            "gamma": self.gamma,
            "max_grad_norm": self.max_grad_norm,
        }
        actual = {key: state.get(key) for key in expected}
        if actual != expected:
            raise ValueError("Step milestone update policy configuration mismatch")
        applied = int(state.get("applied", -1))
        if not 0 <= applied <= len(self.milestones):
            raise ValueError("Step milestone update policy state is invalid")
        self.applied = applied


def serialize_update_policy(policy: ParameterUpdatePolicy) -> dict[str, Any]:
    """Return a checkpoint-safe identity and private-state payload."""

    state = policy.state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("Parameter update policy state_dict() must return a mapping")
    return {"name": str(policy.name), "state": dict(state)}


def restore_update_policy(
    policy: ParameterUpdatePolicy,
    payload: Mapping[str, Any] | None,
) -> None:
    """Restore one policy without silently changing its registered identity."""

    if payload is None:
        if policy.name != StandardUpdatePolicy.name:
            raise ValueError("Checkpoint is missing parameter update policy state")
        return
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint parameter update policy state must be a mapping")
    saved_name = str(payload.get("name", "")).strip().lower()
    if saved_name != str(policy.name).strip().lower():
        raise ValueError(
            "Checkpoint parameter update policy does not match the current policy"
        )
    state = payload.get("state", {})
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint update policy private state must be a mapping")
    policy.load_state_dict(state)
