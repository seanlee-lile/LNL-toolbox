"""Continuous sample-weight contracts and binary RCN importance weights."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import (
    Any,
    Generic,
    Mapping,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import torch
from torch import Tensor

from .base import ContributionResult


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}
_PROBABILITY_ATOL = 1e-6
_PROBABILITY_RTOL = 1e-5
_NEGATIVE_WEIGHT_ATOL = 1e-7


@dataclass(frozen=True)
class BinaryRCNWeightInput:
    """Binary noisy-label posterior and observed labels for one batch."""

    posterior_probabilities: Tensor
    observed_targets: Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WeightResult:
    """Continuous non-negative sample weights and provider-owned statistics."""

    sample_weights: Tensor
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SupervisedWeightInput:
    """Noisy-only signals exposed by the ordinary supervised training step.

    This input deliberately does not contain a posterior estimate. Providers
    that require method-specific evidence must define and receive their own
    typed input instead of inferring it from the current classifier.
    """

    logits: Tensor
    noisy_targets: Tensor
    sample_indices: Tensor
    per_sample_loss: Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)


InputT_contra = TypeVar("InputT_contra", contravariant=True)


@runtime_checkable
class WeightProvider(Protocol[InputT_contra]):
    """Compute one detached continuous weight per observed sample."""

    def compute(self, weight_input: InputT_contra) -> WeightResult:
        """Return weights aligned with the input batch."""


@runtime_checkable
class StatefulWeightProvider(WeightProvider[InputT_contra], Protocol):
    """Optional state contract for history-based or learned weights."""

    def state_dict(self) -> Mapping[str, Any]:
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        ...


def _validate_rate(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    rate = float(value)
    if not math.isfinite(rate) or not 0.0 <= rate < 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1")
    return rate


def validate_binary_rcn_weight_input(
    weight_input: BinaryRCNWeightInput,
) -> tuple[Tensor, Tensor]:
    """Validate the binary noisy-posterior boundary and detach the posterior."""

    if not isinstance(weight_input, BinaryRCNWeightInput):
        raise TypeError("weight_input must be a BinaryRCNWeightInput")

    posterior = weight_input.posterior_probabilities
    targets = weight_input.observed_targets
    if (
        not isinstance(posterior, Tensor)
        or posterior.ndim != 2
        or posterior.shape[1] != 2
    ):
        raise ValueError("posterior_probabilities must have shape [B, 2]")
    if posterior.shape[0] == 0:
        raise ValueError("posterior_probabilities must not be empty")
    if not torch.is_floating_point(posterior):
        raise ValueError("posterior_probabilities must use a floating-point dtype")
    if not bool(torch.isfinite(posterior).all().item()):
        raise ValueError("posterior_probabilities must be finite")
    if bool(((posterior < 0) | (posterior > 1)).any().item()):
        raise ValueError("posterior_probabilities must be within [0, 1]")
    if not torch.allclose(
        posterior.sum(dim=1),
        torch.ones(
            posterior.shape[0],
            dtype=posterior.dtype,
            device=posterior.device,
        ),
        atol=_PROBABILITY_ATOL,
        rtol=_PROBABILITY_RTOL,
    ):
        raise ValueError("each posterior row must sum to one")

    if not isinstance(targets, Tensor) or targets.shape != (posterior.shape[0],):
        raise ValueError("observed_targets must have shape [B]")
    if targets.device != posterior.device:
        raise ValueError(
            "observed_targets and posterior_probabilities must be on the same device"
        )
    if targets.dtype not in _INTEGER_DTYPES:
        raise ValueError("observed_targets must use an integer dtype")
    if bool(((targets != 0) & (targets != 1)).any().item()):
        raise ValueError("observed_targets must contain only binary labels 0 or 1")
    if not isinstance(weight_input.metadata, Mapping):
        raise TypeError("weight metadata must be a mapping")
    return posterior.detach(), targets


def validate_weight_result(
    result: WeightResult,
) -> Tensor:
    """Validate self-contained provider output and return its weights."""

    if not isinstance(result, WeightResult):
        raise TypeError("weight provider must return a WeightResult")
    weights = result.sample_weights
    if not isinstance(weights, Tensor) or weights.ndim != 1:
        raise ValueError("sample_weights must be a one-dimensional tensor")
    if weights.numel() == 0:
        raise ValueError("sample_weights must not be empty")
    if not torch.is_floating_point(weights):
        raise ValueError("sample_weights must use a floating-point dtype")
    if weights.requires_grad:
        raise ValueError("sample_weights must be detached from autograd")
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("sample_weights must be finite")
    if bool((weights < 0).any().item()):
        raise ValueError("sample_weights must be non-negative")

    if not isinstance(result.metrics, Mapping):
        raise TypeError("weight metrics must be a mapping")
    for name, value in result.metrics.items():
        if not isinstance(name, str):
            raise TypeError("weight metric names must be strings")
        if type(value) is not float:
            raise TypeError(
                f"weight metric {name!r} must be a Python float"
            )
        if not math.isfinite(value):
            raise ValueError(f"weight metric {name!r} must be finite")
    return weights


class BinaryRCNImportanceWeightProvider:
    """Paper-exact importance weights for binary asymmetric class noise.

    The posterior input is ``P(noisy_Y = class | X)``. Labels use ``0`` for
    the negative class and ``1`` for the positive class.
    """

    def __init__(self, rho_positive: float, rho_negative: float) -> None:
        self.rho_positive = _validate_rate("rho_positive", rho_positive)
        self.rho_negative = _validate_rate("rho_negative", rho_negative)
        if self.rho_positive + self.rho_negative >= 1.0:
            raise ValueError("rho_positive + rho_negative must be less than 1")

    def compute(self, weight_input: BinaryRCNWeightInput) -> WeightResult:
        posterior, targets = validate_binary_rcn_weight_input(weight_input)
        q = posterior.gather(1, targets[:, None].long()).squeeze(1)
        opposite_rate = torch.where(
            targets == 1,
            torch.as_tensor(
                self.rho_negative,
                dtype=posterior.dtype,
                device=posterior.device,
            ),
            torch.as_tensor(
                self.rho_positive,
                dtype=posterior.dtype,
                device=posterior.device,
            ),
        )
        noise_gap = 1.0 - self.rho_positive - self.rho_negative
        weights = torch.zeros_like(q)
        nonzero = q != 0
        weights[nonzero] = (
            (q[nonzero] - opposite_rate[nonzero])
            / (noise_gap * q[nonzero])
        )

        if not bool(torch.isfinite(weights).all().item()):
            raise ValueError("importance weights must be finite")
        minimum = float(weights.min().item())
        if minimum < -_NEGATIVE_WEIGHT_ATOL:
            raise ValueError(
                "estimated noisy-label posterior produced a negative importance weight"
            )
        weights = weights.clamp_min(0).detach()
        zero_ratio = float((weights == 0).float().mean().item())
        return WeightResult(
            sample_weights=weights,
            metrics={
                "weight_mean": float(weights.mean().item()),
                "weight_min": float(weights.min().item()),
                "weight_max": float(weights.max().item()),
                "zero_weight_ratio": zero_ratio,
            },
        )


InputT = TypeVar("InputT")


class WeightContributionAdapter(Generic[InputT]):
    """Adapt a continuous WeightProvider to the shared contribution contract."""

    def __init__(self, provider: WeightProvider[InputT]) -> None:
        self.provider = provider

    def resolve(self, weight_input: InputT) -> ContributionResult:
        result = self.provider.compute(weight_input)
        weights = validate_weight_result(result)
        return ContributionResult(
            selected_mask=torch.ones_like(weights, dtype=torch.bool),
            sample_weights=weights,
            metrics=dict(result.metrics),
        )

    def state_dict(self) -> dict[str, Any]:
        if hasattr(self.provider, "state_dict"):
            return {"state": dict(self.provider.state_dict())}
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        if not hasattr(self.provider, "load_state_dict"):
            raise ValueError("weight provider is not stateful")
        provider_state = state.get("state", state)
        if not isinstance(provider_state, Mapping):
            raise TypeError("weight provider state must be a mapping")
        self.provider.load_state_dict(provider_state)
