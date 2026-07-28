from __future__ import annotations

"""Binary unbiased risks used by the Natarajan noisy-label experiments."""

import math
from typing import Any

import torch
from torch import Tensor, nn

from lnl_toolbox.losses.torch_losses import loss_for_all_targets


def _rates(rho_positive: float, rho_negative: float) -> tuple[float, float, float]:
    values = (float(rho_positive), float(rho_negative))
    if not all(math.isfinite(value) and 0.0 <= value < 1.0 for value in values):
        raise ValueError("noise rates must be finite and in [0, 1)")
    denominator = 1.0 - sum(values)
    if denominator <= 0.0:
        raise ValueError("binary noise rates must have positive identifiability gap")
    return values[0], values[1], denominator


class NatarajanUnbiasedRisk:
    """Natarajan et al. binary risk under class-dependent label noise.

    ``rho_positive`` is P(tilde y=0 | y=1), and ``rho_negative`` is
    P(tilde y=1 | y=0). The class-conditional noise rates are known inputs.
    """

    name = "natarajan_unbiased"
    requires_transition = False

    def __init__(self, rho_positive: float, rho_negative: float) -> None:
        self.rho_positive, self.rho_negative, self.denominator = _rates(
            rho_positive, rho_negative
        )

    def per_sample_risk(
        self, *, logits: Tensor, noisy_targets: Tensor, base_loss: nn.Module,
        transition: Any = None,
    ) -> Tensor:
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError("binary risk logits must have shape [B, 2]")
        if noisy_targets.ndim != 1 or noisy_targets.shape != (logits.shape[0],):
            raise ValueError("binary noisy targets must have shape [B]")
        if noisy_targets.dtype != torch.long or bool(((noisy_targets < 0) | (noisy_targets > 1)).any()):
            raise ValueError("binary noisy targets must be long labels 0 or 1")
        clean_losses = loss_for_all_targets(base_loss, logits)
        loss_zero, loss_one = clean_losses[:, 0], clean_losses[:, 1]
        observed_zero = (
            (1.0 - self.rho_positive) * loss_zero - self.rho_negative * loss_one
        ) / self.denominator
        observed_one = (
            -self.rho_positive * loss_zero + (1.0 - self.rho_negative) * loss_one
        ) / self.denominator
        return torch.where(noisy_targets == 0, observed_zero, observed_one)

    def state_dict(self) -> dict[str, float]:
        return {
            "rho_positive": self.rho_positive,
            "rho_negative": self.rho_negative,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        current = self.state_dict()
        for key in current:
            if key in state and float(state[key]) != current[key]:
                raise ValueError("binary risk configuration mismatch")


class LabelDependentCostRisk:
    """A reusable class-dependent cost risk for binary observed labels."""

    name = "label_dependent_cost"
    requires_transition = False

    def __init__(self, negative_cost: float = 1.0, positive_cost: float = 1.0) -> None:
        if negative_cost < 0.0 or positive_cost < 0.0:
            raise ValueError("label-dependent costs must be non-negative")
        self.negative_cost = float(negative_cost)
        self.positive_cost = float(positive_cost)

    def per_sample_risk(self, *, logits, noisy_targets, base_loss, transition=None) -> Tensor:
        values = base_loss(logits, noisy_targets)
        costs = torch.where(
            noisy_targets == 1,
            torch.as_tensor(self.positive_cost, device=logits.device, dtype=logits.dtype),
            torch.as_tensor(self.negative_cost, device=logits.device, dtype=logits.dtype),
        )
        return values * costs


NatarajanRisk = NatarajanUnbiasedRisk
BinaryRiskCorrector = NatarajanUnbiasedRisk

__all__ = [
    "BinaryRiskCorrector",
    "LabelDependentCostRisk",
    "NatarajanRisk",
    "NatarajanUnbiasedRisk",
]
