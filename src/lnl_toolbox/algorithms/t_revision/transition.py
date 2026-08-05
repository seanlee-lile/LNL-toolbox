from __future__ import annotations

"""Paper-experiment raw additive transition revision."""

from typing import Iterable

import torch
from torch import Tensor, nn


class AdditiveTransitionRevision(nn.Module):
    """Keep ``T_hat`` fixed and learn an unconstrained additive ``delta``."""

    def __init__(self, initial_transition: Tensor) -> None:
        super().__init__()
        if not torch.is_tensor(initial_transition) or initial_transition.ndim != 2:
            raise ValueError("initial_transition must have square shape [C, C]")
        if initial_transition.shape[0] != initial_transition.shape[1]:
            raise ValueError("initial_transition must have square shape [C, C]")
        if initial_transition.shape[0] < 2 or not torch.is_floating_point(initial_transition):
            raise ValueError("initial_transition must be floating with C >= 2")
        if not bool(torch.isfinite(initial_transition).all().item()):
            raise ValueError("initial_transition must be finite")
        self.register_buffer("initial_transition", initial_transition.detach().clone())
        self.delta = nn.Parameter(torch.zeros_like(initial_transition))

    def forward(self) -> Tensor:
        revised = self.initial_transition + self.delta
        if not bool(torch.isfinite(revised).all().item()):
            raise ValueError("revised transition must be finite")
        return revised

    def diagnostics(self) -> dict[str, object]:
        revised = self().detach()
        row_sums = revised.sum(dim=1)
        return {
            "row_sums": [float(value) for value in row_sums.cpu().tolist()],
            "minimum": float(revised.min().item()),
            "maximum": float(revised.max().item()),
            "diagonal": [float(value) for value in revised.diagonal().cpu().tolist()],
            "finite": bool(torch.isfinite(revised).all().item()),
            "non_negative": bool((revised >= 0).all().item()),
            "row_stochastic": bool(
                (revised >= 0).all().item()
                and torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-6, atol=1e-8)
            ),
            "delta_l1": float(self.delta.detach().abs().sum().item()),
            "delta_frobenius": float(torch.linalg.vector_norm(self.delta.detach()).item()),
        }


def validate_revision_optimizer(
    optimizer: torch.optim.Optimizer,
    model_parameters: Iterable[nn.Parameter],
    revision: AdditiveTransitionRevision,
) -> None:
    """Require the optimizer to own every model parameter and delta exactly once."""

    expected = [parameter for parameter in model_parameters if parameter.requires_grad]
    expected.append(revision.delta)
    actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    actual_ids = [id(parameter) for parameter in actual]
    expected_ids = [id(parameter) for parameter in expected]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("revision optimizer contains duplicate parameters")
    if set(actual_ids) != set(expected_ids) or len(actual_ids) != len(expected_ids):
        raise ValueError(
            "revision optimizer must contain exactly model parameters plus delta"
        )
