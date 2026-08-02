from __future__ import annotations

"""Risk correctors that consume shared transition-matrix artifacts."""

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from lnl_toolbox.losses.torch_losses import loss_for_all_targets
from lnl_toolbox.noise.transition import TransitionProvider


def validate_instance_transitions(
    transitions: Tensor,
    *,
    batch_size: int,
    num_classes: int,
) -> Tensor:
    """Validate row-stochastic per-sample matrices ``[B,C,C]``."""

    if transitions.shape != (batch_size, num_classes, num_classes):
        raise ValueError(
            f"instance transitions must have shape [{batch_size}, {num_classes}, {num_classes}]"
        )
    if not torch.isfinite(transitions).all() or bool((transitions < 0).any()):
        raise ValueError("instance transitions must be finite and non-negative")
    if not torch.allclose(
        transitions.sum(dim=2),
        torch.ones((batch_size, num_classes), device=transitions.device, dtype=transitions.dtype),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("every instance transition row must sum to one")
    return transitions


def forward_instance_corrected_losses(
    logits: Tensor,
    noisy_targets: Tensor,
    transitions: Tensor,
    base_loss: nn.Module,
) -> Tensor:
    """Forward correction for one transition matrix per sample."""

    if logits.ndim != 2 or noisy_targets.shape != (logits.shape[0],):
        raise ValueError("logits and noisy_targets have invalid shapes")
    matrices = validate_instance_transitions(
        transitions, batch_size=logits.shape[0], num_classes=logits.shape[1]
    )
    observed_probabilities = torch.bmm(
        torch.softmax(logits, dim=1).unsqueeze(1), matrices
    ).squeeze(1)
    tiny = torch.finfo(logits.dtype).tiny
    corrected_logits = torch.log(observed_probabilities.clamp_min(tiny))
    if isinstance(base_loss, nn.CrossEntropyLoss):
        return -corrected_logits.gather(1, noisy_targets[:, None]).squeeze(1)
    return base_loss(corrected_logits, noisy_targets)


def instance_importance_reweighted_losses(
    logits: Tensor,
    noisy_targets: Tensor,
    transitions: Tensor,
    base_loss: nn.Module,
    *,
    detach_weights: bool = True,
    maximum_weight: float | None = None,
) -> Tensor:
    """Importance-correct noisy risk using ``p(y|x)/p(tilde-y|x)``."""

    if logits.ndim != 2 or noisy_targets.shape != (logits.shape[0],):
        raise ValueError("logits and noisy_targets have invalid shapes")
    matrices = validate_instance_transitions(
        transitions, batch_size=logits.shape[0], num_classes=logits.shape[1]
    )
    clean = torch.softmax(logits, dim=1)
    noisy = torch.bmm(clean.unsqueeze(1), matrices).squeeze(1)
    clean_observed = clean.gather(1, noisy_targets[:, None]).squeeze(1)
    noisy_observed = noisy.gather(1, noisy_targets[:, None]).squeeze(1)
    weights = clean_observed / noisy_observed.clamp_min(torch.finfo(logits.dtype).tiny)
    if maximum_weight is not None:
        if maximum_weight <= 0:
            raise ValueError("maximum_weight must be positive")
        weights = weights.clamp_max(float(maximum_weight))
    if detach_weights:
        weights = weights.detach()
    losses = base_loss(logits, noisy_targets)
    if losses.ndim != 1 or losses.shape != noisy_targets.shape:
        raise ValueError("base_loss must return per-sample losses")
    return losses * weights


@runtime_checkable
class RiskCorrector(Protocol):
    """Convert a base objective into a per-sample noisy-label risk."""

    def per_sample_risk(
        self,
        *,
        logits: Tensor,
        noisy_targets: Tensor,
        base_loss: nn.Module,
        transition: TransitionProvider,
    ) -> Tensor:
        ...


class ForwardRiskCorrector:
    """Forward correction: map clean posterior through ``T`` before loss."""

    name = "forward"
    requires_transition = True

    def per_sample_risk(self, *, logits, noisy_targets, base_loss, transition):
        if logits.ndim != 2 or noisy_targets.ndim != 1:
            raise ValueError("logits and noisy_targets have invalid shapes")
        if logits.shape[0] != noisy_targets.shape[0]:
            raise ValueError("logits and noisy_targets batch sizes must match")
        matrix = transition.as_tensor(device=logits.device, dtype=logits.dtype)
        if matrix.shape != (logits.shape[1], logits.shape[1]):
            raise ValueError("transition classes must match logits classes")
        probabilities = torch.softmax(logits, dim=1) @ matrix
        observed = probabilities.gather(1, noisy_targets[:, None]).squeeze(1)
        if isinstance(base_loss, nn.CrossEntropyLoss):
            return -torch.log(observed.clamp_min(torch.finfo(logits.dtype).tiny))
        corrected_logits = torch.log(probabilities.clamp_min(torch.finfo(logits.dtype).tiny))
        return base_loss(corrected_logits, noisy_targets)


class BackwardRiskCorrector:
    """Backward correction using ``T^{-1}`` under row-vector convention."""

    name = "backward"
    requires_transition = True

    def per_sample_risk(self, *, logits, noisy_targets, base_loss, transition):
        if logits.ndim != 2 or noisy_targets.ndim != 1:
            raise ValueError("logits and noisy_targets have invalid shapes")
        if noisy_targets.ndim != 1 or noisy_targets.shape[0] != logits.shape[0]:
            raise ValueError("noisy_targets must have shape [B]")
        matrix = transition.as_tensor(device=logits.device, dtype=logits.dtype)
        if matrix.shape != (logits.shape[1], logits.shape[1]):
            raise ValueError("transition classes must match logits classes")
        if not torch.isfinite(matrix).all():
            raise ValueError("transition matrix must be finite")
        condition = torch.linalg.cond(matrix)
        if not torch.isfinite(condition):
            raise ValueError("transition matrix must be invertible")
        losses = loss_for_all_targets(base_loss, logits)
        corrected = torch.linalg.solve(matrix, losses.transpose(0, 1)).transpose(0, 1)
        return corrected.gather(1, noisy_targets[:, None]).squeeze(1)
