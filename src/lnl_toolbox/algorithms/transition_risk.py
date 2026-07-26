from __future__ import annotations

"""Risk correctors that consume shared transition-matrix artifacts."""

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from lnl_toolbox.losses.torch_losses import loss_for_all_targets
from lnl_toolbox.noise.transition import TransitionProvider


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

    def per_sample_risk(self, *, logits, noisy_targets, base_loss, transition):
        if logits.ndim != 2 or noisy_targets.ndim != 1:
            raise ValueError("logits and noisy_targets have invalid shapes")
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
