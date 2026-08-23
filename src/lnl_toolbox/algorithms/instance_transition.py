from __future__ import annotations

"""Generic single-model training with per-sample transition matrices."""

from typing import Any

import torch
from torch import nn

from lnl_toolbox.core import Batch, ExperimentContext, RunState, StepResult
from lnl_toolbox.noise.transition import InstanceTransitionProvider
from .transition_risk import (
    forward_instance_corrected_losses,
    instance_importance_reweighted_losses,
    validate_instance_transitions,
)


def _row_abs_normalize(transitions: torch.Tensor) -> torch.Tensor:
    """Match PDL ``tools.norm`` for a batch of transition matrices."""

    return transitions.abs() / transitions.abs().sum(dim=2, keepdim=True).clamp_min(
        torch.finfo(transitions.dtype).tiny
    )


def pdl_instance_corrected_losses(
    logits: torch.Tensor,
    noisy_targets: torch.Tensor,
    transitions: torch.Tensor,
    *,
    detach_importance_weight: bool = False,
) -> torch.Tensor:
    """PDL's ``beta * NLL(log p_theta(y_tilde|x))`` objective."""

    if logits.ndim != 2 or noisy_targets.shape != (logits.shape[0],):
        raise ValueError("logits and noisy_targets have invalid shapes")
    if transitions.shape != (logits.shape[0], logits.shape[1], logits.shape[1]):
        raise ValueError("PDL transitions have incompatible shapes")
    if not torch.isfinite(transitions).all() or bool((transitions < 0).any()):
        raise ValueError("PDL transitions must be finite and non-negative")
    # Official train_correction consumes the clipped raw matrix directly;
    # unlike generic transition consumers it does not row-normalize it.
    matrices = transitions
    clean = torch.softmax(logits, dim=1)
    noisy = torch.bmm(clean.unsqueeze(1), matrices).squeeze(1)
    target_probability = clean.gather(1, noisy_targets[:, None]).squeeze(1)
    noisy_probability = noisy.gather(1, noisy_targets[:, None]).squeeze(1)
    weight = target_probability / noisy_probability.clamp_min(
        torch.finfo(logits.dtype).tiny
    )
    if detach_importance_weight:
        weight = weight.detach()
    return weight * -torch.log(target_probability.clamp_min(torch.finfo(logits.dtype).tiny))


class InstanceTransitionClassificationAlgorithm:
    """Consume any ``InstanceTransitionProvider`` without paper-specific branches."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module,
        transition: InstanceTransitionProvider,
        device: torch.device,
        *,
        correction: str = "forward",
        detach_importance_weights: bool = True,
        maximum_importance_weight: float | None = None,
    ) -> None:
        correction = str(correction).strip().lower()
        if correction not in {"forward", "importance", "pdl", "pdl_revision"}:
            raise ValueError(
                "correction must be 'forward', 'importance', 'pdl', or 'pdl_revision'"
            )
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.transition = transition
        self.device = torch.device(device)
        self.correction = correction
        self.detach_importance_weights = bool(detach_importance_weights)
        self.maximum_importance_weight = maximum_importance_weight

    def setup(self, context: ExperimentContext) -> None:
        self.model.to(self.device)
        self.loss.to(self.device)

    def on_run_start(self, state: RunState) -> None:
        pass

    def on_cycle_start(self, state: RunState) -> None:
        state.phase = "train"
        self.model.train()

    def step(self, batch: Batch, state: RunState) -> StepResult:
        inputs = batch.payload["input"].to(self.device, non_blocking=True)
        targets = batch.payload["target"].to(self.device, non_blocking=True)
        indices = torch.as_tensor(batch.payload["index"], dtype=torch.long, device=self.device)
        logits = self.model(inputs)
        matrices = self.transition.transition_for(
            inputs, indices, device=logits.device, dtype=logits.dtype
        )
        effective_matrices = matrices
        if self.correction == "pdl_revision":
            revision = getattr(self.model, "T_revision", None)
            if not isinstance(revision, nn.Linear) or revision.bias is not None:
                raise TypeError(
                    "PDL revision requires a bias-free model.T_revision Linear head"
                )
            effective_matrices = _row_abs_normalize(
                matrices + revision.weight.to(matrices).unsqueeze(0)
            )
        if self.correction == "forward":
            per_sample = forward_instance_corrected_losses(
                logits, targets, matrices, self.loss
            )
        elif self.correction == "importance":
            per_sample = instance_importance_reweighted_losses(
                logits, targets, matrices, self.loss,
                detach_weights=self.detach_importance_weights,
                maximum_weight=self.maximum_importance_weight,
            )
        else:
            per_sample = pdl_instance_corrected_losses(
                logits, targets, effective_matrices,
                # The official correction phase re-wraps beta as an
                # independent Variable; revision instead differentiates it.
                detach_importance_weight=self.correction == "pdl",
            )
        if per_sample.ndim != 1 or per_sample.shape != targets.shape:
            raise ValueError("corrected risk must return one loss per sample")
        objective = per_sample.mean()
        if not torch.isfinite(objective):
            raise ValueError("corrected objective must be finite")
        self.optimizer.zero_grad(set_to_none=True)
        objective.backward()
        self.optimizer.step()
        state.step += 1
        count = int(targets.numel())
        metric_logits = logits
        if self.correction in {"pdl", "pdl_revision"}:
            observed = torch.bmm(
                torch.softmax(logits, dim=1).unsqueeze(1), effective_matrices
            ).squeeze(1)
            metric_logits = torch.log(
                observed.clamp_min(torch.finfo(logits.dtype).tiny)
            )
        correct = int((metric_logits.argmax(1) == targets).sum().item())
        return StepResult(metrics={
            "loss": float(objective.detach().item()),
            "accuracy": correct / count,
            "samples": float(count),
        })

    def on_cycle_end(self, state: RunState) -> StepResult:
        return StepResult()

    def on_run_end(self, state: RunState) -> StepResult:
        return StepResult()

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "correction": self.correction,
            "transition_artifact_hash": getattr(self.transition, "artifact_hash", None),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("correction") != self.correction:
            raise ValueError("instance correction configuration mismatch")
        expected_hash = getattr(self.transition, "artifact_hash", None)
        if state.get("transition_artifact_hash") != expected_hash:
            raise ValueError("instance transition artifact mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)

    def close(self) -> None:
        pass
