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
)


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
        if correction not in {"forward", "importance"}:
            raise ValueError("correction must be 'forward' or 'importance'")
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
        if self.correction == "forward":
            per_sample = forward_instance_corrected_losses(logits, targets, matrices, self.loss)
        else:
            per_sample = instance_importance_reweighted_losses(
                logits, targets, matrices, self.loss,
                detach_weights=self.detach_importance_weights,
                maximum_weight=self.maximum_importance_weight,
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
        correct = int((logits.argmax(1) == targets).sum().item())
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
