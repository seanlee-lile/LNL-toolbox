from __future__ import annotations

"""Complete dual-model Co-teaching batch algorithm."""

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from lnl_toolbox.core import Batch, ExperimentContext, RunState, StepResult
from lnl_toolbox.losses.torch_losses import validate_per_sample_loss

from .config import CoTeachingConfig
from .selection import determine_keep_count, stable_small_loss_mask
from .state import CoTeachingState


_PEERS = {"a", "b"}


def _peer_mapping(value: Any, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PEERS:
        raise ValueError(f"Co-teaching {owner} must contain exactly peers a and b")
    return value


class CoTeachingAlgorithm:
    """Own two peers and update each from the other peer's small-loss set."""

    def __init__(
        self,
        *,
        model_a: nn.Module,
        model_b: nn.Module,
        optimizer_a: torch.optim.Optimizer,
        optimizer_b: torch.optim.Optimizer,
        scheduler_a: Any,
        scheduler_b: Any,
        loss: nn.Module,
        device: torch.device,
        method_config: CoTeachingConfig,
    ) -> None:
        if model_a is model_b:
            raise ValueError("Co-teaching peer models must be distinct objects")
        if optimizer_a is optimizer_b:
            raise ValueError("Co-teaching peer optimizers must be distinct objects")
        if scheduler_a is not None and scheduler_a is scheduler_b:
            raise ValueError("Co-teaching peer schedulers must be distinct objects")
        self.model_a = model_a
        self.model_b = model_b
        self.optimizer_a = optimizer_a
        self.optimizer_b = optimizer_b
        self.scheduler_a = scheduler_a
        self.scheduler_b = scheduler_b
        self.loss = loss
        self.device = device
        self.method_config = method_config
        self.private_state = CoTeachingState()

    def setup(self, context: ExperimentContext) -> None:
        del context
        self.model_a.to(self.device)
        self.model_b.to(self.device)
        self.loss.to(self.device)

    def on_run_start(self, state: RunState) -> None:
        del state

    def on_cycle_start(self, state: RunState) -> None:
        state.phase = "train"
        self.model_a.train()
        self.model_b.train()

    def step(self, batch: Batch, state: RunState) -> StepResult:
        payload = batch.payload
        inputs = payload["input"].to(self.device, non_blocking=True)
        targets = payload["target"].to(self.device, non_blocking=True)
        indices = torch.as_tensor(
            payload["index"], dtype=torch.long, device=self.device
        )
        count = int(targets.numel())
        if count == 0 or indices.shape != targets.shape:
            raise ValueError("Co-teaching targets and stable indices must align as [B]")

        # Both forwards precede either optimizer update. The model graphs are disjoint.
        logits_a = self.model_a(inputs)
        logits_b = self.model_b(inputs)
        if logits_a.shape != logits_b.shape or logits_a.ndim != 2:
            raise ValueError("Co-teaching peer logits must have matching shape [B,C]")
        losses_a = validate_per_sample_loss(self.loss(logits_a, targets), count)
        losses_b = validate_per_sample_loss(self.loss(logits_b, targets), count)

        remember_rate = self.method_config.rate_at(state.cycle)
        keep_count = determine_keep_count(count, remember_rate)
        selected_by_a = stable_small_loss_mask(
            losses_a.detach(), indices, keep_count
        )
        selected_by_b = stable_small_loss_mask(
            losses_b.detach(), indices, keep_count
        )
        objective_a = losses_a[selected_by_b].mean()
        objective_b = losses_b[selected_by_a].mean()

        self.optimizer_a.zero_grad(set_to_none=True)
        objective_a.backward()
        self.optimizer_a.step()
        self.private_state.optimizer_steps_a += 1

        self.optimizer_b.zero_grad(set_to_none=True)
        objective_b.backward()
        self.optimizer_b.step()
        self.private_state.optimizer_steps_b += 1
        state.step += 1

        predictions_a = logits_a.argmax(1)
        predictions_b = logits_b.argmax(1)
        ensemble = (
            torch.softmax(logits_a.detach(), dim=1)
            + torch.softmax(logits_b.detach(), dim=1)
        ) / 2.0
        overlap = int((selected_by_a & selected_by_b).sum().item())
        metrics = {
            "loss_a_on_selected_by_b": float(objective_a.detach().item()),
            "loss_b_on_selected_by_a": float(objective_b.detach().item()),
            "all_sample_loss_a": float(losses_a.detach().mean().item()),
            "all_sample_loss_b": float(losses_b.detach().mean().item()),
            "selected_by_a_count": float(selected_by_a.sum().item()),
            "selected_by_b_count": float(selected_by_b.sum().item()),
            "remember_rate": remember_rate,
            "selection_overlap_rate": overlap / keep_count,
            "prediction_agreement_rate": float(
                (predictions_a == predictions_b).float().mean().item()
            ),
            "accuracy_a": float((predictions_a == targets).float().mean().item()),
            "accuracy_b": float((predictions_b == targets).float().mean().item()),
            "accuracy_ensemble": float(
                (ensemble.argmax(1) == targets).float().mean().item()
            ),
            "samples": float(count),
            "optimizer_steps_a": float(self.private_state.optimizer_steps_a),
            "optimizer_steps_b": float(self.private_state.optimizer_steps_b),
        }
        return StepResult(
            metrics=metrics,
            metadata={
                "selected_by_a_indices": indices[selected_by_a].detach().cpu(),
                "selected_by_b_indices": indices[selected_by_b].detach().cpu(),
            },
        )

    def on_cycle_end(self, state: RunState) -> StepResult:
        del state
        return StepResult()

    def on_run_end(self, state: RunState) -> StepResult:
        del state
        return StepResult()

    def step_schedulers(self) -> None:
        if self.scheduler_a is not None:
            self.scheduler_a.step()
        if self.scheduler_b is not None:
            self.scheduler_b.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "a": self.model_a.state_dict(),
                "b": self.model_b.state_dict(),
            },
            "optimizer": {
                "a": self.optimizer_a.state_dict(),
                "b": self.optimizer_b.state_dict(),
            },
            "schedulers": {
                "a": None if self.scheduler_a is None else self.scheduler_a.state_dict(),
                "b": None if self.scheduler_b is None else self.scheduler_b.state_dict(),
            },
            "coteaching_state": self.private_state.state_dict(),
            "method_identity": "coteaching",
            "peer_identity": ("a", "b"),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        models = _peer_mapping(state.get("model"), "model state")
        optimizers = _peer_mapping(state.get("optimizer"), "optimizer state")
        schedulers = _peer_mapping(state.get("schedulers"), "scheduler state")
        if state.get("method_identity") != "coteaching":
            raise ValueError("checkpoint method identity is not Co-teaching")
        if tuple(state.get("peer_identity", ())) != ("a", "b"):
            raise ValueError("Co-teaching checkpoint peer identity changed")
        self.model_a.load_state_dict(models["a"])
        self.model_b.load_state_dict(models["b"])
        self.optimizer_a.load_state_dict(optimizers["a"])
        self.optimizer_b.load_state_dict(optimizers["b"])
        for optimizer in (self.optimizer_a, self.optimizer_b):
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(self.device)
        for name, scheduler in (("a", self.scheduler_a), ("b", self.scheduler_b)):
            saved = schedulers[name]
            if scheduler is None and saved is not None:
                raise ValueError(f"checkpoint enables scheduler {name} but config disables it")
            if scheduler is not None and saved is None:
                raise ValueError(f"checkpoint is missing scheduler {name}")
            if scheduler is not None:
                scheduler.load_state_dict(saved)
        self.private_state.load_state_dict(state.get("coteaching_state"))

    def close(self) -> None:
        pass
