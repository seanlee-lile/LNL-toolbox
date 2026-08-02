from __future__ import annotations

"""Complete CNLCU-S dual-peer batch algorithm."""

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from lnl_toolbox.algorithms.coteaching.selection import determine_keep_count, stable_small_loss_mask
from lnl_toolbox.core import Batch, ExperimentContext, RunState, StepResult
from lnl_toolbox.losses.torch_losses import validate_per_sample_loss

from .config import CNLCUConfig
from .estimators import soft_robust_mean
from .history import PeerLossHistory
from .scoring import cnlcu_soft_score
from .state import CNLCUState


_PEERS = {"a", "b"}


def _peer_mapping(value: Any, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PEERS:
        raise ValueError(f"CNLCU {owner} must contain exactly peers a and b")
    return value


def _validate_peer_parameter_ownership(
    model_a: nn.Module,
    model_b: nn.Module,
    optimizer_a: torch.optim.Optimizer,
    optimizer_b: torch.optim.Optimizer,
) -> None:
    model_parameters = {
        "a": {id(parameter) for parameter in model_a.parameters()},
        "b": {id(parameter) for parameter in model_b.parameters()},
    }
    optimizer_parameters = {
        "a": {
            id(parameter)
            for group in optimizer_a.param_groups
            for parameter in group["params"]
        },
        "b": {
            id(parameter)
            for group in optimizer_b.param_groups
            for parameter in group["params"]
        },
    }
    if model_parameters["a"] & model_parameters["b"]:
        raise ValueError("CNLCU peer model parameter sets must not overlap")
    for peer in ("a", "b"):
        if optimizer_parameters[peer] != model_parameters[peer]:
            raise ValueError(
                f"CNLCU optimizer {peer} parameters must exactly match model {peer}"
            )
    if optimizer_parameters["a"] & optimizer_parameters["b"]:
        raise ValueError("CNLCU peer optimizer parameter sets must not overlap")


class CNLCUAlgorithm:
    """Use uncertainty-aware peer selections for exact cross-updates."""

    def __init__(self, *, model_a: nn.Module, model_b: nn.Module,
                 optimizer_a: torch.optim.Optimizer, optimizer_b: torch.optim.Optimizer,
                 scheduler_a: Any, scheduler_b: Any, loss: nn.Module,
                 device: torch.device, method_config: CNLCUConfig,
                 canonical_global_indices: Tensor) -> None:
        if model_a is model_b or optimizer_a is optimizer_b:
            raise ValueError("CNLCU peer models and optimizers must be distinct")
        if scheduler_a is not None and scheduler_a is scheduler_b:
            raise ValueError("CNLCU peer schedulers must be distinct")
        _validate_peer_parameter_ownership(
            model_a, model_b, optimizer_a, optimizer_b
        )
        self.model_a, self.model_b = model_a, model_b
        self.optimizer_a, self.optimizer_b = optimizer_a, optimizer_b
        self.scheduler_a, self.scheduler_b = scheduler_a, scheduler_b
        self.loss, self.device, self.method_config = loss, device, method_config
        self.private_state = CNLCUState(
            PeerLossHistory(canonical_global_indices, method_config.window_size, "a"),
            PeerLossHistory(canonical_global_indices, method_config.window_size, "b"),
        )

    def setup(self, context: ExperimentContext) -> None:
        del context
        self.model_a.to(self.device); self.model_b.to(self.device); self.loss.to(self.device)

    def on_run_start(self, state: RunState) -> None:
        del state

    def on_cycle_start(self, state: RunState) -> None:
        state.phase = "train"
        self.model_a.train(); self.model_b.train()
        self.private_state.history_a.prepare_epoch(state.cycle)
        self.private_state.history_b.prepare_epoch(state.cycle)

    def _score(self, history: PeerLossHistory, rows: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        values, observed, selected_count = history.lookup_rows(rows)
        robust_mean, lengths = soft_robust_mean(values, observed)
        # The stored value is the number of completed selections in the active
        # epoch window.  The one-count pseudocount makes the first selection
        # attempt well-defined without pretending it is a completed selection.
        effective_count = selected_count + 1
        score, bonus = cnlcu_soft_score(
            robust_mean, lengths, effective_count, self.method_config.sigma_squared
        )
        return score, {"robust_mean": robust_mean, "history_length": lengths,
                       "effective_selected_count": effective_count, "confidence_bonus": bonus}

    def step(self, batch: Batch, state: RunState) -> StepResult:
        payload = batch.payload
        inputs = payload["input"].to(self.device, non_blocking=True)
        targets = payload["target"].to(self.device, non_blocking=True)
        indices = torch.as_tensor(payload["index"], dtype=torch.long, device=self.device)
        count = int(targets.numel())
        if count == 0 or indices.shape != targets.shape:
            raise ValueError("CNLCU targets and stable indices must align as [B]")
        logits_a, logits_b = self.model_a(inputs), self.model_b(inputs)
        if logits_a.ndim != 2 or logits_a.shape != logits_b.shape:
            raise ValueError("CNLCU peer logits must have matching shape [B,C]")
        losses_a = validate_per_sample_loss(self.loss(logits_a, targets), count)
        losses_b = validate_per_sample_loss(self.loss(logits_b, targets), count)
        rows_a = self.private_state.history_a.append(indices, losses_a.detach())
        rows_b = self.private_state.history_b.append(indices, losses_b.detach())
        score_a, detail_a = self._score(self.private_state.history_a, rows_a)
        score_b, detail_b = self._score(self.private_state.history_b, rows_b)
        rate = self.method_config.rate_at(state.cycle)
        keep_count = determine_keep_count(count, rate)
        selected_a = stable_small_loss_mask(score_a.to(self.device), indices, keep_count)
        selected_b = stable_small_loss_mask(score_b.to(self.device), indices, keep_count)
        objective_a, objective_b = losses_a[selected_b].mean(), losses_b[selected_a].mean()
        self.optimizer_a.zero_grad(set_to_none=True); objective_a.backward(); self.optimizer_a.step()
        self.private_state.optimizer_steps_a += 1
        self.optimizer_b.zero_grad(set_to_none=True); objective_b.backward(); self.optimizer_b.step()
        self.private_state.optimizer_steps_b += 1
        self.private_state.history_a.increment_selected(rows_a, selected_a)
        self.private_state.history_b.increment_selected(rows_b, selected_b)
        state.step += 1
        pred_a, pred_b = logits_a.argmax(1), logits_b.argmax(1)
        ensemble = (torch.softmax(logits_a.detach(), 1) + torch.softmax(logits_b.detach(), 1)) / 2
        metrics: dict[str, float] = {
            "loss_a_on_selected_by_b": objective_a.detach().item(),
            "loss_b_on_selected_by_a": objective_b.detach().item(),
            "current_loss_a": losses_a.detach().mean().item(), "current_loss_b": losses_b.detach().mean().item(),
            "robust_mean_a": detail_a["robust_mean"].mean().item(), "robust_mean_b": detail_b["robust_mean"].mean().item(),
            "confidence_bonus_a": detail_a["confidence_bonus"].mean().item(), "confidence_bonus_b": detail_b["confidence_bonus"].mean().item(),
            "uncertainty_score_a": score_a.mean().item(), "uncertainty_score_b": score_b.mean().item(),
            "history_length_a": detail_a["history_length"].float().mean().item(), "history_length_b": detail_b["history_length"].float().mean().item(),
            "effective_selected_count_a": detail_a["effective_selected_count"].float().mean().item(), "effective_selected_count_b": detail_b["effective_selected_count"].float().mean().item(),
            "selected_by_a_count": selected_a.sum().item(), "selected_by_b_count": selected_b.sum().item(),
            "remember_rate": rate,
            "selection_overlap_rate": (selected_a & selected_b).sum().item() / keep_count,
            "prediction_agreement_rate": (pred_a == pred_b).float().mean().item(),
            "accuracy_a": (pred_a == targets).float().mean().item(), "accuracy_b": (pred_b == targets).float().mean().item(),
            "accuracy_ensemble": (ensemble.argmax(1) == targets).float().mean().item(),
            "samples": float(count), "optimizer_steps_a": float(self.private_state.optimizer_steps_a),
            "optimizer_steps_b": float(self.private_state.optimizer_steps_b),
        }
        return StepResult(metrics=metrics, metadata={
            "selected_by_a_indices": indices[selected_a].detach().cpu(),
            "selected_by_b_indices": indices[selected_b].detach().cpu(),
        })

    def on_cycle_end(self, state: RunState) -> StepResult:
        del state; return StepResult()
    def on_run_end(self, state: RunState) -> StepResult:
        del state; return StepResult()
    def step_schedulers(self) -> None:
        if self.scheduler_a is not None: self.scheduler_a.step()
        if self.scheduler_b is not None: self.scheduler_b.step()

    def state_dict(self) -> dict[str, Any]:
        schedule = self.method_config.remember_schedule
        return {"model": {"a": self.model_a.state_dict(), "b": self.model_b.state_dict()},
                "optimizer": {"a": self.optimizer_a.state_dict(), "b": self.optimizer_b.state_dict()},
                "schedulers": {"a": None if self.scheduler_a is None else self.scheduler_a.state_dict(),
                               "b": None if self.scheduler_b is None else self.scheduler_b.state_dict()},
                "cnlcu_state": self.private_state.state_dict(), "method_identity": "cnlcu",
                "peer_identity": ("a", "b"), "variant": self.method_config.variant,
                "sigma_squared": self.method_config.sigma_squared,
                "window_size": self.method_config.window_size,
                "selected_count_scope": "fixed_window",
                "noise_rate": self.method_config.noise_rate,
                "remember_schedule": {
                    "name": "linear",
                    "start": schedule.start,
                    "end": schedule.end,
                    "gradual_epochs": schedule.warmup_epochs,
                }}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        models = _peer_mapping(state.get("model"), "model state")
        optimizers = _peer_mapping(state.get("optimizer"), "optimizer state")
        schedulers = _peer_mapping(state.get("schedulers"), "scheduler state")
        if state.get("method_identity") != "cnlcu" or tuple(state.get("peer_identity", ())) != ("a", "b"):
            raise ValueError("checkpoint method or peer identity is not CNLCU")
        schedule = self.method_config.remember_schedule
        expected_schedule = {
            "name": "linear",
            "start": schedule.start,
            "end": schedule.end,
            "gradual_epochs": schedule.warmup_epochs,
        }
        if (
            state.get("variant") != self.method_config.variant
            or int(state.get("window_size", -1)) != self.method_config.window_size
            or state.get("selected_count_scope") != "fixed_window"
            or float(state.get("sigma_squared", -1)) != self.method_config.sigma_squared
            or float(state.get("noise_rate", -1)) != self.method_config.noise_rate
            or state.get("remember_schedule") != expected_schedule
        ):
            raise ValueError("CNLCU checkpoint method configuration changed")
        self.model_a.load_state_dict(models["a"]); self.model_b.load_state_dict(models["b"])
        self.optimizer_a.load_state_dict(optimizers["a"]); self.optimizer_b.load_state_dict(optimizers["b"])
        for optimizer in (self.optimizer_a, self.optimizer_b):
            for values in optimizer.state.values():
                for key, value in values.items():
                    if torch.is_tensor(value): values[key] = value.to(self.device)
        for name, scheduler in (("a", self.scheduler_a), ("b", self.scheduler_b)):
            saved = schedulers[name]
            if (scheduler is None) != (saved is None):
                raise ValueError(f"CNLCU scheduler {name} configuration changed")
            if scheduler is not None: scheduler.load_state_dict(saved)
        self.private_state.load_state_dict(state.get("cnlcu_state"))

    def close(self) -> None:
        pass


__all__ = ["CNLCUAlgorithm"]
