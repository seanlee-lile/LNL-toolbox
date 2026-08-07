from __future__ import annotations

"""Joint classifier/transition optimizer owner for VolMinNet."""

from typing import Any, Mapping

import torch
from torch import nn

from .objective import volminnet_objective
from .state import VolMinNetState
from .transition import VolMinTransition


class VolMinNetAlgorithm:
    def __init__(
        self,
        *,
        model: nn.Module,
        transition: VolMinTransition,
        classifier_optimizer: torch.optim.Optimizer,
        transition_optimizer: torch.optim.Optimizer,
        classifier_scheduler: Any,
        transition_scheduler: Any,
        lambda_volume: float,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.transition = transition.to(device)
        self.classifier_optimizer = classifier_optimizer
        self.transition_optimizer = transition_optimizer
        self.classifier_scheduler = classifier_scheduler
        self.transition_scheduler = transition_scheduler
        self.lambda_volume = float(lambda_volume)
        self.device = device
        self.state = VolMinNetState()
        model_ids = {id(parameter) for parameter in self.model.parameters()}
        transition_ids = {id(parameter) for parameter in self.transition.parameters()}
        if model_ids & transition_ids:
            raise ValueError("VolMinNet classifier and transition parameters must be disjoint")
        optimizer_model_ids = {
            id(parameter)
            for group in classifier_optimizer.param_groups
            for parameter in group["params"]
        }
        optimizer_transition_ids = {
            id(parameter)
            for group in transition_optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_model_ids != model_ids or optimizer_transition_ids != transition_ids:
            raise ValueError("VolMinNet optimizers must own exactly their model parameters")
        if optimizer_model_ids & optimizer_transition_ids:
            raise ValueError("VolMinNet optimizer parameter sets must be disjoint")

    def train_batch(self, batch: Mapping[str, Any]) -> dict[str, float]:
        self.model.train()
        inputs = batch["input"].to(self.device, non_blocking=True)
        targets = batch["target"].to(self.device, non_blocking=True)
        logits = self.model(inputs)
        transition = self.transition.matrix(dtype=logits.dtype)
        objective, metrics = volminnet_objective(
            logits, targets, transition, lambda_volume=self.lambda_volume
        )
        self.classifier_optimizer.zero_grad(set_to_none=True)
        self.transition_optimizer.zero_grad(set_to_none=True)
        objective.backward()
        for owner, parameters in (
            ("classifier", self.model.parameters()),
            ("transition", self.transition.parameters()),
        ):
            gradients = [parameter.grad for parameter in parameters if parameter.requires_grad]
            if not gradients or any(gradient is None for gradient in gradients):
                raise ValueError(f"VolMinNet {owner} did not receive complete gradients")
            if not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
                raise ValueError(f"VolMinNet {owner} gradients are non-finite")
        self.classifier_optimizer.step()
        self.transition_optimizer.step()
        self.state.global_step += 1
        self.state.classifier_optimizer_steps += 1
        self.state.transition_optimizer_steps += 1
        return {**metrics, "samples": float(targets.numel())}

    def step_schedulers(self) -> None:
        if self.classifier_scheduler is not None:
            self.classifier_scheduler.step()
        if self.transition_scheduler is not None:
            self.transition_scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "transition": self.transition.state_dict(),
            "classifier_optimizer": self.classifier_optimizer.state_dict(),
            "transition_optimizer": self.transition_optimizer.state_dict(),
            "classifier_scheduler": (
                None if self.classifier_scheduler is None else self.classifier_scheduler.state_dict()
            ),
            "transition_scheduler": (
                None if self.transition_scheduler is None else self.transition_scheduler.state_dict()
            ),
            "volminnet_state": self.state.state_dict(),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        self.model.load_state_dict(value["model"])
        self.transition.load_state_dict(value["transition"])
        self.classifier_optimizer.load_state_dict(value["classifier_optimizer"])
        self.transition_optimizer.load_state_dict(value["transition_optimizer"])
        for scheduler, key in (
            (self.classifier_scheduler, "classifier_scheduler"),
            (self.transition_scheduler, "transition_scheduler"),
        ):
            saved = value.get(key)
            if scheduler is None and saved is not None:
                raise ValueError(f"VolMinNet checkpoint has unexpected {key}")
            if scheduler is not None:
                if saved is None:
                    raise ValueError(f"VolMinNet checkpoint is missing {key}")
                scheduler.load_state_dict(saved)
        self.state = VolMinNetState.from_state_dict(value["volminnet_state"])
