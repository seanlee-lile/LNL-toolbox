from __future__ import annotations

"""Two-network computational core for the complete DivideMix workflow."""

from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .config import DivideMixConfig
from .mixmatch import mixmatch_mixup
from .objective import dividemix_objective, unsupervised_weight
from .state import DivideMixState
from .targets import co_guess, co_refine


class DivideMixAlgorithm:
    def __init__(self, *, model_a: nn.Module, model_b: nn.Module, optimizer_a: torch.optim.Optimizer, optimizer_b: torch.optim.Optimizer, scheduler_a: Any, scheduler_b: Any, config: DivideMixConfig, device: torch.device) -> None:
        if model_a is model_b or optimizer_a is optimizer_b:
            raise ValueError("DivideMix peers and optimizers must be distinct")
        parameters_a = {id(value) for group in optimizer_a.param_groups for value in group["params"]}
        parameters_b = {id(value) for group in optimizer_b.param_groups for value in group["params"]}
        if parameters_a != {id(value) for value in model_a.parameters()} or parameters_b != {id(value) for value in model_b.parameters()} or parameters_a & parameters_b:
            raise ValueError("DivideMix optimizer ownership does not match disjoint peer models")
        self.model_a, self.model_b = model_a.to(device), model_b.to(device)
        self.optimizer_a, self.optimizer_b = optimizer_a, optimizer_b
        self.scheduler_a, self.scheduler_b = scheduler_a, scheduler_b
        self.config, self.device = config, device
        self.state = DivideMixState()

    @staticmethod
    def confidence_penalty(logits: Tensor) -> Tensor:
        probabilities = torch.softmax(logits, dim=1)
        return torch.mean(torch.sum(probabilities * torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)), dim=1))

    def warmup_step(self, peer: str, inputs: Tensor, targets: Tensor, *, asymmetric: bool) -> dict[str, float]:
        model, optimizer = self._peer(peer)
        model.train()
        logits = model(inputs.to(self.device))
        ce = torch.nn.functional.cross_entropy(logits, targets.to(self.device))
        penalty = self.confidence_penalty(logits) if asymmetric else logits.new_zeros(())
        objective = ce + self.config.confidence_penalty_weight * penalty
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        self._increment(peer)
        return {"warmup_ce": float(ce.detach().item()), "confidence_penalty": float(penalty.detach().item()), "objective": float(objective.detach().item())}

    def train_peer_step(self, peer: str, labeled_views: tuple[Tensor, ...], unlabeled_views: tuple[Tensor, ...], noisy_targets: Tensor, clean_probability: Tensor, *, epoch: int, batch_index: int, num_batches: int, rng: np.random.Generator | None = None) -> dict[str, float]:
        model, optimizer = self._peer(peer)
        other = self.model_b if peer == "a" else self.model_a
        model.train()
        other.eval()
        labeled_views = tuple(value.to(self.device) for value in labeled_views)
        unlabeled_views = tuple(value.to(self.device) for value in unlabeled_views)
        noisy_targets = noisy_targets.to(self.device)
        clean_probability = clean_probability.to(self.device, dtype=torch.float32)
        refined = co_refine(model, labeled_views, noisy_targets, clean_probability, self.config.temperature)
        guessed = co_guess(model, other, unlabeled_views, self.config.temperature)
        mixed = mixmatch_mixup(labeled_views, unlabeled_views, refined, guessed, alpha=self.config.mixup_alpha, rng=rng)
        logits = model(mixed.inputs)
        labeled_logits = logits[:mixed.labeled_count]
        unlabeled_logits = logits[mixed.labeled_count:]
        progress = self.config.warmup_epochs + epoch + batch_index / max(1, num_batches)
        weight_u = unsupervised_weight(self.config.lambda_u, progress, self.config.warmup_epochs, self.config.rampup_epochs)
        objective, metrics = dividemix_objective(labeled_logits, mixed.targets[:mixed.labeled_count], unlabeled_logits, mixed.targets[mixed.labeled_count:], logits, lambda_u=weight_u, lambda_r=self.config.lambda_r)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        self._increment(peer)
        metrics["mix_lambda"] = mixed.mix_lambda
        return metrics

    def _peer(self, peer: str):
        if peer == "a":
            return self.model_a, self.optimizer_a
        if peer == "b":
            return self.model_b, self.optimizer_b
        raise ValueError("DivideMix peer must be 'a' or 'b'")

    def _increment(self, peer: str) -> None:
        if peer == "a": self.state.optimizer_steps_a += 1
        else: self.state.optimizer_steps_b += 1

    def step_schedulers(self) -> None:
        if self.scheduler_a is not None: self.scheduler_a.step()
        if self.scheduler_b is not None: self.scheduler_b.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "method_identity": "dividemix",
            "config_identity_hash": self.config.identity_hash,
            "model": {"a": self.model_a.state_dict(), "b": self.model_b.state_dict()},
            "optimizer": {"a": self.optimizer_a.state_dict(), "b": self.optimizer_b.state_dict()},
            "schedulers": {"a": None if self.scheduler_a is None else self.scheduler_a.state_dict(), "b": None if self.scheduler_b is None else self.scheduler_b.state_dict()},
            "dividemix_state": self.state.state_dict(),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value.get("method_identity") != "dividemix" or value.get("config_identity_hash") != self.config.identity_hash:
            raise ValueError("DivideMix checkpoint method/config identity mismatch")
        for peer, model in (("a", self.model_a), ("b", self.model_b)):
            model.load_state_dict(value["model"][peer])
        for peer, optimizer in (("a", self.optimizer_a), ("b", self.optimizer_b)):
            optimizer.load_state_dict(value["optimizer"][peer])
            for state in optimizer.state.values():
                for key, item in state.items():
                    if torch.is_tensor(item): state[key] = item.to(self.device)
        for peer, scheduler in (("a", self.scheduler_a), ("b", self.scheduler_b)):
            saved = value["schedulers"][peer]
            if (scheduler is None) != (saved is None):
                raise ValueError(f"DivideMix scheduler {peer} configuration changed")
            if scheduler is not None: scheduler.load_state_dict(saved)
        self.state = DivideMixState.from_state_dict(value["dividemix_state"])
