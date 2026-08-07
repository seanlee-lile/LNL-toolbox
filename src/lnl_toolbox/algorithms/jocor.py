from __future__ import annotations

"""JoCoR joint training over a reusable two-model group."""

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lnl_toolbox.algorithms.multi_model import ModelGroup
from lnl_toolbox.core import Batch, ExperimentContext, RunState, StepResult
from lnl_toolbox.losses.torch_losses import validate_per_sample_loss
from lnl_toolbox.selectors import SelectionInput, Selector
from lnl_toolbox.selectors.base import validate_selection_result


def symmetric_kl_per_sample(logits_1: Tensor, logits_2: Tensor) -> Tensor:
    """Return KL(p1||p2) + KL(p2||p1) for every sample."""

    if logits_1.ndim != 2 or logits_1.shape != logits_2.shape:
        raise ValueError("JoCoR logits must have matching shape [B, C]")
    log_1 = F.log_softmax(logits_1, dim=1)
    log_2 = F.log_softmax(logits_2, dim=1)
    prob_1 = log_1.exp()
    prob_2 = log_2.exp()
    result = (
        F.kl_div(log_1, prob_2, reduction="none").sum(dim=1)
        + F.kl_div(log_2, prob_1, reduction="none").sum(dim=1)
    )
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("JoCoR agreement produced non-finite values")
    return result


def jocor_joint_scores(
    loss_1: Tensor,
    loss_2: Tensor,
    logits_1: Tensor,
    logits_2: Tensor,
    lambda_: float,
) -> Tensor:
    """Build the per-sample objective used for both selection and updating."""

    batch_size = int(logits_1.shape[0])
    first = validate_per_sample_loss(loss_1, batch_size)
    second = validate_per_sample_loss(loss_2, batch_size)
    coefficient = float(lambda_)
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("JoCoR lambda must be in [0, 1]")
    result = (
        (1.0 - coefficient) * (first + second)
        + coefficient * symmetric_kl_per_sample(logits_1, logits_2)
    )
    return validate_per_sample_loss(result, batch_size)


class JoCoRAlgorithm:
    """Two networks sharing one joint small-loss set and optimizer step."""

    def __init__(
        self,
        models: ModelGroup,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module,
        selector: Selector,
        device: torch.device,
        *,
        lambda_: float,
    ) -> None:
        if len(models.models) != 2:
            raise ValueError("JoCoR requires exactly two models")
        coefficient = float(lambda_)
        if not 0.0 <= coefficient <= 1.0:
            raise ValueError("JoCoR lambda must be in [0, 1]")
        self.models = models
        self.optimizer = optimizer
        self.loss = loss
        self.selector = selector
        self.device = device
        self.lambda_ = coefficient

    def setup(self, context: ExperimentContext) -> None:
        self.models.to(self.device)
        self.loss.to(self.device)

    def on_run_start(self, state: RunState) -> None:
        pass

    def on_cycle_start(self, state: RunState) -> None:
        state.phase = "train"
        self.models.train()

    def step(self, batch: Batch, state: RunState) -> StepResult:
        inputs = batch.payload["input"].to(self.device, non_blocking=True)
        targets = batch.payload["target"].to(self.device, non_blocking=True)
        indices = torch.as_tensor(
            batch.payload["index"], dtype=torch.long, device=self.device
        )
        outputs = self.models.logits(inputs)
        names = tuple(outputs)
        logits_1, logits_2 = outputs[names[0]], outputs[names[1]]
        losses_1 = validate_per_sample_loss(
            self.loss(logits_1, targets), int(targets.numel())
        )
        losses_2 = validate_per_sample_loss(
            self.loss(logits_2, targets), int(targets.numel())
        )
        agreement = symmetric_kl_per_sample(logits_1, logits_2)
        joint = (
            (1.0 - self.lambda_) * (losses_1 + losses_2)
            + self.lambda_ * agreement
        )
        selection = self.selector.select(SelectionInput(
            scores=joint.detach(),
            sample_indices=indices,
            metadata={"epoch": state.cycle},
        ))
        selected = validate_selection_result(
            selection,
            batch_size=int(targets.numel()),
            device=joint.device,
        )
        objective = joint[selected].mean()
        self.optimizer.zero_grad(set_to_none=True)
        objective.backward()
        self.optimizer.step()

        ensemble = (logits_1 + logits_2) / 2.0
        count = int(targets.numel())
        state.step += 1
        metrics = {
            "loss": float(objective.detach().item()),
            "all_sample_loss": float(joint.detach().mean().item()),
            "supervised_loss": float(
                (losses_1 + losses_2).detach().mean().item()
            ),
            "agreement_loss": float(agreement.detach().mean().item()),
            "accuracy": float(
                (ensemble.argmax(1) == targets).sum().item() / count
            ),
            "model_1_accuracy": float(
                (logits_1.argmax(1) == targets).sum().item() / count
            ),
            "model_2_accuracy": float(
                (logits_2.argmax(1) == targets).sum().item() / count
            ),
            "samples": float(count),
            "selected_samples": float(selected.sum().item()),
            "selected_ratio": float(selected.float().mean().item()),
            **{str(key): float(value) for key, value in selection.metrics.items()},
        }
        return StepResult(metrics=metrics)

    def on_cycle_end(self, state: RunState) -> StepResult:
        return StepResult()

    def on_run_end(self, state: RunState) -> StepResult:
        return StepResult()

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.models.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lambda": self.lambda_,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if float(state.get("lambda", self.lambda_)) != self.lambda_:
            raise ValueError("JoCoR lambda changed across resume")
        self.models.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)

    def close(self) -> None:
        pass


__all__ = [
    "JoCoRAlgorithm",
    "jocor_joint_scores",
    "symmetric_kl_per_sample",
]
