from __future__ import annotations

"""Reference supervised algorithm used to verify the shared training path."""

from typing import Any

import torch
from torch import nn

from lnl_toolbox.core import Batch, ExperimentContext, RunState, StepResult
from lnl_toolbox.losses.torch_losses import validate_per_sample_loss
from lnl_toolbox.selectors import (
    AllSelector,
    SelectionInput,
    Selector,
)
from lnl_toolbox.treatments import (
    ReductionSpec,
    SelectorContributionAdapter,
    reduce_per_sample_loss,
)


class SupervisedClassificationAlgorithm:
    """Train one classifier with a standard loss and optimizer.

    This is the clean-label baseline and a minimal lifecycle example. Robust
    LNL algorithms should implement the same public hooks.
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 loss: nn.Module, device: torch.device,
                 selector: Selector | None = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.selector = selector or AllSelector()
        self.contribution_adapter = SelectorContributionAdapter(self.selector)
        self.reduction = ReductionSpec()

    def setup(self, context: ExperimentContext) -> None:
        self.model.to(self.device)
        self.loss.to(self.device)

    def on_run_start(self, state: RunState) -> None:
        pass

    def on_cycle_start(self, state: RunState) -> None:
        state.phase = "train"
        self.model.train()

    def step(self, batch: Batch, state: RunState) -> StepResult:
        """Perform one optimization step and return batch-level metrics."""
        inputs = batch.payload["input"].to(self.device, non_blocking=True)
        targets = batch.payload["target"].to(self.device, non_blocking=True)
        sample_indices = torch.as_tensor(
            batch.payload["index"], dtype=torch.long, device=self.device
        )
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        per_sample_loss = validate_per_sample_loss(
            self.loss(logits, targets), int(targets.numel())
        )
        contribution = self.contribution_adapter.resolve(SelectionInput(
            scores=per_sample_loss.detach(),
            sample_indices=sample_indices,
            metadata={"epoch": state.cycle},
        ))
        selected_mask = contribution.selected_mask
        loss = reduce_per_sample_loss(
            per_sample_loss,
            contribution,
            self.reduction,
        )
        loss.backward()
        self.optimizer.step()
        count = int(targets.numel())
        selected_count = int(selected_mask.sum().item())
        correct = int((logits.argmax(1) == targets).sum().item())
        state.step += 1
        return StepResult(metrics={
            "loss": float(loss.detach().item()),
            "all_sample_loss": float(per_sample_loss.detach().mean().item()),
            "accuracy": correct / count,
            "samples": float(count),
            "selected_samples": float(selected_count),
            "selected_ratio": selected_count / count,
        })

    def on_cycle_end(self, state: RunState) -> StepResult:
        return StepResult()

    def on_run_end(self, state: RunState) -> StepResult:
        return StepResult()

    def state_dict(self) -> dict[str, Any]:
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore model and optimizer state, moving optimizer tensors as needed."""
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)

    def close(self) -> None:
        pass
