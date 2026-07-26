from __future__ import annotations

"""Reference supervised algorithm used to verify the shared training path."""

from typing import Any

import torch
from torch import nn

from lnl_toolbox.core import Batch, ExperimentContext, RunState, StepResult
from lnl_toolbox.core.result import PseudoLabelResult, SoftTargetResult
from lnl_toolbox.core.targets import LabelProvider, TargetInput
from lnl_toolbox.losses.torch_losses import loss_for_all_targets, validate_per_sample_loss
from lnl_toolbox.noise.transition import TransitionProvider
from lnl_toolbox.algorithms.transition_risk import RiskCorrector
from lnl_toolbox.algorithms.update_policy import (
    ParameterUpdateInput,
    ParameterUpdatePolicy,
    StandardUpdatePolicy,
    restore_update_policy,
    serialize_update_policy,
)
from lnl_toolbox.selectors import (
    AllSelector,
    SelectionInput,
    Selector,
)
from lnl_toolbox.treatments import (
    ContributionResult,
    ReductionSpec,
    SelectorContributionAdapter,
    WeightContributionAdapter,
    WeightInput,
    WeightProvider,
    reduce_per_sample_loss,
)


class SupervisedClassificationAlgorithm:
    """Train one classifier with a standard loss and optimizer.

    This is the clean-label baseline and a minimal lifecycle example. Robust
    LNL algorithms should implement the same public hooks.
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 loss: nn.Module, device: torch.device,
                 selector: Selector | None = None,
                 update_policy: ParameterUpdatePolicy | None = None,
                 risk_corrector: RiskCorrector | None = None,
                 transition: TransitionProvider | None = None,
                 weight_provider: WeightProvider[WeightInput] | None = None,
                 target_provider: LabelProvider | None = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.selector = selector or AllSelector()
        self.update_policy = update_policy or StandardUpdatePolicy()
        self.risk_corrector = risk_corrector
        self.transition = transition
        self.weight_provider = weight_provider
        self.target_provider = target_provider
        self.contribution_adapter = SelectorContributionAdapter(self.selector)
        self.weight_adapter = (
            None if weight_provider is None else WeightContributionAdapter(weight_provider)
        )
        self.reduction = ReductionSpec()

    def setup(self, context: ExperimentContext) -> None:
        self.model.to(self.device)
        self.loss.to(self.device)
        self.update_policy.setup(context)

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
        logits = self.model(inputs)
        target_mask = None
        target_result = None
        if self.target_provider is not None:
            target_result = self.target_provider.resolve(TargetInput(
                logits=logits.detach(),
                noisy_targets=targets.detach(),
                sample_indices=sample_indices.detach(),
                metadata={"epoch": state.cycle},
            ))
        if self.risk_corrector is not None and target_result is not None:
            raise ValueError("risk correction and target replacement cannot be combined implicitly")
        if self.risk_corrector is not None:
            if self.transition is None:
                raise ValueError("risk correction requires a transition provider")
            per_sample_loss = self.risk_corrector.per_sample_risk(
                logits=logits,
                noisy_targets=targets,
                base_loss=self.loss,
                transition=self.transition,
            )
        elif isinstance(target_result, SoftTargetResult):
            all_target_losses = loss_for_all_targets(self.loss, logits)
            per_sample_loss = (all_target_losses * target_result.targets).sum(dim=1)
            target_mask = target_result.selected_mask
        elif isinstance(target_result, PseudoLabelResult):
            per_sample_loss = self.loss(logits, target_result.labels)
            target_mask = target_result.selected_mask
        elif target_result is not None:
            raise TypeError("target provider returned an unsupported target result")
        else:
            per_sample_loss = self.loss(logits, targets)
        per_sample_loss = validate_per_sample_loss(
            per_sample_loss, int(targets.numel())
        )
        contribution = self.contribution_adapter.resolve(SelectionInput(
            scores=per_sample_loss.detach(),
            sample_indices=sample_indices,
            metadata={"epoch": state.cycle},
        ))
        if self.weight_adapter is not None:
            weight_input = WeightInput(
                logits=logits.detach(),
                noisy_targets=targets.detach(),
                sample_indices=sample_indices.detach(),
                per_sample_loss=per_sample_loss.detach(),
                posterior_probabilities=torch.softmax(logits.detach(), dim=1),
                metadata={"epoch": state.cycle},
            )
            weights = self.weight_adapter.resolve(weight_input)
            contribution = ContributionResult(
                selected_mask=contribution.selected_mask & weights.selected_mask,
                sample_weights=contribution.sample_weights * weights.sample_weights,
                metrics={**contribution.metrics, **weights.metrics},
            )
        if target_mask is not None:
            contribution = ContributionResult(
                selected_mask=contribution.selected_mask & target_mask.to(device=self.device),
                sample_weights=contribution.sample_weights,
                metrics=contribution.metrics,
            )
        selected_mask = contribution.selected_mask
        loss = reduce_per_sample_loss(
            per_sample_loss,
            contribution,
            self.reduction,
        )
        update_result = self.update_policy.update(ParameterUpdateInput(
            objective=loss,
            model=self.model,
            optimizer=self.optimizer,
            run_state=state,
        ))
        count = int(targets.numel())
        selected_count = int(selected_mask.sum().item())
        correct = int((logits.argmax(1) == targets).sum().item())
        state.step += 1
        metrics = {
            "loss": float(loss.detach().item()),
            "all_sample_loss": float(per_sample_loss.detach().mean().item()),
            "accuracy": correct / count,
            "samples": float(count),
            "selected_samples": float(selected_count),
            "selected_ratio": selected_count / count,
        }
        metrics.update({
            f"treatment_{key}": float(value)
            for key, value in contribution.metrics.items()
        })
        metrics.update(update_result.metrics)
        return StepResult(metrics=metrics)

    def on_cycle_end(self, state: RunState) -> StepResult:
        return StepResult()

    def on_run_end(self, state: RunState) -> StepResult:
        return StepResult()

    def state_dict(self) -> dict[str, Any]:
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "parameter_update_policy": serialize_update_policy(self.update_policy),
        }
        if self.weight_adapter is not None:
            weight_state = self.weight_adapter.state_dict()
            if weight_state:
                state["weight_provider"] = weight_state
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore model and optimizer state, moving optimizer tensors as needed."""
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        restore_update_policy(
            self.update_policy,
            state.get("parameter_update_policy"),
        )
        if self.weight_adapter is not None and state.get("weight_provider") is not None:
            self.weight_adapter.load_state_dict(state["weight_provider"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)

    def close(self) -> None:
        pass
