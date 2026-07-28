from __future__ import annotations

"""Reference supervised algorithm used to verify the shared training path."""

from typing import Any, Mapping, Protocol, runtime_checkable

import torch
from torch import nn

from lnl_toolbox.core import Batch, ExperimentContext, RunState, StepResult
from lnl_toolbox.core.result import PseudoLabelResult, SoftTargetResult
from lnl_toolbox.core.targets import LabelProvider, TargetInput
from lnl_toolbox.losses.torch_losses import loss_for_all_targets, validate_per_sample_loss
from lnl_toolbox.noise.transition import TransitionProvider
from lnl_toolbox.algorithms.transition_risk import RiskCorrector
from lnl_toolbox.core.objectives import ObjectiveConsumer, ObjectiveResult
from lnl_toolbox.models.feature_output import FeatureOutput, forward_with_features, validate_objective
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
    SupervisedWeightInput,
    WeightContributionAdapter,
    WeightProvider,
    reduce_per_sample_loss,
)


@runtime_checkable
class Regularizer(Protocol):
    """Generic noisy-only objective added after sample treatment."""

    def __call__(
        self,
        logits: torch.Tensor,
        noisy_targets: torch.Tensor,
        *,
        sample_indices: torch.Tensor,
        selected_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
        metadata: Mapping[str, Any],
    ) -> torch.Tensor:
        ...


def _call_regularizer(
    regularizer: Any,
    *,
    logits: torch.Tensor,
    noisy_targets: torch.Tensor,
    sample_indices: torch.Tensor,
    selected_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
    metadata: Mapping[str, Any],
) -> torch.Tensor:
    """Call either the common callable or compute-style regularizer contract."""

    arguments = {
        "logits": logits,
        "noisy_targets": noisy_targets,
        "sample_indices": sample_indices,
        "selected_mask": selected_mask,
        "rejected_mask": rejected_mask,
        "metadata": metadata,
    }
    compute = getattr(regularizer, "compute", None)
    value = compute(**arguments) if callable(compute) else regularizer(**arguments)
    if not torch.is_tensor(value) or value.ndim not in (0, 1):
        raise ValueError("regularizer must return a scalar or per-sample tensor")
    if value.ndim == 1:
        if value.shape != noisy_targets.shape:
            raise ValueError("regularizer per-sample output must align with the batch")
        value = value.mean()
    if not torch.isfinite(value).all():
        raise ValueError("regularizer returned a non-finite value")
    return value


def _validate_target_sample_indices(
    target_result: SoftTargetResult | PseudoLabelResult,
    batch_sample_indices: torch.Tensor,
) -> None:
    """Reject target-provider output that is not exactly batch-order aligned."""

    result_indices = target_result.sample_indices
    if result_indices.shape != batch_sample_indices.shape:
        raise ValueError(
            "target provider sample index length mismatch: "
            f"batch shape={tuple(batch_sample_indices.shape)}, "
            f"provider shape={tuple(result_indices.shape)}"
        )
    if result_indices.device != batch_sample_indices.device:
        raise ValueError(
            "target provider sample index device mismatch: "
            f"batch device={batch_sample_indices.device}, "
            f"provider device={result_indices.device}"
        )
    comparable_result = result_indices.to(dtype=torch.int64)
    comparable_batch = batch_sample_indices.to(dtype=torch.int64)
    mismatch = torch.nonzero(
        comparable_result != comparable_batch,
        as_tuple=False,
    ).flatten()
    if mismatch.numel() == 0:
        return
    positions = mismatch[:8]
    summary = [
        {
            "position": int(position),
            "batch": int(comparable_batch[position].item()),
            "provider": int(comparable_result[position].item()),
        }
        for position in positions
    ]
    raise ValueError(
        "target provider sample indices are not exactly batch-order aligned; "
        f"mismatch_count={int(mismatch.numel())}, mismatches={summary}, "
        f"batch_indices={comparable_batch.detach().cpu().tolist()}, "
        f"provider_indices={comparable_result.detach().cpu().tolist()}"
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
                 weight_provider: WeightProvider[SupervisedWeightInput] | None = None,
                 target_provider: LabelProvider | None = None,
                 regularizer: Regularizer | None = None,
                 objective_consumer: ObjectiveConsumer | None = None,
                 regularizer_weight: float = 1.0) -> None:
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
        self.regularizer = regularizer
        self.objective_consumer = objective_consumer
        if not torch.isfinite(torch.tensor(float(regularizer_weight))):
            raise ValueError("regularizer_weight must be finite")
        self.regularizer_weight = float(regularizer_weight)
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
        hook = getattr(self.objective_consumer, "on_run_start", None)
        if callable(hook):
            hook(state)

    def on_cycle_start(self, state: RunState) -> None:
        state.phase = "train"
        self.model.train()
        hook = getattr(self.objective_consumer, "on_cycle_start", None)
        if callable(hook):
            hook(state)

    def step(self, batch: Batch, state: RunState) -> StepResult:
        """Perform one optimization step and return batch-level metrics."""
        inputs = batch.payload["input"].to(self.device, non_blocking=True)
        targets = batch.payload["target"].to(self.device, non_blocking=True)
        sample_indices = torch.as_tensor(
            batch.payload["index"], dtype=torch.long, device=self.device
        )
        model_output: FeatureOutput | None = None
        if self.objective_consumer is None:
            logits = self.model(inputs)
        else:
            model_output = forward_with_features(self.model, inputs)
            logits = model_output.logits
        target_mask = None
        target_result = None
        if self.target_provider is not None:
            target_result = self.target_provider.resolve(TargetInput(
                logits=logits.detach(),
                noisy_targets=targets.detach(),
                sample_indices=sample_indices.detach(),
                metadata={"epoch": state.cycle},
            ))
            if isinstance(target_result, (SoftTargetResult, PseudoLabelResult)):
                _validate_target_sample_indices(target_result, sample_indices)
        if self.risk_corrector is not None and target_result is not None:
            raise ValueError("risk correction and target replacement cannot be combined implicitly")
        if self.risk_corrector is not None:
            requires_transition = bool(
                getattr(self.risk_corrector, "requires_transition", True)
            )
            if self.transition is None and requires_transition:
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
            weight_input = SupervisedWeightInput(
                logits=logits.detach(),
                noisy_targets=targets.detach(),
                sample_indices=sample_indices.detach(),
                per_sample_loss=per_sample_loss.detach(),
                metadata={"epoch": state.cycle},
            )
            weights = self.weight_adapter.resolve(weight_input)
            contribution = ContributionResult(
                selected_mask=contribution.selected_mask & weights.selected_mask,
                sample_weights=contribution.sample_weights * weights.sample_weights,
                metrics={**contribution.metrics, **weights.metrics},
                selection_mask=contribution.selection_mask,
            )
        if target_mask is not None:
            contribution = ContributionResult(
                selected_mask=contribution.selected_mask & target_mask.to(device=self.device),
                sample_weights=contribution.sample_weights,
                metrics=contribution.metrics,
                selection_mask=contribution.selection_mask,
            )
        selected_mask = contribution.selected_mask
        selection_mask = contribution.selection_mask
        if selection_mask is None:
            selection_mask = selected_mask
        rejected_mask = ~selection_mask
        objective_result: ObjectiveResult | None = None
        reporting_loss: torch.Tensor | None = None
        objective_metrics: dict[str, float] = {}
        if self.objective_consumer is None:
            loss = reduce_per_sample_loss(
                per_sample_loss,
                contribution,
                self.reduction,
            )
        else:
            assert model_output is not None
            computed = self.objective_consumer.compute(
                model=self.model,
                logits=logits,
                features=model_output.features,
                noisy_targets=targets,
                sample_indices=sample_indices,
                base_loss=self.loss,
                metadata={"epoch": state.cycle},
            )
            if isinstance(computed, ObjectiveResult):
                objective_result = computed
                loss = validate_objective(computed.objective)
                if computed.selected_mask is not None:
                    objective_mask = computed.selected_mask
                    if (
                        not torch.is_tensor(objective_mask)
                        or objective_mask.shape != targets.shape
                        or objective_mask.dtype != torch.bool
                        or objective_mask.device != targets.device
                    ):
                        raise ValueError(
                            "objective selected_mask must be a boolean [B] "
                            "tensor on the target device"
                        )
                    selected_mask = objective_mask
                    selection_mask = objective_mask
                    rejected_mask = ~objective_mask
                if computed.reporting_loss is not None:
                    reporting_loss = validate_objective(
                        computed.reporting_loss
                    )
                for name, value in computed.metrics.items():
                    scalar = float(value)
                    if not torch.isfinite(torch.tensor(scalar)):
                        raise ValueError(
                            f"objective metric {name!r} must be finite"
                        )
                    objective_metrics[str(name)] = scalar
            else:
                loss = validate_objective(computed)
        regularizer_loss = None
        if self.regularizer is not None:
            regularizer_loss = _call_regularizer(
                self.regularizer,
                logits=logits,
                noisy_targets=targets,
                sample_indices=sample_indices,
                selected_mask=selected_mask,
                rejected_mask=rejected_mask,
                metadata={"epoch": state.cycle},
            )
            loss = loss + self.regularizer_weight * regularizer_loss
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
            "loss": float(
                (reporting_loss if reporting_loss is not None else loss)
                .detach()
                .item()
            ),
            "optimization_loss": float(loss.detach().item()),
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
        metrics.update({
            f"update_objective_{key}": value
            for key, value in objective_metrics.items()
        })
        if regularizer_loss is not None:
            metrics["regularizer_loss"] = float(regularizer_loss.detach().item())
        return StepResult(metrics=metrics)

    def on_cycle_end(self, state: RunState) -> StepResult:
        hook = getattr(self.objective_consumer, "on_cycle_end", None)
        if callable(hook):
            hook(state)
        return StepResult()

    def on_run_end(self, state: RunState) -> StepResult:
        hook = getattr(self.objective_consumer, "on_run_end", None)
        if callable(hook):
            hook(state)
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
        if self.regularizer is not None:
            state_dict = getattr(self.regularizer, "state_dict", None)
            if callable(state_dict):
                state["regularizer"] = dict(state_dict())
            state["regularizer_weight"] = self.regularizer_weight
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
        if "regularizer_weight" in state:
            if float(state["regularizer_weight"]) != self.regularizer_weight:
                raise ValueError("regularizer configuration mismatch")
        if self.regularizer is not None and state.get("regularizer") is not None:
            loader = getattr(self.regularizer, "load_state_dict", None)
            if not callable(loader):
                raise ValueError("regularizer is not stateful")
            loader(state["regularizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)

    def close(self) -> None:
        pass
