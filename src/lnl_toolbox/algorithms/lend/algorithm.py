from __future__ import annotations

"""Single-model online LEND batch algorithm."""

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from lnl_toolbox.core import Batch, ExperimentContext, RunState, StepResult
from lnl_toolbox.losses.torch_losses import validate_per_sample_loss
from lnl_toolbox.models.feature_output import forward_with_features

from .config import LENDConfig
from .dilution import author_release_propagation, dilute_labels
from .graph import build_author_release_graph, build_lend_similarity, normalize_lend_graph
from .history import LENDLabelHistory
from .selector import select_lend_samples
from .state import LENDState


class LENDAlgorithm:
    def __init__(self, *, model: nn.Module, optimizer: torch.optim.Optimizer,
                 loss: nn.Module, device: torch.device, method_config: LENDConfig,
                 canonical_global_indices: Tensor, num_classes: int) -> None:
        model_parameters = {id(value) for value in model.parameters()}
        optimizer_parameters = {
            id(value) for group in optimizer.param_groups for value in group["params"]
        }
        if model_parameters != optimizer_parameters:
            raise ValueError("LEND optimizer parameters must exactly match the model")
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.method_config = method_config
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError("LEND requires at least two classes")
        self.private_state = LENDState(
            LENDLabelHistory(canonical_global_indices, self.num_classes)
        )

    def setup(self, context: ExperimentContext) -> None:
        del context
        self.model.to(self.device)
        self.loss.to(self.device)

    def on_run_start(self, state: RunState) -> None:
        del state

    def on_cycle_start(self, state: RunState) -> None:
        state.phase = "training"
        self.model.train()

    def step(self, batch: Batch, state: RunState) -> StepResult:
        payload = batch.payload
        inputs = payload["input"].to(self.device, non_blocking=True)
        targets = payload["target"].to(self.device, non_blocking=True)
        indices = torch.as_tensor(payload["index"], dtype=torch.int64, device=self.device)
        count = int(targets.numel())
        if targets.ndim != 1 or indices.shape != targets.shape or count <= self.method_config.k:
            raise ValueError("LEND batch must align as [B] and satisfy B > k")
        if bool((targets < 0).any()) or bool((targets >= self.num_classes).any()):
            raise ValueError("LEND target is outside the configured class range")
        featured = forward_with_features(self.model, inputs)
        logits, features = featured.logits, featured.features
        if logits.shape != (count, self.num_classes) or features.ndim != 2 or features.shape[0] != count:
            raise ValueError("LEND model must return logits [B,C] and embeddings [B,D]")
        per_sample = validate_per_sample_loss(self.loss(logits, targets), count)
        detached_features = features.detach()
        onehot = functional.one_hot(targets, self.num_classes).to(features.dtype).detach()
        dilution_diagnostics: dict[str, float] = {}
        author = self.method_config.implementation_variant == "author_release"
        if author:
            adjacency, graph, _ = build_author_release_graph(
                detached_features, indices, k=self.method_config.k,
            )
            propagation_input = self.private_state.history.author_input(
                indices, onehot, epoch=state.cycle, beta=self.method_config.beta,
            )
            current = author_release_propagation(
                propagation_input, graph, diagnostics=dilution_diagnostics,
            )
            proposal = self.private_state.history.propose_author(indices, current, epoch=state.cycle)
        else:
            adjacency = build_lend_similarity(
                detached_features, indices, k=self.method_config.k,
                gamma=self.method_config.gamma, metric=self.method_config.metric,
                normalize_features=self.method_config.normalize_features,
            )
            graph = normalize_lend_graph(
                adjacency, zero_degree_policy=self.method_config.zero_degree_policy
            )
            current = dilute_labels(
                onehot, graph.to(dtype=onehot.dtype), alpha=self.method_config.alpha,
                steps=self.method_config.dilution_steps,
                policy=self.method_config.dilution_policy,
                tolerance=self.method_config.dilution_tolerance,
                max_steps=self.method_config.dilution_max_steps,
                diagnostics=dilution_diagnostics,
            )
            proposal = self.private_state.history.propose(
                indices, current, epoch=state.cycle, beta=self.method_config.beta
            )
        history_values = proposal.values.to(self.device, dtype=current.dtype)
        selected = select_lend_samples(targets, history_values)
        selected_count = int(selected.sum().item())
        # In the author path argmax agreement is diagnostic only, never a mask.
        weights = current.gather(1, targets[:, None]).squeeze(1).detach() if author else None
        selected_loss_sum = (weights * per_sample).sum() if author else per_sample[selected].sum()
        objective_value = 0.0
        if not author and not selected_count:
            self.private_state.empty_selection_batches += 1
        if author or selected_count or self.method_config.empty_batch == "zero_gradient":
            # Eq. (7) sums selected losses. batch_mean is an explicit variant.
            # An empty differentiable sum produces zero data gradients while
            # still allowing the optimizer's momentum/weight decay update.
            objective = selected_loss_sum
            if author or self.method_config.reduction == "batch_mean":
                objective = objective / count
            if not bool(torch.isfinite(objective)):
                raise ValueError("LEND selected objective is non-finite")
            self.optimizer.zero_grad(set_to_none=True)
            objective.backward()
            self.optimizer.step()
            self.private_state.optimizer_steps += 1
            objective_value = float(objective.detach().item())
        # Both empty-batch policies still commit the observed label evidence.
        self.private_state.history.commit(proposal)
        state.step += 1
        product = adjacency @ adjacency.T if author else adjacency.transpose(0, 1) @ adjacency
        degree = product.sum(dim=1)
        agreement = selected.float().mean()
        predictions = logits.detach().argmax(dim=1)
        metrics = {
            **{f"dilution_{key}": value for key, value in dilution_diagnostics.items()},
            "samples": float(count),
            "selected_samples": float(selected_count),
            "selected_ratio": float(agreement.item()),
            "selected_train_loss_sum": float(selected_loss_sum.detach().item()),
            "selected_train_objective": objective_value,
            "all_sample_noisy_ce_loss": float(per_sample.detach().mean().item()),
            "diluted_noisy_agreement": float(agreement.item()),
            "history_initialized_ratio": float(self.private_state.history.initialized.float().mean().item()),
            "graph_degree_min": float(degree.min().item()),
            "graph_degree_mean": float(degree.mean().item()),
            "graph_degree_max": float(degree.max().item()),
            "graph_nonzero_ratio": float((adjacency != 0).float().mean().item()),
            "accuracy": float((predictions == targets).float().mean().item()),
            "optimizer_steps": float(self.private_state.optimizer_steps),
            "empty_selection_batches": float(self.private_state.empty_selection_batches),
        }
        metadata = {
            "selected_mask": selected.detach().cpu(),
            "selected_indices": indices[selected].detach().cpu(),
            "diluted_labels": current.detach().cpu(),
            "history_values": history_values.detach().cpu(),
            "batch_indices": indices.detach().cpu(),
            "neighbor_mask": (adjacency != 0).detach().cpu(),
        }
        if author:
            for key in ("selected_samples", "selected_ratio", "selected_train_loss_sum",
                        "selected_train_objective", "dilution_fallback"):
                metrics.pop(key, None)
            metrics.update(
                weighted_loss_sum=float(selected_loss_sum.detach().item()),
                weighted_objective=objective_value,
                propagation_fallback=dilution_diagnostics["fallback"],
                history_used=float(state.cycle >= 10),
            )
            metadata["agreement_mask"] = metadata.pop("selected_mask")
            metadata["agreement_indices"] = metadata.pop("selected_indices")
            metadata["sample_weights"] = weights.cpu()
        return StepResult(metrics=metrics, metadata=metadata)

    def on_cycle_end(self, state: RunState) -> StepResult:
        del state
        return StepResult()

    def on_run_end(self, state: RunState) -> StepResult:
        state.phase = "completed"
        return StepResult()

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "method_identity": "lend",
            "method_config": self.method_config.identity(),
            "num_classes": self.num_classes,
            "lend_state": self.private_state.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("method_identity") != "lend":
            raise ValueError("checkpoint method identity is not LEND")
        saved_config = state.get("method_config")
        if not isinstance(saved_config, dict):
            raise ValueError("LEND checkpoint is missing method configuration")
        # Historical checkpoints have exactly the old paper semantics. Only
        # this newly introduced key may be defaulted; do not relax other checks.
        saved_config = dict(saved_config)
        saved_config.setdefault("implementation_variant", "paper_equation")
        if saved_config != self.method_config.identity():
            raise ValueError("LEND checkpoint method configuration changed")
        if int(state.get("num_classes", -1)) != self.num_classes:
            raise ValueError("LEND checkpoint class count changed")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        for values in self.optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(self.device)
        self.private_state.load_state_dict(state["lend_state"])

    def close(self) -> None:
        pass


__all__ = ["LENDAlgorithm"]
