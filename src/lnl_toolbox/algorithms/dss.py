from __future__ import annotations

"""Composable DSS objective consumer."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from lnl_toolbox.core.objectives import ObjectiveResult
from lnl_toolbox.selectors.dss import DSSSelectorState

from .masked_risk import candidate_masked_cross_entropy


class DSSObjective:
    """BASE + MDA + CCS with official batch-mean optimization semantics."""

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        total_epochs: int,
        *,
        warmup_epochs: int = 30,
        alpha: float = 0.10,
        prior_decay: float = 0.99,
        mda: bool = True,
        ccs: bool = True,
    ) -> None:
        self.selector = DSSSelectorState(
            num_samples,
            num_classes,
            total_epochs,
            warmup_epochs=warmup_epochs,
            alpha=alpha,
            prior_decay=prior_decay,
            mda=mda,
            ccs=ccs,
        )

    def on_cycle_start(self, state: Any) -> None:
        self.selector.on_cycle_start(int(state.cycle))

    def on_cycle_end(self, state: Any) -> None:
        self.selector.on_cycle_end(int(state.cycle))

    def compute(
        self,
        *,
        model: Any,
        logits: Tensor,
        features: Any,
        noisy_targets: Tensor,
        sample_indices: Tensor,
        base_loss: Any,
        metadata: Mapping[str, Any],
    ) -> ObjectiveResult:
        del model, features, base_loss
        epoch = int(metadata["epoch"])
        selected_cpu, excluded_cpu = self.selector.masks(
            sample_indices, noisy_targets
        )
        probabilities = logits.softmax(dim=1).detach()
        self.selector.observe(
            sample_indices, noisy_targets, probabilities, epoch
        )
        selected = selected_cpu.to(logits.device)
        excluded = excluded_cpu.to(logits.device)
        per_sample = candidate_masked_cross_entropy(
            logits, noisy_targets, excluded
        )
        objective = (
            per_sample * selected.to(per_sample.dtype)
        ).mean()
        if bool(selected.any()):
            reporting_loss = per_sample[selected].mean()
        else:
            reporting_loss = per_sample.sum() * 0.0
        return ObjectiveResult(
            objective=objective,
            selected_mask=selected,
            reporting_loss=reporting_loss,
            metrics={
                "excluded_class_ratio": float(excluded.float().mean().item()),
            },
        )

    def state_dict(self) -> dict[str, Any]:
        return {"selector": self.selector.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        selector_state = state.get("selector")
        if not isinstance(selector_state, Mapping):
            raise ValueError("DSS objective state is missing selector state")
        self.selector.load_state_dict(selector_state)


__all__ = ["DSSObjective"]
