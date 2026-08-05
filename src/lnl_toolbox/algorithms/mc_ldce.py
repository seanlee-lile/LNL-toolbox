from __future__ import annotations

"""Global squared-risk objective for MC-LDCE."""

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from lnl_toolbox.models.feature_output import classifier_parameters, validate_objective
from lnl_toolbox.noise.statistics import StatisticArtifact


class MCLDCEObjective:
    name = "mc_ldce"
    requires_features = True

    def __init__(self, statistic: StatisticArtifact | None = None) -> None:
        self.statistic = statistic

    def bind_statistic(self, statistic: StatisticArtifact) -> None:
        if statistic.estimator != self.name:
            raise ValueError("MC-LDCE requires an MC-LDCE statistic artifact")
        self.statistic = statistic

    def compute(
        self,
        *,
        model: nn.Module,
        logits: Tensor,
        features: Tensor,
        noisy_targets: Tensor,
        sample_indices: Tensor,
        base_loss: nn.Module,
        metadata: Mapping[str, Any],
    ) -> Tensor:
        del logits, noisy_targets, sample_indices, base_loss, metadata
        if self.statistic is None:
            raise ValueError("MC-LDCE objective requires a statistic artifact")
        weight, bias = classifier_parameters(model)
        centroid = torch.as_tensor(
            self.statistic.values.copy(), dtype=features.dtype, device=features.device
        )
        if weight.shape != centroid.shape:
            raise ValueError("classifier and centroid dimensions differ")
        scores = features @ weight.transpose(0, 1)
        if bias is not None:
            scores = scores + bias.to(features)
        quadratic = scores.square().sum(dim=1).mean()
        cross = (weight.to(features) * centroid).sum()
        if bias is not None:
            prior = torch.as_tensor(
                self.statistic.metadata["clean_class_prior"],
                dtype=features.dtype,
                device=features.device,
            )
            cross = cross + (bias.to(features) * prior).sum()
        return validate_objective(1.0 + quadratic - 2.0 * cross)

    def state_dict(self) -> dict[str, Any]:
        return {
            "statistic_hash": None
            if self.statistic is None
            else self.statistic.artifact_hash
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        actual = None if self.statistic is None else self.statistic.artifact_hash
        if state.get("statistic_hash") != actual:
            raise ValueError("MC-LDCE statistic artifact hash mismatch")


__all__ = ["MCLDCEObjective"]
