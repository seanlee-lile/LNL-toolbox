from __future__ import annotations

"""Global squared-risk objective for MC-LDCE."""

from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from lnl_toolbox.models.feature_output import classifier_parameters, validate_objective
from lnl_toolbox.noise.statistics import StatisticArtifact


class MCLDCEObjective:
    name = "mc_ldce"
    requires_features = True

    def __init__(self, statistic: StatisticArtifact | None = None) -> None:
        self.statistic: StatisticArtifact | None = None
        self._inverse: np.ndarray | None = None
        self._inverse_tensors: dict[tuple[torch.device, torch.dtype], Tensor] = {}
        if statistic is not None:
            self.bind_statistic(statistic)

    def bind_statistic(self, statistic: StatisticArtifact) -> None:
        if statistic.estimator != self.name:
            raise ValueError("MC-LDCE requires an MC-LDCE statistic artifact")
        coefficient = np.asarray(statistic.metadata["coefficient_matrix"], dtype=np.float64)
        if (coefficient.ndim != 2 or coefficient.shape[0] != coefficient.shape[1]
                or not np.isfinite(coefficient).all()):
            raise ValueError("MC-LDCE coefficient matrix must be finite and square")
        inverse = statistic.metadata.get("coefficient_pseudoinverse")
        # Legacy artifacts do not store the inverse. Compute it once in the
        # same float64 precision and with the same cutoff as the estimator.
        inverse = (
            np.linalg.pinv(coefficient, rcond=float(statistic.metadata["rcond"]))
            if inverse is None else np.asarray(inverse, dtype=np.float64)
        )
        if inverse.shape != coefficient.shape or not np.isfinite(inverse).all():
            raise ValueError("MC-LDCE coefficient pseudoinverse is invalid")
        self.statistic = statistic
        self._inverse = inverse.copy()
        self._inverse_tensors.clear()

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
        del logits, sample_indices, base_loss, metadata
        if self.statistic is None:
            raise ValueError("MC-LDCE objective requires a statistic artifact")
        weight, bias = classifier_parameters(model)
        if bias is not None:
            raise ValueError("MC-LDCE Eq. (4.13) requires a bias-free classifier")
        # Eq. (4.12) is linear in the noisy centroid. A uniform minibatch
        # estimates the full-data risk at the CURRENT network parameters.
        # Keep this expression in autograd, using the same features (including
        # augmentation and dropout) as the quadratic term. M and its priors
        # are estimated globally, not from a possibly single-class minibatch.
        key = (features.device, features.dtype)
        inverse = self._inverse_tensors.get(key)
        if inverse is None:
            inverse = torch.tensor(self._inverse, dtype=features.dtype, device=features.device)
            self._inverse_tensors[key] = inverse
        labels = torch.nn.functional.one_hot(
            noisy_targets, num_classes=inverse.shape[0]
        ).to(features)
        centroid = (features.transpose(0, 1) @ labels / features.shape[0] @ inverse).T
        if weight.shape != centroid.shape:
            raise ValueError("classifier and centroid dimensions differ")
        scores = features @ weight.transpose(0, 1)
        quadratic = scores.square().sum(dim=1).mean()
        cross = (weight.to(features) * centroid).sum()
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
