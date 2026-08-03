from __future__ import annotations

"""CWD risk consumption separated from its statistic estimator."""

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from lnl_toolbox.losses.torch_losses import loss_for_all_targets
from lnl_toolbox.noise.statistics import StatisticArtifact
from lnl_toolbox.models.feature_output import classifier_parameters, validate_objective


class CWDUnbiasedRisk:
    name = "cwd"
    requires_transition = False

    def __init__(self, statistic: StatisticArtifact | None = None, label_flip_matrix=None) -> None:
        self.statistic = statistic
        self.label_flip_matrix = None if label_flip_matrix is None else torch.as_tensor(label_flip_matrix, dtype=torch.float64)

    def per_sample_risk(self, *, logits: Tensor, noisy_targets: Tensor, base_loss: nn.Module, transition: Any = None) -> Tensor:
        if logits.ndim != 2 or noisy_targets.shape != (logits.shape[0],):
            raise ValueError("CWD inputs must have logits [B, C] and targets [B]")
        matrix = self.label_flip_matrix
        if matrix is None and self.statistic is not None:
            value = self.statistic.metadata.get("label_flip_matrix")
            if value is not None:
                matrix = torch.as_tensor(value, dtype=logits.dtype, device=logits.device)
        if matrix is None:
            matrix = torch.eye(logits.shape[1], dtype=logits.dtype, device=logits.device)
        matrix = matrix.to(device=logits.device, dtype=logits.dtype)
        if matrix.shape != (logits.shape[1], logits.shape[1]):
            raise ValueError("CWD label-flip matrix does not match logits classes")
        losses = loss_for_all_targets(base_loss, logits)
        corrected = torch.linalg.solve(matrix, losses.transpose(0, 1)).transpose(0, 1)
        return corrected.gather(1, noisy_targets[:, None].long()).squeeze(1)


CWD = CWDUnbiasedRisk


class CWDGlobalObjective:
    """CWD's statistic-reconstructed squared-risk objective.

    The clean-risk expansion is evaluated from recovered class centroids and
    priors; virtual samples are never materialized.
    """

    name = "cwd"
    requires_features = True

    def __init__(
        self,
        statistic: StatisticArtifact | None = None,
        *,
        loss: str = "squared",
        constant: float = 1.0,
    ) -> None:
        if str(loss).strip().lower() != "squared":
            raise ValueError("CWD currently supports the paper's squared objective")
        if not torch.isfinite(torch.tensor(float(constant))):
            raise ValueError("CWD objective constant must be finite")
        self.statistic = statistic
        self.loss = "squared"
        self.constant = float(constant)

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
            raise ValueError("CWD objective requires a statistic artifact")
        if features.ndim != 2:
            raise ValueError("CWD features must have shape [B, D]")
        centroids = torch.as_tensor(
            self.statistic.values.copy(),
            dtype=features.dtype,
            device=features.device,
        )
        if centroids.ndim != 2:
            raise ValueError("CWD centroids must have shape [C, D]")
        weight, bias = classifier_parameters(model)
        weight = weight.to(device=features.device, dtype=features.dtype)
        if weight.shape != centroids.shape:
            raise ValueError(
                "CWD classifier weight and centroid dimensions do not match: "
                f"weight={tuple(weight.shape)}, centroids={tuple(centroids.shape)}"
            )
        if bias is not None:
            bias = bias.to(device=features.device, dtype=features.dtype)
        scores = features @ weight.transpose(0, 1)
        if bias is not None:
            scores = scores + bias
        quadratic = scores.square().sum(dim=1).mean()
        prior_values = self.statistic.metadata.get("class_prior")
        if prior_values is None:
            prior = torch.full(
                (centroids.shape[0],),
                1.0 / centroids.shape[0],
                dtype=features.dtype,
                device=features.device,
            )
        else:
            prior = torch.as_tensor(prior_values, dtype=features.dtype, device=features.device)
        if prior.shape != (centroids.shape[0],) or not torch.isfinite(prior).all() or bool((prior < 0).any()):
            raise ValueError("CWD class_prior must be a finite non-negative [C] vector")
        prior = prior / prior.sum().clamp_min(torch.finfo(prior.dtype).tiny)
        cross = (weight * centroids).sum()
        if bias is not None:
            cross = cross + (bias * prior).sum()
        return validate_objective(self.constant + quadratic - 2.0 * cross)

    def bind_statistic(self, statistic: StatisticArtifact) -> None:
        if not isinstance(statistic, StatisticArtifact):
            raise TypeError("CWD objective requires a StatisticArtifact")
        self.statistic = statistic

    def state_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "constant": self.constant,
            "statistic_hash": None if self.statistic is None else self.statistic.artifact_hash,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if str(state.get("loss", self.loss)) != self.loss or float(state.get("constant", self.constant)) != self.constant:
            raise ValueError("CWD objective configuration mismatch")
        expected = state.get("statistic_hash")
        actual = None if self.statistic is None else self.statistic.artifact_hash
        if expected != actual:
            raise ValueError("CWD statistic artifact hash mismatch")


__all__ = ["CWD", "CWDGlobalObjective", "CWDUnbiasedRisk"]
