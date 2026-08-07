from __future__ import annotations

"""Shared-covariance GDA and noisy-validation ensemble learning for PCSE."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .statistics import PCSEStatistics


@dataclass(frozen=True)
class GDALayer:
    name: str
    clean_priors: np.ndarray  # [C]
    means: np.ndarray  # [C,D]
    shared_covariance: np.ndarray  # [D,D], includes configured ridge
    covariance_ridge: float

    def __post_init__(self) -> None:
        priors = np.asarray(self.clean_priors, dtype=np.float64)
        means = np.asarray(self.means, dtype=np.float64)
        covariance = np.asarray(self.shared_covariance, dtype=np.float64)
        if means.ndim != 2 or priors.shape != (means.shape[0],):
            raise ValueError("GDA priors and means must have shapes [C] and [C,D]")
        if covariance.shape != (means.shape[1], means.shape[1]):
            raise ValueError("GDA shared covariance must have shape [D,D]")
        if (
            not np.isfinite(self.covariance_ridge)
            or self.covariance_ridge < 0.0
        ):
            raise ValueError("GDA covariance ridge must be finite and non-negative")
        if not (
            np.isfinite(priors).all()
            and np.isfinite(means).all()
            and np.isfinite(covariance).all()
        ):
            raise ValueError("GDA parameters must be finite")
        if (priors <= 0.0).any() or not np.isclose(priors.sum(), 1.0):
            raise ValueError("GDA clean priors must be positive and sum to one")
        if not np.allclose(covariance, covariance.T, atol=1e-9, rtol=1e-7):
            raise ValueError("GDA shared covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "GDA shared covariance is not positive definite"
            ) from exc
        object.__setattr__(self, "clean_priors", priors.copy())
        object.__setattr__(self, "means", means.copy())
        object.__setattr__(self, "shared_covariance", covariance.copy())
        object.__setattr__(
            self, "covariance_ridge", float(self.covariance_ridge)
        )

    def scores(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.means.shape[1]:
            raise ValueError("GDA features must have shape [N,D]")
        if not np.isfinite(values).all():
            raise ValueError("GDA features must be finite")
        precision_means = np.linalg.solve(
            self.shared_covariance, self.means.T
        ).T
        biases = (
            -0.5 * np.einsum("cd,cd->c", self.means, precision_means)
            + np.log(self.clean_priors)
        )
        result = values @ precision_means.T + biases[None, :]
        if not np.isfinite(result).all():
            raise ValueError("GDA discriminant scores are non-finite")
        return result

    def posterior(self, features: np.ndarray) -> np.ndarray:
        scores = self.scores(features)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        posterior = exponentials / exponentials.sum(axis=1, keepdims=True)
        if not np.isfinite(posterior).all():
            raise ValueError("GDA posterior is non-finite")
        return posterior


def fit_gda_layers(
    statistics: PCSEStatistics,
    *,
    covariance_ridge: float,
) -> tuple[GDALayer, ...]:
    if not np.isfinite(covariance_ridge) or covariance_ridge < 0.0:
        raise ValueError("GDA covariance ridge must be finite and non-negative")
    layers: list[GDALayer] = []
    for layer in statistics.layers:
        shared = np.einsum(
            "c,cij->ij",
            statistics.clean_priors,
            layer.clean_covariances,
        )
        shared = 0.5 * (shared + shared.T)
        smallest = float(np.linalg.eigvalsh(shared).min())
        if smallest < -1e-8:
            raise ValueError(
                f"PCSE layer {layer.name!r} recovered a non-PSD covariance: "
                f"minimum eigenvalue={smallest:.6g}"
            )
        regularized = shared + np.eye(shared.shape[0]) * covariance_ridge
        layers.append(
            GDALayer(
                name=layer.name,
                clean_priors=statistics.clean_priors,
                means=layer.clean_means,
                shared_covariance=regularized,
                covariance_ridge=float(covariance_ridge),
            )
        )
    return tuple(layers)


@dataclass
class GDAEnsemble:
    layers: tuple[GDALayer, ...]
    raw_weights: Tensor

    @property
    def weights(self) -> Tensor:
        return torch.softmax(self.raw_weights, dim=0)

    def posterior_from_layer_probabilities(
        self, probabilities: Tensor
    ) -> Tensor:
        if probabilities.ndim != 3:
            raise ValueError(
                "GDA ensemble probabilities must have shape [L,N,C]"
            )
        if probabilities.shape[0] != len(self.layers):
            raise ValueError("GDA ensemble layer count mismatch")
        if not torch.isfinite(probabilities).all():
            raise ValueError("GDA ensemble probabilities must be finite")
        return torch.einsum("l,lnc->nc", self.weights, probabilities)


def fit_ensemble_weights(
    layer_probabilities: np.ndarray,
    noisy_targets: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    raw_weights: Tensor | None = None,
    optimizer_state: dict | None = None,
) -> tuple[Tensor, torch.optim.Optimizer, list[float]]:
    """Optimize positive simplex weights on noisy-validation NLL."""

    probabilities = torch.as_tensor(layer_probabilities, dtype=torch.float64)
    targets = torch.as_tensor(noisy_targets, dtype=torch.long)
    if probabilities.ndim != 3:
        raise ValueError("layer probabilities must have shape [L,N,C]")
    layers, samples, classes = probabilities.shape
    if layers < 2 or samples == 0 or classes < 3:
        raise ValueError("PCSE ensemble requires L>=2, N>0 and C>=3")
    if targets.shape != (samples,) or targets.min() < 0 or targets.max() >= classes:
        raise ValueError("PCSE ensemble noisy targets must have shape [N]")
    if not torch.isfinite(probabilities).all() or bool(
        (probabilities < 0.0).any().item()
    ):
        raise ValueError("PCSE ensemble probabilities must be finite and non-negative")
    if not torch.allclose(
        probabilities.sum(dim=2),
        torch.ones((layers, samples), dtype=torch.float64),
        atol=1e-8,
        rtol=1e-6,
    ):
        raise ValueError("each PCSE layer posterior row must sum to one")
    if epochs < 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("PCSE ensemble optimization settings are invalid")
    parameter = (
        torch.zeros(layers, dtype=torch.float64, requires_grad=True)
        if raw_weights is None
        else raw_weights
    )
    if parameter.shape != (layers,) or not parameter.requires_grad:
        raise ValueError("PCSE ensemble raw weights must be trainable shape [L]")
    optimizer = torch.optim.Adam([parameter], lr=float(learning_rate))
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    losses: list[float] = []
    for _ in range(epochs):
        weights = torch.softmax(parameter, dim=0)
        mixture = torch.einsum("l,lnc->nc", weights, probabilities)
        observed = mixture.gather(1, targets[:, None]).squeeze(1)
        loss = -torch.log(observed.clamp_min(torch.finfo(torch.float64).tiny)).mean()
        if not torch.isfinite(loss):
            raise ValueError("PCSE ensemble NLL became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return parameter, optimizer, losses
