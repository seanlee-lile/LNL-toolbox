from __future__ import annotations

"""FINE active-forgetting and noise-suppression objectives."""

from typing import Any, Mapping

import torch
from torch import Tensor


class ActiveForgettingRegularizer:
    """Official negative-cross-entropy forgetting term on noisy samples."""

    def __init__(self, probability_floor: float = 1e-8) -> None:
        if probability_floor <= 0.0:
            raise ValueError("probability_floor must be positive")
        self.probability_floor = float(probability_floor)

    def __call__(
        self,
        logits: Tensor,
        noisy_targets: Tensor,
        *,
        selected_mask: Tensor | None = None,
        rejected_mask: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        values = torch.log_softmax(logits, dim=1).gather(
            1, noisy_targets.long()[:, None]
        ).squeeze(1)
        mask = rejected_mask if rejected_mask is not None else (
            torch.ones_like(values, dtype=torch.bool) if selected_mask is None else ~selected_mask
        )
        return values[mask].mean() if bool(mask.any()) else values.sum() * 0.0


class FINERegularizer:
    """Official FINE MU/NL terms applied after SED's second-stage mask."""

    name = "fine"

    def __init__(self, beta: float = 0.1, gamma: float = 0.002, probability_floor: float = 1e-7, seed: int = 0) -> None:
        if beta < 0.0 or gamma < 0.0:
            raise ValueError("FINE beta and gamma must be non-negative")
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.probability_floor = float(probability_floor)
        if self.probability_floor <= 0.0:
            raise ValueError("probability_floor must be positive")
        self.seed = int(seed)  # retained in state for configuration compatibility

    def __call__(
        self,
        logits: Tensor,
        noisy_targets: Tensor,
        *,
        selected_mask: Tensor | None = None,
        rejected_mask: Tensor | None = None,
        pseudo_labels: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        mask = rejected_mask if rejected_mask is not None else (
            torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
            if selected_mask is None else ~selected_mask
        )
        if pseudo_labels is None:
            pseudo_labels = logits.detach().argmax(dim=1)
        if pseudo_labels.shape != noisy_targets.shape:
            raise ValueError("FINE pseudo_labels must have shape [B]")
        mask = mask & pseudo_labels.to(noisy_targets.device).long().ne(
            noisy_targets.long()
        )
        if not bool(mask.any()):
            return logits.sum() * 0.0
        probabilities = torch.softmax(logits, dim=1)
        p_target = probabilities.gather(1, noisy_targets.long()[:, None]).squeeze(1)
        suppression = -torch.log((1.0 - p_target).clamp_min(self.probability_floor))
        forgetting = torch.log_softmax(logits, dim=1).gather(
            1, noisy_targets.long()[:, None]
        ).squeeze(1)
        return self.beta * suppression[mask].mean() + self.gamma * forgetting[mask].mean()

    compute = __call__

    def state_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "gamma": self.gamma,
            "probability_floor": self.probability_floor,
            "seed": self.seed,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name in ("beta", "gamma", "probability_floor"):
            if name in state and float(state[name]) != getattr(self, name):
                raise ValueError("FINE configuration mismatch")
        if "seed" in state and int(state["seed"]) != self.seed:
            raise ValueError("FINE configuration mismatch")


AFMU = ActiveForgettingRegularizer
NSNL = FINERegularizer

__all__ = ["AFMU", "ActiveForgettingRegularizer", "FINERegularizer", "NSNL"]
