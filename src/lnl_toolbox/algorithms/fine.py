from __future__ import annotations

"""Composable FINE active-forgetting and negative-learning regularizer."""

from typing import Any, Mapping

import torch
from torch import Tensor


class ActiveForgettingRegularizer:
    """AFMU term operating on the selected noisy-only batch."""

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
        probabilities = torch.softmax(logits, dim=1)
        p_target = probabilities.gather(1, noisy_targets.long()[:, None]).squeeze(1)
        values = -torch.log((1.0 - p_target).clamp_min(self.probability_floor))
        mask = rejected_mask if rejected_mask is not None else (
            torch.ones_like(values, dtype=torch.bool) if selected_mask is None else ~selected_mask
        )
        return values[mask].mean() if bool(mask.any()) else values.sum() * 0.0


class FINERegularizer:
    """FINE objective: active forgetting plus complementary negative learning."""

    name = "fine"

    def __init__(self, beta: float = 0.001, gamma: float = 0.1, probability_floor: float = 1e-8, seed: int = 0) -> None:
        if beta < 0.0 or gamma < 0.0:
            raise ValueError("FINE beta and gamma must be non-negative")
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.probability_floor = float(probability_floor)
        if self.probability_floor <= 0.0:
            raise ValueError("probability_floor must be positive")
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def complementary_labels(self, noisy_targets: Tensor, classes: int) -> Tensor:
        if classes < 2:
            raise ValueError("FINE requires at least two classes")
        choices = torch.arange(classes, device="cpu")
        result = torch.zeros((noisy_targets.numel(), classes), dtype=torch.bool)
        for row, target in enumerate(noisy_targets.detach().cpu().tolist()):
            allowed = choices[choices != int(target)]
            result[row, int(allowed[torch.randint(len(allowed), (1,), generator=self.generator)])] = True
        return result.to(device=noisy_targets.device)

    def __call__(
        self,
        logits: Tensor,
        noisy_targets: Tensor,
        *,
        selected_mask: Tensor | None = None,
        rejected_mask: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        mask = rejected_mask if rejected_mask is not None else (
            torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
            if selected_mask is None else ~selected_mask
        )
        if not bool(mask.any()):
            return logits.sum() * 0.0
        probabilities = torch.softmax(logits, dim=1)
        p_target = probabilities.gather(1, noisy_targets.long()[:, None]).squeeze(1)
        active = -torch.log((1.0 - p_target).clamp_min(self.probability_floor))
        complementary = self.complementary_labels(noisy_targets, logits.shape[1])
        p_negative = (probabilities * complementary.to(probabilities.dtype)).sum(dim=1)
        negative = -torch.log((1.0 - p_negative).clamp_min(self.probability_floor))
        active = active[mask].mean()
        return self.beta * active + self.gamma * negative[mask].mean()

    compute = __call__

    def state_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "gamma": self.gamma,
            "probability_floor": self.probability_floor,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name in ("beta", "gamma", "probability_floor"):
            if name in state and float(state[name]) != getattr(self, name):
                raise ValueError("FINE configuration mismatch")
        if "generator_state" in state:
            self.generator.set_state(state["generator_state"])


AFMU = ActiveForgettingRegularizer
NSNL = FINERegularizer

__all__ = ["AFMU", "ActiveForgettingRegularizer", "FINERegularizer", "NSNL"]
