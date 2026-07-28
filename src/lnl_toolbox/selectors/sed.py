from __future__ import annotations

"""SED selection kept independent from the FINE regularizer."""

import torch

from .base import SelectionInput, SelectionResult, validate_selection_input


class SEDSelector:
    """Select samples whose detached score is below a configurable threshold."""

    def __init__(self, threshold: float = 0.5, keep_rate: float | None = None) -> None:
        if threshold < 0.0:
            raise ValueError("SED threshold must be non-negative")
        if keep_rate is not None and not 0.0 < keep_rate <= 1.0:
            raise ValueError("SED keep_rate must be in (0, 1]")
        self.threshold = float(threshold)
        self.keep_rate = None if keep_rate is None else float(keep_rate)

    def select(self, selection_input: SelectionInput) -> SelectionResult:
        batch = validate_selection_input(selection_input)
        scores = selection_input.scores
        mask = scores <= self.threshold
        if self.keep_rate is not None:
            count = max(1, int(torch.ceil(torch.tensor(batch * self.keep_rate)).item()))
            order = torch.argsort(scores, stable=True)
            mask = torch.zeros_like(mask)
            mask[order[:count]] = True
        if not bool(mask.any()):
            mask[torch.argmin(scores)] = True
        return SelectionResult(mask, {"selected_samples": float(mask.sum()), "selected_ratio": float(mask.float().mean())})


__all__ = ["SEDSelector"]
