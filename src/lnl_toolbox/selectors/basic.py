"""Basic stateless selectors for ordinary supervised training."""

from __future__ import annotations

import math

import torch

from .base import SelectionInput, SelectionResult, validate_selection_input
from .schedules import KeepRateSchedule, build_keep_rate_schedule


class AllSelector:
    """Keep every sample in the batch."""

    def select(self, selection_input: SelectionInput) -> SelectionResult:
        batch_size = validate_selection_input(selection_input)
        mask = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=selection_input.scores.device,
        )
        return SelectionResult(
            selected_mask=mask,
            metrics={
                "selected_samples": float(batch_size),
                "selected_ratio": 1.0,
            },
        )


class SmallLossSelector:
    """Keep the scheduled fraction of samples with the smallest scores."""

    def __init__(self, keep_rate: object, rounding: str = "ceil") -> None:
        self.schedule: KeepRateSchedule = build_keep_rate_schedule(keep_rate)
        self.rounding = str(rounding).strip().lower()
        if self.rounding not in {"ceil", "floor"}:
            raise ValueError("small-loss rounding must be 'ceil' or 'floor'")

    @property
    def keep_rate(self) -> float:
        """Return the epoch-zero rate for fixed-config compatibility."""

        return self.schedule.rate_at(0)

    def select(self, selection_input: SelectionInput) -> SelectionResult:
        batch_size = validate_selection_input(selection_input)
        epoch = selection_input.metadata.get("epoch", 0)
        keep_rate = self.schedule.rate_at(epoch)
        scaled = batch_size * keep_rate
        keep_count = max(
            1,
            int(math.floor(scaled) if self.rounding == "floor" else math.ceil(scaled)),
        )

        # Stable global indices make equal-score selection deterministic and
        # independent of the incoming batch order.
        index_order = torch.argsort(selection_input.sample_indices, stable=True)
        score_order = torch.argsort(
            selection_input.scores[index_order],
            stable=True,
        )
        chosen = index_order[score_order[:keep_count]]
        mask = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=selection_input.scores.device,
        )
        mask[chosen] = True
        return SelectionResult(
            selected_mask=mask,
            metrics={
                "selected_samples": float(keep_count),
                "selected_ratio": float(keep_count / batch_size),
                "keep_rate": keep_rate,
            },
        )
