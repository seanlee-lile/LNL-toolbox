"""Compatibility adapter from legacy hard Selectors to contributions."""

from __future__ import annotations

import torch

from lnl_toolbox.selectors import (
    SelectionInput,
    Selector,
    validate_selection_result,
)

from .base import ContributionResult


class SelectorContributionAdapter:
    """Represent a hard Selector as mask plus identity sample weights."""

    def __init__(self, selector: Selector) -> None:
        self.selector = selector

    def resolve(self, selection_input: SelectionInput) -> ContributionResult:
        selection = self.selector.select(selection_input)
        mask = validate_selection_result(
            selection,
            batch_size=int(selection_input.scores.numel()),
            device=selection_input.scores.device,
        )
        return ContributionResult(
            selected_mask=mask,
            sample_weights=torch.ones_like(selection_input.scores),
            metrics=dict(selection.metrics),
            selection_mask=mask,
        )
