"""Alignment and direction adaptation from reliability to selection scores."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from lnl_toolbox.selectors import SelectionInput, validate_selection_input

from .base import ReliabilityResult, validate_reliability_result


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _validate_expected_sample_indices(
    expected_sample_indices: Tensor,
    *,
    device: torch.device,
) -> None:
    if (
        not isinstance(expected_sample_indices, Tensor)
        or expected_sample_indices.ndim != 1
    ):
        raise ValueError(
            "expected_sample_indices must be a one-dimensional tensor"
        )
    if expected_sample_indices.numel() == 0:
        raise ValueError("expected_sample_indices must not be empty")
    if expected_sample_indices.dtype not in _INTEGER_DTYPES:
        raise ValueError(
            "expected_sample_indices must use an integer dtype"
        )
    if expected_sample_indices.device != device:
        raise ValueError(
            "expected_sample_indices must be on the reliability result device"
        )
    if (
        torch.unique(expected_sample_indices).numel()
        != expected_sample_indices.numel()
    ):
        raise ValueError("expected_sample_indices must be unique")


class ReliabilityToSelectionInputAdapter:
    """Create lower-is-preferred selector scores from reliability evidence.

    The adapter aligns dataset-level evidence to the requested stable sample
    identities and their requested order. It does not invoke a Selector or
    decide any selection policy.
    """

    def adapt(
        self,
        result: ReliabilityResult,
        *,
        expected_sample_indices: Tensor,
        metadata: Mapping[str, Any] | None = None,
    ) -> SelectionInput:
        """Return an aligned input whose score is negative reliability."""

        result_indices, reliability_scores = validate_reliability_result(
            result
        )
        _validate_expected_sample_indices(
            expected_sample_indices,
            device=result_indices.device,
        )
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("selection metadata must be a mapping")

        # Canonical int64 lookup supports arbitrary unique integer identities
        # without relying on contiguous indices or implicit array positions.
        canonical_result_indices = result_indices.to(dtype=torch.int64)
        canonical_expected_indices = expected_sample_indices.to(
            dtype=torch.int64
        )
        sorted_order = torch.argsort(
            canonical_result_indices,
            stable=True,
        )
        sorted_indices = canonical_result_indices[sorted_order]
        lookup_positions = torch.searchsorted(
            sorted_indices,
            canonical_expected_indices,
        )
        in_bounds = lookup_positions < sorted_indices.numel()
        safe_positions = lookup_positions.clamp(
            max=sorted_indices.numel() - 1
        )
        matches = in_bounds & (
            sorted_indices[safe_positions] == canonical_expected_indices
        )
        if not bool(matches.all().item()):
            missing = expected_sample_indices[~matches].detach().cpu().tolist()
            raise ValueError(
                "expected_sample_indices contains indices absent from "
                f"ReliabilityResult: {missing}"
            )

        source_positions = sorted_order[safe_positions]
        aligned_reliability = reliability_scores[source_positions]
        selection_input = SelectionInput(
            scores=-aligned_reliability,
            sample_indices=expected_sample_indices,
            metadata=dict(metadata or {}),
        )
        validate_selection_input(selection_input)
        return selection_input
