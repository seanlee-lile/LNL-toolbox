from __future__ import annotations

"""Opt-in model output contract for consumers of learned features."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FeatureOutput:
    """Logits and a two-dimensional representation from one forward pass."""

    logits: Tensor
    features: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.logits, Tensor) or not isinstance(
            self.features,
            Tensor,
        ):
            raise TypeError("FeatureOutput values must be tensors")
        if self.logits.ndim != 2 or self.features.ndim != 2:
            raise ValueError(
                "FeatureOutput tensors must have shape [B, D]"
            )
        if self.logits.shape[0] != self.features.shape[0]:
            raise ValueError(
                "FeatureOutput logits and features must share batch size"
            )
        if self.logits.device != self.features.device:
            raise ValueError(
                "FeatureOutput logits and features must share a device"
            )
        if not bool(torch.isfinite(self.logits).all()) or not bool(
            torch.isfinite(self.features).all()
        ):
            raise ValueError("FeatureOutput tensors must be finite")


def forward_with_features(
    model: nn.Module,
    inputs: Tensor,
) -> FeatureOutput:
    """Call the explicit feature API without changing ordinary ``forward``."""

    method = getattr(model, "forward_with_features", None)
    if not callable(method):
        raise TypeError(
            f"model {type(model).__name__} does not expose "
            "forward_with_features()"
        )
    output: Any = method(inputs)
    if isinstance(output, FeatureOutput):
        return output
    if isinstance(output, dict) and {
        "logits",
        "features",
    }.issubset(output):
        return FeatureOutput(output["logits"], output["features"])
    if isinstance(output, (tuple, list)) and len(output) == 2:
        return FeatureOutput(output[0], output[1])
    raise TypeError(
        "forward_with_features() must return FeatureOutput or "
        "(logits, features)"
    )


__all__ = ["FeatureOutput", "forward_with_features"]
