from __future__ import annotations

"""Deterministic named hidden-layer extraction for PCSE."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from lnl_toolbox.training.snapshots import (
    FeatureSnapshot,
    collect_feature_snapshot,
)

from .config import PCSEFeatureLayerConfig


@dataclass(frozen=True)
class PCSEFeatureCollection:
    layer_names: tuple[str, ...]
    snapshots: tuple[FeatureSnapshot, ...]

    def __post_init__(self) -> None:
        if len(self.layer_names) < 2 or len(self.layer_names) != len(self.snapshots):
            raise ValueError("PCSE requires aligned snapshots for at least two layers")
        reference = self.snapshots[0]
        for snapshot in self.snapshots[1:]:
            if not np.array_equal(snapshot.global_indices, reference.global_indices):
                raise ValueError("PCSE feature layer stable indices are misaligned")
            if not np.array_equal(snapshot.noisy_targets, reference.noisy_targets):
                raise ValueError("PCSE feature layer noisy targets are misaligned")


def _pool_activation(value: Any, pooling: str) -> Tensor:
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("PCSE hidden layer returned an empty sequence")
        value = value[0]
    if not torch.is_tensor(value) or value.ndim < 2:
        raise ValueError("PCSE hidden activation must have shape [B, ...]")
    if pooling == "global_average":
        if value.ndim == 2:
            return value
        if value.ndim == 4:
            return F.adaptive_avg_pool2d(value, 1).flatten(1)
        return value.flatten(start_dim=2).mean(dim=2)
    if pooling == "flatten":
        return value.flatten(start_dim=1)
    raise ValueError(f"Unsupported PCSE feature pooling: {pooling}")


def extract_named_activation(
    model: nn.Module,
    inputs: Tensor,
    layer: PCSEFeatureLayerConfig,
) -> Tensor:
    """Run one forward pass and return one named hidden activation.

    The hook is always removed, including when the model forward raises.
    """

    try:
        module = model.get_submodule(layer.name)
    except AttributeError as exc:
        raise ValueError(f"Unknown PCSE feature layer: {layer.name!r}") from exc
    captured: list[Any] = []
    handle = module.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        model_output = model(inputs)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError(
            f"PCSE feature layer {layer.name!r} was invoked {len(captured)} times"
        )
    if captured[0] is model_output:
        raise ValueError(
            f"PCSE feature layer {layer.name!r} is the model logits output; "
            "PCSE requires hidden representations"
        )
    return _pool_activation(captured[0], layer.pooling)


def collect_pcse_features(
    model: nn.Module,
    loader: Any,
    device: torch.device | str,
    *,
    dataset: str,
    split: str,
    layers: tuple[PCSEFeatureLayerConfig, ...],
) -> PCSEFeatureCollection:
    snapshots = tuple(
        collect_feature_snapshot(
            model,
            loader,
            device,
            dataset=dataset,
            split=f"{split}:{layer.name}",
            feature_extractor=(
                lambda current_model, inputs, current_layer=layer:
                extract_named_activation(current_model, inputs, current_layer)
            ),
        )
        for layer in layers
    )
    return PCSEFeatureCollection(
        layer_names=tuple(layer.name for layer in layers),
        snapshots=snapshots,
    )
