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
        if not self.layer_names or len(self.layer_names) != len(self.snapshots):
            raise ValueError("PCSE requires aligned snapshots for at least one layer")
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
    if not layers:
        raise ValueError("PCSE requires at least one hidden feature layer")
    widths: list[int] = []

    def extract_all(current_model: nn.Module, inputs: Tensor) -> Tensor:
        captured: list[Any] = [None] * len(layers)
        counts = [0] * len(layers)
        handles = []
        try:
            for index, layer in enumerate(layers):
                try:
                    module = current_model.get_submodule(layer.name)
                except AttributeError as exc:
                    raise ValueError(f"Unknown PCSE feature layer: {layer.name!r}") from exc
                def capture(_module, _inputs, output, position=index):
                    captured[position] = output
                    counts[position] += 1
                handles.append(module.register_forward_hook(capture))
            model_output = current_model(inputs)
        finally:
            for handle in handles:
                handle.remove()
        if any(count != 1 for count in counts):
            raise ValueError("PCSE feature layers must each be invoked exactly once")
        if any(value is model_output for value in captured):
            raise ValueError("PCSE feature layer is the model logits output; PCSE requires hidden representations")
        values = [_pool_activation(value, layer.pooling) for value, layer in zip(captured, layers)]
        current_widths = [value.shape[1] for value in values]
        if widths and widths != current_widths:
            raise ValueError("PCSE feature widths changed between batches")
        widths[:] = current_widths
        return torch.cat(values, dim=1)

    # One forward pass per batch keeps every layer on the same augmented view.
    combined = collect_feature_snapshot(
        model, loader, device, dataset=dataset, split=split,
        feature_extractor=extract_all,
    )
    boundaries = np.cumsum([0, *widths])
    snapshots = tuple(
        FeatureSnapshot(combined.features[:, boundaries[index]:boundaries[index + 1]],
                        combined.noisy_targets, combined.global_indices,
                        dataset, f"{split}:{layer.name}")
        for index, layer in enumerate(layers)
    )
    return PCSEFeatureCollection(
        layer_names=tuple(layer.name for layer in layers),
        snapshots=snapshots,
    )
