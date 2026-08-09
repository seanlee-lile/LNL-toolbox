from __future__ import annotations

"""Independent label-vector predictors used by DLD."""

import math

import torch
from torch import Tensor, nn


def timestep_embedding(timestep: Tensor, dimension: int) -> Tensor:
    if timestep.ndim != 1 or timestep.dtype != torch.int64:
        raise ValueError("DLD timestep embedding expects int64 [B]")
    if dimension <= 0:
        raise ValueError("DLD timestep embedding dimension must be positive")
    half = dimension // 2
    if half == 0:
        return timestep.to(torch.float32)[:, None]
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timestep.to(torch.float32)[:, None] * frequency[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if embedding.shape[1] < dimension:
        embedding = torch.nn.functional.pad(embedding, (0, dimension - embedding.shape[1]))
    return embedding


class DLDLabelPredictor(nn.Module):
    """Predict one `[B,C]` DLD quantity from label and frozen features."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        *,
        hidden_dim: int = 64,
        time_dim: int = 16,
    ) -> None:
        super().__init__()
        if min(num_classes, feature_dim, hidden_dim, time_dim) <= 0:
            raise ValueError("DLD predictor dimensions must be positive")
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.time_dim = int(time_dim)
        width = self.num_classes * 2 + self.feature_dim + self.time_dim
        self.network = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def forward(
        self,
        y_t: Tensor,
        y_n: Tensor,
        condition_features: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        batch = y_t.shape[0]
        if y_t.shape != (batch, self.num_classes) or y_n.shape != y_t.shape:
            raise ValueError("DLD predictor label inputs must have shape [B, C]")
        if condition_features.shape != (batch, self.feature_dim):
            raise ValueError("DLD predictor feature input has the wrong shape")
        if timestep.shape != (batch,) or timestep.dtype != torch.int64:
            raise ValueError("DLD predictor timestep must be int64 [B]")
        if len({y_t.device, y_n.device, condition_features.device, timestep.device}) != 1:
            raise ValueError("DLD predictor inputs must share a device")
        for value in (y_t, y_n, condition_features):
            if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
                raise ValueError("DLD predictor inputs must be finite floating tensors")
        embedded = timestep_embedding(timestep, self.time_dim).to(
            device=y_t.device, dtype=y_t.dtype
        )
        output = self.network(torch.cat((y_t, y_n, condition_features, embedded), dim=1))
        if not bool(torch.isfinite(output).all()):
            raise ValueError("DLD predictor output is non-finite")
        return output


__all__ = ["DLDLabelPredictor", "timestep_embedding"]
