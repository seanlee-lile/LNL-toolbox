from __future__ import annotations

"""Corrected released-code-inspired outlier detection for CNLCU-H."""

import math

import numpy as np
import torch
from torch import Tensor


def _load_local_outlier_factor():
    try:
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError as error:
        raise ImportError(
            "CNLCU-H corrected LOF requires the optional training dependency; "
            'install it with python -m pip install -e ".[train]"'
        ) from error
    return LocalOutlierFactor


def _validate_parameters(
    n_neighbors: int,
    contamination: float,
    minimum_observations: int,
) -> tuple[int, float, int]:
    neighbors = int(n_neighbors)
    minimum = int(minimum_observations)
    ratio = float(contamination)
    if neighbors < 1:
        raise ValueError("CNLCU-H LOF n_neighbors must be at least one")
    if not math.isfinite(ratio) or not 0.0 < ratio < 0.5:
        raise ValueError("CNLCU-H LOF contamination must be finite and in (0,0.5)")
    if minimum < neighbors + 1:
        raise ValueError(
            "CNLCU-H minimum_observations must be at least n_neighbors + 1"
        )
    return neighbors, ratio, minimum


def lof_retained_mask(
    history: Tensor,
    observed: Tensor,
    *,
    n_neighbors: int,
    contamination: float,
    minimum_observations: int,
) -> Tensor:
    """Return per-row retained observations; LOF ``-1`` means outlier."""

    neighbors, ratio, minimum = _validate_parameters(
        n_neighbors, contamination, minimum_observations
    )
    if not torch.is_tensor(history) or history.ndim != 2:
        raise ValueError("CNLCU-H history must have shape [B,W]")
    if not history.is_floating_point():
        raise TypeError("CNLCU-H history must use a floating-point dtype")
    if history.device.type != "cpu":
        raise ValueError("CNLCU-H LOF history must be on CPU")
    if not torch.is_tensor(observed) or observed.shape != history.shape:
        raise ValueError("CNLCU-H observed mask must align with history")
    if observed.dtype != torch.bool or observed.device.type != "cpu":
        raise TypeError("CNLCU-H observed mask must be a CPU bool tensor")
    values = history[observed]
    if values.numel() and (
        not bool(torch.isfinite(values).all()) or bool((values < 0).any())
    ):
        raise ValueError("observed CNLCU-H losses must be finite and non-negative")
    counts = observed.sum(dim=1)
    if bool((counts <= 0).any()):
        raise ValueError("every CNLCU-H sample must have an observed loss")

    retained = observed.clone()
    eligible = torch.nonzero(counts >= minimum, as_tuple=False).flatten()
    if eligible.numel() == 0:
        return retained
    LocalOutlierFactor = _load_local_outlier_factor()
    for row in eligible.tolist():
        columns = torch.nonzero(observed[row], as_tuple=False).flatten()
        samples = history[row, columns].to(torch.float64).numpy().reshape(-1, 1)
        detector = LocalOutlierFactor(
            n_neighbors=neighbors,
            algorithm="auto",
            contamination=ratio,
            n_jobs=1,
        )
        labels = np.asarray(detector.fit_predict(samples))
        if labels.shape != (columns.numel(),) or not bool(
            np.isin(labels, (-1, 1)).all()
        ):
            raise RuntimeError("CNLCU-H LOF returned invalid inlier/outlier labels")
        keep = torch.from_numpy(labels == 1)
        if not bool(keep.any()):
            raise RuntimeError("CNLCU-H LOF rejected every observation for a sample")
        retained[row, columns] = keep
    if bool((retained & ~observed).any()):
        raise RuntimeError("CNLCU-H retained mask included unobserved padding")
    return retained


__all__ = ["lof_retained_mask"]
