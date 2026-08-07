from __future__ import annotations

"""Compatibility helpers from the repository's original Co-teaching stub."""

import numpy as np


def remember_rate(epoch: int, noise_rate: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0 - noise_rate
    progress = min(max(epoch, 0) / warmup_epochs, 1.0)
    return 1.0 - progress * noise_rate


def _small_loss(losses: np.ndarray, count: int) -> np.ndarray:
    losses = np.asarray(losses, dtype=np.float64)
    return np.argsort(losses, kind="stable")[:count]


def coteaching_exchange(
    losses_a: np.ndarray, losses_b: np.ndarray, keep_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices used to update A and B, selected by the peer network."""

    if not 0.0 < keep_rate <= 1.0:
        raise ValueError("keep_rate must be in (0, 1]")
    if np.shape(losses_a) != np.shape(losses_b):
        raise ValueError("both networks must score the same mini-batch")
    count = max(1, int(round(len(losses_a) * keep_rate)))
    selected_by_a = _small_loss(losses_a, count)
    selected_by_b = _small_loss(losses_b, count)
    return selected_by_b, selected_by_a
