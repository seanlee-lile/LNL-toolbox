from __future__ import annotations

import numpy as np


def _target_probability(probabilities: np.ndarray, targets: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    return np.clip(probabilities[np.arange(targets.size), targets], 1e-12, 1.0)


def cross_entropy(probabilities: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return -np.log(_target_probability(probabilities, targets))


def generalized_cross_entropy(
    probabilities: np.ndarray, targets: np.ndarray, q: float = 0.7
) -> np.ndarray:
    if q <= 0.0:
        raise ValueError("q must be positive")
    p = _target_probability(probabilities, targets)
    return (1.0 - np.power(p, q)) / q

