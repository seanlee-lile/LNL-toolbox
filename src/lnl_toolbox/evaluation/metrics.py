from __future__ import annotations

import numpy as np


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    return float(np.mean(predictions == targets)) if targets.size else 0.0


def selection_precision_recall(selected: np.ndarray, is_clean: np.ndarray) -> tuple[float, float]:
    selected = np.asarray(selected, dtype=bool)
    is_clean = np.asarray(is_clean, dtype=bool)
    true_positive = int(np.sum(selected & is_clean))
    precision = true_positive / max(int(selected.sum()), 1)
    recall = true_positive / max(int(is_clean.sum()), 1)
    return precision, recall

