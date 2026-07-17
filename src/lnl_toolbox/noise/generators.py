from __future__ import annotations

"""Reproducible synthetic label-noise generators for controlled benchmarks."""

import numpy as np

from .manifest import NoiseManifest


def _validate(labels: np.ndarray, num_classes: int, rate: float) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError("labels must be within [0, num_classes)")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    return labels


def generate_symmetric(
    labels: np.ndarray, num_classes: int, rate: float, seed: int, dataset: str = "unknown"
) -> NoiseManifest:
    """Replace a fixed fraction of labels with uniformly chosen wrong classes."""
    labels = _validate(labels, num_classes, rate)
    noisy = labels.copy()
    rng = np.random.default_rng(seed)
    count = int(round(rate * labels.size))
    chosen = rng.choice(labels.size, size=count, replace=False)
    offsets = rng.integers(1, num_classes, size=count)
    noisy[chosen] = (labels[chosen] + offsets) % num_classes
    transition = np.full((num_classes, num_classes), rate / (num_classes - 1))
    np.fill_diagonal(transition, 1.0 - rate)
    return NoiseManifest(dataset, "symmetric", seed, rate, labels, noisy, transition)


def generate_pairflip(
    labels: np.ndarray, num_classes: int, rate: float, seed: int, dataset: str = "unknown"
) -> NoiseManifest:
    """Flip each label to the next class with the requested probability."""
    labels = _validate(labels, num_classes, rate)
    noisy = labels.copy()
    rng = np.random.default_rng(seed)
    flip = rng.random(labels.size) < rate
    noisy[flip] = (labels[flip] + 1) % num_classes
    transition = np.eye(num_classes) * (1.0 - rate)
    transition[np.arange(num_classes), (np.arange(num_classes) + 1) % num_classes] = rate
    return NoiseManifest(dataset, "pairflip", seed, rate, labels, noisy, transition)


def generate_instance_dependent(
    labels: np.ndarray,
    class_scores: np.ndarray,
    rate: float,
    seed: int,
    dataset: str = "unknown",
) -> NoiseManifest:
    """Generate IDN from fixed per-sample class scores without leaking clean labels to training."""

    scores = np.asarray(class_scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("class_scores must have shape [N, C]")
    labels = _validate(labels, scores.shape[1], rate)
    if scores.shape[0] != labels.size:
        raise ValueError("class_scores must have shape [N, C]")
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities[np.arange(labels.size), labels] = 0.0
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)

    ambiguity = 1.0 - np.max(probabilities, axis=1)
    ambiguity = ambiguity / max(float(ambiguity.mean()), 1e-12)
    flip_probability = np.clip(rate * ambiguity, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    flip = rng.random(labels.size) < flip_probability
    noisy = labels.copy()
    for index in np.flatnonzero(flip):
        noisy[index] = rng.choice(scores.shape[1], p=probabilities[index])

    per_sample = probabilities * flip_probability[:, None]
    per_sample[np.arange(labels.size), labels] = 1.0 - flip_probability
    return NoiseManifest(
        dataset,
        "instance_dependent",
        seed,
        rate,
        labels,
        noisy,
        per_sample_transition=per_sample,
        metadata={"generator": "score_weighted_idn"},
    )
