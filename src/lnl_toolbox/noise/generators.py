from __future__ import annotations

"""Reproducible synthetic label-noise generators for controlled benchmarks."""

import numpy as np

from .manifest import NoiseManifest
from .transition import validate_transition_matrix


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
    labels: np.ndarray,
    num_classes: int,
    rate: float,
    seed: int,
    dataset: str = "unknown",
    sampling: str = "global",
    rng: str = "default_rng",
) -> NoiseManifest:
    """Replace a fixed fraction of labels with uniformly chosen wrong classes."""
    labels = _validate(labels, num_classes, rate)
    sampling = str(sampling).strip().lower()
    if sampling not in {"global", "per_class", "transition"}:
        raise ValueError(
            "symmetric sampling must be 'global', 'per_class', or 'transition'"
        )
    rng = str(rng).strip().lower()
    if rng not in {"default_rng", "numpy_legacy"}:
        raise ValueError("symmetric rng must be 'default_rng' or 'numpy_legacy'")
    noisy = labels.copy()
    random = np.random.RandomState(seed) if rng == "numpy_legacy" else np.random.default_rng(seed)
    if sampling == "transition":
        transition = np.full(
            (num_classes, num_classes),
            rate / (num_classes - 1),
        )
        np.fill_diagonal(transition, 1.0 - rate)
        if rng == "numpy_legacy":
            for index, label in enumerate(labels):
                noisy[index] = random.multinomial(
                    1, transition[int(label)], size=1
                )[0].argmax()
        else:
            for index, label in enumerate(labels):
                noisy[index] = random.choice(
                    num_classes, p=transition[int(label)]
                )
        chosen = np.flatnonzero(noisy != labels)
    elif sampling == "global":
        count = int(round(rate * labels.size))
        chosen = random.choice(labels.size, size=count, replace=False)
    else:
        chosen_parts = [
            random.choice(
                np.flatnonzero(labels == class_index),
                size=int(round(rate * int(np.sum(labels == class_index)))),
                replace=False,
            )
            for class_index in range(num_classes)
            if np.any(labels == class_index)
        ]
        chosen = np.concatenate(chosen_parts) if chosen_parts else np.empty(0, dtype=np.int64)
    if sampling == "transition":
        pass
    elif rng == "numpy_legacy":
        for index in chosen:
            other_classes = np.delete(np.arange(num_classes), labels[index])
            noisy[index] = random.choice(other_classes)
    else:
        offsets = random.integers(1, num_classes, size=chosen.size)
        noisy[chosen] = (labels[chosen] + offsets) % num_classes
    transition = np.full((num_classes, num_classes), rate / (num_classes - 1))
    np.fill_diagonal(transition, 1.0 - rate)
    return NoiseManifest(
        dataset,
        "symmetric",
        seed,
        rate,
        labels,
        noisy,
        transition,
        metadata={"sampling": sampling, "rng": rng},
    )


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


def generate_class_conditional(
    labels: np.ndarray,
    transition_matrix: np.ndarray,
    rate: float,
    seed: int,
    dataset: str = "unknown",
    rng: str = "numpy_legacy",
) -> NoiseManifest:
    """Sample noisy labels from a row-stochastic class transition matrix."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    matrix = validate_transition_matrix(transition_matrix)
    if labels.size and (labels.min() < 0 or labels.max() >= matrix.shape[0]):
        raise ValueError("labels must be within the transition matrix classes")
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    rng_name = str(rng).strip().lower()
    if rng_name != "numpy_legacy":
        raise ValueError("class_conditional noise requires rng='numpy_legacy'")
    random = np.random.RandomState(seed)
    noisy = labels.copy()
    for position, label in enumerate(labels):
        noisy[position] = int(random.multinomial(1, matrix[int(label)]).argmax())
    return NoiseManifest(
        dataset,
        "class_conditional",
        seed,
        float(rate),
        labels,
        noisy,
        matrix,
        metadata={"rng": rng_name, "transition_source": "configuration"},
    )


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


def generate_pdl_idn(
    inputs: np.ndarray,
    clean_targets: np.ndarray,
    num_classes: int,
    rate: float,
    seed: int,
    dataset: str = "unknown",
) -> NoiseManifest:
    """Generate PDL Algorithm 2 instance-dependent noise.

    Per-sample corruption rates are sampled from a truncated normal and wrong
    classes are distributed by input-dependent random linear scores.
    """

    targets = _validate(clean_targets, num_classes, rate)
    features = np.asarray(inputs, dtype=np.float64)
    if features.shape[0] != targets.size:
        raise ValueError("inputs and clean_targets must contain the same samples")
    features = features.reshape(targets.size, -1)
    if not np.isfinite(features).all():
        raise ValueError("inputs must be finite")
    random = np.random.RandomState(seed)
    rates = random.normal(float(rate), 0.1, size=targets.size)
    invalid = (rates < 0.0) | (rates > 1.0)
    while invalid.any():
        rates[invalid] = random.normal(float(rate), 0.1, size=int(invalid.sum()))
        invalid = (rates < 0.0) | (rates > 1.0)
    weights = random.standard_normal((num_classes, features.shape[1], num_classes))
    transitions = np.zeros((targets.size, num_classes), dtype=np.float64)
    noisy = targets.copy()
    for position, clean_class in enumerate(targets):
        scores = features[position] @ weights[int(clean_class)]
        scores[int(clean_class)] = -np.inf
        classes = np.delete(np.arange(num_classes), int(clean_class))
        wrong_scores = scores[classes]
        wrong = np.exp(wrong_scores - wrong_scores.max())
        wrong /= wrong.sum()
        transitions[position, classes] = rates[position] * wrong
        transitions[position, int(clean_class)] = 1.0 - rates[position]
        noisy[position] = int(random.choice(num_classes, p=transitions[position]))
    return NoiseManifest(dataset, "pdl_instance_dependent", seed, rate, targets,
        noisy, per_sample_transition=transitions, num_classes=num_classes,
        metadata={"generator": "pdl_algorithm_2", "rate_distribution": "truncated_normal",
            "rate_std": 0.1, "transition_convention": "clean_to_noisy_row"})
