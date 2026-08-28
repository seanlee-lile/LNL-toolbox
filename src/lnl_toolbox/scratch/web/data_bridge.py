"""Read-only Scratch-native data facts for the Scratch editor.

The editor must not pull the legacy training data service into the Scratch
runtime.  These facts describe the built-in sources understood by
``scratch.data_runtime``; concrete source loading still happens through the
canonical Data blocks.
"""

from __future__ import annotations


_DATASETS = {
    "cifar10": {"num_classes": 10, "input_shape": [3, 32, 32]},
    "cifar100": {"num_classes": 100, "input_shape": [3, 32, 32]},
    "mnist": {"num_classes": 10, "input_shape": [1, 28, 28]},
    "fashion_mnist": {"num_classes": 10, "input_shape": [1, 28, 28]},
    "synthetic": {"num_classes": 2, "input_shape": [4]},
    "synthetic_classification": {"num_classes": 2, "input_shape": [4]},
}


def _fact(alias: str) -> dict[str, object]:
    name = str(alias).strip().lower()
    if not name:
        raise ValueError("dataset alias is required")
    schema = dict(_DATASETS.get(name, {"num_classes": None, "input_shape": None}))
    return {
        "alias": name,
        "name": name,
        "path": None,
        "adapter": "scratch-native",
        "status": "available" if name in _DATASETS else "unregistered",
        "train_samples": None,
        "test_samples": None,
        "num_classes": schema["num_classes"],
        "input_shape": schema["input_shape"],
        "has_clean_target": True,
        "has_noisy_target": False,
        "has_sample_index": True,
        "layout_validated": True,
        "training_verified": False,
        "inspect_status": "scratch-native",
        "noise_methods": ["none", "symmetric", "pairflip", "class_conditional", "instance_dependent"],
        "profile": None,
        "error": None,
    }


def dataset_catalog_payload() -> dict[str, object]:
    """Return built-in Scratch dataset facts without probing external paths."""

    return {"datasets": [_fact(name) for name in sorted(_DATASETS)]}


def dataset_fact_payload(alias: object) -> dict[str, object]:
    """Return one registered dataset's current, persisted inspection facts."""

    return _fact(str(alias))


__all__ = ["dataset_catalog_payload", "dataset_fact_payload"]
