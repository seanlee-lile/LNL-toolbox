"""Read-only Scratch-native data facts for the Scratch editor.

The editor must not pull the legacy training data service into the Scratch
runtime.  These facts describe the built-in sources and aliases registered in
the shared local catalog. Concrete source loading still happens through the
canonical Scratch Data blocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..data_runtime import registered_dataset_catalog


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


def _registered_fact(record: Mapping[str, object]) -> dict[str, object]:
    alias = str(record.get("alias", "")).strip().lower()
    data = record.get("data") if isinstance(record.get("data"), Mapping) else {}
    data = dict(data)
    profile = record.get("profile") if isinstance(record.get("profile"), Mapping) else {}
    evidence = record.get("evidence") if isinstance(record.get("evidence"), Mapping) else {}
    location = data.get("root") or data.get("path")
    exists = bool(location) and Path(str(location)).expanduser().exists()
    state = str(record.get("state", "registered"))
    status = "ready" if exists and state in {"layout_validated", "training_verified"} else "available" if exists else "unavailable"
    classes = data.get("num_classes") or profile.get("num_classes") or evidence.get("classes")
    input_shape = data.get("input_shape") or profile.get("input_shape")
    counts = profile.get("sample_counts_by_split") if isinstance(profile, Mapping) else {}
    return {
        "alias": alias,
        "name": alias,
        "path": str(location) if location else None,
        "adapter": str(record.get("adapter") or data.get("name") or "registered"),
        "status": status,
        "registration_state": state,
        "train_samples": (counts or {}).get("train") if isinstance(counts, Mapping) else evidence.get("train_samples"),
        "test_samples": (counts or {}).get("test") if isinstance(counts, Mapping) else evidence.get("test_samples"),
        "num_classes": classes,
        "input_shape": input_shape,
        "has_clean_target": str((profile or {}).get("clean_train_labels", "available")) != "unavailable",
        "has_noisy_target": str((profile or {}).get("observed_train_labels", "available")) == "available" and str(record.get("adapter", "")).endswith("n"),
        "has_sample_index": str((profile or {}).get("stable_indices", "available")) != "unavailable",
        "layout_validated": state in {"layout_validated", "training_verified"},
        "training_verified": state == "training_verified",
        "inspect_status": state,
        "noise_methods": ["none", "symmetric", "pairflip", "class_conditional", "instance_dependent"],
        "profile": profile or None,
        "error": record.get("error"),
    }


def dataset_catalog_payload() -> dict[str, object]:
    """Return built-ins plus aliases registered by the shared data page."""

    values = {name: _fact(name) for name in _DATASETS}
    values.update({alias: _registered_fact(record) for alias, record in registered_dataset_catalog().items()})
    return {"datasets": [values[name] for name in sorted(values)]}


def dataset_fact_payload(alias: object) -> dict[str, object]:
    """Return one registered dataset's current, persisted inspection facts."""

    key = str(alias).strip().lower().replace(" ", "-")
    record = registered_dataset_catalog().get(key)
    return _registered_fact(record) if record is not None else _fact(key)


__all__ = ["dataset_catalog_payload", "dataset_fact_payload"]
