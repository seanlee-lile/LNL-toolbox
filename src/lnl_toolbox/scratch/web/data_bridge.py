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
    "synthetic_binary_2d": {"num_classes": 2, "input_shape": [2]},
    "synthetic_binary_high_dim": {"num_classes": 2, "input_shape": [16]},
    "synthetic_multiclass": {"num_classes": 3, "input_shape": [8]},
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
        "has_clean_target": True if name in _DATASETS else None,
        "has_test_clean_target": True if name in _DATASETS else None,
        "has_noisy_target": False if name in _DATASETS else None,
        "has_sample_index": True if name in _DATASETS else None,
        "has_validation_source": False,
        "layout_validated": name in _DATASETS,
        "training_verified": False,
        "inspect_status": "scratch-native",
        "noise_methods": [
            "none", "symmetric", "pairflip", "class_conditional",
            "instance_dependent", "pdl", "binary_asymmetric_rcn",
            "asymmetric_rcn", "external", "external_torch",
        ],
        "profile": None,
        "error": None,
    }


def _available(value: object) -> bool | None:
    if value is True or value == "available":
        return True
    if value is False or value == "unavailable":
        return False
    return None


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
        "has_clean_target": _available(profile.get("clean_train_labels")),
        "has_test_clean_target": _available(profile.get("clean_test_labels")),
        "has_validation_clean_target": _available(profile.get("clean_validation_labels")),
        "has_noisy_target": _available(profile.get("has_noisy_target", evidence.get("has_noisy_target"))),
        "has_sample_index": _available(profile.get("stable_indices")),
        "has_validation_source": bool(
            (profile or {}).get("sample_counts_by_split", {}).get("validation")
            if isinstance((profile or {}).get("sample_counts_by_split"), Mapping) else False
        ),
        "layout_validated": state in {"layout_validated", "training_verified"},
        "training_verified": state == "training_verified",
        "inspect_status": state,
        "noise_methods": [
            "none", "symmetric", "pairflip", "class_conditional",
            "instance_dependent", "pdl", "binary_asymmetric_rcn",
            "asymmetric_rcn", "external", "external_torch",
        ],
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


def _resolve_path(value: object, base_dir: Path | None) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def inspect_source_capabilities(params: Mapping[str, object], *, base_dir: Path | None = None) -> dict[str, object]:
    """Resolve unknown facts with the same source block used by execution.

    No training, noise injection, download or persistent capability cache is
    created here. A nonempty directory alone is never evidence of a dataset.
    """
    from ..blocks.data import load_dataset
    from ..context import ScratchContext

    arguments = {key: value for key, value in params.items()
                 if key in {"dataset", "source_mode", "root", "path", "options"}}
    options = dict(arguments.get("options") or {})
    options["download"] = False
    for values in (arguments, options):
        for key in ("path", "root"):
            if values.get(key):
                values[key] = str(_resolve_path(values[key], base_dir))
    arguments["options"] = options
    ctx = ScratchContext()
    load_dataset(ctx, **arguments)
    train, test, validation = ctx["train_source"], ctx["test_source"], ctx["validation_source"]
    if not train.samples or test is None or not test.samples:
        raise ValueError("数据源需要非空的 train 和独立 test split")
    return {"num_classes": train.num_classes, "has_clean_target": train.has_clean_targets,
            "has_test_clean_target": test.has_clean_targets,
            "has_validation_clean_target": validation.has_clean_targets if validation is not None else None,
            "has_sample_index": True, "has_validation_source": validation is not None and bool(validation.samples),
            "train_samples": len(train.samples), "test_samples": len(test.samples),
            "layout_validated": True, "inspect_status": "scratch-source-inspected"}


def dataset_preflight_payload(params: Mapping[str, object] | object, *, base_dir: Path | None = None) -> dict[str, object]:
    """Check an explicitly configured local dataset source before a run.

    This is deliberately a narrow filesystem gate.  Adapter-specific layout
    validation still belongs to Scratch's source loader; this check prevents
    the Web UI from treating an empty or nonexistent custom path as ready.
    Built-in aliases without an explicit path remain unchanged, while a
    registered alias inherits and checks its persisted root/path.
    """

    if not isinstance(params, Mapping):
        return {"ok": False, "code": "invalid-dataset-params", "error": "dataset parameters must be an object"}
    effective = dict(params)
    options = effective.get("options")
    if isinstance(options, Mapping):
        # ``load_dataset`` applies options after root/path, so mirror that
        # precedence for the preflight decision.
        effective.update(options)
    alias = str(effective.get("dataset") or effective.get("name") or "").strip()
    mode = str(effective.get("source_mode") or effective.get("mode") or "").strip().lower()
    custom_modes = {"custom", "custom_path", "local", "path", "folder"}
    record = None
    if alias:
        record = registered_dataset_catalog().get(alias.lower().replace(" ", "-"))
    candidate = effective.get("path") or effective.get("root")
    if not candidate and alias:
        if isinstance(record, Mapping):
            data = record.get("data")
            if isinstance(data, Mapping):
                candidate = data.get("path") or data.get("root")
    if not candidate:
        if mode in custom_modes:
            return {
                "ok": False,
                "code": "missing-dataset-path",
                "error": "自定义数据源需要填写 path 或 root。",
                "dataset": alias or None,
            }
        return {"ok": True, "skipped": True, "dataset": alias or None}
    resolved = _resolve_path(candidate, base_dir)
    try:
        if not resolved.exists():
            return {"ok": False, "code": "dataset-path-not-found", "error": f"数据集路径不存在：{resolved}", "path": str(resolved), "dataset": alias or None}
        if resolved.is_dir():
            # A directory with no entries cannot be a usable dataset source.
            has_entry = next(resolved.iterdir(), None) is not None
            if not has_entry:
                return {"ok": False, "code": "dataset-path-empty", "error": f"数据集目录为空：{resolved}", "path": str(resolved), "dataset": alias or None}
            kind = "directory"
        elif resolved.is_file():
            if resolved.stat().st_size <= 0:
                return {"ok": False, "code": "dataset-path-empty", "error": f"数据集文件为空：{resolved}", "path": str(resolved), "dataset": alias or None}
            kind = "file"
        else:
            return {"ok": False, "code": "dataset-path-invalid", "error": f"数据集路径不是文件或目录：{resolved}", "path": str(resolved), "dataset": alias or None}
    except OSError as exc:
        return {"ok": False, "code": "dataset-path-inaccessible", "error": f"无法访问数据集路径：{resolved}（{exc}）", "path": str(resolved), "dataset": alias or None}
    return {"ok": True, "dataset": alias or None, "path": str(resolved), "kind": kind, "layout_checked": False}


__all__ = ["dataset_catalog_payload", "dataset_fact_payload", "dataset_preflight_payload"]
