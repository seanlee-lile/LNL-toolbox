"""Read-only data facts for the Scratch editor."""

from __future__ import annotations

from typing import Any


def _data_service() -> Any:
    from importlib import import_module

    module = import_module("lnl_toolbox.training.data_service")
    return module.DataService()


def _known(value: object) -> bool | None:
    if value in {"available", "true", True}:
        return True
    if value in {"unavailable", "false", False}:
        return False
    return None


def _record_state(service: Any, alias: str) -> str | None:
    try:
        return service.catalog.get(alias).effective_state
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _facts_from_report(service: Any, report: Any) -> dict[str, object]:
    profile = report.profile
    profile_data = None if profile is None else profile.to_dict()
    clean = None if profile is None else _known(profile.clean_train_labels.value)
    noisy = None
    if profile is not None:
        noisy = {"noisy": True, "clean": False}.get(profile.noise.status.value)
    state = _record_state(service, report.name)
    methods = ["none"]
    if clean is True:
        methods.extend(("symmetric", "pairflip", "class_conditional", "instance_dependent"))
    if noisy is True:
        methods.append("external")
    return {
        "alias": report.name,
        "name": report.name,
        "path": report.location,
        "adapter": report.adapter,
        "status": report.status,
        "train_samples": report.train_samples,
        "test_samples": report.test_samples,
        "num_classes": report.classes if report.classes is not None else (None if profile is None else profile.num_classes),
        "input_shape": None if profile is None or profile.input_shape is None else list(profile.input_shape),
        "has_clean_target": clean,
        "has_noisy_target": noisy,
        "has_sample_index": None if profile is None else _known(profile.stable_indices.value),
        "layout_validated": state in {"layout_validated", "training_verified"},
        "training_verified": state == "training_verified",
        "inspect_status": state or report.status,
        "noise_methods": methods,
        "profile": profile_data,
        "error": report.error,
    }


def dataset_catalog_payload() -> dict[str, object]:
    """Return registered dataset facts without probing or guessing paths."""

    service = _data_service()
    records = {record.alias for record in service.catalog.records()}
    facts = [
        _facts_from_report(service, report)
        for report in service.list_datasets()
        if report.name in records
    ]
    return {"datasets": facts}


def dataset_fact_payload(alias: object) -> dict[str, object]:
    """Return one registered dataset's current, persisted inspection facts."""

    name = str(alias).strip()
    if not name:
        raise ValueError("dataset alias is required")
    service = _data_service()
    report = service.status(name)
    if report.name not in {record.alias for record in service.catalog.records()}:
        raise ValueError(f"dataset is not registered: {name}")
    return _facts_from_report(service, report)


__all__ = ["dataset_catalog_payload", "dataset_fact_payload"]
