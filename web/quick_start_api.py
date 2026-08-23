from __future__ import annotations

"""Thin JSON boundary for the Quick Start service."""

from collections.abc import Mapping
from typing import Any

from lnl_toolbox.quickstart.models import QuickStartNoiseSelection
from lnl_toolbox.quickstart.service import QuickStartService


SERVICE = QuickStartService()


def _body_mapping(body: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(body, Mapping):
        raise TypeError("Quick Start request body must be a JSON object")
    return body


def _required_text(body: Mapping[str, object], key: str) -> str:
    value = str(body.get(key, "")).strip()
    if not value:
        raise ValueError(f"Quick Start request requires {key}")
    return value


def _selection(body: Mapping[str, object]) -> QuickStartNoiseSelection:
    raw = body.get("noise")
    if not isinstance(raw, Mapping):
        raise ValueError("Quick Start request requires noise selection")
    rate = raw.get("rate")
    return QuickStartNoiseSelection(
        kind=str(raw.get("kind", "clean")),
        key=str(raw.get("key", "clean")),
        rate=None if rate in {None, ""} else float(rate),
        seed=None if raw.get("seed") in {None, ""} else int(raw["seed"]),
    )


def probe_payload(body: Mapping[str, object]) -> dict[str, object]:
    body = _body_mapping(body)
    result = SERVICE.probe(_required_text(body, "path"))
    return {
        "path": result.path,
        "status": result.status,
        "existing_alias": result.existing_alias,
        "candidates": [
            {"adapter": item.adapter, "confidence": item.confidence, "reason": item.reason, "data": item.data}
            for item in result.candidates
        ],
    }


def register_payload(body: Mapping[str, object]) -> dict[str, object]:
    body = _body_mapping(body)
    value = SERVICE.register_and_inspect(
        _required_text(body, "path"),
        selected_adapter=(None if not body.get("adapter") else str(body["adapter"])),
    )
    if hasattr(value, "to_dict"):
        return {"kind": "dataset", "dataset": value.to_dict()}
    return {
        "kind": "probe",
        "path": value.path,
        "status": value.status,
        "existing_alias": value.existing_alias,
        "candidates": [
            {"adapter": item.adapter, "confidence": item.confidence, "reason": item.reason, "data": item.data}
            for item in value.candidates
        ],
    }


def noise_options_payload(alias: str) -> dict[str, object]:
    return SERVICE.noise_options(str(alias).strip())


def method_options_payload(body: Mapping[str, object]) -> dict[str, object]:
    body = _body_mapping(body)
    options = SERVICE.method_options(_required_text(body, "dataset"), _selection(body))
    return {"methods": [item.to_dict() for item in options]}


def plan_payload(body: Mapping[str, object]) -> dict[str, object]:
    body = _body_mapping(body)
    user_inputs = body.get("user_inputs")
    if user_inputs is not None and not isinstance(user_inputs, Mapping):
        raise ValueError("user_inputs must be a JSON object")
    plan = SERVICE.build_plan(
        dataset_alias=_required_text(body, "dataset"),
        noise_selection=_selection(body),
        paper_id=_required_text(body, "paper_id"),
        user_inputs=user_inputs,
    )
    return plan.to_dict()


__all__ = [
    "method_options_payload",
    "noise_options_payload",
    "plan_payload",
    "probe_payload",
    "register_payload",
]
