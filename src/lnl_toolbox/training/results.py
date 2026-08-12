from __future__ import annotations

"""Stable final experiment result contract."""

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from lnl_toolbox.training.planning import method_name


SCHEMA_VERSION = 1


def config_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def load_final_result(run_dir: str | Path) -> dict[str, Any] | None:
    path = Path(run_dir) / "final_metrics.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"final_metrics.json must contain an object: {path}")
    return dict(value)


def is_completed_result(run_dir: str | Path) -> bool:
    value = load_final_result(run_dir)
    if not value:
        return False
    status = value.get("status")
    if status is not None:
        return bool(status == "completed" and value.get("completed", True))
    return any(
        "test" in str(name).lower()
        and "accuracy" in str(name).lower()
        and isinstance(metric, (int, float))
        and not isinstance(metric, bool)
        and math.isfinite(float(metric))
        for name, metric in value.items()
    )


def _last_metric_row(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return {}
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {}
    value = json.loads(rows[-1])
    return dict(value) if isinstance(value, Mapping) else {}


def _primary_metric(raw: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = config.get("evaluation", {}) or {}
    requested = str(evaluation.get("primary", "accuracy"))
    candidates = [
        requested if requested.startswith("test_") else f"test_{requested}",
        "test_accuracy",
        "clean_test_accuracy",
        "test_accuracy_ensemble",
        "test_mean_peer_accuracy",
    ]
    for name in candidates:
        value = raw.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return {"name": name, "value": float(value)}
    for name, value in sorted(raw.items()):
        lowered = str(name).lower()
        if (
            "test" in lowered
            and "accuracy" in lowered
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return {"name": str(name), "value": float(value)}
    raise ValueError("completed run does not expose a finite primary test metric")


def finalize_result(
    run_dir: str | Path,
    config: Mapping[str, Any],
    *,
    runner: str,
    recipe: str | None = None,
) -> dict[str, Any]:
    """Upgrade runner output in place while preserving legacy metric keys."""

    root = Path(run_dir).resolve()
    raw = load_final_result(root) or _last_metric_row(root)
    algorithm_metrics = dict(raw.get("metrics", raw))
    evaluation = config.get("evaluation", {}) or {}
    selection_split = str(
        raw.get("selection_split", evaluation.get("selection_split", "validation"))
    )
    best_epoch = raw.get("best_epoch")
    selection_metric = str(evaluation.get("primary", "accuracy"))
    contract: dict[str, Any] = dict(raw)
    contract.update(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": root.name,
            "event": "final",
            "status": "completed",
            "completed": True,
            "method": method_name(config, runner),
            "runner": runner,
            "recipe": recipe or "custom",
            "seed": int(config.get("seed", 1)),
            "config_hash": config_hash(config),
            "primary_metric": _primary_metric(raw, config),
            "selection": {
                "split": selection_split,
                "metric": selection_metric,
                "best_epoch": best_epoch,
            },
            "test_selection_leakage": bool(
                raw.get("test_selection_leakage", selection_split == "test")
            ),
            "metrics": algorithm_metrics,
        }
    )
    destination = root / "final_metrics.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return contract


__all__ = [
    "SCHEMA_VERSION",
    "config_hash",
    "finalize_result",
    "is_completed_result",
    "load_final_result",
]
