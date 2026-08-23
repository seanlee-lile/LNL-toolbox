from __future__ import annotations

"""Stable final experiment result contract."""

from hashlib import sha256
from datetime import datetime
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


def load_metric_history(run_dir: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read epoch metrics while tolerating one actively-written trailing line."""

    path = Path(run_dir).expanduser().resolve() / "metrics.jsonl"
    if not path.is_file():
        return [], []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    last_nonempty = nonempty[-1] if nonempty else -1
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            if index != last_nonempty:
                errors.append(f"metrics.jsonl line {index + 1}: {exc.msg}")
            continue
        if not isinstance(value, Mapping):
            errors.append(f"metrics.jsonl line {index + 1}: expected an object")
            continue
        rows.append(dict(value))
    return rows, errors


def _finite_metrics(value: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(name): float(metric)
        for name, metric in value.items()
        if isinstance(metric, (int, float))
        and not isinstance(metric, bool)
        and math.isfinite(float(metric))
    }


def inspect_run_result(run_dir: str | Path) -> dict[str, Any]:
    """Return a JSON-safe summary and metric history for one run directory."""

    root = Path(run_dir).expanduser().resolve()
    rows, errors = load_metric_history(root)
    final = load_final_result(root) or {}
    last = rows[-1] if rows else {}
    primary = final.get("primary_metric", {}) or {}
    final_metrics = final.get("metrics", final) if final else {}
    if not isinstance(final_metrics, Mapping):
        final_metrics = {}
    current_epoch = max(
        (
            int(row["epoch"])
            for row in rows
            if isinstance(row.get("epoch"), (int, float))
            and not isinstance(row.get("epoch"), bool)
        ),
        default=None,
    )
    metric_names = sorted(
        {
            name
            for row in rows
            for name in _finite_metrics(row)
            if name not in {"epoch", "global_step", "step"}
        }
    )
    series = [
        {
            "epoch": row.get("epoch"),
            "global_step": row.get("global_step", row.get("step")),
            "metrics": _finite_metrics(row),
        }
        for row in rows
    ]
    return {
        "run_id": str(final.get("run_id", root.name)),
        "run_dir": str(root),
        "method": final.get("method", last.get("method", "-")),
        "seed": final.get("seed"),
        "status": final.get("status", "incomplete" if rows else "pending"),
        "current_epoch": current_epoch,
        "best_epoch": final.get("best_epoch", (final.get("selection", {}) or {}).get("best_epoch")),
        "primary_metric": {
            "name": primary.get("name"),
            "value": primary.get("value"),
        },
        "final_metrics": _finite_metrics(final_metrics),
        "metric_names": metric_names,
        "series": series,
        "errors": errors,
    }


def discover_run_results(path: str | Path) -> list[dict[str, Any]]:
    """Discover complete or in-progress run directories below a selected path."""

    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"run path does not exist: {root}")
    if root.is_file():
        root = root.parent
    directories = {
        candidate.parent
        for pattern in ("metrics.jsonl", "final_metrics.json")
        for candidate in root.rglob(pattern)
    }
    if (root / "metrics.jsonl").is_file() or (root / "final_metrics.json").is_file():
        directories.add(root)
    manifest_runs: dict[Path, dict[str, Any]] = {}
    for manifest_path in root.rglob("sweep_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in manifest.get("runs", []):
            if not isinstance(item, Mapping) or not item.get("run_dir"):
                continue
            manifest_runs[Path(item["run_dir"]).expanduser().resolve()] = {
                "sweep_id": manifest.get("sweep_id", manifest_path.parent.name),
                "status": item.get("status", "pending"),
                "seed": item.get("seed"),
                "overrides": dict(item.get("overrides", {}) or {}),
            }
    results = []
    for directory in sorted(directories):
        result = inspect_run_result(directory)
        if directory in manifest_runs:
            result.update(manifest_runs[directory])
        results.append(result)
    return results


def _component_name(value: object) -> object:
    if isinstance(value, Mapping):
        return value.get("name", value.get("type", "-"))
    return value if value not in (None, "") else "-"


def _checkpoint_progress(value: object) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    phases: list[dict[str, object]] = []
    progress: list[dict[str, object]] = []
    skipped = {"model", "optimizer", "scheduler", "rng_state", "ema"}

    def visit(item: object, prefix: str = "", depth: int = 0) -> None:
        if depth > 6 or not isinstance(item, Mapping):
            return
        for raw_key, child in item.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            lowered = key.lower()
            if isinstance(child, (str, int, float, bool)) or child is None:
                if lowered == "phase" and child not in (None, ""):
                    phases.append({"path": path, "value": child})
                if (
                    lowered in {"global_step", "step", "cycle", "stopped"}
                    or ("completed" in lowered and ("epoch" in lowered or "step" in lowered))
                    or lowered.startswith("optimizer_steps")
                ):
                    progress.append({"path": path, "value": child})
            elif lowered not in skipped:
                visit(child, path, depth + 1)

    visit(value)
    unique_phases = []
    seen_phases: set[tuple[str, str]] = set()
    for item in phases:
        identity = (str(item["path"]), str(item["value"]))
        if identity not in seen_phases:
            unique_phases.append(item)
            seen_phases.add(identity)
    return unique_phases, progress[:40]


def _epoch_schedule(config: Mapping[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def visit(item: object, prefix: str = "", depth: int = 0) -> None:
        if depth > 5 or not isinstance(item, Mapping):
            return
        for raw_key, child in item.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            lowered = key.lower()
            if (
                isinstance(child, (int, float))
                and not isinstance(child, bool)
                and (lowered == "epochs" or lowered.endswith("_epochs"))
            ):
                rows.append({"path": path, "value": child})
            elif isinstance(child, Mapping):
                visit(child, path, depth + 1)

    visit(config)
    return rows


def inspect_resume_run(
    run_dir: str | Path,
    checkpoint: str = "last",
) -> dict[str, Any]:
    """Inspect one run without mutating it or loading training state into a model."""

    if checkpoint not in {"last", "best"}:
        raise ValueError("checkpoint must be 'last' or 'best'")
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {root}")

    inventory = []
    for item in sorted(root.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())):
        stat = item.stat()
        inventory.append({
            "name": item.name,
            "type": "directory" if item.is_dir() else "file",
            "size_bytes": None if item.is_dir() else stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
        })

    config_path = root / "resolved_config.yaml"
    config: dict[str, Any] = {}
    config_yaml = ""
    errors: list[str] = []
    if config_path.is_file():
        try:
            import yaml

            config_yaml = config_path.read_text(encoding="utf-8")
            loaded = yaml.safe_load(config_yaml)
            if not isinstance(loaded, Mapping):
                raise ValueError("resolved_config.yaml must contain a mapping")
            config = dict(loaded)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"configuration: {exc}")
    else:
        errors.append("missing resolved_config.yaml")

    checkpoint_rows = []
    for path in sorted(root.glob("*.pt")):
        stat = path.stat()
        checkpoint_rows.append({
            "name": path.name,
            "selected": path.stem == checkpoint,
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
        })
    checkpoint_path = root / f"{checkpoint}.pt"
    payload: dict[str, Any] = {}
    if checkpoint_path.is_file():
        try:
            from lnl_toolbox.training.checkpoint import read_checkpoint

            payload = read_checkpoint(checkpoint_path, "cpu")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"checkpoint: {exc}")
    else:
        errors.append(f"missing {checkpoint}.pt")

    history, history_errors = load_metric_history(root)
    errors.extend(history_errors)
    latest = history[-1] if history else {}
    metric_epochs = [
        int(row["epoch"])
        for row in history
        if isinstance(row.get("epoch"), (int, float))
        and not isinstance(row.get("epoch"), bool)
    ]
    completed_epoch = payload.get("completed_epoch")
    current_epoch = max(metric_epochs, default=None)
    if current_epoch is None and isinstance(completed_epoch, (int, float)):
        current_epoch = int(completed_epoch) + 1

    phases, checkpoint_progress = _checkpoint_progress(payload)
    latest_phase = latest.get("phase")
    if latest_phase not in (None, ""):
        phases.insert(0, {"path": "metrics.latest.phase", "value": latest_phase})
    phase = phases[0]["value"] if phases else "unknown"
    schedule = _epoch_schedule(config)
    trainer = config.get("trainer", {}) if isinstance(config.get("trainer"), Mapping) else {}
    target_epoch = trainer.get("epochs")
    completed = is_completed_result(root)
    target_reached = (
        isinstance(current_epoch, (int, float))
        and isinstance(target_epoch, (int, float))
        and current_epoch >= target_epoch
    )
    completed = completed or target_reached

    try:
        from lnl_toolbox.training.runners import resolve_runner

        runner = resolve_runner(config).name if config else "-"
    except (KeyError, TypeError, ValueError):
        runner = str((config.get("execution", {}) or {}).get("runner", "-")) if config else "-"

    config_summary = {
        "method": method_name(config, runner) if config else payload.get("method", "-"),
        "runner": runner,
        "seed": config.get("seed", "-") if config else "-",
        "data": _component_name(config.get("data")) if config else "-",
        "noise": _component_name(config.get("noise", "clean")) if config else "-",
        "model": _component_name(config.get("model")) if config else "-",
        "optimizer": _component_name(config.get("optimizer")) if config else "-",
        "target_epochs": target_epoch if target_epoch is not None else "-",
    }
    best = {
        "best_epoch": payload.get("best_epoch", latest.get("best_epoch")),
        "best_validation_accuracy": payload.get(
            "best_validation_accuracy", latest.get("best_validation_accuracy")
        ),
        "selection_split": payload.get("selection_split", latest.get("selection_split")),
        "best_selection_accuracy": payload.get(
            "best_selection_accuracy", latest.get("best_selection_accuracy")
        ),
    }
    latest_metrics = _finite_metrics(latest)
    resumable = not errors and not completed
    status = "ready" if resumable else "completed" if completed and not errors else "blocked"
    if completed:
        errors.append(
            "configured target epoch has already been reached; resume would be a no-op"
            if target_reached
            else "run is already completed; resume would be a no-op"
        )
    return {
        "run_dir": str(root),
        "status": status,
        "resumable": resumable,
        "completed": completed,
        "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "config_summary": config_summary,
        "config_yaml": config_yaml,
        "current_epoch": current_epoch,
        "target_epoch": target_epoch,
        "phase": phase,
        "phase_sources": phases,
        "epoch_schedule": schedule,
        "checkpoint_progress": checkpoint_progress,
        "best": best,
        "latest_metrics": latest_metrics,
        "checkpoints": checkpoint_rows,
        "files": inventory,
        "errors": errors,
    }


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
    "discover_run_results",
    "finalize_result",
    "inspect_run_result",
    "inspect_resume_run",
    "is_completed_result",
    "load_final_result",
    "load_metric_history",
]
