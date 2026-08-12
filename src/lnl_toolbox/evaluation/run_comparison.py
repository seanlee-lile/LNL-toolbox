from __future__ import annotations

"""Aggregate Result Contract files and audit experiment comparability."""

import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Mapping

from lnl_toolbox.training.runners import resolve_runner


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a mapping in {path}")
    return dict(value)


def collect_run_results(root: str | Path) -> list[dict[str, Any]]:
    directory = Path(root).expanduser().resolve()
    paths = sorted(directory.rglob("final_metrics.json"))
    records: list[dict[str, Any]] = []
    for path in paths:
        result = _load_mapping(path)
        config_path = path.parent / "resolved_config.yaml"
        config = _load_mapping(config_path) if config_path.is_file() else {}
        records.append({"run_dir": str(path.parent), "result": result, "config": config})
    if not records:
        raise ValueError(f"no final_metrics.json files found below {directory}")
    return records


def _noise_name(config: Mapping[str, Any]) -> str:
    noise = config.get("noise", {}) or {}
    if not isinstance(noise, Mapping) or not noise:
        return "clean"
    return f"{noise.get('name', noise.get('type', 'configured'))}:{noise.get('rate', '-')}"


def _fairness_value(record: Mapping[str, Any], field: str) -> Any:
    config = record["config"]
    result = record["result"]
    data = config.get("data", {}) or {}
    noise = config.get("noise", {}) or {}
    if field == "dataset":
        return data.get("name")
    if field == "model":
        return config.get("model") or "runner-owned"
    if field == "augmentation":
        return data.get("augment")
    if field == "training_budget":
        return resolve_runner(config).describe(config).training_budget if config else None
    if field == "noise_rate":
        return noise.get("rate") if isinstance(noise, Mapping) else None
    if field == "noise_manifest":
        metrics = result.get("metrics", {}) or {}
        metadata = metrics.get("noise", {}) if isinstance(metrics, Mapping) else {}
        return metadata.get("mapping_hash") if isinstance(metadata, Mapping) else None
    if field == "selection_split":
        return (result.get("selection", {}) or {}).get("split")
    if field == "test_selection_leakage":
        return bool(result.get("test_selection_leakage", False))
    raise KeyError(field)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def compare_runs(root: str | Path) -> dict[str, Any]:
    records = collect_run_results(root)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    for record in records:
        result = record["result"]
        if result.get("status") != "completed":
            failures.append(record)
            continue
        primary = result.get("primary_metric", {}) or {}
        value = primary.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            failures.append(record)
            continue
        key = (
            str(result.get("method", "unknown")),
            _noise_name(record["config"]),
            str(primary.get("name", "unknown")),
        )
        groups.setdefault(key, []).append(record)

    summaries = []
    for (method, noise, metric), members in sorted(groups.items()):
        values = [float(item["result"]["primary_metric"]["value"]) for item in members]
        summaries.append(
            {
                "method": method,
                "noise": noise,
                "metric": metric,
                "n": len(values),
                "mean": mean(values),
                "std": pstdev(values),
                "median": median(values),
                "min": min(values),
                "max": max(values),
            }
        )

    warnings = []
    labels = {
        "dataset": "dataset configurations differ",
        "model": "model configurations differ",
        "augmentation": "augmentation configurations differ",
        "training_budget": "training budgets differ",
        "noise_rate": "noise rates differ",
        "noise_manifest": "Noise Manifests differ",
        "selection_split": "selection splits differ",
        "test_selection_leakage": "test-selection leakage flags differ",
    }
    completed = [item for members in groups.values() for item in members]
    for field, message in labels.items():
        values = {_canonical(_fairness_value(item, field)) for item in completed}
        if len(values) > 1:
            warnings.append(f"WARNING: {message}.")
    return {
        "schema_version": 1,
        "root": str(Path(root).expanduser().resolve()),
        "summaries": summaries,
        "warnings": warnings,
        "failed_runs": [item["run_dir"] for item in failures],
        "run_count": len(records),
    }


def write_report(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    columns = ["method", "noise", "metric", "n", "mean", "std", "median", "min", "max"]
    csv_path = root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary["summaries"])
    markdown = [
        "# Experiment comparison",
        "",
        "| Method | Noise | Metric | N | Mean | Std | Median | Min | Max |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["summaries"]:
        markdown.append(
            f"| {row['method']} | {row['noise']} | {row['metric']} | {row['n']} | "
            f"{row['mean']:.6f} | {row['std']:.6f} | {row['median']:.6f} | "
            f"{row['min']:.6f} | {row['max']:.6f} |"
        )
    markdown.extend(["", "## Fairness warnings", ""])
    markdown.extend(f"- {value}" for value in summary["warnings"])
    if not summary["warnings"]:
        markdown.append("- None")
    markdown.extend(["", "## Failed runs", ""])
    markdown.extend(f"- {value}" for value in summary["failed_runs"])
    if not summary["failed_runs"]:
        markdown.append("- None")
    report_path = root / "report.md"
    report_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {"report": report_path, "csv": csv_path, "json": json_path}


__all__ = ["collect_run_results", "compare_runs", "write_report"]
