from __future__ import annotations

"""Aggregate Result Contract files and audit grouped experiment fairness."""

import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Mapping, Sequence


DEFAULT_GROUP_BY = ("method", "noise.rate", "primary_metric.name")
DEFAULT_REQUIRE_EQUAL = (
    "dataset",
    "model",
    "augmentation",
    "evaluation_protocol",
    "selection_split",
)


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


def _nested(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for component in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(component)
    return value


def _field_value(record: Mapping[str, Any], field: str) -> Any:
    config = record["config"]
    result = record["result"]
    data = config.get("data", {}) or {}
    if field == "method":
        return result.get("method", config.get("method", "unknown"))
    if field == "dataset":
        return data.get("name") if isinstance(data, Mapping) else None
    if field == "model":
        return config.get("model") or "runner-owned"
    if field == "augmentation":
        if not isinstance(data, Mapping):
            return None
        return {
            key: data.get(key)
            for key in sorted(data)
            if "augment" in str(key).lower() or "transform" in str(key).lower()
        }
    if field == "evaluation_protocol":
        return config.get("evaluation", {}) or {}
    if field == "selection_split":
        selection = result.get("selection", {}) or {}
        return selection.get("split", result.get("selection_split"))
    if field == "noise_manifest":
        metrics = result.get("metrics", {}) or {}
        metadata = metrics.get("noise", {}) if isinstance(metrics, Mapping) else {}
        if isinstance(metadata, Mapping):
            return metadata.get("mapping_hash") or metadata.get("manifest_sha256")
        noise = result.get("noise", {}) or {}
        return noise.get("mapping_hash") if isinstance(noise, Mapping) else None
    if field == "seed":
        return config.get("seed", result.get("seed"))
    if field == "test_selection_leakage":
        return bool(result.get("test_selection_leakage", False))
    if field.startswith("primary_metric."):
        return _nested(result, field)
    value = _nested(config, field)
    if value is not None:
        return value
    return _nested(result, field)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _normalize_fields(
    fields: Sequence[str] | None, defaults: tuple[str, ...]
) -> tuple[str, ...]:
    values = defaults if fields is None else tuple(str(item).strip() for item in fields)
    if not values or any(not value for value in values):
        raise ValueError("comparison fields must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("comparison fields must be unique")
    return tuple(values)


def _failed_manifest_runs(root: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(root.rglob("sweep_manifest.json")):
        manifest = _load_mapping(path)
        for item in manifest.get("runs", []):
            if isinstance(item, Mapping) and item.get("status") == "failed":
                failures.append(str(item.get("run_dir", "unknown")))
    return failures


def compare_runs(
    root: str | Path,
    *,
    group_by: Sequence[str] | None = None,
    require_equal: Sequence[str] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    directory = Path(root).expanduser().resolve()
    records = collect_run_results(directory)
    grouping = _normalize_fields(group_by, DEFAULT_GROUP_BY)
    invariants = _normalize_fields(require_equal, DEFAULT_REQUIRE_EQUAL)
    # Metrics with different meanings must never share an aggregate.
    aggregate_fields = grouping
    if "primary_metric.name" not in aggregate_fields:
        aggregate_fields += ("primary_metric.name",)

    valid: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    leakage_records: list[dict[str, Any]] = []
    for record in records:
        result = record["result"]
        primary = result.get("primary_metric", {}) or {}
        value = primary.get("value")
        if result.get("status") != "completed" or not isinstance(value, (int, float)) \
                or isinstance(value, bool) or not math.isfinite(float(value)):
            failed.append(record)
            continue
        if _field_value(record, "test_selection_leakage"):
            leakage_records.append(record)
            if strict:
                excluded.append(
                    {"run_dir": record["run_dir"], "reason": "test_selection_leakage"}
                )
                continue
        valid.append(record)

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    raw_keys: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in valid:
        values = {field: _field_value(record, field) for field in aggregate_fields}
        key = tuple(_canonical(values[field]) for field in aggregate_fields)
        groups.setdefault(key, []).append(record)
        raw_keys[key] = values

    summaries: list[dict[str, Any]] = []
    for key in sorted(groups):
        members = groups[key]
        values = [float(item["result"]["primary_metric"]["value"]) for item in members]
        group = raw_keys[key]
        noise_name = _field_value(members[0], "noise.name") or "clean"
        noise_rate = _field_value(members[0], "noise.rate")
        summaries.append(
            {
                "group": group,
                "method": group.get("method", "mixed"),
                "noise": f"{noise_name}:{noise_rate if noise_rate is not None else '-'}",
                "metric": group["primary_metric.name"],
                "n": len(values),
                "mean": mean(values),
                "std": pstdev(values),
                "median": median(values),
                "min": min(values),
                "max": max(values),
            }
        )

    # Fairness cohorts permit method to vary, while preserving every other
    # explicit research dimension. This supports method-vs-method comparison.
    cohort_fields = tuple(field for field in grouping if field != "method")
    cohorts: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    cohort_values: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in valid:
        values = {field: _field_value(record, field) for field in cohort_fields}
        key = tuple(_canonical(values[field]) for field in cohort_fields)
        cohorts.setdefault(key, []).append(record)
        cohort_values[key] = values

    findings: list[dict[str, Any]] = []
    for key in sorted(cohorts):
        members = cohorts[key]
        label = cohort_values[key]
        for field in invariants:
            if field in grouping:
                continue
            observed = {_canonical(_field_value(item, field)) for item in members}
            if len(observed) > 1:
                findings.append(
                    {
                        "field": field,
                        "group": label,
                        "values": sorted(observed),
                        "message": f"{field} differs within comparable group {label}",
                    }
                )

        by_seed: dict[str, list[dict[str, Any]]] = {}
        for member in members:
            by_seed.setdefault(_canonical(_field_value(member, "seed")), []).append(member)
        for seed, seed_members in sorted(by_seed.items()):
            methods = {_canonical(_field_value(item, "method")) for item in seed_members}
            manifests = {
                _canonical(_field_value(item, "noise_manifest")) for item in seed_members
            }
            if len(methods) > 1 and len(manifests) > 1:
                findings.append(
                    {
                        "field": "noise_manifest",
                        "group": {**label, "seed": json.loads(seed)},
                        "values": sorted(manifests),
                        "message": (
                            "Noise Manifests differ across methods for the same seed "
                            f"and comparable conditions {label}"
                        ),
                    }
                )

    warnings = [f"WARNING: {item['message']}." for item in findings]
    if leakage_records:
        action = "excluded" if strict else "included with warning"
        warnings.append(
            f"WARNING: {len(leakage_records)} run(s) report test-selection leakage; {action}."
        )
    failed_paths = {item["run_dir"] for item in failed}
    failed_paths.update(_failed_manifest_runs(directory))
    compatibility = {
        field: "warning" if any(item["field"] == field for item in findings) else "consistent"
        for field in (*invariants, "noise_manifest")
    }
    compatibility["test_selection_leakage"] = "warning" if leakage_records else "none"
    return {
        "schema_version": 2,
        "root": str(directory),
        "group_by": list(aggregate_fields),
        "require_equal": list(invariants),
        "summaries": summaries,
        "compatibility": compatibility,
        "compatibility_findings": findings,
        "warnings": warnings,
        "excluded_runs": excluded,
        "failed_runs": sorted(failed_paths),
        "run_count": len(records),
        "strict": bool(strict),
    }


def write_report(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    columns = ["method", "noise", "metric", "n", "mean", "std", "median", "min", "max"]
    csv_path = root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
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
    markdown.extend(["", "## Compatibility", ""])
    for field, status in summary.get("compatibility", {}).items():
        markdown.append(f"- {field}: {status}")
    markdown.extend(["", "## Warnings", ""])
    markdown.extend(f"- {value}" for value in summary["warnings"])
    if not summary["warnings"]:
        markdown.append("- None")
    markdown.extend(["", "## Excluded runs", ""])
    markdown.extend(
        f"- {item['run_dir']}: {item['reason']}"
        for item in summary.get("excluded_runs", [])
    )
    if not summary.get("excluded_runs"):
        markdown.append("- None")
    markdown.extend(["", "## Failed runs", ""])
    markdown.extend(f"- {value}" for value in summary["failed_runs"])
    if not summary["failed_runs"]:
        markdown.append("- None")
    report_path = root / "report.md"
    report_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {"report": report_path, "csv": csv_path, "json": json_path}


__all__ = [
    "DEFAULT_GROUP_BY",
    "DEFAULT_REQUIRE_EQUAL",
    "collect_run_results",
    "compare_runs",
    "write_report",
]
