from __future__ import annotations

"""Deterministic, sequential, resumable experiment sweeps."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from lnl_toolbox.core.config_overrides import apply_override
from lnl_toolbox.training.results import config_hash, is_completed_result
from lnl_toolbox.training.runners import resolve_runner
from lnl_toolbox.training.service import ExperimentService


@dataclass(frozen=True, slots=True)
class PlannedRun:
    seed: int
    overrides: tuple[tuple[str, Any], ...]
    config_hash: str
    run_id: str

    @property
    def override_mapping(self) -> dict[str, Any]:
        return dict(self.overrides)


@dataclass(frozen=True, slots=True)
class SweepPlan:
    root: Path
    recipe: str
    seeds: tuple[int, ...]
    matrix: tuple[tuple[str, tuple[Any, ...]], ...]
    runs: tuple[PlannedRun, ...]
    base_config_hash: str
    sweep_hash: str


@dataclass(frozen=True, slots=True)
class SweepResult:
    root: Path
    manifest: dict[str, Any]

    @property
    def failed(self) -> int:
        return sum(item["status"] == "failed" for item in self.manifest["runs"])

    @property
    def completed(self) -> int:
        return sum(item["status"] == "completed" for item in self.manifest["runs"])

    @property
    def skipped(self) -> int:
        return int(self.manifest.get("skipped_count", 0))


def _normalized_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized or any(seed < 0 for seed in normalized):
        raise ValueError("sweep seeds must be a non-empty list of non-negative integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("sweep seeds must be unique")
    return normalized


def _normalized_matrix(
    matrix: Mapping[str, Sequence[Any]] | None,
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    dimensions: list[tuple[str, tuple[Any, ...]]] = []
    for path in sorted((matrix or {}).keys()):
        values = (matrix or {})[path]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"sweep matrix {path!r} must be a sequence of values")
        normalized = tuple(values)
        if not normalized:
            raise ValueError(f"sweep matrix {path!r} must not be empty")
        dimensions.append((str(path), normalized))
    return tuple(dimensions)


def _default_root(config: Mapping[str, Any], recipe: str | None, identity: str) -> Path:
    del config
    name = (recipe or "custom").strip().lower().replace("_", "-")
    return Path("artifacts/sweeps") / f"{name}-{identity[:10]}"


def resolve_planned_config(
    config: Mapping[str, Any], planned: PlannedRun
) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    for path, value in planned.overrides:
        resolved = apply_override(resolved, path, value)
    resolved["seed"] = planned.seed
    if config_hash(resolved) != planned.config_hash:
        raise RuntimeError("planned sweep configuration hash changed during resolution")
    return resolved


def plan_sweep(
    config: Mapping[str, Any],
    seeds: Sequence[int],
    *,
    matrix: Mapping[str, Sequence[Any]] | None = None,
    output_dir: str | Path | None = None,
    recipe: str | None = None,
) -> SweepPlan:
    normalized_seeds = _normalized_seeds(seeds)
    dimensions = _normalized_matrix(matrix)
    base = deepcopy(dict(config))
    combinations = product(*(values for _path, values in dimensions)) if dimensions else [()]
    pending: list[tuple[int, tuple[tuple[str, Any], ...], str]] = []
    for combination in combinations:
        overrides = tuple(
            (dimensions[index][0], value) for index, value in enumerate(combination)
        )
        candidate = deepcopy(base)
        for path, value in overrides:
            candidate = apply_override(candidate, path, value)
        for seed in normalized_seeds:
            resolved = deepcopy(candidate)
            resolved["seed"] = seed
            pending.append((seed, overrides, config_hash(resolved)))
    hashes = [identity for _seed, _overrides, identity in pending]
    if len(set(hashes)) != len(hashes):
        raise ValueError("sweep matrix produces duplicate resolved configurations")
    runs = tuple(
        PlannedRun(
            seed=seed,
            overrides=overrides,
            config_hash=identity,
            run_id=f"run-{index:04d}-{identity[:10]}",
        )
        for index, (seed, overrides, identity) in enumerate(pending, start=1)
    )
    matrix_value = {path: list(values) for path, values in dimensions}
    sweep_identity = config_hash(
        {
            "base_config_hash": config_hash(base),
            "matrix": matrix_value,
            "seeds": list(normalized_seeds),
        }
    )
    # Preserve the v1 default directory for seed-only sweeps so an old
    # manifest remains discoverable and resumable without --output-dir.
    root_identity = sweep_identity if dimensions else config_hash(base)
    root = Path(output_dir or _default_root(base, recipe, root_identity)).expanduser().resolve()
    return SweepPlan(
        root=root,
        recipe=recipe or "custom",
        seeds=normalized_seeds,
        matrix=dimensions,
        runs=runs,
        base_config_hash=config_hash(base),
        sweep_hash=sweep_identity,
    )


def _write_manifest(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _new_manifest(plan: SweepPlan) -> dict[str, Any]:
    runs_root = plan.root / "runs"
    return {
        "schema_version": 2,
        "sweep_id": plan.root.name,
        "recipe": plan.recipe,
        "config_hash": plan.base_config_hash,
        "sweep_hash": plan.sweep_hash,
        "seeds": list(plan.seeds),
        "matrix": {path: list(values) for path, values in plan.matrix},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "runs": [
            {
                "run_id": run.run_id,
                "seed": run.seed,
                "overrides": run.override_mapping,
                "config_hash": run.config_hash,
                "status": "pending",
                "run_dir": str((runs_root / run.run_id).resolve()),
                "error": None,
            }
            for run in plan.runs
        ],
    }


def _validate_existing_manifest(manifest: Mapping[str, Any], plan: SweepPlan) -> None:
    schema = int(manifest.get("schema_version", 1))
    if schema == 1:
        if plan.matrix:
            raise ValueError("legacy sweep manifest cannot be resumed with a matrix")
        if manifest.get("config_hash") != plan.base_config_hash:
            raise ValueError("sweep configuration does not match existing manifest")
        if tuple(item["seed"] for item in manifest.get("runs", [])) != plan.seeds:
            raise ValueError("sweep seeds do not match existing manifest")
        return
    if schema != 2:
        raise ValueError(f"unsupported sweep manifest schema: {schema}")
    if manifest.get("sweep_hash") != plan.sweep_hash:
        raise ValueError("sweep plan does not match existing manifest")
    identities = tuple(item.get("config_hash") for item in manifest.get("runs", []))
    if identities != tuple(item.config_hash for item in plan.runs):
        raise ValueError("sweep run identities do not match existing manifest")


def _legacy_config(config: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    resolved["seed"] = int(item["seed"])
    return resolved


def run_sweep(
    config: Mapping[str, Any],
    seeds: Sequence[int],
    *,
    matrix: Mapping[str, Sequence[Any]] | None = None,
    output_dir: str | Path | None = None,
    recipe: str | None = None,
    service: ExperimentService | None = None,
) -> SweepResult:
    plan = plan_sweep(
        config, seeds, matrix=matrix, output_dir=output_dir, recipe=recipe
    )
    plan.root.mkdir(parents=True, exist_ok=True)
    manifest_path = plan.root / "sweep_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_existing_manifest(manifest, plan)
        for item in manifest.get("runs", []):
            if item.get("status") == "running":
                item["status"] = "interrupted"
    else:
        manifest = _new_manifest(plan)
        _write_manifest(manifest_path, manifest)

    experiment_service = service or ExperimentService()
    planned_by_hash = {item.config_hash: item for item in plan.runs}
    skipped = 0
    for item in manifest["runs"]:
        run_dir = Path(item["run_dir"])
        if item["status"] == "completed" and is_completed_result(run_dir):
            skipped += 1
            continue
        if int(manifest.get("schema_version", 1)) == 1:
            run_config = _legacy_config(config, item)
        else:
            planned = planned_by_hash.get(str(item.get("config_hash")))
            if planned is None:
                raise ValueError("sweep manifest contains an unknown run identity")
            run_config = resolve_planned_config(config, planned)
        item.update(status="running", error=None)
        _write_manifest(manifest_path, manifest)
        try:
            runner = resolve_runner(run_config)
            checkpoint = run_dir / "last.pt"
            resume = checkpoint if checkpoint.is_file() and runner.supports_resume else None
            experiment_service.run(
                run_config,
                run_dir,
                resume,
                recipe=recipe,
                completed_noop=True,
            )
        except Exception as exc:  # sweep isolation intentionally records each failure
            item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        else:
            item.update(status="completed", error=None)
        _write_manifest(manifest_path, manifest)

    failures = sum(item["status"] == "failed" for item in manifest["runs"])
    manifest["skipped_count"] = skipped
    manifest["status"] = "completed_with_failures" if failures else "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)
    return SweepResult(root=plan.root, manifest=manifest)


def sweep_status(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    manifest_path = candidate if candidate.is_file() else candidate / "sweep_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"sweep manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    counts = {
        status: sum(item.get("status") == status for item in runs)
        for status in ("completed", "running", "failed", "interrupted", "pending")
    }
    return {
        "root": str(manifest_path.parent),
        "sweep_id": manifest.get("sweep_id", manifest_path.parent.name),
        "status": manifest.get("status", "unknown"),
        "counts": counts,
        "completed": counts["completed"],
        "total": len(runs),
        "runs": [
            {
                "run_id": item.get("run_id"),
                "seed": item.get("seed"),
                "status": item.get("status", "unknown"),
                "overrides": dict(item.get("overrides", {}) or {}),
                "run_dir": item.get("run_dir"),
                "reason": item.get("error"),
            }
            for item in runs
        ],
        "failed_runs": [
            {
                "seed": item.get("seed"),
                "overrides": dict(item.get("overrides", {}) or {}),
                "reason": item.get("error"),
            }
            for item in runs
            if item.get("status") == "failed"
        ],
    }


__all__ = [
    "PlannedRun",
    "SweepPlan",
    "SweepResult",
    "plan_sweep",
    "resolve_planned_config",
    "run_sweep",
    "sweep_status",
]
