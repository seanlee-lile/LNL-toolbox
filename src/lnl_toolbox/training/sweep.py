from __future__ import annotations

"""Sequential, resumable multi-seed experiment sweeps."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from lnl_toolbox.training.results import config_hash, is_completed_result
from lnl_toolbox.training.runners import resolve_runner
from lnl_toolbox.training.service import ExperimentService


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


def _write_manifest(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _default_root(config: Mapping[str, Any], recipe: str | None) -> Path:
    identity = config_hash(config)[:10]
    name = (recipe or "custom").strip().lower().replace("_", "-")
    return Path("artifacts/sweeps") / f"{name}-{identity}"


def run_sweep(
    config: Mapping[str, Any],
    seeds: Sequence[int],
    *,
    output_dir: str | Path | None = None,
    recipe: str | None = None,
    service: ExperimentService | None = None,
) -> SweepResult:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized or any(seed < 0 for seed in normalized):
        raise ValueError("sweep seeds must be a non-empty list of non-negative integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("sweep seeds must be unique")
    base = deepcopy(dict(config))
    root = Path(output_dir or _default_root(base, recipe)).expanduser().resolve()
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "sweep_manifest.json"
    identity = config_hash(base)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != identity:
            raise ValueError("sweep configuration does not match existing manifest")
        if tuple(item["seed"] for item in manifest.get("runs", [])) != normalized:
            raise ValueError("sweep seeds do not match existing manifest")
    else:
        manifest = {
            "schema_version": 1,
            "sweep_id": root.name,
            "recipe": recipe or "custom",
            "config_hash": identity,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "runs": [
                {
                    "seed": seed,
                    "status": "pending",
                    "run_dir": str((runs_root / f"seed-{seed}").resolve()),
                    "error": None,
                }
                for seed in normalized
            ],
        }
        _write_manifest(manifest_path, manifest)

    experiment_service = service or ExperimentService()
    skipped = 0
    for item in manifest["runs"]:
        run_dir = Path(item["run_dir"])
        if item["status"] == "completed" and is_completed_result(run_dir):
            skipped += 1
            continue
        run_config = deepcopy(base)
        run_config["seed"] = int(item["seed"])
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
    return SweepResult(root=root, manifest=manifest)


__all__ = ["SweepResult", "run_sweep"]
