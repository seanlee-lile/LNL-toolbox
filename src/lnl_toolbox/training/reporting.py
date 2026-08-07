from __future__ import annotations

"""Local, dependency-light run and toolbox reports."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping


REPORT_SCHEMA_VERSION = 1


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _load_config(run_dir: Path, config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is not None:
        return dict(config)
    path = run_dir / "resolved_config.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, Mapping) else {}
    except (ImportError, OSError, TypeError, ValueError):
        return {}


def _write_resolved_config(root: Path, config: Mapping[str, Any] | None) -> None:
    target = root / "resolved_config.yaml"
    if target.exists() or config is None:
        return
    try:
        import yaml

        text = yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True)
    except ImportError:
        text = json.dumps(dict(config), indent=2, ensure_ascii=False)
    target.write_text(text, encoding="utf-8")


def _write_environment(root: Path) -> None:
    target = root / "environment.json"
    if target.exists():
        return
    environment: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }
    try:
        import torch

        environment.update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        })
    except ImportError:
        environment["torch"] = None
    target.write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")


def load_metric_events(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL events and assign compatibility sequence numbers."""

    target = Path(path)
    if not target.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"metrics line {line_number} must be an object")
        row = dict(value)
        row.setdefault("seq", len(rows))
        if "event" not in row:
            row["event"] = "epoch" if "epoch" in row else "metric"
        rows.append(row)
    return rows


def _artifact_index(run_dir: Path) -> list[dict[str, Any]]:
    excluded = {"report.json", "report.md", "artifacts.json", "run_manifest.json"}
    result: list[dict[str, Any]] = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_file() or path.name in excluded:
            continue
        result.append({
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": _file_hash(path),
        })
    metadata_path = run_dir / "artifacts.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for item in metadata.get("registered_artifacts", []):
                if isinstance(item, Mapping):
                    result.append(dict(item))
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return result


def _standard_epoch_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in events if row.get("event") == "epoch" and "epoch" in row]
    for row in rows:
        if "validation_accuracy" not in row and "selection_accuracy" in row:
            row["validation_accuracy"] = row["selection_accuracy"]
        if "validation_loss" not in row and "selection_loss" in row:
            row["validation_loss"] = row["selection_loss"]
    return rows


def write_run_report(
    run_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    runner: str | None = None,
    method: str | None = None,
    status: str | None = None,
    error: str | None = None,
    implementation_status: str = "runnable",
    modularization_status: str = "unified_run_report",
    smoke_status: str = "unknown",
    formal_run_status: str = "not_run",
    paper_fidelity_status: str = "not_audited",
) -> dict[str, Path]:
    """Create the canonical per-run manifest, artifact index and reports."""

    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(root, config)
    _write_environment(root)
    resolved = _load_config(root, config)
    events = load_metric_events(root / "metrics.jsonl")
    final_rows = [row for row in events if row.get("event") == "final"]
    final = final_rows[-1] if final_rows else {}
    final_path = root / "final_metrics.json"
    if final_path.is_file():
        try:
            value = json.loads(final_path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                final = dict(value)
        except json.JSONDecodeError:
            pass
    explicit_final = bool(final)
    if not final and events:
        final = dict(
            next(
                (row for row in reversed(events) if row.get("event") in {"epoch", "step"}),
                events[-1],
            )
        )
    if status is None:
        status = "completed" if explicit_final else "incomplete"
    if not final:
        final = {"status": status}
        if error:
            final["error"] = error
    method_value = resolved.get("method", "supervised")
    if isinstance(method_value, Mapping):
        method_value = method_value.get("name", "supervised")
    method = method or str(method_value)
    runner = runner or str((resolved.get("execution") or {}).get("runner", method))
    seed = resolved.get("seed")
    try:
        from lnl_toolbox.training.checkpoint import upgrade_checkpoint_to_v3

        for checkpoint_name in ("last.pt", "best.pt"):
            try:
                upgrade_checkpoint_to_v3(
                    root / checkpoint_name,
                    config=resolved,
                    runner=runner,
                    method=method,
                    events=len(events) - 1,
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                pass
    except ImportError:
        pass
    artifacts = _artifact_index(root)
    registered_artifacts: list[dict[str, Any]] = []
    registered_path = root / "artifacts.json"
    if registered_path.is_file():
        try:
            previous = json.loads(registered_path.read_text(encoding="utf-8"))
            registered_artifacts = [
                dict(item)
                for item in previous.get("registered_artifacts", [])
                if isinstance(item, Mapping)
            ]
        except (OSError, json.JSONDecodeError, TypeError):
            registered_artifacts = []
    epoch_rows = _standard_epoch_rows(events)
    if not final_path.is_file() and final:
        final_path.write_text(
            json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        artifacts = _artifact_index(root)
    curves_path: Path | None = None
    if epoch_rows:
        try:
            from lnl_toolbox.training.progress import write_training_curves_svg

            curves_path = root / "training_curves.svg"
            write_training_curves_svg(epoch_rows, curves_path)
            artifacts = _artifact_index(root)
        except (ImportError, KeyError, OSError, TypeError, ValueError):
            curves_path = None
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_dir": str(root),
        "runner": runner,
        "method": method,
        "seed": seed,
        "status": status,
        "error": error,
        "config_hash": _json_hash(resolved),
        "git_revision": _git_revision(),
        "event_count": len(events),
        "epoch_count": len(epoch_rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "identity": manifest,
        "implementation_status": implementation_status,
        "modularization_status": modularization_status,
        "smoke_status": smoke_status,
        "formal_run_status": formal_run_status,
        "paper_fidelity_status": paper_fidelity_status,
        "events": {"count": len(events), "epoch_count": len(epoch_rows)},
        "final_metrics": dict(final),
        "artifacts": artifacts,
        "known_limitations": [
            "This report records code execution and does not claim paper-fidelity reproduction.",
        ],
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (root / "artifacts.json").write_text(
        json.dumps(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "artifacts": artifacts,
                "registered_artifacts": registered_artifacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        f"# LNL Run Report: {method}",
        "",
        f"- Status: `{status}`",
        f"- Runner: `{runner}`",
        f"- Seed: `{seed}`",
        f"- Epoch events: `{len(epoch_rows)}`",
        f"- Paper fidelity: `not_audited`",
        "",
        "## Final metrics",
        "",
        "```json",
        json.dumps(final, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- `{item['path']}` ({item['size']} bytes)" for item in artifacts)
    if error:
        lines.extend(("", "## Error", "", f"```text\n{error}\n```"))
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "manifest": root / "run_manifest.json",
        "artifacts": root / "artifacts.json",
        "json": root / "report.json",
        "markdown": root / "report.md",
        **({"curves": curves_path} if curves_path is not None else {}),
    }


class RunSession:
    """Small lifecycle facade used by runners and the registry wrapper."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        config: Mapping[str, Any] | None = None,
        runner: str | None = None,
        method: str | None = None,
        resumed: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = dict(config or {})
        self.runner = runner
        self.method = method
        self.resumed = resumed
        self._sequence = len(load_metric_events(self.run_dir / "metrics.jsonl"))

    def emit(self, event: str, *, phase: str = "default", **values: Any) -> dict[str, Any]:
        row = {
            "event": str(event),
            "seq": self._sequence,
            "phase": str(phase),
            "method": self.method,
            "runner": self.runner,
            **values,
        }
        self._sequence += 1
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        return row

    def start_run(self) -> dict[str, Any]:
        return self.emit("resume" if self.resumed else "run_start")

    def start_phase(self, name: str, *, total_units: int | None = None) -> dict[str, Any]:
        return self.emit("phase_start", phase=name, total_units=total_units)

    def end_phase(self, name: str, *, completed_units: int | None = None) -> dict[str, Any]:
        return self.emit("phase_end", phase=name, completed_units=completed_units)

    def log_epoch(self, epoch: int, *, phase: str = "train", **metrics: Any) -> dict[str, Any]:
        return self.emit(
            "epoch",
            phase=phase,
            unit="epoch",
            completed=int(epoch),
            metrics=dict(metrics),
            artifacts=[],
            epoch=int(epoch),
            **metrics,
        )

    def log_step(self, step: int, *, phase: str = "train", **metrics: Any) -> dict[str, Any]:
        return self.emit(
            "step",
            phase=phase,
            unit="step",
            completed=int(step),
            metrics=dict(metrics),
            artifacts=[],
            global_step=int(step),
            **metrics,
        )

    def register_artifact(
        self,
        name: str,
        path: str | Path,
        *,
        artifact_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Index an existing artifact without changing its location or name."""

        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(target)
        reference = {
            "name": str(name),
            "path": str(target),
            "size": target.stat().st_size,
            "sha256": artifact_hash or _file_hash(target),
            "metadata": dict(metadata or {}),
        }
        index_path = self.run_dir / "artifacts.json"
        existing: list[dict[str, Any]] = []
        if index_path.is_file():
            try:
                value = json.loads(index_path.read_text(encoding="utf-8"))
                existing = [dict(item) for item in value.get("registered_artifacts", [])]
            except (OSError, json.JSONDecodeError, TypeError):
                existing = []
        existing = [item for item in existing if item.get("name") != reference["name"]]
        existing.append(reference)
        index_path.write_text(
            json.dumps({"schema_version": REPORT_SCHEMA_VERSION, "registered_artifacts": existing}, indent=2),
            encoding="utf-8",
        )
        return reference

    def save_checkpoint(
        self,
        payload: Mapping[str, Any],
        path: str | Path,
        *,
        phase: str = "default",
        completed_epoch: int = -1,
        global_step: int | None = None,
        component_states: Mapping[str, Any] | None = None,
        best_metric: Mapping[str, Any] | None = None,
    ) -> Path:
        """Atomically save a legacy payload with the shared v3 envelope."""

        from lnl_toolbox.training.checkpoint import atomic_save, build_v3_envelope

        saved = dict(payload)
        envelope = build_v3_envelope(
            identity={"runner": self.runner, "method": self.method, "seed": self.config.get("seed")},
            progress={
                "phase": phase,
                "completed_epoch": int(completed_epoch),
                "global_step": self._sequence - 1 if global_step is None else int(global_step),
            },
            component_states=dict(component_states or {}),
            config=self.config,
            best_metric=best_metric,
            artifact_refs={"registered": True},
            log_sequence=self._sequence - 1,
        )["checkpoint"]
        saved.setdefault("checkpoint_v3", envelope)
        target = Path(path).expanduser().resolve()
        atomic_save(saved, target)
        self.register_artifact(target.name, target)
        return target

    def recover_metrics_from_checkpoint(self, checkpoint: str | Path) -> int:
        """Discard JSONL events written after a checkpoint's committed sequence."""

        from lnl_toolbox.training.checkpoint import read_v3_checkpoint

        payload = read_v3_checkpoint(checkpoint)
        envelope = payload.get("checkpoint_v3", payload.get("checkpoint", {}))
        sequence = int(envelope.get("log_sequence", -1))
        events = [row for row in load_metric_events(self.run_dir / "metrics.jsonl") if int(row["seq"]) <= sequence]
        target = self.run_dir / "metrics.jsonl"
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in events),
            encoding="utf-8",
        )
        temporary.replace(target)
        self._sequence = len(events)
        return len(events)

    def finish_run(self, **metrics: Any) -> dict[str, Path]:
        if metrics:
            self.emit("final", **metrics)
        return write_run_report(
            self.run_dir,
            config=self.config,
            runner=self.runner,
            method=self.method,
            status="completed",
        )

    def fail_run(self, error: BaseException) -> dict[str, Path]:
        return write_run_report(
            self.run_dir,
            config=self.config,
            runner=self.runner,
            method=self.method,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )


def write_toolbox_report(runs_root: str | Path, output: str | Path) -> dict[str, Path]:
    """Aggregate all per-run reports below a root into JSON and Markdown."""

    root = Path(runs_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for path in sorted(root.rglob("report.json")) if root.exists() else []:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping):
            reports.append(dict(value))
    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "runs_root": str(root),
        "run_count": len(reports),
        "runs": reports,
    }
    json_path = destination / "toolbox_report.json"
    md_path = destination / "toolbox_report.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# LNL Toolbox Report", "", f"Runs: {len(reports)}", "", "| Method | Runner | Status | Smoke | Formal |", "|---|---|---|---|---|"]
    for report in reports:
        identity = report.get("identity", {})
        lines.append(
            f"| {identity.get('method', '-')} | {identity.get('runner', '-')} | "
            f"{identity.get('status', '-')} | {report.get('smoke_status', '-')} | "
            f"{report.get('formal_run_status', '-')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


__all__ = ["RunSession", "load_metric_events", "write_run_report", "write_toolbox_report"]
