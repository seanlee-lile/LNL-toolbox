"""Read-only, best-effort training status for the local Web console."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


@dataclass(frozen=True)
class TrainingContext:
    """Location and bounded identity inferred from one CLI argv."""

    run_dir: Path | None
    config_path: Path | None
    command_kind: str
    total_epochs: int | None


@dataclass(frozen=True)
class TrainingSnapshot:
    """A display-only view; it never controls the child training process."""

    state: str
    stage: str | None
    completed_epoch: int | None
    total_epochs: int | None
    source: str
    metrics: tuple[dict[str, Any], ...] = ()
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["metrics"] = list(self.metrics)
        return value


def _option_value(command: Sequence[str], option: str) -> str | None:
    try:
        index = list(command).index(option)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _resolve_path(value: str | None, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _read_total_epochs(config_path: Path | None) -> int | None:
    if config_path is None or not config_path.is_file():
        return None
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(raw, dict):
        return None
    trainer = raw.get("trainer")
    if isinstance(trainer, dict) and isinstance(trainer.get("epochs"), int):
        return int(trainer["epochs"])
    # Multi-stage runners reuse ``epoch`` for each stage.  A single combined
    # total would make e.g. DivideMix main epoch 1 appear as 1/310, so it is
    # deliberately omitted until a runner publishes a stage-aware contract.
    return None


def infer_training_context(command: Sequence[str], root: str | Path) -> TrainingContext | None:
    """Infer only safe, explicit run locations from ``lnl run``/``resume``."""

    values = list(command)
    root_path = Path(root)
    if "resume" in values:
        index = values.index("resume")
        run_dir = _resolve_path(values[index + 1] if index + 1 < len(values) else None, root_path)
        return TrainingContext(run_dir=run_dir, config_path=None, command_kind="resume", total_epochs=None)
    if "run" not in values:
        return None
    config_path = _resolve_path(_option_value(values, "--config"), root_path)
    run_index = values.index("run")
    if config_path is None and run_index + 1 < len(values):
        candidate = values[run_index + 1]
        if not candidate.startswith("-"):
            config_path = _resolve_path(candidate, root_path)
    output_dir = _resolve_path(_option_value(values, "--output-dir"), root_path)
    total_epochs = _read_total_epochs(config_path)
    override = _option_value(values, "--epochs")
    if override is not None:
        try:
            total_epochs = int(override)
        except ValueError:
            pass
    return TrainingContext(
        run_dir=output_dir,
        config_path=config_path,
        command_kind="run",
        total_epochs=total_epochs,
    )


def _metric_rows(path: Path | None) -> tuple[list[dict[str, Any]], str | None]:
    if path is None:
        return [], None
    metrics_path = path / "metrics.jsonl"
    if not metrics_path.is_file():
        return [], None
    rows: list[dict[str, Any]] = []
    try:
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    except OSError as exc:
        return [], str(exc)
    return rows[-100:], None


def _structured_stdout(lines: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get("event"), str):
            rows.append(row)
    return rows[-100:]


def _stage(row: dict[str, Any]) -> str | None:
    phase = row.get("phase")
    if isinstance(phase, str) and phase:
        return phase
    event = row.get("event")
    if event == "warmup":
        return "warmup"
    if event == "epoch":
        return "main"
    if event == "final":
        return "completed"
    return str(event) if isinstance(event, str) else None


def training_snapshot(
    context: TrainingContext | None,
    *,
    lines: Iterable[str],
    running: bool,
    returncode: int | None,
    cancel_requested: bool,
) -> TrainingSnapshot | None:
    if context is None:
        return None
    rows, warning = _metric_rows(context.run_dir)
    source = "metrics" if rows else "stdout"
    if not rows:
        rows = _structured_stdout(lines)
    latest = rows[-1] if rows else {}
    stage = _stage(latest)
    epoch = latest.get("epoch")
    completed_epoch = int(epoch) if isinstance(epoch, int) else None
    if running:
        state = "running"
    elif cancel_requested:
        state = "cancelled"
    elif returncode == 0:
        state = "completed"
    elif returncode is not None:
        state = "failed"
    else:
        state = "starting"
    return TrainingSnapshot(
        state=state,
        stage=stage,
        completed_epoch=completed_epoch,
        total_epochs=context.total_epochs,
        source=source if rows else "process",
        metrics=tuple(rows),
        warning=warning,
    )
