"""Read-only, best-effort training status for the local Web console."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
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
    total_steps: int | None = None
    validation_targets: str | None = None


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
    completed_step: int | None = None
    total_steps: int | None = None
    epoch_seconds: float | None = None
    estimated_total_seconds: float | None = None
    accuracy: float | None = None
    accuracy_split: str | None = None

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


def _read_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.is_file():
        return {}
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _training_budget(raw: dict[str, Any]) -> tuple[int | None, int | None]:
    trainer = raw.get("trainer")
    if isinstance(trainer, dict) and isinstance(trainer.get("max_steps"), int) and trainer["max_steps"] > 0:
        return None, trainer["max_steps"]
    if isinstance(trainer, dict) and isinstance(trainer.get("epochs"), int):
        return int(trainer["epochs"]), None
    # Multi-stage runners reuse ``epoch`` for each stage.  A single combined
    # total would make e.g. DivideMix main epoch 1 appear as 1/310, so it is
    # deliberately omitted until a runner publishes a stage-aware contract.
    return None, None


def infer_training_context(command: Sequence[str], root: str | Path) -> TrainingContext | None:
    """Infer only safe, explicit run locations from ``lnl run``/``resume``."""

    values = list(command)
    root_path = Path(root)
    if "resume" in values:
        index = values.index("resume")
        run_dir = _resolve_path(values[index + 1] if index + 1 < len(values) else None, root_path)
        if run_dir is not None and run_dir.is_file():
            run_dir = run_dir.parent
        config_path = run_dir / "resolved_config.yaml" if run_dir is not None else None
        config = _read_config(config_path)
        return TrainingContext(run_dir, config_path, "resume", *_training_budget(config), _validation_targets(config))
    if "run" not in values:
        return None
    config_path = _resolve_path(_option_value(values, "--config"), root_path)
    run_index = values.index("run")
    if config_path is None and run_index + 1 < len(values):
        candidate = values[run_index + 1]
        if not candidate.startswith("-"):
            config_path = _resolve_path(candidate, root_path)
    output_dir = _resolve_path(_option_value(values, "--output-dir"), root_path)
    config = _read_config(config_path)
    total_epochs, total_steps = _training_budget(config)
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
        total_steps=total_steps,
        validation_targets=_validation_targets(config),
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
    event = row.get("event")
    if event == "final":
        return "completed"
    for key in ("stage", "phase"):
        phase = row.get(key)
        if isinstance(phase, str) and phase:
            return phase
    if event == "stage1_epoch":
        return "stage1"
    if event == "diffusion_epoch":
        return "diffusion"
    if event == "warmup":
        return "warmup"
    if event in {"epoch", "main_epoch"}:
        return "main"
    return str(event) if isinstance(event, str) else None


def _validation_targets(config: dict[str, Any]) -> str | None:
    noise = config.get("noise")
    value = noise.get("validation_targets") if isinstance(noise, dict) else None
    return value if value in ("clean", "noisy") else None


def _positive_int(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def _epoch(row: dict[str, Any]) -> int | None:
    epoch = row.get("stage_epoch", row.get("epoch"))
    if type(epoch) is not int:
        return None
    if "stage_epoch" not in row and (
        row.get("event") in {"stage1_epoch", "main_epoch", "diffusion_epoch"}
        or row.get("phase") == "final_training"
    ):
        epoch += 1
    return _positive_int(epoch)


def _number(value: Any) -> float | None:
    return float(value) if type(value) in (float, int) and math.isfinite(value) else None


def _accuracy(row: dict[str, Any], validation_targets: str | None) -> tuple[float | None, str | None]:
    targets = row.get("validation_targets", validation_targets)
    validation_split = f"{targets}_validation" if targets in ("clean", "noisy") else "validation"
    selection_split = row.get("selection_split")
    if selection_split == "validation":
        selection_split = validation_split
    # Some legacy standardized rows copy the selection score to validation_accuracy.
    validation = row.get("validation_accuracy")
    selected = row.get("selection_accuracy")
    if selection_split == "test" and selected == validation:
        validation = None
    for value, split in (
        (row.get("revised_noisy_validation_accuracy"), "noisy_validation"),
        (row.get("noisy_validation_accuracy"), "noisy_validation"),
        (row.get("clean_validation_accuracy"), "clean_validation"),
        (row.get("mean_peer_accuracy"), validation_split),
        (row.get("validation_accuracy_ensemble"), validation_split),
        (selected, selection_split if selection_split in ("clean_validation", "noisy_validation") else None),
        (validation, validation_split),
        (selected, selection_split),
        (row.get("test_accuracy"), "test"),
    ):
        accuracy = _number(value)
        if accuracy is not None and 0 <= accuracy <= 1 and split in (
            "validation", "clean_validation", "noisy_validation", "test"
        ):
            return accuracy, split
    return None, None


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
    # A final summary has test metrics of its own. Keep the last epoch for the
    # progress strip so its count, time and accuracy still describe one row.
    progress = latest
    if latest.get("event") == "final":
        progress = next((row for row in reversed(rows[:-1]) if _epoch(row) is not None), {})
    stage = _stage(progress)
    completed_epoch = _epoch(progress)
    total_epochs = _positive_int(progress.get("total_epochs"))
    total_steps = _positive_int(progress.get("total_steps"))
    if stage in (None, "main", "cwd") and "total_steps" not in progress:
        total_steps = total_steps or context.total_steps
    if total_steps is None and stage in (None, "main", "cwd") and "total_epochs" not in progress:
        total_epochs = context.total_epochs
    completed_step = _positive_int(progress.get("global_step"))
    epoch_seconds = _number(progress.get("epoch_seconds"))
    estimated_total_seconds = None
    if epoch_seconds is not None and epoch_seconds > 0:
        epoch_steps = _positive_int(progress.get("epoch_steps"))
        if total_steps and epoch_steps and (completed_step or 0) < total_steps:
            estimated_total_seconds = epoch_seconds * total_steps / epoch_steps
        elif total_steps is None and total_epochs:
            estimated_total_seconds = epoch_seconds * total_epochs
    accuracy, accuracy_split = _accuracy(progress, context.validation_targets)
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
        total_epochs=total_epochs,
        source=source if rows else "process",
        metrics=tuple(rows),
        warning=warning,
        completed_step=completed_step,
        total_steps=total_steps,
        epoch_seconds=epoch_seconds,
        estimated_total_seconds=estimated_total_seconds,
        accuracy=accuracy,
        accuracy_split=accuracy_split,
    )
