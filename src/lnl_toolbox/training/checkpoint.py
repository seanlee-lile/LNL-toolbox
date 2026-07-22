from __future__ import annotations

"""Canonical checkpoint v2 plus safe readers for both historical layouts."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from lnl_toolbox.core import RunState


def atomic_save(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(destination)


def build_checkpoint(
    algorithm,
    run_state: RunState,
    completed_epoch: int,
    config: Mapping[str, Any],
    *,
    scheduler=None,
    best_epoch: int = -1,
    best_validation_accuracy: float = float("-inf"),
    noise: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 2,
        "model": algorithm.model.state_dict(),
        "optimizer": algorithm.optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "run_state": asdict(run_state),
        "completed_epoch": int(completed_epoch),
        "best_epoch": int(best_epoch),
        "best_validation_accuracy": float(best_validation_accuracy),
        "loss": dict(config.get("loss", {"name": "ce"})),
        "config": dict(config),
    }
    if noise is not None:
        payload["noise"] = dict(noise)
    return payload


def save_checkpoint(
    path: str | Path,
    algorithm,
    run_state: RunState,
    completed_epoch: int,
    config: Mapping[str, Any],
    *,
    scheduler=None,
    best_epoch: int = -1,
    best_validation_accuracy: float = float("-inf"),
    noise: Mapping[str, Any] | None = None,
) -> None:
    atomic_save(
        build_checkpoint(
            algorithm,
            run_state,
            completed_epoch,
            config,
            scheduler=scheduler,
            best_epoch=best_epoch,
            best_validation_accuracy=best_validation_accuracy,
            noise=noise,
        ),
        path,
    )


def read_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a mapping")
    return payload


def _restore_model_optimizer(payload: Mapping[str, Any], algorithm) -> str:
    if "model" in payload and "optimizer" in payload:
        algorithm.load_state_dict({
            "model": payload["model"],
            "optimizer": payload["optimizer"],
        })
        return "top-level"
    legacy = payload.get("algorithm")
    if isinstance(legacy, Mapping) and "model" in legacy and "optimizer" in legacy:
        algorithm.load_state_dict(dict(legacy))
        return "nested-algorithm"
    raise ValueError("Checkpoint is missing model or optimizer state")


def load_checkpoint(
    path: str | Path,
    algorithm,
    device: torch.device,
    *,
    scheduler=None,
) -> tuple[RunState, int, dict[str, Any]]:
    payload = read_checkpoint(path, device)
    layout = _restore_model_optimizer(payload, algorithm)

    has_scheduler = "scheduler" in payload
    saved_scheduler = payload.get("scheduler")
    if scheduler is None and saved_scheduler is not None:
        raise ValueError("Checkpoint contains scheduler state but the current run disables it")
    if scheduler is not None:
        if not has_scheduler or saved_scheduler is None:
            raise ValueError("Checkpoint is missing scheduler state required by the current run")
        scheduler.load_state_dict(saved_scheduler)

    state_value = payload.get("run_state")
    if not isinstance(state_value, Mapping):
        raise ValueError("Checkpoint is missing run_state")
    state = RunState(**dict(state_value))
    completed_epoch = int(payload.get("completed_epoch", -1))

    warnings: list[str] = []
    if "best_epoch" not in payload or "best_validation_accuracy" not in payload:
        payload["best_epoch"] = completed_epoch
        payload["best_validation_accuracy"] = float("-inf")
        warnings.append(
            "Legacy checkpoint has no best metric; the next completed epoch will establish it."
        )
    if int(payload.get("format_version", 0)) < 2:
        warnings.append(f"Loaded legacy {layout} checkpoint layout.")
    if warnings:
        payload["_compatibility_warnings"] = warnings
    return state, completed_epoch, payload
