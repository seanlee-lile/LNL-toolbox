from __future__ import annotations

"""Canonical checkpoint v2 plus safe readers for both historical layouts."""

from dataclasses import asdict
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from lnl_toolbox.core import RunState


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("rng state must be a mapping")
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


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
    selection_split: str = "validation",
    best_selection_accuracy: float = float("-inf"),
    noise: Mapping[str, Any] | None = None,
    pipeline: Mapping[str, Any] | None = None,
    early_stopping: Mapping[str, Any] | None = None,
    component_states: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    algorithm_state = algorithm.state_dict()
    payload: dict[str, Any] = {
        "format_version": 2,
        "model": algorithm_state["model"],
        "optimizer": algorithm_state["optimizer"],
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "run_state": asdict(run_state),
        "completed_epoch": int(completed_epoch),
        "best_epoch": int(best_epoch),
        "best_validation_accuracy": float(best_validation_accuracy),
        "selection_split": str(selection_split),
        "best_selection_accuracy": float(best_selection_accuracy),
        "loss": dict(config.get("loss", {"name": "ce"})),
        "config": dict(config),
        "rng_state": capture_rng_state(),
        "algorithm_private_state": {
            key: value
            for key, value in algorithm_state.items()
            if key not in {"model", "optimizer"}
        },
    }
    if "parameter_update_policy" in algorithm_state:
        payload["parameter_update_policy"] = algorithm_state[
            "parameter_update_policy"
        ]
    if noise is not None:
        payload["noise"] = dict(noise)
    if pipeline is not None:
        payload["pipeline"] = dict(pipeline)
    if early_stopping is not None:
        payload["early_stopping"] = dict(early_stopping)
    if component_states is not None:
        payload["component_states"] = dict(component_states)
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
    selection_split: str = "validation",
    best_selection_accuracy: float = float("-inf"),
    noise: Mapping[str, Any] | None = None,
    pipeline: Mapping[str, Any] | None = None,
    early_stopping: Mapping[str, Any] | None = None,
    component_states: Mapping[str, Any] | None = None,
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
            selection_split=selection_split,
            best_selection_accuracy=best_selection_accuracy,
            noise=noise,
            pipeline=pipeline,
            early_stopping=early_stopping,
            component_states=component_states,
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
        state = {
            "model": payload["model"],
            "optimizer": payload["optimizer"],
        }
        if "parameter_update_policy" in payload:
            state["parameter_update_policy"] = payload["parameter_update_policy"]
        state.update(dict(payload.get("algorithm_private_state", {})))
        algorithm.load_state_dict(state)
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
    if "rng_state" in payload:
        restore_rng_state(payload["rng_state"])
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
    payload.setdefault("selection_split", "validation")
    payload.setdefault("best_selection_accuracy", payload["best_validation_accuracy"])
    if int(payload.get("format_version", 0)) < 2:
        warnings.append(f"Loaded legacy {layout} checkpoint layout.")
    if warnings:
        payload["_compatibility_warnings"] = warnings
    return state, completed_epoch, payload
