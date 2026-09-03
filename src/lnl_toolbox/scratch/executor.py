"""Tiny top-to-bottom interpreter for Scratch recipes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import blocks as _builtin_blocks  # noqa: F401
from .context import ScratchContext
from .registry import BlockDefinition, get_block
from .validation import validate_recipe


class ScratchExecutionError(RuntimeError):
    def __init__(
        self,
        path: tuple[int, ...],
        definition: BlockDefinition,
        params: Mapping[str, Any],
        context: Mapping[str, Any],
        cause: Exception,
    ) -> None:
        self.path = path
        self.block_id = definition.id
        self.params = dict(params)
        self.cause = cause
        label = " → ".join(f"step {index + 1}" for index in path)
        super().__init__(
            f"{label} — {definition.name} ({definition.id}) failed. "
            f"Available context: {sorted(context)}. Original error: {cause}"
        )


def _publish_progress(
    context: ScratchContext,
    *,
    block_id: str | None = None,
    path: tuple[int, ...] = (),
    state: str = "running",
    error: str | None = None,
) -> None:
    """Write a small, JSON-safe execution snapshot for the WebUI.

    The progress file is deliberately separate from ``ScratchContext``'s
    runtime objects.  It is an observation channel only; it never controls
    execution and never serializes tensors, models, loaders, or datasets.
    """

    destination_value = context.get("_progress_path")
    if not destination_value:
        return
    destination = Path(str(destination_value))
    previous: dict[str, Any] = {}
    # The outer recipe handler publishes a terminal state after a block has
    # already published its precise failure location.  Retain that location
    # when the terminal call does not provide a new block/path.
    if state == "failed" and block_id is None and not path:
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                previous = existing
        except (OSError, json.JSONDecodeError):
            pass
    payload: dict[str, Any] = {
        "state": state,
        "block": block_id if block_id is not None else previous.get("block"),
        "path": list(path) if path else previous.get("path", []),
        "epoch": context.get("epoch", previous.get("epoch")),
        "batch_idx": context.get("batch_idx", previous.get("batch_idx")),
        "global_step": context.get("global_step", previous.get("global_step")),
        "total_epochs": context.get("_progress_total_epochs", previous.get("total_epochs")),
        "total_batches": context.get("_progress_total_batches", previous.get("total_batches")),
    }
    total_epochs = payload["total_epochs"]
    epoch = payload["epoch"]
    total_batches = payload["total_batches"]
    batch_idx = payload["batch_idx"]
    try:
        if isinstance(total_epochs, int) and total_epochs > 0 and isinstance(epoch, int):
            epoch_offset = max(0, epoch - int(context.get("_progress_start_epoch", 0)))
            within_epoch = 0.0
            if isinstance(total_batches, int) and total_batches > 0 and isinstance(batch_idx, int):
                within_epoch = min(max((batch_idx + 1) / total_batches, 0.0), 1.0)
            payload["fraction"] = min(max((epoch_offset + within_epoch) / total_epochs, 0.0), 1.0)
            payload["completed_epoch"] = min(epoch_offset + (1 if within_epoch >= 1.0 else 0), total_epochs)
    except (TypeError, ValueError, ZeroDivisionError):
        payload["fraction"] = None
    if error:
        payload["error"] = str(error)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(destination)
    except OSError:
        # Progress is best-effort and must never turn a valid Scratch run into
        # a failure merely because the UI status file is unavailable.
        return


def _resolved_params(definition: BlockDefinition, step: Mapping[str, Any]) -> dict[str, Any]:
    supplied = dict(step.get("params", {}))
    for name, schema in definition.params.items():
        if name not in supplied and "default" in schema:
            supplied[name] = schema["default"]
    return supplied


def _runtime_requirements(
    definition: BlockDefinition,
    context: ScratchContext,
    params: Mapping[str, Any],
) -> None:
    missing = []
    for name in definition.requires:
        candidate = params.get(name, name)
        slot = candidate if isinstance(candidate, str) else name
        if not isinstance(slot, str) or slot not in context:
            missing.append(str(slot))
    if missing:
        raise KeyError(f"missing context values: {', '.join(missing)}")


def execute_steps(
    steps: Sequence[Mapping[str, Any]],
    context: ScratchContext,
    *,
    _path: tuple[int, ...] = (),
) -> ScratchContext:
    for index, step in enumerate(steps):
        path = (*_path, index)
        definition = get_block(str(step["block"]))
        params = _resolved_params(definition, step)
        children = step.get("steps", [])
        _publish_progress(context, block_id=definition.id, path=path)
        try:
            _runtime_requirements(definition, context, params)
            if definition.kind == "action":
                definition.execute(context, **params)
            else:
                definition.execute(
                    context,
                    params=params,
                    children=children,
                    execute=lambda nested, nested_context=context: execute_steps(
                        nested, nested_context, _path=path
                    ),
                )
        except ScratchExecutionError:
            raise
        except Exception as exc:
            _publish_progress(context, block_id=definition.id, path=path, state="failed", error=str(exc))
            raise ScratchExecutionError(
                path, definition, params, context, exc
            ) from exc
    return context


def execute_recipe(
    recipe: Mapping[str, Any],
    context: ScratchContext | None = None,
    *,
    runtime_limits: Mapping[str, Any] | None = None,
) -> ScratchContext:
    initial: dict[str, Any] = {}
    if isinstance(recipe, Mapping) and isinstance(recipe.get("settings", {}), Mapping):
        initial.update(recipe.get("settings", {}))
    if context is not None:
        initial.update(context)
    if runtime_limits is not None:
        if not isinstance(runtime_limits, Mapping):
            raise TypeError("runtime_limits must be a mapping")
        allowed = {"max_epochs", "max_batches", "skip_final_test", "fixture"}
        unknown = set(runtime_limits) - allowed
        if unknown:
            raise ValueError(f"unknown runtime limits: {sorted(unknown)}")
        limits = dict(runtime_limits)
        for name in ("max_epochs", "max_batches"):
            if limits.get(name) is not None and (
                not isinstance(limits[name], int) or isinstance(limits[name], bool) or int(limits[name]) < 1
            ):
                raise ValueError(f"runtime limit `{name}` must be a positive integer or null")
        if "skip_final_test" in limits and not isinstance(limits["skip_final_test"], bool):
            raise ValueError("runtime limit `skip_final_test` must be boolean")
        if "fixture" in limits and not isinstance(limits["fixture"], bool):
            raise ValueError("runtime limit `fixture` must be boolean")
        # Fixture/catalog runs are deliberately bounded unless the caller
        # supplies explicit limits.  This keeps the formal recipe settings
        # intact while preventing a tiny synthetic source from entering a
        # paper's full multi-epoch training horizon during validation.
        if limits.get("fixture"):
            limits.setdefault("max_epochs", 1)
            limits.setdefault("max_batches", 1)
        initial["_runtime_limits"] = limits
    validated = validate_recipe(recipe, initial_slots=set(initial))
    result = ScratchContext(validated.get("settings", {}))
    result.update(initial)
    try:
        executed = execute_steps(validated["steps"], result)
    except Exception as exc:
        _publish_progress(result, state="failed", error=str(exc))
        raise
    _publish_progress(result, state="completed", path=())
    return executed
