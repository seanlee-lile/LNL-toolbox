"""Tiny top-to-bottom interpreter for Scratch recipes."""

from __future__ import annotations

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
        initial["_runtime_limits"] = limits
    validated = validate_recipe(recipe, initial_slots=set(initial))
    result = ScratchContext(validated.get("settings", {}))
    result.update(initial)
    return execute_steps(validated["steps"], result)
