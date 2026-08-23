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
) -> ScratchContext:
    result = ScratchContext(recipe.get("settings", {}))
    if context is not None:
        result.update(context)
    validated = validate_recipe(recipe, initial_slots=set(result))
    return execute_steps(validated["steps"], result)
