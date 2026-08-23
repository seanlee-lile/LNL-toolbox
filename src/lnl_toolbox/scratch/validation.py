"""Necessary, user-facing validation for sequential Scratch recipes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from . import blocks as _builtin_blocks  # noqa: F401
from .registry import BlockDefinition, get_block


class ScratchValidationError(ValueError):
    pass


_TYPE_CHECKS = {
    "bool": lambda value: isinstance(value, bool),
    "float": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "int": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "slot": lambda value: isinstance(value, str) and bool(value.strip()),
    "enum": lambda value: isinstance(value, str),
    "dataset": lambda value: isinstance(value, str),
    "path": lambda value: isinstance(value, str),
    "str": lambda value: isinstance(value, str),
    "value": lambda value: True,
}


def _path_label(path: tuple[int, ...]) -> str:
    return " → ".join(f"step {index + 1}" for index in path)


def _resolve_params(
    definition: BlockDefinition,
    supplied: Mapping[str, Any],
    path: tuple[int, ...],
) -> dict[str, Any]:
    unknown = set(supplied) - set(definition.params)
    if unknown:
        raise ScratchValidationError(
            f"{_path_label(path)} — {definition.name}: unknown parameters {sorted(unknown)}"
        )
    resolved: dict[str, Any] = {}
    for name, schema in definition.params.items():
        if name in supplied:
            value = supplied[name]
        elif "default" in schema:
            value = deepcopy(schema["default"])
        elif bool(schema.get("required", False)):
            raise ScratchValidationError(
                f"{_path_label(path)} — {definition.name}: missing parameter `{name}`"
            )
        else:
            continue
        type_name = str(schema.get("type", "value"))
        checker = _TYPE_CHECKS.get(type_name)
        if checker is None:
            raise ScratchValidationError(
                f"{definition.id}: unsupported parameter type `{type_name}`"
            )
        if not checker(value):
            raise ScratchValidationError(
                f"{_path_label(path)} — {definition.name}: parameter `{name}` must be {type_name}"
            )
        options = schema.get("options")
        if type_name == "enum" and isinstance(options, (list, tuple, set)) and value not in options:
            raise ScratchValidationError(
                f"{_path_label(path)} — {definition.name}: parameter `{name}` must be one of {list(options)}"
            )
        if type_name in {"int", "float"}:
            if "min" in schema and value < schema["min"]:
                raise ScratchValidationError(
                    f"{_path_label(path)} — {definition.name}: `{name}` is below its minimum"
                )
            if "max" in schema and value > schema["max"]:
                raise ScratchValidationError(
                    f"{_path_label(path)} — {definition.name}: `{name}` exceeds its maximum"
                )
        resolved[name] = value
    return resolved


def _required_slot(name: str, params: Mapping[str, Any]) -> str:
    """Resolve a requirement parameter, or use a fixed context slot name."""
    value = params.get(name, name)
    if name not in params:
        return name
    if not isinstance(value, str) or not value.strip():
        raise ScratchValidationError(f"context slot `{name}` must be a non-empty string")
    return value


def _provided_slot(name: str, params: Mapping[str, Any]) -> str:
    """Resolve dynamic save_as-style outputs while allowing fixed outputs."""
    value = params.get(name, name)
    return value if isinstance(value, str) and value.strip() else name


def _validate_steps(
    steps: Sequence[Any],
    available: set[str],
    path: tuple[int, ...] = (),
    context: str = "top",
) -> set[str]:
    if not isinstance(steps, list):
        raise ScratchValidationError(f"{_path_label(path) or 'recipe'}: steps must be a list")
    current = set(available)
    for index, raw_step in enumerate(steps):
        step_path = (*path, index)
        if not isinstance(raw_step, Mapping):
            raise ScratchValidationError(f"{_path_label(step_path)} must be a mapping")
        unknown_fields = set(raw_step) - {"block", "params", "steps"}
        if unknown_fields:
            raise ScratchValidationError(
                f"{_path_label(step_path)} contains forbidden fields {sorted(unknown_fields)}"
            )
        block_id = raw_step.get("block")
        if not isinstance(block_id, str):
            raise ScratchValidationError(f"{_path_label(step_path)} requires a block id")
        try:
            definition = get_block(block_id)
        except KeyError as exc:
            raise ScratchValidationError(f"{_path_label(step_path)}: {exc.args[0]}") from exc
        # Legacy paper recipes keep their historical execution shape.  V2 UI
        # placement is enforced for beginner-visible blocks; hidden legacy
        # blocks remain executable for compatibility.
        if definition.beginner_visible and "any" not in definition.placement and context not in definition.placement:
            raise ScratchValidationError(
                f"{_path_label(step_path)} — {definition.name}: block `{definition.id}` "
                f"cannot be placed in {context} context; move it to {', '.join(definition.placement)}"
            )
        supplied = raw_step.get("params", {})
        if not isinstance(supplied, Mapping):
            raise ScratchValidationError(
                f"{_path_label(step_path)} — {definition.name}: params must be a mapping"
            )
        params = _resolve_params(definition, supplied, step_path)
        missing = [
            _required_slot(name, params)
            for name in definition.requires
            if _required_slot(name, params) not in current
        ]
        if missing:
            raise ScratchValidationError(
                f"{_path_label(step_path)} — {definition.name} cannot run; "
                f"missing context values: {', '.join(missing)}; "
                "add an upstream block that provides these keys first"
            )
        children = raw_step.get("steps")
        if definition.kind == "action" and children is not None:
            raise ScratchValidationError(
                f"{_path_label(step_path)} — {definition.name} cannot contain child steps"
            )
        if definition.kind != "action":
            if not isinstance(children, list):
                raise ScratchValidationError(
                    f"{_path_label(step_path)} — {definition.name} requires child steps"
                )
            child_available = current | {
                _provided_slot(name, params) for name in definition.provides
            }
            child_context = "epoch" if definition.id == "epoch_loop" else "batch" if definition.id == "batch_loop" else context
            child_result = _validate_steps(children, child_available, step_path, child_context)
            # Epoch-level evaluation/selection outputs remain available after
            # the loop (for example the best validation checkpoint). Batch
            # intermediates stay scoped to the batch loop.
            if definition.id == "epoch_loop":
                current.update(child_result)
        current.update(_provided_slot(name, params) for name in definition.provides)
    return current


def validate_recipe(
    recipe: Mapping[str, Any],
    *,
    initial_slots: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(recipe, Mapping):
        raise ScratchValidationError("Scratch recipe must be a mapping")
    unknown = set(recipe) - {"schema_version", "name", "description", "settings", "steps"}
    if unknown:
        raise ScratchValidationError(f"Scratch recipe contains unknown fields {sorted(unknown)}")
    if recipe.get("schema_version") != 1:
        raise ScratchValidationError("Scratch recipe schema_version must be 1")
    if not isinstance(recipe.get("name"), str) or not recipe["name"].strip():
        raise ScratchValidationError("Scratch recipe name must not be empty")
    settings = recipe.get("settings", {})
    if not isinstance(settings, Mapping):
        raise ScratchValidationError("Scratch recipe settings must be a mapping")
    steps = recipe.get("steps")
    _validate_steps(steps, set(settings) | set(initial_slots or ()))
    return deepcopy(dict(recipe))
