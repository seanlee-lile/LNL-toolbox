"""Validation for FormulaSpec before registration or execution."""

from __future__ import annotations

import re
from typing import Any, Mapping

from ..registry import get_block
from .schema import FormulaSpec


class FormulaValidationError(ValueError):
    pass


_FORMULA_ID = re.compile(r"^(?:builtin|user)/[a-z0-9][a-z0-9_-]*$")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9 _.-]{0,127}$")


def _source_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormulaValidationError("formula bindings must reference a named slot")
    return value.strip()


def _step_output_name(step_id: str, source: str) -> str:
    return source.split(".", 1)[0] if isinstance(source, str) else source


def _nested_formula_id(block_id: str) -> str | None:
    if block_id.startswith("formula/"):
        return block_id.removeprefix("formula/")
    if block_id.startswith("formula__"):
        return block_id.removeprefix("formula__").replace("__", "/", 1)
    return None


def validate_formula(value: FormulaSpec | Mapping[str, Any], *, stack: tuple[str, ...] = ()) -> FormulaSpec:
    spec = value if isinstance(value, FormulaSpec) else FormulaSpec.from_dict(value)
    if not _FORMULA_ID.fullmatch(spec.id):
        raise FormulaValidationError("formula id must look like `user/my_formula` or `builtin/name`")
    if not _NAME.fullmatch(spec.name.strip()):
        raise FormulaValidationError("formula name must be a non-empty readable name")
    if spec.id in stack:
        chain = " -> ".join((*stack, spec.id))
        raise FormulaValidationError(f"formula cycle detected: {chain}")
    if not spec.inputs:
        raise FormulaValidationError("formula must declare at least one input")
    names = set(spec.inputs) | set(spec.parameters)
    if names & set(spec.outputs):
        raise FormulaValidationError("formula output names must not shadow inputs or parameters")
    for name, input_spec in spec.inputs.items():
        if not name or not name.replace("_", "").isalnum():
            raise FormulaValidationError(f"invalid formula input name: {name}")
        if input_spec.type not in {"tensor", "scalar", "mask", "labels", "features", "probability", "transition", "any"}:
            raise FormulaValidationError(f"unsupported formula input type: {input_spec.type}")
    for name, parameter in spec.parameters.items():
        if not name or not name.replace("_", "").isalnum():
            raise FormulaValidationError(f"invalid formula parameter name: {name}")
        if parameter.type not in {"float", "int", "bool", "enum", "str", "value"}:
            raise FormulaValidationError(f"unsupported formula parameter type: {parameter.type}")
        if parameter.type == "enum" and parameter.options and parameter.default not in parameter.options:
            raise FormulaValidationError(f"parameter `{name}` default is not in options")
        if parameter.minimum is not None and parameter.maximum is not None and parameter.minimum > parameter.maximum:
            raise FormulaValidationError(f"parameter `{name}` has an invalid range")
        if parameter.default is not None:
            if parameter.minimum is not None and parameter.default < parameter.minimum:
                raise FormulaValidationError(f"parameter `{name}` default is below its minimum")
            if parameter.maximum is not None and parameter.default > parameter.maximum:
                raise FormulaValidationError(f"parameter `{name}` default exceeds its maximum")
    if not spec.steps:
        raise FormulaValidationError("formula must contain at least one operation step")
    seen: set[str] = set()
    for step in spec.steps:
        if not step.id or not step.id.replace("_", "").isalnum():
            raise FormulaValidationError(f"invalid or missing step id: {step.id!r}")
        if step.id in seen or step.id in names:
            raise FormulaValidationError(f"duplicate formula step id: {step.id}")
        seen.add(step.id)
        try:
            definition = get_block(step.block)
        except KeyError as exc:
            raise FormulaValidationError(f"step `{step.id}` uses unknown Block `{step.block}`") from exc
        if not definition.formula_safe:
            raise FormulaValidationError(f"Block `{step.block}` is not allowed inside a Formula")
        nested_id = _nested_formula_id(step.block)
        if nested_id:
            # Resolving the block above proves registration; importing here
            # avoids a hard import cycle with the Formula Registry.
            from .registry import get_formula

            nested = get_formula(nested_id)
            validate_formula(nested, stack=(*stack, spec.id))
        for param_name, binding in step.bindings.items():
            if param_name not in definition.params:
                raise FormulaValidationError(f"step `{step.id}` has unknown binding `{param_name}`")
            if definition.params[param_name].get("type") == "slot":
                source = _source_name(binding)
                if source not in names and source not in seen:
                    raise FormulaValidationError(f"step `{step.id}` binding `{param_name}` references unavailable `{source}`")
        for param_name, value_param in step.parameters.items():
            if param_name not in definition.params:
                raise FormulaValidationError(f"step `{step.id}` has unknown parameter `{param_name}`")
            if isinstance(value_param, str) and value_param in spec.parameters:
                continue
        for required in definition.requires:
            if required in step.bindings:
                source = _source_name(step.bindings[required])
                if source not in names and source not in seen:
                    raise FormulaValidationError(f"step `{step.id}` missing earlier source `{source}` for `{required}`")
            elif required not in names and required not in seen:
                raise FormulaValidationError(f"step `{step.id}` missing required binding `{required}`")
        # A step's named result is the only implicit output.  The runtime
        # forces dynamic save_as-style outputs to this name, so there is no
        # hidden reduction or detach between steps.
        names.add(step.id)
    if not spec.outputs:
        raise FormulaValidationError("formula must declare at least one output")
    for name, output in spec.outputs.items():
        if not name or not name.replace("_", "").isalnum():
            raise FormulaValidationError(f"invalid formula output name: {name}")
        source = _step_output_name(output.source, output.source)
        if source not in names or source in spec.inputs or source in spec.parameters:
            raise FormulaValidationError(f"formula output `{name}` references unknown result `{output.source}`")
    return spec
