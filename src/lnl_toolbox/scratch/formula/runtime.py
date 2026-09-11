"""Sequential execution of FormulaSpec through the existing Scratch Blocks."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from ..context import ScratchContext
from ..registry import get_block
from .schema import FormulaSpec
from .validation import validate_formula


def formula_hash(value: FormulaSpec | Mapping[str, Any]) -> str:
    spec = value if isinstance(value, FormulaSpec) else FormulaSpec.from_dict(value)
    payload = json.dumps(spec.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_value(value: Any, local: Mapping[str, Any], parameters: Mapping[str, Any]) -> Any:
    if isinstance(value, list):
        return [_resolve_value(item, local, parameters) if isinstance(item, (list, dict)) or item in parameters else item for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_value(item, local, parameters) if isinstance(item, (list, dict)) or item in parameters else item for item in value)
    if isinstance(value, dict):
        return {key: _resolve_value(item, local, parameters) for key, item in value.items()}
    if isinstance(value, str):
        if value.startswith("$"):
            key = value[1:]
            if key not in local:
                raise KeyError(f"formula parameter/source not found: {key}")
            return local[key]
        if value in parameters:
            return parameters[value]
    return value


def _variant_matches(when: Mapping[str, Any], local: Mapping[str, Any], parameters: Mapping[str, Any]) -> bool:
    """Return whether an executable FormulaSpec variant applies."""
    for key, expected in when.items():
        if key in parameters:
            actual = parameters[key]
            present = True
        else:
            actual = local.get(key)
            present = key in local and actual is not None
        if isinstance(expected, Mapping):
            if "present" in expected and bool(expected["present"]) != present:
                return False
            if "equals" in expected and (not present or actual != expected["equals"]):
                return False
            continue
        elif isinstance(expected, str) and expected in {"__present__", "$present"}:
            if not present:
                return False
        elif isinstance(expected, str) and expected in {"__absent__", "$absent"}:
            if present:
                return False
        elif not present or actual != expected:
            return False
    return True


def execute_formula(
    value: FormulaSpec | Mapping[str, Any],
    context: ScratchContext | Mapping[str, Any],
    *,
    input_bindings: Mapping[str, str] | None = None,
    parameter_values: Mapping[str, Any] | None = None,
    output_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute a formula in an isolated local slot table.

    Existing Block callables are the only source of computation.  The parent
    context is read for inputs and receives only named formula outputs; local
    intermediate slots never leak into the Recipe context.
    """

    spec = validate_formula(value)
    parent = context
    inputs = dict(input_bindings or {})
    supplied_parameters = dict(parameter_values or {})
    local = ScratchContext()
    for name, input_spec in spec.inputs.items():
        source = inputs.get(name, name)
        if not isinstance(source, str) or source not in parent:
            if input_spec.required is False:
                local[name] = None
                continue
            raise KeyError(f"formula input `{name}` is not bound to an available slot")
        local[name] = parent[source]
    resolved_parameters: dict[str, Any] = {}
    for name, parameter in spec.parameters.items():
        if name in supplied_parameters:
            current = supplied_parameters[name]
        else:
            current = parameter.default
        if parameter.type == "float":
            current = float(current)
        elif parameter.type == "int":
            current = int(current)
        elif parameter.type == "bool":
            current = bool(current)
        elif parameter.type == "enum" and parameter.options and current not in parameter.options:
            raise ValueError(f"formula parameter `{name}` must be one of {parameter.options}")
        if parameter.minimum is not None and current < parameter.minimum:
            raise ValueError(f"formula parameter `{name}` is below its minimum")
        if parameter.maximum is not None and current > parameter.maximum:
            raise ValueError(f"formula parameter `{name}` exceeds its maximum")
        resolved_parameters[name] = current
        local[name] = current

    selected_variant = next((variant for variant in spec.variants if _variant_matches(variant.when, local, resolved_parameters)), None)
    if spec.variants and selected_variant is None:
        raise ValueError(f"formula `{spec.id}` has no matching executable variant")
    steps = selected_variant.steps if selected_variant is not None else spec.steps
    selected_outputs = (selected_variant.outputs or spec.outputs) if selected_variant is not None else spec.outputs
    for step in steps:
        definition = get_block(step.block)
        params: dict[str, Any] = {}
        for name, schema in definition.params.items():
            if name in step.bindings:
                raw = step.bindings[name]
            elif name in step.parameters:
                raw = step.parameters[name]
            elif name in definition.requires and name in local:
                raw = name
            elif "default" in schema:
                raw = schema["default"]
            else:
                continue
            if schema.get("type") == "slot":
                if name in definition.provides:
                    if not isinstance(raw, str) or not raw.strip():
                        raise ValueError(f"formula step `{step.id}` output slot `{name}` must be named")
                    params[name] = raw
                    continue
                if not isinstance(raw, str) or raw not in local:
                    raise KeyError(f"formula step `{step.id}` slot `{name}` is not available: {raw!r}")
                params[name] = raw
            else:
                params[name] = _resolve_value(raw, local, resolved_parameters)
        # Dynamic output parameters (save_as, *_as) are wired to the named
        # step result.  This preserves the existing Block contract exactly.
        for provided in definition.provides:
            if provided in definition.params:
                params[provided] = step.id
        definition.execute(local, **params)
        if step.id not in local:
            candidates = [local.get(name) for name in definition.provides if name in local]
            if candidates:
                local[step.id] = candidates[0]
            else:
                raise RuntimeError(f"formula step `{step.id}` did not publish an output")

    bindings = dict(output_bindings or {})
    for name, output in selected_outputs.items():
        source = output.source.split(".", 1)[0]
        if source not in local:
            raise KeyError(f"formula output `{name}` source is unavailable: {output.source}")
        target = bindings.get(name, name)
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"formula output `{name}` requires a target slot")
        parent[target] = local[source]
    provenance = {
        "formula_id": spec.id,
        "formula_name": spec.name,
        "formula_hash": formula_hash(spec),
        "formula_version": formula_hash(spec),
        "formula_snapshot": deepcopy(spec.to_dict()),
    }
    if isinstance(parent, dict):
        records = parent.setdefault("_formula_provenance", [])
        if isinstance(records, list):
            # Nested Formula Blocks execute against this formula's local
            # context. Promote their immutable snapshots to the caller before
            # recording the outer formula, otherwise nested provenance would
            # disappear when the local context is discarded.
            nested_records = local.get("_formula_provenance", [])
            for nested in nested_records if isinstance(nested_records, list) else []:
                if isinstance(nested, dict) and not any(
                    item.get("formula_id") == nested.get("formula_id")
                    and item.get("formula_hash") == nested.get("formula_hash")
                    for item in records
                    if isinstance(item, dict)
                ):
                    records.append(deepcopy(nested))
            if not any(
                item.get("formula_id") == spec.id and item.get("formula_hash") == provenance["formula_hash"]
                for item in records
                if isinstance(item, dict)
            ):
                records.append(provenance)
    return {name: parent[bindings.get(name, name)] for name in selected_outputs}
