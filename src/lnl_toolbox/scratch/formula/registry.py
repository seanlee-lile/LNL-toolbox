"""Formula registry backed by the existing Scratch Block Registry."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ..registry import BLOCKS, BlockDefinition, register_block
from .schema import FormulaSpec
from .validation import validate_formula


FORMULA_BLOCK_PREFIX = "formula__"
FORMULAS: dict[str, FormulaSpec] = {}
_BUILTINS_LOADED = False


def formula_block_id(formula_id: str) -> str:
    return FORMULA_BLOCK_PREFIX + formula_id.replace("/", "__")


def external_formula_id(block_id: str) -> str | None:
    if block_id.startswith("formula/"):
        return block_id.removeprefix("formula/")
    if block_id.startswith(FORMULA_BLOCK_PREFIX):
        return block_id.removeprefix(FORMULA_BLOCK_PREFIX).replace("__", "/", 1)
    return None


def get_formula(formula_id: str) -> FormulaSpec:
    try:
        return FORMULAS[formula_id]
    except KeyError as exc:
        raise KeyError(f"unknown Scratch formula: {formula_id}") from exc


def list_formulas() -> tuple[FormulaSpec, ...]:
    register_builtin_formulas()
    return tuple(FORMULAS[key] for key in sorted(FORMULAS))


def _formula_execute(spec: FormulaSpec):
    def execute(ctx, **params):
        from .runtime import execute_formula

        input_bindings = {name: params.get(name, name) for name in spec.inputs}
        parameter_values = {name: params.get(name, parameter.default) for name, parameter in spec.parameters.items()}
        output_bindings = {}
        if len(spec.outputs) == 1:
            output_name = next(iter(spec.outputs))
            output_bindings[output_name] = params.get("save_as", output_name)
        else:
            for output_name in spec.outputs:
                output_bindings[output_name] = params.get(f"{output_name}_as", output_name)
        execute_formula(spec, ctx, input_bindings=input_bindings, parameter_values=parameter_values, output_bindings=output_bindings)

    return execute


def _register_formula_block(spec: FormulaSpec) -> None:
    block_id = formula_block_id(spec.id)
    if block_id in BLOCKS:
        return
    params: dict[str, dict[str, Any]] = {
        name: {"type": "slot", "default": name, "description": input_spec.description}
        for name, input_spec in spec.inputs.items()
    }
    type_map = {"float": "float", "int": "int", "bool": "bool", "enum": "enum", "str": "str", "value": "value"}
    for name, parameter in spec.parameters.items():
        entry: dict[str, Any] = {"type": type_map.get(parameter.type, "value"), "default": parameter.default}
        if parameter.minimum is not None:
            entry["min"] = parameter.minimum
        if parameter.maximum is not None:
            entry["max"] = parameter.maximum
        if parameter.options is not None:
            entry["options"] = list(parameter.options)
        params[name] = entry
    if len(spec.outputs) == 1:
        params["save_as"] = {"type": "slot", "default": next(iter(spec.outputs))}
        provides = ("save_as",)
    else:
        provides = tuple(f"{name}_as" for name in spec.outputs)
        for name in spec.outputs:
            params[f"{name}_as"] = {"type": "slot", "default": name}
    register_block(BlockDefinition(
        id=block_id,
        name=spec.name,
        category="Formula",
        description=spec.description or "User-composed Scratch formula",
        kind="action",
        params=params,
        requires=tuple(spec.inputs),
        provides=provides,
        execute=_formula_execute(spec),
        placement=("batch", "top", "epoch"),
        stage="train",
        ui_group="⑧ 用户公式",
        formula_safe=True,
        formula_group="formula",
        formula="; ".join(f"{step.id}={step.block}" for step in spec.steps),
        formula_ref=spec.id,
    ))


def register_formula(value: FormulaSpec | Mapping[str, Any], *, replace: bool = False) -> FormulaSpec:
    spec = validate_formula(value)
    if spec.id in FORMULAS and not replace:
        raise ValueError(f"duplicate Scratch formula id: {spec.id}")
    if replace:
        FORMULAS.pop(spec.id, None)
        BLOCKS.pop(formula_block_id(spec.id), None)
    FORMULAS[spec.id] = spec
    _register_formula_block(spec)
    return spec


def validate_and_register_formula(value: FormulaSpec | Mapping[str, Any], *, replace: bool = False) -> FormulaSpec:
    return register_formula(value, replace=replace)


def unregister_formula(formula_id: str) -> FormulaSpec:
    spec = get_formula(formula_id)
    FORMULAS.pop(formula_id, None)
    from ..registry import BLOCKS

    BLOCKS.pop(formula_block_id(formula_id), None)
    return spec


def register_builtin_formulas() -> tuple[FormulaSpec, ...]:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return tuple(FORMULAS.values())
    _BUILTINS_LOADED = True
    from .storage import load_formula_file

    root = Path(__file__).parent / "examples"
    for path in sorted(root.glob("*.yaml")):
        register_formula(load_formula_file(path))
    return tuple(FORMULAS.values())


def reload_formulas(root: str | Path | None = None) -> tuple[FormulaSpec, ...]:
    from .storage import formula_root, list_formula_files, load_formula_file
    for spec_id in list(FORMULAS):
        if spec_id.startswith("user/"):
            unregister_formula(spec_id)
    directory = formula_root(root)
    pending = [load_formula_file(path) for path in list_formula_files(directory.parent)]
    # Register dependencies before dependants; a user may name Formula A in
    # Formula B even when filenames sort in the opposite order.
    while pending:
        progress = False
        remaining = []
        for spec in pending:
            try:
                register_formula(spec)
            except (KeyError, ValueError) as exc:
                if "unknown Scratch formula" in str(exc) or "unknown Scratch block" in str(exc):
                    remaining.append(spec)
                else:
                    raise
            else:
                progress = True
        if not progress:
            missing = ", ".join(spec.id for spec in remaining)
            raise ValueError(f"could not load formula dependencies: {missing}")
        pending = remaining
    return list_formulas()


def collect_formula_provenance(recipe: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return immutable formula snapshots referenced by a Recipe."""
    from .runtime import formula_hash

    result: list[dict[str, Any]] = []

    def visit(steps: Any) -> None:
        if not isinstance(steps, list):
            return
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            block_id = str(step.get("block", ""))
            formula_id = external_formula_id(block_id)
            if formula_id:
                try:
                    spec = get_formula(formula_id)
                except KeyError:
                    pass
                else:
                    item = {
                        "formula_id": spec.id,
                        "formula_name": spec.name,
                        "formula_hash": formula_hash(spec),
                        "formula_snapshot": deepcopy(spec.to_dict()),
                    }
                    if not any(old["formula_id"] == item["formula_id"] and old["formula_hash"] == item["formula_hash"] for old in result):
                        result.append(item)
            visit(step.get("steps"))

    visit(recipe.get("steps", []))
    return result
