"""YAML storage for user formulas in the Scratch user workspace."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import FormulaSpec
from .validation import validate_formula


def formula_root(root: str | Path | None = None) -> Path:
    value = root or os.environ.get("LNL_SCRATCH_WORKSPACE")
    candidates = [Path(value).expanduser()] if value else [Path.home() / ".lnl_toolbox" / "scratch", Path.cwd() / "artifacts" / "scratch"]
    for destination in candidates:
        result = destination / "formulas"
        try:
            result.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return result
    raise PermissionError("unable to create a writable Scratch formula workspace")


def _slug(formula_id: str) -> str:
    if "/" not in formula_id:
        raise ValueError("formula id must contain a namespace, such as user/my_formula")
    namespace, name = formula_id.split("/", 1)
    if namespace != "user":
        raise ValueError("only user formulas may be saved to the user workspace")
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name):
        raise ValueError("formula id contains invalid characters")
    return f"{name}.yaml"


def _safe_path(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if resolved.parent != root.resolve():
        raise ValueError("formula path must stay inside the Scratch formula workspace")
    return resolved


def list_formula_files(root: str | Path | None = None) -> list[Path]:
    return sorted(formula_root(root).glob("*.yaml"))


def save_formula(value: FormulaSpec | Mapping[str, Any], root: str | Path | None = None, *, overwrite: bool = False) -> Path:
    spec = validate_formula(value)
    if not spec.id.startswith("user/"):
        raise ValueError("only user formulas can be saved")
    directory = formula_root(root)
    destination = _safe_path(directory / _slug(spec.id), directory)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"formula already exists: {spec.id}")
    destination.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return destination


def load_formula_file(path: str | Path) -> FormulaSpec:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return validate_formula(payload)


def load_formula(formula_id_or_path: str | Path, root: str | Path | None = None) -> FormulaSpec:
    path = Path(formula_id_or_path)
    if not path.suffix:
        formula_id = str(formula_id_or_path)
        path = formula_root(root) / _slug(formula_id)
    return load_formula_file(_safe_path(path, formula_root(root)))


def delete_formula(formula_id: str, root: str | Path | None = None, *, referenced_by: list[str] | tuple[str, ...] = ()) -> FormulaSpec:
    if referenced_by:
        raise ValueError(f"formula `{formula_id}` is still used by: {', '.join(referenced_by)}")
    spec = load_formula(formula_id, root)
    path = _safe_path(formula_root(root) / _slug(spec.id), formula_root(root))
    path.unlink()
    return spec


def rename_formula(old_id: str, new_id: str, root: str | Path | None = None) -> FormulaSpec:
    spec = load_formula(old_id, root)
    updated = FormulaSpec(
        id=new_id,
        name=spec.name,
        description=spec.description,
        inputs=spec.inputs,
        parameters=spec.parameters,
        steps=spec.steps,
        outputs=spec.outputs,
        metadata=spec.metadata,
    )
    save_formula(updated, root)
    delete_formula(old_id, root)
    return updated


def export_formula(formula_id: str, destination: str | Path | None = None, root: str | Path | None = None) -> str | Path:
    spec = load_formula(formula_id, root)
    text = yaml.safe_dump(spec.to_dict(), sort_keys=False, allow_unicode=True)
    if destination is None:
        return text
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def import_formula(source: str | Path | Mapping[str, Any], root: str | Path | None = None, *, overwrite: bool = False) -> FormulaSpec:
    if isinstance(source, Mapping):
        payload = source
    else:
        text_or_path = Path(source)
        try:
            is_path = text_or_path.is_file()
        except OSError:
            is_path = False
        text = text_or_path.read_text(encoding="utf-8") if is_path else str(source)
        payload = yaml.safe_load(text)
    return validate_formula(save_and_load_payload(payload, root, overwrite=overwrite))


def find_formula_references(recipe_root: str | Path, formula_id: str) -> list[str]:
    """Find recipe names that reference a formula before deletion."""
    root = Path(recipe_root)
    references: list[str] = []
    wanted = {formula_id, "formula/" + formula_id, "formula__" + formula_id.replace("/", "__")}

    def visit(steps: Any) -> bool:
        if not isinstance(steps, list):
            return False
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            if str(step.get("block", "")) in wanted or visit(step.get("steps")):
                return True
        return False

    if root.exists():
        for path in root.rglob("*.y*ml"):
            try:
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(payload, Mapping) and visit(payload.get("steps")):
                references.append(str(path.relative_to(root)).replace("\\", "/"))
    return sorted(references)


def save_and_load_payload(payload: Mapping[str, Any], root: str | Path | None, *, overwrite: bool = False) -> FormulaSpec:
    spec = validate_formula(payload)
    save_formula(spec, root, overwrite=overwrite)
    return spec
