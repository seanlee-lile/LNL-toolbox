"""Machine-readable definitions for Scratch composite operations.

Composite definitions are ordinary :class:`FormulaSpec` values stored beside
the runtime, not a second executor.  They make the public composition of a
composite block inspectable while the existing block remains the executable
oracle until value/gradient equivalence is established.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import FormulaSpec
from .validation import validate_formula


_COMPOSITES: dict[str, FormulaSpec] = {}
_LOADED = False


def _root() -> Path:
    return Path(__file__).with_name("composites")


def register_composite_formula(value: FormulaSpec | dict[str, Any]) -> FormulaSpec:
    spec = validate_formula(value)
    if not spec.id.startswith("builtin/"):
        raise ValueError("composite definitions must use the builtin namespace")
    _COMPOSITES[spec.id] = spec
    return spec


def register_composite_formulas() -> tuple[FormulaSpec, ...]:
    global _LOADED
    if _LOADED:
        return tuple(_COMPOSITES.values())
    # Mark the registry loaded before parsing so a malformed definition cannot
    # recursively trigger another directory scan.
    _LOADED = True
    for path in sorted(_root().glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        register_composite_formula(payload)
    return tuple(_COMPOSITES.values())


def get_composite_formula(block_id: str) -> FormulaSpec:
    register_composite_formulas()
    key = str(block_id)
    if not key.startswith("builtin/"):
        key = "builtin/" + key
    try:
        return _COMPOSITES[key]
    except KeyError as exc:
        raise KeyError(f"unknown Scratch composite definition: {block_id}") from exc


def has_composite_formula(block_id: str) -> bool:
    try:
        get_composite_formula(block_id)
    except KeyError:
        return False
    return True


def list_composite_formulas() -> tuple[FormulaSpec, ...]:
    register_composite_formulas()
    return tuple(_COMPOSITES[key] for key in sorted(_COMPOSITES))


__all__ = [
    "get_composite_formula",
    "has_composite_formula",
    "list_composite_formulas",
    "register_composite_formula",
    "register_composite_formulas",
]
