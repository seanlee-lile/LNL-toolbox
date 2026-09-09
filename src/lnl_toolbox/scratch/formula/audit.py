"""Audits for the three-layer Scratch formula language.

The audit deliberately walks registered definitions and their actual FormulaSpec
steps.  A ``formula_kind`` label without a readable definition is therefore
reported as broken instead of being treated as composable by convention.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..registry import get_block, list_blocks
from .composites import get_composite_formula, register_composite_formulas
from .registry import external_formula_id, get_formula, register_builtin_formulas
from .schema import FormulaSpec


def _formula_for_block(block_id: str, formula_ref: str | None) -> FormulaSpec | None:
    nested_id = external_formula_id(block_id)
    if nested_id:
        try:
            return get_formula(nested_id)
        except KeyError:
            return None
    if not formula_ref:
        return None
    try:
        return get_composite_formula(formula_ref)
    except KeyError:
        try:
            return get_formula(formula_ref)
        except KeyError:
            return None


def _walk_spec(spec: FormulaSpec, *, seen: tuple[str, ...]) -> tuple[set[str], set[str], int, str]:
    if spec.id in seen:
        return set(), set(), 0, "BROKEN_COMPOSITE"
    leaves_primitive: set[str] = set()
    leaves_special: set[str] = set()
    depth = 1
    status = "COMPOSABLE"
    for step in spec.steps:
        definition = get_block(step.block)
        if definition.formula_kind == "primitive":
            leaves_primitive.add(definition.id)
            continue
        if definition.formula_kind == "special":
            leaves_special.add(definition.id)
            continue
        if definition.formula_kind != "composite":
            return set(), set(), depth, "BROKEN_COMPOSITE"
        child = _formula_for_block(definition.id, definition.formula_ref)
        if child is None:
            return set(), set(), depth, "MISSING_DEFINITION"
        p, s, child_depth, child_status = _walk_spec(child, seen=(*seen, spec.id))
        leaves_primitive.update(p)
        leaves_special.update(s)
        depth = max(depth, child_depth + 1)
        if child_status != "COMPOSABLE":
            status = child_status
    return leaves_primitive, leaves_special, depth, status


def formula_closure() -> list[dict[str, Any]]:
    """Return one closure record for every registered formula-bearing block."""
    register_builtin_formulas()
    register_composite_formulas()
    records: list[dict[str, Any]] = []
    for definition in list_blocks():
        if definition.formula_kind is None:
            continue
        if definition.formula_kind != "composite":
            records.append({
                "block_id": definition.id,
                "formula_kind": definition.formula_kind,
                "formula_ref": definition.formula_ref or "",
                "leaf_primitives": definition.id if definition.formula_kind == "primitive" else "",
                "leaf_specials": definition.id if definition.formula_kind == "special" else "",
                "depth": 0,
                "status": "PRIMITIVE" if definition.formula_kind == "primitive" else "SPECIAL",
            })
            continue
        spec = _formula_for_block(definition.id, definition.formula_ref)
        if spec is None:
            records.append({
                "block_id": definition.id,
                "formula_kind": definition.formula_kind,
                "formula_ref": definition.formula_ref or "",
                "leaf_primitives": "",
                "leaf_specials": "",
                "depth": 0,
                "status": "MISSING_DEFINITION",
            })
            continue
        primitives, specials, depth, status = _walk_spec(spec, seen=())
        records.append({
            "block_id": definition.id,
            "formula_kind": definition.formula_kind,
            "formula_ref": definition.formula_ref or "",
            "leaf_primitives": "|".join(sorted(primitives)),
            "leaf_specials": "|".join(sorted(specials)),
            "depth": depth,
            "status": status,
        })
    return records


def formula_classification() -> list[dict[str, Any]]:
    """Return a deterministic classification record for every formula block."""
    register_builtin_formulas()
    rows: list[dict[str, Any]] = []
    for definition in list_blocks():
        if definition.formula_kind is None and definition.formula is None:
            continue
        rows.append({
            "block_id": definition.id,
            "formula_kind": definition.formula_kind or "UNKNOWN",
            "formula_group": definition.formula_group or "",
            "has_formula": bool(definition.formula),
            "reason": definition.formula_ref or definition.description,
        })
    return rows


def write_formula_audit(directory: str | Path = "artifacts/scratch_formula_audit") -> tuple[Path, Path]:
    """Write TSV and Markdown closure/classification reports for reviewers."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    classification_path = destination / "formula_classification.tsv"
    closure_path = destination / "formula_closure.tsv"
    with classification_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["block_id", "formula_kind", "formula_group", "has_formula", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(formula_classification())
    rows = formula_closure()
    with closure_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["block_id", "formula_kind", "formula_ref", "leaf_primitives", "leaf_specials", "depth", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report_path = destination / "formula_closure_report.md"
    report_path.write_text(
        "# Scratch formula closure\n\n"
        f"- Primitive: {sum(row['formula_kind'] == 'primitive' for row in rows)}\n"
        f"- Special: {sum(row['formula_kind'] == 'special' for row in rows)}\n"
        f"- Composite: {sum(row['formula_kind'] == 'composite' for row in rows)}\n"
        f"- Fully expandable: {counts.get('COMPOSABLE', 0)}\n"
        f"- Missing definition: {counts.get('MISSING_DEFINITION', 0)}\n"
        f"- Broken composite: {counts.get('BROKEN_COMPOSITE', 0)}\n"
        f"- Maximum depth: {max((int(row['depth']) for row in rows), default=0)}\n",
        encoding="utf-8",
    )
    return classification_path, closure_path


__all__ = ["formula_classification", "formula_closure", "write_formula_audit"]
