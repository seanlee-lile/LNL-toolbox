"""Audits for the three-layer Scratch formula language.

The audit deliberately walks registered definitions and their actual FormulaSpec
steps.  A ``formula_kind`` label without a readable definition is therefore
reported as broken instead of being treated as composable by convention.
"""

from __future__ import annotations

import csv
import ast
import hashlib
import inspect
import re
import textwrap
from pathlib import Path
from typing import Any

from ..registry import get_block, list_blocks
from .composites import get_composite_formula, register_composite_formulas
from .registry import external_formula_id, get_formula, register_builtin_formulas
from .schema import FormulaSpec


# ``BlockDefinition.params`` includes dynamic output slots (``save_as`` or
# ``*_as``), while FormulaSpec inputs/parameters describe the mathematical
# operands.  Keep the few historical aliases explicit for the audit.
_OUTPUT_PARAM_RE = re.compile(r"(?:^save_as$|_as$)")
_PARAMETER_ALIASES = {"input": ("x", "values", "probabilities"), "minimum": ("epsilon",)}

def _runtime_body(definition: Any) -> str:
    try:
        source = textwrap.dedent(inspect.getsource(definition.execute))
        tree = ast.parse(source)
        function = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if function is None or not function.body:
            return source
        start = getattr(function.body[0], "lineno", function.lineno)
        end = getattr(function.body[-1], "end_lineno", start)
        return "\n".join(source.splitlines()[start - 1:end])
    except (OSError, TypeError, SyntaxError):
        return ""


def _special_reason(definition: Any, body: str) -> tuple[str, str]:
    """Classify a special body from executable evidence, conservatively.

    This intentionally does not consult a Block-ID allow-list.  Retention is
    justified by syntax present in the callable body (state mutation, model
    lifecycle, stochastic sampling, autograd, or numerical estimation).  A
    plain tensor expression therefore becomes an actionable migration finding.
    """
    lowered = body.lower()
    flags: list[str] = []
    if any(token in lowered for token in ("autograd", "backward(", "create_graph", "gradient", "requires_grad")):
        flags.append("autograd")
    if any(token in lowered for token in ("torch.rand", "torch.randn", "torch.randint", "random", "generator")):
        flags.append("random")
    if any(token in lowered for token in ("torch.linalg", "pinv", "solve(", "slogdet", "lstsq", "scipy", "fit_", "estimate_", "collect_")):
        flags.append("solver/statistics")
    if any(token in lowered for token in ("named_parameters", ".eval(", ".train(", "optimizer", "model", "network")):
        flags.append("model/optimizer internal")
    if definition.category in {"State", "Meta", "Optimization"} or any(token in lowered for token in (".zero_(", "ctx[state]", "ctx[\"state\"]", "state_value", "on_cycle_", "epoch")):
        flags.append("state/lifecycle")
    # Retain a special operation only when its callable body exposes a
    # research-level boundary that ordinary tensor primitives cannot express.
    # These checks inspect executable evidence rather than block IDs or a
    # hand-maintained allow-list.
    evidence = (
        (("transition_for", "einsum", "artifact"), "artifact-aware transition algebra"),
        (("paper_volmin_objective", "diagnostics"), "constrained statistical objective with diagnostics"),
        (("torch.eye", "_cwd_swap_matrix", "virtual_prior"), "virtual-prior matrix-system construction"),
        (("bincount", "observed_prior", "centroid", "snapshot"), "snapshot-derived class statistics"),
        (("pseudoinverse", "torch.stack"), "pseudoinverse/statistical recovery"),
        (("_schedule", "alpha_bar", "beta_bar"), "paper diffusion schedule materialization"),
        (("estimator", "estimate("), "estimator lifecycle"),
        (("searchsorted", "identifiability", "opposite_rate"), "stable-index aligned identifiability weighting"),
        (("for c in range", "reference_losses", "reference_transition"), "conditional covariance accumulation"),
        # Iterative mixture fitting is a statistical estimator even when the
        # implementation is self-contained and does not call a helper named
        # ``fit_*``.  Require the characteristic EM state and update loop as
        # executable evidence rather than retaining it by block ID.
        (("for _ in range", "posterior", "variances", "means"), "iterative mixture estimator"),
    )
    for tokens, label in evidence:
        if all(token in lowered for token in tokens):
            flags.append(label)
            break
    if not flags:
        return "CHANGE_TO_COMPOSITE", "plain tensor arithmetic has no state/random/autograd/solver/model evidence"
    return "KEEP_SPECIAL", "; ".join(dict.fromkeys(flags))


def special_audit() -> list[dict[str, Any]]:
    """Audit every registered ``formula_kind=special`` callable body."""
    register_builtin_formulas()
    rows: list[dict[str, Any]] = []
    for definition in list_blocks():
        if definition.formula_kind != "special":
            continue
        body = _runtime_body(definition)
        decision, reason = _special_reason(definition, body)
        rows.append({
            "block_id": definition.id,
            "current_kind": definition.formula_kind,
            "keep_or_change": decision,
            "reason": reason,
            "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest() if body else "",
        })
    return rows


def _formula_parameter_coverage(definition: Any, spec: FormulaSpec) -> dict[str, Any]:
    block_inputs = {
        name for name, schema in definition.params.items()
        if schema.get("type") == "slot" and not _OUTPUT_PARAM_RE.search(name)
    }
    block_values = {
        name for name, schema in definition.params.items()
        if schema.get("type") != "slot" and not _OUTPUT_PARAM_RE.search(name)
    }
    formula_inputs = set(spec.inputs)
    formula_values = set(spec.parameters)
    aliases: dict[str, str] = {}
    missing: list[str] = []
    for name in sorted(block_inputs):
        candidates = (name, *_PARAMETER_ALIASES.get(name, ()))
        target = next((candidate for candidate in candidates if candidate in formula_inputs), None)
        if target is None:
            missing.append(name)
        elif target != name:
            aliases[name] = target
    for name in sorted(block_values):
        candidates = (name, *_PARAMETER_ALIASES.get(name, ()))
        target = next((candidate for candidate in candidates if candidate in formula_values or candidate in formula_inputs), None)
        if target is not None:
            aliases[name] = target
        else:
            missing.append(name)
    # Formula controls without a corresponding Block parameter are also
    # surfaced.  A branch is valid only when it is executable in the FormulaSpec
    # itself; audit annotations are deliberately not accepted as evidence.
    reverse = set(aliases.values()) | block_values
    missing.extend(sorted(formula_values - reverse))
    missing = sorted(set(missing))
    branch_parameters = sorted({key for variant in spec.variants for key in variant.when})
    # Parameters controlling a real executable variant are represented by the
    # variant's `when` selector and are therefore not hidden branch metadata.
    unrepresented = sorted(set(missing) - set(branch_parameters))
    if not unrepresented:
        status = "PASS"
        reason = "all input/value parameters are represented by executable formula inputs, parameters, or variants"
    else:
        status = "FAIL"
        reason = "formula and executable Block expose different result-changing parameters"
    return {
        "block_id": definition.id,
        "formula_ref": definition.formula_ref or "",
        "block_parameters": "|".join(sorted(definition.params)),
        "formula_inputs": "|".join(sorted(formula_inputs)),
        "formula_parameters": "|".join(sorted(formula_values)),
        "aliases": ";".join(f"{key}->{value}" for key, value in sorted(aliases.items())),
        "missing_parameters": "|".join(unrepresented),
        "branch_parameters": "|".join(branch_parameters),
        "branch_documentation": ";".join(f"{variant.name}: {variant.when}" for variant in spec.variants),
        "branch_status": "EXECUTABLE_FORMULA_VARIANT" if spec.variants else "NONE",
        "status": status,
        "reason": reason,
    }


def composite_parameter_coverage() -> list[dict[str, Any]]:
    """Compare canonical composite Block parameters with FormulaSpec values."""
    register_builtin_formulas()
    register_composite_formulas()
    rows: list[dict[str, Any]] = []
    for definition in list_blocks():
        # FormulaSpecs are themselves exposed as generated formula__ blocks;
        # auditing those would only compare a generated schema with itself.
        if definition.formula_kind != "composite" or definition.id.startswith("formula__"):
            continue
        if not definition.formula_ref:
            rows.append({"block_id": definition.id, "formula_ref": "", "block_parameters": "|".join(sorted(definition.params)), "formula_inputs": "", "formula_parameters": "", "aliases": "", "missing_parameters": "", "branch_parameters": "", "branch_documentation": "", "branch_status": "NONE", "status": "FAIL", "reason": "composite Block has no formula_ref"})
            continue
        try:
            spec = get_composite_formula(definition.formula_ref)
        except KeyError:
            try:
                spec = get_formula(definition.formula_ref)
            except KeyError:
                rows.append({"block_id": definition.id, "formula_ref": definition.formula_ref, "block_parameters": "|".join(sorted(definition.params)), "formula_inputs": "", "formula_parameters": "", "aliases": "", "missing_parameters": "", "branch_parameters": "", "branch_documentation": "", "branch_status": "NONE", "status": "FAIL", "reason": "formula definition is missing"})
                continue
        rows.append(_formula_parameter_coverage(definition, spec))
    return rows


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
    # Variants are executable branches, not documentation. Include every
    # branch in the closure so a composite cannot hide a dependency behind a
    # selector.
    executable_steps = list(spec.steps)
    for variant in spec.variants:
        executable_steps.extend(variant.steps)
    for step in executable_steps:
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
    """Write classification, closure, special and parameter reports."""
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
    special_rows = special_audit()
    special_path = destination / "special_audit.tsv"
    with special_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["block_id", "current_kind", "keep_or_change", "reason", "body_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(special_rows)
    coverage_rows = composite_parameter_coverage()
    coverage_path = destination / "composite_parameter_coverage.tsv"
    with coverage_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["block_id", "formula_ref", "block_parameters", "formula_inputs", "formula_parameters", "aliases", "missing_parameters", "branch_parameters", "branch_documentation", "branch_status", "status", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(coverage_rows)
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
        f"- Maximum depth: {max((int(row['depth']) for row in rows), default=0)}\n"
        f"- Special bodies audited: {len(special_rows)}\n"
        f"- Specials retained: {sum(row['keep_or_change'] == 'KEEP_SPECIAL' for row in special_rows)}\n"
        f"- Specials moved to composite: {sum(row['keep_or_change'] == 'CHANGE_TO_COMPOSITE' for row in special_rows)}\n"
        f"- Composite parameter coverage failures: {sum(row['status'] == 'FAIL' for row in coverage_rows)}\n",
        encoding="utf-8",
    )
    return classification_path, closure_path


__all__ = [
    "composite_parameter_coverage",
    "formula_classification",
    "formula_closure",
    "special_audit",
    "write_formula_audit",
]
