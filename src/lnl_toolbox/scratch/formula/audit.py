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

_SPECIAL_REASON_OVERRIDES = {
    "apply_transition": "artifact-aware 2-D/3-D transition dispatch and namespace alignment",
    "binary_risk": "paper-specific unbiased class-dependent risk estimator with identifiability guard",
    "cal_cores2_adjusted_risk": "paper-defined CORES² adjusted risk with square-root prior normalization and distinct numerical floors",
    "cal_covariance_correction": "conditional covariance estimator over retained proxy classes and detached reference statistics",
    "compose_revision_transition": "trainable transition-artifact composition with model parameter access",
    "cwd_observed_statistics": "CWD observed-prior and class-feature statistics from a noisy snapshot",
    "cwd_recover_centroids": "CWD clean-centroid recovery through virtual-system pseudoinverse algebra",
    "cwd_virtual_systems": "CWD virtual-prior/coefficient-system construction indexed by clean class",
    "dual_t_transition_estimation": "Dual-T anchor and argmax-count transition estimator with artifact diagnostics",
    "fit_gmm": "two-component GMM EM lifecycle with clean-component identification",
    "importance_weight_formula": "binary RCN importance weighting with stable-index alignment and identifiability guard",
    "initialize_t_revision_transition": "T-Revision pseudo-anchor transition estimator with artifact materialization",
    "mentor_build_features": "MentorNet frozen-provider feature contract and curriculum epoch semantics",
    "mc_ldce_volmin_objective": "MC-LDCE PaperVolMin likelihood plus constrained transition diagnostics",
    "one_hot_like": "reference-shaped class-cardinality adapter; output width is inferred from a tensor",
    "pdl_estimate_instance_transition": "PDL part-dependent transition estimator over fitted basis artifacts",
    "pdl_fit_basis_matrices": "PDL class-conditioned basis fitting lifecycle over aligned anchors",
    "pdl_fit_part_representation": "PDL nonnegative part-factorization solver and convergence controls",
    "t_revision_importance_ratio": "paper-specific posterior ratio with denominator validation and diagnostic output",
}


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
    """Classify a special body from executable evidence, conservatively."""
    lowered = body.lower()
    flags: list[str] = []
    if any(token in lowered for token in ("autograd", "backward(", "create_graph", "gradient")):
        flags.append("autograd")
    if any(token in lowered for token in ("torch.rand", "torch.randn", "torch.randint", "random", "generator")):
        flags.append("random")
    if any(token in lowered for token in ("torch.linalg", "pinv", "solve(", "slogdet", "scipy", "fit_", "estimate_")):
        flags.append("solver/statistics")
    if any(token in lowered for token in ("named_parameters", ".eval(", ".train(", "optimizer", "model")):
        flags.append("model/optimizer internal")
    if "state" in lowered or definition.category in {"State", "Meta", "Optimization"}:
        flags.append("state/lifecycle")
    if definition.id in _SPECIAL_REASON_OVERRIDES:
        flags.append(_SPECIAL_REASON_OVERRIDES[definition.id])
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
        target = next((candidate for candidate in candidates if candidate in formula_values), None)
        if target is not None:
            aliases[name] = target
        else:
            missing.append(name)
    # Formula controls without an oracle parameter are also surfaced.  This is
    # what catches a YAML epsilon/reduction that had previously been hard-coded
    # by the Python callable.
    reverse = set(aliases.values()) | block_values
    missing.extend(sorted(formula_values - reverse))
    missing = sorted(set(missing))
    # A sequential FormulaSpec intentionally describes the canonical/default
    # path.  Branches that cannot be encoded as sequential steps must be
    # declared by that YAML itself; the audit must not silently whitelist a
    # Python-only parameter by block id.
    branch_metadata = spec.metadata.get("oracle_branches", {})
    if not isinstance(branch_metadata, dict):
        branch_metadata = {}
    dispatch = sorted(set(branch_metadata) & set(missing))
    unrepresented = sorted(set(missing) - set(dispatch))
    if not unrepresented:
        status = "PASS"
        reason = "all input/value parameters are represented; dispatch branches are recorded separately"
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
        "branch_parameters": "|".join(dispatch),
        "branch_documentation": ";".join(f"{name}: {branch_metadata[name]}" for name in dispatch),
        "branch_status": "DOCUMENTED_YAML_ORACLE_BRANCH" if dispatch else "NONE",
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
