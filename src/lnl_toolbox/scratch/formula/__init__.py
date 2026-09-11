"""User-composable, YAML-backed Scratch formulas.

Formulas are deliberately a thin layer over the existing Scratch Block
Registry.  They contain no Python code and do not introduce a second
executor: validation resolves Registry metadata and runtime executes the
registered block callables in order.
"""

from .registry import (
    FORMULA_BLOCK_PREFIX,
    collect_formula_provenance,
    merge_formula_provenance,
    get_formula,
    list_formulas,
    register_formula,
    register_builtin_formulas,
    reload_formulas,
    unregister_formula,
    validate_and_register_formula,
)
from .composites import (
    get_composite_formula,
    has_composite_formula,
    list_composite_formulas,
    register_composite_formulas,
)
from .audit import composite_parameter_coverage, formula_classification, formula_closure, special_audit, write_formula_audit
from .runtime import execute_formula, formula_hash
from .schema import (
    FormulaInputSpec,
    FormulaOutputSpec,
    FormulaParameterSpec,
    FormulaSpec,
    FormulaStepSpec,
    FormulaVariantSpec,
)
from .validation import FormulaValidationError, validate_formula

__all__ = [
    "FORMULA_BLOCK_PREFIX",
    "FormulaInputSpec",
    "FormulaOutputSpec",
    "FormulaParameterSpec",
    "FormulaSpec",
    "FormulaStepSpec",
    "FormulaVariantSpec",
    "FormulaValidationError",
    "collect_formula_provenance",
    "merge_formula_provenance",
    "execute_formula",
    "formula_hash",
    "get_formula",
    "list_formulas",
    "register_builtin_formulas",
    "register_formula",
    "reload_formulas",
    "unregister_formula",
    "validate_and_register_formula",
    "validate_formula",
    "get_composite_formula",
    "has_composite_formula",
    "list_composite_formulas",
    "register_composite_formulas",
    "formula_classification",
    "formula_closure",
    "special_audit",
    "composite_parameter_coverage",
    "write_formula_audit",
]
