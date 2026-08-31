"""User-composable, YAML-backed Scratch formulas.

Formulas are deliberately a thin layer over the existing Scratch Block
Registry.  They contain no Python code and do not introduce a second
executor: validation resolves Registry metadata and runtime executes the
registered block callables in order.
"""

from .registry import (
    FORMULA_BLOCK_PREFIX,
    collect_formula_provenance,
    get_formula,
    list_formulas,
    register_formula,
    register_builtin_formulas,
    reload_formulas,
    unregister_formula,
    validate_and_register_formula,
)
from .runtime import execute_formula, formula_hash
from .schema import (
    FormulaInputSpec,
    FormulaOutputSpec,
    FormulaParameterSpec,
    FormulaSpec,
    FormulaStepSpec,
)
from .validation import FormulaValidationError, validate_formula

__all__ = [
    "FORMULA_BLOCK_PREFIX",
    "FormulaInputSpec",
    "FormulaOutputSpec",
    "FormulaParameterSpec",
    "FormulaSpec",
    "FormulaStepSpec",
    "FormulaValidationError",
    "collect_formula_provenance",
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
]
