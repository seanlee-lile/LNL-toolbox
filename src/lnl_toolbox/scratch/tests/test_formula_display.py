from __future__ import annotations

import re
from pathlib import Path
import unittest

from lnl_toolbox.scratch.registry import list_blocks


WEB_SCRIPT = Path(__file__).resolve().parents[1] / "web" / "scratch.js"


class ScratchFormulaDisplayTest(unittest.TestCase):
    def _display_catalogue(self) -> str:
        source = WEB_SCRIPT.read_text(encoding="utf-8")
        start = source.index("const DISPLAY_FORMULAS = Object.freeze({")
        end = source.index("});", start)
        return source[start:end]

    def test_every_registered_formula_has_a_display_entry(self) -> None:
        catalogue = self._display_catalogue()
        displayed = set(re.findall(r"^\s{2}([A-Za-z0-9_]+):\s+String\.raw", catalogue, re.MULTILINE))
        registered = {block.id for block in list_blocks() if block.formula}
        self.assertEqual(registered - displayed, set())

    def test_display_uses_explicit_mathematical_operations(self) -> None:
        catalogue = self._display_catalogue()
        # These were the ambiguous shorthand strings that made the formula bar
        # claim a different operation than the corresponding Scratch block.
        for obsolete in ("+/-g", "arg lowest_k", "T^T p", "mean(v[indices])", "CE(z, y)"):
            self.assertNotIn(obsolete, catalogue)
        for required in (
            r"subtract: String.raw`z=x-y`",
            r"divide: String.raw`z=x\div y`",
            r"gather_by_label: String.raw`v_i=V_{i,y_i}`",
            r"mean_by_indices: String.raw`\operatorname{mean}_{j\in I}",
            r"select_by_indices: String.raw`v_{\mathrm{selected}}",
            r"select_lowest_scores: String.raw`I=\operatorname{argsort}",
            r"formula__builtin__gce: String.raw`\mathcal{L}_q=",
            r"mean_squared_error: String.raw`e_i=",
            r"importance_weight_formula: String.raw`w_i=\operatorname{detach}",
        ):
            self.assertIn(required, catalogue)

    def test_function_reduction_and_probability_operations_show_equations(self) -> None:
        catalogue = self._display_catalogue()
        # These operations previously fell back to prose in the formula
        # editor even though their runtime semantics have compact equations.
        for required in (
            r"detach: String.raw`z=\operatorname{stopgrad}(x)`",
            r"negative_log: String.raw`z=-\log\left(\max(x,\varepsilon)\right)`",
            r"one_hot: String.raw`O_{i,c}=1[y_i=c]",
            r"row_normalize: String.raw`z_{i,c}=\frac{x_{i,c}}{\max(\sum_jx_{i,j},\varepsilon)}`",
            r"positive_logdet: String.raw`z=\log\det(X),\ \det(X)>0`",
            r"ones_like: String.raw`z_i=1",
            r"zeros_like: String.raw`z_i=0",
            r"masked_mean: String.raw`L(d=selected)=\frac{\sum_i m_i v_i}",
            r"apply_transition: String.raw`\tilde{p}_i=p_iT_i`",
            r"complementary_negative_loss: String.raw`L_i=-\sum_c m_{i,c}",
            r"mae_loss: String.raw`L_i=\frac{1}{C}\sum_c",
            r"partial_label_loss: String.raw`\mathcal{L}_i=\lambda",
            r"prior_kl: String.raw`D=\sum_c\pi_c",
            r"uniform_prior: String.raw`\pi_c=\frac{1}{C}",
            r"safe_divide: String.raw`z=\frac{x}{\max(y,\varepsilon)}`",
        ):
            self.assertIn(required, catalogue)

    def test_render_sites_use_display_catalogue(self) -> None:
        source = WEB_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("function formulaDisplayText(info)", source)
        self.assertIn("function formulaEditorOperationIntro(info)", source)
        self.assertIn("function renderFormulaOrIntro(container, info", source)
        self.assertGreaterEqual(source.count("renderFormulaOrIntro("), 6)
        self.assertNotIn("该运算没有单独公式标注", source)
        self.assertNotIn("暂无独立公式', {compact", source)
        self.assertNotIn("无独立公式', {compact", source)

    def test_operations_without_equations_have_registry_introductions(self) -> None:
        source = WEB_SCRIPT.read_text(encoding="utf-8")
        operations = [
            block for block in list_blocks()
            if block.kind == "action" and block.formula_kind and not block.formula
        ]
        self.assertTrue(operations)
        self.assertEqual([block.id for block in operations if not block.description], [])
        self.assertIn("formulaEditorOperationIntro(info)", source)
        self.assertIn("renderFormulaOrIntro(formula, info, {compact: true})", source)


if __name__ == "__main__":
    unittest.main()
