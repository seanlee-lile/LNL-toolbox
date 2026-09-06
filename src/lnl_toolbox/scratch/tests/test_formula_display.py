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
            r"gather_by_label: String.raw`v_i=V_{i,y_i}`",
            r"mean_by_indices: String.raw`\operatorname{mean}_{j\in I}",
            r"select_by_indices: String.raw`v_{\mathrm{selected}}",
            r"select_lowest_scores: String.raw`I=\operatorname{argsort}",
            r"formula__builtin__gce: String.raw`\mathcal{L}_q=",
            r"mean_squared_error: String.raw`e_i=",
            r"importance_weight_formula: String.raw`w_i=\operatorname{detach}",
        ):
            self.assertIn(required, catalogue)

    def test_render_sites_use_display_catalogue(self) -> None:
        source = WEB_SCRIPT.read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("formulaDisplayText(info)"), 5)
        self.assertIn("function formulaDisplayText(info)", source)


if __name__ == "__main__":
    unittest.main()
