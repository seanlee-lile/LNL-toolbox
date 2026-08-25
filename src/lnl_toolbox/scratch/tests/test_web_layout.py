from __future__ import annotations

from pathlib import Path
import unittest


WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


class ScratchInspectorLayoutTest(unittest.TestCase):
    def test_module_explanation_is_outside_scrollable_inspector(self) -> None:
        html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="module-explanation"', html)
        self.assertIn('id="inspector"', html)
        self.assertLess(html.index('id="module-explanation"'), html.index('id="inspector"'))

        css = (WEB_ROOT / "scratch.css").read_text(encoding="utf-8")
        self.assertIn(".inspector { display: flex; flex-direction: column;", css)
        self.assertIn("#inspector { flex: 1 1 auto;", css)
        self.assertIn("overflow: auto", css)

    def test_render_inspector_updates_fixed_explanation_target(self) -> None:
        javascript = (WEB_ROOT / "scratch.js").read_text(encoding="utf-8")
        self.assertIn("const explanationTarget = $('module-explanation');", javascript)
        self.assertIn("explanationTarget.appendChild(identity);", javascript)
        self.assertNotIn("addSection('1. 名称和说明', identity);", javascript)


if __name__ == "__main__":
    unittest.main()
