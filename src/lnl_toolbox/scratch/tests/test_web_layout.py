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

    def test_palette_has_search_and_category_fold_controls(self) -> None:
        html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="palette-search"', html)
        self.assertIn("单击查看说明；拖动到绿色插入位置", html)
        javascript = (WEB_ROOT / "scratch.js").read_text(encoding="utf-8")
        self.assertIn("paletteQuery", javascript)
        self.assertIn("paletteCollapsed", javascript)
        self.assertIn("palette-category-toggle", javascript)
        self.assertIn("aria-expanded", javascript)
        self.assertIn("Object.entries(info.params || {})", javascript)
        self.assertIn("paletteSelection", javascript)
        self.assertIn("state.paletteSelection = item", javascript)
        self.assertIn("const previewStep = paletteInfo", javascript)
        self.assertIn("node.draggable = true", javascript)
        self.assertIn("可拖动到合法的绿色插入位置", javascript)
        self.assertIn("state.activeInsertionTarget = { parentId, index, context }", javascript)
        self.assertIn('id="formula-builder"', html)
        self.assertIn('id="formula-dialog"', html)
        self.assertIn('id="formula-operation"', html)
        self.assertIn("function formulaCandidates()", javascript)
        self.assertIn("function formulaInsertionReason", javascript)
        self.assertIn("function addFormulaFromDialog", javascript)
        self.assertIn('id="new-formula"', html)
        self.assertIn('id="my-formulas"', html)
        self.assertIn('id="formula-editor-dialog"', html)
        self.assertIn('id="formula-editor-binding-fields"', html)
        self.assertNotIn('id="formula-editor-bindings"', html)
        self.assertIn('id="formula-editor-block"', html)
        self.assertIn('id="formula-editor-save"', html)
        self.assertIn("data-formula-binding", javascript)

    def test_inspector_and_main_columns_are_viewport_bounded(self) -> None:
        css = (WEB_ROOT / "scratch.css").read_text(encoding="utf-8")
        self.assertIn("height: calc(100vh - 58px)", css)
        self.assertIn(".inspector > h2 { position: sticky", css)
        self.assertIn(".module-explanation { position: sticky", css)

    def test_workflow_guide_onboarding_and_result_panel_are_present(self) -> None:
        html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        for marker in (
            'id="workflow-guide"', 'data-workflow="data"', 'data-workflow="result"',
            'id="first-use-guide"', 'id="open-tutorial"', 'id="onboarding-dialog"',
            'id="result-panel"', 'id="result-status"', 'id="result-metrics"',
            'id="result-artifacts"', 'id="result-details"', 'id="output"',
        ):
            self.assertIn(marker, html)

        javascript = (WEB_ROOT / "scratch.js").read_text(encoding="utf-8")
        for marker in (
            "function updateWorkflowGuide()", "function renderRunResult(result)",
            "setResultState('运行中…')", "renderRunResult(result)",
            "$('result-metrics')", "localStorage.setItem('lnl-scratch-first-use-guide-dismissed'",
        ):
            self.assertIn(marker, javascript)

        css = (WEB_ROOT / "scratch.css").read_text(encoding="utf-8")
        self.assertIn(".workflow-guide", css)
        self.assertIn(".first-use-guide", css)
        self.assertIn(".result-panel", css)
        self.assertIn(".workflow-step.active", css)


if __name__ == "__main__":
    unittest.main()
