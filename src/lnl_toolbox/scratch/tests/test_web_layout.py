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
        self.assertIn(".inspector {", css)
        self.assertIn("display: flex; flex-direction: column;", css)
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
        self.assertIn("单击查看说明；用“添加到当前位置/推荐位置”或拖动把手放入绿色插入位", html)
        javascript = (WEB_ROOT / "scratch.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "scratch.css").read_text(encoding="utf-8")
        self.assertIn("paletteQuery", javascript)
        self.assertIn("paletteCollapsed", javascript)
        self.assertIn("palette-category-toggle", javascript)
        self.assertIn("aria-expanded", javascript)
        self.assertIn("Object.entries(info.params || {})", javascript)
        self.assertIn("paletteSelection", javascript)
        self.assertIn("state.paletteSelection = item", javascript)
        self.assertIn("const previewStep = paletteInfo", javascript)
        self.assertNotIn("node.draggable = true", javascript)
        self.assertIn("dragHandle.draggable = true", javascript)
        self.assertIn("palette-target-summary", javascript)
        self.assertIn("palette-target-locate", javascript)
        self.assertIn("palette-status-help", html)
        self.assertIn("✕ 不可添加", html)
        self.assertIn("palette-legend", html)
        self.assertNotIn("颜色 = 类别", html)
        self.assertIn("function collectInsertionTargets", javascript)
        self.assertIn("function recommendedInsertionTarget", javascript)
        self.assertIn("function renderUndoDelete", javascript)
        self.assertIn("className = 'palette-add'", javascript)
        self.assertIn("className = 'drag-handle'", javascript)
        self.assertNotIn("node.onmousedown", javascript)
        self.assertNotIn("node.ondragstart", javascript)
        self.assertIn("event.target.closest('.drag-handle, .palette-add')", javascript)
        self.assertIn("zone.dataset.baseLabel", javascript)
        self.assertIn("✓ 可放置", javascript)
        self.assertIn("state.deletedStep", javascript)
        self.assertIn("state.activeInsertionTarget = target", javascript)
        self.assertIn('id="workspace-position-indicator"', html)
        self.assertIn("function renderPositionIndicator()", javascript)
        self.assertIn("workspace-position-detail", javascript)
        self.assertIn("✕ ${candidate.name || candidate.id}", javascript)
        self.assertIn("✕ ${baseLabel}", javascript)
        self.assertIn('[data-state="invalid"]', css)
        self.assertNotIn('id="formula-builder"', html)
        self.assertNotIn('id="templates"', html)
        self.assertIn('id="formula-dialog"', html)
        self.assertIn('id="formula-operation"', html)
        self.assertIn("function formulaCandidates()", javascript)
        self.assertIn("function formulaInsertionReason", javascript)
        self.assertIn("function addFormulaFromDialog", javascript)
        self.assertNotIn('id="new-formula"', html)
        self.assertNotIn('id="my-formulas"', html)
        self.assertIn('id="formula-editor-dialog"', html)
        self.assertIn('id="formula-editor-binding-fields"', html)
        self.assertIn('id="formula-editor-current-preview"', html)
        self.assertNotIn('id="formula-editor-bindings"', html)
        self.assertIn('id="formula-editor-block"', html)
        self.assertIn('id="formula-editor-save"', html)
        self.assertIn("data-formula-binding", javascript)
        self.assertIn('id="formula-dialog"', html)
        self.assertIn("<h2>快速插入运算</h2>", html)
        self.assertNotIn("<h2>Batch 公式组合器</h2>", html)
        self.assertIn("function renderNestedFormulaPreview", javascript)
        self.assertIn("function formatFormulaCall", javascript)
        self.assertIn("function renderMathFormula", javascript)
        self.assertIn("const MATH_NS", javascript)
        self.assertIn("renderMathFormula(formulaText", javascript)
        self.assertIn("renderFormulaPreview", javascript)
        self.assertIn("formula-chain-step", javascript)
        self.assertIn("formula-palette-formula", javascript)
        self.assertIn("formula-palette-disclosure", javascript)
        self.assertIn("全部公式（${candidates.length}）", javascript)
        self.assertIn("disclosure.append(summary, body)", javascript)
        self.assertIn("已添加步骤：${id}", javascript)
        self.assertIn("while (state.formulaEditor.steps.some", javascript)
        self.assertIn('id="formula-canvas"', html)
        self.assertIn('id="formula-parameter-fields"', html)
        self.assertIn('id="formula-editor-advanced"', html)
        self.assertIn('class="formula-editor-advanced-inline"', html)
        self.assertIn('id="formula-add-parameter"', html)
        self.assertIn("function renderFormulaCanvas", javascript)
        self.assertIn("id=\"formula-equation-builder\"", html)
        self.assertIn("id=\"formula-apply-equation\"", html)
        self.assertIn('value="subtract"', html)
        self.assertIn('value="elementwise_multiply"', html)
        self.assertIn('value="safe_divide"', html)
        self.assertIn("function addFormulaEquation", javascript)
        self.assertIn("openBuiltinFormulaCopy", javascript)
        self.assertIn("使用此模板", javascript)
        self.assertIn("formulas.forEach((item) =>", javascript)
        self.assertIn("builtin-formula-item", javascript)
        self.assertIn("user/${base}", javascript)
        self.assertIn("function renderFormulaParameterFields", javascript)
        self.assertIn("function addFormulaEditorParameter", javascript)
        self.assertIn("renderFormulaParameterFields();", javascript)
        self.assertIn("paletteExpanded", javascript)
        self.assertIn("disclosure.open = state.formulaEditor.paletteExpanded", javascript)
        self.assertIn("disclosure.addEventListener('toggle'", javascript)
        self.assertIn("formula-editor-advanced > summary", css)
        self.assertIn(".formula-canvas", css)
        self.assertIn(".formula-parameter-row", css)
        self.assertIn(".formula-editor-palette { align-self: stretch;", css)
        self.assertIn("overflow-y: auto", css)

    def test_inspector_and_main_columns_are_viewport_bounded(self) -> None:
        css = (WEB_ROOT / "scratch.css").read_text(encoding="utf-8")
        self.assertIn("height: calc(100vh - 64px)", css)
        self.assertIn(".inspector-tabs { position: sticky", css)
        self.assertIn(".module-explanation { position: sticky", css)

    def test_scratch_toolbar_onboarding_and_result_panel_are_present(self) -> None:
        html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        for marker in (
            'id="category-rail"', 'id="new-menu"', 'id="new-single"', 'id="run-full"',
            'id="empty-onboarding"', 'id="onboarding-dialog"',
            'data-inspector-tab="blocks"', 'data-inspector-tab="run"',
            'id="result-panel"', 'id="result-status"', 'id="result-metrics"',
            'id="result-artifacts"', 'id="result-details"', 'id="output"',
            'id="run-guidance"', 'class="preflight-guide"',
            'id="stop-run"', 'id="refresh-run"', 'id="run-progress"',
            'id="run-mode"', 'id="run-mode-help"',
            'id="progress-bar"', 'id="progress-stage"',
            'id="epoch-output"', 'id="epoch-output-list"', 'id="epoch-output-count"',
        ):
            self.assertIn(marker, html)

        javascript = (WEB_ROOT / "scratch.js").read_text(encoding="utf-8")
        for marker in (
            "function setInspectorTab(tab = 'blocks')", "function renderRunResult(result)",
            "setResultState('运行中…')", "renderRunResult(result)",
            "function pollRunJob(jobId)", "async function stopRun()", "renderRunProgress(job)",
            "function renderEpochOutputs(progress = {})", "epoch_outputs",
            "const RUN_PROGRESS_POLL_MS = 4000", "const RUN_STOP_POLL_MS = 500",
            "function makeTemplateCardActivatable(card, item)",
            "async function openTemplateDialog()",
            "$('new-template').onclick = openTemplateDialog",
            "card.setAttribute('role', 'button')", "card.onkeydown",
            "$('result-metrics')", "function detectCompositeRanges(steps)",
            "function applySkeleton(kind)", "await validateCurrentRecipe()",
            "function datasetReadiness()", "function guideForError(error)",
            "function renderGuidance()", "function focusDatasetSource()",
            "function clearErrorPresentation()", "state.runMode === 'full' ? {}",
            "state.runMode = event.target.value === 'full' ? 'full' : 'check'",
            "已修改，旧错误已清除；请重新运行检查。",
            "function chooseSyntheticDataset()", "当前状态：${facts.status || 'unknown'}",
            "USABLE_DATASET_STATUSES", "datasetIsUsable(item)",
            "name: '全部'", "state.paletteCategory === '全部'", "color: '#84cc16'",
            "async function refreshDatasets",
            "const MODEL_OPTIMIZATION_BLOCKS = new Set",
            "function detectModelOptimizationRange(steps, start)",
            "range.modelCount", "range.optimizerCount",
            "const DATA_OVERVIEW_GROUPS = [",
            "label: '数据集'", "label: '数据划分'", "label: '标签噪声'",
            "label: '输入与增强'", "label: '训练批次'",
            "function renderDataConcepts", "function renderDataAdvancedSummary",
            "查看原始 12 步", "返回 5 个数据模块",
        ):
            self.assertIn(marker, javascript)

        css = (WEB_ROOT / "scratch.css").read_text(encoding="utf-8")
        self.assertIn(".category-rail", css)
        self.assertIn('.category-rail button[data-category="全部"]', css)
        self.assertIn(".dataset-refresh", css)
        self.assertIn(".composite-block", css)
        self.assertIn(".result-panel", css)
        self.assertIn(".run-progress", css)
        self.assertIn(".epoch-output-card", css)
        self.assertIn(".run-controls", css)
        self.assertIn(".run-button", css)
        self.assertIn(".math-rendered math", css)
        self.assertIn(".formula-chain-step", css)
        self.assertIn(".formula-editor-current-preview", css)
        self.assertIn(".formula-palette-disclosure", css)
        self.assertIn(".formula-palette-disclosure[open] > summary::before", css)
        self.assertIn(".data-concepts", css)
        self.assertIn(".data-submodule", css)
        self.assertIn(".data-advanced-settings", css)
        self.assertIn(".data-execution-heading", css)

    def test_loop_epochs_are_editable_and_scheduler_is_an_explicit_advanced_action(self) -> None:
        html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        javascript = (WEB_ROOT / "scratch.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "scratch.css").read_text(encoding="utf-8")
        self.assertNotIn('id="add-scheduler"', html)
        self.assertNotIn("function addAdvancedBlock(blockId)", javascript)
        self.assertIn("function appendEpochAdvancedOptions(container, step)", javascript)
        self.assertIn("添加 Create Scheduler", javascript)
        self.assertIn("createSchedulerForEpoch(step)", javascript)
        self.assertIn("const schedulerExists = flatRecipeSteps().some", javascript)
        self.assertIn("addScheduler.disabled = schedulerExists", javascript)
        self.assertNotIn("appendLoopIterationControl(title, steps[range.start])", javascript)
        self.assertIn("function appendLoopIterationControl(container, step)", javascript)
        self.assertIn("step.block !== 'epoch_loop'", javascript)
        self.assertIn("step.params.epochs = value", javascript)
        self.assertIn("className = 'loop-iterations'", javascript)
        self.assertIn(".loop-iterations-input", css)
        self.assertIn(".epoch-advanced", css)

    def test_data_overview_modules_are_vertical_and_editable(self) -> None:
        javascript = (WEB_ROOT / "scratch.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "scratch.css").read_text(encoding="utf-8")
        self.assertIn("function selectDataOverviewGroup(group, steps)", javascript)
        self.assertIn("state.dataOverviewSelection = {groupId: group.id", javascript)
        self.assertIn("item.setAttribute('role', 'button')", javascript)
        self.assertIn("item.onclick = () => selectDataOverviewGroup(group, steps)", javascript)
        self.assertIn("function renderDataOverviewInspector(selection, target, explanationTarget)", javascript)
        self.assertIn("renderParamControl(name, schema, step)", javascript)
        self.assertIn("五个模块按执行顺序排列", javascript)
        self.assertIn("dataBlockMode: 'concepts'", javascript)
        self.assertIn("function dataPaletteItems()", javascript)
        self.assertIn("function addDataConceptAtTarget(group, target)", javascript)
        self.assertIn("application/x-lnl-data-group", javascript)
        self.assertIn("state.dataBlockMode = 'canonical'", javascript)
        self.assertIn("当前数据视图：5 个合并数据积木", javascript)
        self.assertIn("canonical data view is intentionally lossless", javascript)
        self.assertIn("isDataComposite && showingDataSteps", javascript)
        self.assertIn(".data-concepts { display: flex; flex-direction: column;", css)
        self.assertIn(".data-submodule.selected", css)
        self.assertIn(".data-submodule-edit", css)


if __name__ == "__main__":
    unittest.main()
