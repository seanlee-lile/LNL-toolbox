from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import unittest

from web.command_console import STATIC_ASSETS


ROOT = Path(__file__).resolve().parents[1]


class QuickStartAssetsTests(unittest.TestCase):
    def test_assets_are_served_from_explicit_allowlist(self) -> None:
        self.assertEqual(set(STATIC_ASSETS), {"/assets/quick_start.js", "/assets/quick_start.css", "/assets/run_output.js"})
        self.assertTrue(all(path.is_file() for path, _content_type in STATIC_ASSETS.values()))

    def test_parent_traversal_is_not_an_asset(self) -> None:
        self.assertNotIn("/assets/../index.html", STATIC_ASSETS)

    def test_index_loads_external_quick_start_resources(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn('/assets/quick_start.css', html)
        self.assertIn('/assets/quick_start.js', html)
        self.assertIn('/assets/run_output.js', html)
        self.assertIn('window.quickStartController', html)
        self.assertIn('id="training-output"', html)
        self.assertIn('id="context-config-label"', html)
        self.assertIn('paperSelected: function (paper)', html)
        self.assertIn('quickStart ? "当前论文" : "当前实验"', html)
        self.assertIn('state.quickStartPaper.acronym, state.quickStartPaper.title', html)

    def test_existing_modules_remain(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        for module in ('quickstart', 'beginner', 'yaml', 'data', 'sweep', 'results', 'papers', 'advanced'):
            self.assertIn('id: "' + module + '"', html)

    def test_registered_dataset_and_training_guard_are_wired(self) -> None:
        script = (ROOT / "web" / "assets" / "quick_start.js").read_text(encoding="utf-8")
        self.assertIn("esc(item.location)", script)
        self.assertIn('payload.status === "already_registered"', script)
        self.assertIn("qs-required-input", script)
        self.assertIn("window.confirm", script)
        self.assertIn("loadingMarkup", script)
        self.assertIn("请勿重复点击", script)
        self.assertIn("首次登记时检查训练/测试样本", script)
        self.assertNotIn("不重新加载图像", script)
        self.assertIn("finally(endLoading)", script)
        self.assertIn("item.key === state.noiseSelection.key", script)
        self.assertIn('qs-rate")?.addEventListener("change", updateNoise)', script)
        self.assertIn('qs-seed")?.addEventListener("change", updateNoise)', script)
        self.assertIn('disabled aria-disabled="true"', script)
        self.assertIn('["ready", "needs_input"].includes(item.status)', script)
        self.assertIn('renderPlanParameters()', script)
        self.assertIn('field.display_group === "default"', script)
        self.assertIn('field.display_group === "advanced"', script)
        self.assertIn('field.display_group === "restricted"', script)
        self.assertIn('function renderScheduleControl(group, allFields)', script)
        self.assertIn('data-qs-schedule-toggle', script)
        self.assertIn('data-qs-schedule-milestone', script)
        self.assertIn('data-qs-schedule-rate', script)
        self.assertIn('手动修改表格后改用各段明确数值', script)
        self.assertIn('禁止在 Web 修改的专属参数', script)
        self.assertNotIn('summary>公共实验设置', script)
        self.assertNotIn('summary>模型 / 组件选择', script)
        self.assertNotIn('summary>外部资源与运行前置条件', script)
        self.assertIn('parameterPatches()', script)
        self.assertNotIn('qs-advanced-actions', script)
        self.assertIn("训练轮次", (ROOT / "src" / "lnl_toolbox" / "quickstart" / "service.py").read_text(encoding="utf-8"))

    def test_step_schedule_preview_shows_formal_l2rw_rates(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs');
const vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =', 'window.__probe = {state, renderScheduleControl}; window.quickStartController =');
const context = {window: {}};
vm.runInNewContext(source, context);
const fields = [
  {path:'scheduler.name', value:'none', editable:true},
  {path:'scheduler.step_milestones', value:[40000, 60000], editable:true},
  {path:'scheduler.gamma', value:0.1, editable:true},
  {path:'optimizer.lr', value:0.1, editable:true},
  {path:'trainer.max_steps', value:80000, editable:true}
];
const probe = context.window.__probe;
probe.state.segmentedPaths['optimizer.lr'] = true;
const html = probe.renderScheduleControl({prefix:'scheduler', fields:fields.slice(0, 3)}, fields);
if (!html.includes('data-qs-schedule-toggle="scheduler" checked') ||
    !html.includes('40000') || !html.includes('60000') ||
    !html.includes('0.01') || !html.includes('0.001') ||
    !html.includes('data-qs-schedule-rate')) process.exit(1);
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_quick_start_shows_only_effective_training_budget(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs');
const vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {state, renderPlanParameters}; window.quickStartController =');
const context = {window:{}};
vm.runInNewContext(source, context);
const {state, renderPlanParameters} = context.window.__probe;
state.planSchema = {method:'l2rw', fields:[
  {path:'trainer.epochs', value:180, label:'训练轮数', editable:true, visible:true, display_group:'default', kind:'number'},
  {path:'trainer.max_steps', value:80000, label:'总更新步数', editable:true, visible:true, display_group:'default', kind:'number'}
]};
let html = renderPlanParameters();
if (!html.includes('data-qs-param="trainer.max_steps"') ||
    html.includes('data-qs-param="trainer.epochs"')) process.exit(1);
state.parameterDraft['trainer.max_steps'] = '0';
html = renderPlanParameters();
if (!html.includes('data-qs-param="trainer.epochs"') ||
    html.includes('data-qs-param="trainer.max_steps"')) process.exit(2);
state.planSchema = {method:'gce', fields:[
  {path:'trainer.epochs', value:120, label:'训练轮数', editable:true, visible:true, display_group:'default', kind:'number'}
]};
html = renderPlanParameters();
if (!html.includes('data-qs-param="trainer.epochs"')) process.exit(3);
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_selected_paper_updates_quick_start_context(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs');
const vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {state}; window.quickStartController =');
const context = {window:{}, document:{getElementById:() => null}};
vm.runInNewContext(source, context);
const {state} = context.window.__probe;
state.registered = [{name:'CIFAR-10'}];
state.methods = [{paper_id:'l2rw', acronym:'L2RW', title:'Learning to Reweight Examples'}];
state.selectedPaperId = 'l2rw';
let selected = null;
const panel = {innerHTML:'', querySelectorAll:() => [], querySelector:() => null};
context.window.quickStartController.mount(panel, {compact:true, paperSelected:paper => { selected = paper; }});
if (selected?.acronym !== 'L2RW') process.exit(1);
state.selectedPaperId = '';
context.window.quickStartController.render();
if (selected !== null) process.exit(2);
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_parameter_cards_use_field_names_without_category_prefixes(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs');
const vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {renderParameterField}; window.quickStartController =');
const context = {window:{}};
vm.runInNewContext(source, context);
const {renderParameterField} = context.window.__probe;
const augment = renderParameterField({path:'data.augment', label:'训练数据增强',
  presentation:'common', presentation_label:'公共实验设置', value:true, kind:'boolean', editable:true, note:'是否应用随机增强。'});
const model = renderParameterField({path:'model.name', label:'模型架构',
  presentation:'selection', presentation_label:'模型选择', value:'resnet', kind:'string', editable:true, note:''});
if (!augment.includes('<span>训练数据增强</span>') || augment.includes('公共实验设置 ·') ||
    !model.includes('<span>模型架构</span>') || model.includes('模型选择 ·') ||
    model.includes('<small>')) process.exit(1);
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_run_plan_leads_with_training_schedule_without_duplicate_status(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs');
const vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =', 'window.__probe = {state, renderPlan}; window.quickStartController =');
const context = {window:{}};
vm.runInNewContext(source, context);
const {state, renderPlan} = context.window.__probe;
state.plan = {status:'ready', config_kind:'paper_reproduction', summary:'条件完全匹配正式论文复现配置。',
  details:['训练轮次：trainer.epochs=200', '输出目录：temporary']};
state.planSchema = {fields:[
  {path:'trainer.epochs', value:200, editable:true, visible:true, display_group:'default', kind:'number'},
  {path:'scheduler.name', value:'cosine', editable:true, visible:true, display_group:'default', kind:'string'},
  {path:'optimizer.lr', value:0.001, editable:true, visible:true, display_group:'default', kind:'number'}
]};
let html = renderPlan();
if (!html.includes('模型训练计划') || html.indexOf('模型训练计划') > html.indexOf('可修改的训练参数') ||
    html.includes('已修改的运行配置') || html.includes('条件完全匹配正式论文复现配置') ||
    html.includes('已匹配已验收数据资料') || html.includes('训练轮次：trainer.epochs=200') ||
    !html.includes('data-qs-param="trainer.epochs"')) process.exit(1);
state.plan = {status:'needs_input', summary:'需要提供方法输入'};
html = renderPlan();
if (!html.includes('运行前还需处理') || !html.includes('需要提供方法输入')) process.exit(2);
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_only_supported_learning_rate_moves_into_segment_table(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs');
const vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {state, renderPlanParameters, parameterPatches}; window.quickStartController =');
const context = {window:{}};
vm.runInNewContext(source, context);
const {state, renderPlanParameters, parameterPatches} = context.window.__probe;
state.planSchema = {fields:[
  {path:'scheduler.name', value:'multistep', editable:true, visible:true, display_group:'default', kind:'string'},
  {path:'scheduler.milestones', value:[40,80], editable:true, visible:true, display_group:'default', kind:'list'},
  {path:'scheduler.gamma', value:0.1, editable:true, visible:true, display_group:'default', kind:'number'},
  {path:'scheduler.lr_values', value:null, editable:true, visible:true, display_group:'advanced', kind:'list', nullable:true},
  {path:'optimizer.lr', value:0.01, editable:true, visible:true, display_group:'default', kind:'number'},
  {path:'optimizer.momentum', value:0.9, editable:true, visible:true, display_group:'default', kind:'number'},
  {path:'optimizer.weight_decay', value:0.0001, editable:true, visible:true, display_group:'default', kind:'number'},
  {path:'model.name', value:'cifar_cnn8', editable:true, visible:true, display_group:'default', kind:'string'},
  {path:'trainer.epochs', value:120, editable:true, visible:true, display_group:'default', kind:'number'}
]};
let html = renderPlanParameters();
if (!html.includes('qs-segment-table') || !html.includes('data-qs-segment-param="optimizer.lr"') ||
    !html.includes('data-qs-segment-param="optimizer.lr" checked') ||
    !html.includes('data-qs-schedule-rate="scheduler"') ||
    (html.match(/data-qs-param="optimizer.lr"/g) || []).length !== 1 ||
    html.includes('data-qs-segment-param="optimizer.momentum"') ||
    html.includes('data-qs-segment-param="optimizer.weight_decay"') ||
    html.includes('data-qs-segment-param="model.name"') ||
    !html.includes('momentum、weight decay 等仍按整场训练的固定值执行')) process.exit(1);
state.parameterDraft['scheduler.lr_values'] = JSON.stringify([0.003, 0.0007]);
html = renderPlanParameters();
const patch = parameterPatches().find(item => item.path === 'scheduler.lr_values');
if (!html.includes('value="0.003"') || !html.includes('value="0.0007"') ||
    !html.includes('不再按统一倍率推算') || !patch ||
    JSON.stringify(patch.value) !== '[0.003,0.0007]') process.exit(3);
delete state.parameterDraft['scheduler.lr_values'];
state.segmentedPaths['optimizer.lr'] = false;
html = renderPlanParameters();
if (!html.includes('学习率仍在下方编辑') || html.includes('data-qs-schedule-rate="scheduler"') ||
    (html.match(/data-qs-param="optimizer.lr"/g) || []).length !== 1 ||
    !html.includes('data-qs-param="optimizer.momentum"') ||
    parameterPatches().length !== 0) process.exit(2);
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_noise_question_follows_accepted_dataset_facts(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs');
const vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =', 'window.__probe = {state, renderNoise, renderDataset}; window.quickStartController =');
const context = {window:{}};
vm.runInNewContext(source, context);
const {state, renderNoise, renderDataset} = context.window.__probe;
state.dataset = {display_name:'CIFAR-10', train_size:100, test_size:20, num_classes:10};
state.noise = {dataset_state:'clean', noise_status:'clean', status_source:'inspected',
  options:[{key:'clean',label:'保持干净'},{key:'symmetric',label:'对称噪声',requires_rate:true}]};
let html = renderNoise();
if (!html.includes('是否添加人工标签噪声') || html.includes('id="qs-rate"') ||
    !renderDataset().includes('带噪情况：</b>已确认干净')) process.exit(1);
state.noiseSelection = {kind:'synthetic',key:'symmetric',rate:0.2,seed:1};
html = renderNoise();
if (!html.includes('id="qs-rate"') || !html.includes('加噪类型')) process.exit(2);
state.noise = {dataset_state:'native', noise_status:'noisy', noise_rate:{status:'unknown',value:null}};
html = renderNoise();
if (!html.includes('能否提供原始标签的噪声率') || html.includes('id="qs-add-noise"') ||
    !renderDataset().includes('带噪情况：</b>已确认含噪')) process.exit(3);
state.noise = {dataset_state:'unknown', noise_status:'unknown', requires_confirmation:true};
state.labelsConfirmed = false;
html = renderNoise();
if (!html.includes('干净性尚不能确认') || html.includes('id="qs-add-noise"')) process.exit(4);
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
