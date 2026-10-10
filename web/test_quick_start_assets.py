from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import unittest

from web.command_console import STATIC_ASSETS, _config_schema, _parameter_registry


ROOT = Path(__file__).resolve().parents[1]


class QuickStartAssetsTests(unittest.TestCase):
    def test_dataset_path_next_identifies_and_invalidates_previous_selection(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs=require('fs'), vm=require('vm'), assert=require('assert');
let source=fs.readFileSync('web/assets/quick_start.js','utf8');
source=source.replace('window.quickStartController =',
  'window.__probe={state,changeDatasetPath,advanceQuickStart,renderDataset,renderQuickStartCarousel,pickPath}; window.quickStartController =');
source=source.replace('function render() {','function render() { return;');
let mode='detected', calls=[];
const dataset={alias:'new-data',display_name:'new-data',path:'F:/new',train_size:3000,test_size:0,num_classes:10};
const ctx={window:{setTimeout(){return 1;},clearTimeout(){}},document:{getElementById(){return null;}},
  fetch:async(url, options)=>{
    calls.push([url,options ? JSON.parse(options.body):null]);
    if(url.endsWith('/probe')) return {ok:mode!=='error',json:async()=>mode==='error'?{error:'无法读取目录'}:{status:mode,candidates:[{adapter:'cifar10'}]}};
    return {ok:true,json:async()=>url.endsWith('/register')?{kind:'dataset',dataset}:
      url==='/api/datasets'?{datasets:[]}:
      url==='/api/picker'?{cancelled:false,path:'F:/picked'}:
      {requires_confirmation:true,dataset_state:'unknown',options:[]}};
  }};
vm.runInNewContext(source,ctx);
const p=ctx.window.__probe,s=p.state;
(async()=>{
  s.path='F:/old';s.dataset={alias:'old'};s.noise={};s.plan={};s.planSchema={};
  s.parameterDraft={'optimizer.lr':1};s.review={phase:'ready'};s.selectedPaperId='old-paper';
  p.changeDatasetPath('F:/new');
  assert.strictEqual(s.dataset,null);assert.strictEqual(s.plan,null);assert.strictEqual(s.review,null);
  assert.strictEqual(s.selectedPaperId,'');assert.strictEqual(Object.keys(s.parameterDraft).length,0);
  const html=p.renderDataset();
  assert(!html.includes('qs-probe'));assert(!html.includes('qs-reset'));
  assert(!p.renderQuickStartCarousel().match(/id="qs-next"[^>]*disabled/));
  await p.advanceQuickStart();
  assert.strictEqual(s.dataset.alias,'new-data');assert.strictEqual(s.currentStep,1);
  assert.strictEqual(calls.filter(x=>x[0].endsWith('/probe')).length,1);
  assert.strictEqual(calls.find(x=>x[0].endsWith('/register'))[1].path,'F:/new');
  s.registered=[{name:'new-data',location:'F:/new',adapter:'cifar10'}];
  assert(p.renderDataset().includes('使用过的数据集'));
  await p.pickPath();assert.strictEqual(s.path,'F:/picked');assert.strictEqual(s.dataset,null);
  mode='ambiguous';calls=[];await p.advanceQuickStart();
  assert.strictEqual(s.currentStep,0);assert.strictEqual(s.dataset,null);
  assert(!calls.some(x=>x[0].endsWith('/register')));
  mode='error';await p.advanceQuickStart();
  assert.strictEqual(s.currentStep,0);assert(s.error.includes('无法读取目录'));
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_formal_paper_renders_schema_choices_and_nested_list_controls(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        schemas = [_config_schema(recipe) for recipe in _parameter_registry()["formal_recipe_bindings"].values()]
        script = r"""
const fs=require('fs'), vm=require('vm');
let source=fs.readFileSync('web/assets/quick_start.js','utf8');
source=source.replace('window.quickStartController =','window.__probe={state,renderParameterField,renderPlanParameters}; window.quickStartController =');
const ctx={window:{}};
vm.runInNewContext(source,ctx);
const p=ctx.window.__probe;
let fields=0, selects=0;
for(const schema of JSON.parse(fs.readFileSync(0,'utf8'))) {
  p.state.planSchema=schema;
  p.state.dataset={alias:'selected-data',train_size:3000};
  p.state.parameterDraft={}; p.state.segmentedPaths={};
  for(const field of schema.fields.filter(f=>f.visible!==false)) {
    const html=p.renderParameterField(field,false);
    if(!html.includes(field.label) && field.path!=='data.name') throw Error(schema.method+': missing label '+field.path);
    if(field.choices && field.path!=='data.name') {
      if(!html.includes('<select')) throw Error(schema.method+': not dropdown '+field.path);
      for(const choice of field.choices) if(!html.includes(choice.label)) throw Error('missing readable option '+field.path);
      selects++;
    }
    if(field.rows) {
      if(!html.includes('textarea hidden') || !html.includes('data-parameter-list')) throw Error('raw nested JSON control '+field.path);
      for(const row of field.rows) for(const child of row) if(!html.includes(child.label)) throw Error('missing nested label');
    }
    if(html.includes('通用配置参数') || html.includes('可修改的高级实验/实现选项')) throw Error('vague helper');
    fields++;
  }
  const panel=p.renderPlanParameters();
  if(panel.includes('>Normalize<') || panel.includes('>Scale<') || panel.includes('>Center<')) throw Error('old preprocessing names');
}
console.log(JSON.stringify({papers:26,fields,selects}));
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT,
                                input=json.dumps(schemas, ensure_ascii=False), capture_output=True,
                                text=True, encoding="utf-8", check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        stats = json.loads(result.stdout)
        self.assertEqual(stats["papers"], 26)
        self.assertGreater(stats["fields"], 700)
        self.assertGreater(stats["selects"], 150)

    def test_fifth_step_reviews_cli_output_before_final_confirmation(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs'), vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {state, reviewPlan, renderReview, renderPlan, quickStartSteps, renderParameterField, executePlan, updateReview, setPanel(value){panel=value;}, setContext(value){context=value;}}; window.quickStartController =');
source = source.replace('function render() {', 'function render() { window.__renders=(window.__renders||0)+1;');
let failValidation = false, failDry = false, calls = [], executed = 0, submitted = '', savedPatches, polls = 0;
const context = {window:{setTimeout(fn){fn();}}, fetch:async(url, options) => {
  const body = options ? JSON.parse(options.body) : null;
  calls.push([url,body]);
  if (url.endsWith('review-config')) { savedPatches = body.patches; return {ok:true,json:async()=>({path:'review.yaml',content:'checked snapshot',run_id:'planned',output_dir:'artifacts/web-runs/planned',config_path:'artifacts/web-runs/planned/config.yaml'})}; }
  if (url.endsWith('training-config')) {
    if (body.content!=='checked snapshot' || body.run_id!=='planned') throw Error('training snapshot changed');
    return {ok:true,json:async()=>({path:'artifacts/web-runs/planned/config.yaml',output_dir:'artifacts/web-runs/planned',config_path:'artifacts/web-runs/planned/config.yaml'})};
  }
  if (url==='/api/run') {
    polls=0;
    if (!body.command.includes('--dry-run') || !body.command.includes('--review')) throw Error('not a combined review');
  } else polls++;
  const failed=failValidation||failDry;
  const done=failed||polls>=3;
  const phases=failValidation ? [] : ['Quick Start phase: validating', 'Quick Start phase: rehearsing'];
  return {ok:true,json:async()=>({id:'job',running:!done,output_complete:done,returncode:done ? failed ? 1:0 : null,
    lines:failed ? phases.concat(['错误: invalid config for optimizer.lr, expected finite number > 0',
                   'invalid config for trainer.epochs, expected positive integer']) : polls>=2 ? phases.concat(done?['CLI passed']:[]) : []})};
}};
vm.runInNewContext(source, context);
const p = context.window.__probe, s=p.state;
p.setContext({setRequest(request,command){submitted=command;},execute(){executed++;}});
s.dataset={alias:'test'}; s.noise={dataset_state:'clean'}; s.methods=[{paper_id:'p'}];
s.selectedPaperId='p'; s.plan={plan_id:'plan',paper_id:'p',status:'ready',method:'GCE',
  recipe_id:'recipe', command:'lnl run --recipe recipe',dry_run_command:'lnl run --recipe recipe --dry-run'};
s.planSchema={fields:[{path:'optimizer.lr',value:0.01,kind:'number',editable:true,visible:true}]};
s.currentStep=4;
(async()=>{
  if (p.quickStartSteps().length!==5) throw Error('expected five steps');
  const optimizer=p.renderParameterField({path:'t_revision.stage1.optimizer.name',kind:'string',value:'sgd',editable:true,choices:[{value:'sgd',label:'随机梯度下降'},{value:'adam',label:'自适应矩估计'}]});
  if (!optimizer.includes('<select') || !optimizer.includes('value="adam"')) throw Error('optimizer is not a select');
  const fourth=p.renderPlan();
  if (fourth.includes('id="qs-dry"') || fourth.includes('id="qs-run"')) throw Error('fourth step still executes commands');
  failValidation=true; s.parameterDraft={'optimizer.lr':'非法 invalid'};
  await p.reviewPlan();
  if (s.review.phase!=='error' || !p.renderReview().includes('invalid config for trainer.epochs') ||
      calls.filter(x=>x[0]==='/api/run').length!==1 || savedPatches[0].value!=='非法 invalid')
    throw Error('validation failed to return all CLI errors or dry-run ran after failure');
  failValidation=false; s.parameterDraft={}; calls=[];
  const before=context.window.__renders;
  await p.reviewPlan();
  if (s.review.phase!=='ready' || executed!==0 || calls.filter(x=>x[0]==='/api/run').length!==1)
    throw Error('validation/dry-run gate incorrect');
  if (!p.renderReview().includes('artifacts/web-runs/planned/config.yaml') ||
      !calls.find(x=>x[0]==='/api/run')[1].command.includes('--output-dir "artifacts/web-runs/planned"'))
    throw Error('planned training locations missing');
  if (context.window.__renders-before!==1) throw Error('polling rebuilds Quick Start');
  if (calls.some(x=>x[0].endsWith('training-config'))) throw Error('saved before confirmation');
  await p.executePlan();
  if (executed!==1 || submitted!=='lnl run --config "artifacts/web-runs/planned/config.yaml" --output-dir "artifacts/web-runs/planned"') throw Error('did not use reviewed snapshot');
  await p.reviewPlan(); s.parameterDraft={'optimizer.lr':'0.2'}; await p.executePlan();
  if (executed!==1) throw Error('stale review permitted training');
  failDry=true; s.parameterDraft={}; await p.reviewPlan();
  if (s.review.phase!=='error') throw Error('dry-run failure not displayed');
  s.review.phase='validating';
  if (!p.renderReview().includes('qs-spinner')) throw Error('missing validation spinner');
  s.review.phase='rehearsing';
  if (!p.renderReview().includes('qs-spinner')) throw Error('missing dry-run spinner');
  const nodes={};
  const center={querySelector(selector){return nodes[selector]||(nodes[selector]={textContent:'',hidden:false});}};
  p.setPanel({querySelector(){return center;}});
  p.updateReview();
  const spinner=nodes['.qs-spinner'];
  nodes['[data-review-output]'].open=true;
  for(let i=0;i<5;i++) p.updateReview();
  if (nodes['.qs-spinner']!==spinner || !nodes['[data-review-output]'].open || spinner.hidden)
    throw Error('spinner or expanded output replaced');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stage_parameter_cards_preserve_backend_names_and_explanations(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs'), vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {renderParameterField}; window.quickStartController =');
const context = {window:{}};
vm.runInNewContext(source, context);
const render = context.window.__probe.renderParameterField;
for (const path of ['t_revision.stage1.epochs', 't_revision.stage1.optimizer.momentum',
                    'upm.stage1.epochs', 'dld.diffusion.optimizer.direction.lr']) {
  const label = path + '（具体训练阶段）', note = '具体模型及参数作用';
  const html = render({path, label, note, value:1, kind:'number', editable:true});
  if (!html.includes(label) || !html.includes(note))
    throw Error('Parameter text overridden: ' + path);
}
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_navigation_resets_scroll_on_steps_modules_and_page_refresh(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
let source = fs.readFileSync('web/assets/quick_start.js','utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {state, render, attach(target){panel=target;context={compact:true}}}; window.quickStartController =');
let scrolls = 0;
const handlers = {}, dots = [0,1].map(index=>({dataset:{qsStep:String(index)},disabled:false,
  addEventListener(event,callback){this.click=callback}}));
const context = {window:{scrollConsoleToTop(){scrolls++}},
  document:{getElementById(id){return ['qs-prev','qs-next'].includes(id) ?
    {addEventListener(event,callback){handlers[id]=callback}} : null}}};
const panel = {innerHTML:'',querySelector(){return null},querySelectorAll(selector){return selector==='.qs-carousel-dot'?dots:[]}};
vm.runInNewContext(source,context);
const {state,render,attach} = context.window.__probe;
attach(panel);
state.dataset={alias:'mini',display_name:'mini'};
state.noise={requires_confirmation:false,options:[],dataset_state:'clean'};
render(); assert.strictEqual(scrolls,1);
handlers['qs-next'](); assert.strictEqual(state.currentStep,1); assert.strictEqual(scrolls,2);
handlers['qs-prev'](); assert.strictEqual(state.currentStep,0); assert.strictEqual(scrolls,3);
dots[1].click.call(dots[1]); assert.strictEqual(state.currentStep,1); assert.strictEqual(scrolls,4);
render(); assert.strictEqual(scrolls,4); // Background redraw keeps the reading position.
state.currentStep=0; render(); assert.strictEqual(scrolls,5); // Automatic step change.

const page = fs.readFileSync('web/index.html','utf8');
const initialization = page.slice(page.indexOf('window.scrollConsoleToTop ='),page.indexOf('const recipeMode ='));
let position = 700, animation, pageshow;
const node = {classList:{toggle(){},remove(){},contains(){return true}},addEventListener(){}};
const pageContext = {window:{history:{scrollRestoration:'auto'},scrollTo(options){position=options.top},
  requestAnimationFrame(callback){animation=callback},addEventListener(event,callback){if(event==='pageshow')pageshow=callback}},
  state:{module:'quickstart'},yamlEditorPanel:node,runButton:node,
  document:{getElementById(){return node}},parameterPanel:{querySelectorAll(){return []}},
  workspaceMeta(){return {}},
};
for (const name of ['renderModuleButtons','renderWorkspaceTabs','renderQuickStart','renderYaml','renderData',
  'renderSweepV2','renderRunWorkspace','renderResultsV2','renderPapers','installPathPickers',
  'updateModulePreview','updateYamlEditorButton','renderConsoleContext']) pageContext[name]=()=>{};
vm.runInNewContext(initialization,pageContext);
assert.strictEqual(pageContext.window.history.scrollRestoration,'manual');
pageshow(); assert.strictEqual(position,0);
position=500; animation(); assert.strictEqual(position,0);
const switchSource=page.slice(page.indexOf('function switchModule(module)'),page.indexOf('function toggleYamlField'));
vm.runInNewContext(switchSource,pageContext);
position=900; pageContext.switchModule('data'); assert.strictEqual(position,0);
position=400; pageContext.switchModule('papers'); assert.strictEqual(position,0);
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_multistage_timeline_precedes_parameter_editors(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs'), vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {state, renderPlanParameters}; window.quickStartController =');
const context = {window:{}};
vm.runInNewContext(source, context);
const {state, renderPlanParameters} = context.window.__probe;
const fields = [
  ['t_revision.stage1.epochs',20],
  ['t_revision.classifier_initialization.epochs',200],
  ['t_revision.revision.epochs',200],
  ['t_revision.stage1.scheduler.name','none'],
  ['t_revision.classifier_initialization.scheduler.name','multistep'],
  ['t_revision.classifier_initialization.scheduler.milestones',[40,80]],
  ['t_revision.revision.scheduler.name','none'],
].map(([path,value]) => ({path,value,editable:true,visible:true,
  display_group:'default',kind:typeof value === 'number' ? 'number' :
    Array.isArray(value) ? 'list' : 'string'}));
state.planSchema = {method:'t_revision',fields};
let html = renderPlanParameters();
if (!html.includes('训练时间轴') || !html.includes('估计噪声矩阵') ||
    html.indexOf('训练时间轴') > html.indexOf('模型训练计划') ||
    !html.includes('data-qs-schedule-expand="t_revision.stage1.scheduler" aria-expanded="false"') ||
    !html.includes('data-qs-schedule-toggle="t_revision.stage1.scheduler"') ||
    !html.includes('动态学习率') || !html.includes('20 轮'))
  throw Error('T-Revision timeline missing or misplaced');
for (const title of ['初始分类器', '校正分类器', '联合修正']) {
  if (!html.includes('<strong>' + title + '</strong>'))
    throw Error('Missing short stage name: ' + title);
}
if (html.includes('Stage 1') || html.includes('Stage 2A') || html.includes('Stage 2B') ||
    html.includes('<strong>分类器与转移矩阵联合修正</strong>'))
  throw Error('Long or numbered stage title returned');
state.expandedSchedules['t_revision.stage1.scheduler'] = true;
html = renderPlanParameters();
if (!html.includes('data-qs-schedule-expand="t_revision.stage1.scheduler" aria-expanded="true"') ||
    !html.includes('class="qs-schedule-body">'))
  throw Error('expanded schedule editor missing');
state.parameterDraft['t_revision.stage1.epochs'] = '30';
html = renderPlanParameters();
if (!html.includes('30 轮')) throw Error('timeline did not follow edited budget');
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_selected_dataset_restores_canonical_path_and_selection_on_redraw(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {state, registerPath, renderDataset}; window.quickStartController =');
const dataset = {alias:'mini-cifar', display_name:'mini-cifar', adapter:'cifar10',
  path:'F:\\datasets\\mini-cifar', train_size:2700, test_size:300, num_classes:10};
const records = [{name:'other-cifar', adapter:'cifar10', location:'F:\\datasets\\other'},
  {name:dataset.alias, adapter:dataset.adapter, location:dataset.path}];
const context = {window:{setTimeout(){return 1},clearTimeout(){}},
  fetch:async url => ({ok:true,json:async()=>
    url === '/api/quick-start/register' ? {kind:'dataset',dataset} :
    url === '/api/datasets' ? {datasets:records} : {requires_confirmation:true,dataset_state:'unknown'}})};
vm.runInNewContext(source,context);
const {state,registerPath,renderDataset} = context.window.__probe;
(async()=>{
  state.path='./mini-cifar';
  await registerPath(state.path,null,false);
  assert.strictEqual(state.path,dataset.path);
  for (const step of [0,1,0]) {
    state.currentStep=step;
    const html=renderDataset();
    assert(html.includes('id="qs-path" value="'+dataset.path+'"'));
    const select=html.match(/<select id="qs-registered">([\s\S]*?)<\/select>/)[1];
    assert(select.includes('<option value="'+dataset.path+'" selected>mini-cifar'));
    assert.strictEqual((select.match(/ selected/g)||[]).length,1);
    assert(html.includes('✓ mini-cifar'));
  }
  state.dataset=null;
  assert(!renderDataset().match(/<option[^>]* selected/));
})().catch(error=>{console.error(error);process.exitCode=1});
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_updating_noise_keeps_noise_step_and_selected_paper(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs'), vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {state, updateNoise}; window.quickStartController =');
const fields = {'qs-add-noise':{checked:true}, 'qs-noise':{value:'symmetric'},
  'qs-rate':{value:'0.2'}, 'qs-seed':{value:'1'}};
const calls = [];
const context = {
  window:{setTimeout:()=>1, clearTimeout:()=>{}},
  document:{getElementById:id=>fields[id] || null},
  fetch:async url => { calls.push(url); return {ok:true, json:async () =>
    url.includes('/methods') ? {methods:[{paper_id:'dividemix',status:'ready'}]} :
    {paper_id:'dividemix',plan_id:'new-plan',status:'ready'}}; }
};
vm.runInNewContext(source, context);
const {state, updateNoise} = context.window.__probe;
state.dataset = {alias:'cifar10-local'};
state.noise = {dataset_state:'clean', options:[{key:'clean'},
  {key:'symmetric',requires_rate:true}]};
state.selectedPaperId = 'dividemix';
state.plan = {paper_id:'dividemix',plan_id:'old-plan',status:'needs_input'};
state.currentStep = 1;
updateNoise({target:{id:'qs-noise'}});
setTimeout(() => {
  if (state.selectedPaperId !== 'dividemix' || state.plan?.plan_id !== 'new-plan' ||
      state.currentStep !== 1 || !calls.some(url => url.includes('/methods')) ||
      !calls.some(url => url.includes('/plan'))) process.exitCode = 1;
}, 20);
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_noise_inputs_use_guidance_not_internal_paths(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs');
const vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {state, renderPlan}; window.quickStartController =');
const context = {window:{}};
vm.runInNewContext(source, context);
const {state, renderPlan} = context.window.__probe;
state.methods = [{paper_id:'dividemix', required_input_paths:[
  ['requires_noisy_training_labels', [['noise','name']]],
  ['noise_rate_prior', [['noise','rate']]]]}];
state.selectedPaperId = 'dividemix';
state.plan = {status:'needs_input', summary:'raw summary',
  details:['method noise-rate prior is required independently of the dataset true rate'],
  required_user_inputs:['requires_noisy_training_labels','noise_rate_prior']};
let html = renderPlan();
if (!html.includes('返回第 2 步设置标签噪声') || html.includes('方法噪声率先验') ||
    html.includes('noise.name') || html.includes('method noise-rate prior'))
  throw Error('clean DivideMix must guide to the noise step without duplicate prior');
state.selectedPaperId = 'cal';
state.methods = [{paper_id:'cal', required_input_paths:[
  ['config:requires_external_noise_labels', [['noise','path'],['noise','clean_key'],['noise','noisy_key']]]]}];
state.plan = {status:'needs_input', summary:'raw summary',
  details:['CAL requires aligned external clean/noisy label vectors'],
  required_user_inputs:['config:requires_external_noise_labels']};
html = renderPlan();
if (!html.includes('查看论文外部数据') || html.includes('noise.path') ||
    html.includes('noise.clean_key') || html.includes('CAL requires aligned'))
  throw Error('CAL must explain the external-label requirement');
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_formal_papers_keep_one_editable_learning_rate_when_schedule_is_disabled(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        from lnl_toolbox.catalog import default_paper_config, load_papers
        from web import command_console

        schemas = []
        for paper in load_papers(command_console.ROOT):
            recipe = default_paper_config(paper, root=command_console.ROOT)[0]
            schemas.append(command_console._config_schema(recipe.recipe_id))
        self.assertEqual(len(schemas), 26)
        script = r"""
const fs = require('fs');
const vm = require('vm');
const schemas = JSON.parse(fs.readFileSync(0, 'utf8'));
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {state, renderPlanParameters, parameterPatches, scheduleGroups, scheduleEnabled, setScheduleEnabled}; window.quickStartController =');
const context = {window:{}};
vm.runInNewContext(source, context);
const {state, renderPlanParameters, parameterPatches, scheduleGroups, scheduleEnabled, setScheduleEnabled} = context.window.__probe;
for (const schema of schemas) {
  state.planSchema = schema;
  state.parameterDraft = {};
  state.segmentedPaths = {};
  state.scheduleStash = {};
  const rates = schema.fields.filter(field => field.editable && field.visible !== false &&
    field.display_group !== 'restricted' && field.path.includes('optimizer.') && field.path.endsWith('.lr'));
  const check = (label) => {
    const html = renderPlanParameters();
    for (const field of rates) {
      const needle = 'data-qs-param="' + field.path + '"';
      if (html.split(needle).length !== 2)
        throw Error(schema.method + ' ' + label + ': ' + field.path + ' editor count is ' + (html.split(needle).length - 1));
    }
    return html;
  };
  const original = check('original');
  const active = scheduleGroups(schema.fields).filter(scheduleEnabled);
  for (const group of active) {
    if (!group.fields.some(field => field.path === group.prefix + '.lr_values')) continue;
    if (!original.includes('data-qs-schedule-rate="' + group.prefix + '"'))
      throw Error(schema.method + ': independent stage rates missing for ' + group.prefix);
  }
  for (const group of active) setScheduleEnabled(group.prefix, false);
  const disabled = check('disabled');
  for (const field of rates) {
    if (!disabled.includes('data-qs-param="' + field.path + '"'))
      throw Error(schema.method + ': missing fixed rate ' + field.path);
    state.parameterDraft[field.path] = String(Number(field.value) + 0.001);
    const patch = parameterPatches().find(item => item.path === field.path);
    if (!patch || typeof patch.value !== 'number')
      throw Error(schema.method + ': rate is not saved as a number: ' + field.path);
  }
  for (const field of rates) delete state.parameterDraft[field.path];
  for (const group of active) setScheduleEnabled(group.prefix, true);
  check('restored');
  if (parameterPatches().length)
    throw Error(schema.method + ': off/on changed the original schedule: ' + JSON.stringify(parameterPatches()));
}
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script], cwd=ROOT,
            input=json.dumps(schemas), capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_split_editor_shows_current_and_paper_counts(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs');
const vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =', 'window.__probe = {state, renderParameterField}; window.quickStartController =');
const context = {window:{}};
vm.runInNewContext(source, context);
const {state, renderParameterField} = context.window.__probe;
state.dataset = {train_size:2700,training_pool_size:3000};
const html = renderParameterField({path:'data.num_val', label:'验证集样本数', value:300,
  kind:'number', editable:true, split_reference:{count:5000,total:50000}}, false);
if (!html.includes('value="300"') || !html.includes('/ 3000') ||
    !html.includes('原论文配置：5000 / 50000')) process.exit(1);
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_split_summary_tracks_the_current_draft_and_original_pool(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs'), vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =', 'window.__probe = {state, splitAllocationText}; window.quickStartController =');
const context = {window:{}};
vm.runInNewContext(source, context);
const {state, splitAllocationText} = context.window.__probe;
state.dataset = {train_size:2700,training_pool_size:3000};
state.planSchema = {fields:[{path:'data.validation_size',value:300}]};
if (!splitAllocationText().includes('训练 2700，验证 300')) process.exit(1);
state.parameterDraft['data.validation_size'] = '600';
if (!splitAllocationText().includes('训练 2400，验证 600')) process.exit(2);
state.parameterDraft['data.validation_size'] = '3000';
if (!splitAllocationText().includes('不合法')) process.exit(3);
state.parameterDraft = {};
state.planSchema = {fields:[{path:'data.num_val',value:300},{path:'data.num_clean',value:6}]};
if (!splitAllocationText().includes('训练 2694，验证 300，干净参考 6')) process.exit(4);
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

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
        for module in ('quickstart', 'yaml', 'data', 'sweep', 'results', 'papers'):
            self.assertIn('id: "' + module + '"', html)
        self.assertNotIn('id: "advanced"', html)
        self.assertNotIn('id: "beginner"', html)
        self.assertNotIn('完整新手引导', html)
        self.assertNotIn('数据集优先', html)

    def test_registered_dataset_and_training_guard_are_wired(self) -> None:
        script = (ROOT / "web" / "assets" / "quick_start.js").read_text(encoding="utf-8")
        self.assertIn("esc(item.location)", script)
        self.assertIn('payload.status === "already_registered"', script)
        self.assertIn("qs-required-input", script)
        self.assertIn('function renderReview()', script)
        self.assertIn('review.signature !== reviewSignature()', script)
        self.assertIn("loadingMarkup", script)
        self.assertIn("请勿重复点击", script)
        self.assertIn("正在读取数据集…", script)
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
        self.assertIn('function renderScheduleControl(group, allFields, compact = false)', script)
        self.assertIn('data-qs-schedule-toggle', script)
        self.assertIn('data-qs-schedule-milestone', script)
        self.assertIn('data-qs-schedule-rate', script)
        self.assertIn('每一格都可以独立修改', script)
        self.assertIn('禁止在 Web 修改的专属参数', script)
        self.assertNotIn('summary>公共实验设置', script)
        self.assertNotIn('summary>模型 / 组件选择', script)
        self.assertNotIn('summary>外部资源与运行前置条件', script)
        self.assertIn('parameterPatches(true)', script)
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

    def test_run_confirmation_uses_current_budget_not_original_plan_details(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs');
const vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {state, currentRunSummary}; window.quickStartController =');
const context = {window:{}};
vm.runInNewContext(source, context);
const {state, currentRunSummary} = context.window.__probe;
state.dataset = {alias:'cifar10-local'};
state.plan = {plan_id:'test', status:'ready', method:'GCE',
  summary:'旧计划', details:['训练轮次：trainer.epochs=120'], command:'lnl run --recipe gce'};
state.planSchema = {fields:[{path:'trainer.epochs', value:120, kind:'number', editable:true, visible:true}]};
state.parameterDraft = {'trainer.epochs':'1'};
(async () => {
  let confirmation = currentRunSummary(state.plan, []);
  if (!confirmation.includes('训练轮数：1') || confirmation.includes('120') || confirmation.includes('旧计划'))
    throw Error('epoch confirmation is stale: ' + confirmation);
  state.plan.method = 'L2RW';
  state.plan.details = ['训练轮次：trainer.epochs=180'];
  state.planSchema = {fields:[
    {path:'trainer.epochs', value:180, kind:'number', editable:true, visible:true},
    {path:'trainer.max_steps', value:80000, kind:'number', editable:true, visible:true}
  ]};
  state.parameterDraft = {'trainer.max_steps':'2'};
  confirmation = currentRunSummary(state.plan, []);
  if (!confirmation.includes('总更新步数：2') || confirmation.includes('180') ||
      confirmation.includes('80000') || confirmation.includes('训练轮数'))
    throw Error('step confirmation is stale: ' + confirmation);
})().catch(error => { console.error(error); process.exitCode = 1; });
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
    !html.includes('data-qs-param="optimizer.momentum"') ||
    !html.includes('data-qs-param="optimizer.weight_decay"') ||
    !html.includes('data-qs-param="model.name"')) process.exit(1);
state.parameterDraft['scheduler.lr_values'] = JSON.stringify([0.003, 0.0007]);
html = renderPlanParameters();
const patch = parameterPatches().find(item => item.path === 'scheduler.lr_values');
if (!html.includes('value="0.003"') || !html.includes('value="0.0007"') ||
    !html.includes('每一格都可以独立修改') || !patch ||
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

    def test_l2rw_disabled_schedule_returns_learning_rate_to_parameters(self) -> None:
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
state.planSchema = {method:'l2rw', fields:[
  {path:'scheduler.name', value:'none', editable:true, visible:true, display_group:'default', kind:'string'},
  {path:'scheduler.step_milestones', value:[40000,60000], editable:true, visible:true, display_group:'advanced', kind:'list'},
  {path:'scheduler.gamma', value:0.1, editable:true, visible:true, display_group:'default', kind:'number'},
  {path:'optimizer.lr', value:0.1, editable:true, visible:true, display_group:'default', kind:'number'},
  {path:'trainer.max_steps', value:80000, editable:true, visible:true, display_group:'default', kind:'number'}
]};
let html = renderPlanParameters();
if ((html.match(/data-qs-param="optimizer.lr"/g) || []).length !== 1 ||
    !html.includes('data-qs-schedule-rate="scheduler"')) process.exit(1);
state.parameterDraft['scheduler.step_milestones'] = '[]';
html = renderPlanParameters();
if (html.includes('data-qs-schedule-rate="scheduler"') ||
    (html.match(/data-qs-param="optimizer.lr"/g) || []).length !== 1 ||
    !html.includes('value="0.1"') ||
    !parameterPatches().some(item => item.path === 'scheduler.step_milestones')) process.exit(2);
state.parameterDraft['optimizer.lr'] = '0.05';
if (!parameterPatches().some(item => item.path === 'optimizer.lr' && item.value === 0.05)) process.exit(3);
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_schedule_toggle_restores_unsaved_custom_nodes_and_rates(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs');
const vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {state, setScheduleEnabled, renderPlanParameters, parameterPatches}; window.quickStartController =');
const context = {window:{}};
vm.runInNewContext(source, context);
const {state, setScheduleEnabled, renderPlanParameters, parameterPatches} = context.window.__probe;
const common = {editable:true, visible:true, display_group:'default'};
state.planSchema = {method:'gce', fields:[
  {...common, path:'scheduler.name', value:'multistep', kind:'string'},
  {...common, path:'scheduler.milestones', value:[40,80], kind:'list'},
  {...common, path:'scheduler.gamma', value:0.1, kind:'number'},
  {...common, path:'scheduler.lr_values', value:null, kind:'list', nullable:true},
  {...common, path:'optimizer.lr', value:0.01, kind:'number'},
  {...common, path:'trainer.epochs', value:120, kind:'number'}
]};
state.parameterDraft['scheduler.milestones'] = '[30,70]';
state.parameterDraft['scheduler.lr_values'] = '[0.004,0.0008]';
setScheduleEnabled('scheduler', false);
if (!renderPlanParameters().includes('data-qs-param="optimizer.lr"') ||
    !parameterPatches().some(item => item.path === 'scheduler.name' && item.value === 'none')) process.exit(1);
setScheduleEnabled('scheduler', true);
if (JSON.stringify(parameterPatches().find(item => item.path === 'scheduler.milestones')?.value) !== '[30,70]' ||
    JSON.stringify(parameterPatches().find(item => item.path === 'scheduler.lr_values')?.value) !== '[0.004,0.0008]') process.exit(2);
state.planSchema = {method:'l2rw', fields:[
  {...common, path:'scheduler.name', value:'none', kind:'string'},
  {...common, path:'scheduler.step_milestones', value:[40000,60000], kind:'list'},
  {...common, path:'scheduler.gamma', value:0.1, kind:'number'},
  {...common, path:'optimizer.lr', value:0.1, kind:'number'},
  {...common, path:'trainer.max_steps', value:80000, kind:'number'}
]};
state.parameterDraft = {'scheduler.step_milestones':'[30000,50000]'};
state.scheduleStash = {};
setScheduleEnabled('scheduler', false);
if (JSON.stringify(parameterPatches().find(item => item.path === 'scheduler.step_milestones')?.value) !== '[]') process.exit(3);
setScheduleEnabled('scheduler', true);
if (JSON.stringify(parameterPatches().find(item => item.path === 'scheduler.step_milestones')?.value) !== '[30000,50000]') process.exit(4);
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

    def test_binary_noise_selector_creates_two_editable_rates(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs = require('fs');
const vm = require('vm');
let source = fs.readFileSync('web/assets/quick_start.js', 'utf8');
source = source.replace('window.quickStartController =',
  'window.__probe = {state, updateNoise, renderNoise}; window.quickStartController =');
const fields = {
  'qs-add-noise': {checked:true}, 'qs-noise': {value:'binary_asymmetric_rcn'},
  'qs-seed': {value:'1'}
};
const context = {window:{}, document:{getElementById:id => fields[id] || null}};
vm.runInNewContext(source, context);
const {state, updateNoise, renderNoise} = context.window.__probe;
state.noise = {dataset_state:'clean', options:[
  {key:'clean',label:'保持干净'},
  {key:'symmetric',label:'对称噪声',requires_rate:true},
  {key:'binary_asymmetric_rcn',label:'二分类非对称噪声',requires_rate:false}
]};
state.noiseSelection = {kind:'synthetic',key:'symmetric',rate:0.2,seed:1};
updateNoise({target:{id:'qs-noise'}});
if (state.error || state.noiseSelection.key !== 'binary_asymmetric_rcn' ||
    state.noiseSelection.rho_positive !== 0.2 || state.noiseSelection.rho_negative !== 0.1) process.exit(1);
state.dataset = {alias:'binary'};
const html = renderNoise();
if (!html.includes('id="qs-rho-positive"') || !html.includes('id="qs-rho-negative"') ||
    html.includes('id="qs-rate"')) process.exit(2);
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
