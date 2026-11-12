from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import unittest

from web.command_console import STATIC_ASSETS, _config_schema, _parameter_registry


ROOT = Path(__file__).resolve().parents[1]


class QuickStartAssetsTests(unittest.TestCase):
    def test_external_label_path_is_required_on_noise_step(self) -> None:
        script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
let source=fs.readFileSync('web/assets/quick_start.js','utf8');
source=source.replace('window.quickStartController =','window.__probe={state,quickStartCanAdvance,renderNoise}; window.quickStartController =');
const ctx={window:{},document:{getElementById(){return null;},querySelector(){return null;}}};
vm.runInNewContext(source,ctx);
const {state,quickStartCanAdvance,renderNoise}=ctx.window.__probe;
state.dataset={num_classes:10};state.noise={dataset_state:'clean',options:[{key:'external_torch',label:'外部标签文件'}]};
state.noiseSelection={kind:'synthetic',key:'external_torch',path:'',seed:1};
assert(!quickStartCanAdvance({id:'noise'}));assert(renderNoise().includes('请选择外部标签文件后继续'));
state.noiseSelection.path=' F:/labels.pt ';assert(quickStartCanAdvance({id:'noise'}));
state.noiseSelection={kind:'synthetic',key:'symmetric'};assert(quickStartCanAdvance({id:'noise'}));
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_external_noise_label_picker_reuses_existing_web_picker(self) -> None:
        self.assertIsNotNone(shutil.which("node"), "CI requires Node-backed UI verification")
        script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
let source=fs.readFileSync('web/assets/quick_start.js','utf8');
source=source.replace('window.quickStartController =',
 'window.__probe={state,renderNoise,updateNoise}; window.quickStartController =');
source=source.replace('function render() {','function render() { return;');
const index=fs.readFileSync('web/index.html','utf8');
const shared=index.slice(index.indexOf('    async function pickPath('),index.indexOf('    function renderStructured('));
let result={cancelled:false,path:'F:/labels with spaces/带噪标签.pt'},ok=true,calls=[],bindings=0;
const callbacks={},button={dataset:{pickerTarget:'qs-noise-path',pickerMode:'open_file',pickerKind:'all'},
 addEventListener(name,fn){callbacks[name]=fn;bindings++;}};
const controls={
 'qs-noise-path':{value:'F:/old.pt',dispatchEvent(event){if(event.type==='change')ctx.window.__probe.updateNoise();}},
 'qs-noise-pick':button,'qs-noise':{value:'external_torch'},'qs-add-noise':{checked:true},'qs-seed':{value:'73'}
};
const ctx={window:{setTimeout(){return 1;},clearTimeout(){}},
 Event:class {constructor(type){this.type=type;}},statusNode:{},
 document:{getElementById(id){return controls[id]||null;},querySelector(){return null;},
 querySelectorAll(){return [button];}},
 fetch:async(url,options)=>{
  const body=JSON.parse(options.body);calls.push({url,body});
  if(url==='/api/picker')return {ok,json:async()=>result};
  return {ok:true,json:async()=>({methods:[]})};
 }};
vm.runInNewContext(source,ctx);
vm.runInNewContext(shared,ctx);
const p=ctx.window.__probe,s=p.state;
s.dataset={alias:'mini',num_classes:10};
s.noise={dataset_state:'clean',options:[{key:'external_torch',label:'外部标签文件'}]};
s.noiseSelection={kind:'synthetic',key:'external_torch',path:'F:/old.pt',seed:73};
async function click(){callbacks.click();await new Promise(setImmediate);}
(async()=>{
 const html=p.renderNoise();
 assert(html.includes('data-picker-target="qs-noise-path"'));
 assert(html.includes('data-picker-mode="open_file"'));
 assert(html.includes('>浏览</button>'));
 assert(source.includes('window.installPathPickers?.(panel)'));
 assert(!source.includes('function pickNoiseLabelPath'));
 ctx.installPathPickers();ctx.installPathPickers();assert.strictEqual(bindings,1);
 s.plan={plan_id:'old'};
 await click();
 assert.deepStrictEqual(calls[0],{url:'/api/picker',body:{mode:'open_file',kind:'all',initial:'F:/old.pt'}});
 assert.strictEqual(controls['qs-noise-path'].value,result.path);
 assert.strictEqual(s.noiseSelection.path,result.path);
 assert.strictEqual(s.noiseSelection.seed,73);assert.strictEqual(s.plan,null);
 assert(calls.some(c=>c.url==='/api/quick-start/methods'&&c.body.noise.path===result.path));
 const previous=s.noiseSelection;
 result={cancelled:true,path:null};calls=[];await click();
 assert.strictEqual(s.noiseSelection,previous);
 assert.strictEqual(controls['qs-noise-path'].value,previous.path);
 assert.strictEqual(calls.length,1);
 ok=false;result={error:'Windows picker unavailable'};await click();
 assert(ctx.statusNode.textContent.includes('Windows picker unavailable'));
 assert.strictEqual(ctx.statusNode.className,'status failed');
 assert.strictEqual(controls['qs-noise-path'].value,previous.path);
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_selecting_paper_preserves_noise_step_seed(self) -> None:
        self.assertIsNotNone(shutil.which("node"), "CI requires Node-backed UI verification")
        script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('web/assets/quick_start.js','utf8');
const binding=source.split('\n').find(line=>line.includes('panel.querySelectorAll(".qs-method-card")'));
(async()=>{
  for(const seed of [1,73]) {
    const state={noiseSelection:{seed},methods:['l2rw','gce','pcse'].map(paper_id=>
      ({paper_id,recipe_id:paper_id+'-recipe',status:'ready'}))};
    const calls=[];
    let click;
    const button={dataset:{paper:'l2rw'},addEventListener(event,handler){click=handler;}};
    const ctx={state,panel:{querySelectorAll(){return [button];}},render(){},confirmConfigurationChange(){return true;},
      buildPlan(){calls.push(state.noiseSelection.seed);},
      request:async()=>({fields:[{path:'seed',value:1234}]})};
    vm.runInNewContext(binding,ctx);
    for(const method of state.methods) {
      button.dataset.paper=method.paper_id;
      click.call(button);
      await Promise.resolve();await Promise.resolve();
      assert.strictEqual(state.noiseSelection.seed,seed,method.paper_id);
    }
    assert.deepStrictEqual(calls,[seed,seed,seed]);
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dataset_and_paper_switches_confirm_before_discarding_draft(self) -> None:
        script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
let source=fs.readFileSync('web/assets/quick_start.js','utf8');
const binding=source.split('\n').find(line=>line.includes('panel.querySelectorAll(".qs-method-card")'));
source=source.replace('window.quickStartController =',
 'window.__probe={state,changeDatasetPath,confirmConfigurationChange}; window.quickStartController =');
source=source.replace('function render() {','function render() { return;');
let approve=false,prompts=0,builds=0,click;
const ctx={window:{confirm(message){prompts++;assert(message.includes('未保存'));return approve;}}};
vm.runInNewContext(source,ctx);
const p=ctx.window.__probe,s=p.state;
function draft(){
 s.path='F:/old';s.dataset={alias:'local'};s.selectedPaperId='gce';s.plan={plan_id:'old'};
 s.planSchema={fields:[{path:'optimizer.lr',kind:'number',editable:true,value:0.1}]};
 s.parameterDraft={'optimizer.lr':'0.3'};
}
draft();
assert.strictEqual(p.changeDatasetPath('F:/new'),false);
assert.strictEqual(s.path,'F:/old');assert.strictEqual(s.plan.plan_id,'old');
assert.strictEqual(s.parameterDraft['optimizer.lr'],'0.3');
approve=true;assert.strictEqual(p.changeDatasetPath('F:/new'),true);
assert.strictEqual(s.plan,null);assert.strictEqual(Object.keys(s.parameterDraft).length,0);
draft();approve=false;s.methods=[{paper_id:'l2rw',status:'ready'}];
const button={dataset:{paper:'l2rw'},addEventListener(event,fn){click=fn;}};
Object.assign(ctx,{state:s,panel:{querySelectorAll(){return [button];}},render(){},
 confirmConfigurationChange:p.confirmConfigurationChange,buildPlan(){builds++;}});
vm.runInNewContext(binding,ctx);
click.call(button);assert.strictEqual(s.selectedPaperId,'gce');assert.strictEqual(builds,0);
approve=true;click.call(button);assert.strictEqual(s.selectedPaperId,'l2rw');
assert.strictEqual(s.plan,null);assert.strictEqual(builds,1);
assert.strictEqual(Object.keys(s.parameterDraft).length,0);
const before=prompts;click.call(button);assert.strictEqual(prompts,before);assert.strictEqual(builds,1);
'''
        self.assertIsNotNone(shutil.which("node"))
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_compatibility_guide_reuses_loaded_noise_facts(self) -> None:
        self.assertIsNotNone(shutil.which("node"), "CI requires Node-backed UI verification")
        script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
let source=fs.readFileSync('web/assets/quick_start.js','utf8');
source=source.replace('window.quickStartController =',
 'window.__probe={state}; window.quickStartController =');
source=source.replace('function render() {','function render() { return;');
const calls=[];
const ctx={window:{setTimeout(){return 1;},clearTimeout(){}},
 fetch:async url=>{calls.push(url);return {ok:true,json:async()=>({requires_confirmation:true,dataset_state:'unknown',options:[]})};}};
vm.runInNewContext(source,ctx);
const s=ctx.window.__probe.state,c=ctx.window.quickStartController;
(async()=>{
 s.dataset={alias:'fixture'};s.noise=null;
 await c.showCompatibilityGuide();
 assert.deepStrictEqual(calls,['/api/quick-start/noises?dataset=fixture']);
 assert(s.noise);await c.showCompatibilityGuide();assert.strictEqual(calls.length,1);
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stale_plan_and_method_responses_cannot_replace_current_selection(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is unavailable")
        script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
let source=fs.readFileSync('web/assets/quick_start.js','utf8');
source=source.replace('window.quickStartController =',
 'window.__probe={state,buildPlan,loadMethods}; window.quickStartController =');
source=source.replace('function render() {','function render() { return;');
const pending=[];
const ctx={window:{setTimeout(){return 1;},clearTimeout(){}},
 fetch:(url,options)=>new Promise(resolve=>pending.push({url,resolve}))};
vm.runInNewContext(source,ctx);
const p=ctx.window.__probe,s=p.state;
function reply(index,payload){pending[index].resolve({ok:true,json:async()=>payload});}
(async()=>{
 s.dataset={alias:'data'};s.selectedPaperId='gce';
 const old=p.buildPlan({navigate:false});
 s.selectedPaperId='pcse';const current=p.buildPlan({navigate:false});
 reply(1,{plan_id:'new',paper_id:'pcse'});await current;
 reply(0,{plan_id:'old',paper_id:'gce'});await old;
 assert.strictEqual(s.plan.paper_id,'pcse');assert.strictEqual(s.plan.plan_id,'new');
 const oldMethods=p.loadMethods();
 s.noiseSelection={kind:'native',key:'native'};const currentMethods=p.loadMethods();
 reply(3,{methods:[{paper_id:'new-method'}]});await currentMethods;
 reply(2,{methods:[{paper_id:'old-method'}]});await oldMethods;
 assert.strictEqual(s.methods[0].paper_id,'new-method');
 const abandoned=p.buildPlan({navigate:false});s.selectedPaperId='lend';
 reply(4,{plan_id:'abandoned',paper_id:'pcse'});await abandoned;
 assert.strictEqual(s.plan,null);
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

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
const ctx={window:{setTimeout(){return 1;},clearTimeout(){},confirm(){return true;}},document:{getElementById(){return null;}},
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
  s.path='F:/old';s.dataset={alias:'old'};s.noise={};s.plan={};s.planSchema={fields:[{path:'optimizer.lr',value:0.1,kind:'number',editable:true}]};
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
    }
    if(field.rows) {
      if(!html.includes('textarea hidden') || !html.includes('data-parameter-list')) throw Error('raw nested JSON control '+field.path);
      for(const row of field.rows) for(const child of row) if(!html.includes(child.label)) throw Error('missing nested label');
    }
    if(html.includes('通用配置参数') || html.includes('可修改的高级实验/实现选项')) throw Error('vague helper');
  }
  const panel=p.renderPlanParameters();
  if(panel.includes('>Normalize<') || panel.includes('>Scale<') || panel.includes('>Center<')) throw Error('old preprocessing names');
}
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT,
                                input=json.dumps(schemas, ensure_ascii=False), capture_output=True,
                                text=True, encoding="utf-8", check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

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


class ContinuousScheduleRegressionTests(unittest.TestCase):
    def test_jocor_saved_initial_rate_is_the_rate_used_by_training(self) -> None:
        import tempfile, yaml, torch
        from web import command_console as console
        from lnl_toolbox.training.multi_model_experiment import _apply_epoch_optimizer_schedule
        for path in ('optimizer.lr', 'scheduler.initial_lr'):
            with self.subTest(path=path), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                saved = console._save_config({'recipe': 'jocor-cifar10-symmetric05-reproduction',
                         'path': str(Path(directory) / 'edited.yaml'),
                         'patches': [{'path': path, 'value': .003}], 'acknowledge_paper_impact': True})
                config = yaml.safe_load((ROOT / saved['path']).read_text(encoding='utf-8'))
                self.assertEqual(config['optimizer']['lr'], .003)
                self.assertEqual(config['scheduler']['initial_lr'], .003)
                optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=config['optimizer']['lr'])
                _apply_epoch_optimizer_schedule(optimizer, config['scheduler'], epoch=0, total_epochs=200)
                self.assertEqual(optimizer.param_groups[0]['lr'], .003)
                _apply_epoch_optimizer_schedule(optimizer, config['scheduler'], epoch=120, total_epochs=200)
                self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .002)
                self.assertEqual(optimizer.param_groups[0]['betas'][0], .1)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            target = Path(directory) / 'conflict.yaml'
            with self.assertRaisesRegex(ValueError, '起始学习率设置冲突'):
                console._save_config({'recipe': 'jocor-cifar10-symmetric05-reproduction', 'path': str(target),
                    'patches': [{'path': 'optimizer.lr', 'value': .003}, {'path': 'scheduler.initial_lr', 'value': .004}],
                    'acknowledge_paper_impact': True})
            self.assertFalse(target.exists())

    def test_mc_ldce_can_save_its_own_linear_schedule_boundary(self) -> None:
        import tempfile, yaml
        from web import command_console as console
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            saved = console._save_config({'recipe': 'mc-ldce-cifar10-reproduction',
                     'path': str(Path(directory) / 'edited.yaml'),
                     'patches': [{'path': 'scheduler.decay_start', 'value': 81}], 'acknowledge_paper_impact': True})
            config = yaml.safe_load((ROOT / saved['path']).read_text(encoding='utf-8'))
            self.assertEqual(config['scheduler']['decay_start'], 81)
            self.assertNotIn('start_epoch', config['scheduler'])
            for start in (-1, config['trainer']['epochs']):
                with self.subTest(start=start), self.assertRaisesRegex(ValueError, 'decay_start'):
                    console._save_config({'recipe': 'mc-ldce-cifar10-reproduction',
                        'path': str(Path(directory) / 'invalid.yaml'),
                        'patches': [{'path': 'scheduler.decay_start', 'value': start}], 'acknowledge_paper_impact': True})




if __name__ == "__main__":
    unittest.main()
