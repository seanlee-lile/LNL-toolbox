const state = {
  blocks: [],
  recipe: { schema_version: 1, name: 'scratch_recipe', description: '', steps: [] },
  selected: null,
  paletteSelection: null,
  collapsed: new WeakSet(),
  activeInsertionTarget: null,
  drag: null,
  datasets: [],
  datasetFacts: {},
  validated: false,
  errorStepId: null,
  errorMessage: '',
  lastRun: null,
  runtimeLimits: { max_epochs: 1, max_batches: 1, skip_final_test: true },
  paletteQuery: '',
  paletteCollapsed: new Set(),
  formulaEditor: { steps: [], editingId: null },
};
const $ = (id) => document.getElementById(id);
const apiBase = location.pathname.startsWith('/scratch') ? '/api/scratch' : '/api';
let uiIdCounter = 0;

async function api(path, options = {}) {
  const relativePath = path.startsWith('/api/') ? path.slice(4) : path;
  const response = await fetch(apiBase + relativePath, options);
  const body = await response.json();
  if (!response.ok || body.ok === false) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function blockInfo(id) { return state.blocks.find((item) => item.id === id); }
function blockCategory(info) { return String(info?.category || '未分类'); }
function stepsWithInfo() {
  return flatRecipeSteps().map((step) => ({ step, info: blockInfo(step.block) }));
}
function stepProviding(slot) {
  return stepsWithInfo().find(({ step, info }) =>
    (info?.provides || []).some((name) => slotValue(info, step, name) === slot));
}
function isDatasetSourceInfo(info) {
  return Boolean(info && (info.provides || []).some((name) => name === 'train_source'));
}
function makeUiId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function') return globalThis.crypto.randomUUID();
  uiIdCounter += 1;
  return `scratch-ui-${Date.now()}-${uiIdCounter}`;
}
function newStep(id) {
  const step = { block: id, params: {}, _uiId: makeUiId() };
  if (blockInfo(id)?.kind !== 'action') step.steps = [];
  return step;
}

function ensureUiIds(steps) {
  (steps || []).forEach((step) => {
    if (!step._uiId) step._uiId = makeUiId();
    if (Array.isArray(step.steps)) ensureUiIds(step.steps);
  });
}

function findStepById(uiId, steps = state.recipe.steps) {
  for (const step of steps || []) {
    if (step._uiId === uiId) return step;
    const nested = Array.isArray(step.steps) ? findStepById(uiId, step.steps) : null;
    if (nested) return nested;
  }
  return null;
}

function findParentArrayAndIndex(uiId, steps = state.recipe.steps, parentId = '__root__') {
  for (let index = 0; index < (steps || []).length; index += 1) {
    const step = steps[index];
    if (step._uiId === uiId) return { array: steps, index, parentId };
    const nested = Array.isArray(step.steps) ? findParentArrayAndIndex(uiId, step.steps, step._uiId) : null;
    if (nested) return nested;
  }
  return null;
}

function getChildrenArray(parentId) {
  if (parentId === '__root__') return state.recipe.steps;
  const parent = findStepById(parentId);
  if (!parent) return null;
  if (!Array.isArray(parent.steps)) parent.steps = [];
  return parent.steps;
}

function getPlacementContext(parentId) {
  if (parentId === '__root__') return 'top';
  const parent = findStepById(parentId);
  if (!parent) return 'top';
  if (parent.block === 'epoch_loop') return 'epoch';
  if (parent.block === 'batch_loop') return 'batch';
  return 'any';
}

function stripUiFields(value) {
  if (Array.isArray(value)) return value.map(stripUiFields);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !key.startsWith('_ui'))
    .map(([key, child]) => [key, stripUiFields(child)]));
}

function placementAllows(info, parentId) {
  if (!info) return false;
  const context = getPlacementContext(parentId);
  return (info.placement || ['any']).includes('any') || (info.placement || []).includes(context);
}

function containsStep(root, uiId) {
  return root?._uiId === uiId || (root?.steps || []).some((child) => containsStep(child, uiId));
}

function findStepPath(uiId, steps = state.recipe.steps, path = []) {
  for (const step of steps || []) {
    const nextPath = [...path, step];
    if (step._uiId === uiId) return nextPath;
    if (Array.isArray(step.steps)) {
      const nested = findStepPath(uiId, step.steps, nextPath);
      if (nested) return nested;
    }
  }
  return null;
}

function slotValue(info, step, name) {
  const schema = info?.params?.[name] || {};
  const value = step?.params?.[name] ?? schema.default ?? name;
  return typeof value === 'string' && value.trim() ? value : name;
}

function inferAvailableSlot(info, name, available) {
  const schema = info?.params?.[name] || {};
  const configured = schema.default;
  if (configured && available.has(configured)) return configured;
  return null;
}

function inferOutputSlot(info, name, target) {
  return null;
}

function addProvidedKeys(keys, step) {
  const info = blockInfo(step.block);
  (info?.provides || []).forEach((name) => keys.add(slotValue(info, step, name)));
}

function addStepsBefore(keys, steps, endIndex) {
  (steps || []).slice(0, endIndex).forEach((step) => addProvidedKeys(keys, step));
}

function availableKeysBefore(target) {
  const keys = new Set(Object.keys(state.recipe.settings || {}));
  if (!target) return keys;
  if (target.parentId === '__root__') {
    addStepsBefore(keys, state.recipe.steps, target.index);
    return keys;
  }
  const path = findStepPath(target.parentId);
  if (!path) return keys;
  let siblings = state.recipe.steps;
  for (const parent of path) {
    const parentIndex = siblings.indexOf(parent);
    addStepsBefore(keys, siblings, parentIndex);
    addProvidedKeys(keys, parent);
    siblings = Array.isArray(parent.steps) ? parent.steps : [];
  }
  addStepsBefore(keys, siblings, target.index);
  return keys;
}

function loopAlreadyExists(blockId, parentId, movingId = null) {
  const array = getChildrenArray(parentId) || [];
  return array.some((step) => step.block === blockId && step._uiId !== movingId);
}

function availabilityReason(info, target, movingId = null) {
  if (!info || !target) return '请先选择插入位置';
  if (!placementAllows(info, target.parentId)) return `placement 不允许放入 ${target.context} 层`;
  if (info.id === 'epoch_loop' && target.parentId !== '__root__') return 'Epoch Loop 只能放在根层';
  if (info.id === 'batch_loop' && getPlacementContext(target.parentId) !== 'epoch') return 'Batch Loop 只能放入 Epoch Loop';
  if ((info.id === 'epoch_loop' || info.id === 'batch_loop') && loopAlreadyExists(info.id, target.parentId, movingId)) return `${info.name} 当前只允许一个`;
  if (movingId && target.parentId !== '__root__' && containsStep(findStepById(movingId), target.parentId)) return '不能把循环拖入自己的子层';
  const available = availableKeysBefore(target);
  const missing = (info.requires || [])
    .filter((name) => !inferAvailableSlot(info, name, available))
    .map((name) => slotValue(info, null, name));
  if (missing.length) return `前置 slot 不完整：${missing.join(', ')}`;
  return null;
}

function canInsert(info, parentId, movingId = null, index = 0) {
  return availabilityReason(info, { parentId, index, context: getPlacementContext(parentId) }, movingId) === null;
}

function findDropTarget(zone) {
  return { parentId: zone.dataset.parentId || '__root__', index: Number(zone.dataset.index || 0), context: zone.dataset.context || 'top' };
}

function setActiveInsertionTarget(target) {
  state.activeInsertionTarget = target;
  draw();
}

function targetLabel(target) {
  if (!target) return '请先点击一个插入位置';
  const contextNames = { top: '根层', epoch: 'Epoch Loop', batch: 'Batch Loop', any: '当前容器' };
  const next = target.context === 'batch' ? ' 下一步：添加 Forward。' : '';
  return `当前插入位置：${contextNames[target.context] || target.context}，第 ${target.index + 1} 步。${next}`;
}

function defaultInsertionTarget() {
  const epoch = state.recipe.steps.find((step) => step.block === 'epoch_loop');
  const batch = epoch?.steps?.find((step) => step.block === 'batch_loop');
  if (!batch) return null;
  const afterGetBatch = (batch.steps || []).findIndex((step) => step.block === 'get_batch');
  return { parentId: batch._uiId, index: afterGetBatch >= 0 ? afterGetBatch + 1 : (batch.steps || []).length, context: 'batch' };
}

function workflowPresence() {
  const infos = stepsWithInfo();
  return {
    data: infos.some(({ info }) => blockCategory(info) === 'Data' || isDatasetSourceInfo(info)),
    model: infos.some(({ info, step }) => info?.id === 'create_model' || blockCategory(info) === 'Model' || step.block === 'create_model'),
    train: infos.some(({ info, step }) => ['epoch_loop', 'batch_loop'].includes(step.block) || ['Training', 'Control', 'Runtime'].includes(blockCategory(info))),
    formula: infos.some(({ info, step }) => info?.formula || info?.kind === 'formula' || ['forward', 'backward', 'cross_entropy', 'mean_loss'].includes(step.block)),
  };
}

function updateWorkflowGuide() {
  const guide = $('workflow-guide');
  if (!guide) return;
  const presence = workflowPresence();
  const sequence = ['data', 'model', 'train', 'formula'];
  const completed = sequence.filter((key) => presence[key]).length;
  const current = !presence.data ? 'data' : !presence.model ? 'model' : !presence.train ? 'train' : !presence.formula ? 'formula' : !state.validated ? 'check' : !state.lastRun ? 'run' : 'result';
  guide.querySelectorAll('.workflow-step').forEach((node) => {
    const key = node.dataset.workflow;
    node.classList.toggle('active', key === current);
    const isCompleted = sequence.includes(key) ? presence[key] : (key === 'check' && state.validated) || (key === 'run' && Boolean(state.lastRun));
    node.classList.toggle('completed', Boolean(isCompleted) && key !== current);
  });
}

function setResultState(label, className = '') {
  const status = $('result-status');
  if (!status) return;
  status.textContent = label;
  status.className = `result-status ${className}`.trim();
}

function clearResultDetails() {
  if ($('result-metrics')) $('result-metrics').replaceChildren();
  if ($('result-artifacts')) $('result-artifacts').replaceChildren();
}

function showMessage(message, className = '') {
  const text = String(message ?? '');
  const output = $('output');
  if (output) { output.textContent = text; output.className = className; }
  const summary = $('result-summary');
  if (summary) summary.textContent = text;
  if (className === 'error') setResultState('需要处理', 'error');
  else if (className === 'ok') setResultState('已完成', 'ok');
  else if (!state.lastRun) setResultState('尚未运行');
  updateWorkflowGuide();
}

function formatResultValue(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : '—';
  if (typeof value === 'boolean') return value ? '是' : '否';
  if (typeof value === 'string') return value;
  try { return JSON.stringify(value); } catch (error) { return String(value); }
}

function appendMetricCard(name, value) {
  const metrics = $('result-metrics');
  if (!metrics) return;
  const card = document.createElement('div'); card.className = 'metric-card';
  const label = document.createElement('strong'); label.textContent = name;
  const rendered = document.createElement('span'); rendered.textContent = formatResultValue(value);
  card.append(label, rendered); metrics.appendChild(card);
}

function renderRunResult(result) {
  state.lastRun = result;
  clearResultDetails();
  const metrics = Array.isArray(result?.metrics) ? result.metrics : (result?.metrics ? [result.metrics] : []);
  metrics.forEach((item, index) => {
    if (item && typeof item === 'object' && !Array.isArray(item)) Object.entries(item).forEach(([name, value]) => appendMetricCard(metrics.length > 1 ? `${index + 1}. ${name}` : name, value));
    else appendMetricCard(`指标 ${index + 1}`, item);
  });
  if (!metrics.length) {
    const empty = $('result-metrics');
    if (empty) empty.textContent = '本次运行没有返回指标；请展开原始输出查看详细信息。';
  }
  const artifacts = $('result-artifacts');
  if (artifacts && result?.artifact_dir) {
    const heading = document.createElement('strong'); heading.textContent = '产物'; artifacts.appendChild(heading);
    const base = String(result.artifact_dir).replace(/[\\/]$/, '');
    [['目录', base], ['Recipe', `${base}/recipe.yaml`], ['解析后 Recipe', `${base}/resolved_recipe.yaml`], ['运行日志', `${base}/stdout.log`], ['公式溯源', `${base}/formula_provenance.json`]].forEach(([label, path]) => {
      const row = document.createElement('div');
      const name = document.createElement('span'); name.textContent = `${label}：`;
      const value = document.createElement('code'); value.textContent = path;
      row.append(name, value); artifacts.appendChild(row);
    });
  }
  const output = $('output');
  if (output) output.textContent = JSON.stringify(result, null, 2);
  if ($('result-summary')) $('result-summary').textContent = '运行完成。下方显示本次结构验证返回的指标和产物位置；原始 JSON 可展开查看。';
  setResultState('运行完成', 'ok');
  if ($('result-details')) $('result-details').open = false;
  updateWorkflowGuide();
}

function markDirty() {
  state.validated = false;
  state.lastRun = null;
  state.errorStepId = null;
  state.errorMessage = '';
  updateRunState();
}

function recipeDataReady() {
  const dataset = stepsWithInfo().find(({ info }) => isDatasetSourceInfo(info));
  if (dataset) {
    const alias = String(dataset.step.params?.dataset ?? dataset.step.params?.name ?? '').trim();
    if (!alias) return false;
    if (alias === 'synthetic') return true;
    return Boolean(state.datasets.find((item) => item.alias === alias && item.status === 'ready'));
  }
  return false;
}

function updateRunState() {
  const run = $('run');
  if (run) run.disabled = !(state.validated && recipeDataReady());
}

function renderRuntimeLimits() {
  const epochStep = flatRecipeSteps().find((step) => step.block === 'epoch_loop');
  const formalEpochs = epochStep?.params?.epochs ?? blockInfo('epoch_loop')?.params?.epochs?.default ?? '—';
  $('runtime-limits').textContent = `正式配置：${formalEpochs} epochs；本次结构验证：${state.runtimeLimits.max_epochs} epoch / ${state.runtimeLimits.max_batches} batch（跳过最终测试）`;
}

function selectedTarget(step) {
  const location = step ? findParentArrayAndIndex(step._uiId) : null;
  return { parentId: location?.parentId || '__root__', index: location?.index || 0 };
}

function isOutputSlot(name) {
  return name === 'save_as' || name.endsWith('_as');
}

function controlKind(name, schema) {
  if (schema.type === 'slot') return isOutputSlot(name) ? 'output-slot' : 'slot';
  if (schema.type === 'str' && name === 'model') return 'model';
  if (schema.type === 'str' && name === 'optimizer') return 'optimizer';
  return schema.type || 'text';
}

function optionValues(name, schema, step) {
  const kind = controlKind(name, schema);
  if (kind === 'enum') {
    return schema.options || [];
  }
  if (kind === 'model') return [
    'cifar_cnn8', 'cifar_six_conv', 'cifar_resnet18', 'cifar_resnet34',
    'cifar_resnet32', 'cifar_resnet50', 'resnet18', 'resnet34', 'resnet32',
    'resnet50', 'preact_resnet18', 'mlp', 'linear',
  ];
  if (kind === 'optimizer') return ['sgd', 'adam'];
  if (kind === 'dataset') {
    const mode = String(step.params?.source_mode || 'registered');
    const current = step.params[name];
    const values = mode === 'builtin'
      ? ['synthetic']
      : mode === 'registered'
        ? state.datasets.filter((item) => item.status === 'ready').map((item) => item.alias || item.name).filter(Boolean)
        : [];
    return [...new Set(['', ...values, current].filter((value) => value !== undefined))];
  }
  if (kind === 'slot') {
    const keys = [...availableKeysBefore(selectedTarget(step))];
    const current = step.params[name] ?? schema.default;
    if (current && !keys.includes(current)) keys.push(current);
    return keys.sort();
  }
  return [];
}

function renderParamControl(name, schema, step) {
  const kind = controlKind(name, schema);
  const current = step.params[name] ?? schema.default ?? '';
  let input;
  if (['enum', 'model', 'optimizer', 'dataset', 'slot'].includes(kind)) {
    input = document.createElement('select');
    optionValues(name, schema, step).forEach((value) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value || '未选择';
      option.selected = String(value) === String(current);
      input.appendChild(option);
    });
  } else if (kind === 'bool') {
    input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = Boolean(current);
  } else if (kind === 'int' || kind === 'float') {
    input = document.createElement('input');
    input.type = 'number';
    input.value = current;
    if (schema.min !== undefined) input.min = schema.min;
    if (schema.max !== undefined) input.max = schema.max;
    input.step = kind === 'int' ? '1' : 'any';
  } else {
    input = document.createElement('input');
    input.type = 'text';
    input.value = current;
  }
  input.dataset.param = name;
  const updateValue = () => {
    if (kind === 'bool') step.params[name] = input.checked;
    else if (kind === 'int') step.params[name] = Number.parseInt(input.value, 10);
    else if (kind === 'float') step.params[name] = Number.parseFloat(input.value);
    else step.params[name] = input.value;
    markDirty();
    renderPalette();
    if (isDatasetSourceInfo(blockInfo(step.block)) && (name === 'dataset' || name === 'root' || name === 'path')) {
      renderInspector();
      if (name === 'dataset') loadDatasetFacts(step.params[name]);
    }
  };
  input.oninput = updateValue;
  input.onchange = updateValue;
  return input;
}

function syntheticFacts() {
  return {
    alias: 'synthetic', name: 'synthetic', path: null, adapter: 'scratch.synthetic', status: 'ready',
    train_samples: null, test_samples: null, num_classes: null, input_shape: null,
    has_clean_target: true, has_noisy_target: false, has_sample_index: true,
    layout_validated: true, training_verified: true, inspect_status: 'built-in',
    noise_methods: ['none', 'symmetric', 'pairflip', 'class_conditional', 'instance_dependent'],
  };
}

async function loadDatasetFacts(alias) {
  const name = String(alias || '').trim();
  if (!name) return;
  if (name === 'synthetic') {
    state.datasetFacts.synthetic = syntheticFacts();
    renderInspector();
    renderPalette();
    return;
  }
  try {
    state.datasetFacts[name] = await api('/api/dataset/' + encodeURIComponent(name));
  } catch (error) {
    state.datasetFacts[name] = { alias: name, status: 'unknown', noise_methods: [], error: error.message };
  }
  renderInspector();
  renderPalette();
}

function factText(value) {
  if (value === null || value === undefined || value === '') return '未知';
  if (typeof value === 'boolean') return value ? '是' : '否';
  if (Array.isArray(value)) return `[${value.join(', ')}]`;
  return String(value);
}

function renderDatasetFacts(step) {
  const alias = String(step.params?.dataset ?? '').trim();
  const facts = alias === 'synthetic' ? syntheticFacts() : state.datasetFacts[alias];
  const section = document.createElement('div');
  section.className = 'inspector-section dataset-facts';
  const heading = document.createElement('h3'); heading.textContent = '数据集信息'; section.appendChild(heading);
  if (!alias) {
    section.appendChild(Object.assign(document.createElement('div'), {textContent: state.datasets.length ? '请选择已登记数据集' : '暂无已登记数据；前往“本地数据集”管理'}));
    return section;
  }
  if (!facts) {
    section.appendChild(Object.assign(document.createElement('div'), {textContent: '正在读取数据集事实…'}));
    return section;
  }
  const rows = [
    ['名称', facts.name], ['路径', facts.path], ['adapter', facts.adapter],
    ['Train 样本数', facts.train_samples], ['Test 样本数', facts.test_samples],
    ['类别数', facts.num_classes], ['输入 shape', facts.input_shape],
    ['Clean target', facts.has_clean_target], ['Noisy target', facts.has_noisy_target],
    ['Sample index', facts.has_sample_index], ['Inspect 状态', facts.layout_validated],
    ['Training verify 状态', facts.training_verified],
  ];
  rows.forEach(([label, value]) => {
    const row = document.createElement('div'); row.className = 'fact-row';
    row.append(Object.assign(document.createElement('span'), {textContent: label}), Object.assign(document.createElement('span'), {textContent: factText(value)}));
    section.appendChild(row);
  });
  return section;
}

function renderNoiseFacts(step) {
  const section = document.createElement('div'); section.className = 'inspector-section';
  const heading = document.createElement('h3'); heading.textContent = '噪声数据流'; section.appendChild(heading);
  const dataset = stepsWithInfo().find(({ info }) => isDatasetSourceInfo(info))?.step;
  const alias = String(dataset?.params?.dataset ?? '').trim();
  const facts = alias === 'synthetic' ? syntheticFacts() : state.datasetFacts[alias];
  const method = String(step.params?.method || 'none');
  section.append(Object.assign(document.createElement('div'), {textContent: `输入：${facts?.has_clean_target === true ? 'clean labels' : '未知'}`}));
  section.append(Object.assign(document.createElement('div'), {textContent: `操作：${method} rate=${step.params?.rate ?? 0}`}));
  section.append(Object.assign(document.createElement('div'), {textContent: `输出：${method === 'external' ? '已有 noisy labels' : method === 'none' ? '原始 labels' : 'noisy labels'}`}));
  return section;
}

function removeStepById(uiId) {
  const location = findParentArrayAndIndex(uiId);
  if (!location) return null;
  const [step] = location.array.splice(location.index, 1);
  return { step, ...location };
}

function insertStep(parentId, index, step) {
  const target = getChildrenArray(parentId);
  if (!target) return false;
  target.splice(Math.max(0, Math.min(index, target.length)), 0, step);
  markDirty();
  return true;
}

function moveStepToTarget(uiId, target) {
  const location = findParentArrayAndIndex(uiId);
  const step = findStepById(uiId);
  const info = blockInfo(step?.block);
  if (!location || !step || !canInsert(info, target.parentId, uiId, target.index)) return false;
  const targetArray = getChildrenArray(target.parentId);
  if (!targetArray) return false;
  location.array.splice(location.index, 1);
  let index = target.index;
  if (location.array === targetArray && location.index < target.index) index -= 1;
  targetArray.splice(Math.max(0, Math.min(index, targetArray.length)), 0, step);
  markDirty();
  state.activeInsertionTarget = { ...target, index };
  return true;
}

function addStepAtTarget(blockId, target) {
  const info = blockInfo(blockId);
  if (!canInsert(info, target?.parentId, null, target?.index || 0)) return false;
  const step = newStep(blockId);
  const available = availableKeysBefore(target);
  (info.requires || []).forEach((name) => {
    const current = slotValue(info, step, name);
    const inferred = inferAvailableSlot(info, name, available);
    if (inferred && !available.has(current)) step.params[name] = inferred;
  });
  (info.provides || []).forEach((name) => {
    const inferred = inferOutputSlot(info, name, target);
    if (inferred) step.params[name] = inferred;
  });
  if (!insertStep(target.parentId, target.index, step)) return false;
  markDirty();
  state.selected = step;
  state.activeInsertionTarget = { ...target, index: target.index + 1 };
  return true;
}

function markDropZones() {
  document.querySelectorAll('.drop-target').forEach((zone) => {
    const target = findDropTarget(zone);
    const movingId = state.drag?.type === 'step' ? state.drag.id : null;
    const info = state.drag?.info || (state.drag?.type === 'new' ? blockInfo(state.drag.id) : null);
    const active = Boolean(state.activeInsertionTarget
      && state.activeInsertionTarget.parentId === target.parentId
      && state.activeInsertionTarget.index === target.index);
    zone.classList.toggle('drop-active', active);
    zone.classList.toggle('active', active);
    zone.classList.toggle('drop-valid', Boolean(info && canInsert(info, target.parentId, movingId, target.index)));
    zone.classList.toggle('drop-invalid', Boolean(info && !canInsert(info, target.parentId, movingId, target.index)));
  });
}

function handleDrop(event, zone) {
  event.preventDefault();
  const target = findDropTarget(zone);
  const newBlockId = event.dataTransfer.getData('application/x-lnl-new-block');
  const stepId = event.dataTransfer.getData('application/x-lnl-step');
  const reason = newBlockId
    ? availabilityReason(blockInfo(newBlockId), target)
    : stepId ? availabilityReason(blockInfo(findStepById(stepId)?.block), target, stepId) : '未识别拖动内容';
  const success = !reason && (newBlockId ? addStepAtTarget(newBlockId, target) : moveStepToTarget(stepId, target));
  state.drag = null;
  if (!success) {
    showMessage(`不能插入到${target.context}层：${reason || '插入失败'}`, 'error');
    markDropZones();
    return;
  }
  draw();
}

function createDropZone(parentId, index, context) {
  const zone = document.createElement('div');
  zone.className = 'drop-target';
  zone.dataset.parentId = parentId;
  zone.dataset.index = String(index);
  zone.dataset.context = context;
  zone.textContent = context === 'batch' ? '+ 把 Batch 内公式拖到这里' : `+ 添加${context === 'epoch' ? ' Epoch 级' : '循环外'}步骤`;
  zone.onclick = (event) => { event.stopPropagation(); setActiveInsertionTarget({ parentId, index, context }); };
  zone.ondragover = (event) => {
    const info = state.drag?.info;
    if (info && canInsert(info, parentId, state.drag.type === 'step' ? state.drag.id : null, index)) {
      event.preventDefault();
      zone.classList.add('drop-hover');
    }
  };
  zone.ondragleave = () => zone.classList.remove('drop-hover');
  zone.ondrop = (event) => { zone.classList.remove('drop-hover'); handleDrop(event, zone); };
  return zone;
}

function renderPalette() {
  const palette = $('palette');
  palette.innerHTML = '';
  const query = state.paletteQuery.trim().toLowerCase();
  const visible = state.blocks.filter((item) => item.beginner_visible && (!query || [
    item.name, item.id, item.category, item.description,
    ...(item.requires || []), ...(item.provides || []),
  ].join(' ').toLowerCase().includes(query)));
  const categories = [...new Set(visible.map((item) => item.category))];
  if (!categories.length) {
    palette.textContent = query ? '没有匹配的积木' : '暂无可见积木';
    return;
  }
  categories.forEach((category) => {
    const title = document.createElement('h3');
    title.className = 'palette-category-title';
    title.dataset.category = category;
    const collapsed = state.paletteCollapsed.has(category) && !query;
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'palette-category-toggle';
    toggle.setAttribute('aria-expanded', String(!collapsed));
    toggle.title = collapsed ? `展开 ${category}` : `折叠 ${category}`;
    const indicator = document.createElement('span');
    indicator.className = 'palette-category-indicator';
    indicator.textContent = collapsed ? '▸' : '▾';
    const label = document.createElement('span');
    label.textContent = `${category} (${visible.filter((item) => item.category === category).length})`;
    toggle.append(indicator, label);
    toggle.onclick = () => {
      if (state.paletteCollapsed.has(category)) state.paletteCollapsed.delete(category);
      else state.paletteCollapsed.add(category);
      renderPalette();
    };
    title.appendChild(toggle);
    palette.appendChild(title);
    if (collapsed) return;
    visible.filter((item) => item.category === category).forEach((item) => {
      const node = document.createElement('div');
      node.className = 'palette-block';
      node.dataset.category = blockCategory(item);
      node.classList.toggle('palette-selected', Boolean(state.paletteSelection?.id === item.id && !state.selected));
      // Placement and required slots are validated against the concrete drop
      // zone, not the last selected insertion target.
      node.draggable = true;
      node.setAttribute('aria-disabled', 'false');
      node.textContent = item.name;
      const clickReason = availabilityReason(item, state.activeInsertionTarget);
      node.title = clickReason
        ? `${item.description}\n点击插入不可用：${clickReason}\n可拖动到合法的绿色插入位置`
        : `${item.description}\n点击插入到当前插入位置，或拖动到任意合法插入位置`;
      node.ondragstart = (event) => {
        state.paletteSelection = item;
        state.selected = null;
        renderInspector();
        state.drag = { type: 'new', id: item.id, info: item };
        event.dataTransfer.setData('application/x-lnl-new-block', item.id);
        event.dataTransfer.effectAllowed = 'copy';
        markDropZones();
      };
      node.ondragend = () => { state.drag = null; markDropZones(); };
      node.onclick = () => {
        // A palette click should always expose the block contract, even when
        // the current insertion target cannot accept that block.
        state.paletteSelection = item;
        state.selected = null;
        renderInspector();
        const target = state.activeInsertionTarget;
        const clickReason = availabilityReason(item, target);
        if (clickReason || !addStepAtTarget(item.id, target)) { showMessage(`不能插入：${clickReason || '插入失败'}`, 'error'); return; }
        state.paletteSelection = null;
        draw();
      };
      palette.appendChild(node);
    });
  });
}

function renderStepSummary(step, info) {
  if (!info) return '';
  const lines = [];
  if (info.formula) lines.push(info.formula);
  const values = Object.entries(info.params || {}).map(([name, schema]) => {
    const configured = Object.prototype.hasOwnProperty.call(step.params || {}, name)
      ? step.params[name] : schema.default;
    return `${name}=${formatParamValue(configured)}`;
  });
  if (values.length) lines.push(`参数：${values.join(' · ')}`);
  const outputs = (info.provides || []).map((name) => slotValue(info, step, name));
  if (outputs.length) lines.push(`→ ${outputs.join(', ')}`);
  return lines.join('\n');
}

function formatParamValue(value) {
  if (value === undefined) return '未设置';
  if (value === null) return 'null';
  if (typeof value === 'object') {
    try { return JSON.stringify(value); } catch (error) { return String(value); }
  }
  return String(value);
}

function flatRecipeSteps(steps = state.recipe.steps) {
  return (steps || []).flatMap((step) => [step, ...(Array.isArray(step.steps) ? flatRecipeSteps(step.steps) : [])]);
}

function getNextSuggestion() {
  const entries = stepsWithInfo();
  const hasOutput = (slot) => entries.some(({ step, info }) =>
    (info?.provides || []).some((name) => slotValue(info, step, name) === slot));
  const hasId = (id) => entries.some(({ info }) => info?.id === id);
  const dataset = entries.find(({ info }) => isDatasetSourceInfo(info))?.step;
  if (!dataset || !String(dataset.params?.dataset ?? '').trim()) return '下一步：选择数据集';
  if (!hasOutput('noise_state')) return '下一步：决定是否添加标签噪声';
  if (!hasId('create_model')) return '下一步：确认模型';
  if (!hasId('create_optimizer') && !hasId('create_parameter_group_optimizer')) return '下一步：确认优化器';
  if (!hasId('epoch_loop') || !hasId('batch_loop')) return '下一步：建立 Epoch / Batch Loop';
  if (!hasId('forward') && !hasId('forward_feature')) return '下一步：进入 Batch Loop 添加 Forward';
  if (!hasId('mean_loss')) return '下一步：添加概率/损失公式';
  if (!hasId('backward')) return '下一步：添加 Backward';
  if (!hasId('optimizer_step')) return '下一步：添加 Optimizer Step';
  return '下一步：检查 recipe，然后运行 smoke';
}

function drawSteps(steps = state.recipe.steps, parent = $('steps'), parentId = '__root__', context = 'top') {
  parent.innerHTML = '';
  (steps || []).forEach((step, index) => {
    parent.appendChild(createDropZone(parentId, index, context));
    const info = blockInfo(step.block);
    const node = document.createElement('div');
    const loopClass = info && info.kind !== 'action' ? `loop-container loop-block ${step.block === 'epoch_loop' ? 'epoch-loop' : step.block === 'batch_loop' ? 'batch-loop' : ''}` : '';
    const formulaClass = info?.formula ? 'formula-block' : '';
    node.className = `step ${loopClass} ${formulaClass}${state.selected === step ? ' selected' : ''}${state.errorStepId === step._uiId ? ' error-step' : ''}`;
    node.dataset.category = blockCategory(info);
    node.dataset.uiId = step._uiId;
    const header = document.createElement('div');
    header.className = 'step-header';
    const title = document.createElement('span'); title.textContent = info ? info.name : step.block; header.appendChild(title);
    const kind = document.createElement('span'); kind.className = 'kind'; kind.textContent = info ? info.kind : ''; header.appendChild(kind);
    const summary = renderStepSummary(step, info);
    node.appendChild(header);
    if (summary) { const summaryNode = document.createElement('div'); summaryNode.className = 'step-summary'; summaryNode.textContent = summary; node.appendChild(summaryNode); }
    // Selecting a step should work from any non-control area of the block,
    // not only from the title/header text.  Stop propagation so a nested
    // block selects itself rather than its enclosing loop.
    node.onclick = (event) => {
      if (event.target.closest('button, input, select, textarea')) return;
      event.stopPropagation();
      state.selected = step;
      state.paletteSelection = null;
      renderInspector();
      draw();
    };
    node.draggable = true;
    node.onmousedown = (event) => {
      if (event.button !== 0) return;
      const startX = event.clientX;
      const startY = event.clientY;
      let moved = false;
      const onMove = (moveEvent) => {
        if (!moved && Math.hypot(moveEvent.clientX - startX, moveEvent.clientY - startY) < 5) return;
        moved = true;
        state.drag = { type: 'step', id: step._uiId, info };
        const zone = document.elementFromPoint(moveEvent.clientX, moveEvent.clientY)?.closest('.drop-target');
        document.querySelectorAll('.drop-target.drop-hover').forEach((item) => item.classList.remove('drop-hover'));
        if (zone) {
          const target = findDropTarget(zone);
          if (canInsert(info, target.parentId, step._uiId, target.index)) zone.classList.add('drop-hover');
        }
        markDropZones();
        moveEvent.preventDefault();
      };
      const onUp = (upEvent) => {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        if (!moved) { state.drag = null; return; }
        const zone = document.elementFromPoint(upEvent.clientX, upEvent.clientY)?.closest('.drop-target');
        const target = zone ? findDropTarget(zone) : null;
        const reason = target ? availabilityReason(info, target, step._uiId) : '未放在有效 drop zone';
        const success = Boolean(target && !reason && moveStepToTarget(step._uiId, target));
        state.drag = null;
        if (!success) showMessage(`不能拖动：${reason || '插入失败'}`, 'error');
        draw();
        upEvent.preventDefault();
      };
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    };
    node.ondragstart = (event) => {
      state.drag = { type: 'step', id: step._uiId, info };
      event.dataTransfer.setData('application/x-lnl-step', step._uiId);
      event.dataTransfer.effectAllowed = 'move';
      markDropZones();
    };
    node.ondragend = () => { state.drag = null; markDropZones(); };
    if (info && info.kind !== 'action') {
      const toggle = document.createElement('button');
      toggle.textContent = state.collapsed.has(step) ? '展开' : '折叠';
      toggle.onclick = (event) => { event.stopPropagation(); if (state.collapsed.has(step)) state.collapsed.delete(step); else state.collapsed.add(step); draw(); };
      header.appendChild(toggle);
      if (!state.collapsed.has(step)) {
        const nested = document.createElement('div');
        nested.className = 'nested loop-body';
        const childContext = step.block === 'epoch_loop' ? 'epoch' : step.block === 'batch_loop' ? 'batch' : 'any';
        drawSteps(step.steps || (step.steps = []), nested, step._uiId, childContext);
        node.appendChild(nested);
      }
    }
    const copy = document.createElement('button');
    copy.textContent = '复制';
    copy.onclick = (event) => {
      event.stopPropagation();
      const clone = typeof structuredClone === 'function' ? structuredClone(step) : JSON.parse(JSON.stringify(step));
      ensureUiIds([clone]);
      clone._uiId = makeUiId();
      (steps || []).splice(index + 1, 0, clone);
      markDirty();
      state.selected = clone;
      draw();
    };
    const remove = document.createElement('button');
    remove.textContent = '删除';
    remove.onclick = (event) => {
      event.stopPropagation();
      (steps || []).splice(index, 1);
      markDirty();
      if (state.selected === step) state.selected = null;
      state.paletteSelection = null;
      state.activeInsertionTarget = { parentId, index, context };
      draw();
    };
    node.append(copy, remove);
    parent.appendChild(node);
  });
  parent.appendChild(createDropZone(parentId, (steps || []).length, context));
  markDropZones();
}

function renderInspector() {
  const target = $('inspector');
  const explanationTarget = $('module-explanation');
  target.innerHTML = '';
  explanationTarget.innerHTML = '';
  const paletteInfo = !state.selected ? state.paletteSelection : null;
  const previewStep = paletteInfo ? {
    block: paletteInfo.id,
    params: Object.fromEntries(Object.entries(paletteInfo.params || {}).map(([name, schema]) => [name, schema.default ?? ''])),
  } : null;
  const step = state.selected || previewStep;
  if (!step) { explanationTarget.textContent = '点击一个积木查看说明'; target.textContent = '选择积木后可编辑参数'; return; }
  const info = paletteInfo || blockInfo(step.block);
  if (!info) { explanationTarget.textContent = '暂无模块说明'; target.textContent = '无法加载该积木参数'; return; }
  if (state.errorMessage) {
    const error = document.createElement('div'); error.className = 'inspector-error'; error.textContent = state.errorMessage; target.appendChild(error);
  }
  const addSection = (title, content) => {
    const section = document.createElement('div');
    section.className = 'inspector-section';
    const heading = document.createElement('h3'); heading.textContent = title; section.appendChild(heading);
    if (typeof content === 'string') {
      const text = document.createElement('div'); text.textContent = content; section.appendChild(text);
    } else section.appendChild(content);
    target.appendChild(section);
  };
  const identity = document.createElement('div');
  const name = document.createElement('strong'); name.textContent = info.name; identity.appendChild(name);
  const description = document.createElement('p'); description.textContent = info.description || '暂无说明'; identity.appendChild(description);
  explanationTarget.appendChild(identity);

  const formula = document.createElement('div');
  const formulaText = document.createElement('div'); formulaText.textContent = info.formula || '暂无独立公式'; formula.className = 'formula'; formula.appendChild(formulaText);
  if (info.formula_ref) { const ref = document.createElement('small'); ref.textContent = `定义来源：${info.formula_ref}`; formula.appendChild(ref); }
  if (info.paper) { const paper = document.createElement('div'); paper.textContent = `论文：${info.paper}`; formula.appendChild(paper); }
  addSection('2. 公式', formula);

  const io = document.createElement('div');
  const inputs = document.createElement('div'); inputs.textContent = `输入：${(info.requires || []).map((name) => slotValue(info, step, name)).join(', ') || '无'}`; io.appendChild(inputs);
  const outputs = document.createElement('div'); outputs.textContent = `输出：${(info.provides || []).map((name) => slotValue(info, step, name)).join(', ') || '无'}`; io.appendChild(outputs);
  addSection('3. 输入 / 输出', io);

  const params = document.createElement('div');
  if (paletteInfo) {
    info.params && Object.entries(info.params).forEach(([name, schema]) => {
      if (isDatasetSourceInfo(info) && name === 'path' && step.params?.source_mode !== 'custom_path') return;
      const wrap = document.createElement('div'); wrap.className = 'param';
      const label = document.createElement('label'); label.textContent = name; wrap.appendChild(label);
      const value = document.createElement('span'); value.className = 'param-preview';
      value.textContent = formatParamValue(step.params[name] ?? schema.default);
      wrap.appendChild(value);
      params.appendChild(wrap);
    });
  } else {
    info.params && Object.entries(info.params).forEach(([name, schema]) => {
      if (isDatasetSourceInfo(info) && name === 'path' && step.params?.source_mode !== 'custom_path') return;
      const wrap = document.createElement('div'); wrap.className = 'param';
      const label = document.createElement('label'); label.textContent = name; wrap.appendChild(label);
      wrap.appendChild(renderParamControl(name, schema, step));
      params.appendChild(wrap);
    });
  }
  addSection('4. 参数', params);
  if (!paletteInfo && isDatasetSourceInfo(info)) target.appendChild(renderDatasetFacts(step));
  if (!paletteInfo && (info.provides || []).some((name) => name === 'noise_state' || name === 'noisy_train_split')) target.appendChild(renderNoiseFacts(step));
}

function draw() {
  $('recipe-name').value = state.recipe.name;
  renderRuntimeLimits();
  $('next-suggestion').textContent = getNextSuggestion();
  renderPalette();
  drawSteps();
  renderInspector();
  updateRunState();
  if (!state.lastRun) showMessage(targetLabel(state.activeInsertionTarget));
  else updateWorkflowGuide();
}

function formulaCandidates() {
  return state.blocks.filter((info) => info.beginner_visible
    && !info.paper
    && info.kind === 'action'
    && info.formula_safe
    && (info.placement || []).some((place) => place === 'batch' || place === 'any')
    && (info.formula || Object.keys(info.params || {}).some((name) => name === 'save_as' || name.endsWith('_as'))));
}

function formulaTarget() {
  return state.activeInsertionTarget?.context === 'batch' ? state.activeInsertionTarget : null;
}

function formulaFieldValue(input, schema) {
  if (schema.type === 'bool') return input.checked;
  if (schema.type === 'int') return Number.parseInt(input.value, 10);
  if (schema.type === 'float') return Number.parseFloat(input.value);
  if (schema.type === 'value') {
    const raw = input.value.trim();
    if (!raw) return null;
    try { return JSON.parse(raw); } catch (error) { throw new Error(`参数 ${input.dataset.formulaParam} 必须是 JSON 值`); }
  }
  return input.value;
}

function formulaSlotOptions(target, schema) {
  const values = [...availableKeysBefore(target)];
  if (schema.default && !values.includes(schema.default)) values.push(schema.default);
  return values.sort();
}

function appendFormulaField(container, name, schema, target) {
  const wrap = document.createElement('div');
  wrap.className = 'formula-field';
  const label = document.createElement('label');
  label.textContent = isOutputSlot(name) ? `${name}（输出 slot）` : name;
  wrap.appendChild(label);
  const current = schema.default ?? (schema.type === 'value' ? [] : '');
  let input;
  if (schema.type === 'enum') {
    input = document.createElement('select');
    (schema.options || []).forEach((optionValue) => {
      const option = document.createElement('option');
      option.value = optionValue;
      option.textContent = optionValue;
      input.appendChild(option);
    });
    input.value = String(current);
  } else if (schema.type === 'bool') {
    input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = Boolean(current);
  } else if (schema.type === 'int' || schema.type === 'float') {
    input = document.createElement('input');
    input.type = 'number';
    input.value = current;
    input.step = schema.type === 'int' ? '1' : 'any';
    if (schema.min !== undefined) input.min = schema.min;
    if (schema.max !== undefined) input.max = schema.max;
  } else if (schema.type === 'value') {
    input = document.createElement('textarea');
    input.value = typeof current === 'string' ? current : JSON.stringify(current);
    input.placeholder = '例如： ["loss_a", "loss_b"] 或 0.5';
  } else {
    input = document.createElement('input');
    input.type = 'text';
    input.value = current;
    if (schema.type === 'slot') {
      const listId = `formula-slots-${name}`;
      input.setAttribute('list', listId);
      const list = document.createElement('datalist');
      list.id = listId;
      formulaSlotOptions(target, schema).forEach((value) => {
        const option = document.createElement('option'); option.value = value; list.appendChild(option);
      });
      wrap.appendChild(list);
    }
  }
  input.dataset.formulaParam = name;
  input.dataset.formulaType = schema.type || 'value';
  wrap.appendChild(input);
  if (schema.type === 'slot') {
    const hint = document.createElement('small');
    hint.textContent = '可填写上游 slot；下拉建议来自当前 Batch 插入位置之前的输出。';
    wrap.appendChild(hint);
  }
  container.appendChild(wrap);
}

function renderFormulaBuilderFields() {
  const select = $('formula-operation');
  const preview = $('formula-operation-preview');
  const targetText = $('formula-target');
  const fields = $('formula-fields');
  const addButton = $('formula-add');
  const candidates = formulaCandidates();
  const selectedId = select.value;
  select.innerHTML = '';
  candidates.forEach((info) => {
    const option = document.createElement('option');
    option.value = info.id;
    option.textContent = `${info.name} · ${info.category}`;
    select.appendChild(option);
  });
  if (candidates.some((info) => info.id === selectedId)) select.value = selectedId;
  const info = blockInfo(select.value) || candidates[0];
  if (!info) {
    preview.textContent = '暂无可组合的公共 Batch 公式积木。';
    targetText.textContent = '请先加载积木 Registry。';
    targetText.className = 'formula-target invalid';
    fields.innerHTML = '';
    addButton.disabled = true;
    return;
  }
  const target = formulaTarget();
  preview.textContent = `${info.description || ''}\n公式：${info.formula || '该运算没有单独公式标注，但可作为公式链的一步。'}\n输入：${(info.requires || []).join(', ') || '无'}`;
  if (target) {
    targetText.textContent = `目标：Batch Loop，第 ${target.index + 1} 步（将生成 ${info.name}）`;
    targetText.className = 'formula-target';
  } else {
    targetText.textContent = '请先在中间区域点击 Batch 内的绿色插入位置，再打开公式组合器。';
    targetText.className = 'formula-target invalid';
  }
  fields.innerHTML = '';
  Object.entries(info.params || {}).forEach(([name, schema]) => appendFormulaField(fields, name, schema, target));
  addButton.disabled = !target;
}

function formulaParamsFromDialog(info) {
  const params = {};
  $('formula-fields').querySelectorAll('[data-formula-param]').forEach((input) => {
    const name = input.dataset.formulaParam;
    params[name] = formulaFieldValue(input, info.params[name]);
  });
  return params;
}

function formulaInsertionReason(info, target, params) {
  if (!target) return '请先选择 Batch 内的绿色插入位置';
  if (!placementAllows(info, target.parentId)) return `placement 不允许放入 ${target.context} 层`;
  const available = availableKeysBefore(target);
  const missing = (info.requires || []).map((name) => {
    const configured = params[name] ?? info.params?.[name]?.default ?? name;
    return [name, configured];
  }).filter(([, slot]) => typeof slot !== 'string' || !available.has(slot));
  if (missing.length) return `前置 slot 不完整：${missing.map(([name, slot]) => `${name}=${slot}`).join(', ')}`;
  if (info.id === 'weighted_sum' && Array.isArray(params.terms)) {
    const unknown = params.terms.filter((slot) => typeof slot !== 'string' || !available.has(slot));
    if (unknown.length) return `Weighted Sum 找不到 term slot：${unknown.join(', ')}`;
  }
  return null;
}

function addFormulaFromDialog() {
  const info = blockInfo($('formula-operation').value);
  const target = formulaTarget();
  if (!info || !target) { renderFormulaBuilderFields(); return; }
  let params;
  try { params = formulaParamsFromDialog(info); } catch (error) { showMessage(error.message, 'error'); return; }
  const reason = formulaInsertionReason(info, target, params);
  if (reason) { $('formula-target').textContent = reason; $('formula-target').className = 'formula-target invalid'; showMessage(reason, 'error'); return; }
  const step = newStep(info.id);
  step.params = params;
  insertStep(target.parentId, target.index, step);
  state.selected = step;
  state.paletteSelection = null;
  state.activeInsertionTarget = { ...target, index: target.index + 1 };
  $('formula-dialog').close();
  draw();
}

function formulaEditorCandidates() {
  return state.blocks.filter((info) => info.formula_safe && !info.paper && info.kind === 'action')
    .sort((a, b) => `${a.formula_group || a.category}:${a.name}`.localeCompare(`${b.formula_group || b.category}:${b.name}`));
}

function parseEditorInputs() {
  return String($('formula-inputs').value || '').split(/[\n,]/).map((item) => item.trim()).filter(Boolean);
}

function parseEditorParameters() {
  const result = {};
  String($('formula-parameters').value || '').split(/\n/).map((item) => item.trim()).filter(Boolean).forEach((line) => {
    const match = line.match(/^([A-Za-z][A-Za-z0-9_]*)\s*(?::\s*([A-Za-z]+))?\s*=\s*(.+)$/);
    if (!match) throw new Error(`参数格式错误：${line}，应为 name:type=default`);
    const [, name, type = 'float', raw] = match;
    let value = raw.trim();
    if (type === 'float') value = Number.parseFloat(value);
    else if (type === 'int') value = Number.parseInt(value, 10);
    else if (type === 'bool') value = value === 'true';
    else if (type === 'value') { try { value = JSON.parse(value); } catch (error) { throw new Error(`参数 ${name} 的 value 必须是 JSON`); } }
    if ((type === 'float' || type === 'int') && !Number.isFinite(value)) throw new Error(`参数 ${name} 的默认值无效`);
    result[name] = { type, default: value };
  });
  return result;
}

function formulaEditorSources() {
  const sources = [...parseEditorInputs()];
  try { sources.push(...Object.keys(parseEditorParameters())); } catch (error) { /* save reports the format error */ }
  sources.push(...state.formulaEditor.steps.map((step) => step.id));
  return [...new Set(sources)];
}

function renderFormulaEditorBindingFields() {
  const container = $('formula-editor-binding-fields');
  const info = blockInfo($('formula-editor-block')?.value);
  if (!container || !info) return;
  container.innerHTML = '';
  const sources = formulaEditorSources();
  Object.entries(info.params || {}).forEach(([name, schema]) => {
    if (schema.type !== 'slot' || isOutputSlot(name)) return;
    const label = document.createElement('label');
    label.textContent = `${name}${(info.requires || []).includes(name) ? '（必需）' : ''}`;
    const select = document.createElement('select');
    select.dataset.formulaBinding = name;
    const configured = schema.default && sources.includes(schema.default) ? schema.default : '';
    const options = [...new Set(['', ...sources, configured].filter(Boolean))];
    options.forEach((source) => {
      const option = document.createElement('option');
      option.value = source;
      option.textContent = source;
      option.selected = source === configured;
      select.appendChild(option);
    });
    label.appendChild(select);
    container.appendChild(label);
  });
  if (!container.children.length) container.textContent = '该操作没有需要连接的输入插口。';
}

function renderFormulaEditor() {
  const select = $('formula-editor-block');
  if (!select) return;
  const candidates = formulaEditorCandidates();
  const current = select.value;
  select.innerHTML = '';
  candidates.forEach((info) => {
    const option = document.createElement('option'); option.value = info.id; option.textContent = `${info.name} · ${info.formula_group || info.category}`; select.appendChild(option);
  });
  if (candidates.some((info) => info.id === current)) select.value = current;
  renderFormulaEditorBindingFields();
  const list = $('formula-editor-step-list'); list.innerHTML = '';
  state.formulaEditor.steps.forEach((step, index) => {
    const row = document.createElement('div'); row.className = 'formula-step-item';
    const label = document.createElement('span'); label.innerHTML = `${index + 1}. <code>${step.id}</code> ← ${step.block}`;
    const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = '删除'; remove.onclick = () => { state.formulaEditor.steps.splice(index, 1); renderFormulaEditor(); };
    row.append(label, remove); list.appendChild(row);
  });
  const output = $('formula-output-source'); output.innerHTML = '';
  state.formulaEditor.steps.forEach((step) => { const option = document.createElement('option'); option.value = step.id; option.textContent = step.id; output.appendChild(option); });
  const last = state.formulaEditor.steps.at(-1)?.id || '';
  if (last) output.value = output.value || last;
  const preview = state.formulaEditor.steps.length
    ? `公式预览：\n${state.formulaEditor.steps.map((step, index) => `${index + 1}. ${step.id} = ${step.block}`).join('\n')}`
    : '公式预览：尚未添加步骤';
  $('formula-preview').textContent = preview;
}

function resetFormulaEditor() {
  state.formulaEditor.steps = []; state.formulaEditor.editingId = null;
  $('formula-id').value = 'user/my_formula'; $('formula-name').value = 'My Formula'; $('formula-description').value = '';
  $('formula-inputs').value = 'logits\ntargets'; $('formula-parameters').value = 'epsilon:float=1e-8'; $('formula-output-name').value = 'loss';
  $('formula-editor-step-id').value = ''; $('formula-editor-step-parameters').value = '';
  renderFormulaEditor();
  $('formula-editor-validation').textContent = '';
}

function loadFormulaIntoEditor(item) {
  $('formula-list-dialog')?.close();
  state.formulaEditor.editingId = item.id;
  state.formulaEditor.steps = (item.steps || []).map((step) => ({ id: step.id, block: step.block, bindings: {...(step.bindings || {})}, parameters: {...(step.parameters || {})} }));
  $('formula-id').value = item.id; $('formula-name').value = item.name || ''; $('formula-description').value = item.description || '';
  $('formula-inputs').value = Object.keys(item.inputs || {}).join('\n');
  $('formula-parameters').value = Object.entries(item.parameters || {}).map(([name, schema]) => `${name}:${schema.type || 'float'}=${JSON.stringify(schema.default)}`).join('\n');
  const outputName = Object.keys(item.outputs || {})[0] || 'loss';
  $('formula-output-name').value = outputName;
  renderFormulaEditor(); $('formula-output-source').value = item.outputs?.[outputName]?.source?.split('.')[0] || '';
  $('formula-editor-validation').textContent = '';
  $('formula-editor-dialog').showModal();
}

function addFormulaEditorStep() {
  const block = $('formula-editor-block').value;
  const info = blockInfo(block);
  const id = $('formula-editor-step-id').value.trim();
  if (!info || !id) { $('formula-editor-validation').textContent = '步骤需要选择公共 Formula-safe Block 并填写唯一 step id'; return; }
  if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(id) || state.formulaEditor.steps.some((step) => step.id === id)) { $('formula-editor-validation').textContent = 'step id 必须唯一且为字母/数字/下划线'; return; }
  const bindings = {};
  $('formula-editor-binding-fields').querySelectorAll('[data-formula-binding]').forEach((input) => {
    if (input.value) bindings[input.dataset.formulaBinding] = input.value;
  });
  let parameters = {};
  const rawParameters = $('formula-editor-step-parameters').value.trim();
  if (rawParameters) { try { parameters = JSON.parse(rawParameters); } catch (error) { $('formula-editor-validation').textContent = 'step parameters 必须是 JSON 对象'; return; } }
  state.formulaEditor.steps.push({ id, block, bindings, parameters });
  $('formula-editor-step-id').value = ''; $('formula-editor-step-parameters').value = '';
  $('formula-editor-validation').textContent = '';
  renderFormulaEditor();
}

async function saveFormulaEditor() {
  const errorTarget = $('formula-editor-validation');
  try {
    const id = $('formula-id').value.trim(); const name = $('formula-name').value.trim();
    const inputs = Object.fromEntries(parseEditorInputs().map((input) => [input, { description: '' }]));
    const parameters = parseEditorParameters();
    const outputName = $('formula-output-name').value.trim(); const outputSource = $('formula-output-source').value;
    if (!outputName || !outputSource) throw new Error('请指定公式输出及其来源');
    const formula = { id, name, description: $('formula-description').value.trim(), inputs, parameters, steps: state.formulaEditor.steps, outputs: { [outputName]: { source: outputSource } }, metadata: { origin: 'scratch-web' } };
    const result = await api('/api/formulas', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ formula, replace: Boolean(state.formulaEditor.editingId) }) });
    await refreshBlocksAfterFormulaChange();
    $('formula-editor-dialog').close();
    state.formulaEditor.editingId = null;
    showMessage(`公式已保存：${result.formula.name}（${result.formula.id}）`, 'ok');
  } catch (error) { errorTarget.textContent = error.message; }
}

async function refreshBlocksAfterFormulaChange() {
  state.blocks = await api('/api/blocks');
  draw();
}

async function showMyFormulas() {
  try {
    const formulas = await api('/api/formulas');
    state.blocks = await api('/api/blocks');
    renderPalette();
    const list = $('formula-list'); list.innerHTML = '';
    formulas.filter((item) => item.id.startsWith('user/')).forEach((item) => {
      const row = document.createElement('div'); row.className = 'formula-list-item';
      const text = document.createElement('span'); text.innerHTML = `<strong>${item.name}</strong><small>${item.id} · ${item.formula_hash.slice(0, 12)}</small>`;
      const actions = document.createElement('span');
      const editButton = document.createElement('button'); editButton.textContent = '编辑'; editButton.onclick = () => loadFormulaIntoEditor(item);
      const copyButton = document.createElement('button'); copyButton.textContent = '复制'; copyButton.onclick = async () => { const copy = {...item, id: item.id + '_copy', name: `${item.name} Copy`}; delete copy.formula_hash; delete copy.block_id; await api('/api/formulas', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({formula: copy}) }); await refreshBlocksAfterFormulaChange(); showMyFormulas(); };
      const renameButton = document.createElement('button'); renameButton.textContent = '重命名'; renameButton.onclick = async () => { const next = window.prompt('新的 Formula ID（user/xxx）', item.id); if (!next || next === item.id) return; await api('/api/formula/rename', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({old_id: item.id, new_id: next}) }); await refreshBlocksAfterFormulaChange(); showMyFormulas(); };
      const exportButton = document.createElement('button'); exportButton.textContent = '导出'; exportButton.onclick = async () => { const value = await api('/api/formula/' + encodeURIComponent(item.id) + '/export'); const blob = new Blob([value], {type: 'text/yaml'}); const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = item.id.split('/').pop() + '.yaml'; link.click(); URL.revokeObjectURL(link.href); };
      const deleteButton = document.createElement('button'); deleteButton.textContent = '删除'; deleteButton.onclick = async () => { if (!window.confirm(`删除 ${item.name}？`)) return; await api('/api/formula/delete', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({id: item.id}) }); await refreshBlocksAfterFormulaChange(); showMyFormulas(); };
      actions.append(editButton, copyButton, renameButton, exportButton, deleteButton); row.append(text, actions); list.appendChild(row);
    });
    if (!list.children.length) list.textContent = '暂无用户公式；点击“新建公式”创建。';
    $('formula-list-dialog').showModal();
  } catch (error) { showError(error); }
}

async function loadDefaultRecipe() {
  state.recipe = await api('/api/default-recipe');
  ensureUiIds(state.recipe.steps);
  state.selected = null;
  state.paletteSelection = null;
  state.lastRun = null;
  state.validated = false;
  state.activeInsertionTarget = defaultInsertionTarget();
  draw();
}

function payload() {
  state.recipe.name = $('recipe-name').value.trim() || 'scratch_recipe';
  return { recipe: stripUiFields(state.recipe) };
}

function runPayload() {
  return { ...payload(), runtime_limits: state.runtimeLimits };
}

function highlightError(message) {
  const path = [...String(message).matchAll(/step (\d+)/g)].map((match) => Number(match[1]) - 1);
  let steps = state.recipe.steps; let target = null;
  path.forEach((index, depth) => {
    if (steps && steps[index]) {
      target = steps[index];
      if (depth < path.length - 1) state.collapsed.delete(target);
      steps = target.steps || [];
    }
  });
  if (target) {
    state.selected = target;
    state.errorStepId = target._uiId;
    draw();
    document.querySelector(`[data-ui-id="${target._uiId}"]`)?.scrollIntoView({block: 'center'});
  }
}
function showError(error) { state.errorMessage = error.message; highlightError(error.message); showMessage(error.message, 'error'); }

function renderTemplateList(templates, query = '') {
  const list = $('template-list');
  list.innerHTML = '';
  const needle = query.trim().toLowerCase();
  templates.filter((item) => !needle || `${item.name} ${item.id} ${item.status}`.toLowerCase().includes(needle)).forEach((item) => {
    const card = document.createElement('div'); card.className = 'template-card';
    const name = document.createElement('strong'); name.textContent = item.name; card.appendChild(name);
    const status = document.createElement('small');
    status.textContent = templateStatusLabel(item.status);
    card.appendChild(status);
    const open = document.createElement('button'); open.textContent = '打开';
    open.onclick = () => openTemplate(item);
    card.appendChild(open); list.appendChild(card);
  });
  if (!list.children.length) list.textContent = '没有匹配的模板';
}

function templateStatusLabel(status) {
  return status === 'formula-ready' ? '公式积木已展开' : status === 'template-ready' ? '论文模板' : '旧版积木模板，尚未逐公式展开';
}

function renderFeaturedExamples(examples) {
  const list = $('featured-examples');
  list.innerHTML = '';
  examples.forEach((item) => {
    const card = document.createElement('div'); card.className = 'template-card featured-example';
    const name = document.createElement('strong'); name.textContent = `${item.name} 论文范例`; card.appendChild(name);
    const status = document.createElement('small'); status.textContent = `${templateStatusLabel(item.status)}，可直接查看和运行`; card.appendChild(status);
    const open = document.createElement('button'); open.textContent = '打开范例'; open.onclick = () => openTemplate(item);
    card.appendChild(open); list.appendChild(card);
  });
}

async function openTemplate(item) {
  try {
    state.recipe = await api('/api/recipe/' + encodeURIComponent(item.path));
    ensureUiIds(state.recipe.steps);
    state.selected = null;
    state.paletteSelection = null;
    state.lastRun = null;
    state.validated = false;
    state.activeInsertionTarget = defaultInsertionTarget();
    $('template-dialog').close();
    draw();
  } catch (error) { showError(error); }
}

function openOnboarding() {
  const dialog = $('onboarding-dialog');
  if (dialog && typeof dialog.showModal === 'function') dialog.showModal();
}

$('help').onclick = openOnboarding;
$('open-tutorial').onclick = openOnboarding;
$('tutorial-done').onclick = () => $('onboarding-dialog').close();
$('dismiss-guide').onclick = () => {
  $('first-use-guide').classList.add('is-hidden');
  try { localStorage.setItem('lnl-scratch-first-use-guide-dismissed', '1'); } catch (error) { /* storage is optional */ }
};
try {
  if (localStorage.getItem('lnl-scratch-first-use-guide-dismissed') === '1') $('first-use-guide').classList.add('is-hidden');
} catch (error) { /* storage is optional */ }

$('new').onclick = async () => { try { await loadDefaultRecipe(); } catch (error) { showError(error); } };
$('check').onclick = async () => {
  try {
    await api('/api/validate', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload()) });
    state.validated = true;
    updateRunState();
    const dataReady = recipeDataReady();
    showMessage(dataReady ? '检查通过，已允许运行' : '结构检查通过，但请先选择有效数据集', dataReady ? 'ok' : 'error');
  } catch (error) {
    state.validated = false;
    updateRunState();
    showError(error);
  }
};
$('save').onclick = async () => { try { const result = await api('/api/save', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload()) }); showMessage(`已保存: ${result.path}`); } catch (error) { showError(error); } };
$('run').onclick = async () => {
  if ($('run').disabled) return;
  setResultState('运行中…');
  if ($('result-summary')) $('result-summary').textContent = '正在执行结构验证，请稍候…';
  try {
    const result = await api('/api/run', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(runPayload()) });
    renderRunResult(result);
  } catch (error) { state.lastRun = null; showError(error); }
};
$('open').onclick = async () => { try { const names = await api('/api/recipes'); showMessage(`我的 Recipe：${names.join('、') || '暂无已保存 Recipe'}`); } catch (error) { showError(error); } };
$('templates').onclick = async () => {
  try {
    const [templates, examples] = await Promise.all([api('/api/templates'), api('/api/examples')]);
    renderFeaturedExamples(examples);
    renderTemplateList(templates);
    $('template-search').value = '';
    $('template-search').oninput = () => renderTemplateList(templates, $('template-search').value);
    $('template-dialog').showModal();
  } catch (error) { showError(error); }
};
$('formula-builder').onclick = () => {
  renderFormulaBuilderFields();
  $('formula-dialog').showModal();
};
$('formula-operation').onchange = renderFormulaBuilderFields;
$('formula-add').onclick = addFormulaFromDialog;
$('formula-cancel').onclick = () => $('formula-dialog').close();
$('new-formula').onclick = () => { resetFormulaEditor(); $('formula-editor-dialog').showModal(); };
$('formula-editor-block').onchange = renderFormulaEditor;
$('formula-inputs').oninput = renderFormulaEditorBindingFields;
$('formula-parameters').oninput = renderFormulaEditorBindingFields;
$('formula-editor-add-step').onclick = addFormulaEditorStep;
$('formula-editor-save').onclick = saveFormulaEditor;
$('formula-editor-cancel').onclick = () => $('formula-editor-dialog').close();
$('my-formulas').onclick = showMyFormulas;
$('recipe-name').oninput = markDirty;
$('palette-search').oninput = (event) => {
  state.paletteQuery = event.target.value;
  renderPalette();
};

api('/api/blocks').then((blocks) => {
  state.blocks = blocks;
  return Promise.all([api('/api/entry-recipe'), api('/api/datasets')]);
}).then(([recipe, datasetPayload]) => {
  state.recipe = recipe;
  state.datasets = Array.isArray(datasetPayload) ? datasetPayload : (datasetPayload.datasets || []);
  state.datasets.forEach((item) => { if (item.alias) state.datasetFacts[item.alias] = item; });
  ensureUiIds(state.recipe.steps);
  state.activeInsertionTarget = defaultInsertionTarget();
  draw();
  const selectedDataset = stepsWithInfo().find(({ info }) => isDatasetSourceInfo(info))?.step;
  if (selectedDataset?.params?.dataset) loadDatasetFacts(selectedDataset.params.dataset);
}).catch((error) => { showMessage(error.message, 'error'); });
