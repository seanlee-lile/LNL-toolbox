const state = {
  blocks: [],
  recipe: { schema_version: 1, name: 'scratch_recipe', description: '', steps: [] },
  selected: null,
  collapsed: new WeakSet(),
  activeInsertionTarget: null,
  drag: null,
  datasets: [],
  datasetFacts: {},
  validated: false,
  errorStepId: null,
  errorMessage: '',
  runtimeLimits: { max_epochs: 1, max_batches: 1, skip_final_test: true },
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
  if (info?.id === 'small_loss_indices' && name === 'input') {
    return [...available].sort().find((key) => /(?:^|_)loss(?:_[ab])?_per_sample$/.test(key)) || null;
  }
  return null;
}

function inferOutputSlot(info, name, target) {
  if (info?.id !== 'small_loss_indices' || name !== 'save_as') return null;
  const siblings = getChildrenArray(target?.parentId) || [];
  const outputs = siblings
    .filter((step) => step.block === 'small_loss_indices')
    .map((step) => step.params?.save_as || blockInfo(step.block)?.params?.save_as?.default);
  if (outputs.includes('selected_b') && !outputs.includes('selected_a')) return 'selected_a';
  if (outputs.includes('selected_a') && !outputs.includes('selected_b')) return 'selected_b';
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
  if (info.id === 'configure_noise') {
    const selectedDataset = flatRecipeSteps().find((step) => step.block === 'select_dataset');
    const alias = String(selectedDataset?.params?.dataset ?? '').trim();
    const facts = alias === 'synthetic' ? syntheticFacts() : state.datasetFacts[alias];
    if (!alias || !facts || facts.status !== 'ready') return '请先选择并成功识别数据集。';
    if (!(facts.noise_methods || []).length) return '当前数据集没有可用的标签噪声实现。';
  }
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

function showMessage(message, className = '') {
  $('output').textContent = message;
  $('output').className = className;
}

function markDirty() {
  state.validated = false;
  state.errorStepId = null;
  state.errorMessage = '';
  updateRunState();
}

function recipeDataReady() {
  const dataset = flatRecipeSteps().find((step) => step.block === 'select_dataset');
  if (dataset) {
    const alias = String(dataset.params?.dataset ?? '').trim();
    if (!alias) return false;
    if (alias === 'synthetic') return true;
    return Boolean(state.datasets.find((item) => item.alias === alias && item.status === 'ready'));
  }
  return flatRecipeSteps().some((step) => step.block.startsWith('load_') || step.block.startsWith('prepare_'));
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
    if (step.block === 'configure_noise' && name === 'method') {
      const dataset = flatRecipeSteps().find((item) => item.block === 'select_dataset');
      const alias = String(dataset?.params?.dataset ?? '').trim();
      const facts = alias === 'synthetic' ? syntheticFacts() : state.datasetFacts[alias];
      return facts?.noise_methods || ['none'];
    }
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
    if (step.block === 'select_dataset' && (name === 'source_mode' || name === 'dataset')) {
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
  const dataset = flatRecipeSteps().find((item) => item.block === 'select_dataset');
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
  const visible = state.blocks.filter((item) => item.beginner_visible);
  const categories = [...new Set(visible.map((item) => item.category))];
  categories.forEach((category) => {
    const title = document.createElement('h3'); title.textContent = category; palette.appendChild(title);
    visible.filter((item) => item.category === category).forEach((item) => {
      const node = document.createElement('div');
      const reason = availabilityReason(item, state.activeInsertionTarget);
      const enabled = !reason;
      node.className = 'palette-block';
      node.classList.toggle('palette-disabled', !enabled);
      node.classList.toggle('disabled', !enabled);
      node.draggable = enabled;
      node.setAttribute('aria-disabled', String(!enabled));
      node.textContent = item.name;
      node.title = enabled ? item.description : `${item.description}\n当前不可用：${reason}`;
      node.ondragstart = (event) => {
        if (!enabled) { event.preventDefault(); return; }
        state.drag = { type: 'new', id: item.id, info: item };
        event.dataTransfer.setData('application/x-lnl-new-block', item.id);
        event.dataTransfer.effectAllowed = 'copy';
        markDropZones();
      };
      node.ondragend = () => { state.drag = null; markDropZones(); };
      node.onclick = () => {
        const target = state.activeInsertionTarget;
        const clickReason = availabilityReason(item, target);
        if (clickReason || !addStepAtTarget(item.id, target)) { showMessage(`不能插入：${clickReason || '插入失败'}`, 'error'); return; }
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
  const keyNames = ['q', 'noise_rate', 'warmup_epochs', 'remember_rate', 'keep_rate'];
  const values = keyNames.filter((name) => info.params?.[name]).map((name) => `${name}=${step.params?.[name] ?? info.params[name].default ?? ''}`);
  if (values.length) lines.push(values.join(' · '));
  const outputs = (info.provides || []).map((name) => slotValue(info, step, name));
  if (outputs.length) lines.push(`→ ${outputs.join(', ')}`);
  return lines.join('\n');
}

function flatRecipeSteps(steps = state.recipe.steps) {
  return (steps || []).flatMap((step) => [step, ...(Array.isArray(step.steps) ? flatRecipeSteps(step.steps) : [])]);
}

function getNextSuggestion() {
  const ids = new Set(flatRecipeSteps().map((step) => step.block));
  const hasDataset = [...ids].some((id) => id.startsWith('load_') || id === 'select_dataset');
  const datasetStep = flatRecipeSteps().find((step) => step.block === 'select_dataset');
  if (!hasDataset || (datasetStep && !String(datasetStep.params?.dataset ?? '').trim())) return '下一步：选择数据集';
  if (!ids.has('configure_noise') && !ids.has('apply_symmetric_noise')) return '下一步：决定是否添加标签噪声';
  if (!ids.has('create_model')) return '下一步：确认模型';
  if (!ids.has('create_optimizer')) return '下一步：确认优化器';
  if (!ids.has('epoch_loop') || !ids.has('batch_loop')) return '下一步：建立 Epoch / Batch Loop';
  if (!ids.has('forward') && !ids.has('forward_feature')) return '下一步：进入 Batch Loop 添加 Forward';
  if (!ids.has('mean_loss')) return '下一步：添加概率/损失公式';
  if (!ids.has('backward')) return '下一步：添加 Backward';
  if (!ids.has('optimizer_step')) return '下一步：添加 Optimizer Step';
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
    node.dataset.uiId = step._uiId;
    const header = document.createElement('div');
    header.className = 'step-header';
    const title = document.createElement('span'); title.textContent = info ? info.name : step.block; header.appendChild(title);
    const kind = document.createElement('span'); kind.className = 'kind'; kind.textContent = info ? info.kind : ''; header.appendChild(kind);
    const summary = renderStepSummary(step, info);
    header.onclick = () => { state.selected = step; renderInspector(); draw(); };
    node.appendChild(header);
    if (summary) { const summaryNode = document.createElement('div'); summaryNode.className = 'step-summary'; summaryNode.textContent = summary; node.appendChild(summaryNode); }
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
  const step = state.selected;
  if (!step) { explanationTarget.textContent = '点击一个积木查看说明'; target.textContent = '选择积木后可编辑参数'; return; }
  const info = blockInfo(step.block);
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
  info.params && Object.entries(info.params).forEach(([name, schema]) => {
    if (step.block === 'select_dataset' && name === 'path' && step.params?.source_mode !== 'custom_path') return;
    const wrap = document.createElement('div'); wrap.className = 'param';
    const label = document.createElement('label'); label.textContent = name; wrap.appendChild(label);
    wrap.appendChild(renderParamControl(name, schema, step));
    params.appendChild(wrap);
  });
  addSection('4. 参数', params);
  if (step.block === 'select_dataset') target.appendChild(renderDatasetFacts(step));
  if (step.block === 'configure_noise') target.appendChild(renderNoiseFacts(step));
}

function draw() {
  $('recipe-name').value = state.recipe.name;
  renderRuntimeLimits();
  $('next-suggestion').textContent = getNextSuggestion();
  renderPalette();
  drawSteps();
  renderInspector();
  updateRunState();
  showMessage(targetLabel(state.activeInsertionTarget));
}

async function loadDefaultRecipe() {
  state.recipe = await api('/api/default-recipe');
  ensureUiIds(state.recipe.steps);
  state.selected = null;
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
    status.textContent = item.status === 'formula-ready' ? '公式积木已展开' : item.status === 'template-ready' ? '论文模板' : '旧版积木模板，尚未逐公式展开';
    card.appendChild(status);
    const open = document.createElement('button'); open.textContent = '打开';
    open.onclick = () => openTemplate(item);
    card.appendChild(open); list.appendChild(card);
  });
  if (!list.children.length) list.textContent = '没有匹配的模板';
}

function renderFeaturedExamples(examples) {
  const list = $('featured-examples');
  list.innerHTML = '';
  examples.forEach((item) => {
    const card = document.createElement('div'); card.className = 'template-card featured-example';
    const name = document.createElement('strong'); name.textContent = `${item.name} 论文范例`; card.appendChild(name);
    const status = document.createElement('small'); status.textContent = '公式积木已展开，可直接查看和运行'; card.appendChild(status);
    const open = document.createElement('button'); open.textContent = '打开范例'; open.onclick = () => openTemplate(item);
    card.appendChild(open); list.appendChild(card);
  });
}

async function openTemplate(item) {
  try {
    state.recipe = await api('/api/recipe/' + encodeURIComponent(item.path));
    ensureUiIds(state.recipe.steps);
    state.selected = null;
    state.validated = false;
    state.activeInsertionTarget = defaultInsertionTarget();
    $('template-dialog').close();
    draw();
  } catch (error) { showError(error); }
}

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
$('run').onclick = async () => { if ($('run').disabled) return; try { const result = await api('/api/run', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(runPayload()) }); showMessage(JSON.stringify(result, null, 2)); } catch (error) { showError(error); } };
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
$('recipe-name').oninput = markDirty;

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
  const selectedDataset = flatRecipeSteps().find((step) => step.block === 'select_dataset');
  if (selectedDataset?.params?.dataset) loadDatasetFacts(selectedDataset.params.dataset);
}).catch((error) => { showMessage(error.message, 'error'); });
