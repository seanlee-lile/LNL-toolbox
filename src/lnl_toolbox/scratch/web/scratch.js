const state = {
  blocks: [],
  formulas: [],
  recipe: { schema_version: 1, name: 'scratch_recipe', description: '', steps: [] },
  selected: null,
  adjacentSelection: null,
  adjacentExpanded: new Set(),
  paletteSelection: null,
  paletteDraft: null,
  collapsed: new WeakSet(),
  activeInsertionTarget: null,
  drag: null,
  datasets: [],
  datasetFacts: {},
  validated: false,
  errorStepId: null,
  errorMessage: '',
  errorPayload: null,
  errorGuide: null,
  lastRun: null,
  runMode: 'check',
  runtimeLimits: { max_epochs: 1, max_batches: 1, skip_final_test: true },
  paletteQuery: '',
  paletteCategory: '数据',
  paletteCollapsed: new Set(),
  compositeExpanded: new Set(),
  dataExecutionView: new Set(),
  dataBlockMode: 'concepts',
  dataOverviewSelection: null,
  compositeUngrouped: new Set(),
  deletedStep: null,
  inspectorTab: 'blocks',
  running: false,
  jobId: null,
  runPollTimer: null,
  runProgress: null,
  runStopping: false,
  formulaEditor: {
    steps: [], editingId: null, paletteExpanded: true, expression: null, expressionSelection: null, expressionDrag: null,
    activeOutput: 'loss', outputExpressions: {loss: null}, inputSchemas: {}, parameterSchemas: {}, outputSchemas: {},
    parameterStates: {},
  },
};
const $ = (id) => document.getElementById(id);
const apiBase = location.pathname.startsWith('/scratch') ? '/api/scratch' : '/api';
let uiIdCounter = 0;
// Epoch snapshots are written by the worker; the UI only needs to read them
// every few seconds. Stopping remains faster so the button feels responsive.
const RUN_PROGRESS_POLL_MS = 4000;
const RUN_STOP_POLL_MS = 500;
const USABLE_DATASET_STATUSES = new Set(['ready', 'available', 'built-in', 'builtin']);

async function api(path, options = {}) {
  const relativePath = path.startsWith('/api/') ? path.slice(4) : path;
  const response = await fetch(apiBase + relativePath, options);
  const body = await response.json();
  if (!response.ok || body.ok === false) {
    const error = new Error(body.error || `HTTP ${response.status}`);
    error.payload = body;
    throw error;
  }
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

function paletteDraftFor(info) {
  if (!state.paletteDraft || state.paletteDraft.block !== info.id || state.paletteDraft.recipe !== state.recipe) {
    state.paletteDraft = {block: info.id, params: {}, recipe: state.recipe};
  }
  return state.paletteDraft;
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
  const location = findParentArrayAndIndex(parentId);
  return location ? getPlacementContext(location.parentId) : 'top';
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

function inferAvailableSlot(info, name, available, step = null) {
  const supplied = step?.params || {};
  if (Object.prototype.hasOwnProperty.call(supplied, name)
      && (typeof supplied[name] !== 'string' || !supplied[name].trim())) return null;
  const configured = slotValue(info, step, name);
  return available.has(configured) ? configured : null;
}

function inferOutputSlot(info, name, target) {
  return null;
}

function addProvidedKeys(keys, step) {
  const info = blockInfo(step.block);
  (info?.provides || []).forEach((name) => keys.add(slotValue(info, step, name)));
}

function addStepsBefore(keys, steps, endIndex, excludedId = null) {
  (steps || []).slice(0, endIndex).forEach((step) => {
    if (step._uiId === excludedId) return;
    addProvidedKeys(keys, step);
    // Match recipe validation: epoch outputs survive the loop, while batch
    // temporaries and conditional child outputs do not escape their scope.
    if (step.block === 'epoch_loop') addStepsBefore(keys, step.steps, step.steps?.length || 0, excludedId);
  });
}

function availableKeysBefore(target, excludedId = null) {
  const keys = new Set(Object.keys(state.recipe.settings || {}));
  if (!target) return keys;
  if (target.parentId === '__root__') {
    addStepsBefore(keys, state.recipe.steps, target.index, excludedId);
    return keys;
  }
  const path = findStepPath(target.parentId);
  if (!path) return keys;
  let siblings = state.recipe.steps;
  for (const parent of path) {
    const parentIndex = siblings.indexOf(parent);
    addStepsBefore(keys, siblings, parentIndex, excludedId);
    addProvidedKeys(keys, parent);
    siblings = Array.isArray(parent.steps) ? parent.steps : [];
  }
  addStepsBefore(keys, siblings, target.index, excludedId);
  return keys;
}

function availabilityReason(info, target, movingId = null, params = null, allowIncomplete = false) {
  if (!target) return '请先选择插入位置';
  if (!info) return '积木定义尚未加载，请刷新积木库';
  const parent = target.parentId === '__root__' ? null : findStepById(target.parentId);
  const children = target.parentId === '__root__' ? state.recipe.steps : parent?.steps;
  if (!Array.isArray(children) || !Number.isInteger(target.index) || target.index < 0 || target.index > children.length
      || (parent && blockInfo(parent.block)?.kind === 'action')) return '插入位置已失效，请重新点击插入位';
  const names = { top: '根层（循环外）', epoch: '每轮训练（Epoch）', batch: '每批训练（Batch）', any: '任意层级' };
  if (!placementAllows(info, target.parentId)) return `当前是${names[getPlacementContext(target.parentId)]}；此积木允许：${(info.placement || ['any']).map((value) => names[value] || value).join('、')}`;
  if (movingId && target.parentId !== '__root__' && containsStep(findStepById(movingId), target.parentId)) return '不能把循环拖入自己的子层';
  const movingStep = movingId ? findStepById(movingId) : null;
  if (movingId && !movingStep) return '待移动积木已不存在，请重新选择';
  const candidate = params ? {params} : movingStep || (state.paletteDraft?.block === info.id && state.paletteDraft.recipe === state.recipe ? state.paletteDraft : null);
  const available = availableKeysBefore(target, movingId);
  const missing = (info.requires || [])
    .filter((name) => !inferAvailableSlot(info, name, available, candidate))
    .map((name) => slotValue(info, candidate, name));
  if (missing.length && !allowIncomplete) return `缺少前置输入：${missing.join(', ')}。可先添加积木，再在右侧连接输入`;
  if (movingStep) return moveDependencyReason(movingStep, target);
  return null;
}

function moveDependencyReason(movingStep, target) {
  // Inspect a copy so a rejected drag never mutates the recipe. Only newly
  // broken connections block a move; unrelated unfinished steps remain editable.
  function missingConnections(steps, available, missing = new Map(), context = 'top') {
    const current = new Set(available);
    for (const step of steps) {
      const info = blockInfo(step.block);
      if (info && info.beginner_visible !== false && !(info.placement || ['any']).some((place) => place === 'any' || place === context)) {
        missing.set(`${step._uiId}:placement`, `${info.name || step.block} 不支持移动后的 ${context} 层级`);
      }
      for (const name of info?.requires || []) {
        const slot = slotValue(info, step, name);
        if (!current.has(slot)) missing.set(`${step._uiId}:${slot}`, `${info.name || step.block} 需要 ${slot}`);
      }
      const childKeys = new Set(current);
      addProvidedKeys(childKeys, step);
      if (Array.isArray(step.steps)) {
        const childContext = step.block === 'epoch_loop' ? 'epoch' : step.block === 'batch_loop' ? 'batch' : context;
        const result = missingConnections(step.steps, childKeys, missing, childContext);
        if (step.block === 'epoch_loop') result.current.forEach((key) => current.add(key));
      }
      addProvidedKeys(current, step);
    }
    return {current, missing};
  }
  const initial = new Set(Object.keys(state.recipe.settings || {}));
  const before = missingConnections(state.recipe.steps, initial).missing;
  const copy = (steps, parentId = '__root__') => {
    const result = [];
    for (let index = 0; index <= steps.length; index += 1) {
      if (parentId === target.parentId && index === target.index) result.push(movingStep);
      const step = steps[index];
      if (!step || step._uiId === movingStep._uiId) continue;
      result.push(Array.isArray(step.steps) ? {...step, steps: copy(step.steps, step._uiId)} : step);
    }
    return result;
  };
  const after = missingConnections(copy(state.recipe.steps), initial).missing;
  const broken = [...after].filter(([key]) => !before.has(key)).map(([, reason]) => reason);
  return broken.length ? `移动会断开输入连接或违反层级限制：${broken.join('；')}。请同时调整依赖积木的位置` : null;
}

function canInsert(info, parentId, movingId = null, index = 0, allowIncomplete = false) {
  return availabilityReason(info, { parentId, index, context: getPlacementContext(parentId) }, movingId, null, allowIncomplete) === null;
}

function findDropTarget(zone) {
  return { parentId: zone.dataset.parentId || '__root__', index: Number(zone.dataset.index || 0), context: zone.dataset.context || 'top', compositeId: zone.dataset.compositeId || null };
}

function setActiveInsertionTarget(target) {
  state.activeInsertionTarget = target;
  draw();
}

function targetLabel(target) {
  if (!target) return '请先点击一个插入位置';
  const contextNames = { top: '根层', epoch: 'Epoch Loop', batch: 'Batch Loop', any: '当前容器' };
  const focus = target.focus === 'algorithm-core' ? '算法核心' : `第 ${target.index + 1} 步`;
  return `当前插入位置：${contextNames[getPlacementContext(target.parentId)] || target.context}，${focus}。`;
}

function renderPositionIndicator() {
  const indicator = $('workspace-position-indicator');
  if (!indicator) return;
  const detail = $('workspace-position-detail');
  const constraint = $('workspace-position-constraint');
  const status = $('workspace-position-status');
  const target = state.activeInsertionTarget;
  const contextNames = { top: '根层', epoch: 'Epoch Loop', batch: 'Batch Loop', any: '当前容器' };
  const context = target ? (contextNames[getPlacementContext(target.parentId)] || target.context) : null;
  // A selected workspace block is also a candidate: selecting a block and
  // then clicking another insertion slot should explain whether that block
  // can be moved there, just like a palette block being dragged in.
  const selectedInfo = state.selected ? blockInfo(state.selected.block) : null;
  const candidate = state.drag?.info || state.paletteSelection || selectedInfo;
  const movingId = state.drag?.type === 'step'
    ? state.drag.id
    : (!state.drag && state.selected ? state.selected._uiId : null);

  indicator.dataset.state = target ? 'ready' : 'empty';
  if (detail) detail.textContent = target ? `${context} · 第 ${target.index + 1} 个插入位` : '尚未选择插入位置';
  if (constraint) {
    if (!target) {
      constraint.textContent = '先选插入位，再选积木：这里会说明能否放置，以及缺少哪些输入。';
    } else {
      const slotCount = availableKeysBefore(target).size;
      constraint.textContent = `位置需支持${context}；输入可以添加后再连接。此处已有 ${slotCount} 项输入。可放置不代表已通过运行检查。`;
    }
  }
  if (!status) return;
  if (!candidate) {
    status.textContent = target ? '○ 请选择左侧积木查看是否可放置' : '○ 等待选择插入位';
    return;
  }
  const reason = availabilityReason(candidate, target, movingId);
  if (reason) {
    const configurable = !movingId && !availabilityReason(candidate, target, null, null, true);
    indicator.dataset.state = configurable ? 'ready' : 'invalid';
    status.textContent = configurable ? `○ 可添加，待连接：${reason}` : `✕ ${candidate.name || candidate.id}：${reason}`;
  } else {
    indicator.dataset.state = 'valid';
    status.textContent = `✓ ${candidate.name || candidate.id} 可以放置在这里`;
  }
}

function defaultInsertionTarget() {
  const epoch = state.recipe.steps.find((step) => step.block === 'epoch_loop');
  const batch = epoch?.steps?.find((step) => step.block === 'batch_loop');
  if (!batch) return null;
  const coreIndex = (batch.steps || []).findIndex((step) =>
    ['forward', 'forward_feature', 'forward_with_state'].includes(step.block));
  const index = coreIndex >= 0 ? coreIndex + 1 : (batch.steps || []).length;
  return { parentId: batch._uiId, index, context: 'batch', focus: 'algorithm-core' };
}

function collectInsertionTargets(steps = state.recipe.steps, parentId = '__root__', context = 'top', targets = []) {
  const children = steps || [];
  for (let index = 0; index <= children.length; index += 1) targets.push({parentId, index, context});
  children.forEach((step) => {
    if (!Array.isArray(step.steps)) return;
    const childContext = getPlacementContext(step._uiId);
    collectInsertionTargets(step.steps, step._uiId, childContext, targets);
  });
  return targets;
}

function revealInsertionTarget(target) {
  const path = target?.parentId && target.parentId !== '__root__' ? findStepPath(target.parentId) : [];
  let siblings = state.recipe.steps;
  let containerId = '__root__';
  (path || []).forEach((ancestor) => {
    const ancestorIndex = siblings.indexOf(ancestor);
    detectCompositeRanges(siblings).forEach((range) => {
      if (ancestorIndex < range.start || ancestorIndex >= range.end) return;
      const key = compositeKey(containerId, {...range, steps: siblings});
      if (!state.compositeUngrouped.has(key)) state.compositeExpanded.add(key);
    });
    if (blockInfo(ancestor.block)?.kind !== 'action') state.collapsed.delete(ancestor);
    siblings = Array.isArray(ancestor.steps) ? ancestor.steps : [];
    containerId = ancestor._uiId;
  });
}

function recommendedInsertionTarget(info, movingId = null) {
  const current = state.activeInsertionTarget;
  if (current && canInsert(info, current.parentId, movingId, current.index)) return current;
  return collectInsertionTargets().find((target) => canInsert(info, target.parentId, movingId, target.index)) || null;
}

function locateBlockInsertion(info) {
  const target = recommendedInsertionTarget(info);
  if (!target) {
    showMessage(`没有找到可插入位置：${availabilityReason(info, state.activeInsertionTarget) || '请先补齐前置 slot'}`, 'error');
    return false;
  }
  revealInsertionTarget(target);
  state.activeInsertionTarget = target;
  state.paletteSelection = info;
  state.selected = null;
  draw();
  requestAnimationFrame(() => {
    const zone = [...document.querySelectorAll('.drop-target')].find((item) => {
      const candidate = findDropTarget(item);
      return candidate.parentId === target.parentId && candidate.index === target.index;
    });
    zone?.scrollIntoView({block: 'center'});
  });
  return true;
}

const UI_CATEGORIES = [
  { name: '全部', icon: '▤', color: '#84cc16' },
  { name: '数据', icon: '▦', color: '#38bdf8' },
  { name: '模型与训练', icon: '◈', color: '#a78bfa' },
  { name: '算法操作', icon: 'ƒ', color: '#fb7185' },
  { name: '评估', icon: '◒', color: '#c4b5fd' },
  { name: '我的', icon: '♡', color: '#f472b6' },
];

function uiCategory(info) {
  if (isUserOwnedInfo(info)) return '我的';
  const raw = `${info?.ui_group || ''} ${info?.category || ''} ${info?.stage || ''}`.toLowerCase();
  if (/data|dataset|loader|noise|view|role|split|manifest|数据/.test(raw)) return '数据';
  if (/evaluation|evaluate|metric|评估/.test(raw)) return '评估';
  if (/model|network|backbone|optimizer|scheduler|epoch|batch|training|control|runtime|train|loop|模型|训练|更新|参数/.test(raw)) return '模型与训练';
  return '算法操作';
}

function isUserOwnedInfo(info) {
  if (!info) return false;
  const origin = String(info.origin || info.metadata?.origin || '').toLowerCase();
  const id = String(info.id || '').toLowerCase();
  return origin === 'user' || origin.includes('scratch-web') || id.startsWith('user/') || id.includes('__user__');
}

function categoryInfo(name) { return UI_CATEGORIES.find((item) => item.name === name) || UI_CATEGORIES.find((item) => item.name === '模型与训练') || UI_CATEGORIES[0]; }

function setResultState(label, className = '') {
  const status = $('result-status');
  if (!status) return;
  status.textContent = label;
  status.className = `result-status ${className}`.trim();
}

function clearRunPolling() {
  if (state.runPollTimer) window.clearTimeout(state.runPollTimer);
  state.runPollTimer = null;
}

function resetRunTracking() {
  clearRunPolling();
  state.jobId = null;
  state.runProgress = null;
  state.runStopping = false;
  state.running = false;
  const panel = $('run-progress');
  if (panel) panel.hidden = true;
}

function renderRunProgress(job = null) {
  const panel = $('run-progress');
  const bar = $('progress-bar');
  const percent = $('progress-percent');
  const stage = $('progress-stage');
  const position = $('progress-position');
  const block = $('progress-block');
  if (!panel || !bar || !percent || !stage || !position || !block) return;
  const progress = job?.progress || state.runProgress || {};
  renderEpochOutputs(progress);
  const rawFraction = Number(progress.fraction);
  const fraction = Number.isFinite(rawFraction)
    ? Math.min(Math.max(rawFraction, 0), 1)
    : (job?.status === 'completed' ? 1 : null);
  panel.hidden = !job && !state.runProgress;
  bar.value = fraction ?? 0;
  percent.textContent = fraction === null ? '—' : `${Math.round(fraction * 100)}%`;
  const stateLabel = {starting: '正在启动', running: '运行中', stopping: '正在停止', completed: '已完成', cancelled: '已取消', failed: '失败'};
  stage.textContent = stateLabel[job?.status || progress.state] || progress.state || '准备中';
  const epoch = Number.isInteger(progress.epoch) ? progress.epoch + 1 : null;
  const totalEpochs = Number.isInteger(progress.total_epochs) ? progress.total_epochs : null;
  const batch = Number.isInteger(progress.batch_idx) ? progress.batch_idx + 1 : null;
  const totalBatches = Number.isInteger(progress.total_batches) ? progress.total_batches : null;
  const epochText = epoch === null ? 'epoch —' : `epoch ${epoch}${totalEpochs === null ? '' : ` / ${totalEpochs}`}`;
  const batchText = batch === null ? 'batch —' : `batch ${batch}${totalBatches === null ? '' : ` / ${totalBatches}`}`;
  position.textContent = `${epochText} · ${batchText} · global step ${Number.isInteger(progress.global_step) ? progress.global_step : '—'}`;
  block.textContent = progress.block ? `当前积木：${progress.block}` : (progress.error ? `错误：${progress.error}` : '正在准备执行…');
}

function renderEpochOutputs(progress = {}) {
  const list = $('epoch-output-list');
  const count = $('epoch-output-count');
  if (!list || !count) return;
  const rows = Array.isArray(progress.epoch_outputs) ? progress.epoch_outputs : [];
  count.textContent = `${rows.length} 轮`;
  list.replaceChildren();
  if (!rows.length) {
    const empty = document.createElement('div');
    empty.className = 'epoch-output-empty';
    empty.textContent = '完成一个 epoch 后，这里会实时显示该轮的标量输出。';
    list.appendChild(empty);
    return;
  }
  rows.forEach((row, index) => {
    if (!row || typeof row !== 'object' || Array.isArray(row)) return;
    const card = document.createElement('article');
    card.className = 'epoch-output-card';
    const heading = document.createElement('header');
    const title = document.createElement('strong');
    const epoch = Number.isInteger(row.epoch) ? row.epoch + 1 : index + 1;
    title.textContent = `Epoch ${epoch}`;
    const ordinal = document.createElement('span');
    ordinal.textContent = `第 ${index + 1} 次完成`;
    heading.append(title, ordinal);
    card.appendChild(heading);
    const values = document.createElement('div');
    values.className = 'epoch-output-values';
    Object.entries(row).forEach(([name, value]) => {
      if (name === 'epoch') return;
      const item = document.createElement('div');
      item.className = 'epoch-output-value';
      const label = document.createElement('strong');
      label.textContent = `${name}: `;
      const rendered = document.createElement('code');
      rendered.textContent = formatResultValue(value);
      item.append(label, rendered);
      values.appendChild(item);
    });
    if (!values.childElementCount) {
      values.textContent = '本轮没有可显示的标量输出。';
    }
    card.appendChild(values);
    list.appendChild(card);
  });
  list.scrollTop = list.scrollHeight;
}

function renderRunJob(job) {
  if (!job) return;
  state.jobId = job.id || state.jobId;
  state.runStopping = Boolean(job.running && job.cancel_requested);
  state.runProgress = job.progress || state.runProgress;
  renderRunProgress(job);
  const output = $('output');
  if (output && Array.isArray(job.lines)) {
    output.textContent = job.lines.join('\n') || '正在启动 Scratch 运行进程…';
  }
  if (job.running) {
    state.running = true;
    setResultState(job.cancel_requested ? '正在停止…' : '运行中…', 'running');
    if ($('result-summary')) $('result-summary').textContent = job.cancel_requested
      ? '已发出停止请求，正在等待运行进程退出…'
      : '训练正在后台执行；实时进度会在此处更新。';
    updateRunState();
    return;
  }
  state.running = false;
  clearRunPolling();
  updateRunState();
  if (job.status === 'cancelled' || job.cancel_requested) {
    setResultState('已停止', 'error');
    if ($('result-summary')) $('result-summary').textContent = '运行已停止；已经生成的中间产物仍保留在运行目录中。';
    return;
  }
  if (job.returncode === 0) {
    const result = job.structured && typeof job.structured === 'object' ? job.structured : {
      ok: true, metrics: [], artifact_dir: job.artifact_dir,
    };
    renderRunResult({...result, artifact_dir: result.artifact_dir || job.artifact_dir});
    return;
  }
  const error = new Error(job.error || 'Scratch 运行失败，请展开原始输出查看原因。');
  error.payload = {ok: false, error: error.message, code: job.error_code, block_id: job.block_id, params: job.params};
  showError(error);
}

async function pollRunJob(jobId) {
  if (!jobId) return;
  try {
    const job = await api(`/jobs/${encodeURIComponent(jobId)}`);
    renderRunJob(job);
    if (job.running) state.runPollTimer = window.setTimeout(() => pollRunJob(jobId), RUN_PROGRESS_POLL_MS);
  } catch (error) {
    clearRunPolling(); state.running = false; updateRunState(); showError(error);
  }
}

async function stopRun() {
  if (!state.jobId || !state.running) return;
  state.runStopping = true;
  const button = $('stop-run');
  if (button) { button.disabled = true; button.textContent = '正在停止…'; }
  try {
    const job = await api(`/jobs/${encodeURIComponent(state.jobId)}/cancel`, {method: 'POST'});
    renderRunJob(job);
    if (job.running) state.runPollTimer = window.setTimeout(() => pollRunJob(state.jobId), RUN_STOP_POLL_MS);
  } catch (error) {
    state.runStopping = false;
    if (button) { button.disabled = false; button.textContent = '■ 停止运行'; }
    showError(error);
  }
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
  setInspectorTab(state.inspectorTab);
}

function setInspectorTab(tab = 'blocks') {
  state.inspectorTab = tab === 'run' ? 'run' : 'blocks';
  document.querySelectorAll('[data-inspector-tab]').forEach((button) => {
    const active = button.dataset.inspectorTab === state.inspectorTab;
    button.setAttribute('aria-selected', String(active));
    button.classList.toggle('active', active);
  });
  const blockPanel = $('inspector-block-tab');
  const runPanel = $('inspector-run-tab');
  if (blockPanel) blockPanel.hidden = state.inspectorTab !== 'blocks';
  if (runPanel) runPanel.hidden = state.inspectorTab !== 'run';
}

function focusDatasetSource() {
  let entry = datasetSourceEntry();
  if (!entry) {
    const target = state.activeInsertionTarget || defaultInsertionTarget() || {parentId: '__root__', index: 0, context: 'top'};
    if (!addStepAtTarget('load_dataset', target)) {
      showMessage('无法自动添加 load_dataset，请从左侧“数据”分类拖入“加载数据集”。', 'error');
      return;
    }
    entry = datasetSourceEntry();
  }
  if (!entry) return;
  state.selected = entry.step;
  state.paletteSelection = null;
  state.inspectorTab = 'blocks';
  draw();
  requestAnimationFrame(() => document.querySelector(`[data-ui-id="${entry.step._uiId}"]`)?.scrollIntoView({block: 'center'}));
}

function chooseSyntheticDataset() {
  let entry = datasetSourceEntry();
  if (!entry) {
    const target = state.activeInsertionTarget || defaultInsertionTarget() || {parentId: '__root__', index: 0, context: 'top'};
    if (!addStepAtTarget('load_dataset', target)) {
      showMessage('无法自动添加数据集积木，请从左侧“数据”分类拖入“加载数据集”。', 'error');
      return;
    }
    entry = datasetSourceEntry();
  }
  if (!entry) return;
  entry.step.params = {...entry.step.params, dataset: 'synthetic', source_mode: 'builtin'};
  markDirty();
  state.selected = entry.step;
  state.paletteSelection = null;
  state.inspectorTab = 'blocks';
  draw();
  loadDatasetFacts('synthetic');
}

function focusErrorStep() {
  const step = state.errorStepId ? findStepById(state.errorStepId) : state.selected;
  if (!step) {
    state.inspectorTab = 'blocks';
    draw();
    return;
  }
  state.selected = step;
  state.paletteSelection = null;
  state.inspectorTab = 'blocks';
  draw();
  requestAnimationFrame(() => document.querySelector(`[data-ui-id="${step._uiId}"]`)?.scrollIntoView({block: 'center'}));
}

function openRunDetails() {
  state.inspectorTab = 'run';
  draw();
  const details = $('result-details');
  if (details) details.open = true;
}

function runGuideAction(action) {
  if (action === 'focus-dataset') return focusDatasetSource();
  if (action === 'use-synthetic') return chooseSyntheticDataset();
  if (action === 'focus-error') return focusErrorStep();
  if (action === 'open-run-details') return openRunDetails();
  return undefined;
}

function guideForError(error) {
  const payload = error?.payload || {};
  const code = String(error?.guidanceCode || error?.code || payload.code || '').toLowerCase();
  const message = String(error?.message || payload.error || '');
  if (code === 'missing-dataset-block') {
    return {
      title: '运行前需要一个数据集入口',
      summary: '当前 Recipe 还没有 load_dataset。先添加它，系统才知道从哪里读取 train/test 数据。',
      steps: ['点击“添加加载数据集”，或从左侧“数据”分类拖入“加载数据集”。', '在右侧“积木”面板的 dataset 参数中选择数据源。', '只想确认流程时，可以先使用内置 synthetic。'],
      actions: [{id: 'focus-dataset', label: '添加 / 查看数据集积木', primary: true}, {id: 'use-synthetic', label: '直接使用 synthetic'}],
    };
  }
  if (code === 'missing-dataset-selection' || /选择有效数据集|尚未选择 dataset|dataset.*(required|missing)/i.test(message)) {
    return {
      title: '请选择数据集',
      summary: '数据集不是在运行面板里选，而是在“加载数据集”积木的右侧参数中选。',
      steps: ['点击“查看数据集积木”，系统会自动定位到 load_dataset。', '在右侧“数据集”下拉框中选择可用数据集。', '选择后可在同一面板查看 train/test 样本、类别数和标签能力。'],
      actions: [{id: 'focus-dataset', label: '定位到数据集选择', primary: true}, {id: 'use-synthetic', label: '使用内置 synthetic 试跑'}],
    };
  }
  if (code === 'missing-dataset-path' || code === 'dataset-not-found' || code === 'dataset-unavailable' || /数据集.*(不可用|找不到|path|路径)/i.test(message)) {
    return {
      title: '数据集资源还没有准备好',
      summary: message || '当前数据源没有可用的本地资源。',
      steps: ['回到“积木”标签，打开 load_dataset 的 dataset/source_mode/path 参数。', '注册数据集或填写正确路径，并确认 train/test source 都能访问。', '如果只是先验证算法连接，可以切换到内置 synthetic。'],
      actions: [{id: 'focus-dataset', label: '检查数据源参数', primary: true}, {id: 'use-synthetic', label: '切换 synthetic 试跑'}],
    };
  }
  if (code === 'dataset-model-class-mismatch' || /类别数不一致|num_classes.*(match|匹配)/i.test(message)) {
    return {
      title: '数据集和模型类别数不匹配',
      summary: message || '模型输出的类别数必须与所选数据集一致，否则标签索引会越界。',
      steps: ['点击“查看出错积木”定位 create_model。', '如果使用 CIFAR-100，请将 num_classes 设为 100；如果运行 GCE CIFAR-10，请选择 local-cifar10 或 CIFAR-10。', '修改后重新点击“检查”再运行；不要用路径或标签替换掩盖类别数不一致。'],
      actions: [{id: 'focus-error', label: '定位模型类别数', primary: true}, {id: 'focus-dataset', label: '重新选择数据集'}],
    };
  }
  if (code === 'missing-slot' || /slot|requires|前置 slot|缺少.*(输入|依赖)|missing.*input/i.test(message)) {
    return {
      title: '这个积木缺少前置输入',
      summary: '它需要的 slot 尚未由前面的积木产生，不能直接运行。',
      steps: ['点击“查看出错积木”，查看右侧输入 / 输出和当前可用 slots。', '把产生这些 slots 的积木拖到它之前，或在参数下拉框改选已有 slot。', '再次运行前，确认绿色插入位和连接顺序正确。'],
      actions: [{id: 'focus-error', label: '查看出错积木', primary: true}],
    };
  }
  if (code === 'placement' || /placement|不能插入|只能放入|循环.*(根层|Epoch|Batch)/i.test(message)) {
    return {
      title: '积木放置位置不符合规则',
      summary: '这个积木只能放在特定层级；红色插入位表示当前位置不合法。',
      steps: ['把积木拖到绿色插入位，不要拖到普通积木正文上。', 'Epoch Loop 只能放在根层，Batch Loop 只能放在 Epoch Loop 内。', '公式、选择和更新积木通常应放在 Batch Loop 内。'],
      actions: [{id: 'focus-error', label: '查看出错积木', primary: true}],
    };
  }
  if (code === 'parameter' || /参数|parameter|必须是|范围|finite|invalid/i.test(message)) {
    return {
      title: '参数需要修正',
      summary: '某个参数的类型、范围或格式不符合该积木的定义。',
      steps: ['点击“查看出错积木”，右侧会显示参数类型、默认值和限制。', '数字参数请填写有限值；枚举、模型、优化器和 slot 参数请从下拉框选择。', '修正后会自动取消旧的检查结果，需要重新点击运行。'],
      actions: [{id: 'focus-error', label: '编辑出错参数', primary: true}],
    };
  }
  if (code === 'formula' || /公式|formula|operation/i.test(message)) {
    return {
      title: '公式连接还不完整',
      summary: '公式步骤必须使用已有输入，并且最终输出要连接到一个真实 slot。',
      steps: ['打开右侧“积木”标签，确认每一步的输入 slot 都来自前置积木。', '在公式编辑器中使用类型化参数，不要留下空的步骤输入。', '确认公式最终输出被后续 loss / update 积木消费。'],
      actions: [{id: 'focus-error', label: '查看公式积木', primary: true}],
    };
  }
  if (code === 'resource' || /manifest|文件不存在|资源|permission|权限|external|依赖/i.test(message)) {
    return {
      title: '运行资源或环境不可用',
      summary: message || '数据文件、噪声 manifest、模型资源或运行环境没有准备好。',
      steps: ['先看右侧“运行”标签的原始输出，确认具体缺失的是数据、manifest 还是依赖。', '回到对应数据 / 噪声 / 模型积木检查 path、artifact 和设备参数。', '确认资源后重新运行；不要用空文件或 observed label 代替真实输入。'],
      actions: [{id: 'open-run-details', label: '展开运行日志', primary: true}, {id: 'focus-error', label: '查看相关积木'}],
    };
  }
  return {
    title: '运行没有完成',
    summary: message || '系统返回了未分类的运行错误。',
    steps: ['先查看出错积木的输入 / 输出和当前可用 slots。', '再展开运行原始输出，确认是连接、参数、数据资源还是运行环境问题。', '修正后重新运行；每次编辑都会重新进行结构检查。'],
    actions: [{id: 'focus-error', label: '查看出错积木', primary: true}, {id: 'open-run-details', label: '展开运行日志'}],
  };
}

function renderGuidance() {
  const target = $('run-guidance');
  if (!target) return;
  target.replaceChildren();
  const guide = state.errorGuide;
  target.hidden = !guide;
  if (!guide) return;
  const heading = document.createElement('strong'); heading.className = 'run-guidance-title'; heading.textContent = `怎么处理：${guide.title}`; target.appendChild(heading);
  const summary = document.createElement('p'); summary.className = 'run-guidance-summary'; summary.textContent = guide.summary; target.appendChild(summary);
  const list = document.createElement('ol'); list.className = 'run-guidance-steps';
  (guide.steps || []).forEach((step) => list.appendChild(Object.assign(document.createElement('li'), {textContent: step})));
  target.appendChild(list);
  const actions = document.createElement('div'); actions.className = 'run-guidance-actions';
  (guide.actions || []).forEach((item) => {
    const button = document.createElement('button'); button.type = 'button'; button.textContent = item.label; if (item.primary) button.className = 'primary-action'; button.onclick = () => runGuideAction(item.id); actions.appendChild(button);
  });
  target.appendChild(actions);
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
  state.errorStepId = null;
  state.errorMessage = '';
  state.errorPayload = null;
  state.errorGuide = null;
  clearErrorPresentation();
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
  setInspectorTab('run');
}

function markDirty() {
  const hadError = Boolean(state.errorMessage || state.errorStepId || state.errorPayload || state.errorGuide);
  state.validated = false;
  state.lastRun = null;
  state.errorStepId = null;
  state.errorMessage = '';
  state.errorPayload = null;
  state.errorGuide = null;
  clearErrorPresentation();
  if (hadError) {
    setResultState('待重新运行');
    const summary = $('result-summary');
    if (summary) summary.textContent = '已修改，旧错误已清除；请重新运行检查。';
  }
  updateRunState();
}

function clearErrorPresentation() {
  document.querySelectorAll('.inspector-error').forEach((node) => node.remove());
  document.querySelectorAll('.error-step').forEach((node) => node.classList.remove('error-step'));
  renderGuidance();
}

function recipeDataReady() {
  return datasetReadiness().ok;
}

function datasetSourceEntry() {
  return stepsWithInfo().find(({ info }) => isDatasetSourceInfo(info)) || null;
}

function datasetIsUsable(item) {
  return Boolean(item && (USABLE_DATASET_STATUSES.has(String(item.status || '').toLowerCase()) || item.alias === 'synthetic'));
}

function datasetClassCount(entry, alias, item) {
  if (alias === 'synthetic') {
    const options = entry?.step?.params?.options;
    const value = options && typeof options === 'object' ? options.classes : entry?.step?.params?.classes;
    const classes = Number(value ?? 2);
    return Number.isFinite(classes) && classes > 0 ? classes : null;
  }
  const classes = Number(item?.num_classes);
  return Number.isFinite(classes) && classes > 0 ? classes : null;
}

function datasetModelCompatibility(entry, alias, item) {
  const classes = datasetClassCount(entry, alias, item);
  if (classes === null) return null; // custom paths may not expose semantics yet
  const conflicts = stepsWithInfo()
    .filter(({ info, step }) => info?.id === 'create_model' && step.params?.num_classes !== undefined)
    .filter(({ step }) => Number(step.params.num_classes) !== classes);
  if (!conflicts.length) return null;
  const requested = [...new Set(conflicts.map(({ step }) => Number(step.params.num_classes)))].join(', ');
  const models = conflicts.map(({ step }) => String(step.params?.model || 'model')).join(', ');
  return {
    ok: false,
    code: 'dataset-model-class-mismatch',
    title: '数据集与模型类别数不一致',
    message: `当前数据集“${alias}”有 ${classes} 个类别，但模型（${models}）配置为 ${requested} 类。请改用匹配的数据集，或在 create_model 的 num_classes 中填写 ${classes}。`,
    step: conflicts[0].step,
  };
}

function datasetReadiness() {
  const entry = datasetSourceEntry();
  if (!entry) {
    return {
      ok: false,
      code: 'missing-dataset-block',
      title: '还没有加载数据集积木',
      message: '当前 Recipe 没有 load_dataset。运行前请先添加一个“加载数据集”积木。',
      step: null,
    };
  }
  const params = entry.step.params || {};
  const mode = String(params.source_mode || 'registered');
  const alias = String(params.dataset ?? params.name ?? '').trim();
  const path = String(params.path || '').trim();
  if (mode === 'custom_path') {
    if (!path) {
      return {
        ok: false,
        code: 'missing-dataset-path',
        title: '数据集路径还没有填写',
        message: '当前数据源选择了自定义路径，但 path 为空。请在右侧积木参数中填写可访问的数据路径。',
        step: entry.step,
      };
    }
    return { ok: true, code: 'custom-path', alias: path, step: entry.step };
  }
  if (!alias) {
    return {
      ok: false,
      code: 'missing-dataset-selection',
      title: '请选择数据集',
      message: 'load_dataset 已添加，但还没有选择 dataset。请点击右侧“积木”标签，在“数据集”参数下拉框中选择一个数据集。',
      step: entry.step,
    };
  }
  if (alias === 'synthetic') {
    return datasetModelCompatibility(entry, alias, null) || { ok: true, code: 'synthetic', alias, step: entry.step };
  }
  const item = state.datasets.find((candidate) => String(candidate.alias || candidate.name || '') === alias);
  if (datasetIsUsable(item)) {
    return datasetModelCompatibility(entry, alias, item) || { ok: true, code: 'ready', alias, item, step: entry.step };
  }
  return {
    ok: false,
    code: item ? 'dataset-unavailable' : 'dataset-not-found',
    title: item ? `数据集“${alias}”当前不可用` : `找不到数据集“${alias}”`,
    message: item
      ? `数据集“${alias}”已配置，但当前状态为 ${item.status || 'unknown'}。请检查数据资源，或先切换到内置 synthetic 试跑。`
      : `数据集“${alias}”不在当前 Scratch 数据目录中。请从右侧 dataset 下拉框重新选择，或使用内置 synthetic 试跑。`,
    step: entry.step,
    item,
  };
}

function updateRunState() {
  const run = $('run');
  if (run) {
    run.disabled = Boolean(state.running);
    run.textContent = state.running ? '运行中…' : '▶ 试运行';
  }
  const fullRun = $('run-full');
  if (fullRun) fullRun.disabled = Boolean(state.running);
  const stop = $('stop-run');
  if (stop) {
    stop.disabled = !state.running || !state.jobId || state.runStopping;
    stop.textContent = state.runStopping ? '正在停止…' : '■ 停止运行';
  }
  const refresh = $('refresh-run');
  if (refresh) refresh.disabled = !state.jobId;
  const mode = $('run-mode');
  if (mode) mode.disabled = Boolean(state.running);
}

function renderRuntimeLimits() {
  const epochStep = flatRecipeSteps().find((step) => step.block === 'epoch_loop');
  const formalEpochs = epochStep?.params?.epochs ?? blockInfo('epoch_loop')?.params?.epochs?.default ?? '—';
  const mode = state.runMode === 'full' ? '完整运行：按 Recipe 正式 epochs / 全部 batch' : `结构验证：${state.runtimeLimits.max_epochs} epoch / ${state.runtimeLimits.max_batches} batch（跳过最终测试）`;
  const banner = $('runtime-limits');
  if (banner) banner.textContent = `正式配置：${formalEpochs} epochs；${mode}`;
  const help = $('run-mode-help');
  if (help) help.textContent = state.runMode === 'full'
    ? '完整运行将使用正式轮次和 batch；数据量较大时可能需要较长时间，可随时点击“停止运行”。'
    : '默认只跑一轮做结构验证；切换到“完整运行”后会按当前 Recipe 的正式轮次和 batch 执行。';
  const selector = $('run-mode');
  if (selector && selector.value !== state.runMode) selector.value = state.runMode;
}

function selectedTarget(step) {
  if (step === state.paletteDraft) return state.activeInsertionTarget;
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
        ? state.datasets.filter(datasetIsUsable).map((item) => item.alias || item.name).filter(Boolean)
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
  if (kind === 'dataset') {
    input.classList.add('dataset-selector');
    input.setAttribute('aria-label', 'dataset（必选）：选择数据集');
    input.title = '运行前必须选择一个数据集；选项来自当前 Scratch 数据目录。';
  }
  const updateValue = () => {
    if (kind === 'bool') step.params[name] = input.checked;
    else if (kind === 'int') step.params[name] = Number.parseInt(input.value, 10);
    else if (kind === 'float') step.params[name] = Number.parseFloat(input.value);
    else step.params[name] = input.value;
    if (step !== state.paletteDraft) markDirty();
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

async function refreshDatasets({announce = false} = {}) {
  try {
    const payload = await api('/api/datasets');
    state.datasets = Array.isArray(payload) ? payload : (payload.datasets || []);
    state.datasets.forEach((item) => { if (item.alias) state.datasetFacts[item.alias] = item; });
    renderInspector();
    renderPalette();
    if (announce) showMessage(`已同步 ${state.datasets.length} 个数据集登记`, 'ok');
  } catch (error) {
    if (announce) showError(error);
  }
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
  const refresh = document.createElement('button'); refresh.type = 'button'; refresh.className = 'secondary dataset-refresh';
  refresh.textContent = '同步已登记数据集'; refresh.title = '读取另一处“数据集登记”面板保存的本地 catalog';
  refresh.onclick = () => refreshDatasets({announce: true}); section.appendChild(refresh);
  if (!alias) {
    const guide = document.createElement('div'); guide.className = 'dataset-selection-guide';
    guide.appendChild(Object.assign(document.createElement('strong'), {textContent: '这里选择数据集'}));
    guide.appendChild(Object.assign(document.createElement('p'), {textContent: state.datasets.length ? '打开上方 dataset 下拉框，选择一个可用数据源。' : '当前没有已登记数据；可以先使用内置 synthetic 验证算法连接。'}));
    const actions = document.createElement('div'); actions.className = 'dataset-guide-actions';
    const synthetic = document.createElement('button'); synthetic.type = 'button'; synthetic.textContent = '使用 synthetic'; synthetic.onclick = chooseSyntheticDataset; actions.appendChild(synthetic);
    guide.appendChild(actions); section.appendChild(guide);
    return section;
  }
  if (!facts) {
    section.appendChild(Object.assign(document.createElement('div'), {textContent: '正在读取数据集事实…'}));
    return section;
  }
  if (!datasetIsUsable(facts) && alias !== 'synthetic') {
    const warning = document.createElement('div'); warning.className = 'dataset-warning';
    warning.appendChild(Object.assign(document.createElement('strong'), {textContent: '数据源尚未就绪'}));
    warning.appendChild(Object.assign(document.createElement('p'), {textContent: `当前状态：${facts.status || 'unknown'}。请检查资源，或切换到 synthetic 先验证流程。`}));
    const synthetic = document.createElement('button'); synthetic.type = 'button'; synthetic.textContent = '切换 synthetic'; synthetic.onclick = chooseSyntheticDataset; warning.appendChild(synthetic);
    section.appendChild(warning);
  }
  const rows = [
    ['名称', facts.name], ['状态', facts.status], ['路径', facts.path], ['adapter', facts.adapter],
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

function renderUndoDelete() {
  const button = $('undo-delete');
  if (!button) return;
  button.hidden = !state.deletedStep;
  button.onclick = () => {
    const deleted = state.deletedStep;
    if (!deleted || !deleted.array) return;
    deleted.array.splice(Math.min(deleted.index, deleted.array.length), 0, deleted.step);
    state.deletedStep = null;
    state.selected = deleted.step;
    state.activeInsertionTarget = {parentId: deleted.parentId, index: deleted.index, context: deleted.context};
    revealInsertionTarget(state.activeInsertionTarget);
    markDirty();
    draw();
  };
}

function insertionComposite(steps, target) {
  if (target.compositeId) return steps.find((step) => step._uiComposite?.id === target.compositeId)?._uiComposite;
  const left = steps[target.index - 1]?._uiComposite;
  const right = steps[target.index]?._uiComposite;
  return left && left.id === right?.id ? left : null;
}

function insertStep(parentId, index, step, insertionTarget = {index}) {
  const target = getChildrenArray(parentId);
  if (!target) return false;
  const group = insertionComposite(target, insertionTarget);
  if (group) step._uiComposite = group;
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
  const group = insertionComposite(targetArray, target);
  location.array.splice(location.index, 1);
  let index = target.index;
  if (location.array === targetArray && location.index < target.index) index -= 1;
  targetArray.splice(Math.max(0, Math.min(index, targetArray.length)), 0, step);
  if (group) step._uiComposite = group;
  else delete step._uiComposite;
  markDirty();
  state.activeInsertionTarget = { ...target, index };
  revealInsertionTarget(target);
  state.selected = step;
  return true;
}

function addStepAtTarget(blockId, target) {
  const info = blockInfo(blockId);
  if (!target || !canInsert(info, target.parentId, null, target.index, true)) return false;
  const step = newStep(blockId);
  if (state.paletteDraft?.block === blockId && state.paletteDraft.recipe === state.recipe) step.params = {...state.paletteDraft.params};
  const available = availableKeysBefore(target);
  (info.requires || []).forEach((name) => {
    const current = slotValue(info, step, name);
    const inferred = inferAvailableSlot(info, name, available, step);
    if (inferred && !available.has(current)) step.params[name] = inferred;
  });
  (info.provides || []).forEach((name) => {
    const inferred = inferOutputSlot(info, name, target);
    if (inferred) step.params[name] = inferred;
  });
  if (!insertStep(target.parentId, target.index, step, target)) return false;
  markDirty();
  state.selected = step;
  state.activeInsertionTarget = { ...target, index: target.index + 1 };
  // Make an inserted nested step immediately visible.  Without expanding its
  // containing composite/loop, the recipe changes but the new block appears
  // to have vanished from the workspace.
  revealInsertionTarget(target);
  return true;
}

function addDataConceptAtTarget(group, target) {
  if (!group || !target) return false;
  if (getPlacementContext(target.parentId) !== 'top') return false;
  const inserted = [];
  for (let offset = 0; offset < group.blocks.length; offset += 1) {
    const blockId = group.blocks[offset];
    const info = blockInfo(blockId);
    const stepTarget = {...target, index: target.index + offset};
    if (!info || !canInsert(info, stepTarget.parentId, null, stepTarget.index, true)) return false;
    if (!addStepAtTarget(blockId, stepTarget)) return false;
    inserted.push(state.selected);
  }
  state.dataOverviewSelection = {groupId: group.id, stepIds: inserted.map((step) => step._uiId)};
  state.selected = inserted[0] || null;
  state.paletteSelection = null;
  state.activeInsertionTarget = {...target, index: target.index + group.blocks.length};
  return Boolean(inserted.length);
}

function markDropZones() {
  document.querySelectorAll('.drop-target').forEach((zone) => {
    const target = findDropTarget(zone);
    const movingId = state.drag?.type === 'step' ? state.drag.id : null;
    const info = state.drag?.info || (state.drag?.type === 'new' ? blockInfo(state.drag.id) : null);
    const active = Boolean(state.activeInsertionTarget
      && state.activeInsertionTarget.parentId === target.parentId
      && state.activeInsertionTarget.index === target.index
      && (state.activeInsertionTarget.compositeId || null) === target.compositeId);
    zone.classList.toggle('drop-active', active);
    zone.classList.toggle('active', active);
    const valid = Boolean(info && canInsert(info, target.parentId, movingId, target.index, !movingId));
    zone.classList.toggle('drop-valid', valid);
    zone.classList.toggle('drop-invalid', Boolean(info && !valid));
    const baseLabel = zone.dataset.baseLabel || dropZoneLabel(target.parentId, target.index, target.context);
    // During a drag, keep the drop zone itself as the explanation surface:
    // valid targets are green, while invalid targets state the exact contract
    // or placement reason instead of relying on colour alone.
    if (info) {
      const pending = valid && availabilityReason(info, target, movingId);
      zone.textContent = valid ? `${baseLabel} ${pending ? '○ 可放置，之后连接输入' : '✓ 可放置'}` : `✕ ${baseLabel}（${availabilityReason(info, target, movingId)}）`;
    } else {
      zone.textContent = baseLabel;
    }
  });
  renderPositionIndicator();
}

function handleDrop(event, zone) {
  event.preventDefault();
  const target = findDropTarget(zone);
  const dataGroupId = event.dataTransfer.getData('application/x-lnl-data-group');
  const newBlockId = event.dataTransfer.getData('application/x-lnl-new-block');
  const stepId = event.dataTransfer.getData('application/x-lnl-step');
  const dataGroup = dataGroupId ? DATA_OVERVIEW_GROUPS.find((group) => group.id === dataGroupId) : null;
  const reason = dataGroup
    ? availabilityReason(dataConceptInfo(dataGroup), target)
    : newBlockId
    ? availabilityReason(blockInfo(newBlockId), target, null, null, true)
    : stepId ? availabilityReason(blockInfo(findStepById(stepId)?.block), target, stepId) : '未识别拖动内容';
  const success = !reason && (dataGroup
    ? addDataConceptAtTarget(dataGroup, target)
    : newBlockId ? addStepAtTarget(newBlockId, target) : moveStepToTarget(stepId, target));
  state.drag = null;
  if (!success) {
    showMessage(`✕ 不能插入到${target.context}层：${reason || '插入失败'}`, 'error');
    markDropZones();
    return;
  }
  draw();
}

function createDropZone(parentId, index, context, compositeId = null) {
  const zone = document.createElement('div');
  zone.className = 'drop-target';
  zone.dataset.parentId = parentId;
  zone.dataset.index = String(index);
  zone.dataset.context = context;
  if (compositeId) zone.dataset.compositeId = compositeId;
  zone.dataset.baseLabel = dropZoneLabel(parentId, index, context);
  zone.textContent = zone.dataset.baseLabel;
  zone.onclick = (event) => { event.stopPropagation(); setActiveInsertionTarget(findDropTarget(zone)); };
  zone.ondragover = (event) => {
    const info = state.drag?.info;
    if (info && canInsert(info, parentId, state.drag.type === 'step' ? state.drag.id : null, index, state.drag.type === 'new')) {
      event.preventDefault();
      zone.classList.add('drop-hover');
    }
  };
  zone.ondragleave = () => zone.classList.remove('drop-hover');
  zone.ondrop = (event) => { zone.classList.remove('drop-hover'); handleDrop(event, zone); };
  return zone;
}

function dropZoneLabel(parentId, index, context) {
  if (context === 'batch') {
    const siblings = getChildrenArray(parentId) || [];
    const before = siblings.slice(0, index);
    const afterForward = before.some((step) => ['forward', 'forward_feature', 'forward_with_state'].includes(step.block));
    const hasUpdate = before.some((step) => ['backward', 'optimizer_step', 'virtual_parameter_update'].includes(step.block));
    if (afterForward && !hasUpdate) return '+ 添加公式、选择、状态或其他算法步骤';
    if (hasUpdate || index >= siblings.length) return '+ 添加模型更新';
    return '+ 添加 Batch 操作';
  }
  if (context === 'epoch') return '+ 添加 Epoch 级操作';
  return '+ 添加实验步骤';
}

function renderPalette() {
  const palette = $('palette');
  const rail = $('category-rail');
  if (!palette || !rail) return;
  palette.innerHTML = '';
  rail.innerHTML = '';
  const query = state.paletteQuery.trim().toLowerCase();
  const target = state.activeInsertionTarget;
  const categoryMatches = (item) => state.paletteCategory === '全部'
    || (state.paletteCategory === '我的' ? isUserOwnedInfo(item) : uiCategory(item) === state.paletteCategory);
  const paletteItems = dataPaletteItems();
  const searchable = paletteItems.filter((item) => item.beginner_visible && (!query || [
    item.name, item.id, item.category, item.ui_group, item.description,
    ...(item.requires || []), ...(item.provides || []),
  ].join(' ').toLowerCase().includes(query)) && categoryMatches(item));
  const placementMatches = (item) => !target || placementAllows(item, target.parentId);
  // Position-incompatible blocks stay discoverable while searching, but the
  // normal palette shows only blocks that can live in the selected container.
  const visible = searchable.filter((item) => placementMatches(item) || Boolean(query));
  const hiddenByPlacement = searchable.length - visible.length;
  const availableKeys = target ? availableKeysBefore(target) : new Set();
  const recommendationScore = (item) => {
    if (!target || !placementMatches(item)) return -1;
    const required = item.requires || [];
    const ready = required.filter((name) => availableKeys.has(slotValue(item, null, name))).length;
    const score = required.length ? ready / required.length : 0.25;
    return (required.length && ready === required.length ? 100 : 0)
      + score * 30
      + (item.formula_kind ? 15 : 0)
      + (item.stage === 'train' && target.context === 'batch' ? 5 : 0);
  };
  const recommended = query || !target ? [] : [...visible]
    .map((item, index) => ({item, score: recommendationScore(item), index}))
    .filter(({score}) => score >= 100)
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .slice(0, 8)
    .map(({item}) => item);
  const targetSummary = $('palette-target-summary');
  if (targetSummary) targetSummary.textContent = target ? targetLabel(target) : '请先点击中间的插入位置';
  const targetLocate = $('palette-target-locate');
  if (targetLocate) {
    targetLocate.hidden = !state.paletteSelection;
    targetLocate.onclick = () => state.paletteSelection && locateBlockInsertion(state.paletteSelection);
  }
  UI_CATEGORIES.forEach((category) => {
    const categoryVisible = visible.filter((item) => uiCategory(item) === category.name);
    const railButton = document.createElement('button');
    railButton.type = 'button';
    railButton.dataset.category = category.name;
    railButton.style.setProperty('--category-color', category.color);
    railButton.classList.toggle('active', state.paletteCategory === category.name);
    railButton.setAttribute('aria-pressed', String(state.paletteCategory === category.name));
    const icon = document.createElement('span'); icon.className = 'rail-icon'; icon.textContent = category.icon;
    const label = document.createElement('span'); label.textContent = category.name;
    railButton.append(icon, label);
    railButton.onclick = () => { state.paletteCategory = category.name; renderPalette(); };
    rail.appendChild(railButton);
  });
  const count = $('palette-count');
  if (count) count.textContent = `${visible.length} 个${hiddenByPlacement ? `（隐藏 ${hiddenByPlacement} 个位置不符）` : ''}`;
  if (state.paletteCategory === '数据') {
    const modeNote = document.createElement('div');
    modeNote.className = 'palette-data-mode-note';
    modeNote.textContent = state.dataBlockMode === 'canonical'
      ? '当前数据视图：原始 12 个执行积木'
      : '当前数据视图：5 个合并数据积木（插入后仍展开为原始执行步骤）';
    palette.appendChild(modeNote);
  }

  if (state.paletteCategory === '我的') {
    const actions = document.createElement('div'); actions.className = 'my-palette-actions';
    [['新建公式', '用数学运算创建一个用户公式', () => { resetFormulaEditor(); $('formula-editor-dialog').showModal(); }],
      ['公式模板与我的公式', '打开内置公式模板，或编辑已保存的用户公式', showMyFormulas],
      ['我的组合块', '组合块只在界面中折叠，不会新增运行时 Block', () => showMessage('组合块是 UI 视图；展开或解除组合不会改变 Recipe。')]].forEach(([name, description, action]) => {
      const card = document.createElement('button'); card.type = 'button'; card.className = 'my-palette-card';
      card.innerHTML = `<strong>${name}</strong><small>${description}</small>`; card.onclick = action; actions.appendChild(card);
    });
    palette.appendChild(actions);
  }
  if (state.paletteCategory === '算法操作') {
    const actions = document.createElement('div'); actions.className = 'my-palette-actions';
    const custom = document.createElement('button'); custom.type = 'button'; custom.className = 'my-palette-card';
    custom.innerHTML = '<strong>＋ 创建自定义公式</strong><small>把已有公式积木组合成一个可复用的用户公式。</small>';
    custom.onclick = () => { resetFormulaEditor(); $('formula-editor-dialog').showModal(); };
    actions.appendChild(custom); palette.appendChild(actions);
  }
  if (!visible.length) {
    palette.appendChild(Object.assign(document.createElement('div'), {className: 'palette-empty', textContent: query ? '没有匹配的积木' : state.paletteCategory === '我的' ? '暂无用户公式；从上方创建一个。' : '该分类暂无可见积木'}));
    renderPositionIndicator();
    return;
  }
  if (recommended.length) {
    const heading = document.createElement('div');
    heading.className = 'palette-section-heading';
    heading.innerHTML = '<strong>推荐下一步</strong><small>根据当前位置和已有输入自动筛选</small>';
    palette.appendChild(heading);
    const recommendationBar = document.createElement('div');
    recommendationBar.className = 'palette-recommendations';
    recommended.forEach((item) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'palette-recommendation';
      button.dataset.blockId = item.id;
      button.textContent = item.name;
      button.title = item.description || item.name;
      button.onclick = () => {
        state.paletteSelection = item;
        state.selected = null;
        paletteDraftFor(item);
        renderInspector();
        renderPalette();
      };
      recommendationBar.appendChild(button);
    });
    palette.appendChild(recommendationBar);
  }
  const groups = new Map();
  visible.forEach((item) => { const group = item.formula_group || item.category || '通用'; if (!groups.has(group)) groups.set(group, []); groups.get(group).push(item); });
  groups.forEach((items, category) => {
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
    label.textContent = `${category} (${items.length})`;
    toggle.append(indicator, label);
    toggle.onclick = () => {
      if (state.paletteCollapsed.has(category)) state.paletteCollapsed.delete(category);
      else state.paletteCollapsed.add(category);
      renderPalette();
    };
    title.appendChild(toggle);
    palette.appendChild(title);
    if (collapsed) return;
    items.forEach((item) => {
      const dataGroup = item.dataGroupId ? DATA_OVERVIEW_GROUPS.find((group) => group.id === item.dataGroupId) : null;
      const node = document.createElement('div');
      node.className = 'palette-block';
      node.dataset.category = blockCategory(item);
      node.dataset.uiCategory = uiCategory(item);
      node.dataset.blockId = item.id;
      node.classList.toggle('palette-selected', Boolean(state.paletteSelection?.id === item.id && !state.selected));
      const header = document.createElement('div'); header.className = 'palette-block-header';
      const title = document.createElement('strong'); title.textContent = item.name;
      const handle = document.createElement('span'); handle.className = 'drag-handle'; handle.textContent = '⠿'; handle.title = '拖动把手：添加到绿色插入位'; handle.setAttribute('role', 'img'); handle.setAttribute('aria-label', '拖动把手');
      header.append(title, handle); node.appendChild(header);
      const reason = target ? availabilityReason(item, target) : '请先选择插入位置';
      const placeable = target && !availabilityReason(item, target, null, null, true);
      const status = document.createElement('div'); status.className = 'palette-availability';
      if (!target) { status.dataset.state = 'empty'; status.textContent = '○ 先选插入位'; }
      else if (reason) { status.dataset.state = placeable ? 'empty' : 'invalid'; status.textContent = placeable ? `○ 待连接：${reason}` : `✕ ${reason}`; }
      else { status.dataset.state = 'valid'; status.textContent = '✓ 可添加'; }
      node.appendChild(status);
      const actions = document.createElement('div'); actions.className = 'palette-block-actions';
      const add = document.createElement('button'); add.type = 'button'; add.className = 'palette-add';
      const recommended = target && reason ? recommendedInsertionTarget(item) : null;
      add.textContent = !target ? '选择插入位' : placeable ? (reason ? '添加并配置' : '添加到当前位置') : recommended ? '添加到推荐位置' : '查看放置限制';
      add.disabled = !target;
      add.title = reason || '添加到当前插入位置';
      add.onclick = (event) => {
        event.stopPropagation();
        state.paletteSelection = item;
        state.selected = null;
        paletteDraftFor(item);
        if (!placeable && recommended) {
          revealInsertionTarget(recommended);
          const inserted = dataGroup ? addDataConceptAtTarget(dataGroup, recommended) : addStepAtTarget(item.id, recommended);
          if (inserted) { state.paletteSelection = null; draw(); return; }
        }
        if (!placeable) { renderInspector(); renderPalette(); setInspectorTab('blocks'); return; }
        const inserted = dataGroup ? addDataConceptAtTarget(dataGroup, target) : addStepAtTarget(item.id, target);
        if (inserted) { state.paletteSelection = null; draw(); }
        else showMessage('添加失败：请重新选择一个绿色插入位置', 'error');
      };
      actions.appendChild(add); node.appendChild(actions);
      handle.draggable = true;
      handle.ondragstart = (event) => {
        state.paletteSelection = item;
        state.selected = null;
        renderInspector();
        state.drag = { type: 'new', id: item.id, info: item };
        if (dataGroup) event.dataTransfer.setData('application/x-lnl-data-group', dataGroup.id);
        else event.dataTransfer.setData('application/x-lnl-new-block', item.id);
        event.dataTransfer.effectAllowed = 'copy';
        markDropZones();
      };
      handle.ondragend = () => { state.drag = null; markDropZones(); };
      node.onclick = (event) => {
        if (event.target.closest('.drag-handle, .palette-add')) return;
        // Clicking a palette item only selects it; insertion is an explicit
        // action or a drag from the handle.
        state.paletteSelection = item;
        state.selected = null;
        state.dataOverviewSelection = null;
        renderInspector();
        renderPalette();
      };
      node.title = `${item.description || ''}\n${reason || '可添加到当前插入位置'}\n点击查看详情；使用“添加到当前位置/推荐位置”或拖动把手`;
      palette.appendChild(node);
    });
  });
  renderPositionIndicator();
}

function renderStepSummary(step, info) {
  if (!info) return '';
  const inputs = (info.requires || []).slice(0, 3).map((name) => slotValue(info, step, name));
  const outputs = (info.provides || []).slice(0, 3).map((name) => slotValue(info, step, name));
  const inputText = inputs.length ? inputs.join(', ') : '无输入';
  const outputText = outputs.length ? outputs.join(', ') : '无输出';
  return `${inputText} → ${outputText}`;
}

function formatParamValue(value) {
  if (value === undefined) return '未设置';
  if (value === null) return 'null';
  if (typeof value === 'object') {
    try { return JSON.stringify(value); } catch (error) { return String(value); }
  }
  return String(value);
}

// Scratch keeps formula metadata as plain text so it can be saved in the
// existing Registry/Recipe format.  This small renderer turns the TeX-like
// notation used by the Registry into native MathML in the browser.  It is
// deliberately a renderer, not a second formula parser: it never evaluates
// expressions and always keeps the original source available as a fallback.
const MATH_NS = 'http://www.w3.org/1998/Math/MathML';
const MATH_COMMANDS = {
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', theta: 'θ',
  lambda: 'λ', mu: 'μ', pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ', phi: 'φ',
  psi: 'ψ', omega: 'ω', Gamma: 'Γ', Delta: 'Δ', Sigma: 'Σ', Phi: 'Φ',
  ell: 'ℓ', varepsilon: 'ϵ', varphi: 'ϕ', vartheta: 'ϑ', kappa: 'κ',
  nu: 'ν', xi: 'ξ', chi: 'χ', zeta: 'ζ',
  in: '∈', notin: '∉', le: '≤', ge: '≥', neq: '≠', ne: '≠', approx: '≈',
  geq: '≥', leq: '≤', partial: '∂', nabla: '∇', lceil: '⌈', rceil: '⌉',
  propto: '∝', times: '×', cdot: '·', div: '÷',
  pm: '±', rightarrow: '→', leftarrow: '←', infty: '∞', sum: 'Σ', prod: 'Π',
};
const MATH_OPERATOR_CHARS = new Set(['=', '+', '-', '*', '/', '·', '×', '±', '⊙',
  '(', ')', '[', ']', '{', '}', ',', ':', ';', '|', '‖', '→', '←', '≤', '≥',
  '≠', '<', '>', '∈', 'Σ', 'Π', '∑', '∏']);
const MATH_GREEK = new Set('αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ'.split(''));
const MATH_WORD_SYMBOLS = {sum: 'Σ', prod: 'Π'};
const MATH_OPERATOR_COMMANDS = new Set([
  'argmax', 'argmin', 'clip', 'cos', 'Cov', 'CE', 'count', 'det', 'EMA', 'exp',
  'GMM', 'log', 'max', 'mean', 'min', 'normalize', 'pinv', 'Quantile',
  'ReverseDLD', 'row-normalize', 'shape', 'softmax', 'swap', 'Top', 'Uniform', 'var',
]);

// Registry formulas are intentionally kept as runtime metadata.  A few of the
// historical strings were shorthand (or used ambiguous variable names), which
// made the formula bar show a mathematically different expression than the
// operation actually performs.  This display-only catalogue gives each
// canonical operation an explicit, indexed equation without changing execution
// metadata or user-authored formulas.
const DISPLAY_FORMULAS = Object.freeze({
  binary_risk: String.raw`\ell_{i,0}=-\log p_{i,0},\ \ell_{i,1}=-\log p_{i,1}; \tilde{\ell}_{i,0}=\frac{(1-\rho_+)\ell_{i,0}-\rho_-\ell_{i,1}}{1-\rho_+-\rho_-}; \tilde{\ell}_{i,1}=\frac{-\rho_+\ell_{i,0}+(1-\rho_-)\ell_{i,1}}{1-\rho_+-\rho_-}`,
  compose_revision_transition: String.raw`T'_i=\operatorname{row-normalize}([T_i+\Delta T]_+),\ [u]_+=\max(u,0)`,
  dld_accelerated_inference: String.raw`\hat{y}_0=\operatorname{ReverseDLD}(y_T; K=\text{steps})`,
  formula__builtin__apl: String.raw`\mathcal{L}_{\mathrm{APL}}=\lambda_A\frac{-\log p_{i,y_i}}{\sum_c -\log p_{i,c}}+\lambda_P\left(-\sum_{c\ne y_i}p_{i,c}\log \bar{y}_{i,c}\right);\ \log\bar{y}_{i,y_i}=0`,
  formula__builtin__confidence_score: String.raw`p_i=\operatorname{softmax}(z_i),\ s_i=p_{i,y_i}`,
  formula__builtin__gce: String.raw`\mathcal{L}_q=\frac{1}{N}\sum_i\frac{1-p_{i,y_i}^{q}}{q},\ p_i=\operatorname{softmax}(z_i),\ q>0`,
  formula__builtin__standard_ce: String.raw`\mathcal{L}=\frac{1}{N}\sum_i-\log p_{i,y_i},\ p_i=\operatorname{softmax}(z_i)`,
  formula__builtin__transition_corrected_risk: String.raw`\mathcal{L}=\frac{1}{N}\sum_i-\log[(p_iT_i)_{y_i}],\ p_i=\operatorname{softmax}(z_i)`,
  formula__builtin__weighted_ce: String.raw`\mathcal{L}=\frac{1}{N}\sum_i w_i[-\log p_{i,y_i}],\ p_i=\operatorname{softmax}(z_i)`,
  formula__builtin__fine_warmup_loss: String.raw`L_{\mathrm{warmup}}=\frac{1}{N}\sum_i-\log p_{i,\tilde{y}_i}`,
  softmax: String.raw`p_{i,c}=\frac{\exp(z_{i,c}/\tau)}{\sum_j\exp(z_{i,j}/\tau)},\ \tau>0`,
  cal_cores2_adjusted_risk: String.raw`\mathcal{L}=\frac{1}{N}\sum_i\left[-\log p_{i,y_i}-\alpha\sum_c\bar{\pi}_c\log p_{i,c}\right],\ \bar{\pi}_c=\frac{\sqrt{\pi_c}}{\sum_j\sqrt{\pi_j}}`,
  cal_covariance_correction: String.raw`\Delta=\sum_c\hat{\pi}_c\sum_j\operatorname{Cov}\left(1[\tilde{y}=j],\ell_j\mid\hat{y}=c\right)`,
  dss_masked_training_loss: String.raw`\mathcal{L}=\frac{1}{B}\sum_i s_i\left[\log\sum_{c\notin E_i}\exp(z_{i,c})-z_{i,y_i}\right]`,
  mc_ldce_objective: String.raw`J=1+\frac{1}{N}\sum_i\left\|h_iW^{\mathsf{T}}\right\|_2^2-2\langle W,\mu_{\mathrm{clean}}\rangle_F`,
  mc_ldce_volmin_objective: String.raw`\mathcal{L}=\frac{1}{N}\sum_i-\log[(p_iT)_{y_i}]+\lambda\log\det(T),\ p_i=\operatorname{softmax}(z_i)`,
  mean_loss: String.raw`L=\frac{1}{N}\sum_i\ell_i`,
  mean_squared_error: String.raw`e_i=\operatorname{mean}_d[(x_{i,d}-y_{i,d})^2],\ \operatorname{MSE}_{\mathrm{per\text{-}sample}}=e_i,\ \operatorname{MSE}_{\mathrm{scalar}}=\operatorname{mean}_i e_i`,
  nce_loss: String.raw`L_{\mathrm{NCE},i}=\frac{-\log p_{i,y_i}}{\sum_c[-\log p_{i,c}]},\ p_i=\operatorname{softmax}(z_i)`,
  per_sample_ce: String.raw`\ell_i=-\log p_{i,y_i},\ p_i=\operatorname{softmax}(z_i)`,
  rce_loss: String.raw`L_{\mathrm{RCE},i}=-\sum_c p_{i,c}\log\bar{y}_{i,c},\ \log\bar{y}_{i,y_i}=0,\ \log\bar{y}_{i,c\ne y_i}=\log\alpha`,
  sed_rejected_regularizer: String.raw`R_i=\beta\frac{\log p_{i,\tilde y_i}}{C}+\gamma\frac{-\log(1-p_{i,\tilde y_i^{\mathrm{comp}}})}{C},\ \tilde y_i^{\mathrm{comp}}\sim\operatorname{Uniform}(\mathcal{Y}\setminus\{\tilde y_i\})`,
  soft_target_cross_entropy: String.raw`\mathcal{L}=\frac{1}{N}\sum_i-\sum_c q_{i,c}\log p_{i,c},\ p_i=\operatorname{softmax}(z_i)`,
  symmetric_kl: String.raw`D_{\mathrm{SKL},i}=\sum_c p^a_{i,c}\log\frac{p^a_{i,c}}{p^b_{i,c}}+\sum_c p^b_{i,c}\log\frac{p^b_{i,c}}{p^a_{i,c}}`,
  apply_transition: String.raw`\tilde{p}_i=p_iT_i`,
  backward_correction: String.raw`\tilde{\ell}_{i,y_i}=[\ell_iT^{-\mathsf{T}}]_{y_i}`,
  complementary_negative_loss: String.raw`L_i=-\sum_c m_{i,c}\log\left(1-\sigma(z_{i,c})\right)`,
  mae_loss: String.raw`L_i=\frac{1}{C}\sum_c\left|\operatorname{softmax}(z_i)_c-1[y_i=c]\right|`,
  masked_mean: String.raw`L(d=selected)=\frac{\sum_i m_i v_i}{\max(\sum_i m_i,\varepsilon)},\quad L(d=batch)=\frac{1}{N}\sum_i v_i`,
  partial_label_loss: String.raw`\mathcal{L}_i=\lambda\left[-\sum_c w_{i,c}h_{i,c}\log p_{i,c}\right]+(1-\lambda)\left[-\sum_c w_{i,c}q_{i,c}\log p_{i,c}\right],\quad q_{i,c}=\frac{c_{i,c}}{\sum_jc_{i,j}},\ h_i=\operatorname{onehot}(\arg\max_c c_{i,c}),\ \lambda=hard_weight`,
  prior_kl: String.raw`D=\sum_c\pi_c\log\frac{\pi_c}{\bar{p}_c},\ \bar{p}=\frac{1}{N}\sum_i p_i`,
  weighted_sum: String.raw`y=\sum_{k=1}^{K}w_kx_k`,
  constant: String.raw`z=c`,
  reciprocal: String.raw`z=1/x`,
  mask_to_indices: String.raw`I=\operatorname{where}(m)`,
  formula__builtin__gather_by_label: String.raw`v_i=V_{i,y_i}`,
  formula__builtin__negative_log: String.raw`z=-\log(\max(x,\varepsilon))`,
  formula__builtin__safe_divide: String.raw`z=\frac{x}{\max(y,\varepsilon)}`,
  formula__builtin__row_normalize: String.raw`z_{i,c}=\frac{x_{i,c}}{\max(\sum_jx_{i,j},\varepsilon)}`,
  formula__builtin__weighted_blend: String.raw`z=w\,left+(1-w)\,right`,
  formula__builtin__sharpen_distribution: String.raw`q'_c=\frac{q_c^{1/T}}{\sum_jq_j^{1/T}}`,
  formula__builtin__soft_target_cross_entropy: String.raw`L=-\operatorname{mean}_i\sum_cq_{i,c}\log p_{i,c}`,
  formula__builtin__masked_mean: String.raw`L=\operatorname{mean}_{i:m_i=1}v_i`,
  formula__builtin__mean_squared_error: String.raw`L=\operatorname{mean}((x-y)^2)`,
  formula__builtin__weighted_sum: String.raw`y=\sum_kw_kx_k`,
  formula__builtin__per_sample_ce: String.raw`\ell_i=-\log p_{i,y_i}`,
  formula__builtin__nce_loss: String.raw`L_i=\frac{-\log p_{i,y_i}}{\sum_c-\log p_{i,c}}`,
  formula__builtin__rce_loss: String.raw`L_i=-\sum_cp_{i,c}\log\bar y_{i,c}`,
  formula__builtin__symmetric_kl: String.raw`D_{\mathrm{SKL}}=\operatorname{KL}(p_a\Vert p_b)+\operatorname{KL}(p_b\Vert p_a)`,
  formula__builtin__mean_loss: String.raw`L=\operatorname{mean}_i\ell_i`,
  formula__builtin__mean_by_indices: String.raw`L=\operatorname{mean}_{i\in I}v_i`,
  formula__builtin__compose_transition: String.raw`T=\operatorname{row\_normalize}(T_1T_2)`,
  formula__builtin__mae_loss: String.raw`L_i=\operatorname{mean}_c|p_{i,c}-1[y_i=c]|`,
  formula__builtin__prior_kl: String.raw`D=\sum_c\pi_c\log(\pi_c/\bar p_c)`,
  formula__builtin__normalize_nonnegative_weights: String.raw`w=\bar w/\sum_j\bar w_j`,
  formula__builtin__binary_risk: String.raw`\tilde{\ell}_{i,y}=\frac{(1-\rho_y)\ell_{i,y}-\rho_{1-y}\ell_{i,1-y}}{1-\rho_+-\rho_-}`,
  formula__builtin__cal_cores2_adjusted_risk: String.raw`\mathcal{L}=\operatorname{mean}_i\left[\ell_{i,y_i}-\alpha\sum_c\bar\pi_c\ell_{i,c}\right]`,
  formula__builtin__cnlcu_soft_score: String.raw`\ell_i^*=\tilde\mu_i-\frac{\sigma^2\left(t_i+\sigma^2\log(2t_i)/t_i^2\right)}{n_i-\sigma^2}`,
  formula__builtin__one_hot_like: String.raw`O_{i,c}=1[y_i=c],\ c=1,\ldots,C=\operatorname{shape}_{-1}(R)`,
  formula__builtin__t_revision_importance_ratio: String.raw`w_i=\frac{g_{\tilde y_i}(x_i)}{\tilde g_{\tilde y_i}(x_i)}`,
  virtual_parameter_update: String.raw`\theta'=\theta-\alpha\nabla_{\theta}L`,
  step_milestone_update: String.raw`\operatorname{lr}_t=\operatorname{lr}_0\gamma^{\sum_m1[t\ge m]},\ t=\operatorname{epoch}\cdot S+\operatorname{batch}+1`,
  cwd_global_objective: String.raw`\mathcal{L}=1+\frac{1}{N}\sum_i(h_i^{\mathsf{T}}w+b)^2-2w^{\mathsf{T}}(\mu_1-\mu_0)-2b(\pi_1-\pi_0)`,
  cwd_virtual_systems: String.raw`C^{(k)}=\sum_{s,t}\tilde{\pi}^{(k)}_s\tilde{T}^{(k)}_{s,t}P_{s,t}^{\mathsf{T}}`,
  cwd_coefficient_pseudoinverse: String.raw`C^{(k)+}=\operatorname{pinv}(C^{(k)})`,
  cwd_observed_statistics: String.raw`\tilde{\pi}_c=\frac{n_c}{N},\ \tilde{\mu}_c=\frac{1}{N}\sum_i h_i1[\tilde{y}_i=c]`,
  cwd_recover_centroids: String.raw`\hat{M}=\left[\sum_k\tilde{M}C^{(k)+}-(C-1)\tilde{M}\right]^{\mathsf{T}}`,
  dld_sample_forward_state: String.raw`y_t=y_0+\bar{\alpha}_t d+\bar{\beta}_t\epsilon,\ d=y_n-y_0`,
  dld_pre_correct_labels: String.raw`d_i=y_{n,i}-y_{0,i},\ (y_0,y_n)=\operatorname{PreCorrect}(p_w,p_s,\tilde{y})`,
  dss_ccs_trend_exclusion: String.raw`E_{i,c}=1\left[z_{i,c}>\Phi^{-1}(1-\alpha)\right],\ E_{i,y_i}=0`,
  dss_mda_marginal_adjustment: String.raw`p'_{i,c}=\frac{p_{i,c}/(Cm_c)}{\sum_jp_{i,j}/(Cm_j)}`,
  dss_warmup_lifecycle: String.raw`S\leftarrow\operatorname{on\_cycle\_start}(S,t)`,
  create_fine_state: String.raw`\operatorname{EMA}_t=m\operatorname{EMA}_{t-1}+(1-m)f_t;\ \operatorname{SCS/SCR}\text{ update epoch snapshots}`,
  fine_scr_reweight: String.raw`w_i=\exp\left[-\frac{(\max_c p_{i,c}-\hat{\mu}_{\hat{c}_i})^2}{2\hat{\sigma}_{\hat{c}_i}^2/n_\sigma^2}\right]`,
  fine_scs_select: String.raw`clean_i=1\left[p_{i,\tilde{y}_i}\ge\tau_{\mathrm{global}}\,\tau_{\mathrm{class}(\tilde{y}_i)}\right]`,
  fine_snapshot_predictions: String.raw`p_{t,i}=f_t(x_i),\ p^{\mathrm{EMA}}_{t,i}=f^{\mathrm{EMA}}_t(x_i),\ \text{aligned by stable index}`,
  fine_warmup_loss: String.raw`L_{\mathrm{warmup}}=\frac{1}{N}\sum_i-\log p_{i,\tilde{y}_i}`,
  fit_gmm: String.raw`w_i=P(z_i=\mathrm{clean}\mid\ell_i),\ z_i\sim\mathrm{GMM}_2`,
  mc_ldce_recover_statistic: String.raw`\hat{\mu}=\left(\tilde{\mu}\,A^+\right)^{\mathsf{T}},\ A=\sum_{i,j}\pi_iT_{i,j}\,\operatorname{swap}(i,j)^{\mathsf{T}}`,
  create_mentor_provider: String.raw`q_t=mq_{t-1}+(1-m)\operatorname{Quantile}_{p}(\ell_t)\quad(\text{provider state})`,
  pcse_recover_layer_statistics: String.raw`\mu=R^{\mathsf{T}}\tilde{\mu},\ S=R^{\mathsf{T}}\tilde{S},\ \Sigma=S-\mu\mu^{\mathsf{T}}`,
  upm_update_eta: String.raw`\eta_i\leftarrow\Pi_{[0,1]}\left(\eta_i+\alpha\frac{\partial\log p(\tilde{y}_i\mid x_i)}{\partial\eta_i}\right)`,
  masked_gradient_update: String.raw`g_i\leftarrow\alpha m_i g_i+\lambda\operatorname{sign}(\theta_i),\ m_i=1[i\in S]`,
  parameter_criticality_mask: String.raw`s_j=|g_j\theta_j|,\ S=\operatorname{Top}_k(s),\ k=\lceil(1-\tau)m\rceil,\ \tau=\text{noise rate}`,
  cal_materialize_proxy_artifact: String.raw`\text{proxy}=\operatorname{CORES}^2(\operatorname{argmax} f_{\mathrm{warmup}}(x),\ \ell_{\mathrm{adjusted}};\ \ell\in[\ell_{\min},\ell_{\max}])`,
  cnlcu_soft_score: String.raw`\ell_i^*=\tilde{\mu}_i-\frac{\sigma^2\left(t_i+\sigma^2\log(2t_i)/t_i^2\right)}{n_i-\sigma^2}`,
  indices_to_mask: String.raw`m_j=1[j\in I]`,
  mean_by_indices: String.raw`\operatorname{mean}_{j\in I}v_j=\frac{1}{|I|}\sum_{j\in I}v_j`,
  numel: String.raw`n=|x|`,
  class_count: String.raw`C=\operatorname{shape}(R)_{-1}`,
  select_class_column: String.raw`v_i=V_{i,c_0}`,
  select_by_indices: String.raw`v_{\mathrm{selected}}=(v_j)_{j\in I}`,
  select_lowest_scores: String.raw`I=\operatorname{argsort}_{\mathrm{stable\ score\ then\ index}}(s)_{1:k},\ k=\min\!\left(N,\max\!\left(m_{\min},R_r(Nf)\right)\right)`,
  linear_rate_schedule: String.raw`r(t)=r_0+\operatorname{clip}(t/T,0,1)(r_1-r_0)`,
  create_dss_state: String.raw`S=(H,M,Z,A,E)\quad\text{(indexed history, marginal, trend, selected, excluded)}`,
  indexed_accumulate: String.raw`S_i\leftarrow S_i+v_i`,
  affine_transform: String.raw`z=\alpha x+\beta`,
  add: String.raw`z=x+y`,
  subtract: String.raw`z=x-y`,
  divide: String.raw`z=x\div y`,
  strict_divide: String.raw`z=\frac{x}{y},\ y>\varepsilon`,
  safe_divide: String.raw`z=\frac{x}{\max(y,\varepsilon)}`,
  negative_log: String.raw`z=-\log\left(\max(x,\varepsilon)\right)`,
  detach: String.raw`z=\operatorname{stopgrad}(x)`,
  one_hot: String.raw`O_{i,c}=1[y_i=c],\ c=1,\ldots,C`,
  ones_like: String.raw`z_i=1,\ \operatorname{shape}(z)=\operatorname{shape}(x),\ \operatorname{dtype}(z)=float32`,
  zeros_like: String.raw`z_i=0,\ \operatorname{shape}(z)=\operatorname{shape}(x),\ \operatorname{dtype}(z)=\operatorname{dtype}(x)`,
  row_normalize: String.raw`z_{i,c}=\frac{x_{i,c}}{\max(\sum_jx_{i,j},\varepsilon)}`,
  positive_logdet: String.raw`z=\log\det(X),\ \det(X)>0`,
  uniform_prior: String.raw`\pi_c=\frac{1}{C},\ c=1,\ldots,C`,
  reduce_sum: String.raw`z=\sum_{\mathrm{dim}}x`,
  reduce_mean: String.raw`z=\operatorname{mean}_{\mathrm{dim}}(x)`,
  reduce_max: String.raw`z=\max_{\mathrm{dim}}(x)`,
  reduce_min: String.raw`z=\min_{\mathrm{dim}}(x)`,
  reshape_tensor: String.raw`z=\operatorname{reshape}(x,\mathrm{shape})`,
  flatten_per_sample: String.raw`z_i=\operatorname{reshape}(x_i,-1)`,
  broadcast_sample_weight: String.raw`w'=\operatorname{reshape}(w,\operatorname{rank}(x))`,
  unsqueeze: String.raw`z=\operatorname{unsqueeze}(x,\mathrm{dim})`,
  squeeze: String.raw`z=\operatorname{squeeze}(x,\mathrm{dim})`,
  transpose_dims: String.raw`z=\operatorname{transpose}(x,d_0,d_1)`,
  compare: String.raw`m=(x\ \operatorname{op}\ y)`,
  where: String.raw`z=\operatorname{where}(m,x,y)`,
  logical_not: String.raw`z=\neg m`,
  logical_and: String.raw`z=m_1\land m_2`,
  logical_or: String.raw`z=m_1\lor m_2`,
  argmax: String.raw`z=\operatorname{argmax}_{\mathrm{dim}}(x)`,
  gather: String.raw`z=\operatorname{gather}(x,i,\mathrm{dim})`,
  index_select: String.raw`z=\operatorname{index\_select}(x,i,\mathrm{dim})`,
  batched_matmul: String.raw`Z_i=X_iY_i`,
  maximum: String.raw`z=\max(x,y)`,
  minimum: String.raw`z=\min(x,y)`,
  matrix_multiply: String.raw`Z=XY`,
  exp: String.raw`z=\exp(x)`,
  log: String.raw`z=\log(x)`,
  sqrt: String.raw`z=\sqrt{x}`,
  abs: String.raw`z=|x|`,
  sign: String.raw`z=\operatorname{sign}(x)`,
  log_softmax: String.raw`\log p=\log\operatorname{softmax}(z)`,
  clamp_min: String.raw`z=\max(x,c)\quad\text{(elementwise)}`,
  elementwise_multiply: String.raw`z_i=x_i y_i`,
  elementwise_power: String.raw`z_i=x_i^{q}`,
  gather_by_label: String.raw`v_i=V_{i,y_i}`,
  negate: String.raw`z_i=-x_i`,
  one_hot_like: String.raw`O_{i,c}=1[y_i=c],\ c=1,\ldots,C=\operatorname{shape}_{-1}(R)`,
  sharpen_distribution: String.raw`q'_{i,c}=\frac{q_{i,c}^{1/T}}{\sum_jq_{i,j}^{1/T}}`,
  sum_last_dimension: String.raw`z_i=\sum_cx_{i,c}`,
  sum_values: String.raw`s=\sum_i x_i`,
  weighted_blend: String.raw`z_i=w_i x_i+(1-w_i)y_i,\ w_i\in[0,1]`,
  compose_transition: String.raw`T_{i,k}=\sum_jT^{(1)}_{i,j}T^{(2)}_{j,k},\ \bar{T}_{i,k}=\frac{T_{i,k}}{\sum_lT_{i,l}}`,
  dual_t_transition_estimation: String.raw`T=T_{\mathrm{club}}T_{\mathrm{spade}},\ (T_{\mathrm{spade}})_{a,b}=\frac{\#\{\operatorname{argmax} q=a,\tilde{y}=b\}}{\#\{\operatorname{argmax} q=a\}}`,
  pdl_fit_basis_matrices: String.raw`M_c=\operatorname{argmin}_{M\ge0}\sum_r\left\|W_{a_c(r)}M-q_{a_c(r)}\right\|_2^2`,
  pdl_estimate_instance_transition: String.raw`T(x)=\sum_{r=1}^{R}\beta_r(x)M_r`,
  pdl_fit_part_representation: String.raw`\operatorname{min}_{H,W\ge0}\|X-WH\|_F^2,\ \text{then normalize rows of }W`,
  initialize_t_revision_transition: String.raw`\hat{T}_{c,:}=q(x_c),\ x_c=\operatorname{argmax}_x q_c(x)`,
  importance_weight_formula: String.raw`w_i=\operatorname{detach}\!\left[\frac{q_{i,\tilde{y}_i}-\rho_{\mathrm{opp}(\tilde{y}_i)}}{(1-\rho_0-\rho_1)q_{i,\tilde{y}_i}}\right]_+`,
  mentor_build_features: String.raw`v_i=(\ell_i,\ell_i-q_t,y_i,e_t)`,
  mentor_predict_sample_weights: String.raw`w_i=M(v_i)\quad\text{with burn-in and dropout policy}`,
  nonnegative_projection: String.raw`w_i=[s x_i]_+=\max(sx_i,0),\ s\in\{+1,-1\}`,
  normalize_nonnegative_weights: String.raw`w_i=\bar{w}_i/\sum_j\bar{w}_j\ (\sum_j\bar{w}_j>0);\ w_i=0\ (\sum_j\bar{w}_j=0)`,
  t_revision_importance_ratio: String.raw`w_i=\frac{g_{\tilde{y}_i}(x_i)}{(g(x_i)T_i)_{\tilde{y}_i}}`,
});

function formulaDisplayText(info) {
  if (typeof info === 'string') return DISPLAY_FORMULAS[info] || info;
  if (!info) return '';
  return DISPLAY_FORMULAS[info.id] || info.formula || '';
}

// Not every Formula Editor operation has a standalone equation.  Keep those
// operations useful in the palette by showing their registry description
// instead of the old empty-formula placeholder.  The fallback is deliberately
// generic so newly registered formula operations get an explanation
// without a UI-specific block-id branch.
function formulaEditorOperationIntro(info) {
  if (!info) return '这是一个可组合的数学运算。';
  const description = String(info.description || '').trim();
  if (description) return description;
  const name = String(info.name || info.id || '这个运算').trim();
  const inputCount = Array.isArray(info.requires) ? info.requires.length : 0;
  if (!inputCount) return `${name}：根据当前参数生成一个公式结果。`;
  if (inputCount === 1) return `${name}：接收一个输入并生成新的张量或数值。`;
  return `${name}：组合 ${inputCount} 个输入，生成新的张量或数值。`;
}

function renderFormulaOrIntro(container, info, {compact = false} = {}) {
  if (!container) return;
  const formula = formulaDisplayText(info);
  if (formula) {
    renderMathFormula(container, formula, {compact});
    return;
  }
  container.replaceChildren();
  container.classList.add('formula-operation-intro');
  container.classList.toggle('formula-operation-intro-compact', compact);
  container.textContent = formulaEditorOperationIntro(info);
}

function mathElement(tag, text = null) {
  const element = document.createElementNS(MATH_NS, tag);
  if (text !== null) element.textContent = text;
  return element;
}

function mathRow(...children) {
  const row = mathElement('mrow');
  children.filter(Boolean).forEach((child) => row.appendChild(child));
  return row;
}

function mathGroupEnd(source, start, opening = '{', closing = '}') {
  let depth = 0;
  for (let index = start; index < source.length; index += 1) {
    if (source[index] === opening) depth += 1;
    else if (source[index] === closing) {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return -1;
}

function normalizeMathSource(value) {
  return String(value || '')
    .replace(/\\(?:left|right|quad|qquad|[,;!])/g, ' ')
    .replace(/\\\s+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function parseMathAtom(source, start) {
  let index = start;
  while (index < source.length && /\s/.test(source[index])) index += 1;
  if (index >= source.length) return {node: null, next: index};
  const character = source[index];
  if (character === '{') {
    const end = mathGroupEnd(source, index);
    if (end > index) return {node: parseMathExpression(source.slice(index + 1, end)), next: end + 1};
  }
  if (character === '(' || character === '[') {
    const closing = character === '(' ? ')' : ']';
    const end = mathGroupEnd(source, index, character, closing);
    if (end > index) {
      const group = mathRow(mathElement('mo', character), parseMathExpression(source.slice(index + 1, end)), mathElement('mo', closing));
      return {node: group, next: end + 1};
    }
  }
  if (character === '\\') {
    const match = source.slice(index + 1).match(/^[A-Za-z]+/);
    if (match) {
      const command = match[0];
      const next = index + 1 + command.length;
      if (command === 'frac') {
        const numerator = parseMathAtom(source, next);
        const denominator = parseMathAtom(source, numerator.next);
        const fraction = mathElement('mfrac');
        fraction.append(numerator.node || mathElement('mi', '□'), denominator.node || mathElement('mi', '□'));
        return {node: fraction, next: denominator.next};
      }
      if (command === 'sqrt') {
        const content = parseMathAtom(source, next);
        const root = mathElement('msqrt'); root.append(content.node || mathElement('mi', '□'));
        return {node: root, next: content.next};
      }
      if (['text', 'mathrm', 'operatorname', 'mathbf', 'mathbb', 'mathcal', 'mathsf'].includes(command)) {
        const content = parseMathAtom(source, next);
        if (['operatorname', 'text'].includes(command)) {
          return {node: mathElement('mtext', content.node?.textContent || ''), next: content.next};
        }
        const node = content.node || mathElement('mi', '□');
        node.setAttribute('mathvariant', command === 'mathbb' ? 'double-struck' : command === 'mathcal' ? 'script' : command === 'mathbf' ? 'bold' : 'normal');
        return {node, next: content.next};
      }
      if (['tilde', 'hat', 'bar', 'vec'].includes(command)) {
        const content = parseMathAtom(source, next);
        const accent = {tilde: '˜', hat: 'ˆ', bar: '¯', vec: '→'}[command];
        const mover = mathElement('mover'); mover.append(content.node || mathElement('mi', '□'), mathElement('mo', accent));
        mover.setAttribute('accent', 'true');
        return {node: mover, next: content.next};
      }
      if (MATH_OPERATOR_COMMANDS.has(command)) {
        return {node: mathElement('mo', command), next};
      }
      const symbol = MATH_COMMANDS[command] || command;
      const node = MATH_GREEK.has(symbol) ? mathElement('mi', symbol) : mathElement('mo', symbol);
      if (!MATH_GREEK.has(symbol) && /^[A-Za-z]+$/.test(symbol)) node.setAttribute('mathvariant', 'normal');
      return {node, next};
    }
    return {node: mathElement('mo', '\\'), next: index + 1};
  }
  if (/\d/.test(character)) {
    const match = source.slice(index).match(/^\d+(?:\.\d+)?/);
    return {node: mathElement('mn', match[0]), next: index + match[0].length};
  }
  if (/[A-Za-z]/.test(character) || MATH_GREEK.has(character)) {
    let end = index + 1;
    while (end < source.length && /[A-Za-z]/.test(source[end])) end += 1;
    const word = source.slice(index, end);
    const symbol = MATH_WORD_SYMBOLS[word];
    const node = symbol ? mathElement('mo', symbol) : mathElement('mi', word);
    if (!symbol && word.length > 1) node.setAttribute('mathvariant', 'normal');
    return {node, next: end};
  }
  if (MATH_OPERATOR_CHARS.has(character)) return {node: mathElement('mo', character), next: index + 1};
  return {node: mathElement('mtext', character), next: index + 1};
}

function applyMathScript(nodes, script, kind) {
  if (!nodes.length) return;
  const base = nodes.pop();
  const tag = base.localName || base.tagName?.toLowerCase();
  if (kind === 'sub' && tag === 'msup') {
    const combined = mathElement('msubsup');
    combined.append(base.firstElementChild, script, base.lastElementChild);
    nodes.push(combined);
    return;
  }
  if (kind === 'sup' && tag === 'msub') {
    const combined = mathElement('msubsup');
    combined.append(base.firstElementChild, base.lastElementChild, script);
    nodes.push(combined);
    return;
  }
  const scripted = mathElement(kind === 'sub' ? 'msub' : 'msup');
  scripted.append(base, script); nodes.push(scripted);
}

function parseMathExpression(value) {
  const source = normalizeMathSource(value);
  const nodes = [];
  let index = 0;
  while (index < source.length) {
    if (/\s/.test(source[index])) { index += 1; continue; }
    if (source[index] === '^' || source[index] === '_') {
      const parsed = parseMathAtom(source, index + 1);
      if (parsed.node) applyMathScript(nodes, parsed.node, source[index] === '^' ? 'sup' : 'sub');
      index = parsed.next;
      continue;
    }
    if (source[index] === "'") {
      if (nodes.length) applyMathScript(nodes, mathElement('mo', '′'), 'sup');
      index += 1; continue;
    }
    const parsed = parseMathAtom(source, index);
    if (!parsed.node) { index = parsed.next; continue; }
    // A slash is rendered as a fraction for the immediately adjacent atoms;
    // grouped numerators/denominators therefore remain visually readable.
    if (parsed.node.localName === 'mo' && parsed.node.textContent === '/' && nodes.length) {
      const right = parseMathAtom(source, parsed.next);
      if (right.node) {
        const numerator = nodes.pop();
        const fraction = mathElement('mfrac'); fraction.append(numerator, right.node); nodes.push(fraction);
        index = right.next; continue;
      }
    }
    nodes.push(parsed.node); index = parsed.next;
  }
  return mathRow(...nodes);
}

function splitMathLines(value) {
  const source = String(value || '');
  const lines = []; let start = 0; let depth = 0;
  for (let index = 0; index < source.length; index += 1) {
    if ('{(['.includes(source[index])) depth += 1;
    else if ('})]'.includes(source[index])) depth = Math.max(0, depth - 1);
    else if (source[index] === ';' && depth === 0) { lines.push(source.slice(start, index)); start = index + 1; }
  }
  lines.push(source.slice(start));
  return lines.map((line) => line.trim()).filter(Boolean);
}

function renderMathFormula(container, formula, {compact = false} = {}) {
  if (!container) return;
  container.replaceChildren();
  const source = normalizeMathSource(formula);
  container.classList.add('math-rendered');
  container.classList.toggle('math-rendered-compact', compact);
  if (!source || /^(暂无公式|暂无独立公式|无独立公式)$/.test(source)) {
    const empty = document.createElement('span'); empty.className = 'math-empty'; empty.textContent = source || '暂无公式'; container.appendChild(empty); return;
  }
  splitMathLines(source).forEach((line) => {
    const math = mathElement('math');
    math.setAttribute('display', 'block');
    math.setAttribute('aria-label', line);
    math.appendChild(parseMathExpression(line));
    const row = document.createElement('div'); row.className = 'math-line'; row.appendChild(math); container.appendChild(row);
  });
  const sourceNode = document.createElement('span');
  sourceNode.className = 'math-source'; sourceNode.textContent = source; sourceNode.title = '公式展示记号';
  container.appendChild(sourceNode);
}

function renderFormulaPreview(container, formula, label = '公式预览') {
  if (!container) return;
  container.replaceChildren();
  const heading = document.createElement('strong'); heading.className = 'math-preview-heading'; heading.textContent = label; container.appendChild(heading);
  const rendered = document.createElement('div'); renderMathFormula(rendered, formula); container.appendChild(rendered);
}

function flatRecipeSteps(steps = state.recipe.steps) {
  return (steps || []).flatMap((step) => [step, ...(Array.isArray(step.steps) ? flatRecipeSteps(step.steps) : [])]);
}

const COMPOSITE_DEFINITIONS = [
  { id: 'environment', label: '实验环境', icon: '⚙', blocks: ['set_seed', 'select_device'], description: '固定随机性并选择运行设备。' },
  { id: 'prepare-data', label: '数据', icon: '▦', blocks: ['load_dataset', 'inspect_dataset_semantics', 'create_dataset_split', 'select_label_source', 'apply_noise', 'build_noise_manifest', 'configure_preprocessing', 'configure_views', 'assign_data_roles', 'configure_loader', 'build_prepared_data', 'build_loaders'], description: '用五个用户概念配置数据；展开后可查看完整的 Scratch 数据执行步骤。' },
  { id: 'model-optimization', label: '模型与优化', icon: '◈', blocks: ['create_model', 'create_optimizer'], description: '按模型/优化器依赖将连续的单模型或多模型初始化步骤成组展示。' },
  { id: 'training-loop', label: '训练循环', icon: '↻', blocks: ['epoch_loop'], description: 'Epoch / Batch 的 C 形训练容器。' },
  { id: 'prepare-batch', label: '准备 Batch', icon: '▤', blocks: ['get_batch', 'move_batch_to_device'], description: '读取 Batch 并移动到当前设备。' },
  { id: 'update-model', label: '更新模型', icon: '↟', blocks: ['backward', 'optimizer_step'], description: '反向传播并执行优化器更新。' },
  { id: 'validate-best', label: '验证并保留最佳', icon: '✓', blocks: ['evaluate_accuracy', 'track_best_model'], description: '评估当前模型并按指标保留最佳状态。' },
  { id: 'final-evaluation', label: '最终评估', icon: '◒', blocks: ['restore_best_model', 'evaluate_accuracy', 'record_metrics'], description: '恢复最佳状态并记录最终测试指标。', optionalPrefix: true },
];

// Data remains a single executable composite, but its collapsed presentation
// follows the concepts a noise-learning researcher actually configures.  The
// mapping is display-only: the canonical twelve data blocks and their slots
// remain unchanged underneath.
const DATA_OVERVIEW_GROUPS = [
  { id: 'dataset', label: '数据集', icon: '▦', blocks: ['load_dataset', 'inspect_dataset_semantics'], description: '选择训练源和测试源，并确认类别与标签能力。' },
  { id: 'split', label: '数据划分', icon: '÷', blocks: ['create_dataset_split', 'select_label_source'], description: '决定训练/验证如何划分，以及各角色使用哪种标签。' },
  { id: 'noise', label: '标签噪声', icon: '≈', blocks: ['apply_noise', 'build_noise_manifest'], description: '设置噪声类型、比例和随机种子；manifest 自动记录样本对应关系。' },
  { id: 'input', label: '输入与增强', icon: '✣', blocks: ['configure_preprocessing', 'configure_views'], description: '把原始样本变成模型输入，并按需要提供 weak/strong 视图。' },
  { id: 'batch', label: '训练批次', icon: '▤', blocks: ['assign_data_roles', 'configure_loader', 'build_prepared_data', 'build_loaders'], description: '设置 batch 并把数据交给 train、validation、test 等用途。' },
];
const DATA_CANONICAL_BLOCK_IDS = new Set(DATA_OVERVIEW_GROUPS.flatMap((group) => group.blocks));
const DATA_CONCEPT_PREFIX = 'data-concept:';

function dataConceptInfo(group) {
  return {
    id: `${DATA_CONCEPT_PREFIX}${group.id}`,
    dataGroupId: group.id,
    name: group.label,
    category: 'Data',
    ui_group: '① 数据准备',
    stage: 'data',
    kind: 'action',
    beginner_visible: true,
    description: group.description,
    requires: [],
    provides: [],
    placement: ['top'],
    params: {},
  };
}

function isDataConceptInfo(info) {
  return Boolean(info?.dataGroupId && String(info.id || '').startsWith(DATA_CONCEPT_PREFIX));
}

function dataConceptById(id) {
  const group = DATA_OVERVIEW_GROUPS.find((item) => `${DATA_CONCEPT_PREFIX}${item.id}` === id);
  return group ? dataConceptInfo(group) : null;
}

function dataPaletteItems() {
  const canonical = state.blocks.filter((item) => !DATA_CANONICAL_BLOCK_IDS.has(item.id));
  if (state.dataBlockMode === 'canonical') return state.blocks;
  return [...canonical, ...DATA_OVERVIEW_GROUPS.map(dataConceptInfo)];
}

function dataOverviewStep(steps, block) {
  return (steps || []).find((step) => step?.block === block) || null;
}

function dataOverviewSummary(group, steps) {
  const first = dataOverviewStep(steps, group.blocks[0]);
  const find = (block) => dataOverviewStep(steps, block);
  if (group.id === 'dataset') return `数据集：${first?.params?.dataset || '未选择'}`;
  if (group.id === 'split') {
    const split = find('create_dataset_split');
    const size = split?.params?.validation_size;
    return `验证集：${size === undefined ? '自动' : size} · 标签按 Recipe 保护`;
  }
  if (group.id === 'noise') {
    const noise = find('apply_noise');
    const name = noise?.params?.name || 'none';
    const rate = noise?.params?.rate;
    return `类型：${name} · 比例：${rate === undefined ? '—' : rate}`;
  }
  if (group.id === 'input') {
    const preprocessing = find('configure_preprocessing');
    const views = find('configure_views');
    const viewNames = Array.isArray(views?.params?.views) ? views.params.views.join(' / ') : 'weak';
    return `预处理：${preprocessing?.params?.preprocessing || 'standard'} · 视图：${viewNames}`;
  }
  const loader = find('configure_loader');
  const roles = find('assign_data_roles');
  const roleNames = Array.isArray(roles?.params?.roles) ? roles.params.roles.join('、') : 'train / test';
  return `Batch：${loader?.params?.batch_size || '—'} · 用途：${roleNames}`;
}

function selectDataOverviewGroup(group, steps) {
  const members = group.blocks.map((block) => dataOverviewStep(steps, block)).filter(Boolean);
  if (!members.length) return;
  state.dataOverviewSelection = {groupId: group.id, stepIds: members.map((step) => step._uiId)};
  state.selected = members[0];
  state.paletteSelection = null;
  state.adjacentSelection = null;
  draw();
}

function renderDataConcepts(steps) {
  const overview = document.createElement('div');
  overview.className = 'data-concepts';
  overview.setAttribute('aria-label', '数据设置模块');
  DATA_OVERVIEW_GROUPS.forEach((group) => {
    const item = document.createElement('div');
    const members = group.blocks.map((block) => dataOverviewStep(steps, block)).filter(Boolean);
    const selected = state.dataOverviewSelection?.groupId === group.id
      && members.some((step) => state.dataOverviewSelection.stepIds?.includes(step._uiId));
    item.className = `data-submodule${selected ? ' selected' : ''}`;
    item.dataset.dataGroup = group.id;
    item.setAttribute('role', 'button');
    item.setAttribute('tabindex', '0');
    item.setAttribute('aria-pressed', selected ? 'true' : 'false');
    item.title = '点击查看并编辑该数据模块的参数';
    const heading = document.createElement('div');
    heading.className = 'data-submodule-heading';
    const icon = document.createElement('span');
    icon.className = 'data-submodule-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = group.icon;
    const title = document.createElement('strong');
    title.textContent = group.label;
    heading.append(icon, title);
    const summary = document.createElement('span');
    summary.className = 'data-submodule-summary';
    summary.textContent = dataOverviewSummary(group, steps);
    const description = document.createElement('small');
    description.textContent = group.description;
    const edit = document.createElement('span');
    edit.className = 'data-submodule-edit';
    edit.textContent = members.length ? '点击编辑参数 →' : '暂无可编辑步骤';
    item.append(heading, summary, description, edit);
    item.onclick = () => selectDataOverviewGroup(group, steps);
    item.onkeydown = (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        selectDataOverviewGroup(group, steps);
      }
    };
    overview.appendChild(item);
  });
  const hint = document.createElement('p');
  hint.className = 'data-concepts-hint';
  hint.textContent = '五个模块按执行顺序排列。点击任一模块即可在右侧编辑其中每个实际数据积木的参数；底层 12 步仍保持不变。';
  overview.appendChild(hint);
  return overview;
}

function renderDataOverviewInspector(selection, target, explanationTarget) {
  const group = DATA_OVERVIEW_GROUPS.find((item) => item.id === selection?.groupId);
  if (!group) return false;
  const members = (selection.stepIds || []).map((id) => findStepById(id)).filter(Boolean);
  if (!members.length) return false;
  const title = document.createElement('strong');
  title.textContent = group.label;
  const description = document.createElement('p');
  description.textContent = `${group.description} 下面的参数直接写回对应的 Scratch 数据积木。`;
  explanationTarget.append(title, description);
  members.forEach((step) => {
    const info = blockInfo(step.block);
    if (!info) return;
    const section = document.createElement('section');
    section.className = 'inspector-section data-overview-inspector-section';
    const heading = document.createElement('h3');
    heading.textContent = info.name || step.block;
    section.appendChild(heading);
    const detail = document.createElement('p');
    detail.className = 'data-overview-step-description';
    detail.textContent = info.description || '';
    section.appendChild(detail);
    Object.entries(info.params || {}).forEach(([name, schema]) => {
      if (isDatasetSourceInfo(info) && name === 'path' && step.params?.source_mode !== 'custom_path') return;
      const wrap = document.createElement('div');
      wrap.className = 'param';
      const label = document.createElement('label');
      label.textContent = isDatasetSourceInfo(info) && name === 'dataset' ? 'dataset（必选）' : name;
      wrap.append(label, renderParamControl(name, schema, step));
      section.appendChild(wrap);
    });
    target.appendChild(section);
    if (isDatasetSourceInfo(info)) target.appendChild(renderDatasetFacts(step));
    if ((info.provides || []).some((name) => name === 'noise_state' || name === 'noisy_train_split')) {
      target.appendChild(renderNoiseFacts(step));
    }
  });
  return true;
}

function renderDataAdvancedSummary(steps) {
  const details = document.createElement('details');
  details.className = 'data-advanced-settings';
  const summary = document.createElement('summary');
  summary.textContent = '高级数据设置';
  details.appendChild(summary);
  const body = document.createElement('div');
  body.className = 'data-advanced-body';
  const label = dataOverviewStep(steps, 'select_label_source');
  const roles = dataOverviewStep(steps, 'assign_data_roles');
  const views = dataOverviewStep(steps, 'configure_views');
  const loader = dataOverviewStep(steps, 'configure_loader');
  const rows = [
    ['标签使用', `训练：${label?.params?.train || 'observed'} · 验证：${label?.params?.validation || 'clean'} · 测试：${label?.params?.test || 'clean'}`],
    ['数据用途', Array.isArray(roles?.params?.roles) ? roles.params.roles.join('、') : 'train、test'],
    ['视图', Array.isArray(views?.params?.views) ? views.params.views.join('、') : 'weak'],
    ['Loader 细节', `workers：${loader?.params?.num_workers ?? 'auto'} · drop_last：${loader?.params?.drop_last ? '是' : '否'}`],
  ];
  rows.forEach(([name, value]) => {
    const row = document.createElement('div');
    row.className = 'data-advanced-row';
    const label = document.createElement('strong');
    label.textContent = name;
    const content = document.createElement('span');
    content.textContent = value;
    row.append(label, content);
    body.appendChild(row);
  });
  const hint = document.createElement('p');
  hint.textContent = '这些选项影响实验语义或防止标签泄漏；论文模板可由 Recipe 自动锁定。需要修改时，请展开实际执行步骤后编辑对应积木。';
  body.appendChild(hint);
  details.appendChild(body);
  return details;
}

const MODEL_OPTIMIZATION_BLOCKS = new Set(['create_model', 'create_optimizer', 'create_parameter_group_optimizer']);

function detectModelOptimizationRange(steps, start) {
  if (steps?.[start]?.block !== 'create_model') return null;
  let end = start;
  let modelCount = 0;
  let optimizerCount = 0;
  while (end < (steps || []).length && MODEL_OPTIMIZATION_BLOCKS.has(steps[end]?.block)) {
    if (steps[end].block === 'create_model') modelCount += 1;
    else optimizerCount += 1;
    end += 1;
  }
  // A model-only prefix is not an optimization group. Keeping it as a normal
  // step also avoids swallowing a later, unrelated optimizer setup.
  if (modelCount === 0 || optimizerCount === 0) return null;
  const modelSlots = new Set();
  for (let index = start; index < end; index += 1) {
    const step = steps[index];
    if (step.block === 'create_model') {
      modelSlots.add(String(step.params?.save_as || 'model'));
      continue;
    }
    // Parameter-group optimizers resolve their source slots from `groups`;
    // the model/optimizer composite should not second-guess that list.
    if (step.block === 'create_parameter_group_optimizer') continue;
    // A single-module optimizer is only part of this group when its model has
    // already been created in the same contiguous setup segment.
    if (!modelSlots.has(String(step.params?.model || 'model'))) return null;
  }
  return { start, end, modelCount, optimizerCount };
}

function detectCompositeRanges(steps) {
  const ranges = [];
  let index = 0;
  while (index < (steps || []).length) {
    const membership = steps[index]._uiComposite;
    if (membership) {
      let end = index + 1;
      while (end < steps.length && steps[end]._uiComposite?.id === membership.id) end += 1;
      const definition = COMPOSITE_DEFINITIONS.find((item) => item.id === membership.definitionId);
      if (definition) {
        const members = steps.slice(index, end);
        ranges.push({definition, start: index, end, groupId: membership.id,
          modelCount: members.filter((step) => step.block === 'create_model').length,
          optimizerCount: members.filter((step) => ['create_optimizer', 'create_parameter_group_optimizer'].includes(step.block)).length});
        index = end;
        continue;
      }
    }
    let match = null;
    for (const definition of COMPOSITE_DEFINITIONS) {
      const ids = definition.blocks;
      if (definition.id === 'model-optimization') {
        const dynamic = detectModelOptimizationRange(steps, index);
        if (dynamic) match = { definition, ...dynamic };
        if (match) break;
        continue;
      }
      const exact = ids.every((id, offset) => steps[index + offset]?.block === id);
      const optional = definition.optionalPrefix && steps[index]?.block === 'evaluate_accuracy'
        && Boolean(steps[index]?.params?.final) && steps[index + 1]?.block === 'record_metrics';
      if (exact || optional) {
        match = { definition, start: index, end: index + (optional ? 2 : ids.length) };
        break;
      }
    }
    if (match && steps.slice(match.start, match.end).every((step) => !step._uiComposite)) {
      const membership = {id: makeUiId(), definitionId: match.definition.id};
      steps.slice(match.start, match.end).forEach((step) => { step._uiComposite = membership; });
      match.groupId = membership.id;
      ranges.push(match); index = match.end;
    } else index += 1;
  }
  return ranges;
}

function compositeKey(parentId, range) {
  return `${parentId}:${range.groupId || range.steps?.[range.start]?._uiComposite?.id}`;
}

function appendLoopIterationControl(container, step) {
  if (!step || step.block !== 'epoch_loop') return;
  const info = blockInfo(step.block);
  const schema = info?.params?.epochs || {type: 'int', min: 0, default: 1};
  step.params = step.params || {};
  const wrap = document.createElement('label');
  wrap.className = 'loop-iterations';
  wrap.title = '直接修改训练循环执行的 epoch 轮次';
  const label = document.createElement('span');
  label.textContent = '轮次';
  const input = document.createElement('input');
  input.className = 'loop-iterations-input';
  input.type = 'number';
  input.min = schema.min ?? 0;
  if (schema.max !== undefined) input.max = schema.max;
  input.step = '1';
  input.value = String(step.params.epochs ?? schema.default ?? 1);
  input.setAttribute('aria-label', '训练轮次');
  const commit = () => {
    const value = Number.parseInt(input.value, 10);
    const minimum = Number(schema.min ?? 0);
    const maximum = schema.max === undefined ? Number.POSITIVE_INFINITY : Number(schema.max);
    if (!Number.isFinite(value) || value < minimum || value > maximum) {
      input.value = String(step.params.epochs ?? schema.default ?? 1);
      input.setAttribute('aria-invalid', 'true');
      return;
    }
    input.removeAttribute('aria-invalid');
    if (step.params.epochs !== value) {
      step.params.epochs = value;
      markDirty();
      renderRuntimeLimits();
    }
  };
  input.oninput = (event) => event.stopPropagation();
  input.onchange = (event) => { event.stopPropagation(); commit(); };
  input.onkeydown = (event) => event.stopPropagation();
  wrap.append(label, input);
  container.appendChild(wrap);
}

function createSchedulerForEpoch(epochStep) {
  const info = blockInfo('create_scheduler');
  if (!info || !epochStep) return false;
  if (flatRecipeSteps().some((step) => step.block === 'create_scheduler')) {
    showMessage('Create Scheduler 已经添加；当前 Recipe 只能通过此入口添加一次。', 'error');
    return false;
  }
  const epochIndex = state.recipe.steps.indexOf(epochStep);
  if (epochIndex < 0) {
    showMessage('当前 Epoch Loop 已失效，请重新选择训练循环。', 'error');
    return false;
  }
  const targets = collectInsertionTargets()
    .filter((target) => target.parentId === '__root__' && target.index <= epochIndex)
    .sort((left, right) => right.index - left.index);
  let selectedTarget = null;
  let optimizer = null;
  for (const target of targets) {
    const available = availableKeysBefore(target);
    const candidate = available.has('optimizer')
      ? 'optimizer'
      : [...available].find((key) => /optimizer/i.test(String(key)));
    if (!candidate) continue;
    if (!availabilityReason(info, target, null, {optimizer: candidate})) {
      selectedTarget = target;
      optimizer = candidate;
      break;
    }
  }
  if (!selectedTarget) {
    showMessage('无法添加 Create Scheduler：请先在 Epoch Loop 前创建并连接一个 optimizer。', 'error');
    return false;
  }
  const draft = paletteDraftFor(info);
  draft.params = {...draft.params, optimizer};
  revealInsertionTarget(selectedTarget);
  state.activeInsertionTarget = selectedTarget;
  if (!addStepAtTarget('create_scheduler', selectedTarget)) {
    showMessage('Create Scheduler 添加失败，请检查 optimizer 连接。', 'error');
    return false;
  }
  state.paletteSelection = null;
  state.paletteDraft = null;
  draw();
  showMessage(`已在 Epoch Loop 前添加 Create Scheduler（${optimizer}）。可在右侧继续编辑参数。`, 'ok');
  return true;
}

function appendEpochAdvancedOptions(container, step) {
  if (!step || step.block !== 'epoch_loop') return;
  const details = document.createElement('details');
  details.className = 'epoch-advanced';
  const summary = document.createElement('summary');
  summary.textContent = '高级选项';
  const menu = document.createElement('div');
  menu.className = 'epoch-advanced-menu';
  const addScheduler = document.createElement('button');
  addScheduler.type = 'button';
  const schedulerExists = flatRecipeSteps().some((candidate) => candidate.block === 'create_scheduler');
  addScheduler.textContent = schedulerExists ? 'Create Scheduler 已添加' : '添加 Create Scheduler';
  addScheduler.disabled = schedulerExists;
  addScheduler.title = schedulerExists ? '当前 Recipe 已有 Create Scheduler' : '只添加一次学习率调度器';
  addScheduler.onclick = (event) => {
    event.stopPropagation();
    if (createSchedulerForEpoch(step)) details.open = false;
  };
  const hint = document.createElement('small');
  hint.textContent = '调度器是训练配置，会自动放在 Epoch Loop 前的合法位置。';
  menu.append(addScheduler, hint);
  details.append(summary, menu);
  details.onclick = (event) => event.stopPropagation();
  container.appendChild(details);
}

function renderStepNode(step, index, steps, parent, parentId, context, options = {}) {
  const info = blockInfo(step.block);
  const node = document.createElement('div');
  const loopClass = info && info.kind !== 'action' ? `loop-container loop-block ${step.block === 'epoch_loop' ? 'epoch-loop' : step.block === 'batch_loop' ? 'batch-loop' : ''}` : '';
  const formulaClass = info?.formula ? 'formula-block' : '';
  const isSelected = state.selected === step || (state.selected?._uiId && state.selected._uiId === step._uiId);
  node.className = `step ${loopClass} ${formulaClass}${isSelected ? ' selected' : ''}${state.errorStepId === step._uiId ? ' error-step' : ''}`;
  node.dataset.category = blockCategory(info);
  node.dataset.uiCategory = uiCategory(info);
  node.dataset.uiId = step._uiId;
  const header = document.createElement('div');
  header.className = 'step-header';
  const title = document.createElement('span'); title.className = 'step-title'; title.textContent = info ? info.name : step.block; header.appendChild(title);
  appendLoopIterationControl(header, step);
  if (!options.suppressEpochAdvanced) appendEpochAdvancedOptions(header, step);
  const kind = document.createElement('span'); kind.className = 'kind'; kind.textContent = info ? info.kind : ''; header.appendChild(kind);
  const summary = renderStepSummary(step, info);
  node.appendChild(header);
  if (summary) { const summaryNode = document.createElement('div'); summaryNode.className = 'step-summary'; summaryNode.textContent = summary; node.appendChild(summaryNode); }
  node.onclick = (event) => {
    if (event.target.closest('button, input, select, textarea, .drag-handle')) return;
    event.stopPropagation();
    state.dataOverviewSelection = null;
    state.selected = step;
    state.paletteSelection = null;
    state.activeInsertionTarget = {parentId, index: index + 1, context: getPlacementContext(parentId), compositeId: step._uiComposite?.id || null};
    renderInspector();
    draw();
  };
  const dragHandle = document.createElement('span');
  dragHandle.className = 'drag-handle';
  dragHandle.textContent = '⠿';
  dragHandle.title = '拖动把手：移动到绿色插入位';
  dragHandle.setAttribute('role', 'img');
  dragHandle.setAttribute('aria-label', '拖动把手');
  dragHandle.draggable = true;
  dragHandle.ondragstart = (event) => {
    state.drag = { type: 'step', id: step._uiId, info };
    state.selected = step;
    event.dataTransfer.setData('application/x-lnl-step', step._uiId);
    event.dataTransfer.effectAllowed = 'move';
    markDropZones();
  };
  dragHandle.ondragend = () => { state.drag = null; markDropZones(); };
  header.insertBefore(dragHandle, header.firstChild);
  if (info && info.kind !== 'action') {
    const toggle = document.createElement('button'); toggle.textContent = state.collapsed.has(step) ? '展开' : '折叠';
    toggle.onclick = (event) => { event.stopPropagation(); if (state.collapsed.has(step)) state.collapsed.delete(step); else state.collapsed.add(step); draw(); };
    header.appendChild(toggle);
    if (!state.collapsed.has(step)) {
      const nested = document.createElement('div'); nested.className = 'nested loop-body';
      const childContext = getPlacementContext(step._uiId);
      drawSteps(step.steps || (step.steps = []), nested, step._uiId, childContext); node.appendChild(nested);
    }
  }
  const controls = document.createElement('span'); controls.className = 'step-controls';
  const copy = document.createElement('button'); copy.textContent = '复制';
  copy.onclick = (event) => { event.stopPropagation(); const clone = typeof structuredClone === 'function' ? structuredClone(step) : JSON.parse(JSON.stringify(step)); ensureUiIds([clone]); clone._uiId = makeUiId(); (steps || []).splice(index + 1, 0, clone); markDirty(); state.selected = clone; draw(); };
  const remove = document.createElement('button'); remove.textContent = '删除';
  remove.onclick = (event) => {
    event.stopPropagation();
    state.deletedStep = {step, array: steps, index, parentId, context};
    (steps || []).splice(index, 1);
    markDirty();
    if (state.selected === step) state.selected = null;
    state.paletteSelection = null;
    state.activeInsertionTarget = {parentId, index, context};
    draw();
    showMessage(`已删除“${info?.name || step.block}”，可点击“撤销删除”恢复`, 'ok');
  };
  controls.append(copy, remove); node.appendChild(controls);
  parent.appendChild(node);
  return node;
}

function adjacentPairAt(steps, index) {
  const first = steps[index], second = steps[index + 1];
  if (!first || !second || first._uiComposite?.id !== second._uiComposite?.id) return null;
  if (first.block === 'set_seed' && second.block === 'select_device') {
    return {name: '设置实验环境', description: '设置随机种子，并选择计算设备。', members: [first, second]};
  }
  if (first.block === 'load_dataset' && second.block === 'inspect_dataset_semantics') {
    return {name: '加载并检查数据', description: '加载数据源和标签，然后检查数据的类别、标签和划分信息。', members: [first, second]};
  }
  return null;
}

function renderAdjacentPair(pair, index, steps, parent, parentId, context) {
  const first = pair.members[0];
  const key = first._uiId;
  const node = document.createElement('div');
  node.className = 'step adjacent-pair';
  node.dataset.uiCategory = uiCategory(blockInfo(first.block));
  node.dataset.pairId = key;
  node.classList.toggle('selected', state.selected === first && state.adjacentSelection?.members[0] === first);
  const header = document.createElement('div'); header.className = 'step-header';
  const title = document.createElement('strong'); title.className = 'step-title'; title.textContent = pair.name;
  const inspect = () => {
    state.dataOverviewSelection = null;
    state.selected = first;
    state.adjacentSelection = pair;
    state.paletteSelection = null;
    state.activeInsertionTarget = {parentId, index: index + 2, context, compositeId: first._uiComposite?.id || null};
    draw();
    setInspectorTab('blocks');
  };
  const edit = document.createElement('button'); edit.type = 'button'; edit.textContent = '编辑参数';
  edit.onclick = (event) => { event.stopPropagation(); inspect(); };
  const expand = document.createElement('button'); expand.type = 'button';
  expand.textContent = state.adjacentExpanded.has(key) ? '收起步骤' : '查看原步骤';
  expand.onclick = (event) => {
    event.stopPropagation();
    if (state.adjacentExpanded.has(key)) state.adjacentExpanded.delete(key); else state.adjacentExpanded.add(key);
    draw();
  };
  header.append(title, edit, expand); node.appendChild(header);
  const summary = document.createElement('div'); summary.className = 'step-summary';
  summary.textContent = first.block === 'set_seed'
    ? `随机种子：${first.params?.seed ?? blockInfo(first.block)?.params?.seed?.default} · 设备：${pair.members[1].params?.device ?? 'auto'}`
    : `数据集：${first.params?.dataset || '未选择'} · 包含数据语义检查`;
  node.appendChild(summary);
  node.onclick = (event) => { if (!event.target.closest('button, input, select, textarea, .drag-handle')) { event.stopPropagation(); inspect(); } };
  if (state.adjacentExpanded.has(key)) {
    const body = document.createElement('div'); body.className = 'nested';
    pair.members.forEach((step, offset) => renderStepNode(step, index + offset, steps, body, parentId, context));
    node.appendChild(body);
  }
  parent.appendChild(node);
}

function renderPairInspector(pair, target, explanationTarget) {
  const title = document.createElement('strong'); title.textContent = pair.name;
  const description = document.createElement('p'); description.textContent = pair.description;
  explanationTarget.append(title, description);
  for (const step of pair.members) {
    const info = blockInfo(step.block);
    const section = document.createElement('section'); section.className = 'inspector-section';
    const heading = document.createElement('h3'); heading.textContent = info.name; section.appendChild(heading);
    for (const [name, schema] of Object.entries(info.params || {})) {
      if (isDatasetSourceInfo(info) && name === 'path' && step.params?.source_mode !== 'custom_path') continue;
      const wrap = document.createElement('div'); wrap.className = 'param';
      const label = document.createElement('label'); label.textContent = name;
      wrap.append(label, renderParamControl(name, schema, step)); section.appendChild(wrap);
    }
    target.appendChild(section);
    if (isDatasetSourceInfo(info)) target.appendChild(renderDatasetFacts(step));
  }
}

function createCompositeShell(range, steps, parentId, context, key) {
  const definition = range.definition;
  const isDataComposite = definition.id === 'prepare-data';
  const showingDataSteps = isDataComposite && state.dataBlockMode === 'canonical';
  const shell = document.createElement('section'); shell.className = 'composite-block'; shell.dataset.compositeId = definition.id; shell.dataset.uiCategory = uiCategory({category: definition.id});
  const accent = {environment: '#f59e0b', 'prepare-data': '#38bdf8', 'model-optimization': '#a78bfa', 'training-loop': '#f59e0b', 'prepare-batch': '#38bdf8', 'update-model': '#facc15', 'validate-best': '#c4b5fd', 'final-evaluation': '#2dd4bf'}[definition.id] || '#38bdf8';
  shell.style.setProperty('--composite-accent', accent);
  const header = document.createElement('div'); header.className = 'composite-header';
  const countLabel = definition.id === 'model-optimization' && range.modelCount
    ? `${range.modelCount} 个模型 · ${range.optimizerCount} 个优化器 · ${range.end - range.start} 个步骤`
    : `${range.end - range.start} 个步骤`;
  const title = document.createElement('div'); title.className = 'composite-title'; title.innerHTML = `<span class="composite-icon">${definition.icon}</span><strong>${definition.label}</strong><small>${countLabel}</small>`;
  if (definition.id === 'training-loop') {
    appendEpochAdvancedOptions(title, steps[range.start]);
  }
  const actions = document.createElement('div'); actions.className = 'composite-actions';
  const toggle = document.createElement('button'); toggle.type = 'button';
  toggle.textContent = state.compositeExpanded.has(key) ? '收起' : '展开';
  toggle.onclick = (event) => { event.stopPropagation(); if (state.compositeExpanded.has(key)) state.compositeExpanded.delete(key); else state.compositeExpanded.add(key); draw(); };
  const ungroup = document.createElement('button'); ungroup.type = 'button'; ungroup.textContent = '解除组合';
  ungroup.onclick = (event) => { event.stopPropagation(); state.compositeUngrouped.add(key); draw(); };
  actions.append(toggle);
  if (isDataComposite && state.compositeExpanded.has(key)) {
    const switchView = document.createElement('button');
    switchView.type = 'button';
    switchView.textContent = showingDataSteps ? '返回 5 个数据模块' : '查看原始 12 步';
    switchView.title = showingDataSteps ? '返回数据区的用户概念视图' : '切换到可编辑的底层数据积木';
    switchView.onclick = (event) => {
      event.stopPropagation();
      if (showingDataSteps) {
        state.dataBlockMode = 'concepts';
        state.dataExecutionView.delete(key);
      } else {
        state.dataBlockMode = 'canonical';
        state.dataExecutionView.add(key);
      }
      state.dataOverviewSelection = null;
      state.paletteSelection = null;
      draw();
    };
    actions.appendChild(switchView);
  }
  actions.appendChild(ungroup); header.append(title, actions); shell.appendChild(header);
  if (isDataComposite && state.compositeExpanded.has(key) && !showingDataSteps) {
    const dataSteps = steps.slice(range.start, range.end);
    shell.appendChild(renderDataConcepts(dataSteps));
    shell.appendChild(renderDataAdvancedSummary(dataSteps));
  }
  const hint = document.createElement('p'); hint.className = 'composite-hint'; hint.textContent = definition.description; shell.appendChild(hint);
  if (state.compositeExpanded.has(key) && (!isDataComposite || showingDataSteps)) {
    const body = document.createElement('div'); body.className = 'composite-body';
    if (isDataComposite && showingDataSteps) {
      const heading = document.createElement('div');
      heading.className = 'data-execution-heading';
      heading.innerHTML = '<strong>实际执行步骤</strong><span>以下仍是原来的 Scratch 数据积木，顺序和插口不变。</span>';
      body.appendChild(heading);
    }
    for (let index = range.start; index < range.end; index += 1) {
      body.appendChild(createDropZone(parentId, index, context, range.groupId));
      // The canonical data view is intentionally lossless: do not collapse
      // the first two data operations into the generic adjacent-pair card.
      // Users who chose “原始 12 步” must see and edit every operation as an
      // independent Scratch block.
      const pair = isDataComposite && showingDataSteps
        ? null
        : index + 1 < range.end ? adjacentPairAt(steps, index) : null;
      if (pair) { renderAdjacentPair(pair, index, steps, body, parentId, context); index += 1; }
      else renderStepNode(steps[index], index, steps, body, parentId, context, {suppressEpochAdvanced: true});
    }
    body.appendChild(createDropZone(parentId, range.end, context, range.groupId)); shell.appendChild(body);
  }
  return shell;
}

function drawSteps(steps = state.recipe.steps, parent = $('steps'), parentId = '__root__', context = 'top') {
  parent.innerHTML = '';
  const ranges = detectCompositeRanges(steps);
  let index = 0;
  while (index < (steps || []).length) {
    const range = ranges.find((item) => item.start === index);
    const key = range ? compositeKey(parentId, {...range, steps}) : null;
    parent.appendChild(createDropZone(parentId, index, context));
    if (range && !state.compositeUngrouped.has(key)) {
      parent.appendChild(createCompositeShell({...range, steps}, steps, parentId, context, key));
      index = range.end;
      continue;
    }
    const pair = adjacentPairAt(steps, index);
    if (pair) { renderAdjacentPair(pair, index, steps, parent, parentId, context); index += 2; }
    else { renderStepNode(steps[index], index, steps, parent, parentId, context); index += 1; }
  }
  parent.appendChild(createDropZone(parentId, (steps || []).length, context));
  markDropZones();
}

function renderInspector() {
  const target = $('inspector');
  const explanationTarget = $('module-explanation');
  target.innerHTML = '';
  explanationTarget.innerHTML = '';
  if (state.dataOverviewSelection
    && state.selected?._uiId
    && state.dataOverviewSelection.stepIds?.includes(state.selected._uiId)
    && renderDataOverviewInspector(state.dataOverviewSelection, target, explanationTarget)) return;
  state.dataOverviewSelection = null;
  const pairLocation = state.selected ? findParentArrayAndIndex(state.selected._uiId) : null;
  const pair = pairLocation ? adjacentPairAt(pairLocation.array, pairLocation.index) : null;
  if (pair && state.adjacentSelection?.members[0] === state.selected) {
    renderPairInspector(pair, target, explanationTarget);
    return;
  }
  const paletteInfo = !state.selected ? state.paletteSelection : null;
  const previewStep = paletteInfo ? paletteDraftFor(paletteInfo) : null;
  const step = state.selected || previewStep;
  if (!step) {
    explanationTarget.textContent = '点击一个积木查看说明';
    const empty = document.createElement('div'); empty.className = 'inspector-empty-guide';
    empty.appendChild(Object.assign(document.createElement('strong'), {textContent: '还没有选中积木'}));
    empty.appendChild(Object.assign(document.createElement('p'), {textContent: '运行前至少需要：数据集 → 模型 → Loader → Training Loop。先选中一个积木，右侧会显示它的参数和输入输出。'}));
    const button = document.createElement('button'); button.type = 'button'; button.className = 'primary-action'; button.textContent = '添加 / 查看数据集积木'; button.onclick = focusDatasetSource;
    empty.appendChild(button); target.appendChild(empty); return;
  }
  const info = paletteInfo || blockInfo(step.block);
  if (!info) { explanationTarget.textContent = '暂无模块说明'; target.textContent = '无法加载该积木参数'; return; }
  if (isDataConceptInfo(info)) {
    const group = DATA_OVERVIEW_GROUPS.find((item) => item.id === info.dataGroupId);
    const existing = group
      ? group.blocks.map((block) => flatRecipeSteps().find((candidate) => candidate.block === block)).filter(Boolean)
      : [];
    if (group && existing.length === group.blocks.length) {
      renderDataOverviewInspector({groupId: group.id, stepIds: existing.map((candidate) => candidate._uiId)}, target, explanationTarget);
    } else {
      const identity = document.createElement('div');
      const name = document.createElement('strong'); name.textContent = info.name;
      const description = document.createElement('p'); description.textContent = `${info.description} 添加后会展开为对应的底层数据积木，并可逐项编辑参数。`;
      identity.append(name, description);
      explanationTarget.appendChild(identity);
      const section = document.createElement('section'); section.className = 'inspector-section';
      const heading = document.createElement('h3'); heading.textContent = '包含的执行步骤'; section.appendChild(heading);
      (group?.blocks || []).forEach((block) => {
        const canonical = blockInfo(block);
        if (canonical) section.appendChild(Object.assign(document.createElement('p'), {textContent: `• ${canonical.name}`}));
      });
      target.appendChild(section);
    }
    return;
  }
  if (state.errorMessage) {
    const error = document.createElement('div'); error.className = 'inspector-error';
    const reason = document.createElement('strong'); reason.textContent = '⚠ ' + state.errorMessage; error.appendChild(reason);
    const details = state.errorPayload || {};
    if (details.block_id) error.appendChild(Object.assign(document.createElement('div'), {textContent: `积木：${details.block_id}`}));
    if (details.params && typeof details.params === 'object') error.appendChild(Object.assign(document.createElement('div'), {textContent: `参数：${JSON.stringify(details.params)}`}));
    if (state.selected) {
      const target = selectedTarget(state.selected);
      const available = [...availableKeysBefore(target)].sort();
      error.appendChild(Object.assign(document.createElement('div'), {textContent: `当前可用 slots：${available.join(', ') || '无'}`}));
    }
    if (state.errorGuide) {
      const guideButton = document.createElement('button'); guideButton.type = 'button'; guideButton.textContent = '查看处理引导'; guideButton.onclick = () => setInspectorTab('run'); error.appendChild(guideButton);
    }
    target.appendChild(error);
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
  formula.className = 'formula';
  const formulaText = document.createElement('div');
  renderFormulaOrIntro(formulaText, info);
  formula.appendChild(formulaText);
  if (info.formula_ref) { const ref = document.createElement('small'); ref.textContent = `定义来源：${info.formula_ref}`; formula.appendChild(ref); }
  if (info.paper) { const paper = document.createElement('div'); paper.textContent = `论文：${info.paper}`; formula.appendChild(paper); }
  addSection(formulaDisplayText(info) ? '2. 公式' : '2. 作用说明', formula);

  const io = document.createElement('div');
  const inputs = document.createElement('div'); inputs.textContent = `输入：${(info.requires || []).map((name) => slotValue(info, step, name)).join(', ') || '无'}`; io.appendChild(inputs);
  const outputs = document.createElement('div'); outputs.textContent = `输出：${(info.provides || []).map((name) => slotValue(info, step, name)).join(', ') || '无'}`; io.appendChild(outputs);
  addSection('3. 输入 / 输出', io);

  const params = document.createElement('div');
  if (paletteInfo) {
    const help = document.createElement('p');
    help.textContent = '待添加积木：先选择输入连接和输出名称，再添加或拖动。这里只配置新积木，不会修改已有步骤。';
    params.appendChild(help);
    info.params && Object.entries(info.params).forEach(([name, schema]) => {
      if (isDatasetSourceInfo(info) && name === 'path' && step.params?.source_mode !== 'custom_path') return;
      const wrap = document.createElement('div'); wrap.className = 'param';
      const label = document.createElement('label'); label.textContent = isDatasetSourceInfo(info) && name === 'dataset' ? 'dataset（必选）' : name; wrap.appendChild(label);
      wrap.appendChild(renderParamControl(name, schema, step));
      params.appendChild(wrap);
    });
  } else {
    info.params && Object.entries(info.params).forEach(([name, schema]) => {
      if (isDatasetSourceInfo(info) && name === 'path' && step.params?.source_mode !== 'custom_path') return;
      const wrap = document.createElement('div'); wrap.className = 'param';
      const label = document.createElement('label'); label.textContent = isDatasetSourceInfo(info) && name === 'dataset' ? 'dataset（必选）' : name; wrap.appendChild(label);
      wrap.appendChild(renderParamControl(name, schema, step));
      params.appendChild(wrap);
    });
  }
  addSection('4. 参数', params);
  if (!paletteInfo && isDatasetSourceInfo(info)) target.appendChild(renderDatasetFacts(step));
  if (!paletteInfo && (info.provides || []).some((name) => name === 'noise_state' || name === 'noisy_train_split')) target.appendChild(renderNoiseFacts(step));
}

function draw() {
  if ($('recipe-name')) $('recipe-name').value = state.recipe.name;
  renderRuntimeLimits();
  const empty = $('empty-onboarding');
  if (empty) empty.hidden = flatRecipeSteps().length > 0;
  renderPalette();
  drawSteps();
  // Selection is keyed by the stable UI id as well as object identity.  This
  // keeps the highlight reliable when a nested recipe array is re-rendered
  // after an explicit palette insertion.
  if (state.selected?._uiId) {
    document.querySelector(`[data-ui-id="${state.selected._uiId}"]`)?.classList.add('selected');
  }
  renderInspector();
  renderUndoDelete();
  renderGuidance();
  updateRunState();
  if (!state.lastRun) showMessage(targetLabel(state.activeInsertionTarget));
  setInspectorTab(state.inspectorTab);
}

function formulaCandidates() {
  return state.blocks.filter((info) => info.beginner_visible
    && !info.paper
    && info.kind === 'action'
    && info.formula_kind
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
  preview.replaceChildren();
  const description = document.createElement('p'); description.className = 'formula-preview-description'; description.textContent = info.description || '';
  preview.appendChild(description);
  const formulaLabel = document.createElement('div'); formulaLabel.className = 'formula-preview-label'; formulaLabel.textContent = formulaDisplayText(info) ? '数学公式' : '运算说明'; preview.appendChild(formulaLabel);
  const formulaVisual = document.createElement('div'); renderFormulaOrIntro(formulaVisual, info); preview.appendChild(formulaVisual);
  const inputSummary = document.createElement('div'); inputSummary.className = 'formula-preview-inputs'; inputSummary.textContent = `输入：${(info.requires || []).join(', ') || '无'}`; preview.appendChild(inputSummary);
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
  const reason = availabilityReason(info, target, null, params);
  if (reason) return reason;
  const available = availableKeysBefore(target);
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
  return state.blocks.filter((info) => info.formula_kind && !info.paper && info.kind === 'action')
    .sort((a, b) => `${a.formula_group || a.category}:${a.name}`.localeCompare(`${b.formula_group || b.category}:${b.name}`));
}

// Formula palette groups come from Block metadata.  Keep structural math
// operations in their own visible groups instead of collapsing them into the
// generic "基础" bucket.
const FORMULA_EDITOR_GROUPS = ['基础', '函数', '归约', '形状', '条件', '索引', '线性代数', '概率', '概率 / Loss', '更多'];
const FORMULA_EDITOR_KIND_GROUPS = ['基础运算', '特殊运算', '公式模板'];
function formulaEditorGroup(info) {
  const declaredGroup = String(info?.formula_group || '').trim();
  if (FORMULA_EDITOR_GROUPS.includes(declaredGroup)) return declaredGroup;
  const kindGroups = {primitive: '基础', composite: '概率 / Loss', special: '更多'};
  return kindGroups[String(info?.formula_kind || '')] || '更多';
}

function formulaEditorKindGroup(info) {
  const kind = String(info?.formula_kind || '');
  if (kind === 'primitive') return '基础运算';
  if (kind === 'special') return '特殊运算';
  if (kind === 'composite') return '公式模板';
  return '特殊运算';
}

function formulaDefinitionForBlock(info) {
  const reference = String(info?.formula_ref || '');
  return state.formulas.find((item) => item && item.id === reference) || null;
}

function expressionHole() { return {kind: 'hole'}; }

function parseEditorInputs() {
  return String($('formula-inputs').value || '').split(/[\n,]/).map((item) => item.trim()).filter(Boolean);
}

function cloneExpressionValue(value) {
  if (value === undefined) return undefined;
  if (typeof structuredClone === 'function') return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}

function cloneSchema(value) {
  return value && typeof value === 'object' ? cloneExpressionValue(value) : {};
}

function parameterValueFromRaw(raw, type) {
  let value = String(raw ?? '').trim();
  if (type === 'float') value = Number.parseFloat(value);
  else if (type === 'int') value = Number.parseInt(value, 10);
  else if (type === 'bool') value = value === 'true';
  else if (type === 'value') {
    try { value = JSON.parse(value); } catch (error) { throw new Error('value 参数必须是合法 JSON'); }
  }
  if ((type === 'float' || type === 'int') && !Number.isFinite(value)) throw new Error('参数默认值无效');
  return value;
}

function parameterDefaultText(schema = {}) {
  if (!Object.prototype.hasOwnProperty.call(schema, 'default')) return '';
  const value = schema.default;
  if (schema.type === 'value') return JSON.stringify(value);
  if (schema.type === 'bool') return value ? 'true' : 'false';
  if (value === null) return 'null';
  return String(value);
}

function parameterSchemaLine(name, schema = {}) {
  const type = schema.type || 'float';
  return Object.prototype.hasOwnProperty.call(schema, 'default') ? `${name}:${type}=${parameterDefaultText(schema)}` : `${name}:${type}`;
}

function formulaInputRows() {
  return [...new Set(parseEditorInputs())];
}

function updateFormulaInputText() {
  const field = $('formula-inputs');
  const rows = $('formula-input-fields')?.querySelectorAll('[data-formula-input-row]') || [];
  if (!field || !rows.length) return;
  const names = [];
  rows.forEach((row) => {
    const name = row.querySelector('[data-formula-input-name]')?.value.trim() || '';
    const previous = row.dataset.formulaInputOriginal || '';
    if (previous && name && previous !== name) {
      renameExpressionSymbol(state.formulaEditor.expression, 'input', previous, name);
      if (state.formulaEditor.inputSchemas[previous]) {
        state.formulaEditor.inputSchemas[name] = state.formulaEditor.inputSchemas[previous];
        delete state.formulaEditor.inputSchemas[previous];
      }
    }
    row.dataset.formulaInputOriginal = name;
    if (name && !names.includes(name)) {
      names.push(name);
      if (!state.formulaEditor.inputSchemas[name]) state.formulaEditor.inputSchemas[name] = {description: '', type: 'tensor'};
    }
  });
  field.value = names.join('\n');
  Object.keys(state.formulaEditor.inputSchemas).forEach((name) => { if (!names.includes(name)) delete state.formulaEditor.inputSchemas[name]; });
  renderFormulaEditorBindingFields();
  renderFormulaCanvas();
}

function renderFormulaInputFields() {
  const container = $('formula-input-fields');
  if (!container) return;
  container.replaceChildren();
  const rows = formulaInputRows();
  if (!rows.length) {
    const empty = document.createElement('div'); empty.className = 'formula-input-empty'; empty.textContent = '暂无输入；点击“＋ 添加输入”创建一个。'; container.appendChild(empty); return;
  }
  rows.forEach((item) => {
    const row = document.createElement('div'); row.className = 'formula-input-row'; row.dataset.formulaInputRow = '1';
    row.dataset.formulaInputOriginal = item;
    const input = document.createElement('input'); input.value = item; input.dataset.formulaInputName = '1'; input.placeholder = 'loss_a'; input.autocomplete = 'off';
    const insert = document.createElement('button'); insert.type = 'button'; insert.className = 'secondary-button formula-symbol-insert'; insert.textContent = '插入'; insert.title = `将 ${item} 插入选中的表达式位置`; insert.onclick = () => insertExpressionSymbol('input', item);
    const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'secondary-button'; remove.textContent = '×'; remove.title = `删除输入 ${item}`;
    remove.onclick = () => { state.formulaEditor.expression = removeExpressionSymbol(state.formulaEditor.expression, 'input', item); state.formulaEditor.expressionSelection = null; row.remove(); updateFormulaInputText(); if (!container.querySelector('[data-formula-input-row]')) renderFormulaInputFields(); renderFormulaCanvas(); };
    input.oninput = updateFormulaInputText;
    row.append(input, insert, remove); container.appendChild(row);
  });
}

function addFormulaEditorInput() {
  const field = $('formula-inputs');
  if (!field) return;
  const existing = formulaInputRows();
  let index = existing.length + 1;
  let name = `input_${index}`;
  while (existing.includes(name)) { index += 1; name = `input_${index}`; }
  field.value = [...existing, name].join('\n');
  state.formulaEditor.inputSchemas[name] = {description: '', type: 'tensor'};
  renderFormulaInputFields();
  const inputs = $('formula-input-fields')?.querySelectorAll('[data-formula-input-name]') || [];
  const input = inputs[inputs.length - 1];
  if (input) { input.focus(); input.select(); }
}

function parseEditorParameters() {
  const result = {};
  String($('formula-parameters').value || '').split(/\n/).map((item) => item.trim()).filter(Boolean).forEach((line) => {
    const match = line.match(/^([A-Za-z][A-Za-z0-9_]*)\s*(?::\s*([A-Za-z]+))?(?:\s*=\s*(.*))?$/);
    if (!match) throw new Error(`参数格式错误：${line}，应为 name:type=default`);
    const [, name, type = 'float', raw] = match;
    const schema = {...cloneSchema(state.formulaEditor.parameterSchemas[name]), type};
    if (raw !== undefined && raw.trim() !== '') {
      let value;
      try { value = parameterValueFromRaw(raw, type); } catch (error) { throw new Error(`参数 ${name}：${error.message}`); }
      schema.default = value;
    } else delete schema.default;
    result[name] = schema;
  });
  return result;
}

function formulaParameterRows() {
  const rows = [];
  String($('formula-parameters')?.value || '').split(/\n/).map((item) => item.trim()).filter(Boolean).forEach((line) => {
    const match = line.match(/^([A-Za-z][A-Za-z0-9_]*)\s*(?::\s*([A-Za-z]+))?(?:\s*=\s*(.*))?$/);
    if (match) rows.push({name: match[1], type: match[2] || 'float', raw: (match[3] ?? '').trim()});
  });
  return rows;
}

function updateFormulaParameterText() {
  const field = $('formula-parameters');
  const rows = $('formula-parameter-fields')?.querySelectorAll('[data-formula-parameter-row]') || [];
  // An empty row list is a valid state after deleting the final parameter.
  // Still rewrite the source textarea and schema map so the deleted variable
  // cannot be rendered back on the next operation switch.
  if (!field) return;
  const previousNames = new Set(Object.keys(state.formulaEditor.parameterSchemas || {}));
  const lines = [];
  rows.forEach((row) => {
    const name = row.querySelector('[data-formula-parameter-name]')?.value.trim() || '';
    const previous = row.dataset.formulaParameterOriginal || '';
    if (previous && name && previous !== name) {
      renameExpressionSymbol(state.formulaEditor.expression, 'parameter', previous, name);
      renameFormulaParameterState(previous, name);
      if (state.formulaEditor.parameterSchemas[previous]) {
        state.formulaEditor.parameterSchemas[name] = state.formulaEditor.parameterSchemas[previous];
        delete state.formulaEditor.parameterSchemas[previous];
      }
    }
    row.dataset.formulaParameterOriginal = name;
    const type = row.querySelector('[data-formula-parameter-type]')?.value || 'float';
    const value = row.querySelector('[data-formula-parameter-default]')?.value ?? '';
    if (name) {
      transitionFormulaParameter(name, FORMULA_PARAMETER_STATUS.ACTIVE, 'user');
      const schema = {...cloneSchema(state.formulaEditor.parameterSchemas[name]), type};
      if (String(value).trim() === '') delete schema.default;
      else { try { schema.default = parameterValueFromRaw(value, type); } catch (error) { schema.default = value; } }
      state.formulaEditor.parameterSchemas[name] = schema;
      lines.push(`${name}:${type}=${value}`);
    }
  });
  field.value = lines.join('\n');
  const names = rows ? [...rows].map((row) => row.querySelector('[data-formula-parameter-name]')?.value.trim() || '').filter(Boolean) : [];
  previousNames.forEach((name) => { if (!names.includes(name)) transitionFormulaParameter(name, FORMULA_PARAMETER_STATUS.REMOVED, 'user'); });
  Object.keys(state.formulaEditor.parameterSchemas).forEach((name) => { if (!names.includes(name)) delete state.formulaEditor.parameterSchemas[name]; });
  renderFormulaEditorBindingFields();
  renderFormulaEditorStepParameterFields();
  renderFormulaCanvas();
}

function renderFormulaParameterFields() {
  const container = $('formula-parameter-fields');
  if (!container) return;
  container.replaceChildren();
  const rows = formulaParameterRows();
  if (!rows.length) {
    const empty = document.createElement('div'); empty.className = 'formula-parameter-empty'; empty.textContent = '暂无参数；点击“＋ 添加参数”创建一个。'; container.appendChild(empty); return;
  }
  rows.forEach((item) => {
    const row = document.createElement('div'); row.className = 'formula-parameter-row'; row.dataset.formulaParameterRow = '1';
    row.dataset.formulaParameterOriginal = item.name;
    const nameLabel = document.createElement('label'); nameLabel.textContent = '名称';
    const name = document.createElement('input'); name.value = item.name; name.dataset.formulaParameterName = '1'; name.placeholder = 'q'; name.autocomplete = 'off'; nameLabel.appendChild(name);
    const valueLabel = document.createElement('label'); valueLabel.textContent = '默认值';
    const value = document.createElement('input'); value.value = item.raw; value.dataset.formulaParameterDefault = '1'; value.placeholder = '0.7'; value.autocomplete = 'off'; valueLabel.appendChild(value);
    const typeLabel = document.createElement('label'); typeLabel.textContent = '类型';
    const type = document.createElement('select'); type.dataset.formulaParameterType = '1';
    [['float', '小数'], ['int', '整数'], ['bool', '开关'], ['enum', '枚举'], ['str', '文本'], ['value', '结构化值']].forEach(([key, label]) => { const option = document.createElement('option'); option.value = key; option.textContent = label; type.appendChild(option); });
    type.value = ['float', 'int', 'bool', 'enum', 'str', 'value'].includes(item.type) ? item.type : 'float'; typeLabel.appendChild(type);
    const schema = state.formulaEditor.parameterSchemas[item.name] || {};
    if (schema.description || schema.minimum !== undefined || schema.maximum !== undefined || Array.isArray(schema.options)) {
      row.title = [schema.description, schema.minimum !== undefined ? `最小值 ${schema.minimum}` : '', schema.maximum !== undefined ? `最大值 ${schema.maximum}` : '', Array.isArray(schema.options) ? `选项：${schema.options.join(', ')}` : ''].filter(Boolean).join('；');
    }
    const insert = document.createElement('button'); insert.type = 'button'; insert.className = 'secondary-button formula-symbol-insert'; insert.textContent = '插入'; insert.title = `将 ${item.name} 插入选中的表达式位置`; insert.onclick = () => insertExpressionSymbol('parameter', item.name);
    const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'secondary-button'; remove.textContent = '删除'; remove.title = `删除参数 ${item.name}`;
    remove.onclick = () => {
      transitionFormulaParameter(item.name, FORMULA_PARAMETER_STATUS.REMOVED, 'user');
      delete state.formulaEditor.parameterSchemas[item.name];
      state.formulaEditor.expression = removeExpressionSymbol(state.formulaEditor.expression, 'parameter', item.name);
      state.formulaEditor.expressionSelection = null;
      row.remove();
      updateFormulaParameterText();
      if (!container.querySelector('[data-formula-parameter-row]')) renderFormulaParameterFields();
      renderFormulaCanvas();
    };
    [name, value, type].forEach((fieldInput) => { fieldInput.oninput = updateFormulaParameterText; fieldInput.onchange = updateFormulaParameterText; });
    row.append(nameLabel, valueLabel, typeLabel, insert, remove); container.appendChild(row);
  });
}

function addFormulaEditorParameter() {
  const field = $('formula-parameters');
  if (!field) return;
  const existing = formulaParameterRows().map((item) => item.name);
  let index = existing.length + 1;
  let name = `parameter_${index}`;
  while (existing.includes(name)) { index += 1; name = `parameter_${index}`; }
  field.value = [...String(field.value || '').split(/\n/).map((item) => item.trim()).filter(Boolean), `${name}:float=0`].join('\n');
  state.formulaEditor.parameterSchemas[name] = {type: 'float', default: 0};
  transitionFormulaParameter(name, FORMULA_PARAMETER_STATUS.ACTIVE, 'user');
  renderFormulaParameterFields();
  const inputs = $('formula-parameter-fields')?.querySelectorAll('[data-formula-parameter-name]') || [];
  const input = inputs[inputs.length - 1];
  if (input) { input.focus(); input.select(); }
}

// Operation parameters are formula variables, not hidden values buried in a
// step call.  When a user inserts an operation, expose every scalar
// non-slot parameter in the visible parameter editor so it can be renamed or
// edited like any other formula variable.  Array-valued controls such as
// weighted-sum terms remain expression arrays instead of becoming opaque JSON
// parameters.
function formulaEditorParameterType(schema = {}) {
  const type = schema.type || 'float';
  return ['float', 'int', 'bool', 'enum', 'str'].includes(type) ? type : null;
}

function normalizeFormulaEditorSchema(schema = {}, valueOverride) {
  const normalized = cloneSchema(schema);
  const value = valueOverride !== undefined ? valueOverride : normalized.default;
  // The editor's numeric parser intentionally rejects non-finite numbers.
  // Keep special values such as -Infinity editable as text; Scratch runtime
  // operations that accept a float still coerce the saved value explicitly.
  if (normalized.type === 'float' && typeof value === 'number' && !Number.isFinite(value)) {
    normalized.type = 'str';
    normalized.default = String(value);
  }
  return normalized;
}

const FORMULA_PARAMETER_STATUS = Object.freeze({ACTIVE: 'active', REMOVED: 'removed'});

function formulaParameterStates() {
  if (!state.formulaEditor.parameterStates) state.formulaEditor.parameterStates = {};
  return state.formulaEditor.parameterStates;
}

function transitionFormulaParameter(name, status, source = 'editor') {
  if (!name) return;
  formulaParameterStates()[name] = {status, source, updatedAt: Date.now()};
}

function formulaParameterWasRemoved(name) {
  return formulaParameterStates()[name]?.status === FORMULA_PARAMETER_STATUS.REMOVED;
}

function renameFormulaParameterState(previous, next) {
  if (!previous || !next || previous === next) return;
  const states = formulaParameterStates();
  if (states[previous]) states[next] = states[previous];
  delete states[previous];
}

function ensureFormulaEditorParameters(info, parameterValues = {}, options = {}) {
  const field = $('formula-parameters');
  if (!field || !info) return;
  const force = Boolean(options.force);
  const existing = formulaParameterRows();
  const names = new Set(existing.map((item) => item.name));
  const lines = existing.map((item) => `${item.name}:${item.type}${item.raw ? `=${item.raw}` : ''}`);
  let changed = false;
  Object.entries(info.params || {}).forEach(([name, schema = {}]) => {
    const type = formulaEditorParameterType(schema);
    if (schema.type === 'slot' || isOutputSlot(name) || !type) return;
    if (names.has(name)) { transitionFormulaParameter(name, FORMULA_PARAMETER_STATUS.ACTIVE, 'existing'); return; }
    // An explicit user deletion wins over automatic discovery, including
    // insertion of another operation that happens to use the same parameter
    // name (for example clamp_min and safe_divide both use `minimum`).  The
    // user can deliberately re-enable it with “＋ 添加参数”, which performs
    // the REMOVED -> ACTIVE transition through addFormulaEditorParameter().
    if (formulaParameterWasRemoved(name)) return;
    transitionFormulaParameter(name, FORMULA_PARAMETER_STATUS.ACTIVE, force ? 'operation' : 'discovered');
    const hasOverride = Object.prototype.hasOwnProperty.call(parameterValues, name);
    const baseSchema = cloneSchema(schema);
    if (hasOverride) baseSchema.default = cloneExpressionValue(parameterValues[name]);
    const normalized = normalizeFormulaEditorSchema(baseSchema);
    state.formulaEditor.parameterSchemas[name] = normalized;
    lines.push(parameterSchemaLine(name, normalized));
    names.add(name);
    changed = true;
  });
  if (changed) {
    field.value = lines.join('\n');
    renderFormulaParameterFields();
  }
}

function formulaEditorParameterNode(name, schema, existing = null) {
  if (existing) return existing;
  const type = schema?.type || 'float';
  const parameterNames = new Set(formulaParameterRows().map((item) => item.name));
  if (!isOutputSlot(name) && formulaEditorParameterType(schema) && parameterNames.has(name)) {
    return expressionParameter(name);
  }
  return expressionConstant(schema?.default);
}

// Formula files created before the expression canvas may contain scalar
// operation parameters only inside a step.  Promote those values to the same
// visible parameter list used by newly inserted operations, preserving the
// step's actual value as the default.  This keeps every equation editable
// without exposing implementation-only output slots such as save_as.
function exposeExpressionParameters(root, seen = new WeakSet()) {
  if (!root || typeof root !== 'object' || seen.has(root)) return;
  seen.add(root);
  if (root.kind === 'operation') {
    const info = blockInfo(root.block);
    if (info) {
      const values = {};
      Object.entries(root.parameters || {}).forEach(([name, child]) => {
        if (child?.kind === 'constant') values[name] = child.value;
      });
      ensureFormulaEditorParameters(info, values);
      Object.entries(info.params || {}).forEach(([name, schema]) => {
        if (!formulaEditorParameterType(schema) || schema.type === 'slot' || isOutputSlot(name)) return;
        const existing = root.parameters?.[name];
        if (!existing) {
          const defaultValue = Object.prototype.hasOwnProperty.call(values, name) ? values[name] : schema.default;
          root.parameters[name] = formulaEditorParameterNode(name, {...schema, default: defaultValue});
        } else if (existing.kind === 'constant' && formulaParameterRows().some((item) => item.name === name)) {
          root.parameters[name] = expressionParameter(name);
        }
      });
    }
  }
  expressionNodeEntries(root).forEach(([, child]) => exposeExpressionParameters(child, seen));
}

function expressionInput(name) { return {kind: 'input', name: String(name)}; }
function expressionParameter(name) { return {kind: 'parameter', name: String(name)}; }
function expressionConstant(value) { return {kind: 'constant', value}; }

function formulaOutputEntries() {
  const outputs = state.formulaEditor.outputExpressions || {};
  if (!Object.keys(outputs).length) outputs.loss = state.formulaEditor.expression || null;
  return Object.entries(outputs);
}

// UI-only placeholders make an incomplete equation explicit while it is being
// edited.  They are deliberately rejected by the FormulaSpec serializer.
function expressionContainsHole(node, seen = new WeakSet()) {
  if (!node) return false;
  if (node.kind === 'hole') return true;
  if (typeof node !== 'object') return false;
  if (seen.has(node)) return false;
  seen.add(node);
  return expressionNodeEntries(node).some(([, child]) => expressionContainsHole(child, seen));
}

function firstExpressionHole(node, seen = new WeakSet()) {
  if (!node || typeof node !== 'object') return null;
  if (node.kind === 'hole') return node;
  if (seen.has(node)) return null;
  seen.add(node);
  for (const [, child] of expressionNodeEntries(node)) {
    const hole = firstExpressionHole(child, seen);
    if (hole) return hole;
  }
  return null;
}

function rememberActiveFormulaOutput() {
  const name = state.formulaEditor.activeOutput || 'loss';
  if (!state.formulaEditor.outputExpressions) state.formulaEditor.outputExpressions = {};
  state.formulaEditor.outputExpressions[name] = state.formulaEditor.expression || null;
}

function activateFormulaOutput(name) {
  rememberActiveFormulaOutput();
  state.formulaEditor.activeOutput = name;
  state.formulaEditor.expression = state.formulaEditor.outputExpressions[name] || null;
  state.formulaEditor.expressionSelection = null;
  const field = $('formula-output-name'); if (field) field.value = name;
  renderFormulaEditor();
}

function outputNameIsValid(name) {
  return /^[A-Za-z][A-Za-z0-9_]*$/.test(String(name || '').trim());
}

function renderFormulaOutputFields() {
  const container = $('formula-output-fields');
  if (!container) return;
  rememberActiveFormulaOutput();
  container.replaceChildren();
  formulaOutputEntries().forEach(([outputName, expression]) => {
    const row = document.createElement('div'); row.className = `formula-output-row${outputName === state.formulaEditor.activeOutput ? ' active' : ''}`; row.dataset.formulaOutputRow = '1'; row.dataset.formulaOutputOriginal = outputName;
    const name = document.createElement('input'); name.value = outputName; name.placeholder = 'loss'; name.dataset.formulaOutputName = '1'; name.title = '输出名称';
    name.onchange = () => {
      const next = name.value.trim(); const previous = row.dataset.formulaOutputOriginal;
      if (next === previous) return;
      if (!outputNameIsValid(next)) { name.value = previous; return; }
      if (Object.prototype.hasOwnProperty.call(state.formulaEditor.outputExpressions, next)) {
        name.value = previous;
        const validation = $('formula-editor-validation');
        if (validation) validation.textContent = `输出名称 ${next} 已存在`;
        return;
      }
      state.formulaEditor.outputExpressions[next] = state.formulaEditor.outputExpressions[previous];
      state.formulaEditor.outputSchemas[next] = state.formulaEditor.outputSchemas[previous] || {};
      delete state.formulaEditor.outputExpressions[previous]; delete state.formulaEditor.outputSchemas[previous];
      if (state.formulaEditor.activeOutput === previous) state.formulaEditor.activeOutput = next;
      row.dataset.formulaOutputOriginal = next; state.formulaEditor.expression = state.formulaEditor.outputExpressions[state.formulaEditor.activeOutput] || null;
      const hidden = $('formula-output-name'); if (hidden) hidden.value = state.formulaEditor.activeOutput;
      renderFormulaOutputFields(); renderFormulaCanvas();
    };
    const equals = document.createElement('span'); equals.textContent = '='; equals.className = 'formula-output-equals';
    const summary = document.createElement('button'); summary.type = 'button'; summary.className = 'formula-output-summary'; summary.textContent = expression ? expressionNodeText(expression) : '□'; summary.title = '编辑这个输出的公式'; summary.onclick = () => activateFormulaOutput(outputName);
    const edit = document.createElement('button'); edit.type = 'button'; edit.textContent = '编辑'; edit.title = '切换到这个输出'; edit.onclick = () => activateFormulaOutput(outputName);
    const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = '×'; remove.title = '删除输出'; remove.disabled = formulaOutputEntries().length <= 1;
    remove.onclick = () => { rememberActiveFormulaOutput(); delete state.formulaEditor.outputExpressions[outputName]; delete state.formulaEditor.outputSchemas[outputName]; const next = formulaOutputEntries()[0]?.[0] || 'loss'; state.formulaEditor.activeOutput = next; state.formulaEditor.expression = state.formulaEditor.outputExpressions[next] || null; renderFormulaEditor(); };
    row.append(name, equals, summary, edit, remove); container.appendChild(row);
  });
}

function addFormulaEditorOutput() {
  rememberActiveFormulaOutput();
  let index = formulaOutputEntries().length + 1; let name = `output_${index}`;
  while (Object.prototype.hasOwnProperty.call(state.formulaEditor.outputExpressions, name)) { index += 1; name = `output_${index}`; }
  state.formulaEditor.outputExpressions[name] = expressionHole(); state.formulaEditor.outputSchemas[name] = {};
  activateFormulaOutput(name);
}

function insertExpressionSymbol(kind, name) {
  const node = kind === 'parameter' ? expressionParameter(name) : expressionInput(name);
  const selected = state.formulaEditor.expressionSelection;
  if (state.formulaEditor.expression && selected) {
    state.formulaEditor.expression = replaceExpressionReference(state.formulaEditor.expression, selected, node);
  } else if (!state.formulaEditor.expression) {
    state.formulaEditor.expression = node;
  } else {
    state.formulaEditor.expressionSelection = state.formulaEditor.expression;
    const validation = $('formula-editor-validation');
    if (validation) validation.textContent = `已选中 ${name}；请先在画布中选择要替换的位置`;
    renderFormulaCanvas();
    return;
  }
  state.formulaEditor.expressionSelection = firstExpressionHole(state.formulaEditor.expression) || node;
  syncFormulaStepsFromExpression();
  renderFormulaEditor();
}

function renameExpressionSymbol(root, kind, previous, next) {
  if (!root || !previous || !next) return root;
  if ((root.kind === kind) && root.name === previous) root.name = next;
  expressionNodeEntries(root).forEach(([, child]) => renameExpressionSymbol(child, kind, previous, next));
  return root;
}

function removeExpressionSymbol(root, kind, name) {
  if (!root) return null;
  if (root.kind === kind && root.name === name) return null;
  if (root.kind === 'array') {
    root.items = root.items.map((child) => removeExpressionSymbol(child, kind, name)).filter(Boolean);
    return root;
  }
  if (root.kind === 'operation') {
    Object.entries(root.bindings || {}).forEach(([slot, child]) => {
      const next = removeExpressionSymbol(child, kind, name);
      if (next) root.bindings[slot] = next; else delete root.bindings[slot];
    });
    Object.entries(root.parameters || {}).forEach(([parameter, child]) => {
      const next = removeExpressionSymbol(child, kind, name);
      if (next) root.parameters[parameter] = next; else delete root.parameters[parameter];
    });
  }
  return root;
}

function expressionFromValue(value, expressions, inputNames, parameterNames) {
  if (Array.isArray(value)) return {kind: 'array', items: value.map((item) => expressionFromValue(item, expressions, inputNames, parameterNames))};
  if (value && typeof value === 'object') return expressionConstant(JSON.parse(JSON.stringify(value)));
  if (typeof value === 'string') {
    if (expressions.has(value)) return expressions.get(value);
    const source = value.split('.')[0];
    if (expressions.has(source)) return expressions.get(source);
    if (inputNames.has(value)) return expressionInput(value);
    if (parameterNames.has(value)) return expressionParameter(value);
  }
  return expressionConstant(value);
}

function formulaStepsToExpressionMap(steps, inputNames = new Set(), parameterNames = new Set()) {
  const expressions = new Map();
  (steps || []).forEach((step) => {
    const bindings = {};
    Object.entries(step.bindings || {}).forEach(([name, value]) => { bindings[name] = expressionFromValue(value, expressions, inputNames, parameterNames); });
    const parameters = {};
    Object.entries(step.parameters || {}).forEach(([name, value]) => { parameters[name] = expressionFromValue(value, expressions, inputNames, parameterNames); });
    expressions.set(step.id, {kind: 'operation', block: step.block, bindings, parameters, sourceStep: step.id});
  });
  return expressions;
}

function formulaStepsToExpression(steps, outputSource, inputNames = new Set(), parameterNames = new Set()) {
  const expressions = formulaStepsToExpressionMap(steps, inputNames, parameterNames);
  return expressions.get(outputSource) || expressions.get(steps?.at(-1)?.id) || null;
}

function formulaStepsToExpressions(steps, outputs, inputNames = new Set(), parameterNames = new Set()) {
  const expressions = formulaStepsToExpressionMap(steps, inputNames, parameterNames);
  return Object.fromEntries(Object.keys(outputs || {}).map((name) => [name, expressions.get(outputs[name]?.source?.split('.')[0]) || null]));
}

function expressionNodeEntries(node) {
  if (!node) return [];
  if (node.kind === 'array') return node.items.map((child, index) => [`[${index + 1}]`, child]);
  if (node.kind !== 'operation') return [];
  const outputSlots = new Set(Object.entries(blockInfo(node.block)?.params || {})
    .filter(([, schema]) => schema?.type === 'slot')
    .map(([name]) => name));
  return [
    ...Object.entries(node.bindings || {}),
    ...Object.entries(node.parameters || {}).filter(([name]) => name !== 'save_as' && !outputSlots.has(name)),
  ];
}

function expressionNodeText(node) {
  if (!node) return '（空）';
  if (node.kind === 'hole') return '□';
  if (node.kind === 'input' || node.kind === 'parameter') return node.name;
  if (node.kind === 'constant') {
    if (typeof node.value === 'string') return node.value;
    const rendered = JSON.stringify(node.value);
    return rendered === undefined ? String(node.value) : rendered;
  }
  if (node.kind === 'array') return `[${node.items.map(expressionNodeText).join(', ')}]`;
  const entries = expressionNodeEntries(node);
  const values = entries.map(([, child]) => expressionNodeText(child));
  const symbols = {add: '＋', subtract: '−', elementwise_multiply: '×', divide: '÷', matrix_multiply: '·'};
  if (symbols[node.block] && values.length >= 2) return `${values[0]} ${symbols[node.block]} ${values[1]}`;
  if (node.block === 'maximum' && values.length >= 2) return `max(${values[0]}, ${values[1]})`;
  if (node.block === 'minimum' && values.length >= 2) return `min(${values[0]}, ${values[1]})`;
  if (['exp', 'log', 'sqrt', 'abs', 'sign'].includes(node.block) && values.length) return `${node.block}(${values[0]})`;
  if (node.block === 'log_softmax' && values.length) {
    const input = expressionNodeText(node.bindings?.logits || values[0]);
    return `log_softmax(${input})`;
  }
  if (node.block === 'softmax' && values.length) {
    const input = expressionNodeText(node.bindings?.logits || values[0]);
    const temperature = node.parameters?.temperature ? expressionNodeText(node.parameters.temperature) : null;
    return `softmax(${input}${temperature ? ` ÷ ${temperature}` : ''})`;
  }
  if (node.block === 'gather_by_label' && values.length >= 2) return `${values[0]}[${values[1]}]`;
  if (node.block === 'clamp_min' && values.length) return `clamp(${values[0]}, ${values[1] || 'ε'})`;
  if (node.block === 'negative_log' && values.length) {
    const input = expressionNodeText(node.bindings?.input || values[0]);
    const minimum = node.parameters?.minimum ? expressionNodeText(node.parameters.minimum) : 'ε';
    return `−log(max(${input}, ${minimum}))`;
  }
  if (node.block === 'sum_last_dimension' && values.length) return `Σ(${values[0]})`;
  if (node.block === 'one_hot_like' && values.length) return `one_hot(${values[0]})`;
  if (node.block === 'ones_like') return '1';
  if (node.block === 'affine_transform' && values.length) {
    const scale = node.parameters?.scale ? expressionNodeText(node.parameters.scale) : '1';
    const bias = node.parameters?.bias ? expressionNodeText(node.parameters.bias) : '0';
    return `(${scale} × ${values[0]}) ＋ ${bias}`;
  }
  if (node.block === 'weighted_sum') {
    const terms = node.parameters?.terms?.items || [];
    const weights = node.parameters?.weights?.items || [];
    if (terms.length) return terms.map((term, index) => `${weights[index] ? expressionNodeText(weights[index]) : '1'} × ${expressionNodeText(term)}`).join(' ＋ ');
  }
  if (node.block === 'mean_loss' && values.length) return `mean(${values[0]})`;
  if (node.block === 'elementwise_power' && values.length >= 2) return `${values[0]}^${values[1]}`;
  if (node.block === 'negate' && values.length) return `−(${values[0]})`;
  if (node.block === 'weighted_blend' && values.length >= 3) {
    return `${values[2]} × ${values[0]} ＋ (1 − ${values[2]}) × ${values[1]}`;
  }
  if (node.block === 'sharpen_distribution' && values.length) {
    const temperature = node.parameters?.temperature ? expressionNodeText(node.parameters.temperature) : 'T';
    return `normalize(${values[0]}^(1/${temperature}))`;
  }
  if (node.block === 'row_normalize' && values.length) return `normalize_rows(${values[0]})`;
  if (node.block === 'apply_transition' && values.length >= 2) return `${values[0]} × ${values[1]}`;
  if (node.block === 'detach' && values.length) return `stop_gradient(${values[0]})`;
  if (node.block === 'one_hot' && values.length) return `one_hot(${values[0]})`;
  if (node.block === 'zeros_like' && values.length) return `zeros_like(${values[0]})`;
  if (node.block === 'uniform_prior' && values.length) return `uniform_prior(${values[0]})`;
  if (node.block === 'safe_divide' && values.length >= 2) {
    const numerator = expressionNodeText(node.bindings?.numerator || values[0]);
    const denominator = expressionNodeText(node.bindings?.denominator || values[1]);
    const minimum = node.parameters?.minimum ? expressionNodeText(node.parameters.minimum) : 'ε';
    return `${numerator} ÷ max(${denominator}, ${minimum})`;
  }
  if (node.block === 'quantile' && values.length) {
    const q = node.parameters?.q ? expressionNodeText(node.parameters.q) : 'q';
    return `quantile(${values[0]}, ${q})`;
  }
  if (node.block === 'threshold_mask' && values.length) {
    const threshold = node.parameters?.threshold ? expressionNodeText(node.parameters.threshold) : 'τ';
    const comparison = node.parameters?.comparison ? expressionNodeText(node.parameters.comparison) : '≥';
    return `1[${values[0]} ${comparison} ${threshold}]`;
  }
  if (node.block === 'top_k_mask' && values.length) {
    const k = node.parameters?.k ? expressionNodeText(node.parameters.k) : 'k';
    return `TopKMask(${values[0]}, ${k})`;
  }
  if (node.block === 'mask_to_indices' && values.length) return `indices(${values[0]})`;
  if (node.block === 'invert_mask' && values.length) return `1 − ${values[0]}`;
  const info = blockInfo(node.block);
  return `${info?.name || node.block}(${values.join(', ')})`;
}

function expressionNodeValue(node) {
  if (!node) return null;
  if (node.kind === 'hole') return null;
  if (node.kind === 'input' || node.kind === 'parameter') return node.name;
  if (node.kind === 'constant') return node.value;
  if (node.kind === 'array') return node.items.map(expressionNodeValue);
  return null;
}

function cloneExpressionNode(node, seen = new WeakMap()) {
  if (!node || typeof node !== 'object') return node;
  if (seen.has(node)) return seen.get(node);
  const copy = {kind: node.kind}; seen.set(node, copy);
  if (node.kind === 'input' || node.kind === 'parameter') copy.name = node.name;
  else if (node.kind === 'constant') copy.value = cloneExpressionValue(node.value);
  else if (node.kind === 'array') copy.items = (node.items || []).map((item) => cloneExpressionNode(item, seen));
  else if (node.kind === 'operation') {
    copy.block = node.block;
    copy.bindings = Object.fromEntries(Object.entries(node.bindings || {}).map(([name, child]) => [name, cloneExpressionNode(child, seen)]));
    copy.parameters = Object.fromEntries(Object.entries(node.parameters || {}).map(([name, child]) => [name, cloneExpressionNode(child, seen)]));
  }
  return copy;
}

function expressionNodeChildSet(parent, key, child) {
  if (!parent) return;
  if (parent.kind === 'array') parent.items[Number(String(key).slice(1)) - 1] = child;
  else if (parent.kind === 'operation') {
    if (Object.prototype.hasOwnProperty.call(parent.bindings || {}, key)) parent.bindings[key] = child;
    else parent.parameters[key] = child;
  }
}

function duplicateExpressionNode(root, target) {
  if (!root || !target) return root;
  if (root === target) return cloneExpressionNode(root);
  const walk = (node) => {
    if (!node) return false;
    if (node.kind === 'array') {
      const index = node.items.indexOf(target);
      if (index >= 0) { node.items.splice(index + 1, 0, cloneExpressionNode(target)); return true; }
      return node.items.some(walk);
    }
    if (node.kind === 'operation') {
      for (const collection of [node.bindings || {}, node.parameters || {}]) {
        for (const [name, child] of Object.entries(collection)) {
          if (child === target) { collection[name] = cloneExpressionNode(child); return true; }
          if (walk(child)) return true;
        }
      }
    }
    return false;
  };
  walk(root);
  return root;
}

function insertExpressionSibling(root, target, position = 'after', node = expressionConstant(0)) {
  if (!root || !target) return false;
  if (root.kind === 'array') {
    const index = root.items.indexOf(target);
    if (index >= 0) { root.items.splice(index + (position === 'before' ? 0 : 1), 0, node); return true; }
    return root.items.some((child) => insertExpressionSibling(child, target, position, node));
  }
  if (root.kind === 'operation') return [...Object.values(root.bindings || {}), ...Object.values(root.parameters || {})].some((child) => insertExpressionSibling(child, target, position, node));
  return false;
}

function moveExpressionArrayItem(root, target, direction) {
  if (!root || !target) return false;
  if (root.kind === 'array') {
    const index = root.items.indexOf(target);
    if (index >= 0) {
      const next = index + Number(direction);
      if (next >= 0 && next < root.items.length) [root.items[index], root.items[next]] = [root.items[next], root.items[index]];
      return true;
    }
    return root.items.some((child) => moveExpressionArrayItem(child, target, direction));
  }
  if (root.kind === 'operation') return [...Object.values(root.bindings || {}), ...Object.values(root.parameters || {})].some((child) => moveExpressionArrayItem(child, target, direction));
  return false;
}

function expressionArrayItem(node) {
  const inputs = formulaInputRows();
  return inputs.length ? expressionInput(inputs[0]) : expressionConstant(0);
}

function appendExpressionNodeControl(parent, label, title, handler) {
  const button = document.createElement('button'); button.type = 'button'; button.className = 'formula-expression-node-control'; button.textContent = label; button.title = title;
  button.onclick = (event) => { event.stopPropagation(); handler(); }; return button;
}

function renderExpressionNode(node, parent = null, key = null) {
  const wrapper = document.createElement('div');
  wrapper.className = `formula-expression-node formula-expression-${node?.kind || 'empty'}${state.formulaEditor.expressionSelection === node ? ' selected' : ''}`;
  wrapper.dataset.expressionKind = node?.kind || 'empty';
  if (node?.block) wrapper.dataset.expressionBlock = node.block;
  wrapper.setAttribute('role', 'button'); wrapper.tabIndex = 0;
  const label = document.createElement('span'); label.className = 'formula-expression-label'; label.textContent = expressionNodeText(node); wrapper.appendChild(label);
  wrapper.onclick = (event) => { event.stopPropagation(); selectExpressionNode(node); };
  wrapper.onkeydown = (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectExpressionNode(node); } };
  const controls = document.createElement('span'); controls.className = 'formula-expression-node-controls';
  controls.appendChild(appendExpressionNodeControl(null, '复制', '复制此节点', () => { state.formulaEditor.expression = duplicateExpressionNode(state.formulaEditor.expression, node); state.formulaEditor.expressionSelection = node; syncFormulaStepsFromExpression(); renderFormulaEditor(); }));
  controls.appendChild(appendExpressionNodeControl(null, '替换', '在下方操作栏替换此节点', () => { state.formulaEditor.expressionSelection = node; renderFormulaCanvas(); }));
  if (parent?.kind === 'array') {
    controls.appendChild(appendExpressionNodeControl(null, '前插', '在数组中插入前项', () => { insertExpressionSibling(state.formulaEditor.expression, node, 'before', expressionArrayItem(node)); syncFormulaStepsFromExpression(); renderFormulaEditor(); }));
    controls.appendChild(appendExpressionNodeControl(null, '后插', '在数组中插入后项', () => { insertExpressionSibling(state.formulaEditor.expression, node, 'after', expressionArrayItem(node)); syncFormulaStepsFromExpression(); renderFormulaEditor(); }));
    controls.appendChild(appendExpressionNodeControl(null, '↑', '数组项上移', () => { moveExpressionArrayItem(state.formulaEditor.expression, node, -1); syncFormulaStepsFromExpression(); renderFormulaEditor(); }));
    controls.appendChild(appendExpressionNodeControl(null, '↓', '数组项下移', () => { moveExpressionArrayItem(state.formulaEditor.expression, node, 1); syncFormulaStepsFromExpression(); renderFormulaEditor(); }));
  }
  if (node?.kind === 'array') controls.appendChild(appendExpressionNodeControl(null, '＋元素', '向数组末尾添加元素', () => { node.items.push(expressionArrayItem(node)); syncFormulaStepsFromExpression(); renderFormulaEditor(); }));
  controls.appendChild(appendExpressionNodeControl(null, '删除', '删除此节点', () => { state.formulaEditor.expression = pruneExpressionNode(state.formulaEditor.expression, node); state.formulaEditor.expressionSelection = null; syncFormulaStepsFromExpression(); renderFormulaEditor(); }));
  wrapper.appendChild(controls);
  const entries = expressionNodeEntries(node);
  if (entries.length) {
    const children = document.createElement('div'); children.className = 'formula-expression-children';
    entries.forEach(([name, child]) => {
      const item = document.createElement('div'); item.className = 'formula-expression-child';
      if (node.kind === 'array') {
        item.draggable = true; item.dataset.expressionArrayItem = '1';
        item.ondragstart = (event) => { state.formulaEditor.expressionDrag = {parent: node, child}; item.classList.add('dragging'); if (event.dataTransfer) { event.dataTransfer.effectAllowed = 'move'; event.dataTransfer.setData('text/plain', name); } };
        item.ondragover = (event) => { if (state.formulaEditor.expressionDrag?.parent === node) { event.preventDefault(); item.classList.add('drag-over'); if (event.dataTransfer) event.dataTransfer.dropEffect = 'move'; } };
        item.ondragleave = () => item.classList.remove('drag-over');
        item.ondrop = (event) => {
          event.preventDefault(); item.classList.remove('drag-over'); const drag = state.formulaEditor.expressionDrag;
          if (!drag || drag.parent !== node || drag.child === child) return;
          const from = node.items.indexOf(drag.child); const to = node.items.indexOf(child);
          if (from >= 0 && to >= 0) { const [moved] = node.items.splice(from, 1); node.items.splice(from < to ? to - 1 : to, 0, moved); }
          state.formulaEditor.expressionDrag = null; syncFormulaStepsFromExpression(); renderFormulaEditor();
        };
        item.ondragend = () => { state.formulaEditor.expressionDrag = null; item.classList.remove('dragging', 'drag-over'); };
      }
      const entry = document.createElement('small'); entry.textContent = name; item.append(entry, renderExpressionNode(child, node, name)); children.appendChild(item);
    });
    wrapper.appendChild(children);
  }
  return wrapper;
}

function appendMathExpressionChild(parent, node) {
  const child = renderMathExpressionNode(node); parent.appendChild(child); return child;
}

function renderMathExpressionNode(node) {
  const wrapper = document.createElement('span');
  wrapper.className = `formula-expression-math-node formula-expression-math-${node?.kind || 'empty'}${state.formulaEditor.expressionSelection === node ? ' selected' : ''}`;
  wrapper.tabIndex = 0; wrapper.setAttribute('role', 'button');
  wrapper.onclick = (event) => { event.stopPropagation(); selectExpressionNode(node); };
  wrapper.onkeydown = (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectExpressionNode(node); } };
  if (!node) { wrapper.textContent = '（空）'; return wrapper; }
  if (node.kind === 'hole') { wrapper.textContent = '□'; return wrapper; }
  if (node.kind === 'input' || node.kind === 'parameter') { wrapper.textContent = node.name; return wrapper; }
  if (node.kind === 'constant') { wrapper.textContent = expressionNodeText(node); return wrapper; }
  if (node.kind === 'array') {
    wrapper.append('['); node.items.forEach((item, index) => { if (index) wrapper.append(', '); appendMathExpressionChild(wrapper, item); }); wrapper.append(']'); return wrapper;
  }
  const entries = expressionNodeEntries(node); const values = entries.map(([, child]) => child);
  const symbols = {add: ' ＋ ', subtract: ' − ', elementwise_multiply: ' × ', divide: ' ÷ ', matrix_multiply: ' · '};
  if (symbols[node.block] && values.length >= 2) { appendMathExpressionChild(wrapper, values[0]); wrapper.append(symbols[node.block]); appendMathExpressionChild(wrapper, values[1]); return wrapper; }
  if (node.block === 'maximum' && values.length >= 2) { wrapper.append('max('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(', '); appendMathExpressionChild(wrapper, values[1]); wrapper.append(')'); return wrapper; }
  if (node.block === 'minimum' && values.length >= 2) { wrapper.append('min('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(', '); appendMathExpressionChild(wrapper, values[1]); wrapper.append(')'); return wrapper; }
  if (['exp', 'log', 'sqrt', 'abs', 'sign'].includes(node.block) && values.length) {
    wrapper.append(`${node.block}(`); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper;
  }
  if (node.block === 'log_softmax' && values.length) {
    wrapper.append('log_softmax('); appendMathExpressionChild(wrapper, node.bindings?.logits || values[0]); wrapper.append(')'); return wrapper;
  }
  if (node.block === 'negative_log' && values.length) {
    const input = node.bindings?.input || values[0]; const minimum = node.parameters?.minimum || expressionConstant('ε');
    wrapper.append('−log(max('); appendMathExpressionChild(wrapper, input); wrapper.append(', '); appendMathExpressionChild(wrapper, minimum); wrapper.append('))'); return wrapper;
  }
  if (node.block === 'mean_loss' && values.length) { wrapper.append('mean('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper; }
  if (node.block === 'softmax' && values.length) {
    const input = node.bindings?.logits || values[0]; const temperature = node.parameters?.temperature;
    wrapper.append('softmax('); appendMathExpressionChild(wrapper, input);
    if (temperature) { wrapper.append(' ÷ '); appendMathExpressionChild(wrapper, temperature); }
    wrapper.append(')'); return wrapper;
  }
  if (node.block === 'gather_by_label' && values.length >= 2) { appendMathExpressionChild(wrapper, values[0]); wrapper.append('['); appendMathExpressionChild(wrapper, values[1]); wrapper.append(']'); return wrapper; }
  if (node.block === 'elementwise_power' && values.length >= 2) { appendMathExpressionChild(wrapper, values[0]); wrapper.append('^'); appendMathExpressionChild(wrapper, values[1]); return wrapper; }
  if (node.block === 'affine_transform' && values.length) {
    const input = node.bindings?.input || expressionHole();
    const scale = node.parameters?.scale || expressionConstant(1);
    const bias = node.parameters?.bias || expressionConstant(0);
    wrapper.append('('); appendMathExpressionChild(wrapper, scale); wrapper.append(' × '); appendMathExpressionChild(wrapper, input); wrapper.append(') ＋ '); appendMathExpressionChild(wrapper, bias); return wrapper;
  }
  if (node.block === 'clamp_min' && values.length) {
    wrapper.append('max('); appendMathExpressionChild(wrapper, node.bindings?.input || values[0]); wrapper.append(', '); appendMathExpressionChild(wrapper, node.parameters?.minimum || expressionConstant('ε')); wrapper.append(')'); return wrapper;
  }
  if (node.block === 'sum_last_dimension' && values.length) { wrapper.append('Σ('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper; }
  if (node.block === 'sum_values' && values.length) { wrapper.append('Σᵢ('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper; }
  if (node.block === 'mean_squared_error' && values.length >= 2) {
    wrapper.append('mean(('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(' − '); appendMathExpressionChild(wrapper, values[1]); wrapper.append(')²)'); return wrapper;
  }
  if (node.block === 'per_sample_ce' && values.length >= 2) {
    wrapper.append('−log(softmax('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')['); appendMathExpressionChild(wrapper, values[1]); wrapper.append('])'); return wrapper;
  }
  if (node.block === 'soft_target_cross_entropy' && values.length >= 2) {
    wrapper.append('−Σ '); appendMathExpressionChild(wrapper, values[1]); wrapper.append(' × log softmax('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper;
  }
  if (node.block === 'symmetric_kl' && values.length >= 2) {
    wrapper.append('SKL('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(', '); appendMathExpressionChild(wrapper, values[1]); wrapper.append(')'); return wrapper;
  }
  if (node.block === 'select_by_indices' && values.length >= 2) { appendMathExpressionChild(wrapper, values[0]); wrapper.append('['); appendMathExpressionChild(wrapper, values[1]); wrapper.append(']'); return wrapper; }
  if (node.block === 'indices_to_mask' && values.length >= 2) { wrapper.append('1[j ∈ '); appendMathExpressionChild(wrapper, values[0]); wrapper.append(']'); return wrapper; }
  if (node.block === 'top_k_confidence' && values.length) { wrapper.append('TopK('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper; }
  if (node.block === 'select_all' && values.length) { wrapper.append('All('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper; }
  if (node.block === 'compose_transition' && values.length >= 2) { wrapper.append('normalize_rows('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(' × '); appendMathExpressionChild(wrapper, values[1]); wrapper.append(')'); return wrapper; }
  if (node.block === 'nonnegative_projection' && values.length) { wrapper.append('max(0, '); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper; }
  if (node.block === 'normalize_nonnegative_weights' && values.length) { appendMathExpressionChild(wrapper, values[0]); wrapper.append(' / Σ('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper; }
  if (node.block === 'one_hot_like' && values.length >= 2) { wrapper.append('one_hot('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(', classes('); appendMathExpressionChild(wrapper, values[1]); wrapper.append('))'); return wrapper; }
  if (node.block === 'one_hot' && values.length) { wrapper.append('one_hot('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper; }
  if (node.block === 'agreement_mask' && values.length >= 2) { wrapper.append('agree('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(', '); appendMathExpressionChild(wrapper, values[1]); wrapper.append(')'); return wrapper; }
  if (node.block === 'mask_logits' && values.length >= 2) { appendMathExpressionChild(wrapper, values[0]); wrapper.append(' ⊙ '); appendMathExpressionChild(wrapper, values[1]); return wrapper; }
  if (node.block === 'weighted_blend' && values.length >= 3) {
    const left = node.bindings?.left || expressionHole();
    const right = node.bindings?.right || expressionHole();
    const weight = node.bindings?.weight || expressionHole();
    appendMathExpressionChild(wrapper, weight); wrapper.append(' × '); appendMathExpressionChild(wrapper, left); wrapper.append(' ＋ (1 − '); appendMathExpressionChild(wrapper, weight); wrapper.append(') × '); appendMathExpressionChild(wrapper, right); return wrapper;
  }
  if (node.block === 'sharpen_distribution' && values.length) {
    const input = node.bindings?.input || expressionHole(); const temperature = node.parameters?.temperature || expressionConstant('T');
    wrapper.append('normalize('); appendMathExpressionChild(wrapper, input); wrapper.append('^(1/'); appendMathExpressionChild(wrapper, temperature); wrapper.append('))'); return wrapper;
  }
  if (node.block === 'row_normalize' && values.length) { wrapper.append('normalize_rows('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper; }
  if (node.block === 'apply_transition' && values.length >= 2) { appendMathExpressionChild(wrapper, values[0]); wrapper.append(' × '); appendMathExpressionChild(wrapper, values[1]); return wrapper; }
  if (node.block === 'safe_divide' && values.length >= 2) {
    const numerator = node.bindings?.numerator || values[0]; const denominator = node.bindings?.denominator || values[1]; const minimum = node.parameters?.minimum || expressionConstant('ε');
    appendMathExpressionChild(wrapper, numerator); wrapper.append(' ÷ max('); appendMathExpressionChild(wrapper, denominator); wrapper.append(', '); appendMathExpressionChild(wrapper, minimum); wrapper.append(')'); return wrapper;
  }
  if (node.block === 'detach' && values.length) { wrapper.append('stop_gradient('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper; }
  if (node.block === 'one_hot' && values.length) { wrapper.append('one_hot('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper; }
  if (node.block === 'zeros_like' && values.length) { wrapper.append('zeros_like('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper; }
  if (node.block === 'uniform_prior' && values.length) { wrapper.append('uniform_prior('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper; }
  if (node.block === 'quantile' && values.length) { wrapper.append('quantile('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(', '); appendMathExpressionChild(wrapper, node.parameters?.q || expressionConstant('q')); wrapper.append(')'); return wrapper; }
  if (node.block === 'threshold_mask' && values.length) { wrapper.append('1['); appendMathExpressionChild(wrapper, values[0]); wrapper.append(' '); appendMathExpressionChild(wrapper, node.parameters?.comparison || expressionConstant('≥')); wrapper.append(' '); appendMathExpressionChild(wrapper, node.parameters?.threshold || expressionConstant('τ')); wrapper.append(']'); return wrapper; }
  if (node.block === 'mask_to_indices' && values.length) { wrapper.append('indices('); appendMathExpressionChild(wrapper, values[0]); wrapper.append(')'); return wrapper; }
  if (node.block === 'invert_mask' && values.length) { wrapper.append('1 − '); appendMathExpressionChild(wrapper, values[0]); return wrapper; }
  if (node.block === 'weighted_sum') {
    const terms = node.parameters?.terms?.items || []; const weights = node.parameters?.weights?.items || [];
    wrapper.append('Σ(');
    terms.forEach((term, index) => { if (index) wrapper.append(' ＋ '); if (weights[index]) { appendMathExpressionChild(wrapper, weights[index]); wrapper.append(' × '); } appendMathExpressionChild(wrapper, term); });
    wrapper.append(')'); return wrapper;
  }
  const info = blockInfo(node.block); wrapper.append(`${info?.name || node.block}(`); values.forEach((child, index) => { if (index) wrapper.append(', '); appendMathExpressionChild(wrapper, child); }); wrapper.append(')'); return wrapper;
}

function replaceExpressionReference(root, target, replacement) {
  if (root === target) return replacement;
  expressionNodeEntries(root).forEach(([name, child]) => {
    const next = replaceExpressionReference(child, target, replacement);
    if (root.kind === 'operation') {
      if (Object.prototype.hasOwnProperty.call(root.bindings || {}, name)) root.bindings[name] = next;
      else if (Object.prototype.hasOwnProperty.call(root.parameters || {}, name)) root.parameters[name] = next;
    } else if (root.kind === 'array') root.items[Number(name.slice(1)) - 1] = next;
  });
  return root;
}

// Expression operations deliberately work on every node kind.  The renderer
// and future formula operations use this small API instead of implementing
// input/parameter/constant/array-specific mutations of their own.
function selectExpressionNode(node) {
  if (node?.kind === 'operation') exposeExpressionParameters(node);
  state.formulaEditor.expressionSelection = node || null;
  renderFormulaCanvas();
  return node;
}

function replaceExpressionNode(root, target, replacement) {
  return replaceExpressionReference(root, target, replacement);
}

function removeExpressionNode(root, target) {
  return pruneExpressionNode(root, target);
}

function wrapExpressionNode(root, target, blockId) {
  const info = blockInfo(blockId);
  if (!info || !target) return root;
  ensureFormulaEditorParameters(info, {}, {force: true});
  const binding = (info.requires || [])[0];
  const wrapper = {kind: 'operation', block: blockId, bindings: binding ? {[binding]: target} : {}, parameters: {}};
  Object.entries(info.params || {}).forEach(([name, schema]) => {
    if (schema.type !== 'slot') wrapper.parameters[name] = formulaEditorParameterNode(name, schema);
  });
  return replaceExpressionReference(root, target, wrapper);
}

function pruneExpressionNode(root, target) {
  if (root === target) return null;
  const entries = expressionNodeEntries(root);
  for (const [name, child] of entries) {
    if (child === target) {
      if (root.kind === 'array') {
        root.items.splice(Number(name.slice(1)) - 1, 1);
        return root;
      }
      const siblings = entries.filter(([key]) => key !== name).map(([, value]) => value).filter(Boolean);
      if (siblings.length) return siblings[0];
      return null;
    }
    const next = pruneExpressionNode(child, target);
    if (next !== child) {
      if (root.kind === 'operation') {
        if (Object.prototype.hasOwnProperty.call(root.bindings || {}, name)) root.bindings[name] = next;
        else if (Object.prototype.hasOwnProperty.call(root.parameters || {}, name)) root.parameters[name] = next;
      } else if (root.kind === 'array') {
        const index = Number(name.slice(1)) - 1;
        if (next === null) root.items.splice(index, 1); else root.items[index] = next;
        return root;
      }
      return root;
    }
  }
  return root;
}

function expressionOperationCandidates() {
  return formulaEditorCandidates().filter((info) => info.formula_kind);
}

function expressionInputCandidates() {
  return formulaInputRows().map((name) => expressionInput(name));
}

function expressionParameterCandidates() {
  return formulaParameterRows().map((item) => expressionParameter(item.name));
}

function expressionSymbolNames(root, kind, names = new Set()) {
  if (!root) return names;
  if (root.kind === kind && root.name) names.add(root.name);
  expressionNodeEntries(root).forEach(([, child]) => expressionSymbolNames(child, kind, names));
  return names;
}

function expressionFreshOperand(slotName, usedInputs = new Set(), usedParameters = new Set()) {
  const inputs = formulaInputRows();
  const parameters = formulaParameterRows().map((item) => item.name);
  const lower = String(slotName || '').toLowerCase();
  const parameterName = parameters.find((name) => {
    const value = name.toLowerCase();
    return !usedParameters.has(name) && (lower.includes('weight') || lower.includes('scale') || lower.includes('bias') || lower.includes('q') || lower.includes('epsilon') || lower.includes('minimum') || value.includes(lower));
  });
  if (parameterName) { usedParameters.add(parameterName); return expressionParameter(parameterName); }
  const preferred = inputs.find((name) => {
    const value = name.toLowerCase();
    return !usedInputs.has(name) && ((lower.includes('logit') && value.includes('logit')) || ((lower.includes('label') || lower.includes('target')) && (value.includes('label') || value.includes('target'))));
  });
  const inputName = preferred || inputs.find((name) => !usedInputs.has(name)) || inputs[0] || slotName || 'input';
  if (inputs.includes(inputName)) usedInputs.add(inputName);
  return expressionInput(inputName);
}

function expressionOperationNode(info, selected = null, current = null) {
  const bindings = {};
  const parameters = {};
  const required = info.requires || [];
  // Every required input starts as an explicit UI-only hole.  An existing
  // selection occupies the first input so binary composition reads naturally
  // as `selected + □`; a selected hole is simply replaced by this template.
  required.forEach((slotName, index) => { bindings[slotName] = selected && index === 0 && selected.kind !== 'hole' ? selected : expressionHole(); });
  Object.entries(info.params || {}).forEach(([name, schema]) => {
    // Slot parameters are connections, not mathematical operands. In
    // particular, save_as is generated during serialization and must never
    // appear as an extra argument in the equation canvas.
    if (schema.type === 'slot') return;
    if (name === 'terms') {
      const defaults = Array.isArray(schema.default) ? schema.default : [];
      parameters[name] = {kind: 'array', items: defaults.length ? defaults.map((item) => expressionConstant(item)) : [expressionHole()]};
    } else if (name === 'weights') {
      const defaults = Array.isArray(schema.default) ? schema.default : [];
      parameters[name] = {kind: 'array', items: defaults.length ? defaults.map((item) => expressionConstant(item)) : [expressionConstant(1)]};
    } else {
      const matching = formulaParameterRows().find((item) => item.name.toLowerCase() === name.toLowerCase());
      parameters[name] = matching ? expressionParameter(matching.name) : expressionConstant(schema.default);
    }
  });
  return {kind: 'operation', block: info.id, bindings, parameters};
}

function addExpressionOperation(blockId) {
  const info = blockInfo(blockId);
  if (!info) return;
  ensureFormulaEditorParameters(info, {}, {force: true});
  const selected = state.formulaEditor.expressionSelection;
  const current = state.formulaEditor.expression;
  const node = expressionOperationNode(info, selected, current);
  if (!current) state.formulaEditor.expression = node;
  else if (selected) state.formulaEditor.expression = replaceExpressionReference(current, selected, node);
  else state.formulaEditor.expression = replaceExpressionReference(current, current, node);
  state.formulaEditor.expressionSelection = firstExpressionHole(node) || node;
  syncFormulaStepsFromExpression();
  renderFormulaEditor();
}

function expressionReplaceOperation(target, blockId) {
  const info = blockInfo(blockId);
  if (!info) return;
  ensureFormulaEditorParameters(info, {}, {force: true});
  const oldValues = expressionNodeEntries(target).map(([, child]) => child);
  const bindings = {};
  let valueIndex = 0;
  Object.entries(info.params || {}).forEach(([name, schema]) => {
    if (schema.type === 'slot' && !isOutputSlot(name)) bindings[name] = oldValues[valueIndex++] || expressionInput(formulaInputRows()[0] || name);
  });
  const parameters = {};
  Object.entries(info.params || {}).forEach(([name, schema]) => {
    if (schema.type !== 'slot') parameters[name] = formulaEditorParameterNode(name, schema, target.parameters?.[name]);
  });
  target.block = blockId; target.bindings = bindings; target.parameters = parameters;
}

function formulaVariantSelectorValue(spec, target, key) {
  const isInput = Boolean(spec?.inputs && Object.prototype.hasOwnProperty.call(spec.inputs, key));
  const node = isInput ? target?.bindings?.[key] : target?.parameters?.[key];
  if (isInput) return {present: Boolean(node && node.kind !== 'hole'), known: true, value: expressionNodeValue(node)};
  if (node?.kind === 'constant') return {present: true, known: true, value: node.value};
  if (node?.kind === 'parameter') {
    const editorSchema = state.formulaEditor.parameterSchemas?.[node.name];
    if (editorSchema && Object.prototype.hasOwnProperty.call(editorSchema, 'default')) return {present: true, known: true, value: editorSchema.default};
    const specSchema = spec?.parameters?.[node.name];
    if (specSchema && Object.prototype.hasOwnProperty.call(specSchema, 'default')) return {present: true, known: true, value: specSchema.default};
    return {present: true, known: false, value: undefined};
  }
  if (!node) {
    const schema = spec?.parameters?.[key];
    if (schema && Object.prototype.hasOwnProperty.call(schema, 'default')) return {present: true, known: true, value: schema.default};
    return {present: false, known: false, value: undefined};
  }
  return {present: true, known: true, value: expressionNodeValue(node)};
}

function formulaVariantMatchesExpression(spec, target, variant) {
  return Object.entries(variant?.when || {}).every(([key, expected]) => {
    const actual = formulaVariantSelectorValue(spec, target, key);
    if (expected && typeof expected === 'object' && !Array.isArray(expected)) {
      if (Object.prototype.hasOwnProperty.call(expected, 'present') && Boolean(expected.present) !== actual.present) return false;
      if (Object.prototype.hasOwnProperty.call(expected, 'equals') && (!actual.known || actual.value !== expected.equals)) return false;
      return true;
    }
    if (expected === '__present__' || expected === '$present') return actual.present;
    if (expected === '__absent__' || expected === '$absent') return !actual.present;
    return actual.known && actual.value === expected;
  });
}

function formulaVariantForExpression(spec, target) {
  return (Array.isArray(spec?.variants) ? spec.variants : []).find((variant) => formulaVariantMatchesExpression(spec, target, variant)) || null;
}

// Replace a composite node with the expression described by its FormulaSpec.
// This is an editor-only transformation: execution still serializes the
// resulting expression through the existing FormulaSpec step format.  When a
// FormulaSpec has executable variants, expand the branch selected by the
// node's current inputs/parameters rather than always showing the default.
function expandCompositeExpression(target) {
  if (!target || target.kind !== 'operation') return false;
  const info = blockInfo(target.block);
  const spec = formulaDefinitionForBlock(info);
  if (!info || info.formula_kind !== 'composite' || !spec) return false;
  const selectedVariant = formulaVariantForExpression(spec, target);
  const steps = selectedVariant?.steps || spec.steps || [];
  const outputs = selectedVariant?.outputs && Object.keys(selectedVariant.outputs).length ? selectedVariant.outputs : spec.outputs;
  const values = new Map();
  const resolve = (value) => {
    if (Array.isArray(value)) return {kind: 'array', items: value.map(resolve)};
    if (value && typeof value === 'object') return expressionConstant(cloneExpressionValue(value));
    if (typeof value === 'string') {
      if (Object.prototype.hasOwnProperty.call(spec.inputs, value)) return target.bindings?.[value] || expressionHole();
      if (Object.prototype.hasOwnProperty.call(spec.parameters, value)) return target.parameters?.[value] || expressionConstant(spec.parameters[value].default);
      if (value.startsWith('$') && Object.prototype.hasOwnProperty.call(spec.parameters, value.slice(1))) return target.parameters?.[value.slice(1)] || expressionConstant(spec.parameters[value.slice(1)].default);
      if (values.has(value)) return values.get(value);
    }
    return expressionConstant(value);
  };
  steps.forEach((step) => {
    const bindings = Object.fromEntries(Object.entries(step.bindings || {}).map(([name, value]) => [name, resolve(value)]));
    const parameters = Object.fromEntries(Object.entries(step.parameters || {}).map(([name, value]) => [name, resolve(value)]));
    values.set(step.id, {kind: 'operation', block: step.block, bindings, parameters});
  });
  const output = Object.values(outputs || {})[0];
  const expanded = output ? values.get(String(output.source).split('.')[0]) : null;
  if (!expanded) return false;
  state.formulaEditor.expression = replaceExpressionReference(state.formulaEditor.expression, target, expanded);
  state.formulaEditor.expressionSelection = expanded;
  syncFormulaStepsFromExpression();
  renderFormulaEditor();
  return true;
}

function renderFormulaExpressionActions() {
  const container = $('formula-expression-actions');
  if (!container) return;
  container.replaceChildren();
  const selected = state.formulaEditor.expressionSelection;
  if (!selected) { container.textContent = '点击公式中的任意部分以选中；输入和参数也可以作为独立对象编辑。'; return; }
  const selectedLabel = document.createElement('strong'); selectedLabel.textContent = `已选中：${expressionNodeText(selected)}`; container.appendChild(selectedLabel);
  const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = '删除此表达式'; remove.onclick = deleteSelectedFormulaExpression; container.appendChild(remove);
  const selectedInfo = selected.kind === 'operation' ? blockInfo(selected.block) : null;
  if (selectedInfo?.formula_kind === 'composite' && formulaDefinitionForBlock(selectedInfo)) {
    const expand = document.createElement('button'); expand.type = 'button'; expand.textContent = '展开后编辑'; expand.title = '把公式模板替换为它的基础运算链'; expand.onclick = () => expandCompositeExpression(selected); container.appendChild(expand);
  }
  const replace = document.createElement('select'); replace.title = '替换当前节点';
  const replacementGroups = [
    {label: '输入', values: expressionInputCandidates().map((node) => ({value: `input:${node.name}`, label: node.name}))},
    {label: '参数', values: expressionParameterCandidates().map((node) => ({value: `parameter:${node.name}`, label: node.name}))},
  ];
  if (selected.kind === 'operation') replacementGroups.push({label: '运算', values: expressionOperationCandidates().map((info) => ({value: `operation:${info.id}`, label: info.name, selected: info.id === selected.block}))});
  replacementGroups.forEach((group) => {
    if (!group.values.length) return;
    const optgroup = document.createElement('optgroup'); optgroup.label = group.label;
    group.values.forEach((item) => { const option = document.createElement('option'); option.value = item.value; option.textContent = item.label; option.selected = Boolean(item.selected); optgroup.appendChild(option); });
    replace.appendChild(optgroup);
  });
  if (replace.options.length) {
    const replaceButton = document.createElement('button'); replaceButton.type = 'button'; replaceButton.textContent = '替换'; replaceButton.onclick = () => {
      const [kind, name] = String(replace.value || '').split(':');
      const replacement = kind === 'input' ? expressionInput(name) : kind === 'parameter' ? expressionParameter(name) : null;
      if (replacement) state.formulaEditor.expression = replaceExpressionReference(state.formulaEditor.expression, selected, replacement);
      else if (kind === 'operation') expressionReplaceOperation(selected, name);
      state.formulaEditor.expressionSelection = replacement || selected;
      syncFormulaStepsFromExpression(); renderFormulaEditor();
    };
    container.append(replace, replaceButton);
  }
  if (selected.kind === 'constant') {
    const constant = document.createElement('input'); constant.type = 'text'; constant.value = parameterDefaultText({default: selected.value}); constant.title = '编辑常量（数字、文本或结构化值）';
    const constantButton = document.createElement('button'); constantButton.type = 'button'; constantButton.textContent = '更新常量'; constantButton.onclick = () => {
      const raw = constant.value.trim(); let value = raw;
      try { value = JSON.parse(raw); } catch (error) { /* plain text constants remain strings */ }
      selected.value = value; syncFormulaStepsFromExpression(); renderFormulaEditor();
    };
    container.append(constant, constantButton);
  }
  const unary = expressionOperationCandidates().filter((info) => (info.requires || []).length === 1);
  if (unary.length) {
    const wrap = document.createElement('select'); wrap.title = '用一元运算包裹'; unary.forEach((info) => { const option = document.createElement('option'); option.value = info.id; option.textContent = `包裹为：${info.name}`; wrap.appendChild(option); });
    const wrapButton = document.createElement('button'); wrapButton.type = 'button'; wrapButton.textContent = '包裹'; wrapButton.onclick = () => { const info = blockInfo(wrap.value); ensureFormulaEditorParameters(info, {}, {force: true}); const binding = (info.requires || [])[0]; const wrapper = {kind: 'operation', block: wrap.value, bindings: {[binding]: selected}, parameters: {}}; Object.entries(info.params || {}).forEach(([name, schema]) => { if (schema.type !== 'slot') wrapper.parameters[name] = formulaEditorParameterNode(name, schema); }); state.formulaEditor.expression = replaceExpressionReference(state.formulaEditor.expression, selected, wrapper); state.formulaEditor.expressionSelection = wrapper; syncFormulaStepsFromExpression(); renderFormulaEditor(); }; container.append(wrap, wrapButton);
  }
}

function expressionToFormulaStepsForOutputs(outputExpressions, outputOrder = null) {
  const steps = []; const used = new Set(); const generated = new WeakMap(); const sources = {};
  const safeName = (value) => String(value || 'step').replace(/[^A-Za-z0-9_]/g, '_').replace(/^[^A-Za-z]+/, '') || 'step';
  const value = (node) => {
    if (!node) return null;
    if (node.kind === 'hole') return null;
    if (node.kind === 'input' || node.kind === 'parameter') return node.name;
    if (node.kind === 'constant') return node.value;
    if (node.kind === 'array') return node.items.map(value);
    return visit(node, false);
  };
  const visit = (node, requestedId = null) => {
    if (!node || node.kind !== 'operation') return value(node);
    if (generated.has(node)) return generated.get(node);
    let id = requestedId || safeName(node.block);
    let suffix = 2;
    while (used.has(id)) id = `${safeName(requestedId || node.block)}_${suffix++}`;
    used.add(id);
    generated.set(node, id);
    const bindings = {}; Object.entries(node.bindings || {}).forEach(([name, child]) => { bindings[name] = value(child); });
    const parameters = {}; Object.entries(node.parameters || {}).forEach(([name, child]) => { parameters[name] = value(child); });
    // A literal occupying a slot is a real Formula operand, not a valid slot
    // name.  Materialize it as an implicit constant step so ``x + 1`` can be
    // saved and reopened without exposing this implementation detail.
    const definition = blockInfo(node.block);
    Object.entries(node.bindings || {}).forEach(([name, child]) => {
      if (child?.kind !== 'constant' || definition?.params?.[name]?.type !== 'slot') return;
      let constantId = generated.get(child);
      if (!constantId) {
        constantId = safeName(`${node.block}_${name}_constant`); let constantSuffix = 2;
        while (used.has(constantId)) constantId = `${safeName(node.block)}_${name}_constant_${constantSuffix++}`;
        used.add(constantId); generated.set(child, constantId);
        steps.push({id: constantId, block: 'constant', bindings: {}, parameters: {value: child.value}});
      }
      bindings[name] = constantId;
    });
    steps.push({id, block: node.block, bindings, parameters});
    return id;
  };
  const entries = outputOrder || Object.keys(outputExpressions || {});
  entries.forEach((name) => { if (outputExpressions?.[name]) sources[name] = visit(outputExpressions[name], name); });
  return {steps, sources};
}

function expressionToFormulaSteps(root, outputName) {
  if (!root) return [];
  return expressionToFormulaStepsForOutputs({[outputName || 'loss']: root}, [outputName || 'loss']).steps;
}

function syncFormulaStepsFromExpression() {
  rememberActiveFormulaOutput();
  const outputNames = Object.keys(state.formulaEditor.outputExpressions || {});
  const serialized = expressionToFormulaStepsForOutputs(state.formulaEditor.outputExpressions || {}, outputNames);
  state.formulaEditor.steps = serialized.steps;
  const output = $('formula-output-source');
  const activeSource = serialized.sources[state.formulaEditor.activeOutput];
  if (output && activeSource && [...output.options].some((option) => option.value === activeSource)) output.value = activeSource;
}

function renderFormulaStructureTree() {
  const tree = $('formula-expression-tree');
  if (!tree) return;
  tree.replaceChildren();
  if (state.formulaEditor.expression) tree.appendChild(renderExpressionNode(state.formulaEditor.expression));
  else tree.textContent = '尚未创建表达式结构';
}

function renderFormulaCanvas() {
  const canvas = $('formula-canvas');
  if (!canvas) return;
  rememberActiveFormulaOutput();
  renderFormulaOutputFields();
  const active = state.formulaEditor.activeOutput || 'loss';
  if (!state.formulaEditor.expression && state.formulaEditor.outputExpressions?.[active]) state.formulaEditor.expression = state.formulaEditor.outputExpressions[active];
  if (!state.formulaEditor.expression && state.formulaEditor.steps.length) {
    const inputs = new Set(parseEditorInputs());
    let parameters = new Set(); try { parameters = new Set(Object.keys(parseEditorParameters())); } catch (error) { /* save reports malformed parameters */ }
    const source = $('formula-output-source')?.value || state.formulaEditor.steps.at(-1)?.id;
    state.formulaEditor.expression = formulaStepsToExpression(state.formulaEditor.steps, source, inputs, parameters);
  }
  // Existing FormulaSpec steps may carry scalar arguments that were never
  // promoted to the editor's parameter section.  Normalize that view once
  // before rendering so every operation exposes the same editable variables.
  exposeExpressionParameters(state.formulaEditor.expression);
  canvas.replaceChildren();
  if (!state.formulaEditor.expression) { canvas.textContent = '公式画布：尚未添加步骤'; renderFormulaStructureTree(); renderFormulaExpressionActions(); return; }
  const heading = document.createElement('div'); heading.className = 'formula-canvas-heading'; heading.textContent = `${state.formulaEditor.activeOutput || '输出'} =`; canvas.appendChild(heading);
  canvas.appendChild(renderMathExpressionNode(state.formulaEditor.expression));
  renderFormulaStructureTree();
  renderFormulaExpressionActions();
}

function formulaEquationSources() {
  const sources = parseEditorInputs();
  sources.push(...state.formulaEditor.steps.map((step) => step.id));
  return [...new Set(sources)];
}

function renderFormulaEquationBuilder() {
  const left = $('formula-equation-left');
  const right = $('formula-equation-right');
  if (!left || !right) return;
  const sources = formulaEquationSources();
  const previousLeft = left.value;
  const previousRight = right.value;
  [left, right].forEach((select) => {
    select.replaceChildren();
    sources.forEach((source) => { const option = document.createElement('option'); option.value = source; option.textContent = source; select.appendChild(option); });
  });
  if (sources.includes(previousLeft)) left.value = previousLeft;
  if (sources.includes(previousRight)) right.value = previousRight;
  if (!left.value && sources.length) left.value = sources[0];
  if (!right.value && sources.length > 1) right.value = sources[1];
  const apply = $('formula-apply-equation');
  if (apply) apply.disabled = sources.length < 2;
}

function addFormulaEquation() {
  const left = $('formula-equation-left')?.value || '';
  const right = $('formula-equation-right')?.value || '';
  const block = $('formula-equation-operator')?.value || 'add';
  const outputName = $('formula-output-name')?.value.trim() || 'loss';
  const validation = $('formula-editor-validation');
  if (!left || !right) { if (validation) validation.textContent = '请先定义至少两个输入变量，再组合等式'; return; }
  const base = outputName.replace(/[^A-Za-z0-9_]/g, '_').replace(/^[^A-Za-z]+/, '') || 'sum';
  let id = base;
  let suffix = 2;
  while (state.formulaEditor.steps.some((step) => step.id === id)) id = `${base}_${suffix++}`;
  const bindings = block === 'safe_divide'
    ? {numerator: left, denominator: right}
    : {left, right};
  state.formulaEditor.steps.push({id, block, bindings, parameters: {}});
  renderFormulaEditor();
  const output = $('formula-output-source');
  if (output) output.value = id;
  renderFormulaCanvas();
  const operator = {add: '＋', subtract: '−', elementwise_multiply: '×', safe_divide: '÷'}[block] || '运算';
  if (validation) validation.textContent = `已添加等式：${outputName} = ${left} ${operator} ${right}`;
}

function formulaEditorSources() {
  const sources = [...parseEditorInputs()];
  try { sources.push(...Object.keys(parseEditorParameters())); } catch (error) { /* save reports the format error */ }
  sources.push(...state.formulaEditor.steps.map((step) => step.id));
  return [...new Set(sources)];
}

function formulaExpressionValue(source, expressions) {
  if (typeof source !== 'string') return JSON.stringify(source);
  return expressions[source] || source;
}

function formatFormulaCall(name, args) {
  if (!args.length) return `${name}()`;
  const body = args.map((arg) => `  ${String(arg).replace(/\n/g, '\n  ')}`).join(',\n');
  return `${name}(\n${body}\n)`;
}

function formulaStepExpression(step, expressions) {
  const info = blockInfo(step.block) || {};
  const bindings = step.bindings || {};
  const args = (info.requires || Object.keys(bindings)).map((name) =>
    formulaExpressionValue(bindings[name] || name, expressions));
  Object.entries(step.parameters || {}).forEach(([name, value]) => {
    if (!(info.requires || []).includes(name)) {
      args.push(`${name}=${formulaExpressionValue(value, expressions)}`);
    }
  });
  return formatFormulaCall(step.block, args);
}

function renderNestedFormulaPreview(steps) {
  const expressions = {};
  (steps || []).forEach((step) => {
    expressions[step.id] = formulaStepExpression(step, expressions);
  });
  const output = $('formula-output-source')?.value || steps?.at(-1)?.id;
  return output && expressions[output] ? expressions[output] : '尚未选择输出';
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

function renderFormulaEditorStepParameterFields() {
  const container = $('formula-editor-step-parameter-fields');
  const info = blockInfo($('formula-editor-block')?.value);
  if (!container) return;
  container.innerHTML = '';
  if (!info) return;
  const parameters = Object.entries(info.params || {}).filter(([name, schema]) => schema.type !== 'slot' || isOutputSlot(name));
  if (!parameters.length) { container.textContent = '该操作没有额外参数。'; return; }
  parameters.forEach(([name, schema]) => {
    const before = container.children.length;
    appendFormulaField(container, name, schema, null);
    const field = container.children[before]?.querySelector('[data-formula-param]');
    if (field) field.dataset.formulaStepParam = name;
  });
}

function deleteSelectedFormulaExpression() {
  const selected = state.formulaEditor.expressionSelection;
  if (!selected || !state.formulaEditor.expression) return false;
  state.formulaEditor.expression = pruneExpressionNode(state.formulaEditor.expression, selected);
  state.formulaEditor.expressionSelection = null;
  syncFormulaStepsFromExpression();
  renderFormulaEditor();
  return true;
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
  renderFormulaEditorStepParameterFields();
  const currentPreview = $('formula-editor-current-preview');
  const selectedInfo = blockInfo(select.value);
  if (currentPreview) {
    currentPreview.replaceChildren();
    const label = document.createElement('strong'); label.textContent = selectedInfo ? `当前运算：${selectedInfo.name}` : '当前运算'; currentPreview.appendChild(label);
    const math = document.createElement('div'); renderFormulaOrIntro(math, selectedInfo, {compact: true}); currentPreview.appendChild(math);
    if (selectedInfo?.formula_kind === 'composite' && selectedInfo.formula_ref) {
      const composition = document.createElement('details'); composition.className = 'formula-composition-details';
      composition.appendChild(Object.assign(document.createElement('summary'), {textContent: '查看组成'}));
      const definition = formulaDefinitionForBlock(selectedInfo);
      if (definition) {
        const chain = document.createElement('ol'); chain.className = 'formula-composition-chain';
        (definition.steps || []).forEach((step) => {
          const item = document.createElement('li'); const operation = blockInfo(step.block);
          item.textContent = operation ? `${operation.name}（${step.block}）` : step.block;
          chain.appendChild(item);
        });
        composition.appendChild(chain);
      } else {
        composition.appendChild(Object.assign(document.createElement('code'), {textContent: selectedInfo.formula || `定义：${selectedInfo.formula_ref}`}));
      }
      const expandHint = document.createElement('small'); expandHint.textContent = '选中公式节点后可“展开后编辑”；展开只改变编辑表达式，不改变执行器。'; composition.appendChild(expandHint);
      currentPreview.appendChild(composition);
    }
  }
  const editorPalette = $('formula-editor-palette');
  if (editorPalette) {
    editorPalette.innerHTML = '';
    const disclosure = document.createElement('details');
    disclosure.className = 'formula-palette-disclosure';
    disclosure.open = state.formulaEditor.paletteExpanded !== false;
    disclosure.addEventListener('toggle', () => { state.formulaEditor.paletteExpanded = disclosure.open; });
    const summary = document.createElement('summary');
    summary.textContent = `全部运算（${candidates.length}）`;
    summary.title = '展开或收起全部数学运算';
    const body = document.createElement('div');
    body.className = 'formula-palette-body';
    const groups = new Map(FORMULA_EDITOR_KIND_GROUPS.map((group) => [group, []]));
    candidates.forEach((info) => {
      const group = formulaEditorKindGroup(info);
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push(info);
    });
    groups.forEach((items, groupName) => {
      if (!items.length) return;
      const group = document.createElement('details');
      group.className = 'formula-operation-group';
      group.open = true;
      const groupSummary = document.createElement('summary');
      groupSummary.textContent = `${groupName}（${items.length}）`;
      const groupBody = document.createElement('div');
      groupBody.className = 'formula-operation-group-body';
      const appendOperation = (info) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'formula-palette-block';
        const name = document.createElement('strong');
        name.textContent = info.name;
        button.appendChild(name);
        const formula = document.createElement('div');
        formula.className = 'formula-palette-formula';
        renderFormulaOrIntro(formula, info, {compact: true});
        button.appendChild(formula);
        button.title = info.description || '点击插入这个数学运算';
        button.onclick = () => addExpressionOperation(info.id);
        groupBody.appendChild(button);
      };
      if (groupName === '基础运算') {
        const subgroups = new Map(FORMULA_EDITOR_GROUPS.map((name) => [name, []]));
        items.forEach((info) => { const name = formulaEditorGroup(info); if (!subgroups.has(name)) subgroups.set(name, []); subgroups.get(name).push(info); });
        subgroups.forEach((subitems, subgroupName) => {
          if (!subitems.length) return;
          const subgroup = document.createElement('details'); subgroup.className = 'formula-operation-subgroup'; subgroup.open = true;
          subgroup.appendChild(Object.assign(document.createElement('summary'), {textContent: `${subgroupName}（${subitems.length}）`}));
          const subgroupBody = document.createElement('div'); subgroupBody.className = 'formula-operation-subgroup-body';
          subitems.forEach((info) => { const previous = groupBody.children.length; appendOperation(info); subgroupBody.appendChild(groupBody.children[previous]); });
          subgroup.appendChild(subgroupBody); groupBody.appendChild(subgroup);
        });
      } else items.forEach(appendOperation);
      group.append(groupSummary, groupBody);
      body.appendChild(group);
    });
    disclosure.append(summary, body);
    editorPalette.appendChild(disclosure);
  }
  const list = $('formula-editor-step-list'); list.innerHTML = '';
  state.formulaEditor.steps.forEach((step, index) => {
    const row = document.createElement('div'); row.className = 'formula-step-item';
    const info = blockInfo(step.block) || {};
    const body = document.createElement('div'); body.className = 'formula-step-body';
    const label = document.createElement('span'); label.innerHTML = `${index + 1}. <code>${step.id}</code> ← ${info.name || step.block}`;
    body.appendChild(label);
    const visual = document.createElement('div'); visual.className = 'formula-step-math'; renderFormulaOrIntro(visual, info, {compact: true}); body.appendChild(visual);
    const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = '删除'; remove.onclick = () => { state.formulaEditor.steps.splice(index, 1); const inputs = new Set(parseEditorInputs()); let parameters = new Set(); try { parameters = new Set(Object.keys(parseEditorParameters())); } catch (error) { /* save reports malformed parameters */ } state.formulaEditor.expression = formulaStepsToExpression(state.formulaEditor.steps, state.formulaEditor.steps.at(-1)?.id, inputs, parameters); state.formulaEditor.expressionSelection = null; renderFormulaEditor(); };
    row.append(body, remove); list.appendChild(row);
  });
  const output = $('formula-output-source'); output.innerHTML = '';
  rememberActiveFormulaOutput();
  const serializedOutputs = expressionToFormulaStepsForOutputs(
    state.formulaEditor.outputExpressions || {},
    Object.keys(state.formulaEditor.outputExpressions || {}),
  );
  Object.entries(serializedOutputs.sources).forEach(([outputName, source]) => {
    const option = document.createElement('option');
    option.value = source;
    option.textContent = `${outputName} → ${source}`;
    option.dataset.formulaOutputName = outputName;
    output.appendChild(option);
  });
  const activeSource = serializedOutputs.sources[state.formulaEditor.activeOutput];
  if (activeSource && [...output.options].some((option) => option.value === activeSource)) output.value = activeSource;
  const preview = $('formula-preview');
  if (!state.formulaEditor.steps.length) {
    preview.replaceChildren();
    preview.appendChild(Object.assign(document.createElement('span'), {textContent: '公式预览：尚未添加步骤'}));
  } else {
    preview.replaceChildren();
    const heading = document.createElement('strong'); heading.className = 'math-preview-heading'; heading.textContent = '公式预览（按步骤组合）'; preview.appendChild(heading);
    state.formulaEditor.steps.forEach((step, index) => {
      const info = blockInfo(step.block) || {};
      const visual = document.createElement('div'); visual.className = 'formula-chain-step';
      const stepLabel = document.createElement('small'); stepLabel.textContent = `${index + 1}. ${info.name || step.block}`; visual.appendChild(stepLabel);
      const math = document.createElement('div'); renderFormulaOrIntro(math, info, {compact: true}); visual.appendChild(math); preview.appendChild(visual);
      if (index < state.formulaEditor.steps.length - 1) preview.appendChild(Object.assign(document.createElement('div'), {className: 'formula-chain-arrow', textContent: '↓'}));
    });
    const expression = document.createElement('details'); expression.className = 'formula-expression-details';
    expression.appendChild(Object.assign(document.createElement('summary'), {textContent: '查看组合表达式'}));
    expression.appendChild(Object.assign(document.createElement('code'), {textContent: renderNestedFormulaPreview(state.formulaEditor.steps)}));
    preview.appendChild(expression);
  }
  renderFormulaInputFields();
  renderFormulaParameterFields();
  renderFormulaCanvas();
}

function resetFormulaEditor() {
  state.formulaEditor.steps = []; state.formulaEditor.expression = null; state.formulaEditor.expressionSelection = null; state.formulaEditor.editingId = null; state.formulaEditor.paletteExpanded = true; state.formulaEditor.parameterStates = {};
  state.formulaEditor.activeOutput = 'loss'; state.formulaEditor.outputExpressions = {loss: null}; state.formulaEditor.outputSchemas = {};
  state.formulaEditor.inputSchemas = {logits: {description: '', type: 'tensor'}, targets: {description: '', type: 'labels'}};
  state.formulaEditor.parameterSchemas = {epsilon: {type: 'float', default: 1e-8}};
  state.formulaEditor.outputSchemas = {};
  $('formula-id').value = 'user/my_formula'; $('formula-name').value = 'My Formula'; $('formula-description').value = '';
  $('formula-inputs').value = 'logits\ntargets'; $('formula-parameters').value = 'epsilon:float=1e-8'; $('formula-output-name').value = 'loss';
  $('formula-editor-step-id').value = '';
  if ($('formula-editor-advanced')) $('formula-editor-advanced').open = false;
  renderFormulaEditor();
  $('formula-editor-validation').textContent = '';
}

function loadFormulaIntoEditor(item) {
  $('formula-list-dialog')?.close();
  state.formulaEditor.editingId = item.id;
  state.formulaEditor.steps = (item.steps || []).map((step) => ({ id: step.id, block: step.block, bindings: {...(step.bindings || {})}, parameters: {...(step.parameters || {})} }));
  state.formulaEditor.inputSchemas = cloneSchema(item.inputs || {});
  state.formulaEditor.parameterSchemas = cloneSchema(item.parameters || {});
  state.formulaEditor.outputSchemas = cloneSchema(item.outputs || {});
  state.formulaEditor.parameterStates = {};
  const outputNames = Object.keys(item.outputs || {});
  state.formulaEditor.activeOutput = outputNames[0] || 'loss';
  state.formulaEditor.outputExpressions = outputNames.length ? formulaStepsToExpressions(item.steps || [], item.outputs, new Set(Object.keys(item.inputs || {})), new Set(Object.keys(item.parameters || {}))) : {loss: null};
  $('formula-id').value = item.id; $('formula-name').value = item.name || ''; $('formula-description').value = item.description || '';
  $('formula-inputs').value = Object.keys(item.inputs || {}).join('\n');
  $('formula-parameters').value = Object.entries(item.parameters || {}).map(([name, schema]) => parameterSchemaLine(name, schema)).join('\n');
  const outputName = state.formulaEditor.activeOutput;
  $('formula-output-name').value = outputName;
  const inputNames = new Set(Object.keys(item.inputs || {}));
  const parameterNames = new Set(Object.keys(item.parameters || {}));
  const outputSource = item.outputs?.[outputName]?.source?.split('.')[0] || state.formulaEditor.steps.at(-1)?.id;
  state.formulaEditor.expression = state.formulaEditor.outputExpressions[outputName] || formulaStepsToExpression(state.formulaEditor.steps, outputSource, inputNames, parameterNames);
  state.formulaEditor.expressionSelection = null;
  if ($('formula-editor-advanced')) $('formula-editor-advanced').open = false;
  renderFormulaEditor();
  const loadedSource = item.outputs?.[outputName]?.source?.split('.')[0];
  if (loadedSource && $('formula-output-source') && [...$('formula-output-source').options].some((option) => option.value === loadedSource)) $('formula-output-source').value = loadedSource;
  $('formula-editor-validation').textContent = '';
  $('formula-editor-dialog').showModal();
}

function openBuiltinFormulaCopy(item, formulas = []) {
  const used = new Set(formulas.map((formula) => formula.id));
  const base = String(item.id || 'formula').replace(/^builtin\//, '').replace(/[^A-Za-z0-9_]/g, '_') || 'formula';
  let id = `user/${base}`;
  let suffix = 2;
  while (used.has(id)) id = `user/${base}_${suffix++}`;
  const copy = JSON.parse(JSON.stringify(item));
  copy.id = id;
  copy.name = `${item.name || base}（我的副本）`;
  delete copy.formula_hash;
  delete copy.block_id;
  loadFormulaIntoEditor(copy);
  state.formulaEditor.editingId = null;
  $('formula-editor-validation').textContent = `已复制为 ${id}；保存后写入“我的公式”`;
}

function addFormulaEditorStep() {
  const block = $('formula-editor-block').value;
  const info = blockInfo(block);
  const validation = $('formula-editor-validation');
  if (!info) { validation.textContent = '请先选择一个公共数学运算'; return; }
  ensureFormulaEditorParameters(info, {}, {force: true});
  const idInput = $('formula-editor-step-id');
  let id = idInput.value.trim();
  if (!id) {
    const base = String(block || 'step').replace(/[^A-Za-z0-9_]/g, '_').replace(/^[^A-Za-z]+/, '') || 'step';
    id = base;
    let suffix = 2;
    while (state.formulaEditor.steps.some((step) => step.id === id)) id = `${base}_${suffix++}`;
    idInput.value = id;
  }
  if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(id) || state.formulaEditor.steps.some((step) => step.id === id)) { $('formula-editor-validation').textContent = 'step id 必须唯一且为字母/数字/下划线'; return; }
  const bindings = {};
  $('formula-editor-binding-fields').querySelectorAll('[data-formula-binding]').forEach((input) => {
    if (input.value) bindings[input.dataset.formulaBinding] = input.value;
  });
  let parameters = {};
  try {
    $('formula-editor-step-parameter-fields').querySelectorAll('[data-formula-step-param]').forEach((input) => {
      parameters[input.dataset.formulaStepParam] = formulaFieldValue(input, info.params[input.dataset.formulaStepParam]);
    });
  } catch (error) { $('formula-editor-validation').textContent = error.message; return; }
  state.formulaEditor.steps.push({ id, block, bindings, parameters });
  const inputNames = new Set(parseEditorInputs());
  let parameterNames = new Set(); try { parameterNames = new Set(Object.keys(parseEditorParameters())); } catch (error) { /* save reports malformed parameters */ }
  state.formulaEditor.expression = formulaStepsToExpression(state.formulaEditor.steps, id, inputNames, parameterNames);
  state.formulaEditor.outputExpressions[state.formulaEditor.activeOutput || 'loss'] = state.formulaEditor.expression;
  state.formulaEditor.expressionSelection = null;
  $('formula-editor-step-id').value = '';
  validation.textContent = `已添加步骤：${id}`;
  renderFormulaEditor();
}

async function saveFormulaEditor() {
  const errorTarget = $('formula-editor-validation');
  try {
    rememberActiveFormulaOutput();
    const id = $('formula-id').value.trim(); const name = $('formula-name').value.trim();
    const inputNames = parseEditorInputs();
    const inputs = Object.fromEntries(inputNames.map((input) => [input, {...cloneSchema(state.formulaEditor.inputSchemas[input]), description: state.formulaEditor.inputSchemas[input]?.description || ''}]));
    const parameters = parseEditorParameters();
    const outputEntries = formulaOutputEntries();
    if (!outputEntries.length || outputEntries.some(([, root]) => !root || expressionContainsHole(root))) {
      throw new Error('请先填满所有 □ 并为每个输出创建表达式');
    }
    const outputExpressions = Object.fromEntries(outputEntries);
    const serialized = expressionToFormulaStepsForOutputs(outputExpressions, outputEntries.map(([outputName]) => outputName));
    if (!serialized.steps.length || outputEntries.some(([outputName]) => !serialized.sources[outputName])) {
      throw new Error('每个输出都必须连接到一个运算结果');
    }
    state.formulaEditor.steps = serialized.steps;
    const outputs = Object.fromEntries(outputEntries.map(([outputName]) => [
      outputName,
      {...cloneSchema(state.formulaEditor.outputSchemas[outputName]), source: serialized.sources[outputName]},
    ]));
    const formula = { id, name, description: $('formula-description').value.trim(), inputs, parameters, steps: serialized.steps, outputs, metadata: { origin: 'scratch-web' } };
    const result = await api('/api/formulas', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ formula, replace: Boolean(state.formulaEditor.editingId) }) });
    await refreshBlocksAfterFormulaChange();
    $('formula-editor-dialog').close();
    state.formulaEditor.editingId = null;
    showMessage(`公式已保存：${result.formula.name}（${result.formula.id}）`, 'ok');
  } catch (error) { errorTarget.textContent = error.message; }
}

async function refreshBlocksAfterFormulaChange() {
  [state.blocks, state.formulas] = await Promise.all([api('/api/blocks'), api('/api/formulas')]);
  draw();
}

async function showMyFormulas() {
  try {
    const formulas = await api('/api/formulas');
    [state.blocks, state.formulas] = await Promise.all([api('/api/blocks'), api('/api/formulas')]);
    renderPalette();
    const list = $('formula-list'); list.innerHTML = '';
    formulas.forEach((item) => {
      const builtin = item.id.startsWith('builtin/');
      const row = document.createElement('div'); row.className = `formula-list-item${builtin ? ' builtin-formula-item' : ''}`;
      const text = document.createElement('span'); text.innerHTML = `<strong>${item.name}</strong><small>${builtin ? '内置模板' : '我的公式'} · ${item.id} · ${(item.formula_hash || '').slice(0, 12)}</small>`;
      const actions = document.createElement('span');
      if (builtin) {
        const useButton = document.createElement('button'); useButton.textContent = '使用此模板'; useButton.title = '复制为 user/ 公式后打开编辑器'; useButton.onclick = () => openBuiltinFormulaCopy(item, formulas);
        actions.appendChild(useButton);
      } else {
        const editButton = document.createElement('button'); editButton.textContent = '编辑'; editButton.onclick = () => loadFormulaIntoEditor(item);
        const copyButton = document.createElement('button'); copyButton.textContent = '复制'; copyButton.onclick = async () => { const copy = {...item, id: item.id + '_copy', name: `${item.name} Copy`}; delete copy.formula_hash; delete copy.block_id; await api('/api/formulas', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({formula: copy}) }); await refreshBlocksAfterFormulaChange(); showMyFormulas(); };
        const renameButton = document.createElement('button'); renameButton.textContent = '重命名'; renameButton.onclick = async () => { const next = window.prompt('新的 Formula ID（user/xxx）', item.id); if (!next || next === item.id) return; await api('/api/formula/rename', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({old_id: item.id, new_id: next}) }); await refreshBlocksAfterFormulaChange(); showMyFormulas(); };
        const exportButton = document.createElement('button'); exportButton.textContent = '导出'; exportButton.onclick = async () => { const value = await api('/api/formula/' + encodeURIComponent(item.id) + '/export'); const blob = new Blob([value], {type: 'text/yaml'}); const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = item.id.split('/').pop() + '.yaml'; link.click(); URL.revokeObjectURL(link.href); };
        const deleteButton = document.createElement('button'); deleteButton.textContent = '删除'; deleteButton.onclick = async () => { if (!window.confirm(`删除 ${item.name}？`)) return; await api('/api/formula/delete', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({id: item.id}) }); await refreshBlocksAfterFormulaChange(); showMyFormulas(); };
        actions.append(editButton, copyButton, renameButton, exportButton, deleteButton);
      }
      row.append(text, actions); list.appendChild(row);
    });
    if (!list.children.length) list.textContent = '暂无公式模板；点击“新建公式”创建。';
    $('formula-list-dialog').showModal();
  } catch (error) { showError(error); }
}

function makeRecipeStep(block, params = {}, steps = null) {
  const step = newStep(block);
  step.params = {...params};
  if (Array.isArray(steps)) step.steps = steps;
  return step;
}

function canonicalDataSteps() {
  return [
    makeRecipeStep('set_seed', {seed: 1}),
    makeRecipeStep('select_device', {device: 'auto', save_as: 'device'}),
    makeRecipeStep('load_dataset', {dataset: 'synthetic', options: {samples: 64, features: 4, classes: 2}, save_as: 'data_plan'}),
    makeRecipeStep('inspect_dataset_semantics', {data_plan: 'data_plan', save_as: 'dataset_semantics'}),
    makeRecipeStep('create_dataset_split', {data_plan: 'data_plan', validation_size: 8, split_strategy: 'random', split_seed: 1}),
    makeRecipeStep('select_label_source', {data_plan: 'data_plan', train: 'observed', validation: 'clean', test: 'clean'}),
    makeRecipeStep('apply_noise', {data_plan: 'data_plan', name: 'none', rate: 0, seed: 1}),
    makeRecipeStep('build_noise_manifest', {data_plan: 'data_plan', required: false}),
    makeRecipeStep('configure_preprocessing', {data_plan: 'data_plan', preprocessing: 'tensor_only', augment: false}),
    makeRecipeStep('configure_views', {data_plan: 'data_plan', views: ['weak']}),
    makeRecipeStep('assign_data_roles', {data_plan: 'data_plan', roles: ['train', 'train_eval', 'validation', 'test']}),
    makeRecipeStep('configure_loader', {data_plan: 'data_plan', batch_size: 16, num_workers: 0, pin_memory: false}),
    makeRecipeStep('build_prepared_data', {data_plan: 'data_plan', save_as: 'prepared_data'}),
    makeRecipeStep('build_loaders', {role_datasets: 'role_datasets', loader_spec: 'loader_spec'}),
  ];
}

function singleSkeletonRecipe() {
  const batchSteps = [
    makeRecipeStep('get_batch', {batch: 'batch', input_as: 'images', label_as: 'labels', index_as: 'indices'}),
    makeRecipeStep('move_batch_to_device', {input: 'images', labels: 'labels', device: 'device'}),
    makeRecipeStep('zero_grad', {optimizer: 'optimizer'}),
    makeRecipeStep('forward', {model: 'model', input: 'images', save_as: 'logits'}),
    makeRecipeStep('per_sample_ce', {logits: 'logits', labels: 'labels', save_as: 'loss_per_sample'}),
    makeRecipeStep('mean_loss', {input: 'loss_per_sample', save_as: 'loss'}),
    makeRecipeStep('backward', {loss: 'loss', model: 'model'}),
    makeRecipeStep('optimizer_step', {optimizer: 'optimizer'}),
  ];
  const epochSteps = [
    makeRecipeStep('batch_loop', {loader: 'train_loader', global_step_as: 'global_step'}, batchSteps),
    makeRecipeStep('evaluate_accuracy', {model: 'model', loader: 'validation_loader', save_as: 'validation_accuracy', final: false, target_source: 'observed'}),
    makeRecipeStep('track_best_model', {model: 'model', metric: 'validation_accuracy', metric_name: 'validation_accuracy', save_as: 'best_model_state'}),
  ];
  return {
    schema_version: 1, name: '新建单模型算法', description: 'Scratch 标准单模型骨架',
    settings: {}, steps: [...canonicalDataSteps(), makeRecipeStep('create_model', {model: 'mlp', num_classes: 2, input_dim: 4, hidden: 32, device: 'device', save_as: 'model'}), makeRecipeStep('create_optimizer', {optimizer: 'sgd', model: 'model', lr: 0.01, momentum: 0.9, save_as: 'optimizer'}), makeRecipeStep('epoch_loop', {epochs: 3, start_epoch: 0}, epochSteps), makeRecipeStep('restore_best_model', {model: 'model', state: 'best_model_state'}), makeRecipeStep('evaluate_accuracy', {model: 'model', loader: 'test_loader', save_as: 'test_accuracy', final: true, target_source: 'observed'}), makeRecipeStep('record_metrics', {values: ['test_accuracy']})],
  };
}

function dualSkeletonRecipe() {
  const dualBatch = [
    makeRecipeStep('get_batch', {batch: 'batch', input_as: 'images', label_as: 'labels', index_as: 'indices'}),
    makeRecipeStep('move_batch_to_device', {input: 'images', labels: 'labels', device: 'device'}),
    makeRecipeStep('forward', {model: 'model_a', input: 'images', save_as: 'logits_a'}),
    makeRecipeStep('forward', {model: 'model_b', input: 'images', save_as: 'logits_b'}),
    makeRecipeStep('per_sample_ce', {logits: 'logits_a', labels: 'labels', save_as: 'loss_a_per_sample'}),
    makeRecipeStep('mean_loss', {input: 'loss_a_per_sample', save_as: 'loss_a'}),
    makeRecipeStep('backward', {loss: 'loss_a', model: 'model_a'}),
    makeRecipeStep('optimizer_step', {optimizer: 'optimizer_a'}),
    makeRecipeStep('per_sample_ce', {logits: 'logits_b', labels: 'labels', save_as: 'loss_b_per_sample'}),
    makeRecipeStep('mean_loss', {input: 'loss_b_per_sample', save_as: 'loss_b'}),
    makeRecipeStep('backward', {loss: 'loss_b', model: 'model_b'}),
    makeRecipeStep('optimizer_step', {optimizer: 'optimizer_b'}),
  ];
  const epochSteps = [makeRecipeStep('batch_loop', {loader: 'train_loader', global_step_as: 'global_step'}, dualBatch), makeRecipeStep('evaluate_accuracy', {model: 'model_a', loader: 'validation_loader', save_as: 'validation_accuracy', target_source: 'observed'})];
  return {
    schema_version: 1, name: '新建双模型算法', description: 'Scratch 双模型训练骨架', settings: {},
    steps: [...canonicalDataSteps(), makeRecipeStep('create_model', {model: 'mlp', num_classes: 2, input_dim: 4, hidden: 32, device: 'device', save_as: 'model_a'}), makeRecipeStep('create_model', {model: 'mlp', num_classes: 2, input_dim: 4, hidden: 32, device: 'device', save_as: 'model_b'}), makeRecipeStep('create_optimizer', {optimizer: 'sgd', model: 'model_a', lr: 0.01, save_as: 'optimizer_a'}), makeRecipeStep('create_optimizer', {optimizer: 'sgd', model: 'model_b', lr: 0.01, save_as: 'optimizer_b'}), makeRecipeStep('epoch_loop', {epochs: 3, start_epoch: 0}, epochSteps), makeRecipeStep('evaluate_accuracy', {model: 'model_a', loader: 'test_loader', save_as: 'test_accuracy', final: true, target_source: 'observed'}), makeRecipeStep('record_metrics', {values: ['test_accuracy']})],
  };
}

function blankRecipe() { return {schema_version: 1, name: '空白 Scratch 算法', description: '', settings: {}, steps: []}; }

function applySkeleton(kind) {
  resetRunTracking();
  state.recipe = kind === 'single' ? singleSkeletonRecipe() : kind === 'dual' ? dualSkeletonRecipe() : blankRecipe();
  ensureUiIds(state.recipe.steps); state.selected = null; state.paletteSelection = null; state.dataOverviewSelection = null; state.dataBlockMode = 'concepts'; state.deletedStep = null; state.lastRun = null; state.validated = false; state.errorStepId = null; state.errorMessage = ''; state.errorPayload = null; state.errorGuide = null; state.activeInsertionTarget = defaultInsertionTarget(); state.compositeExpanded.clear(); state.dataExecutionView.clear(); state.compositeUngrouped.clear();
  const menu = $('new-menu'); if (menu) menu.hidden = true; const newButton = $('new'); if (newButton) newButton.setAttribute('aria-expanded', 'false');
  draw();
}

function renderRecipeList(names) {
  const list = $('recipe-list'); if (!list) return; list.innerHTML = '';
  if (!names.length) { list.textContent = '暂无已保存 Recipe'; return; }
  names.forEach((name) => { const row = document.createElement('button'); row.type = 'button'; row.className = 'recipe-list-row'; row.textContent = name; row.onclick = async () => { try { resetRunTracking(); state.recipe = await api('/api/recipe/' + encodeURIComponent(name)); ensureUiIds(state.recipe.steps); state.selected = null; state.paletteSelection = null; state.dataOverviewSelection = null; state.dataBlockMode = 'concepts'; state.deletedStep = null; state.validated = false; state.lastRun = null; state.errorMessage = ''; state.errorPayload = null; state.errorGuide = null; state.errorStepId = null; state.dataExecutionView.clear(); $('recipe-dialog').close(); draw(); } catch (error) { showError(error); } }; list.appendChild(row); });
}

async function loadDefaultRecipe() {
  resetRunTracking();
  state.recipe = await api('/api/default-recipe');
  ensureUiIds(state.recipe.steps);
  state.selected = null;
  state.dataOverviewSelection = null;
  state.dataBlockMode = 'concepts';
  state.paletteSelection = null;
  state.deletedStep = null;
  state.lastRun = null;
  state.validated = false;
  state.errorMessage = '';
  state.errorPayload = null;
  state.errorGuide = null;
  state.dataExecutionView.clear();
  state.errorStepId = null;
  state.activeInsertionTarget = defaultInsertionTarget();
  draw();
}

function payload() {
  state.recipe.name = $('recipe-name').value.trim() || 'scratch_recipe';
  return { recipe: stripUiFields(state.recipe) };
}

function runPayload() {
  const runtime_limits = state.runMode === 'full' ? {} : {...state.runtimeLimits};
  return { ...payload(), runtime_limits };
}

function highlightError(message, payload = null) {
  const indexedPath = Array.isArray(payload?.path) ? payload.path.filter((item) => Number.isInteger(item)).map(Number) : [];
  const explicitBlock = payload?.block_id ? flatRecipeSteps().find((step) => step.block === payload.block_id) : null;
  if (explicitBlock && !indexedPath.length) {
    state.selected = explicitBlock;
    state.errorStepId = explicitBlock._uiId;
    draw();
    document.querySelector(`[data-ui-id="${explicitBlock._uiId}"]`)?.scrollIntoView({block: 'center'});
    return;
  }
  const path = indexedPath.length ? indexedPath : [...String(message).matchAll(/(?:step|steps\[)\s*(\d+)/g)].map((match) => Number(match[1]) - 1);
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
  } else {
    state.errorStepId = flatRecipeSteps()[0]?._uiId || null;
    draw();
  }
}
function showError(error) {
  state.errorMessage = error.message;
  state.errorPayload = error.payload || null;
  state.errorGuide = guideForError(error);
  highlightError(error.message, error.payload);
  showMessage(error.message, 'error');
  renderGuidance();
}

function renderTemplateList(templates, query = '') {
  const list = $('template-list');
  list.innerHTML = '';
  const needle = query.trim().toLowerCase();
  templates.filter((item) => !needle || `${item.name} ${item.id} ${item.status}`.toLowerCase().includes(needle)).forEach((item) => {
    const card = document.createElement('div'); card.className = 'template-card';
    makeTemplateCardActivatable(card, item);
    const name = document.createElement('strong'); name.textContent = item.name; card.appendChild(name);
    const status = document.createElement('small');
    status.textContent = templateStatusLabel(item.status);
    card.appendChild(status);
    const open = document.createElement('button'); open.textContent = '打开';
    open.onclick = (event) => { event.stopPropagation(); openTemplate(item); };
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
    makeTemplateCardActivatable(card, item);
    const name = document.createElement('strong'); name.textContent = `${item.name} 论文范例`; card.appendChild(name);
    const status = document.createElement('small'); status.textContent = `${templateStatusLabel(item.status)}，可直接查看和运行`; card.appendChild(status);
    const open = document.createElement('button'); open.textContent = '打开范例'; open.onclick = (event) => { event.stopPropagation(); openTemplate(item); };
    card.appendChild(open); list.appendChild(card);
  });
}

function makeTemplateCardActivatable(card, item) {
  // The whole paper row is an open target; the explicit button remains for
  // discoverability and keyboard users. This avoids requiring a precise click
  // on a small button in the New → paper template flow.
  card.tabIndex = 0;
  card.setAttribute('role', 'button');
  card.onclick = (event) => {
    if (!event.target.closest('button')) openTemplate(item);
  };
  card.onkeydown = (event) => {
    if ((event.key === 'Enter' || event.key === ' ') && event.target === card) {
      event.preventDefault();
      openTemplate(item);
    }
  };
}

async function openTemplate(item) {
  try {
    resetRunTracking();
    state.recipe = await api('/api/recipe/' + encodeURIComponent(item.path));
    ensureUiIds(state.recipe.steps);
    state.selected = null;
    state.paletteSelection = null;
    state.dataOverviewSelection = null;
    state.dataBlockMode = 'concepts';
    state.deletedStep = null;
    state.lastRun = null;
    state.validated = false;
    state.errorMessage = '';
    state.errorPayload = null;
    state.errorGuide = null;
    state.dataExecutionView.clear();
    state.errorStepId = null;
    state.activeInsertionTarget = defaultInsertionTarget();
    $('template-dialog').close();
    draw();
  } catch (error) { showError(error); }
}

function openOnboarding() {
  const dialog = $('onboarding-dialog');
  if (dialog && typeof dialog.showModal === 'function') dialog.showModal();
}

if ($('help')) $('help').onclick = openOnboarding;
if ($('tutorial-done')) $('tutorial-done').onclick = () => $('onboarding-dialog').close();

if ($('new')) $('new').onclick = (event) => {
  event.stopPropagation();
  const menu = $('new-menu'); if (!menu) return;
  menu.hidden = !menu.hidden; $('new').setAttribute('aria-expanded', String(!menu.hidden));
};
document.addEventListener('click', (event) => { if (!event.target.closest('.new-menu-wrap')) { const menu = $('new-menu'); if (menu) menu.hidden = true; if ($('new')) $('new').setAttribute('aria-expanded', 'false'); } });
if ($('new-single')) $('new-single').onclick = () => applySkeleton('single');
if ($('new-dual')) $('new-dual').onclick = () => applySkeleton('dual');
if ($('new-blank')) $('new-blank').onclick = () => applySkeleton('blank');
if ($('new-template')) $('new-template').onclick = openTemplateDialog;
if ($('empty-single')) $('empty-single').onclick = () => applySkeleton('single');
if ($('empty-dual')) $('empty-dual').onclick = () => applySkeleton('dual');
if ($('empty-blank')) $('empty-blank').onclick = () => applySkeleton('blank');
if ($('empty-template')) $('empty-template').onclick = openTemplateDialog;
document.querySelectorAll('[data-inspector-tab]').forEach((button) => { button.onclick = () => setInspectorTab(button.dataset.inspectorTab); });

async function validateCurrentRecipe() {
  await api('/api/validate', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload()) });
  state.validated = true; updateRunState();
  const readiness = datasetReadiness();
  if (!readiness.ok) {
    state.validated = false;
    updateRunState();
    const error = new Error(readiness.message);
    error.guidanceCode = readiness.code;
    error.payload = {ok: false, error: readiness.message, code: readiness.code, block_id: readiness.step?.block || 'load_dataset'};
    throw error;
  }
  state.errorStepId = null;
  state.errorMessage = '';
  state.errorPayload = null;
  state.errorGuide = null;
  clearErrorPresentation();
  return true;
}

if ($('check')) $('check').onclick = async () => {
  try {
    await validateCurrentRecipe(); showMessage('检查通过，已允许运行', 'ok');
  } catch (error) {
    state.validated = false;
    updateRunState();
    showError(error);
  }
};
if ($('save')) $('save').onclick = async () => { try { const result = await api('/api/save', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload()) }); showMessage(`已保存: ${result.path}`); } catch (error) { showError(error); } };
if ($('run-mode')) $('run-mode').onchange = (event) => {
  state.runMode = event.target.value === 'full' ? 'full' : 'check';
  renderRuntimeLimits();
};
if ($('stop-run')) $('stop-run').onclick = stopRun;
if ($('refresh-run')) $('refresh-run').onclick = () => state.jobId && pollRunJob(state.jobId);
async function startRun(mode = 'check') {
  if (state.running) return;
  state.runMode = mode === 'full' ? 'full' : 'check';
  renderRuntimeLimits();
  resetRunTracking();
  state.running = true; updateRunState(); setInspectorTab('run'); setResultState('运行中…');
  $('result-status')?.classList.add('running');
  if ($('result-summary')) $('result-summary').textContent = '正在检查数据集、连接和参数，检查通过后会启动后台运行任务。';
  renderRunProgress({status: 'starting', progress: {state: 'starting'}});
  try {
    await validateCurrentRecipe();
    const result = await api('/api/run', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(runPayload()) });
    if (result?.id) {
      state.jobId = result.id;
      renderRunJob(result);
      await pollRunJob(result.id);
    } else {
      state.running = false;
      renderRunResult(result);
    }
  } catch (error) {
    state.lastRun = null; state.running = false; clearRunPolling(); updateRunState(); showError(error);
  }
}
if ($('run')) $('run').onclick = () => startRun('check');
if ($('run-full')) $('run-full').onclick = () => startRun('full');
if ($('open')) $('open').onclick = async () => { try { const names = await api('/api/recipes'); renderRecipeList(names); $('recipe-dialog').showModal(); } catch (error) { showError(error); } };
async function openTemplateDialog() {
  try {
    const menu = $('new-menu');
    if (menu) menu.hidden = true;
    if ($('new')) $('new').setAttribute('aria-expanded', 'false');
    const [templates, examples] = await Promise.all([api('/api/templates'), api('/api/examples')]);
    renderFeaturedExamples(examples);
    renderTemplateList(templates);
    $('template-search').value = '';
    $('template-search').oninput = () => renderTemplateList(templates, $('template-search').value);
    $('template-dialog').showModal();
  } catch (error) { showError(error); }
}
if ($('templates')) $('templates').onclick = openTemplateDialog;
if ($('formula-builder')) $('formula-builder').onclick = () => {
  renderFormulaBuilderFields();
  $('formula-dialog').showModal();
};
if ($('formula-operation')) $('formula-operation').onchange = renderFormulaBuilderFields;
if ($('formula-add')) $('formula-add').onclick = addFormulaFromDialog;
if ($('formula-cancel')) $('formula-cancel').onclick = () => $('formula-dialog').close();
if ($('new-formula')) $('new-formula').onclick = () => { resetFormulaEditor(); $('formula-editor-dialog').showModal(); };
if ($('formula-editor-block')) $('formula-editor-block').onchange = renderFormulaEditor;
if ($('formula-inputs')) $('formula-inputs').oninput = () => { renderFormulaInputFields(); renderFormulaEditorBindingFields(); renderFormulaCanvas(); };
if ($('formula-parameters')) $('formula-parameters').oninput = () => { renderFormulaParameterFields(); renderFormulaEditorBindingFields(); renderFormulaCanvas(); };
if ($('formula-output-source')) $('formula-output-source').onchange = renderFormulaEditor;
if ($('formula-output-name')) $('formula-output-name').oninput = renderFormulaCanvas;
if ($('formula-add-input')) $('formula-add-input').onclick = addFormulaEditorInput;
if ($('formula-add-parameter')) $('formula-add-parameter').onclick = addFormulaEditorParameter;
if ($('formula-add-output')) $('formula-add-output').onclick = addFormulaEditorOutput;
if ($('formula-apply-equation')) $('formula-apply-equation').onclick = addFormulaEquation;
if ($('formula-editor-add-step')) $('formula-editor-add-step').onclick = addFormulaEditorStep;
if ($('formula-editor-save')) $('formula-editor-save').onclick = saveFormulaEditor;
if ($('formula-editor-cancel')) $('formula-editor-cancel').onclick = () => $('formula-editor-dialog').close();
if ($('my-formulas')) $('my-formulas').onclick = showMyFormulas;
if ($('recipe-name')) $('recipe-name').oninput = markDirty;
if ($('palette-search')) $('palette-search').oninput = (event) => {
  state.paletteQuery = event.target.value;
  renderPalette();
};

document.addEventListener('keydown', (event) => {
  if (event.key !== 'Backspace') return;
  const dialog = $('formula-editor-dialog');
  if (!dialog?.open) return;
  const target = event.target;
  if (target?.closest?.('input, textarea, select, button, a, [contenteditable="true"]')) return;
  if (state.formulaEditor.expressionSelection && deleteSelectedFormulaExpression()) event.preventDefault();
});

Promise.all([api('/api/blocks'), api('/api/formulas')]).then(([blocks, formulas]) => {
  state.blocks = blocks;
  state.formulas = formulas;
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
