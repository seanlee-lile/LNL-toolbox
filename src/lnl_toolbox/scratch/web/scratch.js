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
  errorPayload: null,
  errorGuide: null,
  lastRun: null,
  runtimeLimits: { max_epochs: 1, max_batches: 1, skip_final_test: true },
  paletteQuery: '',
  paletteCategory: '数据',
  paletteCollapsed: new Set(),
  compositeExpanded: new Set(),
  compositeUngrouped: new Set(),
  inspectorTab: 'blocks',
  running: false,
  jobId: null,
  runPollTimer: null,
  runProgress: null,
  runStopping: false,
  formulaEditor: { steps: [], editingId: null },
};
const $ = (id) => document.getElementById(id);
const apiBase = location.pathname.startsWith('/scratch') ? '/api/scratch' : '/api';
let uiIdCounter = 0;
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

const UI_CATEGORIES = [
  { name: '全部', icon: '▤', color: '#84cc16' },
  { name: '数据', icon: '▦', color: '#38bdf8' },
  { name: '模型', icon: '◈', color: '#a78bfa' },
  { name: '训练', icon: '↻', color: '#f59e0b' },
  { name: '公式', icon: 'ƒ', color: '#fb7185' },
  { name: '选择', icon: '⌁', color: '#34d399' },
  { name: '状态', icon: '▣', color: '#818cf8' },
  { name: '统计', icon: '∑', color: '#2dd4bf' },
  { name: '更新', icon: '↟', color: '#facc15' },
  { name: '评估', icon: '◒', color: '#c4b5fd' },
  { name: '我的', icon: '♡', color: '#f472b6' },
];

function uiCategory(info) {
  if (isUserOwnedInfo(info)) return '我的';
  const raw = `${info?.ui_group || ''} ${info?.category || ''} ${info?.stage || ''}`.toLowerCase();
  if (info?.id?.startsWith('formula__') || /formula|tensor|loss|gather|softmax|log|power|mean|cross/.test(raw)) return '公式';
  if (/data|dataset|loader|noise|view|role|split|manifest|数据/.test(raw)) return '数据';
  if (/model|network|backbone|模型/.test(raw)) return '模型';
  if (/selection|select|mask|weight|sample|选择|筛/.test(raw)) return '选择';
  if (/state|history|ema|accumulator|memory|状态|历史/.test(raw)) return '状态';
  if (/statistic|posterior|transition|graph|snapshot|统计|后验|转移/.test(raw)) return '统计';
  if (/meta|parameter|gradient|optimizer|update|更新|参数/.test(raw)) return '更新';
  if (/evaluation|evaluate|metric|评估/.test(raw)) return '评估';
  if (/scheduler|epoch|batch|training|control|runtime|train|loop|训练/.test(raw)) return '训练';
  if (String(info?.id || '').startsWith('user/')) return '我的';
  return '训练';
}

function isUserOwnedInfo(info) {
  if (!info) return false;
  const origin = String(info.origin || info.metadata?.origin || '').toLowerCase();
  const id = String(info.id || '').toLowerCase();
  return origin === 'user' || origin.includes('scratch-web') || id.startsWith('user/') || id.includes('__user__');
}

function categoryInfo(name) { return UI_CATEGORIES.find((item) => item.name === name) || UI_CATEGORIES.find((item) => item.name === '训练') || UI_CATEGORIES[0]; }

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
    if (job.running) state.runPollTimer = window.setTimeout(() => pollRunJob(jobId), 400);
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
    if (job.running) state.runPollTimer = window.setTimeout(() => pollRunJob(state.jobId), 250);
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
  state.errorGuide = null;
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
  state.validated = false;
  state.lastRun = null;
  state.errorStepId = null;
  state.errorMessage = '';
  state.errorPayload = null;
  state.errorGuide = null;
  updateRunState();
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
    run.textContent = state.running ? '运行中…' : '▶ 运行';
  }
  const stop = $('stop-run');
  if (stop) {
    stop.disabled = !state.running || !state.jobId || state.runStopping;
    stop.textContent = state.runStopping ? '正在停止…' : '■ 停止运行';
  }
  const refresh = $('refresh-run');
  if (refresh) refresh.disabled = !state.jobId;
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
  zone.textContent = dropZoneLabel(parentId, index, context);
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
  const categoryMatches = (item) => state.paletteCategory === '全部'
    || (state.paletteCategory === '我的' ? isUserOwnedInfo(item) : uiCategory(item) === state.paletteCategory);
  const visible = state.blocks.filter((item) => item.beginner_visible && (!query || [
    item.name, item.id, item.category, item.ui_group, item.description,
    ...(item.requires || []), ...(item.provides || []),
  ].join(' ').toLowerCase().includes(query)) && categoryMatches(item));
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
  if (count) count.textContent = `${visible.length} 个`;

  if (state.paletteCategory === '我的') {
    const actions = document.createElement('div'); actions.className = 'my-palette-actions';
    [['新建公式', '创建一个 Formula-safe 用户公式', () => { resetFormulaEditor(); $('formula-editor-dialog').showModal(); }],
      ['我的公式', '编辑或导出已保存公式', showMyFormulas],
      ['我的组合块', '组合块只在界面中折叠，不会新增运行时 Block', () => showMessage('组合块是 UI 视图；展开或解除组合不会改变 Recipe。')]].forEach(([name, description, action]) => {
      const card = document.createElement('button'); card.type = 'button'; card.className = 'my-palette-card';
      card.innerHTML = `<strong>${name}</strong><small>${description}</small>`; card.onclick = action; actions.appendChild(card);
    });
    palette.appendChild(actions);
  }
  if (!visible.length) {
    palette.appendChild(Object.assign(document.createElement('div'), {className: 'palette-empty', textContent: query ? '没有匹配的积木' : state.paletteCategory === '我的' ? '暂无用户公式；从上方创建一个。' : '该分类暂无可见积木'}));
    return;
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
      const node = document.createElement('div');
      node.className = 'palette-block';
      node.dataset.category = blockCategory(item);
      node.dataset.uiCategory = uiCategory(item);
      node.dataset.blockId = item.id;
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
  in: '∈', le: '≤', ge: '≥', neq: '≠', times: '×', cdot: '·',
  pm: '±', rightarrow: '→', leftarrow: '←', infty: '∞', sum: 'Σ', prod: 'Π',
};
const MATH_OPERATOR_CHARS = new Set(['=', '+', '-', '*', '/', '·', '×', '±', '⊙',
  '(', ')', '[', ']', '{', '}', ',', ':', ';', '|', '‖', '→', '←', '≤', '≥',
  '≠', '<', '>', '∈', 'Σ', 'Π', '∑', '∏']);
const MATH_GREEK = new Set('αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ'.split(''));
const MATH_WORD_SYMBOLS = {sum: 'Σ', prod: 'Π'};

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
    .replace(/\\left|\\right/g, '')
    .replace(/\\[,;!]/g, ' ')
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
      if (command === 'text' || command === 'mathrm') {
        const content = parseMathAtom(source, next);
        const text = content.node?.textContent || '';
        return {node: mathElement('mtext', text), next: content.next};
      }
      if (['tilde', 'hat', 'bar', 'vec'].includes(command)) {
        const content = parseMathAtom(source, next);
        const accent = {tilde: '˜', hat: 'ˆ', bar: '¯', vec: '→'}[command];
        const mover = mathElement('mover'); mover.append(content.node || mathElement('mi', '□'), mathElement('mo', accent));
        mover.setAttribute('accent', 'true');
        return {node: mover, next: content.next};
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
  const source = String(formula || '').trim();
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
  sourceNode.className = 'math-source'; sourceNode.textContent = source; sourceNode.title = '原始公式记号';
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
  { id: 'prepare-data', label: '准备数据', icon: '▦', blocks: ['load_dataset', 'inspect_dataset_semantics', 'create_dataset_split', 'select_label_source', 'apply_noise', 'build_noise_manifest', 'configure_preprocessing', 'configure_views', 'assign_data_roles', 'configure_loader', 'build_prepared_data', 'build_loaders'], description: '统一的数据源、划分、噪声、视图、角色和 Loader 链。' },
  { id: 'model-optimization', label: '模型与优化', icon: '◈', blocks: ['create_model', 'create_optimizer'], description: '创建模型并连接其优化器。' },
  { id: 'training-loop', label: '训练循环', icon: '↻', blocks: ['epoch_loop'], description: 'Epoch / Batch 的 C 形训练容器。' },
  { id: 'prepare-batch', label: '准备 Batch', icon: '▤', blocks: ['get_batch', 'move_batch_to_device'], description: '读取 Batch 并移动到当前设备。' },
  { id: 'update-model', label: '更新模型', icon: '↟', blocks: ['backward', 'optimizer_step'], description: '反向传播并执行优化器更新。' },
  { id: 'validate-best', label: '验证并保留最佳', icon: '✓', blocks: ['evaluate_accuracy', 'track_best_model'], description: '评估当前模型并按指标保留最佳状态。' },
  { id: 'final-evaluation', label: '最终评估', icon: '◒', blocks: ['restore_best_model', 'evaluate_accuracy', 'record_metrics'], description: '恢复最佳状态并记录最终测试指标。', optionalPrefix: true },
];

function detectCompositeRanges(steps) {
  const ranges = [];
  let index = 0;
  while (index < (steps || []).length) {
    let match = null;
    for (const definition of COMPOSITE_DEFINITIONS) {
      const ids = definition.blocks;
      const exact = ids.every((id, offset) => steps[index + offset]?.block === id);
      const optional = definition.optionalPrefix && steps[index]?.block === 'evaluate_accuracy'
        && Boolean(steps[index]?.params?.final) && steps[index + 1]?.block === 'record_metrics';
      if (exact || optional) {
        match = { definition, start: index, end: index + (optional ? 2 : ids.length) };
        break;
      }
    }
    if (match) { ranges.push(match); index = match.end; } else index += 1;
  }
  return ranges;
}

function compositeKey(parentId, range) {
  return `${parentId}:${range.definition.id}:${range.start}:${range.end}:${range.steps?.[range.start]?._uiId || ''}`;
}

function renderStepNode(step, index, steps, parent, parentId, context) {
  const info = blockInfo(step.block);
  const node = document.createElement('div');
  const loopClass = info && info.kind !== 'action' ? `loop-container loop-block ${step.block === 'epoch_loop' ? 'epoch-loop' : step.block === 'batch_loop' ? 'batch-loop' : ''}` : '';
  const formulaClass = info?.formula ? 'formula-block' : '';
  node.className = `step ${loopClass} ${formulaClass}${state.selected === step ? ' selected' : ''}${state.errorStepId === step._uiId ? ' error-step' : ''}`;
  node.dataset.category = blockCategory(info);
  node.dataset.uiCategory = uiCategory(info);
  node.dataset.uiId = step._uiId;
  const header = document.createElement('div');
  header.className = 'step-header';
  const title = document.createElement('span'); title.textContent = info ? info.name : step.block; header.appendChild(title);
  const kind = document.createElement('span'); kind.className = 'kind'; kind.textContent = info ? info.kind : ''; header.appendChild(kind);
  const summary = renderStepSummary(step, info);
  node.appendChild(header);
  if (summary) { const summaryNode = document.createElement('div'); summaryNode.className = 'step-summary'; summaryNode.textContent = summary; node.appendChild(summaryNode); }
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
    if (event.button !== 0 || event.target.closest('button, input, select, textarea')) return;
    const startX = event.clientX; const startY = event.clientY; let moved = false;
    const onMove = (moveEvent) => {
      if (!moved && Math.hypot(moveEvent.clientX - startX, moveEvent.clientY - startY) < 5) return;
      moved = true;
      state.drag = { type: 'step', id: step._uiId, info };
      const zone = document.elementFromPoint(moveEvent.clientX, moveEvent.clientY)?.closest('.drop-target');
      document.querySelectorAll('.drop-target.drop-hover').forEach((item) => item.classList.remove('drop-hover'));
      if (zone) { const target = findDropTarget(zone); if (canInsert(info, target.parentId, step._uiId, target.index)) zone.classList.add('drop-hover'); }
      markDropZones(); moveEvent.preventDefault();
    };
    const onUp = (upEvent) => {
      document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp);
      if (!moved) { state.drag = null; return; }
      const zone = document.elementFromPoint(upEvent.clientX, upEvent.clientY)?.closest('.drop-target');
      const target = zone ? findDropTarget(zone) : null;
      const reason = target ? availabilityReason(info, target, step._uiId) : '未放在有效 drop zone';
      const success = Boolean(target && !reason && moveStepToTarget(step._uiId, target)); state.drag = null;
      if (!success) showMessage(`不能拖动：${reason || '插入失败'}`, 'error');
      draw(); upEvent.preventDefault();
    };
    document.addEventListener('mousemove', onMove); document.addEventListener('mouseup', onUp);
  };
  node.ondragstart = (event) => { state.drag = { type: 'step', id: step._uiId, info }; event.dataTransfer.setData('application/x-lnl-step', step._uiId); event.dataTransfer.effectAllowed = 'move'; markDropZones(); };
  node.ondragend = () => { state.drag = null; markDropZones(); };
  if (info && info.kind !== 'action') {
    const toggle = document.createElement('button'); toggle.textContent = state.collapsed.has(step) ? '展开' : '折叠';
    toggle.onclick = (event) => { event.stopPropagation(); if (state.collapsed.has(step)) state.collapsed.delete(step); else state.collapsed.add(step); draw(); };
    header.appendChild(toggle);
    if (!state.collapsed.has(step)) {
      const nested = document.createElement('div'); nested.className = 'nested loop-body';
      const childContext = step.block === 'epoch_loop' ? 'epoch' : step.block === 'batch_loop' ? 'batch' : 'any';
      drawSteps(step.steps || (step.steps = []), nested, step._uiId, childContext); node.appendChild(nested);
    }
  }
  const controls = document.createElement('span'); controls.className = 'step-controls';
  const copy = document.createElement('button'); copy.textContent = '复制';
  copy.onclick = (event) => { event.stopPropagation(); const clone = typeof structuredClone === 'function' ? structuredClone(step) : JSON.parse(JSON.stringify(step)); ensureUiIds([clone]); clone._uiId = makeUiId(); (steps || []).splice(index + 1, 0, clone); markDirty(); state.selected = clone; draw(); };
  const remove = document.createElement('button'); remove.textContent = '删除';
  remove.onclick = (event) => { event.stopPropagation(); (steps || []).splice(index, 1); markDirty(); if (state.selected === step) state.selected = null; state.paletteSelection = null; state.activeInsertionTarget = { parentId, index, context }; draw(); };
  controls.append(copy, remove); node.appendChild(controls);
  parent.appendChild(node);
  return node;
}

function createCompositeShell(range, steps, parentId, context, key) {
  const definition = range.definition;
  const shell = document.createElement('section'); shell.className = 'composite-block'; shell.dataset.compositeId = definition.id; shell.dataset.uiCategory = uiCategory({category: definition.id});
  const accent = {environment: '#f59e0b', 'prepare-data': '#38bdf8', 'model-optimization': '#a78bfa', 'training-loop': '#f59e0b', 'prepare-batch': '#38bdf8', 'update-model': '#facc15', 'validate-best': '#c4b5fd', 'final-evaluation': '#2dd4bf'}[definition.id] || '#38bdf8';
  shell.style.setProperty('--composite-accent', accent);
  const header = document.createElement('div'); header.className = 'composite-header';
  const title = document.createElement('div'); title.className = 'composite-title'; title.innerHTML = `<span class="composite-icon">${definition.icon}</span><strong>${definition.label}</strong><small>${range.end - range.start} 个步骤</small>`;
  const actions = document.createElement('div'); actions.className = 'composite-actions';
  const toggle = document.createElement('button'); toggle.type = 'button'; toggle.textContent = state.compositeExpanded.has(key) ? '收起' : '展开';
  toggle.onclick = (event) => { event.stopPropagation(); if (state.compositeExpanded.has(key)) state.compositeExpanded.delete(key); else state.compositeExpanded.add(key); draw(); };
  const ungroup = document.createElement('button'); ungroup.type = 'button'; ungroup.textContent = '解除组合';
  ungroup.onclick = (event) => { event.stopPropagation(); state.compositeUngrouped.add(key); draw(); };
  actions.append(toggle, ungroup); header.append(title, actions); shell.appendChild(header);
  const hint = document.createElement('p'); hint.className = 'composite-hint'; hint.textContent = definition.description; shell.appendChild(hint);
  if (state.compositeExpanded.has(key)) {
    const body = document.createElement('div'); body.className = 'composite-body';
    for (let index = range.start; index < range.end; index += 1) { body.appendChild(createDropZone(parentId, index, context)); renderStepNode(steps[index], index, steps, body, parentId, context); }
    body.appendChild(createDropZone(parentId, range.end, context)); shell.appendChild(body);
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
    renderStepNode(steps[index], index, steps, parent, parentId, context);
    index += 1;
  }
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
  renderMathFormula(formulaText, info.formula || '');
  formula.appendChild(formulaText);
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
      const label = document.createElement('label'); label.textContent = isDatasetSourceInfo(info) && name === 'dataset' ? 'dataset（必选）' : name; wrap.appendChild(label);
      const value = document.createElement('span'); value.className = 'param-preview';
      value.textContent = formatParamValue(step.params[name] ?? schema.default);
      wrap.appendChild(value);
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
  renderInspector();
  renderGuidance();
  updateRunState();
  if (!state.lastRun) showMessage(targetLabel(state.activeInsertionTarget));
  setInspectorTab(state.inspectorTab);
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
  preview.replaceChildren();
  const description = document.createElement('p'); description.className = 'formula-preview-description'; description.textContent = info.description || '';
  preview.appendChild(description);
  const formulaLabel = document.createElement('div'); formulaLabel.className = 'formula-preview-label'; formulaLabel.textContent = '数学公式'; preview.appendChild(formulaLabel);
  const formulaVisual = document.createElement('div'); renderMathFormula(formulaVisual, info.formula || '该运算没有单独公式标注，但可作为公式链的一步。'); preview.appendChild(formulaVisual);
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
    const math = document.createElement('div'); renderMathFormula(math, selectedInfo?.formula || '暂无独立公式', {compact: true}); currentPreview.appendChild(math);
  }
  const editorPalette = $('formula-editor-palette');
  if (editorPalette) {
    editorPalette.innerHTML = '';
    candidates.forEach((info) => {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'formula-palette-block';
      const name = document.createElement('strong'); name.textContent = info.name; button.appendChild(name);
      const formula = document.createElement('div'); formula.className = 'formula-palette-formula'; renderMathFormula(formula, info.formula || '无独立公式', {compact: true}); button.appendChild(formula);
      button.title = info.description || info.id;
      button.onclick = () => { select.value = info.id; renderFormulaEditor(); };
      editorPalette.appendChild(button);
    });
  }
  const list = $('formula-editor-step-list'); list.innerHTML = '';
  state.formulaEditor.steps.forEach((step, index) => {
    const row = document.createElement('div'); row.className = 'formula-step-item';
    const info = blockInfo(step.block) || {};
    const body = document.createElement('div'); body.className = 'formula-step-body';
    const label = document.createElement('span'); label.innerHTML = `${index + 1}. <code>${step.id}</code> ← ${info.name || step.block}`;
    body.appendChild(label);
    const visual = document.createElement('div'); visual.className = 'formula-step-math'; renderMathFormula(visual, info.formula || `${step.block}(${(info.requires || []).join(', ')})`, {compact: true}); body.appendChild(visual);
    const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = '删除'; remove.onclick = () => { state.formulaEditor.steps.splice(index, 1); renderFormulaEditor(); };
    row.append(body, remove); list.appendChild(row);
  });
  const output = $('formula-output-source'); output.innerHTML = '';
  state.formulaEditor.steps.forEach((step) => { const option = document.createElement('option'); option.value = step.id; option.textContent = step.id; output.appendChild(option); });
  const last = state.formulaEditor.steps.at(-1)?.id || '';
  if (last) output.value = output.value || last;
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
      const math = document.createElement('div'); renderMathFormula(math, info.formula || `${step.block}(${(info.requires || []).join(', ')})`, {compact: true}); visual.appendChild(math); preview.appendChild(visual);
      if (index < state.formulaEditor.steps.length - 1) preview.appendChild(Object.assign(document.createElement('div'), {className: 'formula-chain-arrow', textContent: '↓'}));
    });
    const expression = document.createElement('details'); expression.className = 'formula-expression-details';
    expression.appendChild(Object.assign(document.createElement('summary'), {textContent: '查看组合表达式'}));
    expression.appendChild(Object.assign(document.createElement('code'), {textContent: renderNestedFormulaPreview(state.formulaEditor.steps)}));
    preview.appendChild(expression);
  }
}

function resetFormulaEditor() {
  state.formulaEditor.steps = []; state.formulaEditor.editingId = null;
  $('formula-id').value = 'user/my_formula'; $('formula-name').value = 'My Formula'; $('formula-description').value = '';
  $('formula-inputs').value = 'logits\ntargets'; $('formula-parameters').value = 'epsilon:float=1e-8'; $('formula-output-name').value = 'loss';
  $('formula-editor-step-id').value = '';
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
  try {
    $('formula-editor-step-parameter-fields').querySelectorAll('[data-formula-step-param]').forEach((input) => {
      parameters[input.dataset.formulaStepParam] = formulaFieldValue(input, info.params[input.dataset.formulaStepParam]);
    });
  } catch (error) { $('formula-editor-validation').textContent = error.message; return; }
  state.formulaEditor.steps.push({ id, block, bindings, parameters });
  $('formula-editor-step-id').value = '';
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
    makeRecipeStep('forward', {model: 'model', input: 'images', save_as: 'logits'}),
    makeRecipeStep('per_sample_ce', {logits: 'logits', labels: 'labels', save_as: 'loss_per_sample'}),
    makeRecipeStep('mean_loss', {input: 'loss_per_sample', save_as: 'loss'}),
    makeRecipeStep('backward', {loss: 'loss', model: 'model'}),
    makeRecipeStep('optimizer_step', {optimizer: 'optimizer'}),
  ];
  const epochSteps = [
    makeRecipeStep('batch_loop', {loader: 'train_loader', max_steps: 1, global_step_as: 'global_step'}, batchSteps),
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
  const epochSteps = [makeRecipeStep('batch_loop', {loader: 'train_loader', max_steps: 1, global_step_as: 'global_step'}, dualBatch), makeRecipeStep('evaluate_accuracy', {model: 'model_a', loader: 'validation_loader', save_as: 'validation_accuracy', target_source: 'observed'})];
  return {
    schema_version: 1, name: '新建双模型算法', description: 'Scratch 双模型训练骨架', settings: {},
    steps: [...canonicalDataSteps(), makeRecipeStep('create_model', {model: 'mlp', num_classes: 2, input_dim: 4, hidden: 32, device: 'device', save_as: 'model_a'}), makeRecipeStep('create_model', {model: 'mlp', num_classes: 2, input_dim: 4, hidden: 32, device: 'device', save_as: 'model_b'}), makeRecipeStep('create_optimizer', {optimizer: 'sgd', model: 'model_a', lr: 0.01, save_as: 'optimizer_a'}), makeRecipeStep('create_optimizer', {optimizer: 'sgd', model: 'model_b', lr: 0.01, save_as: 'optimizer_b'}), makeRecipeStep('epoch_loop', {epochs: 3, start_epoch: 0}, epochSteps), makeRecipeStep('evaluate_accuracy', {model: 'model_a', loader: 'test_loader', save_as: 'test_accuracy', final: true, target_source: 'observed'}), makeRecipeStep('record_metrics', {values: ['test_accuracy']})],
  };
}

function blankRecipe() { return {schema_version: 1, name: '空白 Scratch 算法', description: '', settings: {}, steps: []}; }

function applySkeleton(kind) {
  resetRunTracking();
  state.recipe = kind === 'single' ? singleSkeletonRecipe() : kind === 'dual' ? dualSkeletonRecipe() : blankRecipe();
  ensureUiIds(state.recipe.steps); state.selected = null; state.paletteSelection = null; state.lastRun = null; state.validated = false; state.errorStepId = null; state.errorMessage = ''; state.errorPayload = null; state.errorGuide = null; state.activeInsertionTarget = defaultInsertionTarget(); state.compositeExpanded.clear(); state.compositeUngrouped.clear();
  const menu = $('new-menu'); if (menu) menu.hidden = true; const newButton = $('new'); if (newButton) newButton.setAttribute('aria-expanded', 'false');
  draw();
}

function renderRecipeList(names) {
  const list = $('recipe-list'); if (!list) return; list.innerHTML = '';
  if (!names.length) { list.textContent = '暂无已保存 Recipe'; return; }
  names.forEach((name) => { const row = document.createElement('button'); row.type = 'button'; row.className = 'recipe-list-row'; row.textContent = name; row.onclick = async () => { try { resetRunTracking(); state.recipe = await api('/api/recipe/' + encodeURIComponent(name)); ensureUiIds(state.recipe.steps); state.selected = null; state.paletteSelection = null; state.validated = false; state.lastRun = null; state.errorMessage = ''; state.errorPayload = null; state.errorGuide = null; state.errorStepId = null; $('recipe-dialog').close(); draw(); } catch (error) { showError(error); } }; list.appendChild(row); });
}

async function loadDefaultRecipe() {
  resetRunTracking();
  state.recipe = await api('/api/default-recipe');
  ensureUiIds(state.recipe.steps);
  state.selected = null;
  state.paletteSelection = null;
  state.lastRun = null;
  state.validated = false;
  state.errorMessage = '';
  state.errorPayload = null;
  state.errorGuide = null;
  state.errorStepId = null;
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
    resetRunTracking();
    state.recipe = await api('/api/recipe/' + encodeURIComponent(item.path));
    ensureUiIds(state.recipe.steps);
    state.selected = null;
    state.paletteSelection = null;
    state.lastRun = null;
    state.validated = false;
    state.errorMessage = '';
    state.errorPayload = null;
    state.errorGuide = null;
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
if ($('new-template')) $('new-template').onclick = () => $('templates').click();
if ($('empty-single')) $('empty-single').onclick = () => applySkeleton('single');
if ($('empty-dual')) $('empty-dual').onclick = () => applySkeleton('dual');
if ($('empty-blank')) $('empty-blank').onclick = () => applySkeleton('blank');
if ($('empty-template')) $('empty-template').onclick = () => $('templates').click();
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
if ($('stop-run')) $('stop-run').onclick = stopRun;
if ($('refresh-run')) $('refresh-run').onclick = () => state.jobId && pollRunJob(state.jobId);
if ($('run')) $('run').onclick = async () => {
  if (state.running) return;
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
};
if ($('open')) $('open').onclick = async () => { try { const names = await api('/api/recipes'); renderRecipeList(names); $('recipe-dialog').showModal(); } catch (error) { showError(error); } };
if ($('templates')) $('templates').onclick = async () => {
  try {
    const [templates, examples] = await Promise.all([api('/api/templates'), api('/api/examples')]);
    renderFeaturedExamples(examples);
    renderTemplateList(templates);
    $('template-search').value = '';
    $('template-search').oninput = () => renderTemplateList(templates, $('template-search').value);
    $('template-dialog').showModal();
  } catch (error) { showError(error); }
};
if ($('formula-builder')) $('formula-builder').onclick = () => {
  renderFormulaBuilderFields();
  $('formula-dialog').showModal();
};
if ($('formula-operation')) $('formula-operation').onchange = renderFormulaBuilderFields;
if ($('formula-add')) $('formula-add').onclick = addFormulaFromDialog;
if ($('formula-cancel')) $('formula-cancel').onclick = () => $('formula-dialog').close();
if ($('new-formula')) $('new-formula').onclick = () => { resetFormulaEditor(); $('formula-editor-dialog').showModal(); };
if ($('formula-editor-block')) $('formula-editor-block').onchange = renderFormulaEditor;
if ($('formula-inputs')) $('formula-inputs').oninput = renderFormulaEditorBindingFields;
if ($('formula-parameters')) $('formula-parameters').oninput = renderFormulaEditorBindingFields;
if ($('formula-output-source')) $('formula-output-source').onchange = renderFormulaEditor;
if ($('formula-editor-add-step')) $('formula-editor-add-step').onclick = addFormulaEditorStep;
if ($('formula-editor-save')) $('formula-editor-save').onclick = saveFormulaEditor;
if ($('formula-editor-cancel')) $('formula-editor-cancel').onclick = () => $('formula-editor-dialog').close();
if ($('my-formulas')) $('my-formulas').onclick = showMyFormulas;
if ($('recipe-name')) $('recipe-name').oninput = markDirty;
if ($('palette-search')) $('palette-search').oninput = (event) => {
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
