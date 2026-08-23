const state = { blocks: [], recipe: { schema_version: 1, name: 'scratch_recipe', description: '', steps: [] }, selected: null, collapsed: new WeakSet() };
const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok || body.ok === false) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function blockInfo(id) { return state.blocks.find((item) => item.id === id); }
function newStep(id) { return { block: id, params: {} }; }

function renderPalette() {
  const palette = $('palette'); palette.innerHTML = '';
  const categories = [...new Set(state.blocks.map((item) => item.category))];
  categories.forEach((category) => {
    const title = document.createElement('h3'); title.textContent = category; palette.appendChild(title);
    state.blocks.filter((item) => item.category === category).forEach((item) => {
      const node = document.createElement('div'); node.className = 'palette-block'; node.draggable = true; node.textContent = item.name;
      node.title = item.description; node.ondragstart = (event) => event.dataTransfer.setData('block-id', item.id);
      node.onclick = () => addStep(item.id); palette.appendChild(node);
    });
  });
}

function drawSteps(steps = state.recipe.steps, parent = $('steps')) {
  parent.innerHTML = '';
  const drop = document.createElement('div'); drop.className = 'drop-target';
  drop.ondragover = (event) => event.preventDefault(); drop.ondrop = (event) => { event.preventDefault(); const id = event.dataTransfer.getData('block-id'); if (id) steps.push(newStep(id)); draw(); };
  parent.appendChild(drop);
  steps.forEach((step, index) => {
    const info = blockInfo(step.block); const node = document.createElement('div'); node.className = 'step' + (state.selected === step ? ' selected' : '');
    node.draggable = true; node.innerHTML = `<span>${info ? info.name : step.block}</span><span class="kind">${info ? info.kind : ''}</span>`;
    node.onclick = () => { state.selected = step; renderInspector(); draw(); };
    node.ondragstart = (event) => event.dataTransfer.setData('move-index', String(index));
    node.ondragover = (event) => event.preventDefault(); node.ondrop = (event) => { event.preventDefault(); const from = Number(event.dataTransfer.getData('move-index')); if (!Number.isNaN(from)) { const [moved] = steps.splice(from, 1); steps.splice(index, 0, moved); draw(); } };
    if (info && info.kind !== 'action') {
      const toggle = document.createElement('button'); toggle.textContent = state.collapsed.has(step) ? '展开' : '折叠';
      toggle.onclick = (event) => { event.stopPropagation(); if (state.collapsed.has(step)) state.collapsed.delete(step); else state.collapsed.add(step); draw(); };
      node.appendChild(toggle);
      if (!state.collapsed.has(step)) { const nested = document.createElement('div'); nested.className = 'nested'; drawSteps(step.steps || (step.steps = []), nested); node.appendChild(nested); }
    }
    const copy = document.createElement('button'); copy.textContent = '复制'; copy.onclick = (event) => { event.stopPropagation(); steps.splice(index + 1, 0, structuredClone(step)); draw(); };
    const remove = document.createElement('button'); remove.textContent = '删除'; remove.onclick = (event) => { event.stopPropagation(); steps.splice(index, 1); if (state.selected === step) state.selected = null; draw(); };
    node.append(copy, remove); parent.appendChild(node);
  });
}

function renderInspector() {
  const target = $('inspector'); target.innerHTML = ''; const step = state.selected;
  if (!step) { target.textContent = '点击一个积木编辑参数'; return; }
  const info = blockInfo(step.block); if (!info) return;
  info.params && Object.entries(info.params).forEach(([name, schema]) => {
    const wrap = document.createElement('div'); wrap.className = 'param'; const label = document.createElement('label'); label.textContent = name; wrap.appendChild(label);
    const input = document.createElement('input'); input.value = step.params[name] ?? schema.default ?? '';
    input.type = schema.type === 'int' || schema.type === 'float' ? 'number' : schema.type === 'bool' ? 'checkbox' : 'text';
    if (input.type === 'checkbox') input.checked = step.params[name] ?? schema.default ?? false;
    input.onchange = () => { step.params[name] = input.type === 'checkbox' ? input.checked : (schema.type === 'int' ? Number.parseInt(input.value, 10) : schema.type === 'float' ? Number.parseFloat(input.value) : input.value); };
    wrap.appendChild(input); target.appendChild(wrap);
  });
}

function draw() { $('recipe-name').value = state.recipe.name; drawSteps(); renderInspector(); }
function addStep(id) { const step = newStep(id); state.recipe.steps.push(step); state.selected = step; draw(); }
function payload() { state.recipe.name = $('recipe-name').value.trim() || 'scratch_recipe'; return { recipe: state.recipe }; }
function highlightError(message) {
  const path = [...String(message).matchAll(/step (\d+)/g)].map((match) => Number(match[1]) - 1);
  let steps = state.recipe.steps; let target = null;
  path.forEach((index) => { if (steps && steps[index]) { target = steps[index]; steps = target.steps || []; } });
  if (target) { state.selected = target; draw(); }
}
function showError(error) { $('output').textContent = error.message; $('output').className = 'error'; highlightError(error.message); }

$('new').onclick = () => { state.recipe = { schema_version: 1, name: 'scratch_recipe', description: '', steps: [] }; state.selected = null; draw(); };
$('check').onclick = async () => { try { await api('/api/validate', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload()) }); $('output').textContent = '检查通过'; $('output').className = 'ok'; } catch (error) { showError(error); } };
$('save').onclick = async () => { try { const result = await api('/api/save', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload()) }); $('output').textContent = `已保存: ${result.path}`; } catch (error) { showError(error); } };
$('run').onclick = async () => { try { const result = await api('/api/run', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload()) }); $('output').textContent = JSON.stringify(result, null, 2); } catch (error) { showError(error); } };
$('open').onclick = async () => { try { const names = await api('/api/recipes'); const choice = prompt('输入 recipe 文件名:\n' + names.join('\n')); if (choice) { state.recipe = await api('/api/recipe/' + encodeURIComponent(choice)); state.selected = null; draw(); } } catch (error) { showError(error); } };

api('/api/blocks').then((blocks) => { state.blocks = blocks; renderPalette(); draw(); }).catch((error) => { $('output').textContent = error.message; });
