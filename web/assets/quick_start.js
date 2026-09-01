(function () {
  "use strict";

  const state = {
    path: "",
    probe: null,
    dataset: null,
    registered: [],
    noise: null,
    noiseSelection: { kind: "clean", key: "clean", rate: null, seed: 1 },
    methods: [],
    selectedPaperId: "",
    methodInputs: {},
    labelsConfirmed: false,
    plan: null,
    loading: false,
    loadingMessage: "",
    loadingSlow: false,
    error: ""
  };
  let context = null;
  let panel = null;
  let loadingTimer = null;

  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (char) {
      return {"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[char];
    });
  }
  function statusLabel(status) {
    return ({ready:"基础条件满足", needs_input:"需要补充输入", unsupported:"当前实现不适用", metadata_error:"兼容性元数据不完整"})[status] || status;
  }
  function planStatusLabel(status) {
    return status === "ready" ? "已通过训练前检查 / 可运行" : statusLabel(status);
  }
  function beginLoading(message) {
    state.loading = true;
    state.loadingMessage = message || "正在处理…";
    state.loadingSlow = false;
    if (loadingTimer) window.clearTimeout(loadingTimer);
    loadingTimer = window.setTimeout(function () {
      if (state.loading) {
        state.loadingSlow = true;
        render();
      }
    }, 2200);
    render();
  }
  function endLoading() {
    state.loading = false;
    state.loadingMessage = "";
    state.loadingSlow = false;
    if (loadingTimer) window.clearTimeout(loadingTimer);
    loadingTimer = null;
    render();
  }
  function request(url, options, message) {
    beginLoading(message);
    return fetch(url, options).then(function (response) {
      return response.json().then(function (payload) {
        if (!response.ok) throw new Error(payload.error || "Quick Start 请求失败");
        return payload;
      });
    }).finally(endLoading);
  }
  function post(url, body, message) {
    return request(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)}, message);
  }
  function loadingMarkup() {
    if (!state.loading) return "";
    return '<div class="qs-loading" role="status" aria-live="polite"><span class="qs-spinner" aria-hidden="true"></span><div><strong>' + esc(state.loadingMessage) + '</strong>' + (state.loadingSlow ? '<p>仍在处理，数据集扫描可能需要一些时间，请勿重复点击。</p>' : '<p>请稍候，界面会在完成后自动更新。</p>') + '</div></div>';
  }
  function setStatus(message, kind) {
    if (context && context.status) context.status(message, kind || "");
  }
  function loadRegistered() {
    return request("/api/datasets", undefined, "正在读取已登记数据集…").then(function (payload) {
      state.registered = (payload.datasets || []).filter(function (item) { return item.location; });
    }).catch(function (error) { state.error = String(error.message || error); });
  }
  function pickPath() {
    return post("/api/picker", {mode:"folder", kind:"all", initial:state.path}, "正在打开路径选择器…").then(function (payload) {
      if (!payload.cancelled) { state.path = payload.path || state.path; render(); }
    }).catch(function (error) { state.error = String(error.message || error); render(); });
  }
  function renderProbe() {
    if (!state.probe) return "";
    if (state.probe.status === "detected") return '<div class="qs-feedback qs-success">✓ 检测到 ' + esc(state.probe.candidates[0].adapter) + '，可以继续登记并 inspect。</div>';
    if (state.probe.status === "already_registered") return '<div class="qs-feedback qs-success">✓ 已登记：' + esc(state.probe.existing_alias) + '</div>';
    if (state.probe.status === "ambiguous") return '<div class="qs-feedback qs-warning"><strong>检测到多个可能格式</strong>' + state.probe.candidates.map(function (item) { return '<button type="button" class="secondary qs-candidate" data-adapter="' + esc(item.adapter) + '">' + esc(item.adapter) + ' · ' + esc(item.reason) + '</button>'; }).join("") + '</div>';
    return '<div class="qs-feedback qs-error">暂时无法自动识别此路径。可以转到“本地数据集”手动登记。</div>';
  }
  function renderDataset() {
    const registered = state.registered.length ? '<label>使用已登记数据集<select id="qs-registered"><option value="">选择一个</option>' + state.registered.map(function (item) { return '<option value="' + esc(item.location) + '">' + esc(item.name + " · " + item.adapter) + '</option>'; }).join("") + '</select></label>' : "";
    const summary = state.dataset ? '<div class="qs-dataset-card"><strong>✓ ' + esc(state.dataset.display_name) + '</strong><p>' + esc(state.dataset.train_size) + ' train / ' + esc(state.dataset.test_size) + ' test · ' + esc(state.dataset.num_classes) + ' 类</p><p>标签：' + esc(state.dataset.clean_train_labels) + '；噪声来源：' + esc(state.dataset.noise_origin) + '</p></div>' : "";
    return '<section class="qs-step"><h3>1. 数据集</h3><p>选择一个数据集目录或文件；系统会自动识别、登记并实际加载检查。</p><div class="qs-path-row"><input id="qs-path" value="' + esc(state.path) + '" placeholder="例如 F:\\datasets\\cifar10"><button type="button" id="qs-pick" class="secondary">浏览</button></div><div class="qs-actions"><button type="button" id="qs-probe" class="primary">自动识别并继续</button>' + (state.dataset ? '<button type="button" id="qs-reset" class="secondary">更换数据集</button>' : "") + '</div>' + registered + renderProbe() + summary + '</section>';
  }
  function renderNoise() {
    if (!state.dataset || !state.noise) return "";
    const options = (state.noise.options || []).map(function (item) { return '<option value="' + esc(item.key) + '"' + (item.key === state.noiseSelection.key ? ' selected' : '') + '>' + esc(item.label) + '</option>'; }).join("");
    const native = state.noise.dataset_state === "native";
    const unknown = state.noise.requires_confirmation && !state.labelsConfirmed;
    return '<section class="qs-step"><h3>2. 标签噪声</h3>' + (native ? '<div class="qs-feedback qs-success">✓ 该数据集本身包含真实世界标签噪声。Quick Start 直接使用原始 noisy labels，不叠加人工噪声。</div>' : unknown ? '<div class="qs-feedback qs-warning"><strong>训练标签的噪声状态未知</strong><p>继续只表示使用当前 observed labels，不会把它们声明为干净标签。</p><button type="button" id="qs-confirm-labels" class="secondary">确认并检查可用方法</button></div>' : '<p>选择保持干净，或添加工具箱已有的人工噪声能力。</p><label>噪声类型<select id="qs-noise">' + options + '</select></label><div class="qs-noise-inputs"><label>噪声率<input id="qs-rate" type="number" min="0" max="1" step="0.01" value="' + esc(state.noiseSelection.rate == null ? "0.2" : state.noiseSelection.rate) + '"></label><label>随机种子<input id="qs-seed" type="number" step="1" value="' + esc(state.noiseSelection.seed == null ? "1" : state.noiseSelection.seed) + '"></label></div><p class="helper">人工噪声选项来自 noise capability catalog，不来自已有 recipe。</p>') + '</section>';
  }
  function methodCard(item) {
    const selected = item.paper_id === state.selectedPaperId ? " selected" : "";
    const selectable = item.status === "ready" || item.status === "needs_input";
    return '<button type="button" class="qs-method-card qs-status-' + esc(item.status) + selected + '" data-paper="' + esc(item.paper_id) + '"' + (selectable ? "" : ' disabled aria-disabled="true"') + '><strong>' + esc(item.acronym) + '</strong><span>' + esc(item.title) + '</span><small>' + esc(item.venue) + ' ' + esc(item.year) + ' · ' + esc(statusLabel(item.status)) + '</small><p>' + esc(item.summary) + '</p>' + (item.reasons && item.reasons.length ? '<small>' + esc(item.reasons.join("；")) + '</small>' : "") + '</button>';
  }
  function renderMethods() {
    if (!state.dataset || !state.methods.length) return "";
    const groups = ["ready", "needs_input", "unsupported", "metadata_error"];
    return '<section class="qs-step"><h3>3. 方法</h3>' + groups.map(function (status) { const items = state.methods.filter(function (item) { return item.status === status; }); return items.length ? '<div class="qs-method-group"><h4>' + statusLabel(status) + '</h4>' + items.map(methodCard).join("") + '</div>' : ""; }).join("") + '</section>';
  }
  function renderPlan() {
    if (!state.plan) return "";
    const plan = state.plan;
    const method = state.methods.find(function (item) { return item.paper_id === state.selectedPaperId; }) || {};
    const pathMap = {};
    (method.required_input_paths || []).forEach(function (item) { pathMap[item[0]] = item[1] || []; });
    const unresolved = plan.required_user_inputs || [];
    const fields = [];
    const manual = [];
    unresolved.forEach(function (name) {
      if (name === "noise_rate_prior") {
        fields.push('<label>方法噪声率先验<input class="qs-required-input" data-input-key="noise_rate_prior" type="number" min="0" max="1" step="0.01" value="' + esc(state.methodInputs.noise_rate_prior || "") + '"></label>');
        return;
      }
      const paths = pathMap[name] || [];
      if (!paths.length) { manual.push(name); return; }
      paths.forEach(function (path) {
        const key = path.join(".");
        fields.push('<label>' + esc(key) + '<input class="qs-required-input" data-input-key="' + esc(key) + '" value="' + esc(state.methodInputs[key] == null ? "" : state.methodInputs[key]) + '" placeholder="输入数字、字符串或 JSON"></label>');
      });
    });
    const inputs = fields.length ? '<div class="qs-feedback qs-warning"><strong>补充方法输入</strong><div class="qs-required-inputs">' + fields.join("") + '</div><button type="button" id="qs-apply-inputs" class="secondary">应用并重新检查</button></div>' : "";
    const manualNote = manual.length ? '<div class="qs-feedback qs-warning">以下条件属于数据集事实或外部产物，不能在 Quick Start 中伪造：' + esc(manual.join("、")) + '。请先在“本地数据集”或 YAML 编辑器中补齐。</div>' : "";
    return '<section class="qs-step"><h3>4. 配置来源与运行</h3><div class="qs-plan-card"><strong>' + esc(plan.config_kind === "paper_reproduction" ? "论文复现配置" : "Toolbox 适配配置") + '</strong><p>' + esc(plan.summary) + '</p><p>状态：' + esc(planStatusLabel(plan.status)) + '</p>' + (plan.details || []).map(function (item) { return '<p class="helper">' + esc(item) + '</p>'; }).join("") + '</div>' + inputs + manualNote + '<div class="qs-actions"><button type="button" id="qs-dry" class="secondary" ' + (plan.status === "ready" ? "" : "disabled") + '>预演</button><button type="button" id="qs-run" class="primary" ' + (plan.status === "ready" ? "" : "disabled") + '>确认并开始训练</button></div></section>';
  }
  function render() {
    if (!panel) return;
    panel.innerHTML = '<div class="qs-root' + (state.loading ? ' qs-is-loading' : '') + '"><h2>Quick Start</h2>' + loadingMarkup() + (state.error ? '<div class="qs-feedback qs-error">' + esc(state.error) + '</div>' : "") + renderDataset() + renderNoise() + renderMethods() + renderPlan() + '</div>';
    const path = document.getElementById("qs-path");
    path && path.addEventListener("input", function () { state.path = this.value; });
    document.getElementById("qs-pick")?.addEventListener("click", pickPath);
    document.getElementById("qs-registered")?.addEventListener("change", function () { if (this.value) registerPath(this.value, null); });
    document.getElementById("qs-probe")?.addEventListener("click", probeAndRegister);
    document.getElementById("qs-reset")?.addEventListener("click", function () { state.dataset = null; state.noise = null; state.methods = []; state.plan = null; state.methodInputs = {}; state.labelsConfirmed = false; render(); });
    panel.querySelectorAll(".qs-candidate").forEach(function (button) { button.addEventListener("click", function () { registerPath(state.path, this.dataset.adapter); }); });
    document.getElementById("qs-noise")?.addEventListener("change", updateNoise);
    document.getElementById("qs-rate")?.addEventListener("change", updateNoise);
    document.getElementById("qs-seed")?.addEventListener("change", updateNoise);
    document.getElementById("qs-confirm-labels")?.addEventListener("click", function () { state.labelsConfirmed = true; render(); loadMethods(); });
    panel.querySelectorAll(".qs-method-card").forEach(function (button) { button.addEventListener("click", function () { const item = state.methods.find(function (entry) { return entry.paper_id === button.dataset.paper; }); if (!item || !["ready", "needs_input"].includes(item.status)) return; state.selectedPaperId = this.dataset.paper; state.methodInputs = {}; buildPlan(); }); });
    panel.querySelectorAll(".qs-required-input").forEach(function (input) { input.addEventListener("input", function () { state.methodInputs[this.dataset.inputKey] = this.value; }); });
    document.getElementById("qs-apply-inputs")?.addEventListener("click", buildPlan);
    document.getElementById("qs-dry")?.addEventListener("click", function () { if (state.plan) { context.setRequest({command:state.plan.dry_run_command}, state.plan.dry_run_command); context.execute(); } });
    document.getElementById("qs-run")?.addEventListener("click", function () { if (state.plan) { const summary = [state.plan.summary].concat(state.plan.details || []).join("\n"); if (window.confirm("即将启动训练：\n\n" + summary + "\n\n确认继续？")) { context.setRequest({command:state.plan.command}, state.plan.command); context.execute(); } } });
  }
  function updateNoise() {
    const key = document.getElementById("qs-noise")?.value || "clean";
    state.noiseSelection = {kind:key === "clean" ? "clean" : "synthetic", key:key, rate:key === "clean" ? null : Number(document.getElementById("qs-rate")?.value || 0.2), seed:Number(document.getElementById("qs-seed")?.value || 1)};
    state.methods = [];
    state.methodInputs = {};
    state.plan = null;
    render();
    loadMethods();
  }
  function probeAndRegister() {
    state.path = document.getElementById("qs-path")?.value || state.path;
    if (!state.path) { state.error = "请先选择数据集路径"; render(); return; }
    state.error = "";
    post("/api/quick-start/probe", {path:state.path}, "正在识别数据集格式…").then(function (payload) { state.probe = payload; render(); if (payload.status === "detected" || payload.status === "already_registered") registerPath(state.path, null); }).catch(function (error) { state.error = String(error.message || error); render(); });
  }
  function registerPath(path, adapter) {
    post("/api/quick-start/register", {path:path, adapter:adapter || ""}, "正在登记数据集并检查样本…").then(function (payload) { if (payload.kind !== "dataset") { state.probe = payload; render(); return; } state.dataset = payload.dataset; state.error = ""; return loadRegistered().then(loadNoise); }).catch(function (error) { state.error = String(error.message || error); render(); });
  }
  function loadNoise() {
    return request("/api/quick-start/noises?dataset=" + encodeURIComponent(state.dataset.alias), undefined, "正在读取标签噪声能力…").then(function (payload) { state.noise = payload; state.labelsConfirmed = !payload.requires_confirmation; if (payload.dataset_state === "native") state.noiseSelection = {kind:"native", key:"native", rate:null, seed:null}; else state.noiseSelection = {kind:"clean", key:"clean", rate:null, seed:1}; render(); return payload.requires_confirmation ? null : loadMethods(); });
  }
  function loadMethods() {
    if (!state.dataset || !state.noiseSelection) return Promise.resolve();
    return post("/api/quick-start/methods", {dataset:state.dataset.alias, noise:state.noiseSelection}, "正在批量检查论文方法兼容性…").then(function (payload) { state.methods = payload.methods || []; render(); }).catch(function (error) { state.error = String(error.message || error); render(); });
  }
  function buildPlan() {
    if (!state.dataset || !state.selectedPaperId) return;
    const userInputs = {};
    Object.keys(state.methodInputs).forEach(function (key) { const raw = state.methodInputs[key]; if (raw === "") return; try { userInputs[key] = JSON.parse(raw); } catch (_error) { userInputs[key] = raw; } });
    post("/api/quick-start/plan", {dataset:state.dataset.alias, paper_id:state.selectedPaperId, noise:state.noiseSelection, user_inputs:userInputs}, "正在生成运行计划…").then(function (payload) { state.plan = payload; render(); }).catch(function (error) { state.error = String(error.message || error); render(); });
  }
  function mount(target, options) {
    panel = target;
    context = options || {};
    if (!state.registered.length) loadRegistered().then(render);
    render();
  }
  window.quickStartController = {mount:mount, render:render, onModuleEnter:mount};
}());
