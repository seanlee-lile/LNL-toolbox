(function () {
  "use strict";

  const state = {
    path: "",
    probe: null,
    dataset: null,
    registered: [],
    noise: null,
    noiseSelection: { kind: "clean", key: "clean", rate: null, seed: 1 },
    nativeRateMode: "unknown",
    methods: [],
    selectedPaperId: "",
    pendingPaper: null,
    methodInputs: {},
    labelsConfirmed: false,
    seedTouched: false,
    plan: null,
    planSchema: null,
    scheduleStash: {},
    parameterDraft: {},
    segmentedPaths: {},
    parameterError: "",
    methodListExpanded: false,
    currentStep: 0,
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
  function mountExternalResources(node, source, openPaper, onDownloaded, compact = false) {
    if (!node) return;
    const query = new URLSearchParams(source);
    let timer = null;
    let wasDownloading = false;
    async function refresh() {
      if (!node.isConnected) { if (timer) clearTimeout(timer); return; }
      try {
        const response = await fetch("/api/external-resources?" + query);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "无法检查外部数据");
        if (!node.isConnected) return;
        const items = payload.resources || [];
        const downloading = items.some(function (item) { return item.status === "downloading"; });
        const completed = wasDownloading && !downloading && items.every(function (item) { return item.ready; });
        wasDownloading = downloading;
        node.innerHTML = items.length ? '<div class="' + (compact ? 'qs-feedback' : 'guide-card') + '">' + (compact ? '' : '<h3>外部数据</h3>') + items.map(function (item) {
          const label = item.ready ? "已就绪" : item.status === "downloading" ? "正在下载…" : item.status === "failed" ? "下载失败" : "尚未准备";
          return '<div><strong>' + (compact ? '外部数据 ' : '') + esc(item.title) + '：' + label + '</strong>' + (compact ? '' : '<p class="helper">' + esc(item.note) + '</p><p>文件放置路径：<code style="overflow-wrap:anywhere">' + esc(item.path) + '</code></p>') +
            (item.error ? '<p class="status failed">' + esc(item.error) + '</p>' : '') +
            '<div class="actions">' + (!item.ready && item.downloadable ? '<button type="button" data-external-download="' + esc(item.id) + '"' + (item.status === "downloading" ? ' disabled' : '') + '>一键下载</button>' : '') +
            (item.url && !compact ? '<a href="' + esc(item.url) + '" target="_blank" rel="noreferrer">自行下载</a>' : '') +
            (openPaper && !item.ready ? '<button type="button" class="secondary" data-external-paper>前往论文栏查看准备说明</button>' : '') + '</div></div>';
        }).join('') + '<button type="button" class="secondary" data-external-refresh>刷新状态</button></div>' : '';
        node.querySelector("[data-external-refresh]")?.addEventListener("click", refresh);
        node.querySelectorAll("[data-external-paper]").forEach(function (button) { button.onclick = openPaper; });
        node.querySelectorAll("[data-external-download]").forEach(function (button) {
          button.onclick = async function () {
            button.disabled = true;
            button.textContent = "正在开始下载…";
            try {
              const result = await fetch("/api/external-resources/download", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(Object.assign({}, source, {resource:button.dataset.externalDownload}))});
              const body = await result.json();
              if (!result.ok) throw new Error(body.error || "下载失败");
              wasDownloading = true;
              await refresh();
            } catch (error) {
              if (!node.isConnected) return;
              button.disabled = false;
              button.textContent = "重试下载";
              const message = document.createElement("p");
              message.className = "status failed";
              message.textContent = "下载失败：" + error.message + "。请前往论文栏查看下载链接和放置路径。";
              node.appendChild(message);
            }
          };
        });
        if (timer) clearTimeout(timer);
        if (downloading) timer = setTimeout(refresh, 1000);
        if (completed && onDownloaded) onDownloaded();
      } catch (error) {
        if (!node.isConnected) return;
        node.replaceChildren();
        const message = document.createElement("p");
        message.textContent = "外部数据检查失败：" + error.message;
        node.appendChild(message);
        if (openPaper) {
          const button = document.createElement("button");
          button.textContent = "前往论文栏";
          button.onclick = openPaper;
          node.appendChild(button);
        }
      }
    }
    node.textContent = "正在检查外部数据…";
    refresh();
  }
  window.paperExternalResources = {mount:mountExternalResources};
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
    const slowHint = state.loadingMessage.startsWith("正在登记数据集")
      ? "首次验收需要读取训练和测试样本，请勿重复点击。"
      : "正在处理当前步骤；已验收数据集不会重新读取图像，请勿重复点击。";
    return '<div class="qs-loading" role="status" aria-live="polite"><span class="qs-spinner" aria-hidden="true"></span><div><strong>' + esc(state.loadingMessage) + '</strong>' + (state.loadingSlow ? '<p>' + slowHint + '</p>' : '<p>请稍候，界面会在完成后自动更新。</p>') + '</div></div>';
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
    return '<div class="qs-feedback qs-error">暂时无法自动识别此路径。可以转到“数据集”工作区手动登记。</div>';
  }
  function renderDataset() {
    const registered = state.registered.length ? '<label>使用已登记数据集<select id="qs-registered"><option value="">选择一个</option>' + state.registered.map(function (item) { return '<option value="' + esc(item.location) + '">' + esc(item.name + " · " + item.adapter) + '</option>'; }).join("") + '</select></label>' : "";
    const noise = state.noise || state.dataset;
    const status = noise?.noise_status || "unknown";
    const statusLabel = status === "clean" ? "已确认干净" : status === "noisy" ? "已确认含噪" : "标签干净性未确认";
    const evidence = noise?.status_source === "inspected" ? "验收时对照观测标签与干净标签" : noise?.status_source === "declared" ? "依据已登记的数据集声明" : "尚无足够证据";
    const rate = noise?.noise_rate;
    const rateText = status === "noisy" ? (rate?.value == null ? "原始噪声率尚未知晓" : "原始噪声率 " + Number(rate.value).toLocaleString() + (rate.status === "estimated" ? "（估计值）" : "（已知值）")) : "";
    const summary = state.dataset ? '<div class="qs-dataset-card"><strong>✓ ' + esc(state.dataset.display_name) + '</strong><p>' + esc(state.dataset.train_size) + ' train / ' + esc(state.dataset.test_size) + ' test · ' + esc(state.dataset.num_classes) + ' 类</p><p><b>带噪情况：</b>' + esc(statusLabel) + '；' + esc(evidence) + '</p>' + (rateText ? '<p>' + esc(rateText) + '</p>' : '') + '</div>' : "";
    return '<section class="qs-step"><h3>1. 数据集</h3><p>“浏览”只选择数据路径；点击“自动识别并继续”后，系统会识别并登记。若数据格式不明确，会先让你选择格式。</p><div class="qs-path-row"><input id="qs-path" value="' + esc(state.path) + '" placeholder="例如 F:\\datasets\\cifar10"><button type="button" id="qs-pick" class="secondary">浏览</button></div><div class="qs-actions"><button type="button" id="qs-probe" class="primary">自动识别并继续</button>' + (state.dataset ? '<button type="button" id="qs-reset" class="secondary">更换数据集</button>' : "") + '</div>' + registered + renderProbe() + summary + '</section>';
  }
  function renderNoise() {
    if (!state.dataset || !state.noise) return "";
    const native = ["native", "noisy"].includes(state.noise.dataset_state);
    const unknown = state.noise.requires_confirmation && !state.labelsConfirmed;
    const rate = state.noise.noise_rate || {};
    const noisy = native ? '<div class="qs-feedback qs-warning"><strong>训练标签已含噪声</strong><p>直接使用数据集原始观测标签，不再次叠加人工噪声。</p>' +
      (rate.value == null ? '<label>能否提供原始标签的噪声率？<select id="qs-native-rate-status"><option value="unknown"' + (state.nativeRateMode === "unknown" ? ' selected' : '') + '>目前不知道，继续使用原始标签</option><option value="known"' + (state.nativeRateMode === "known" ? ' selected' : '') + '>知道准确值</option><option value="estimated"' + (state.nativeRateMode === "estimated" ? ' selected' : '') + '>只有估计值</option></select></label>' +
        (state.nativeRateMode === "unknown" ? '<p class="helper">不知道时不填写推测数字；需要噪声率的方法会继续提示补充输入。</p>' : '<div class="qs-noise-inputs"><label>原始噪声率（0 到 1）<input id="qs-native-rate-value" type="number" min="0" max="1" step="0.001" placeholder="例如 0.2"></label><label>数值依据<input id="qs-native-rate-source" placeholder="例如官方标注说明或本地测量"></label></div><button type="button" id="qs-save-native-rate" class="secondary">保存为数据集事实</button><p class="helper">这里填写的是已有标签的实际或估计错误比例，不会重新生成标签，也不是论文方法的噪声率先验。</p>') : '<p>原始噪声率：' + esc(rate.value) + (rate.status === "estimated" ? '（估计值）' : '（已知值）') + (rate.provenance ? '；依据：' + esc(rate.provenance) : '') + '</p>') + '</div>' : "";
    const addNoise = state.noiseSelection.kind === "synthetic";
    const syntheticOptions = (state.noise.options || []).filter(function (item) { return item.key !== "clean"; });
    const selected = syntheticOptions.find(function (item) { return item.key === state.noiseSelection.key; }) || syntheticOptions[0];
    const clean = '<div class="qs-feedback qs-success"><strong>训练标签已确认干净</strong><p>可直接使用原始标签，也可以为噪声学习实验主动生成标签噪声；这不会修改磁盘上的原始数据集。</p></div>' +
      '<label class="qs-noise-choice"><input id="qs-add-noise" type="checkbox"' + (addNoise ? ' checked' : '') + '>是否添加人工标签噪声？</label>' +
      (addNoise ? '<label>加噪类型<select id="qs-noise">' + syntheticOptions.map(function (item) { return '<option value="' + esc(item.key) + '"' + (item.key === selected?.key ? ' selected' : '') + '>' + esc(item.label) + '</option>'; }).join("") + '</select></label>' +
        (selected?.description ? '<p class="helper">' + esc(selected.description) + '</p>' : '') +
        (selected?.key === "binary_asymmetric_rcn" ? '<div class="qs-noise-inputs"><label>正类翻转率<input id="qs-rho-positive" type="number" min="0" max="1" step="0.01" value="' + esc(state.noiseSelection.rho_positive ?? 0.2) + '"></label><label>负类翻转率<input id="qs-rho-negative" type="number" min="0" max="1" step="0.01" value="' + esc(state.noiseSelection.rho_negative ?? 0.1) + '"></label></div><p class="helper">分别表示真实正类被标成负类、真实负类被标成正类的概率；两者之和必须小于 1。</p>' :
        selected?.requires_rate ? '<label>人工噪声率<input id="qs-rate" type="number" min="0" max="1" step="0.01" value="' + esc(state.noiseSelection.rate == null ? "0.2" : state.noiseSelection.rate) + '"></label><p class="helper">这是新生成的错误标签比例，不是数据集原本的噪声率。</p>' : '') : '<p class="helper">不加噪：按登记的干净标签直接训练；不需要噪声率。</p>');
    const unknownMessage = '<div class="qs-feedback qs-warning"><strong>训练标签的干净性尚不能确认</strong><p>现有数据资料不足以判断其是否含噪，不能默认它是干净数据并提供人工加噪。可先在“数据集”中补充有依据的标签事实；也可以只按当前观测标签继续检查方法。</p><button type="button" id="qs-open-dataset-facts" class="secondary">查看数据集事实</button> <button type="button" id="qs-confirm-labels" class="secondary">按当前标签继续</button></div>';
    return '<section id="qs-noise-step" class="qs-step"><h3>2. 标签噪声</h3>' + (native ? noisy : unknown || state.noise.dataset_state === "unknown" ? unknownMessage : clean) +
      '<label>实验随机种子<input id="qs-seed" type="number" min="0" step="1" value="' + esc(state.noiseSelection.seed == null ? "1" : state.noiseSelection.seed) + '"></label><p class="helper">同一个种子用于本次实验的数据划分、人工加噪（若启用）和其他随机步骤。</p></section>';
  }
  function methodCard(item) {
    const selected = item.paper_id === state.selectedPaperId ? " selected" : "";
    const selectable = item.status === "ready" || item.status === "needs_input";
    const reasons = item.status === "needs_input" ? "选择后查看需要补充的条件" : (item.reasons || []).join("；");
    return '<button type="button" class="qs-method-card qs-status-' + esc(item.status) + selected + '" data-paper="' + esc(item.paper_id) + '"' + (selectable ? "" : ' disabled aria-disabled="true"') + '><strong>' + esc(item.acronym) + '</strong><span>' + esc(item.title) + '</span><small>' + esc(item.venue) + ' ' + esc(item.year) + ' · ' + esc(statusLabel(item.status)) + '</small><p>' + esc(item.summary) + '</p>' + (reasons ? '<small>' + esc(reasons) + '</small>' : "") + '</button>';
  }
  function renderMethods() {
    if (!state.dataset || !state.methods.length) return "";
    const groups = ["ready", "needs_input", "unsupported", "metadata_error"];
    const selected = state.methods.find(function (item) { return item.paper_id === state.selectedPaperId; });
    const expanded = !state.selectedPaperId || state.methodListExpanded;
    const selectedSummary = selected
      ? '<span class="qs-method-selection">已选择：' + esc(selected.acronym || selected.paper_id) + ' · ' + esc(selected.title || '') + '；点击标题可更换论文</span>'
      : '<span class="qs-method-selection">请选择一篇论文方法；兼容性状态会按数据集和噪声显示</span>';
    const cards = groups.map(function (status) { const items = state.methods.filter(function (item) { return item.status === status; }); return items.length ? '<div class="qs-method-group"><h4>' + statusLabel(status) + '</h4>' + items.map(methodCard).join("") + '</div>' : ""; }).join("");
    return '<details id="qs-methods-panel" class="qs-step qs-methods-panel"' + (expanded ? ' open' : '') + '><summary><strong>3. 方法（' + state.methods.length + '）</strong>' + selectedSummary + '</summary><div class="qs-methods-body">' + cards + '</div></details>';
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
    let needsNoiseSelection = false;
    let needsExternalLabels = false;
    let needsPrior = false;
    unresolved.forEach(function (name) {
      if (name === "noise_rate_prior") {
        needsPrior = true;
        return;
      }
      if (name === "config:requires_external_noise_labels") {
        needsExternalLabels = true;
        return;
      }
      const paths = pathMap[name] || [];
      if (!paths.length) { manual.push(name); return; }
      paths.forEach(function (path) {
        const key = path.join(".");
        if (["noise.name", "noise.rho_positive", "noise.rho_negative"].includes(key)) {
          needsNoiseSelection = true;
          return;
        }
        const labels = {
          "risk.rho_positive":"正类标签翻转率", "risk.rho_negative":"负类标签翻转率",
          "pipeline.transition_estimator.matrix":"当前噪声的类别转移矩阵（K×K）",
          "transition.matrix":"当前噪声的类别转移矩阵（K×K）",
          "transition.artifact":"转移矩阵文件路径",
          "trusted_validation.source":"可信验证样本来源",
          "trusted_validation.manifest":"可信验证样本清单路径",
          "pipeline.weight_provider.artifact_path":"Mentor 网络权重文件路径"
        };
        const label = labels[key];
        if (!label) { manual.push("请在论文配置中补充该方法的专用输入"); return; }
        const value = state.methodInputs[key] == null ? "" : state.methodInputs[key];
        if (key.endsWith(".matrix")) {
          fields.push('<label>' + label + '<textarea class="qs-required-input" data-input-key="' + esc(key) + '" placeholder="按类别顺序填写矩阵，例如 [[0.8,0.2],[0.2,0.8]]">' + esc(value) + '</textarea><small>只能填写与当前噪声类型、比例和类别顺序相符的矩阵；不能沿用另一种噪声的论文矩阵。</small></label>');
        } else {
          fields.push('<label>' + label + '<input class="qs-required-input" data-input-key="' + esc(key) + '" value="' + esc(value) + '" placeholder="请填写有依据的值"></label>');
        }
      });
    });
    if (needsPrior && !needsNoiseSelection) {
      fields.unshift('<label>方法使用的噪声率估计值<input class="qs-required-input" data-input-key="noise_rate_prior" type="number" min="0" max="1" step="0.01" value="' + esc(state.methodInputs.noise_rate_prior || "") + '"></label>');
    }
    const inputs = fields.length ? '<div class="qs-feedback qs-warning"><strong>补充方法输入</strong><div class="qs-required-inputs">' + fields.join("") + '</div><button type="button" id="qs-apply-inputs" class="secondary">应用并重新检查</button></div>' : "";
    const noisePrompt = needsNoiseSelection ? '<div class="qs-feedback qs-warning">请先在第 2 步选择标签噪声，并填写该噪声类型需要的比例。<button type="button" id="qs-back-to-noise" class="secondary">返回第 2 步设置标签噪声</button></div>' : "";
    const externalChoice = state.noiseSelection.kind === "synthetic" ? "第 2 步选择的人工噪声不能替代该文件；原论文配置中的文件路径和标签字段也不能用于另一种噪声。" : "当前的标签选择没有提供这组外部标签；不能把干净标签或未知来源的带噪标签当作该文件。";
    const externalPrompt = needsExternalLabels ? '<div class="qs-feedback qs-warning">CAL 当前实现需要与训练样本逐一对应的外部干净／带噪标签文件。' + externalChoice + '请在“论文”的外部数据栏目准备文件，并使用匹配该文件的实验配置。<button type="button" id="qs-open-paper-resources" class="secondary">查看论文外部数据</button></div>' : "";
    const manualLabels = {clean_train_labels:"可核实的干净训练标签", dataset_noise_rate:"数据集原始噪声率", noise_manifest:"与当前样本对应的噪声记录", method_requirements:"方法兼容性资料"};
    const manualNote = manual.length ? '<div class="qs-feedback qs-warning">还缺少不能凭空生成的资料：' + esc(Array.from(new Set(manual.map(function (name) { return manualLabels[name] || (name.startsWith("pretrained:") ? "预训练模型文件" : name.startsWith("请在") ? name : "方法专用输入"); }))).join("、")) + '。请在数据集或论文配置中核对。</div>' : "";
    const canRun = plan.status === "ready" && state.planSchema && !state.parameterError;
    const details = unresolved.length ? [] : (plan.details || []);
    const problem = plan.status === "ready" ? "" : '<div class="qs-feedback qs-warning"><strong>运行前还需处理</strong><p>' + esc(unresolved.length ? "请按下方提示补齐当前方法的条件。" : plan.summary) + '</p>' + details.map(function (detail) { return '<p>' + esc(detail) + '</p>'; }).join("") + '</div>';
    const transitionNote = plan.status === "ready" && state.selectedPaperId === "loss-correction" && ["symmetric", "pairflip", "binary_asymmetric_rcn"].includes(state.noiseSelection.key) ? '<p class="qs-feedback">原论文配置的转移矩阵不适用于当前噪声；已按第 2 步的类型、比例和当前类别数生成对应矩阵，原 Recipe 未改动。</p>' : "";
    return '<section class="qs-step"><h3>4. 运行计划</h3>' + problem + noisePrompt + externalPrompt + inputs + manualNote + transitionNote + renderPlanParameters() + '<div class="qs-actions"><button type="button" id="qs-dry" class="secondary" ' + (canRun ? "" : "disabled") + '>预演</button><button type="button" id="qs-run" class="primary" ' + (canRun ? "" : "disabled") + '>确认并开始训练</button></div></section>';
  }

  function parameterValue(field) {
    return Object.prototype.hasOwnProperty.call(state.parameterDraft, field.path)
      ? state.parameterDraft[field.path] : field.value;
  }
  function renderParameterField(field, segmentable) {
    segmentable = Boolean(segmentable && field.editable && field.display_group !== "restricted");
    const value = parameterValue(field);
    const currentTrainCount = Number(state.dataset?.train_size);
    const split = field.split_reference && Number.isInteger(currentTrainCount) && currentTrainCount > 0;
    const selectedDataset = field.path === "data.name";
    const readonly = !field.editable || selectedDataset;
    const label = selectedDataset ? "当前数据集" : field.label || field.presentation_label || field.path;
    let control;
    if (field.kind === "boolean") {
      control = '<input type="checkbox" data-qs-param="' + esc(field.path) + '"' + (value ? ' checked' : '') + (readonly ? ' disabled' : '') + '>';
    } else if (field.kind === "list" || field.kind === "object") {
      control = '<textarea data-qs-param="' + esc(field.path) + '"' + (readonly ? ' disabled' : '') + '>' + esc(JSON.stringify(value)) + '</textarea>';
    } else {
      const shown = selectedDataset ? state.dataset.alias : value == null ? "" : value;
      control = '<input type="' + (field.kind === "number" ? "number" : "text") + '" value="' + esc(shown) + '" data-qs-param="' + esc(field.path) + '"' + (split ? ' min="0" max="' + esc(currentTrainCount - 1) + '" step="1"' : '') + (readonly ? ' disabled' : '') + '>';
    }
    const noteText = segmentable ? '勾选后在上方分段表中编辑。' : readonly && field.lock_reason ? field.lock_reason : field.note;
    if (split) control = '<div class="qs-split-value">' + control + '<span>/ ' + esc(currentTrainCount) + '</span></div>';
    const paperSplit = field.split_reference ? '<small>原论文配置：' + esc(field.split_reference.count) + ' / ' + esc(field.split_reference.total) + (split && currentTrainCount !== field.split_reference.total ? '；当前值按相同比例换算并取整。' : '') + '</small>' : '';
    const note = paperSplit + (noteText ? '<small>' + esc(noteText) + '</small>' : '');
    if (segmentable) {
      return '<div class="qs-parameter-field"><div class="qs-parameter-title"><strong>' + esc(label) + '</strong><label class="qs-segment-choice"><input type="checkbox" data-qs-segment-param="' + esc(field.path) + '">加入分段表</label></div>' + control + note + '</div>';
    }
    return '<label class="qs-parameter-field"><span>' + esc(label) + '</span>' + control + note + '</label>';
  }
  function scheduleGroups(fields) {
    const groups = [];
    fields.filter(function (field) { return field.editable && field.path.endsWith(".name") && (field.path === "scheduler.name" || field.path.includes(".scheduler.")); }).forEach(function (nameField) {
      const prefix = nameField.path.slice(0, -5);
      groups.push({prefix:prefix, fields:fields.filter(function (field) { return field.editable && field.path.startsWith(prefix + "."); })});
    });
    return groups;
  }
  function scheduleValue(field) {
    return field ? parameterValue(field) : null;
  }
  function scheduleMilestones(field) {
    const value = scheduleValue(field);
    if (Array.isArray(value)) return value.map(Number);
    try { const parsed = JSON.parse(value); return Array.isArray(parsed) ? parsed.map(Number) : []; } catch (_) { return []; }
  }
  function scheduleExplicitRates(field) {
    const value = scheduleValue(field);
    if (Array.isArray(value)) return value.map(Number);
    if (value == null) return null;
    try { const parsed = JSON.parse(value); return Array.isArray(parsed) ? parsed.map(Number) : null; } catch (_) { return null; }
  }
  function scheduleBaseField(prefix, allFields) {
    const before = prefix === "scheduler" ? "" : prefix.split(".scheduler")[0] + ".";
    const after = prefix.includes(".scheduler.") ? prefix.split(".scheduler.")[1] + "." : "";
    return allFields.find(function (field) { return field.path === before + "optimizer." + after + "lr"; }) ||
      allFields.find(function (field) { return field.path.startsWith(before + "optimizer.") && field.path.endsWith("_lr"); });
  }
  function scheduleEnabled(group) {
    const name = String(scheduleValue(group.fields.find(function (field) { return field.path === group.prefix + ".name"; })) || "none");
    const steps = group.fields.find(function (field) { return field.path === group.prefix + ".step_milestones"; });
    return steps ? scheduleMilestones(steps).length > 0 : name !== "none";
  }
  function setScheduleEnabled(prefix, enabled) {
    const fields = state.planSchema?.fields || [];
    const find = function (suffix) { return fields.find(function (field) { return field.path === prefix + "." + suffix; }); };
    const nameField = find("name");
    const milestones = find("step_milestones") || find("milestones");
    const explicitField = find("lr_values");
    if (!nameField) return;
    if (!enabled) {
      state.scheduleStash[prefix] = {
        name: String(scheduleValue(nameField) || "none"),
        milestones: milestones ? scheduleMilestones(milestones) : null,
        rates: explicitField ? scheduleExplicitRates(explicitField) : null,
      };
      state.parameterDraft[nameField.path] = "none";
      if (explicitField) state.parameterDraft[explicitField.path] = null;
      if (milestones?.path.endsWith("step_milestones")) state.parameterDraft[milestones.path] = "[]";
      return;
    }
    const saved = state.scheduleStash[prefix];
    if (milestones?.path.endsWith("step_milestones")) {
      state.parameterDraft[milestones.path] = JSON.stringify(saved?.milestones ?? milestones.value ?? []);
      state.parameterDraft[nameField.path] = saved?.name ?? "none";
    } else {
      state.parameterDraft[nameField.path] = saved?.name && saved.name !== "none" ? saved.name :
        nameField.value !== "none" ? nameField.value : milestones ? "multistep" : "cosine";
    }
    if (explicitField && saved?.rates != null) state.parameterDraft[explicitField.path] = JSON.stringify(saved.rates);
  }
  function renderScheduleControl(group, allFields) {
    const lookup = new Map(group.fields.map(function (field) { return [field.path, field]; }));
    const prefix = group.prefix;
    const nameField = lookup.get(prefix + ".name");
    const name = String(scheduleValue(nameField) || "none");
    const milestoneField = lookup.get(prefix + ".step_milestones") || lookup.get(prefix + ".milestones");
    const milestones = scheduleMilestones(milestoneField);
    const stepBased = Boolean(milestoneField && milestoneField.path.endsWith("step_milestones"));
    const enabled = scheduleEnabled(group);
    const before = prefix === "scheduler" ? "" : prefix.split(".scheduler")[0] + ".";
    const baseField = scheduleBaseField(prefix, allFields);
    const baseRate = Number(scheduleValue(baseField));
    const gammaField = lookup.get(prefix + ".gamma");
    const gamma = Number(scheduleValue(gammaField) ?? 0.1);
    const explicitField = lookup.get(prefix + ".lr_values");
    const explicitRates = scheduleExplicitRates(explicitField);
    const canSegment = Boolean(baseField?.editable && (explicitField?.editable || gammaField?.editable));
    const budgetField = allFields.find(function (field) { return field.path === (stepBased ? "trainer.max_steps" : before + "epochs"); }) ||
      allFields.find(function (field) { return field.path === "trainer.epochs" || field.path.endsWith(".training.epochs"); });
    const budget = Number(scheduleValue(budgetField)) || Math.max(1, ...milestones);
    const title = prefix === "scheduler" ? "主模型学习率阶段" : prefix.replace(/\.scheduler(\.|$)/, " · ").replaceAll("_", " ") + "学习率阶段";
    const available = ["none"];
    if (lookup.has(prefix + ".milestones")) available.push("multistep");
    if (lookup.has(prefix + ".t_max") || name === "cosine" || available.length === 1) available.push("cosine");
    if (lookup.has(prefix + ".start_epoch") && name !== "linear_decay") available.push("linear_after");
    if (name === "linear_decay") available.push("linear_decay");
    if (!available.includes(name)) available.push(name);
    const mode = stepBased ? '<p class="helper">按累计更新步数在指定节点切换学习率。</p>' : '<label class="qs-schedule-mode">变化方式<select data-qs-param="' + esc(nameField.path) + '">' + available.map(function (item) { return '<option value="' + esc(item) + '"' + (item === name ? ' selected' : '') + '>' + esc(({none:"不变化",multistep:explicitField ? "按节点设置学习率" : "到节点后按倍率调整",cosine:"平滑下降",linear_after:"指定轮次后线性下降",linear_decay:"线性下降"})[item] || item) + '</option>'; }).join("") + '</select></label>';
    const showNodes = stepBased || name === "multistep";
    const ticks = (showNodes ? milestones : []).map(function (point, index) {
      const position = Math.max(0, Math.min(100, 100 * point / budget));
      return '<span class="qs-schedule-tick" style="left:' + position + '%" title="' + esc((stepBased ? "第 " + point + " 步" : "第 " + point + " 轮")) + '"></span>';
    }).join("");
    const segmented = Boolean(canSegment && state.segmentedPaths[baseField.path] !== false);
    const columns = milestones.map(function (point, index) {
      const rate = explicitRates?.[index] ?? (Number.isFinite(baseRate) && Number.isFinite(gamma) ? Number((baseRate * Math.pow(gamma, index + 1)).toPrecision(7)) : "");
      return {point:point, index:index, rate:rate};
    });
    const segmentTable = showNodes && milestoneField ? '<div class="qs-segment-table-wrap"><table class="qs-segment-table"><thead><tr><th scope="col">参数 / 阶段</th><th scope="col">初始段</th>' + columns.map(function (column) {
      return '<th scope="col"><label>节点 ' + (column.index + 1) + '（' + (stepBased ? '步' : '轮') + '）<input type="number" min="1" step="1" data-qs-schedule-milestone="' + esc(prefix) + '" data-index="' + column.index + '" value="' + esc(column.point) + '"></label><button type="button" class="secondary" data-qs-schedule-remove="' + esc(prefix) + '" data-index="' + column.index + '">移除节点</button></th>';
    }).join("") + '</tr></thead><tbody><tr><th scope="row">' + (canSegment ? '<label class="qs-segment-choice"><input type="checkbox" data-qs-segment-param="' + esc(baseField.path) + '"' + (segmented ? ' checked' : '') + '>学习率</label>' : '学习率') + '</th>' + (segmented ? '<td><input aria-label="初始段学习率" type="number" min="0" step="any" data-qs-param="' + esc(baseField.path) + '" value="' + esc(scheduleValue(baseField)) + '"></td>' + columns.map(function (column) {
      return '<td><input aria-label="节点 ' + (column.index + 1) + ' 后学习率" type="number" min="0" step="any" data-qs-schedule-rate="' + esc(prefix) + '" data-index="' + column.index + '" value="' + esc(column.rate) + '"></td>';
    }).join("") : '<td colspan="' + (columns.length + 1) + '" class="qs-segment-placeholder">学习率仍在下方编辑；勾选后移入此表，按阶段查看和调整。</td>') + '</tr></tbody></table></div><p class="helper">' + (segmented ? explicitField ? '表中是各段实际学习率；默认值来自当前论文配置，每一格都可以独立修改。' : '本方法的节点共用一个衰减倍率，修改一格会联动其他节点。' : '这里只切换学习率的编辑位置；若要关闭分段，请取消上方“启用分段学习率”。') + '</p>' : '';
    const otherFields = group.fields.filter(function (field) { return ![nameField.path, milestoneField?.path, gammaField?.path, explicitField?.path].includes(field.path); });
    return '<section class="qs-schedule" data-schedule="' + esc(prefix) + '"><div class="qs-schedule-head"><strong>' + esc(title) + '</strong><label class="qs-schedule-switch"><input type="checkbox" data-qs-schedule-toggle="' + esc(prefix) + '"' + (enabled ? ' checked' : '') + '>' + (stepBased || name === 'multistep' ? '启用分段学习率' : '启用学习率变化') + '</label></div>' +
      (enabled ? mode + '<p class="helper">' + (stepBased ? '横轴是累计参数更新步数。' : '横轴是训练轮次。') + '目前训练端只支持学习率分段；momentum、weight decay 等仍按整场训练的固定值执行。</p>' +
        '<div class="qs-schedule-track"><span>0</span><div class="qs-schedule-bar">' + ticks + '</div><span>' + esc(budget) + '</span></div>' +
        (showNodes && milestoneField ? segmentTable + '<button type="button" class="secondary" data-qs-schedule-add="' + esc(prefix) + '">添加节点</button>' : '<p class="helper">' + (name === "cosine" ? '学习率从起点平滑下降到终点，不使用突变节点。' : '当前方式不使用额外节点。') + '</p>') +
        (gammaField && !explicitField ? '<p class="helper">当前默认倍率：' + esc(gamma) + '；各节点共用此倍率。</p>' : '') +
        (otherFields.length ? '<div class="qs-parameter-grid">' + otherFields.map(function (field) { return renderParameterField(field, false); }).join("") + '</div>' : '') :
        '<p class="helper">不应用此调度阶段；学习率保持本阶段的起始值。开启后可编辑变化方式和节点。</p>') + '</section>';
  }
  function renderPlanParameters() {
    if (state.parameterError) return '<div class="qs-feedback qs-error">参数读取失败：' + esc(state.parameterError) + '</div>';
    if (!state.planSchema) return '<p class="helper">正在加载当前运行配置的参数…</p>';
    // Dataset, noise and the single experiment seed are controlled in the
    // earlier steps; never present duplicate editors for the same setting.
    const earlierStepPaths = new Set(["data.name", "noise.name", "noise.rate", "noise.seed", "seed"]);
    const fields = (state.planSchema.fields || []).filter(function (field) { return field.visible !== false && !earlierStepPaths.has(field.path); });
    const maxSteps = fields.find(function (field) { return field.path === "trainer.max_steps"; });
    const stepBudget = state.planSchema.method === "l2rw" && Number(scheduleValue(maxSteps)) > 0;
    const unusedBudgetPaths = new Set(stepBudget ? ["trainer.epochs"] : ["trainer.max_steps"]);
    const schedules = scheduleGroups(fields);
    const schedulePaths = new Set(schedules.flatMap(function (group) { return group.fields.map(function (field) { return field.path; }); }));
    const segmentablePaths = new Set(schedules.flatMap(function (group) {
      if (!scheduleEnabled(group)) return [];
      const name = String(scheduleValue(group.fields.find(function (field) { return field.path === group.prefix + ".name"; })) || "none");
      const hasNodes = group.fields.some(function (field) { return field.path === group.prefix + ".step_milestones" || field.path === group.prefix + ".milestones"; });
      const gamma = group.fields.find(function (field) { return field.path === group.prefix + ".gamma"; });
      const explicit = group.fields.find(function (field) { return field.path === group.prefix + ".lr_values"; });
      const base = scheduleBaseField(group.prefix, fields);
      return hasNodes && (explicit?.editable || gamma?.editable) && base?.editable && (name === "multistep" || group.fields.some(function (field) { return field.path === group.prefix + ".step_milestones"; })) ? [base.path] : [];
    }));
    const regular = fields.filter(function (field) { return !unusedBudgetPaths.has(field.path) && !schedulePaths.has(field.path) && !(segmentablePaths.has(field.path) && state.segmentedPaths[field.path] !== false); });
    const defaults = regular.filter(function (field) { return field.display_group === "default"; });
    const advanced = regular.filter(function (field) { return field.display_group === "advanced"; });
    const restricted = fields.filter(function (field) { return field.display_group === "restricted"; });
    const changed = Object.keys(state.parameterDraft).length > 0;
    const schedulePlan = schedules.length ? '<div class="qs-training-plan"><h4>模型训练计划</h4><p class="helper">先看学习率会怎样随训练进度变化。关闭阶段时保持起始学习率；开启后可选择变化方式，并在有节点的方式下调整节点及对应学习率。</p>' + schedules.map(function (group) { return renderScheduleControl(group, fields); }).join("") + '</div>' : '';
    return '<div class="qs-parameter-panel">' + schedulePlan + '<h4>可修改的训练参数</h4><div class="qs-parameter-grid">' + defaults.map(function (field) { return renderParameterField(field, segmentablePaths.has(field.path)); }).join("") + '</div>' +
      (advanced.length ? '<details class="qs-parameter-advanced"><summary>高级参数（' + advanced.length + '）</summary><div class="qs-parameter-grid">' + advanced.map(function (field) { return renderParameterField(field, segmentablePaths.has(field.path)); }).join("") + '</div></details>' : '') +
      (restricted.length ? '<details class="qs-parameter-advanced"><summary>禁止在 Web 修改的专属参数（' + restricted.length + '）</summary><p class="helper">这些值由配置契约固定；此处仅供查看。</p><div class="qs-parameter-grid">' + restricted.map(function (field) { return renderParameterField(field, false); }).join("") + '</div></details>' : '') +
      (changed ? '<p class="qs-feedback qs-warning">参数已修改。预演或运行前会保存为本次自定义配置并重新检查；不会改动论文原始 Recipe。</p>' : '') + '</div>';
  }

  function quickStartSteps() {
    const steps = [{id:"dataset", label:"数据集", render:renderDataset}];
    if (state.dataset && state.noise) steps.push({id:"noise", label:"标签噪声", render:renderNoise});
    if (state.dataset && state.noise && state.methods.length) steps.push({id:"methods", label:"方法兼容性", render:renderMethods});
    if (state.plan) steps.push({id:"plan", label:"运行计划", render:renderPlan});
    return steps;
  }

  function quickStartStepIndex(id, steps) {
    return (steps || quickStartSteps()).findIndex(function (step) { return step.id === id; });
  }

  function quickStartCanAdvance(step) {
    if (!step) return false;
    if (step.id === "dataset") return Boolean(state.dataset);
    if (step.id === "noise") return Boolean(state.noise) && (!state.noise.requires_confirmation || state.labelsConfirmed);
    if (step.id === "methods") return Boolean(state.selectedPaperId && state.plan);
    return false;
  }

  function renderQuickStartCarousel() {
    const steps = quickStartSteps();
    if (!steps.length) return "";
    state.currentStep = Math.max(0, Math.min(Number(state.currentStep) || 0, steps.length - 1));
    const active = steps[state.currentStep];
    const progress = steps.map(function (step, index) {
      const disabled = index > state.currentStep && !quickStartCanAdvance(steps[index - 1]);
      return '<button type="button" class="qs-carousel-dot' + (index === state.currentStep ? ' active' : '') + '" data-qs-step="' + index + '"' + (disabled ? ' disabled' : '') + ' aria-label="第 ' + (index + 1) + ' 步：' + esc(step.label) + '">' + (index + 1) + '<span>' + esc(step.label) + '</span></button>';
    }).join('<span class="qs-carousel-connector" aria-hidden="true"></span>');
    const previousDisabled = state.currentStep === 0;
    const nextDisabled = state.currentStep >= steps.length - 1 || !quickStartCanAdvance(active);
    return '<div class="qs-carousel" aria-label="Quick Start 步骤导航"><div class="qs-carousel-slide" aria-live="polite"><div class="qs-carousel-card"><div class="qs-carousel-nav"><button type="button" id="qs-prev" class="secondary"' + (previousDisabled ? ' disabled' : '') + '>&lt; 上一步</button><div class="qs-carousel-progress">' + progress + '</div><button type="button" id="qs-next" class="primary"' + (nextDisabled ? ' disabled' : '') + '>下一步 &gt;</button></div>' + active.render() + '</div></div><div class="qs-carousel-position">' + (state.currentStep + 1) + ' / ' + steps.length + '</div></div>';
  }

  function renderNextActions() {
    if (!context || !context.compact) return "";
    const intro = state.dataset
      ? "数据集已登记并通过基础加载检查。使用下方步骤导航逐步确认标签噪声、论文方法兼容性和可执行训练计划。"
      : "先选择或登记数据集，然后按步骤完成标签噪声、方法兼容性和运行计划。";
    return '<section class="qs-step qs-next-actions"><h3>完整方法兼容性引导</h3><p>' + intro + '</p>' + renderQuickStartCarousel() + '</section>';
  }
  function render() {
    if (!panel) return;
    if (state.dataset && context && context.datasetReady) context.datasetReady(state.dataset);
    if (context && context.paperSelected) context.paperSelected(state.methods.find(function (item) { return item.paper_id === state.selectedPaperId; }) || null);
    const compact = Boolean(context && context.compact);
    const guidedFlow = compact ? renderNextActions() : renderNoise() + renderMethods() + renderPlan();
    panel.innerHTML = '<div class="qs-root' + (state.loading ? ' qs-is-loading' : '') + '">' + (compact ? '' : '<h2>Quick Start</h2>') + loadingMarkup() + (state.error ? '<div class="qs-feedback qs-error">' + esc(state.error) + '</div>' : "") + (state.selectedPaperId ? '<div id="qs-external-resources"></div>' : '') + (compact ? "" : renderDataset()) + guidedFlow + '</div>';
    const selectedResourceMethod = state.methods.find(function (item) { return item.paper_id === state.selectedPaperId; });
    if (selectedResourceMethod?.recipe_id) {
      const currentPlan = state.plan?.paper_id === state.selectedPaperId ? state.plan : null;
      const source = {recipe:currentPlan?.recipe_id || selectedResourceMethod.recipe_id, dataset:state.dataset.alias, paper_id:state.selectedPaperId, noise:JSON.stringify(state.noiseSelection)};
      if (state.plan?.paper_id === state.selectedPaperId && state.plan.generated_config_path) source.config_path = state.plan.generated_config_path;
      const paperId = state.selectedPaperId;
      mountExternalResources(document.getElementById("qs-external-resources"), source, function () {
        if (context?.openPaperDetails) context.openPaperDetails(paperId);
        else context?.switchModule?.("papers");
      }, function () { if (state.selectedPaperId === paperId) buildPlan(); }, true);
    }
    const path = document.getElementById("qs-path");
    path && path.addEventListener("input", function () { state.path = this.value; });
    document.getElementById("qs-pick")?.addEventListener("click", pickPath);
    document.getElementById("qs-registered")?.addEventListener("change", function () { if (this.value) registerPath(this.value, null, true); });
    document.getElementById("qs-probe")?.addEventListener("click", probeAndRegister);
    document.getElementById("qs-reset")?.addEventListener("click", function () { state.dataset = null; state.noise = null; state.methods = []; state.plan = null; state.planSchema = null; state.parameterDraft = {}; state.segmentedPaths = {}; state.scheduleStash = {}; state.methodInputs = {}; state.selectedPaperId = ""; state.labelsConfirmed = false; state.seedTouched = false; state.methodListExpanded = false; state.currentStep = 0; render(); });
    panel.querySelectorAll(".qs-candidate").forEach(function (button) { button.addEventListener("click", function () { registerPath(state.path, this.dataset.adapter); }); });
    document.getElementById("qs-noise")?.addEventListener("change", updateNoise);
    document.getElementById("qs-add-noise")?.addEventListener("change", updateNoise);
    document.getElementById("qs-rate")?.addEventListener("change", updateNoise);
    document.getElementById("qs-rho-positive")?.addEventListener("change", updateNoise);
    document.getElementById("qs-rho-negative")?.addEventListener("change", updateNoise);
    document.getElementById("qs-seed")?.addEventListener("change", updateNoise);
    document.getElementById("qs-native-rate-status")?.addEventListener("change", function () { state.nativeRateMode = this.value; render(); });
    document.getElementById("qs-save-native-rate")?.addEventListener("click", saveNativeRate);
    document.getElementById("qs-open-dataset-facts")?.addEventListener("click", function () { context?.switchModule?.("data"); });
    document.getElementById("qs-confirm-labels")?.addEventListener("click", async function () { state.labelsConfirmed = true; render(); await loadMethods(); await continuePendingPaper(); });
    document.getElementById("qs-methods-panel")?.addEventListener("toggle", function () { state.methodListExpanded = this.open; });
    document.getElementById("qs-prev")?.addEventListener("click", function () { state.currentStep = Math.max(0, state.currentStep - 1); render(); });
    document.getElementById("qs-next")?.addEventListener("click", function () { const steps = quickStartSteps(); if (!quickStartCanAdvance(steps[state.currentStep])) return; state.currentStep = Math.min(steps.length - 1, state.currentStep + 1); render(); });
    panel.querySelectorAll(".qs-carousel-dot").forEach(function (button) { button.addEventListener("click", function () { const target = Number(this.dataset.qsStep); if (!this.disabled) { state.currentStep = target; render(); } }); });
    panel.querySelectorAll(".qs-method-card").forEach(function (button) { button.addEventListener("click", function () { const item = state.methods.find(function (entry) { return entry.paper_id === button.dataset.paper; }); if (!item || !["ready", "needs_input"].includes(item.status)) return; state.selectedPaperId = this.dataset.paper; state.methodInputs = {}; state.planSchema = null; state.parameterDraft = {}; state.segmentedPaths = {}; state.scheduleStash = {}; state.methodListExpanded = false; render(); if (state.seedTouched || !item.recipe_id) { buildPlan(); return; } request("/api/config-schema?recipe=" + encodeURIComponent(item.recipe_id), undefined, "正在读取论文默认随机种子…").then(function (schema) { if (state.selectedPaperId !== item.paper_id) return; const seed = (schema.fields || []).find(function (field) { return field.path === "seed"; }); if (seed) state.noiseSelection.seed = Number(seed.value); render(); buildPlan(); }).catch(function (error) { state.error = String(error.message || error); render(); }); }); });
    panel.querySelectorAll(".qs-required-input").forEach(function (input) { input.addEventListener("input", function () { state.methodInputs[this.dataset.inputKey] = this.value; }); });
    document.getElementById("qs-back-to-noise")?.addEventListener("click", function () {
      const index = quickStartStepIndex("noise");
      if (index >= 0) state.currentStep = index;
      render();
      document.getElementById("qs-noise-step")?.scrollIntoView({behavior:"smooth", block:"start"});
    });
    document.getElementById("qs-open-paper-resources")?.addEventListener("click", function () {
      if (context?.openPaperDetails) context.openPaperDetails(state.selectedPaperId);
      else context?.switchModule?.("papers");
    });
    function markParameterEdited() {
      const note = panel.querySelector(".qs-parameter-panel > .qs-feedback");
      if (!note) panel.querySelector(".qs-parameter-panel")?.insertAdjacentHTML("beforeend", '<p class="qs-feedback qs-warning">参数已修改。预演或运行前会保存为本次自定义配置并重新检查；不会改动论文原始 Recipe。</p>');
    }
    panel.querySelectorAll("[data-qs-param]").forEach(function (input) { input.addEventListener("input", function () {
      const path = this.dataset.qsParam;
      state.parameterDraft[path] = this.type === "checkbox" ? this.checked : this.value;
      const field = (state.planSchema?.fields || []).find(function (item) { return item.path === path; });
      for (const linkedPath of field?.linked_fields || []) {
        const linked = Array.from(panel.querySelectorAll("[data-qs-param]")).find(function (item) { return item.dataset.qsParam === linkedPath; });
        if (linked && !linked.disabled) {
          if (linked.type === "checkbox") linked.checked = input.checked;
          else linked.value = input.value;
          state.parameterDraft[linkedPath] = state.parameterDraft[path];
        }
      }
      markParameterEdited();
    }); });
    panel.querySelector('[data-qs-param="trainer.max_steps"]')?.addEventListener("change", function () { render(); });
    panel.querySelectorAll("[data-qs-segment-param]").forEach(function (checkbox) { checkbox.addEventListener("change", function () {
      state.segmentedPaths[this.dataset.qsSegmentParam] = this.checked;
      render();
    }); });
    panel.querySelectorAll(".qs-schedule-mode select").forEach(function (select) { select.addEventListener("change", render); });
    function scheduleField(prefix, suffix) {
      return (state.planSchema?.fields || []).find(function (field) { return field.path === prefix + "." + suffix; });
    }
    function scheduleMilestoneField(prefix) {
      return scheduleField(prefix, "step_milestones") || scheduleField(prefix, "milestones");
    }
    function scheduleExplicitField(prefix) { return scheduleField(prefix, "lr_values"); }
    function currentScheduleRates(prefix) {
      const field = scheduleExplicitField(prefix);
      const explicit = scheduleExplicitRates(field);
      if (explicit) return explicit;
      const fields = state.planSchema?.fields || [];
      const base = Number(scheduleValue(scheduleBaseField(prefix, fields)));
      const gamma = Number(scheduleValue(scheduleField(prefix, "gamma")) ?? 0.1);
      return scheduleMilestones(scheduleMilestoneField(prefix)).map(function (_, index) {
        return Number((base * Math.pow(gamma, index + 1)).toPrecision(7));
      });
    }
    panel.querySelectorAll("[data-qs-schedule-toggle]").forEach(function (toggle) { toggle.addEventListener("change", function () {
      setScheduleEnabled(this.dataset.qsScheduleToggle, this.checked);
      render();
    }); });
    panel.querySelectorAll("[data-qs-schedule-milestone]").forEach(function (input) { input.addEventListener("change", function () {
      const field = scheduleMilestoneField(this.dataset.qsScheduleMilestone);
      if (!field) return;
      const milestones = scheduleMilestones(field);
      const explicitField = scheduleExplicitField(this.dataset.qsScheduleMilestone);
      const explicit = scheduleExplicitRates(explicitField);
      const value = Number(this.value);
      if (!Number.isInteger(value) || value < 1 || milestones.some(function (item, index) { return index !== Number(input.dataset.index) && item === value; })) {
        this.setCustomValidity("节点必须是正整数，且不能重复"); this.reportValidity(); return;
      }
      this.setCustomValidity("");
      milestones[Number(this.dataset.index)] = value;
      const ordered = milestones.map(function (point, index) { return {point:point, rate:explicit?.[index]}; }).sort(function (a, b) { return a.point - b.point; });
      state.parameterDraft[field.path] = JSON.stringify(ordered.map(function (item) { return item.point; }));
      if (explicitField && explicit) state.parameterDraft[explicitField.path] = JSON.stringify(ordered.map(function (item) { return item.rate; }));
      render();
    }); });
    panel.querySelectorAll("[data-qs-schedule-add]").forEach(function (button) { button.addEventListener("click", function () {
      const field = scheduleMilestoneField(this.dataset.qsScheduleAdd);
      if (!field) return;
      const milestones = scheduleMilestones(field);
      const explicitField = scheduleExplicitField(this.dataset.qsScheduleAdd);
      const explicit = scheduleExplicitRates(explicitField);
      const next = (milestones.length ? Math.max(...milestones) : 0) + 1;
      milestones.push(next);
      state.parameterDraft[field.path] = JSON.stringify(milestones);
      if (explicitField && explicit) state.parameterDraft[explicitField.path] = JSON.stringify(explicit.concat(explicit.at(-1) ?? Number(scheduleValue(scheduleBaseField(this.dataset.qsScheduleAdd, state.planSchema?.fields || [])))));
      render();
    }); });
    panel.querySelectorAll("[data-qs-schedule-remove]").forEach(function (button) { button.addEventListener("click", function () {
      const field = scheduleMilestoneField(this.dataset.qsScheduleRemove);
      if (!field) return;
      const milestones = scheduleMilestones(field);
      const explicitField = scheduleExplicitField(this.dataset.qsScheduleRemove);
      const explicit = scheduleExplicitRates(explicitField);
      milestones.splice(Number(this.dataset.index), 1);
      state.parameterDraft[field.path] = JSON.stringify(milestones);
      if (explicitField && explicit) { explicit.splice(Number(this.dataset.index), 1); state.parameterDraft[explicitField.path] = JSON.stringify(explicit); }
      render();
    }); });
    panel.querySelectorAll("[data-qs-schedule-rate]").forEach(function (input) { input.addEventListener("change", function () {
      const prefix = this.dataset.qsScheduleRate;
      const gammaField = scheduleField(prefix, "gamma");
      const explicitField = scheduleExplicitField(prefix);
      const group = scheduleGroups(state.planSchema?.fields || []).find(function (item) { return item.prefix === prefix; });
      if ((!gammaField && !explicitField) || !group) return;
      const before = prefix === "scheduler" ? "" : prefix.split(".scheduler")[0] + ".";
      const after = prefix.includes(".scheduler.") ? prefix.split(".scheduler.")[1] + "." : "";
      const baseField = (state.planSchema.fields || []).find(function (field) { return field.path === before + "optimizer." + after + "lr"; }) ||
        (state.planSchema.fields || []).find(function (field) { return field.path.startsWith(before + "optimizer.") && field.path.endsWith("_lr"); });
      const base = Number(scheduleValue(baseField));
      const rate = Number(this.value);
      if (!(base > 0 && rate > 0 && Number.isFinite(rate))) {
        this.setCustomValidity("节点后的学习率必须是正数，且起始学习率有效"); this.reportValidity(); return;
      }
      this.setCustomValidity("");
      if (explicitField) {
        const values = currentScheduleRates(prefix);
        values[Number(this.dataset.index)] = rate;
        state.parameterDraft[explicitField.path] = JSON.stringify(values);
        render();
        return;
      }
      state.parameterDraft[gammaField.path] = String(Math.pow(rate / base, 1 / (Number(this.dataset.index) + 1)));
      render();
    }); });
    document.getElementById("qs-apply-inputs")?.addEventListener("click", buildPlan);
    document.getElementById("qs-dry")?.addEventListener("click", function () { executePlan(true); });
    document.getElementById("qs-run")?.addEventListener("click", function () { executePlan(false); });
  }
  function loadPlanSchema(plan) {
    const selected = state.methods.find(function (item) { return item.paper_id === plan.paper_id; });
    const source = plan.recipe_id ? "recipe=" + encodeURIComponent(plan.recipe_id)
      : plan.generated_config_path ? "path=" + encodeURIComponent(plan.generated_config_path) + "&recipe_hint=" + encodeURIComponent(selected?.recipe_id || "") : "";
    if (!source) return Promise.resolve();
    return request("/api/config-schema?" + source, undefined, "正在读取本次运行的参数…").then(function (schema) {
      if (state.plan?.plan_id !== plan.plan_id) return;
      state.planSchema = schema;
      state.parameterError = "";
      render();
    }).catch(function (error) {
      if (state.plan?.plan_id !== plan.plan_id) return;
      state.parameterError = String(error.message || error);
      render();
    });
  }
  function parameterPatches() {
    const fields = state.planSchema?.fields || [];
    const patches = [];
    for (const field of fields) {
      if (!Object.prototype.hasOwnProperty.call(state.parameterDraft, field.path)) continue;
      if (!field.editable || field.visible === false || field.path === "data.name") continue;
      const raw = state.parameterDraft[field.path];
      let value = raw;
      if (field.kind === "number") {
        if (field.nullable && String(raw).trim() === "") value = null;
        else {
          value = Number(raw);
          if (String(raw).trim() === "" || !Number.isFinite(value) || (field.number_type === "integer" && !Number.isInteger(value)))
            throw new Error(field.path + " 必须填写" + (field.number_type === "integer" ? "整数" : "数字"));
        }
      } else if (field.kind === "list" || field.kind === "object") {
        if (field.nullable && raw == null) value = null;
        else {
          try { value = JSON.parse(raw); } catch (_) { throw new Error(field.path + " 必须填写合法 JSON"); }
          if (field.kind === "list" && !Array.isArray(value)) throw new Error(field.path + " 必须是 JSON 列表");
          if (field.kind === "object" && (value === null || Array.isArray(value) || typeof value !== "object")) throw new Error(field.path + " 必须是 JSON 对象");
        }
      }
      if (JSON.stringify(value) !== JSON.stringify(field.value)) patches.push({path:field.path, value:value});
    }
    return patches;
  }
  function commandForSavedConfig(command, plan, path) {
    const original = plan.recipe_id ? "--recipe " + plan.recipe_id : "--config " + plan.generated_config_path;
    if (!command || !command.includes(original)) throw new Error("运行计划的配置来源已变化，请重新生成计划");
    return command.replace(original, "--config " + path);
  }
  function currentRunSummary(plan, patches) {
    const fields = state.planSchema?.fields || [];
    const field = function (path) { return fields.find(function (item) { return item.path === path; }); };
    const maxSteps = field("trainer.max_steps");
    const epochs = field("trainer.epochs");
    const budget = maxSteps && Number(parameterValue(maxSteps)) > 0
      ? "总更新步数：" + parameterValue(maxSteps)
      : epochs ? "训练轮数：" + parameterValue(epochs) : "";
    return ["论文方法：" + plan.method, "数据集：" + state.dataset.alias, budget,
      patches.length ? "已修改 " + patches.length + " 个参数，将另存为本次自定义配置。" : ""
    ].filter(Boolean).join("\n");
  }
  async function executePlan(dryRun) {
    const plan = state.plan;
    if (!plan || plan.status !== "ready" || !state.planSchema || state.parameterError) return;
    try {
      const patches = parameterPatches();
      if (!dryRun && !window.confirm("即将启动训练：\n\n" + currentRunSummary(plan, patches) + "\n\n确认继续？")) return;
      let command = dryRun ? plan.dry_run_command : plan.command;
      if (patches.length) {
        const selected = state.methods.find(function (item) { return item.paper_id === plan.paper_id; });
        const source = plan.recipe_id ? {recipe:plan.recipe_id} : {source_path:plan.generated_config_path, recipe_hint:selected?.recipe_id || ""};
        const destination = "artifacts/web-quick-start/configs/" + plan.plan_id + "-edited.yaml";
        const saved = await post("/api/configs", {
          ...source, path:destination, patches:patches, dataset_alias:state.dataset.alias,
          overwrite:true, acknowledge_paper_impact:true
        }, "正在保存并检查本次修改后的运行配置…");
        if (state.plan?.plan_id !== plan.plan_id) return;
        command = commandForSavedConfig(command, plan, saved.path);
      }
      context.setRequest({command:command}, command);
      context.execute();
    } catch (error) {
      state.error = String(error.message || error);
      render();
    }
  }
  function updateNoise(event) {
    if (event?.target?.id === "qs-seed") state.seedTouched = true;
    const native = ["native", "noisy"].includes(state.noise?.dataset_state);
    const addNoise = !native && state.noise?.dataset_state === "clean" && Boolean(document.getElementById("qs-add-noise")?.checked);
    const key = native ? "native" : addNoise ? (document.getElementById("qs-noise")?.value || (state.noise.options || []).find(function (item) { return item.key !== "clean"; })?.key) : "clean";
    const selected = (state.noise?.options || []).find(function (item) { return item.key === key; });
    const rawRate = document.getElementById("qs-rate")?.value;
    const rate = addNoise && selected?.requires_rate ? Number(rawRate ?? state.noiseSelection.rate ?? 0.2) : null;
    const binary = addNoise && key === "binary_asymmetric_rcn";
    const positiveInput = document.getElementById("qs-rho-positive");
    const negativeInput = document.getElementById("qs-rho-negative");
    const positive = binary ? Number(positiveInput?.value ?? state.noiseSelection.rho_positive ?? 0.2) : null;
    const negative = binary ? Number(negativeInput?.value ?? state.noiseSelection.rho_negative ?? 0.1) : null;
    if (binary && (positiveInput?.value === "" || negativeInput?.value === "" || !Number.isFinite(positive) || !Number.isFinite(negative) || positive < 0 || negative < 0 || positive + negative >= 1)) {
      state.error = "正类和负类翻转率都必须非负，且两者之和小于 1"; render(); return;
    }
    if (addNoise && selected?.requires_rate && (rawRate === "" || !Number.isFinite(rate) || rate < 0 || rate > 1)) {
      state.error = "人工噪声率必须是 0 到 1 之间的数字"; render(); return;
    }
    state.error = "";
    state.noiseSelection = {kind:native ? "native" : addNoise ? "synthetic" : "clean", key:key, rate:rate, seed:Number(document.getElementById("qs-seed")?.value ?? 1), rho_positive:positive, rho_negative:negative};
    state.methods = [];
    state.selectedPaperId = "";
    state.methodListExpanded = false;
    state.methodInputs = {};
    state.plan = null;
    state.planSchema = null;
    state.parameterDraft = {};
    state.segmentedPaths = {};
    state.scheduleStash = {};
    state.currentStep = 1;
    render();
    loadMethods();
  }
  function saveNativeRate() {
    const value = Number(document.getElementById("qs-native-rate-value")?.value);
    const source = document.getElementById("qs-native-rate-source")?.value.trim() || "";
    if (!Number.isFinite(value) || value < 0 || value > 1 || document.getElementById("qs-native-rate-value")?.value === "" || !source) {
      state.error = "请填写 0 到 1 之间的原始噪声率，并说明数字的依据"; render(); return;
    }
    const rate = {status:state.nativeRateMode, value:value, provenance:source};
    post("/api/datasets/" + encodeURIComponent(state.dataset.alias) + "/declarations", {declarations:{noise_rate:rate}}, "正在保存数据集噪声率资料…")
      .then(function () { state.error = ""; return loadNoise(); })
      .catch(function (error) { state.error = String(error.message || error); render(); });
  }
  function probeAndRegister() {
    state.path = document.getElementById("qs-path")?.value || state.path;
    if (!state.path) { state.error = "请先选择数据集路径"; render(); return; }
    state.error = "";
    post("/api/quick-start/probe", {path:state.path}, "正在识别数据集格式…").then(function (payload) { state.probe = payload; render(); if (payload.status === "detected" || payload.status === "already_registered") registerPath(state.path, null, payload.status === "already_registered"); }).catch(function (error) { state.error = String(error.message || error); render(); });
  }
  function registerPath(path, adapter, alreadyRegistered) {
    const message = alreadyRegistered ? "正在读取数据集验收记录（未验收时才检查样本）…" : "正在登记数据集并检查训练/测试样本…";
    return post("/api/quick-start/register", {path:path, adapter:adapter || ""}, message).then(function (payload) { if (payload.kind !== "dataset") { state.probe = payload; render(); return; } state.dataset = payload.dataset; state.error = ""; state.selectedPaperId = ""; state.plan = null; state.planSchema = null; state.parameterDraft = {}; state.segmentedPaths = {}; state.scheduleStash = {}; state.methodListExpanded = false; state.currentStep = 1; return loadRegistered().then(function () { render(); return loadNoise(); }); }).catch(function (error) { state.error = String(error.message || error); render(); return null; });
  }
  function loadNoise() {
    return request("/api/quick-start/noises?dataset=" + encodeURIComponent(state.dataset.alias), undefined, "正在读取已验收数据集的标签与噪声资料…").then(function (payload) { state.noise = payload; state.nativeRateMode = "unknown"; state.labelsConfirmed = !payload.requires_confirmation; if (["native", "noisy"].includes(payload.dataset_state)) state.noiseSelection = {kind:"native", key:"native", rate:null, seed:state.noiseSelection.seed ?? 1}; else state.noiseSelection = {kind:"clean", key:"clean", rate:null, seed:state.noiseSelection.seed ?? 1}; render(); return payload.requires_confirmation ? null : loadMethods(); });
  }
  function refreshNoiseFacts() {
    if (!state.dataset) return Promise.resolve();
    return request("/api/quick-start/noises?dataset=" + encodeURIComponent(state.dataset.alias), undefined, "正在更新数据集标签资料…").then(function (payload) {
      if (JSON.stringify(payload) === JSON.stringify(state.noise)) return;
      state.noise = payload;
      state.nativeRateMode = "unknown";
      state.labelsConfirmed = !payload.requires_confirmation;
      state.methods = [];
      state.selectedPaperId = "";
      state.plan = null;
      state.planSchema = null;
      state.parameterDraft = {};
      state.segmentedPaths = {};
      state.scheduleStash = {};
      if (["native", "noisy"].includes(payload.dataset_state)) state.noiseSelection = {kind:"native", key:"native", rate:null, seed:state.noiseSelection.seed ?? 1};
      else if (payload.dataset_state === "unknown" || state.noiseSelection.kind === "native") state.noiseSelection = {kind:"clean", key:"clean", rate:null, seed:state.noiseSelection.seed ?? 1};
      render();
      return payload.requires_confirmation ? null : loadMethods();
    }).catch(function (error) { state.error = String(error.message || error); render(); });
  }
  function loadMethods() {
    if (!state.dataset || !state.noiseSelection) return Promise.resolve();
    return post("/api/quick-start/methods", {dataset:state.dataset.alias, noise:state.noiseSelection}, "正在用已验收数据资料匹配 26 篇论文方法…").then(function (payload) { state.methods = payload.methods || []; render(); }).catch(function (error) { state.error = String(error.message || error); render(); });
  }
  function buildPlan() {
    if (!state.dataset || !state.selectedPaperId) return;
    const userInputs = {};
    Object.keys(state.methodInputs).forEach(function (key) { const raw = state.methodInputs[key]; if (raw === "") return; try { userInputs[key] = JSON.parse(raw); } catch (_error) { userInputs[key] = raw; } });
    post("/api/quick-start/plan", {dataset:state.dataset.alias, paper_id:state.selectedPaperId, noise:state.noiseSelection, user_inputs:userInputs}, "正在核对配置与已验收数据资料…").then(function (payload) { state.plan = payload; state.planSchema = null; state.parameterDraft = {}; state.segmentedPaths = {}; state.scheduleStash = {}; state.parameterError = ""; const index = quickStartStepIndex("plan"); if (index >= 0) state.currentStep = index; render(); return loadPlanSchema(payload); }).catch(function (error) { state.error = String(error.message || error); render(); });
  }

  function schemaValue(schema, path) {
    const field = (schema?.fields || []).find(function (item) { return item.path === path; });
    return field ? field.value : undefined;
  }

  async function continuePendingPaper() {
    const pending = state.pendingPaper;
    if (!pending || !state.dataset || !state.noise) return;
    if (state.noise.requires_confirmation && !state.labelsConfirmed) return;
    const datasetState = state.noise.dataset_state;
    const configuredNoise = schemaValue(pending.schema, "noise.name");
    const configuredRate = schemaValue(pending.schema, "noise.rate");
    const configuredSeed = schemaValue(pending.schema, "seed");
    if (configuredSeed != null) state.noiseSelection.seed = Number(configuredSeed);
    if (["native", "noisy"].includes(datasetState)) {
      state.noiseSelection = {kind:"native", key:"native", rate:null, seed:state.noiseSelection.seed ?? 1};
    } else if (datasetState === "clean") {
      if (!configuredNoise || configuredNoise === "clean") {
        state.noiseSelection = {kind:"clean", key:"clean", rate:null, seed:state.noiseSelection.seed ?? 1};
      } else {
        const option = (state.noise.options || []).find(function (item) { return item.key === configuredNoise; });
        if (!option) {
          state.error = "Quick Start 不提供该论文配置的噪声类型：" + configuredNoise;
          state.currentStep = quickStartStepIndex("noise");
          render();
          return;
        }
        state.noiseSelection = {kind:"synthetic", key:option.key, rate:option.requires_rate ? Number(configuredRate) : null, seed:state.noiseSelection.seed ?? 1,
          rho_positive:configuredNoise === "binary_asymmetric_rcn" ? Number(schemaValue(pending.schema, "noise.rho_positive")) : null,
          rho_negative:configuredNoise === "binary_asymmetric_rcn" ? Number(schemaValue(pending.schema, "noise.rho_negative")) : null};
      }
    }
    state.pendingPaper = null;
    state.selectedPaperId = pending.paperId;
    state.methodInputs = {};
    state.plan = null;
    state.planSchema = null;
    state.parameterDraft = {};
    state.segmentedPaths = {};
    state.scheduleStash = {};
    state.methodListExpanded = false;
    await loadMethods();
    buildPlan();
  }

  async function openPaper(paperId, recipeId, adapter) {
    state.error = "";
    state.pendingPaper = null;
    context?.switchModule?.("quickstart");
    try {
      const schema = await request("/api/config-schema?recipe=" + encodeURIComponent(recipeId), undefined, "正在读取论文配置…");
      if (!state.registered.length) await loadRegistered();
      const requiredAdapter = schemaValue(schema, "data.name") || adapter;
      const registered = state.registered.find(function (item) { return item.adapter === requiredAdapter; });
      if (!state.dataset || state.dataset.alias !== registered?.name) {
        if (!registered) {
          state.currentStep = 0;
          state.error = "请先登记论文配置需要的数据集（" + requiredAdapter + "），再开始训练。";
          render();
          return;
        }
        await registerPath(registered.location, null, true);
      }
      if (!state.dataset || !state.noise) {
        state.currentStep = 0;
        render();
        return;
      }
      state.pendingPaper = {paperId:paperId, recipeId:recipeId, schema:schema};
      if (state.noise.requires_confirmation && !state.labelsConfirmed) {
        state.currentStep = quickStartStepIndex("noise");
        render();
        return;
      }
      await continuePendingPaper();
    } catch (error) {
      state.error = String(error.message || error);
      render();
    }
  }
  function mount(target, options) {
    panel = target;
    context = options || {};
    if (!state.registered.length) loadRegistered().then(render);
    render();
    if (state.dataset) refreshNoiseFacts();
  }
  window.quickStartController = {mount:mount, render:render, onModuleEnter:mount, openPaper:openPaper, showCompatibilityGuide:function () { render(); return state.noise ? Promise.resolve() : loadNoise(); }};
}());
