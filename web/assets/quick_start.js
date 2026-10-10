(function () {
  "use strict";

  window.parameterChoiceHelp = {
    options(choices, current) {
      return choices.map(function (option) {
        return '<option value="' + esc(option.value) + '" data-description="' + esc(option.description || '') + '"' + (option.value === String(current) ? ' selected' : '') + (option.unsupported ? ' disabled' : '') + '>' + esc(option.label) + '</option>';
      }).join('');
    },
    render(choices, current) {
      const description = choices?.find(function (option) { return option.value === String(current); })?.description || '';
      return '<small class="parameter-choice-note" data-parameter-choice-note' + (description ? '' : ' hidden') + '>' + esc(description) + '</small>';
    },
    update(input) {
      if (input.tagName !== 'SELECT') return;
      const note = input.parentElement.querySelector('[data-parameter-choice-note]');
      if (!note) return;
      note.textContent = input.selectedOptions[0]?.dataset.description || '';
      note.hidden = !note.textContent;
    }
  };

  // Both parameter editors use the same typed controls for lists containing
  // selectable strings. The hidden JSON value remains the API representation.
  window.parameterListEditor = {
    render(field, value, readonly) {
      if (typeof value === "string") { try { value = JSON.parse(value); } catch (_) { value = field.value; } }
      return field.rows.map(function (row, index) {
        return '<fieldset><legend>' + (index + 1) + '</legend>' + row.map(function (child) {
          const current = value?.[index]?.[child.key] ?? child.value;
          const attrs = ' data-parameter-list="' + esc(field.path) + '" data-list-index="' + index + '" data-list-key="' + esc(child.key) + '" data-list-kind="' + esc(child.kind) + '"' + (readonly ? ' disabled' : '');
          let control;
          if (child.choices) {
            const options = child.choices.some(function (option) { return option.value === String(current); }) ? child.choices : [{value:String(current), label:"当前配置含未支持选项", unsupported:true}, ...child.choices];
            control = '<select' + attrs + '>' + window.parameterChoiceHelp.options(options, current) + '</select>';
          } else {
            control = '<input type="' + (child.kind === 'boolean' ? 'checkbox' : 'number') + '" step="any"' + attrs + (child.kind === 'boolean' ? (current ? ' checked' : '') : ' value="' + esc(current) + '"') + '>';
          }
          return '<label><span>' + esc(child.label) + '</span>' + control + (child.choices ? window.parameterChoiceHelp.render(child.choices, current) : '') + '<small>' + esc(child.note) + '</small></label>';
        }).join('') + '</fieldset>';
      }).join('');
    },
    bind(root, findBacking) {
      root.querySelectorAll('[data-parameter-list]').forEach(function (input) {
        input.addEventListener(input.tagName === 'SELECT' ? 'change' : 'input', function () {
          window.parameterChoiceHelp.update(input);
          const backing = findBacking(input.dataset.parameterList);
          const value = JSON.parse(backing.value);
          value[Number(input.dataset.listIndex)][input.dataset.listKey] = input.dataset.listKind === 'boolean' ? input.checked : input.dataset.listKind === 'number' ? Number(input.value) : input.value;
          backing.value = JSON.stringify(value);
          backing.dispatchEvent(new Event('input', {bubbles:true}));
        });
      });
    }
  };

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
    expandedSchedules: {},
    parameterDraft: {},
    segmentedPaths: {},
    parameterError: "",
    review: null,
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
  let renderedStep = null;
  let reviewToken = 0;

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
    return request("/api/datasets", undefined, "正在读取数据集…").then(function (payload) {
      state.registered = (payload.datasets || []).filter(function (item) { return item.location; });
    }).catch(function (error) { state.error = String(error.message || error); });
  }
  function pickPath() {
    return post("/api/picker", {mode:"folder", kind:"all", initial:state.path}, "正在打开路径选择器…").then(function (payload) {
      if (!payload.cancelled) changeDatasetPath(payload.path || state.path);
    }).catch(function (error) { state.error = String(error.message || error); render(); });
  }
  function changeDatasetPath(path, redraw = true) {
    path = String(path || "").trim();
    if (path === state.path) return;
    state.path = path;
    state.probe = null;
    state.dataset = null;
    state.noise = null;
    state.methods = [];
    state.plan = null;
    state.planSchema = null;
    state.parameterDraft = {};
    state.segmentedPaths = {};
    state.scheduleStash = {};
    state.expandedSchedules = {};
    state.methodInputs = {};
    state.selectedPaperId = "";
    state.pendingPaper = null;
    state.labelsConfirmed = false;
    state.seedTouched = false;
    state.methodListExpanded = false;
    state.parameterError = "";
    state.review = null;
    reviewToken++;
    state.error = "";
    state.currentStep = 0;
    if (redraw) render();
  }
  function renderProbe() {
    if (!state.probe) return "";
    if (state.probe.status === "detected") return '<div class="qs-feedback qs-success">✓ 检测到 ' + esc(state.probe.candidates[0].adapter) + '，可以继续登记并 inspect。</div>';
    if (state.probe.status === "already_registered") return '<div class="qs-feedback qs-success">✓ 已登记：' + esc(state.probe.existing_alias) + '</div>';
    if (state.probe.status === "ambiguous") return '<div class="qs-feedback qs-warning"><strong>检测到多个可能格式</strong>' + state.probe.candidates.map(function (item) { return '<button type="button" class="secondary qs-candidate" data-adapter="' + esc(item.adapter) + '">' + esc(item.adapter) + ' · ' + esc(item.reason) + '</button>'; }).join("") + '</div>';
    return '<div class="qs-feedback qs-error">暂时无法自动识别此路径。可以转到“数据集”工作区手动登记。</div>';
  }
  function renderDataset() {
    const registered = state.registered.length ? '<label>使用过的数据集<select id="qs-registered"><option value="">选择一个</option>' + state.registered.map(function (item) { return '<option value="' + esc(item.location) + '"' + (item.name === state.dataset?.alias ? ' selected' : '') + '>' + esc(item.name + " · " + item.adapter) + '</option>'; }).join("") + '</select></label>' : "";
    const noise = state.noise || state.dataset;
    const status = noise?.noise_status || "unknown";
    const statusLabel = status === "clean" ? "已确认干净" : status === "noisy" ? "已确认含噪" : "标签干净性未确认";
    const evidence = noise?.status_source === "inspected" ? "验收时对照观测标签与干净标签" : noise?.status_source === "declared" ? "依据已登记的数据集声明" : "尚无足够证据";
    const rate = noise?.noise_rate;
    const rateText = status === "noisy" ? (rate?.value == null ? "原始噪声率尚未知晓" : "原始噪声率 " + Number(rate.value).toLocaleString() + (rate.status === "estimated" ? "（估计值）" : "（已知值）")) : "";
    const summary = state.dataset ? '<div class="qs-dataset-card"><strong>✓ ' + esc(state.dataset.display_name) + '</strong><p>' + esc(state.dataset.train_size) + ' train / ' + esc(state.dataset.test_size) + ' test · ' + esc(state.dataset.num_classes) + ' 类</p><p><b>带噪情况：</b>' + esc(statusLabel) + '；' + esc(evidence) + '</p>' + (rateText ? '<p>' + esc(rateText) + '</p>' : '') + '</div>' : "";
    return '<section class="qs-step"><h3>1. 数据集</h3><p>选择数据目录或压缩包，点击“下一步”后识别并登记。支持官方 CIFAR 文件和通用 dataset.yaml＋samples.csv；ZIP、tar.gz 会安全解压后使用同一识别流程。若格式不明确，会先让你选择。</p><div class="qs-path-row"><input id="qs-path" value="' + esc(state.path) + '" placeholder="数据目录或 .zip / .tar.gz 路径"><button type="button" id="qs-pick" class="secondary">选择目录</button></div>' + registered + renderProbe() + summary + '</section>';
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
    const needsClassDependentNoise = unresolved.includes("config:requires_class_dependent_noise");
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
      fields.unshift('<label>方法噪声率<input class="qs-required-input" data-input-key="noise_rate_prior" type="number" min="0" max="1" step="0.01" value="' + esc(state.methodInputs.noise_rate_prior || "") + '"></label>');
    }
    const inputs = fields.length ? '<div class="qs-feedback qs-warning"><strong>补充方法输入</strong><div class="qs-required-inputs">' + fields.join("") + '</div><button type="button" id="qs-apply-inputs" class="secondary">应用并重新检查</button></div>' : "";
    const noisePrompt = needsNoiseSelection ? '<div class="qs-feedback qs-warning">' + (needsClassDependentNoise ? '当前实现需要类别相关噪声。' : '当前配置需要带噪训练标签。') + '<button type="button" id="qs-back-to-noise" class="secondary">返回第 2 步设置标签噪声</button></div>' : "";
    const externalPrompt = needsExternalLabels ? '<div class="qs-feedback qs-warning">当前 CAL 配置使用外部标签文件。<button type="button" id="qs-open-paper-resources" class="secondary">查看论文外部数据</button></div>' : "";
    const manualLabels = {clean_train_labels:"可核实的干净训练标签", dataset_noise_rate:"数据集原始噪声率", noise_manifest:"与当前样本对应的噪声记录", method_requirements:"方法兼容性资料"};
    const manualNote = manual.length ? '<div class="qs-feedback qs-warning">还缺少不能凭空生成的资料：' + esc(Array.from(new Set(manual.map(function (name) { return manualLabels[name] || (name.startsWith("pretrained:") ? "预训练模型文件" : name.startsWith("请在") ? name : "方法专用输入"); }))).join("、")) + '。请在数据集或论文配置中核对。</div>' : "";
    const details = unresolved.length ? [] : (plan.details || []);
    const problem = plan.status === "ready" || unresolved.length ? "" : '<div class="qs-feedback qs-warning"><strong>运行前还需处理</strong><p>' + esc(plan.summary) + '</p>' + details.map(function (detail) { return '<p>' + esc(detail) + '</p>'; }).join("") + '</div>';
    const transitionNote = plan.status === "ready" && state.selectedPaperId === "loss-correction" && ["symmetric", "pairflip", "binary_asymmetric_rcn"].includes(state.noiseSelection.key) ? '<p class="qs-feedback">原论文配置的转移矩阵不适用于当前噪声；已按第 2 步的类型、比例和当前类别数生成对应矩阵，原 Recipe 未改动。</p>' : "";
    return '<section class="qs-step"><h3>4. 运行计划</h3>' + problem + noisePrompt + externalPrompt + inputs + manualNote + transitionNote + renderPlanParameters() + '</section>';
  }

  function reviewSignature() {
    return JSON.stringify([state.plan?.plan_id, state.plan?.status, state.plan?.command, state.plan?.dry_run_command,
      state.planSchema?.fields.map(function (field) { return [field.path, field.value]; }),
      state.parameterDraft, state.methodInputs, state.dataset?.alias]);
  }
  function reviewTitle(review) {
    return ({saving:"正在准备参数审查…", validating:"正在启动检查环境并审查参数…", rehearsing:"参数检查通过，正在预演…", ready:"参数检查和预演均已通过", started:"训练已启动", error:"检查未通过"})[review?.phase] || "请返回运行计划，点击下一步进行检查";
  }
  function reviewLines(review) {
    return review?.lines || [];
  }
  function reviewInformation(review, ready) {
    return [(ready || review?.phase === "started") ? review?.summary : "", review?.locations].filter(Boolean).join("\n\n");
  }
  function updateReview() {
    const center = panel?.querySelector(".qs-review-center");
    if (!center) return;
    const review = state.review;
    const ready = review?.phase === "ready" && review.signature === reviewSignature();
    const text = function (selector, value) {
      const node = center.querySelector(selector);
      if (node.textContent !== value) node.textContent = value;
      return node;
    };
    center.querySelector(".qs-spinner").hidden = !["saving", "validating", "rehearsing"].includes(review?.phase);
    text("[data-review-title]", reviewTitle(review));
    const information = reviewInformation(review, ready);
    text("[data-review-locations]", information).hidden = !information;
    text("[data-review-error]", review?.error || "").hidden = review?.phase !== "error";
    const lines = reviewLines(review);
    center.querySelector("[data-review-output]").hidden = !lines.length;
    text("[data-review-output] pre", lines.join("\n"));
    center.querySelector("#qs-run").disabled = !ready;
  }
  function renderReview() {
    const review = state.review;
    const ready = review?.phase === "ready" && review.signature === reviewSignature();
    const busy = ["saving", "validating", "rehearsing"].includes(review?.phase);
    const lines = reviewLines(review);
    const information = reviewInformation(review, ready);
    return '<section class="qs-step"><h3>5. 检查与确认</h3><div class="qs-review-center" aria-live="polite"><span class="qs-spinner" aria-hidden="true"' + (busy ? '' : ' hidden') + '></span><h4 data-review-title>' + esc(reviewTitle(review)) + '</h4><pre data-review-locations' + (information ? '' : ' hidden') + '>' + esc(information) + '</pre><pre data-review-error class="qs-review-errors" role="alert"' + (review?.phase === "error" ? '' : ' hidden') + '>' + esc(review?.error) + '</pre><details data-review-output' + (lines.length ? '' : ' hidden') + '><summary>完整 CLI 输出</summary><pre>' + esc(lines.join("\n")) + '</pre></details><button type="button" id="qs-run" class="primary"' + (ready ? '' : ' disabled') + '>确认并开始训练</button></div></section>';
  }

  function parameterValue(field) {
    return Object.prototype.hasOwnProperty.call(state.parameterDraft, field.path)
      ? state.parameterDraft[field.path] : field.value;
  }
  function trainingStageName(path) {
    return ({
      "t_revision.stage1.epochs":"初始分类器",
      "t_revision.classifier_initialization.epochs":"校正分类器",
      "t_revision.revision.epochs":"联合修正",
      "posterior_stage.epochs":"初始模型",
      "final_stage.epochs":"最终模型",
      "pretraining_stage.epochs":"特征预训练",
      "transition_stage.epochs":"噪声矩阵估计",
      "ensemble_stage.epochs":"集成训练",
      "dividemix.warmup.epochs":"双模型预热",
      "dividemix.training.epochs":"协同训练",
      "upm.stage1.epochs":"初始分类器",
      "upm.main.epochs":"软标签分类器",
      "dld.diffusion.epochs":"扩散训练",
      "warmup.epochs":"预热",
      "transition.epochs":"噪声矩阵估计",
      "lend.training.epochs":"主模型",
      "trainer.epochs":"主模型"
    })[path] || "";
  }
  function trainingBudgetLabel(field) {
    const stage = trainingStageName(field?.path);
    return field?.label || (stage ? field.path + "（" + stage + "训练轮数）" : field?.path || "");
  }
  function preprocessingExplanation(value) {
    return ({
      standard: "先将像素缩放到 0–1，再按颜色通道减去均值并除以标准差。",
      tensor_only: "将像素缩放到 0–1；不减均值、不除以标准差，此模式不支持随机数据增强。",
      gce2018: "从每张图片的各像素位置减去训练集在对应位置的平均值；不除以标准差。"
    })[value] || "使用当前配置指定的输入预处理方案。";
  }
  function renderParameterField(field, segmentable) {
    segmentable = Boolean(segmentable && field.editable && field.display_group !== "restricted");
    const value = parameterValue(field);
    const currentTrainCount = Number(state.dataset?.training_pool_size ?? state.dataset?.train_size);
    const split = field.split_reference && Number.isInteger(currentTrainCount) && currentTrainCount > 0;
    const selectedDataset = field.path === "data.name";
    const readonly = !field.editable || selectedDataset;
    const label = selectedDataset ? "当前数据集" : trainingStageName(field.path) ? trainingBudgetLabel(field) : field.label || field.presentation_label || field.path;
    const choices = selectedDataset ? null : field.choices;
    let control;
    if (field.kind === "boolean") {
      control = '<input type="checkbox" data-qs-param="' + esc(field.path) + '"' + (value ? ' checked' : '') + (readonly ? ' disabled' : '') + '>';
    } else if (field.rows) {
      control = '<textarea hidden data-qs-param="' + esc(field.path) + '">' + esc(typeof value === 'string' ? value : JSON.stringify(value)) + '</textarea>' + window.parameterListEditor.render(field, value, readonly);
    } else if (field.kind === "list" || field.kind === "object") {
      control = '<textarea data-qs-param="' + esc(field.path) + '"' + (readonly ? ' disabled' : '') + '>' + esc(JSON.stringify(value)) + '</textarea>';
    } else if (choices) {
      const current = String(value ?? "");
      const options = choices.some(function (option) { return option.value === current; })
        ? choices : [{value:current, label:"当前配置含未支持选项", unsupported:true}, ...choices];
      control = '<select data-qs-param="' + esc(field.path) + '"' + (readonly ? ' disabled' : '') + '>' + window.parameterChoiceHelp.options(options, current) + '</select>';
    } else {
      const shown = selectedDataset ? state.dataset.alias : value == null ? "" : value;
      control = '<input type="' + (field.kind === "number" ? "number" : "text") + '" value="' + esc(shown) + '" data-qs-param="' + esc(field.path) + '"' + (split ? ' min="0" max="' + esc(currentTrainCount - 1) + '" step="1"' : '') + (readonly ? ' disabled' : '') + '>';
    }
    const noteText = (field.note || '') + (segmentable ? '勾选后在上方分段表中编辑。' : readonly && !selectedDataset ? '此项由训练流程固定，不能在网页修改。' : '');
    if (split) control = '<div class="qs-split-value">' + control + '<span>/ ' + esc(currentTrainCount) + '</span></div>';
    const paperSplit = field.split_reference ? '<small>原论文配置：' + esc(field.split_reference.count) + ' / ' + esc(field.split_reference.total) + (split && currentTrainCount !== field.split_reference.total ? '；初始值按此比例换算，可自行修改。' : '') + '</small>' : '';
    const preprocessingNote = choices && field.path === "data.preprocessing"
      ? '<small data-qs-preprocessing-note>' + esc(preprocessingExplanation(value)) + '</small>' : '';
    const note = (choices ? window.parameterChoiceHelp.render(choices, value) : '') + paperSplit + (preprocessingNote || (noteText ? '<small>' + esc(noteText) + '</small>' : '')) +
      (preprocessingNote && readonly ? '<small>此项由训练流程固定。</small>' : '');
    if (segmentable) {
      return '<div class="qs-parameter-field"><div class="qs-parameter-title"><strong>' + esc(label) + '</strong><label class="qs-segment-choice"><input type="checkbox" data-qs-segment-param="' + esc(field.path) + '">加入分段表</label></div>' + control + note + '</div>';
    }
    const wrapper = field.rows ? "div" : "label";
    return '<' + wrapper + ' class="qs-parameter-field"><span>' + esc(label) + '</span>' + control + note + '</' + wrapper + '>';
  }

  function splitAllocationText() {
    const total = Number(state.dataset?.training_pool_size ?? state.dataset?.train_size);
    const fields = state.planSchema?.fields || [];
    const validation = fields.find(function (field) { return ["data.num_val", "warmup.noisy_validation_size", "data.validation_size"].includes(field.path); });
    if (!validation || !Number.isInteger(total) || total <= 0) return "";
    const heldOut = Number(parameterValue(validation));
    const trustedField = fields.find(function (field) { return field.path === "data.num_clean"; });
    const trusted = trustedField ? Number(parameterValue(trustedField)) : 0;
    if (![heldOut, trusted].every(function (value) { return Number.isInteger(value) && value >= 0; }) || heldOut + trusted >= total) {
      return "划分数量不合法：必须为非负整数，并至少留下一个训练样本。";
    }
    return "原始训练池 " + total + "：训练 " + (total - heldOut - trusted) + "，验证 " + heldOut + (trustedField ? "，干净参考 " + trusted : "") + "。独立测试集保持不变。";
  }
  function scheduleGroups(fields) {
    const groups = [];
    fields.filter(function (field) { return field.editable && field.path.endsWith(".name") && (field.path === "scheduler.name" || field.path.includes(".scheduler.")); }).forEach(function (nameField) {
      const prefix = nameField.path.slice(0, -5);
      groups.push({prefix:prefix, fields:fields.filter(function (field) { return field.editable && field.path.startsWith(prefix + "."); })});
    });
    return groups;
  }
  function renderTrainingTimeline(fields) {
    const stages = {
      t_revision: [
        ["初始分类器", "用带噪标签训练，供噪声矩阵估计使用", "t_revision.stage1.epochs"],
        ["估计噪声矩阵", "由模型预测估计", null],
        ["校正分类器", "固定噪声矩阵，使用校正损失训练", "t_revision.classifier_initialization.epochs"],
        ["联合修正", "同时更新分类器与噪声矩阵修正量", "t_revision.revision.epochs"],
      ],
      dual_t: [
        ["初始模型", "学习带噪标签", "posterior_stage.epochs"],
        ["估计噪声矩阵", "计算转移关系", null],
        ["最终模型", "使用估计矩阵训练", "final_stage.epochs"],
      ],
      pcse: [
        ["准备特征模型", "训练或加载模型", "pretraining_stage.epochs"],
        ["估计噪声矩阵", "学习转移关系", "transition_stage.epochs"],
        ["提取特征与统计", "建立校正信息", null],
        ["集成训练", "组合阶段结果", "ensemble_stage.epochs"],
      ],
      dividemix: [
        ["双模型预热", "分别训练两个模型", "dividemix.warmup.epochs"],
        ["协同训练", "划分样本并互相指导", "dividemix.training.epochs"],
      ],
    }[state.planSchema?.method];
    if (!stages || stages.filter(function (stage) { return stage[2] && fields.some(function (field) { return field.path === stage[2]; }); }).length < 2) return "";
    const cards = stages.map(function (stage, index) {
      const field = fields.find(function (item) { return item.path === stage[2]; });
      const mode = state.planSchema.method === "pcse" && stage[2] === "pretraining_stage.epochs"
        ? fields.find(function (item) { return item.path === "pretraining_stage.mode"; }) : null;
      const count = field && (!mode || parameterValue(mode) === "train") ? Number(parameterValue(field)) : null;
      const duration = Number.isFinite(count) && count > 0 ? count + " 轮" : "自动";
      return '<li><span class="qs-stage-number">' + (index + 1) + '</span><strong>' + esc(trainingStageName(stage[2]) || stage[0]) + '</strong><span class="qs-stage-duration">' + esc(duration) + '</span><small>' + esc(stage[1]) + '</small></li>';
    }).join("");
    return '<section class="qs-stage-timeline" aria-label="训练时间轴"><h4>训练时间轴</h4><ol>' + cards + '</ol></section>';
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
  function scheduleBudgetField(group, fields) {
    const stage = group.prefix === "scheduler" ? "" : group.prefix.split(".scheduler")[0] + ".";
    if (group.fields.some(function (field) { return field.path === group.prefix + ".step_milestones"; })) {
      const steps = fields.find(function (field) { return field.path === "trainer.max_steps"; });
      if (Number(scheduleValue(steps)) > 0) return steps;
    }
    const paths = stage
      ? [stage + "epochs", stage + "training.epochs", stage + "trainer.epochs", "trainer.epochs"]
      : [state.planSchema?.method + ".training.epochs", "trainer.epochs"];
    for (const path of paths) {
      const field = fields.find(function (item) { return item.path === path; });
      if (field) return field;
    }
    return null;
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
  function renderScheduleControl(group, allFields, compact = false) {
    const lookup = new Map(group.fields.map(function (field) { return [field.path, field]; }));
    const prefix = group.prefix;
    const nameField = lookup.get(prefix + ".name");
    const name = String(scheduleValue(nameField) || "none");
    const milestoneField = lookup.get(prefix + ".step_milestones") || lookup.get(prefix + ".milestones");
    const milestones = scheduleMilestones(milestoneField);
    const stepBased = Boolean(milestoneField && milestoneField.path.endsWith("step_milestones"));
    const enabled = scheduleEnabled(group);
    const baseField = scheduleBaseField(prefix, allFields);
    const baseRate = Number(scheduleValue(baseField));
    const gammaField = lookup.get(prefix + ".gamma");
    const gamma = Number(scheduleValue(gammaField) ?? 0.1);
    const explicitField = lookup.get(prefix + ".lr_values");
    const explicitRates = scheduleExplicitRates(explicitField);
    const canSegment = Boolean(baseField?.editable && (explicitField?.editable || gammaField?.editable));
    const budgetField = scheduleBudgetField(group, allFields);
    const budget = Number(scheduleValue(budgetField)) || Math.max(1, ...milestones);
    const stageName = trainingStageName(budgetField?.path);
    const component = ({classifier:"分类器", transition:"转移矩阵", direction:"方向网络", noise:"噪声网络"})[prefix.split(".scheduler.")[1]];
    const shortTitle = stageName ? stageName + (component ? " " + component : "") : prefix === "scheduler" ? "主模型" : prefix.replace(/\.scheduler(\.|$)/, " ").replaceAll("_", " ");
    const title = shortTitle + "学习率阶段";
    const budgetSummary = budgetField ? '<p class="helper" data-qs-budget="' + esc(budgetField.path) + '">' + esc(trainingBudgetLabel(budgetField)) + '：' + esc(scheduleValue(budgetField)) + '</p>' : '';
    const modeLabel = !enabled ? "固定学习率" : name === "multistep" ? "按节点变化" : name === "cosine" ? "平滑下降" : "学习率变化";
    const budgetStatus = budgetField ? " / " + scheduleValue(budgetField) + (budgetField.path.endsWith("max_steps") ? " 步" : " 轮") : "";
    const available = [];
    if (lookup.has(prefix + ".milestones")) available.push("multistep");
    if (lookup.has(prefix + ".t_max") || name === "cosine" || available.length === 0) available.push("cosine");
    if (lookup.has(prefix + ".start_epoch") && name !== "linear_decay") available.push("linear_after");
    if (name === "linear_decay") available.push("linear_decay");
    if (name !== "none" && !available.includes(name)) available.push(name);
    const allowedModes = nameField.choices ? available.filter(function (item) { return nameField.choices.some(function (option) { return option.value === item; }); }) : available;
    const mode = stepBased ? '<p class="helper">按累计更新步数在指定节点切换学习率。</p>' : '<label class="qs-schedule-mode">变化方式<select data-qs-param="' + esc(nameField.path) + '">' + allowedModes.map(function (item) { return '<option value="' + esc(item) + '"' + (item === name ? ' selected' : '') + '>' + esc(({none:"不变化",multistep:explicitField ? "按节点设置学习率" : "到节点后按倍率调整",cosine:"平滑下降",linear_after:"指定轮次后线性下降",linear_decay:"线性下降",step:"固定间隔衰减"})[item] || "当前变化方式") + '</option>'; }).join("") + '</select></label>';
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
    const toggle = '<label class="qs-schedule-switch"><input type="checkbox" data-qs-schedule-toggle="' + esc(prefix) + '"' + (enabled ? ' checked' : '') + '>动态学习率</label>';
    return (compact ? '<div class="qs-schedule qs-schedule-compact" data-schedule="' + esc(prefix) + '"><div class="qs-schedule-compact-head"><button type="button" class="qs-schedule-expand" data-qs-schedule-expand="' + esc(prefix) + '" aria-expanded="' + Boolean(state.expandedSchedules[prefix]) + '"><span class="qs-schedule-arrow" aria-hidden="true">' + (state.expandedSchedules[prefix] ? '▾' : '▸') + '</span><strong>' + esc(shortTitle) + ' 学习率</strong><span class="qs-schedule-status">' + esc(modeLabel + budgetStatus) + '</span></button>' + toggle + '</div><div class="qs-schedule-body"' + (state.expandedSchedules[prefix] ? '' : ' hidden') + '>' : '<section class="qs-schedule" data-schedule="' + esc(prefix) + '"><div class="qs-schedule-head"><strong>' + esc(title) + '</strong>' + toggle + '</div>') +
      budgetSummary + (enabled ? mode +
        (!showNodes && baseField?.editable ? '<div class="qs-parameter-grid">' + renderParameterField(baseField, false) + '</div>' : '') +
        (showNodes ? '<div class="qs-schedule-track"><span>0</span><div class="qs-schedule-bar">' + ticks + '</div><span>' + esc(budget) + '</span></div>' : '') +
        (showNodes && milestoneField ? segmentTable + '<button type="button" class="secondary" data-qs-schedule-add="' + esc(prefix) + '">添加节点</button>' : '<p class="helper">' + (name === "cosine" ? '学习率从起点平滑下降到终点，不使用突变节点。' : '当前方式不使用额外节点。') + '</p>') +
        (gammaField && !explicitField ? '<p class="helper">当前默认倍率：' + esc(gamma) + '；各节点共用此倍率。</p>' : '') +
        (otherFields.length ? '<div class="qs-parameter-grid">' + otherFields.map(function (field) { return renderParameterField(field, false); }).join("") + '</div>' : '') :
        '<p class="helper">学习率保持本阶段的起始值；训练阶段仍会执行。</p>') + (compact ? '</div></div>' : '</section>');
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
    const movedPaths = new Set();
    for (const group of schedules) {
      if (!scheduleEnabled(group)) continue;
      const name = scheduleValue(group.fields.find(function (field) { return field.path === group.prefix + ".name"; }));
      const hasNodes = name === "multistep" || group.fields.some(function (field) { return field.path === group.prefix + ".step_milestones"; });
      const base = scheduleBaseField(group.prefix, fields);
      if (!hasNodes && base?.editable) movedPaths.add(base.path);
    }
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
    const regular = fields.filter(function (field) { return !unusedBudgetPaths.has(field.path) && !movedPaths.has(field.path) && !schedulePaths.has(field.path) && !(segmentablePaths.has(field.path) && state.segmentedPaths[field.path] !== false); });
    const defaults = regular.filter(function (field) { return field.display_group === "default"; });
    const advanced = regular.filter(function (field) { return field.display_group === "advanced"; });
    const restricted = fields.filter(function (field) { return field.display_group === "restricted"; });
    const changed = Object.keys(state.parameterDraft).length > 0;
    const timeline = renderTrainingTimeline(fields);
    const schedulePlan = schedules.length ? '<div class="qs-training-plan"><h4>模型训练计划 · 学习率设置</h4>' + (timeline ? '<p class="helper">各阶段独立设置；关闭变化后仍会训练。</p>' : '') + schedules.map(function (group) { return renderScheduleControl(group, fields, Boolean(timeline)); }).join("") + '</div>' : '';
    return '<div class="qs-parameter-panel">' + timeline + schedulePlan + '<h4>可修改的训练参数</h4><p class="helper" data-qs-split-summary>' + esc(splitAllocationText()) + '</p><div class="qs-parameter-grid">' + defaults.map(function (field) { return renderParameterField(field, segmentablePaths.has(field.path)); }).join("") + '</div>' +
      (advanced.length ? '<details class="qs-parameter-advanced"><summary>高级参数（' + advanced.length + '）</summary><div class="qs-parameter-grid">' + advanced.map(function (field) { return renderParameterField(field, segmentablePaths.has(field.path)); }).join("") + '</div></details>' : '') +
      (restricted.length ? '<details class="qs-parameter-advanced"><summary>禁止在 Web 修改的专属参数（' + restricted.length + '）</summary><p class="helper">这些值由配置契约固定；此处仅供查看。</p><div class="qs-parameter-grid">' + restricted.map(function (field) { return renderParameterField(field, false); }).join("") + '</div></details>' : '') +
      (changed ? '<p class="qs-feedback qs-warning">参数已修改。预演或运行前会保存为本次自定义配置并重新检查；不会改动论文原始 Recipe。</p>' : '') + '</div>';
  }

  function quickStartSteps() {
    const steps = [{id:"dataset", label:"数据集", render:renderDataset}];
    if (state.dataset && state.noise) steps.push({id:"noise", label:"标签噪声", render:renderNoise});
    if (state.dataset && state.noise && state.methods.length) steps.push({id:"methods", label:"方法兼容性", render:renderMethods});
    if (state.plan) {
      steps.push({id:"plan", label:"运行计划", render:renderPlan});
      steps.push({id:"review", label:"检查与确认", render:renderReview});
    }
    return steps;
  }

  function quickStartStepIndex(id, steps) {
    return (steps || quickStartSteps()).findIndex(function (step) { return step.id === id; });
  }

  function quickStartCanAdvance(step) {
    if (!step) return false;
    if (step.id === "dataset") return Boolean(state.dataset || state.path.trim());
    if (step.id === "noise") return Boolean(state.noise) && (!state.noise.requires_confirmation || state.labelsConfirmed);
    if (step.id === "methods") return Boolean(state.selectedPaperId && state.plan);
    if (step.id === "plan") return Boolean(state.plan?.status === "ready" && state.planSchema && !state.parameterError);
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
    const nextDisabled = state.loading || (active.id !== "dataset" && state.currentStep >= steps.length - 1) || !quickStartCanAdvance(active);
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
    path && path.addEventListener("input", function () {
      changeDatasetPath(this.value, false);
      panel.querySelector(".qs-dataset-card")?.remove();
      const next = document.getElementById("qs-next");
      if (next) next.disabled = state.loading || !state.path;
    });
    path && path.addEventListener("change", render);
    document.getElementById("qs-pick")?.addEventListener("click", pickPath);
    document.getElementById("qs-registered")?.addEventListener("change", function () {
      if (this.value) { changeDatasetPath(this.value); registerPath(state.path, null, true); }
    });
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
    document.getElementById("qs-confirm-labels")?.addEventListener("click", async function () { state.labelsConfirmed = true; render(); await loadMethods(); if (state.pendingPaper) await continuePendingPaper(); else recheckSelectedPaper(); });
    document.getElementById("qs-methods-panel")?.addEventListener("toggle", function () { state.methodListExpanded = this.open; });
    document.getElementById("qs-prev")?.addEventListener("click", function () { state.currentStep = Math.max(0, state.currentStep - 1); render(); });
    document.getElementById("qs-next")?.addEventListener("click", advanceQuickStart);
    panel.querySelectorAll(".qs-carousel-dot").forEach(function (button) { button.addEventListener("click", function () { const target = Number(this.dataset.qsStep); if (!this.disabled) { state.currentStep = target; if (quickStartSteps()[target].id === "review") reviewPlan(); else render(); } }); });
    panel.querySelectorAll(".qs-method-card").forEach(function (button) { button.addEventListener("click", function () { const item = state.methods.find(function (entry) { return entry.paper_id === button.dataset.paper; }); if (!item || !["ready", "needs_input"].includes(item.status)) return; state.selectedPaperId = this.dataset.paper; state.methodInputs = {}; state.planSchema = null; state.parameterDraft = {}; state.segmentedPaths = {}; state.scheduleStash = {}; state.methodListExpanded = false; render(); if (state.seedTouched || !item.recipe_id) { buildPlan(); return; } request("/api/config-schema?recipe=" + encodeURIComponent(item.recipe_id), undefined, "正在读取配置…").then(function (schema) { if (state.selectedPaperId !== item.paper_id) return; const seed = (schema.fields || []).find(function (field) { return field.path === "seed"; }); if (seed) state.noiseSelection.seed = Number(seed.value); render(); buildPlan(); }).catch(function (error) { state.error = String(error.message || error); render(); }); }); });
    panel.querySelectorAll(".qs-required-input").forEach(function (input) { input.addEventListener("input", function () { state.methodInputs[this.dataset.inputKey] = this.value; }); });
    document.getElementById("qs-back-to-noise")?.addEventListener("click", function () {
      const index = quickStartStepIndex("noise");
      if (index >= 0) state.currentStep = index;
      render();
      window.scrollConsoleToTop?.();
    });
    document.getElementById("qs-open-paper-resources")?.addEventListener("click", function () {
      if (context?.openPaperDetails) context.openPaperDetails(state.selectedPaperId);
      else context?.switchModule?.("papers");
    });
    function markParameterEdited() {
      const note = panel.querySelector(".qs-parameter-panel > .qs-feedback");
      if (!note) panel.querySelector(".qs-parameter-panel")?.insertAdjacentHTML("beforeend", '<p class="qs-feedback qs-warning">参数已修改。预演或运行前会保存为本次自定义配置并重新检查；不会改动论文原始 Recipe。</p>');
    }
    panel.querySelectorAll("[data-qs-param]").forEach(function (input) { input.addEventListener(input.tagName === "SELECT" ? "change" : "input", function () {
      window.parameterChoiceHelp.update(input);
      const path = this.dataset.qsParam;
      state.parameterDraft[path] = this.type === "checkbox" ? this.checked : this.value;
      const splitSummary = panel.querySelector("[data-qs-split-summary]");
      if (splitSummary) splitSummary.textContent = splitAllocationText();
      if (path === "data.preprocessing") {
        const explanation = panel.querySelector("[data-qs-preprocessing-note]");
        if (explanation) explanation.textContent = preprocessingExplanation(this.value);
      }
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
    window.parameterListEditor.bind(panel, function (path) { return Array.from(panel.querySelectorAll('[data-qs-param]')).find(function (input) { return input.dataset.qsParam === path; }); });
    const budgetPaths = new Set(scheduleGroups(state.planSchema?.fields || []).map(function (group) { return scheduleBudgetField(group, state.planSchema.fields)?.path; }).filter(Boolean));
    for (const field of state.planSchema?.fields || []) {
      if (field.path.endsWith(".epochs")) budgetPaths.add(field.path);
    }
    budgetPaths.add("trainer.max_steps");
    panel.querySelectorAll("[data-qs-param]").forEach(function (input) {
      if (budgetPaths.has(input.dataset.qsParam)) input.addEventListener("change", render);
    });
    panel.querySelectorAll("[data-qs-segment-param]").forEach(function (checkbox) { checkbox.addEventListener("change", function () {
      state.segmentedPaths[this.dataset.qsSegmentParam] = this.checked;
      render();
    }); });
    panel.querySelectorAll("[data-qs-schedule-expand]").forEach(function (button) { button.addEventListener("click", function () {
      const prefix = this.dataset.qsScheduleExpand;
      state.expandedSchedules[prefix] = !state.expandedSchedules[prefix];
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
      if (this.checked) state.expandedSchedules[this.dataset.qsScheduleToggle] = true;
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
    document.getElementById("qs-run")?.addEventListener("click", function () { executePlan(); });
    if (renderedStep !== state.currentStep) {
      renderedStep = state.currentStep;
      window.scrollConsoleToTop?.();
    }
  }
  function loadPlanSchema(plan) {
    const selected = state.methods.find(function (item) { return item.paper_id === plan.paper_id; });
    const source = plan.recipe_id ? "recipe=" + encodeURIComponent(plan.recipe_id)
      : plan.generated_config_path ? "path=" + encodeURIComponent(plan.generated_config_path) + "&recipe_hint=" + encodeURIComponent(selected?.recipe_id || "") : "";
    if (!source) return Promise.resolve();
    return request("/api/config-schema?" + source, undefined, "正在读取参数…").then(function (schema) {
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
  function parameterPatches(forReview = false) {
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
          if (String(raw).trim() === "" || !Number.isFinite(value) || (field.number_type === "integer" && !Number.isInteger(value))) {
            if (forReview) value = raw;
            else throw new Error(field.path + " 必须填写" + (field.number_type === "integer" ? "整数" : "数字"));
          }
        }
      } else if (field.kind === "list" || field.kind === "object") {
        if (field.nullable && raw == null) value = null;
        else {
          try { value = JSON.parse(raw); } catch (_) { if (forReview) value = raw; else throw new Error(field.path + " 必须填写合法 JSON"); }
          if (!forReview && field.kind === "list" && !Array.isArray(value)) throw new Error(field.path + " 必须是 JSON 列表");
          if (!forReview && field.kind === "object" && (value === null || Array.isArray(value) || typeof value !== "object")) throw new Error(field.path + " 必须是 JSON 对象");
        }
      }
      if (JSON.stringify(value) !== JSON.stringify(field.value)) patches.push({path:field.path, value:value});
    }
    return patches;
  }
  function commandForSavedConfig(command, plan, path) {
    const original = plan.recipe_id ? "--recipe " + plan.recipe_id : "--config " + plan.generated_config_path;
    if (!command || !command.includes(original)) throw new Error("运行计划的配置来源已变化，请重新生成计划");
    return command.replace(original, '--config "' + path + '"');
  }
  function commandForRun(command, plan, saved) {
    // Replace any plan output directory so preview and training share one location.
    return commandForSavedConfig(command, plan, saved.path)
      .replace(/\s+--output-dir(?:=|\s+)(?:"[^"]*"|'[^']*'|\S+)/g, "")
      + ' --output-dir "' + saved.output_dir + '"';
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
  async function reviewPlan() {
    const plan = state.plan;
    if (!plan || plan.status !== "ready" || !state.planSchema || state.parameterError) return;
    const token = ++reviewToken;
    const signature = reviewSignature();
    const current = function () { return token === reviewToken && signature === reviewSignature() && quickStartSteps()[state.currentStep]?.id === "review"; };
    state.review = {phase:"saving", signature, lines:[]};
    render();
    async function jsonRequest(url, body) {
      const response = await fetch(url, body ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)} : undefined);
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "检查请求失败");
      return result;
    }
    async function cli(command, phase) {
      if (!current()) return false;
      state.review.phase = phase; state.review.lines = []; updateReview();
      let job = await jsonRequest("/api/run", {command});
      while (current()) {
        state.review.lines = job.lines || [];
        if (state.review.lines.includes("Quick Start phase: rehearsing")) state.review.phase = "rehearsing";
        updateReview();
        if (!job.running && job.output_complete) {
          if (job.returncode !== 0) throw new Error((job.lines || []).join("\n") || job.error || "CLI 检查失败");
          return true;
        }
        await new Promise(function (resolve) { window.setTimeout(resolve, 350); });
        if (!current()) return false;
        job = await jsonRequest("/api/jobs/" + encodeURIComponent(job.id));
      }
      return false;
    }
    try {
      const patches = parameterPatches(true);
      const selected = state.methods.find(function (item) { return item.paper_id === plan.paper_id; });
      const source = plan.recipe_id ? {recipe:plan.recipe_id} : {source_path:plan.generated_config_path, recipe_hint:selected?.recipe_id || ""};
      const saved = await jsonRequest("/api/quick-start/review-config", {...source, patches, dataset_alias:state.dataset.alias});
      if (!current()) return;
      state.review.locations = "本次训练保存位置（开始训练时创建）\n运行目录：" + saved.output_dir + "\n配置文件：" + saved.config_path + "\n训练日志、指标与模型权重保存在同一运行目录。";
      updateReview();
      if (!await cli(commandForRun(plan.dry_run_command, plan, saved) + " --review", "validating")) return;
      state.review.phase = "ready";
      state.review.snapshot = {...source, content:saved.content, dataset_alias:state.dataset.alias, run_id:saved.run_id};
      state.review.summary = currentRunSummary(plan, patches);
    } catch (error) {
      if (!current()) return;
      state.review.phase = "error";
      state.review.error = String(error.message || error);
    }
    if (current()) updateReview();
  }
  async function executePlan() {
    const review = state.review;
    if (review?.phase !== "ready" || review.signature !== reviewSignature()) return;
    review.phase = "saving"; updateReview();
    try {
      const response = await fetch("/api/quick-start/training-config", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(review.snapshot)});
      const saved = await response.json();
      if (!response.ok) throw new Error(saved.error || "保存训练配置失败");
      if (state.review !== review || review.signature !== reviewSignature()) return;
      review.command = commandForRun(state.plan.command, state.plan, saved);
      review.locations = "本次训练保存位置\n运行目录：" + saved.output_dir + "\n配置文件：" + saved.config_path + "\n训练日志、指标与模型权重保存在同一运行目录。";
      review.phase = "started";
      context.setRequest({command:review.command}, review.command);
      context.execute();
      render();
    } catch (error) {
      review.phase = "error"; review.error = String(error.message || error); updateReview();
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
    state.methodListExpanded = false;
    state.methodInputs = {};
    state.plan = null;
    state.planSchema = null;
    state.parameterDraft = {};
    state.segmentedPaths = {};
    state.scheduleStash = {};
    render();
    loadMethods().then(function () { recheckSelectedPaper({navigate:false}); });
  }
  function saveNativeRate() {
    const value = Number(document.getElementById("qs-native-rate-value")?.value);
    const source = document.getElementById("qs-native-rate-source")?.value.trim() || "";
    if (!Number.isFinite(value) || value < 0 || value > 1 || document.getElementById("qs-native-rate-value")?.value === "" || !source) {
      state.error = "请填写 0 到 1 之间的原始噪声率，并说明数字的依据"; render(); return;
    }
    const rate = {status:state.nativeRateMode, value:value, provenance:source};
    post("/api/datasets/" + encodeURIComponent(state.dataset.alias) + "/declarations", {declarations:{noise_rate:rate}}, "正在保存数据集噪声率…")
      .then(function () { state.error = ""; return loadNoise(); })
      .catch(function (error) { state.error = String(error.message || error); render(); });
  }
  async function advanceQuickStart() {
    if (state.loading) return;
    const steps = quickStartSteps();
    const active = steps[state.currentStep];
    if (!quickStartCanAdvance(active)) return;
    if (active.id === "dataset" && !state.dataset) {
      await probeAndRegister();
      return;
    }
    state.currentStep = Math.min(steps.length - 1, state.currentStep + 1);
    if (steps[state.currentStep].id === "review") reviewPlan();
    else render();
  }
  function probeAndRegister() {
    state.path = document.getElementById("qs-path")?.value || state.path;
    if (!state.path) { state.error = "请先选择数据集路径"; render(); return; }
    state.error = "";
    return post("/api/quick-start/probe", {path:state.path}, "正在识别数据集格式…").then(function (payload) { state.probe = payload; render(); if (payload.status === "detected" || payload.status === "already_registered") return registerPath(state.path, null, payload.status === "already_registered"); }).catch(function (error) { state.error = String(error.message || error); render(); });
  }
  function registerPath(path, adapter, alreadyRegistered) {
    const message = "正在读取数据集…";
    return post("/api/quick-start/register", {path:path, adapter:adapter || ""}, message).then(function (payload) { if (payload.kind !== "dataset") { state.probe = payload; render(); return; } state.dataset = payload.dataset; state.path = payload.dataset.path; state.error = ""; state.selectedPaperId = ""; state.plan = null; state.planSchema = null; state.parameterDraft = {}; state.segmentedPaths = {}; state.scheduleStash = {}; state.methodListExpanded = false; state.currentStep = 0; return loadRegistered().then(function () { render(); return loadNoise(); }).then(function () { if (state.noise) state.currentStep = quickStartStepIndex("noise"); render(); }); }).catch(function (error) { state.error = String(error.message || error); render(); return null; });
  }
  function loadNoise() {
    return request("/api/quick-start/noises?dataset=" + encodeURIComponent(state.dataset.alias), undefined, "正在读取标签与噪声…").then(function (payload) { state.noise = payload; state.nativeRateMode = "unknown"; state.labelsConfirmed = !payload.requires_confirmation; if (["native", "noisy"].includes(payload.dataset_state)) state.noiseSelection = {kind:"native", key:"native", rate:null, seed:state.noiseSelection.seed ?? 1}; else state.noiseSelection = {kind:"clean", key:"clean", rate:null, seed:state.noiseSelection.seed ?? 1}; if (state.selectedPaperId) { state.plan = null; state.planSchema = null; state.methodInputs = {}; } render(); return payload.requires_confirmation ? null : loadMethods().then(recheckSelectedPaper); });
  }
  function refreshNoiseFacts() {
    if (!state.dataset) return Promise.resolve();
    return request("/api/quick-start/noises?dataset=" + encodeURIComponent(state.dataset.alias), undefined, "正在更新标签…").then(function (payload) {
      if (JSON.stringify(payload) === JSON.stringify(state.noise)) return;
      state.noise = payload;
      state.nativeRateMode = "unknown";
      state.labelsConfirmed = !payload.requires_confirmation;
      state.methods = [];
      state.plan = null;
      state.planSchema = null;
      state.methodInputs = {};
      state.parameterDraft = {};
      state.segmentedPaths = {};
      state.scheduleStash = {};
      if (["native", "noisy"].includes(payload.dataset_state)) state.noiseSelection = {kind:"native", key:"native", rate:null, seed:state.noiseSelection.seed ?? 1};
      else if (payload.dataset_state === "unknown" || state.noiseSelection.kind === "native") state.noiseSelection = {kind:"clean", key:"clean", rate:null, seed:state.noiseSelection.seed ?? 1};
      render();
      return payload.requires_confirmation ? null : loadMethods().then(recheckSelectedPaper);
    }).catch(function (error) { state.error = String(error.message || error); render(); });
  }
  function loadMethods() {
    if (!state.dataset || !state.noiseSelection) return Promise.resolve();
    return post("/api/quick-start/methods", {dataset:state.dataset.alias, noise:state.noiseSelection}, "正在匹配论文方法…").then(function (payload) { state.methods = payload.methods || []; render(); }).catch(function (error) { state.error = String(error.message || error); render(); });
  }
  function buildPlan(options) {
    if (!state.dataset || !state.selectedPaperId) return;
    const navigate = options?.navigate !== false;
    const userInputs = {};
    Object.keys(state.methodInputs).forEach(function (key) { const raw = state.methodInputs[key]; if (raw === "") return; try { userInputs[key] = JSON.parse(raw); } catch (_error) { userInputs[key] = raw; } });
    post("/api/quick-start/plan", {dataset:state.dataset.alias, paper_id:state.selectedPaperId, noise:state.noiseSelection, user_inputs:userInputs}, "正在核对…").then(function (payload) { state.plan = payload; state.planSchema = null; state.parameterDraft = {}; state.segmentedPaths = {}; state.scheduleStash = {}; state.parameterError = ""; const index = quickStartStepIndex("plan"); if (navigate && index >= 0) state.currentStep = index; render(); return loadPlanSchema(payload); }).catch(function (error) { state.error = String(error.message || error); render(); });
  }
  function recheckSelectedPaper(options) {
    if (!state.selectedPaperId || state.error) return;
    const selected = state.methods.find(function (item) { return item.paper_id === state.selectedPaperId; });
    if (selected && ["ready", "needs_input"].includes(selected.status)) {
      buildPlan(options);
    } else {
      state.error = "所选论文不适用当前噪声设置。";
      if (options?.navigate !== false) state.currentStep = quickStartStepIndex("methods");
      render();
    }
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
    renderedStep = null;
    if (!state.registered.length) loadRegistered().then(render);
    render();
    if (state.dataset) refreshNoiseFacts();
  }
  window.quickStartController = {mount:mount, render:render, onModuleEnter:mount, openPaper:openPaper, showCompatibilityGuide:function () { render(); return state.noise ? Promise.resolve() : loadNoise(); }};
}());
