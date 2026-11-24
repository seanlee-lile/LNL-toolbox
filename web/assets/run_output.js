(function () {
  "use strict";

  let detailMode = false;
  let latestJob = null;

  function duration(seconds) {
    if (seconds < 60) return seconds.toFixed(1) + " 秒";
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return minutes + " 分钟";
    return Math.floor(minutes / 60) + " 小时 " + (minutes % 60) + " 分钟";
  }

  function simpleSummary(training) {
    if (!training) return "";
    const labels = {starting: "正在启动", running: "训练中", completed: "已完成", failed: "训练失败", cancelled: "已取消"};
    const stages = {
      warmup: "预热", pretraining: "预训练", stage1: "预训练",
      mentor_feedback: "反馈模型训练", mentor_teacher: "导师模型训练",
      posterior: "后验模型训练", transition: "转移矩阵估计",
      cwd_transition: "转移矩阵估计", transition_initialization: "初始化转移矩阵",
      statistics: "估计类别统计量", gda: "拟合类别分布", precorrection: "预校正标签",
      classifier_initialization: "分类器初始化", correction_training: "校正训练",
      correction: "校正训练", revision: "修正训练", diffusion: "扩散训练", ensemble: "集成校准",
      main: "正式训练", final: "正式训练", final_training: "正式训练",
      robust: "正式训练", cwd: "正式训练",
    };
    const stage = stages[training.stage];
    let heading = labels[training.state] || training.state;
    if (stage) heading = training.state === "running" ? stage : heading + " · " + stage;
    if (training.completed_epoch != null) {
      const total = training.total_epochs == null ? "" : " / " + training.total_epochs;
      heading += " · 第 " + training.completed_epoch + total + " 轮";
    }
    const parts = [heading];
    if (Number.isFinite(training.epoch_seconds)) parts.push("本轮耗时 " + duration(training.epoch_seconds));
    if (training.state === "running" && Number.isFinite(training.estimated_total_seconds)) {
      parts.push("预计本阶段总耗时 " + duration(training.estimated_total_seconds));
    }
    const splits = {
      noisy_validation: "带噪验证准确率", clean_validation: "干净验证准确率",
      validation: "验证准确率", test: "测试准确率",
    };
    if (Number.isFinite(training.accuracy) && splits[training.accuracy_split]) {
      parts.push(splits[training.accuracy_split] + " " + (training.accuracy * 100).toFixed(1) + "%");
    }
    if (training.warning) parts.push("状态读取提示 " + training.warning);
    return parts.join(" ｜ ");
  }

  function render(job) {
    latestJob = job;
    const panel = document.getElementById("training-output");
    const summary = document.getElementById("training-summary");
    const metrics = document.getElementById("training-metrics");
    if (!panel || !summary || !metrics) return;
    const training = job && job.training;
    const upper = document.getElementById("qs-training-output");
    if (upper) upper.hidden = !training;
    summary.classList.toggle("hidden", Boolean(upper && training));
    panel.classList.toggle("hidden", !training);
    if (!training) {
      const raw = document.getElementById("output");
      if (raw) raw.classList.remove("hidden");
      metrics.classList.add("hidden");
      return;
    }
    summary.textContent = simpleSummary(training);
    if (upper) {
      document.getElementById("qs-training-summary").textContent = simpleSummary(training);
      const bar = document.getElementById("qs-training-progress");
      const label = document.getElementById("qs-training-progress-label");
      const stepBased = Number(training.total_steps) > 0;
      const total = Number(stepBased ? training.total_steps : training.total_epochs);
      const completed = Number(stepBased ? training.completed_step : training.completed_epoch) || 0;
      if (training.state === "completed") {
        bar.value = 100;
        label.textContent = "已完成（100%）";
      } else if (total > 0) {
        const percent = Math.max(0, Math.min(100, completed / total * 100));
        bar.value = percent;
        label.textContent = completed + " / " + total + (stepBased ? " 步" : " 轮") + "（" + Math.floor(percent) + "%）";
      } else {
        if (["failed", "cancelled"].includes(training.state)) bar.value = 0;
        else bar.removeAttribute("value");
        label.textContent = training.state === "starting" ? "正在启动训练…" : "";
      }
    }
    metrics.textContent = (training.metrics || []).map(function (row) {
      return JSON.stringify(row);
    }).join("\n") || "尚无训练指标。";
    applyToggle(detailMode);
  }

  function applyToggle(detail) {
    const raw = document.getElementById("output");
    const metrics = document.getElementById("training-metrics");
    if (raw) raw.classList.remove("hidden");
    if (metrics) metrics.classList.toggle("hidden", !detail);
    const simple = document.getElementById("training-simple");
    const detailed = document.getElementById("training-detail");
    if (simple) simple.classList.toggle("primary", !detail);
    if (detailed) detailed.classList.toggle("primary", detail);
  }

  function toggle(detail) {
    detailMode = Boolean(detail);
    applyToggle(detailMode);
  }

  window.trainingOutput = {render: render, toggle: toggle, refresh:function () { render(latestJob); }};
}());
