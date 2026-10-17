(function () {
  "use strict";

  let detailMode = false;
  let latestJob = null;

  function text(value) { return value == null ? "-" : String(value); }

  function simpleSummary(training) {
    if (!training) return "";
    const labels = {starting: "正在启动", running: "训练中", completed: "已完成", failed: "训练失败", cancelled: "已取消"};
    const parts = [labels[training.state] || training.state];
    if (training.stage) parts.push("阶段：" + training.stage);
    if (training.completed_epoch != null) {
      const total = training.total_epochs == null ? "" : "/" + training.total_epochs;
      parts.push("最近完成 epoch：" + training.completed_epoch + total);
    } else if (training.total_epochs != null) {
      parts.push("总 epoch：" + training.total_epochs);
    }
    if (training.warning) parts.push("状态读取提示：" + training.warning);
    return parts.join(" · ");
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
    }).join("\n") || "尚无已 flush 的 metrics.jsonl 事件。";
    applyToggle(detailMode);
  }

  function applyToggle(detail) {
    const raw = document.getElementById("output");
    const metrics = document.getElementById("training-metrics");
    // The complete console log is permanent; detail mode only controls metrics.
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
