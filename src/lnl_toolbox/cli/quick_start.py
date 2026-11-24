from __future__ import annotations

"""Terminal Quick Start using the Web planner and the existing paper editor."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from uuid import uuid4

from lnl_toolbox.catalog import find_project_root, recipe_by_id, resolve_config_paths
from lnl_toolbox.cli import PromptCancelled, PromptSession
from lnl_toolbox.cli.i18n import save_language, text
from lnl_toolbox.cli.paper_editor import PaperEditor
from lnl_toolbox.quickstart.models import QuickStartDatasetSummary, QuickStartNoiseSelection
from lnl_toolbox.quickstart.service import QuickStartService


class _GoBack(Exception):
    """Leave the current Quick Start prompt and return to its parent."""


def _terminal_session(*, input=None, output=None, write=None):
    from prompt_toolkit import PromptSession as ToolkitSession
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings

    keys = KeyBindings()
    terminal = ToolkitSession(input=input, output=output, key_bindings=keys)

    # Preserve ordinary cursor movement when editing a path or another value.
    @keys.add("left", filter=Condition(lambda: not terminal.default_buffer.text))
    @keys.add("escape")
    def back(event):
        event.app.exit(exception=_GoBack())

    def read(prompt):
        return terminal.prompt(text("[←/Esc 返回] ", "[←/Esc Back] ") + prompt)

    return PromptSession(read=read, write=write)


class QuickStartEditor(PaperEditor):
    _SCHEDULE_FIELDS = {
        "multistep": ("milestones", "gamma", "lr_values"),
        "cosine": ("t_max", "eta_min"),
        "linear_after": ("start_epoch", "end_epoch", "decay_start"),
        "linear_decay": ("start_epoch", "end_epoch", "initial_lr", "final_lr", "beta1_before", "beta1_after"),
        "step": ("step_size", "gamma"),
    }

    def __init__(self, method, service, dataset, noise, inputs):
        super().__init__(method, recipe=recipe_by_id(method.recipe_id), guided=True)
        self.dataset = dataset.alias
        self.pool_size = dataset.training_pool_size or dataset.train_size
        self.review_payload = service.review_payload(self.recipe.id, self.dataset, noise, inputs, [])
        overrides = self.review_payload["_quick_start_overrides"]
        self.config = service._apply_overrides(self.baseline, overrides)
        self.source_schema = self.rules._config_schema(
            self.recipe.id, quickstart_train_count=self.review_payload["_quickstart_train_count"],
            quickstart_dataset=self.dataset, quickstart_data_service=service.data_service,
        )
        self.draft = {}
        self.segmented = {}
        self.restricted = False

    def _protected(self, path):
        # These controls are owned by the preceding steps of the wizard.
        protected = {
            "noise", "seed", "data.name", "data.root", "data.path", "data.noise_path",
            "data.labels_path", "data.annotation_root", "data.seed", "local_dataset",
        }
        return any(path == key or path.startswith(key + ".") for key in protected)

    def schema(self):
        schema = deepcopy(self.source_schema)
        fields = {field["path"]: field for field in schema["fields"]}
        for path, field in fields.items():
            field["value"] = deepcopy(self.draft.get(path, field["value"]))
        for path, field in fields.items():
            if field.get("model_name") and path.endswith((".variant", ".weights")):
                parent = path.rsplit(".", 1)[0]
                field["model_name"] = fields.get(parent + ".name", {}).get("value", field["model_name"])
                field["model_variant"] = fields.get(parent + ".variant", {}).get("value", field.get("model_variant"))
                field["choices"] = self.rules._parameter_choices(field, schema["method"])
        return schema

    def _visible(self):
        return {field["path"]: field for field in self.schema()["fields"]
                if field.get("visible", True) and not self._protected(field["path"])}

    def schedule_groups(self, fields=None):
        fields = self._visible() if fields is None else fields
        groups = {}
        for path, name in fields.items():
            if not (name["editable"] and path.endswith(".name")
                    and (path == "scheduler.name" or ".scheduler." in path)):
                continue
            prefix = path[:-5]
            stage, _, owner = prefix.partition("scheduler")
            base = fields.get(stage + "optimizer" + owner + ".lr")
            if base is None:
                base = next((field for key, field in fields.items() if key.startswith(stage + "optimizer.")
                             and key.endswith("_lr")), None)
            controls = {key[len(prefix)+1:]: f for key, f in fields.items() if key.startswith(prefix + ".")}
            node = controls.get("step_milestones") or controls.get("milestones")
            stepped = "step_milestones" in controls
            mode = name["value"] or "none"
            enabled = bool(node["value"]) if stepped else mode != "none"
            can_segment = bool(base and base["editable"] and node and
                               (stepped or mode == "multistep") and
                               any(controls.get(key, {}).get("editable") for key in ("gamma", "lr_values")))
            groups[prefix] = {
                "path": prefix, "fields": controls, "base": base, "node": node, "stepped": stepped,
                "value": mode, "enabled": enabled, "can_segment": can_segment,
                "segmented": can_segment and self.segmented.get(base["path"], True),
                "choices": [choice["value"] for choice in name.get("choices") or []],
            }
        return groups

    @classmethod
    def active_schedule_fields(cls, group):
        if group["stepped"]:
            return {"step_milestones", "gamma", "lr_values"}
        active = set(cls._SCHEDULE_FIELDS.get(group["value"], ()))
        if group["enabled"] and "beta1_before" in group["fields"]:
            active.update(("start_epoch", "beta1_before", "beta1_after"))
        return active

    def fields(self, fields=None):
        fields = self._visible() if fields is None else fields
        groups = self.schedule_groups(fields)
        managed = {field["path"] for group in groups.values() for field in group["fields"].values()}
        for group in groups.values():
            base = group["base"]
            if base and base["editable"] and group["enabled"] and (
                group["segmented"] or not (group["stepped"] or group["value"] == "multistep")
            ):
                managed.add(base["path"])
        steps = fields.get("trainer.max_steps", {}).get("value") or 0
        budget = "trainer.epochs" if self.source_schema["method"] == "l2rw" and steps > 0 else "trainer.max_steps"
        regular = []
        for path, field in fields.items():
            if path in managed or path == budget:
                continue
            if path.endswith(".early_stopping_monitor"):
                switch = fields.get(path.removesuffix("_monitor"))
                if switch is not None and not switch["value"]:
                    continue
            regular.append(field)
        return regular

    def plan_entries(self):
        visible = self._visible()
        fields = self.fields(visible)
        groups = self.schedule_groups(visible)
        regular = {field["path"]: field for field in fields}
        model_cards = {}
        entries = []
        for field in fields:
            parent, _, key = field["path"].rpartition(".")
            if (parent == "model" or parent.endswith(".model")) and key in {"name", "variant", "weights"}:
                if parent not in model_cards:
                    model_cards[parent] = {**field, "path": parent, "children": []}
                    entries.append(("group", model_cards[parent]))
                model_cards[parent]["children"].append(field)
            elif field["path"].endswith(".early_stopping") and field["path"] + "_monitor" in visible:
                monitor = regular.get(field["path"] + "_monitor")
                entries.append(("group", {**field, "children": [field, monitor] if monitor else [field]}))
            elif not (field["path"].endswith(".early_stopping_monitor")
                      and field["path"].removesuffix("_monitor") in regular):
                entries.append(("field", field))
        if self.restricted:
            return [(kind, field) for kind, field in entries if field.get("display_group") == "restricted"]
        stages = self.timeline(visible)
        timeline = [("info", {"path": text("训练时间轴", "Training timeline"), "value": " → ".join(stages)})] if stages else []
        return timeline + [("schedule", group) for group in groups.values()] + [
            (kind, field) for kind, field in entries
            if field.get("display_group") == "default" or self.advanced and field.get("display_group") == "advanced"
        ]

    def timeline(self, fields=None):
        stages = {
            "t_revision": (("初始分类器", "Initial classifier", "t_revision.stage1.epochs"),
                           ("估计噪声矩阵", "Estimate transition matrix", None),
                           ("校正分类器", "Correct classifier", "t_revision.classifier_initialization.epochs"),
                           ("联合修正", "Joint correction", "t_revision.revision.epochs")),
            "dual_t": (("初始模型", "Initial model", "posterior_stage.epochs"),
                       ("估计噪声矩阵", "Estimate transition matrix", None),
                       ("最终模型", "Final model", "final_stage.epochs")),
            "pcse": (("特征预训练", "Feature pretraining", "pretraining_stage.epochs"),
                     ("估计噪声矩阵", "Estimate transition matrix", "transition_stage.epochs"),
                     ("提取特征与统计", "Extract features and statistics", None),
                     ("集成训练", "Ensemble training", "ensemble_stage.epochs")),
            "dividemix": (("双模型预热", "Warm up both models", "dividemix.warmup.epochs"),
                          ("协同训练", "Co-training", "dividemix.training.epochs")),
        }.get(self.source_schema["method"], ())
        fields = self._visible() if fields is None else fields
        if sum(path in fields for _, _, path in stages) < 2:
            return []
        return [
            text(zh, en) + (text(f"（{fields[path]['value']} 轮）", f" ({fields[path]['value']} epochs)") if path in fields else "")
            for zh, en, path in stages
        ]

    def split_summary(self):
        fields = self._visible()
        total = self.pool_size
        validation = next((fields[key]["value"] for key in
                           ("data.num_val", "warmup.noisy_validation_size", "data.validation_size") if key in fields), None)
        if validation is None or not total:
            return ""
        trusted = fields.get("data.num_clean", {}).get("value", 0)
        if min(validation, trusted) < 0 or validation + trusted >= total:
            return text("划分不合法：至少保留一个训练样本。", "Invalid split: keep at least one training sample.")
        return text(f"训练池 {total} · 训练 {total-validation-trusted} · 验证 {validation}",
                    f"Pool {total} · train {total-validation-trusted} · validation {validation}") + (
            text(f" · 干净参考 {trusted}", f" · trusted {trusted}") if "data.num_clean" in fields else ""
        )

    def schedule_budget(self, group):
        fields = self._visible()
        prefix = group["path"]
        if prefix == "scheduler" and self.source_schema["method"] == "dividemix":
            return sum(fields.get(path, {}).get("value", 0) for path in
                       ("dividemix.warmup.epochs", "dividemix.training.epochs"))
        steps = fields.get("trainer.max_steps", {}).get("value") or 0
        if group["stepped"] and steps > 0:
            return steps
        stage = prefix.partition("scheduler")[0]
        paths = [stage + suffix for suffix in ("epochs", "training.epochs", "trainer.epochs")] if stage else [
            self.source_schema["method"] + ".training.epochs"
        ]
        return next((fields[path]["value"] for path in [*paths, "trainer.epochs"] if path in fields), None)

    def changes(self, reference=None):
        original = {f["path"]: f["value"] for f in self.source_schema["fields"]}
        return [(path, original[path], value) for path, value in sorted(self.draft.items())
                if original[path] != value]

    def snapshot(self):
        return deepcopy((self.draft, self.schedule_stash, self.segmented))

    def restore(self, snapshot):
        self.draft, self.schedule_stash, self.segmented = snapshot

    def update(self, patches):
        fields = self._visible()
        draft = self.draft.copy()
        for path, value in patches:
            field = fields.get(path)
            if not field or not field["editable"] or field.get("display_group") == "restricted":
                raise ValueError(text(f"此参数不可修改：{path}", f"This parameter cannot be edited: {path}"))
            draft[path] = self.rules._coerce_patch_value(field, value)
            for linked in field.get("linked_fields", ()):
                if linked in fields and fields[linked]["editable"]:
                    draft[linked] = draft[path]
        self.draft = draft

    def review_patches(self):
        fields = self._visible()
        groups = self.schedule_groups(fields)
        inactive = {f["path"] for group in groups.values()
                    for suffix, f in group["fields"].items()
                    if suffix != "name" and suffix not in self.active_schedule_fields(group)}
        patches = []
        for source in self.source_schema["fields"]:
            path = source["path"]
            if path not in fields or path in inactive or not source["editable"]:
                continue
            switch = source["kind"] == "boolean" and path.endswith((".early_stopping", ".warmup_early_stopping"))
            if path not in self.draft and not switch:
                continue
            value = fields[path]["value"]
            if switch or source.get("virtual") or value != source["value"]:
                patches.append({"path": path, "value": value})
        return patches

    def review_config(self):
        config, _ = self.rules.prepare_config(
            {**self.review_payload, "patches": self.review_patches()}, for_review=True,
        )
        return config

    def set_schedule_mode(self, prefix, name):
        group = self.schedule_groups()[prefix]
        if name not in group["choices"] or group["stepped"]:
            raise ValueError(text("当前阶段不支持该调度模式。", "This schedule mode is unavailable for this stage."))
        patches = [(group["fields"]["name"]["path"], name)]
        active = self.active_schedule_fields({**group, "value": name, "enabled": name != "none"})
        for suffix, field in group["fields"].items():
            if suffix in active and field.get("virtual") and field["editable"] and field["path"] not in self.draft:
                patches.append((field["path"], field["value"]))
        self.update(patches)

    def toggle_schedule(self, prefix):
        group = self.schedule_groups()[prefix]
        if "none" not in group["choices"]:
            raise ValueError(text("此阶段的调度固定。", "This stage has a fixed schedule."))
        fields = group["fields"]
        if group["enabled"]:
            stash = {suffix: fields[suffix]["value"] for suffix in ("name", "milestones", "step_milestones", "lr_values") if suffix in fields}
            self.update([(fields[key]["path"], value) for key, value in
                         (("name", "none"), ("step_milestones", []), ("lr_values", None))
                         if key in fields and fields[key]["editable"]])
            self.schedule_stash[prefix] = stash
        else:
            stash = self.schedule_stash.get(prefix, {})
            if not group["stepped"]:
                name = stash.get("name") or next(choice for choice in group["choices"] if choice != "none")
                self.set_schedule_mode(prefix, name)
            original = {f["path"]: f["value"] for f in self.source_schema["fields"]}
            patches = [(fields[key]["path"], stash.get(key, original[fields[key]["path"]]))
                       for key in ("step_milestones", "lr_values") if key in fields and fields[key]["editable"]]
            if group["stepped"] and "name" in stash:
                patches.append((fields["name"]["path"], stash["name"]))
            self.update(patches)

    def edit(self, *, input=None, output=None):
        while True:
            workspace = self.workspace(input=input, output=output)
            result = workspace.app.run(pre_run=workspace.start)
            if isinstance(result, tuple) and result[0] == "language":
                save_language(result[1])
                continue
            return result == "review"


def _dataset(session, service):
    reports = [report for report in service.data_service.list_datasets() if report.location]
    register = object()
    choice = session.choose(text("选择数据集", "Choose a dataset"), [
        *((f"{report.name} ({report.adapter})", report.name) for report in reports),
        (text("识别新的目录或压缩包", "Detect a new directory or archive"), register),
        (text("退出", "Exit"), None),
    ])
    if choice is None:
        return None
    if choice is not register:
        report = service.data_service.status(choice)
        if report.status != "ready" or report.profile is None:
            report = service.data_service.inspect(choice)
        return service._summary(choice, service._require_ready_report(report, choice))
    path = session.path(text("本机数据目录或压缩包", "Local dataset directory or archive"), must_exist=True)
    if path is None:
        return False
    detected = service.register_and_inspect(str(path))
    if isinstance(detected, QuickStartDatasetSummary):
        return detected
    if detected.status == "ambiguous":
        adapter = session.choose(text("选择识别到的数据格式", "Choose a detected format"), [
            *((f"{item.adapter} · {item.reason}", item.adapter) for item in detected.candidates),
            (text("返回", "Back"), None),
        ])
        if adapter is None:
            return False
        detected = service.register_and_inspect(str(path), selected_adapter=adapter)
        if isinstance(detected, QuickStartDatasetSummary):
            return detected
    raise ValueError(text(f"无法识别数据集：{path}", f"Could not identify dataset: {path}"))


def _matrix(raw, classes):
    import math

    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != classes or any(
        not isinstance(row, list) or len(row) != classes or any(
            type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in row
        ) or not math.isclose(sum(row), 1, rel_tol=0, abs_tol=1e-8)
        for row in rows
    ):
        raise ValueError(text("转移矩阵必须与类别数一致，各项非负且每行和为 1。",
                              "Use a square matrix matching the class count, with nonnegative rows summing to 1."))
    return tuple(tuple(row) for row in rows)


def _noise(session, service, dataset):
    facts = service.noise_options(dataset.alias)
    state = facts["dataset_state"]
    if facts["requires_confirmation"] and not session.confirm(text(
        "标签干净性未知，是否按当前观测标签继续？", "Label cleanliness is unknown. Continue with observed labels?"
    )):
        return None, None
    options = facts["options"]
    key = session.choose(text("选择标签噪声", "Choose label noise"), [
        *((item["label"], item["key"]) for item in options), (text("上一步", "Back"), None),
    ])
    if key is None:
        return None, None
    seed = session.integer(text("实验随机种子", "Experiment seed"), default=1, minimum=0)
    if key == "native":
        prior = facts["noise_rate"]["value"]
        if prior is None:
            rate_status = session.choose(text("能否提供原始标签的噪声率？", "Can you provide the original noise rate?"), [
                (text("未知", "Unknown"), None), (text("准确值", "Known"), "known"),
                (text("估计值", "Estimated"), "estimated"),
            ])
            if rate_status:
                prior = session.floating(text("原始噪声率", "Original noise rate"), minimum=0, maximum=1)
                source = session.text(text("数值依据", "Source of the estimate"), required=True)
                service.data_service.update_declarations(dataset.alias, {
                    "noise_rate": {"status": rate_status, "value": prior, "provenance": source},
                })
        else:
            session.write(text(f"已登记原始噪声率 {prior}", f"Recorded original noise rate: {prior}"))
        return QuickStartNoiseSelection("native", key, seed=seed), prior
    if key == "clean":
        return QuickStartNoiseSelection("clean", key, seed=seed), None
    if state != "clean":
        raise ValueError(text("人工加噪需要已确认的干净标签。", "Synthetic noise requires confirmed clean labels."))
    option = next(item for item in options if item["key"] == key)
    if option.get("description"):
        session.write(option["description"])
    values = {}
    if key == "class_conditional":
        while True:
            try:
                raw = session.text(text("类别转移矩阵（JSON）", "Class transition matrix (JSON)"), required=True)
                values["transition_matrix"] = _matrix(raw, dataset.num_classes)
                break
            except ValueError as exc:
                session.write(str(exc))
    elif key == "external_torch":
        values["path"] = str(session.path(text("外部标签文件", "External labels file"), must_exist=True,
                                          file_only=True, required=True).resolve())
    elif key == "binary_asymmetric_rcn":
        while True:
            positive = session.floating(text("正类翻转率", "Positive flip rate"), default=.2, minimum=0, maximum=1)
            negative = session.floating(text("负类翻转率", "Negative flip rate"), default=.1, minimum=0, maximum=1)
            if positive + negative < 1:
                values.update(rho_positive=positive, rho_negative=negative)
                break
            session.write(text("两项之和必须小于 1。", "The sum must be below 1."))
    elif option.get("requires_rate"):
        values["rate"] = session.floating(text("人工噪声率", "Synthetic noise rate"), default=.2, minimum=0, maximum=1)
    return QuickStartNoiseSelection("synthetic", key, seed=seed, **values), None


def _method_inputs(session, option, prior):
    inputs = {}
    if prior is not None:
        inputs["noise_rate_prior"] = prior
    path_groups = dict(option.required_input_paths)
    for required in option.required_user_inputs:
        if required == "noise_rate_prior":
            if prior is None:
                inputs[required] = session.floating(text("方法使用的噪声率先验", "Method noise-rate prior"), minimum=0, maximum=1)
            continue
        paths = path_groups.get(required, ())
        if not paths or any(parts[0] == "noise" for parts in paths):
            session.write(text(f"当前选择还缺少条件：{required}。可返回调整标签噪声或更换方法。",
                               f"Missing requirement: {required}. Go back to change the noise or method."))
            return None
        for parts in paths:
            path = ".".join(parts)
            raw = session.text(path, required=True)
            try:
                inputs[path] = json.loads(raw)
            except json.JSONDecodeError:
                inputs[path] = raw
    return inputs


def _review(session, service, editor):
    config = resolve_config_paths(editor.review_config(), find_project_root())
    trainer = service.experiment_service
    runner = trainer.preflight(config, check_data=True)
    session.write(f"  {editor.paper.acronym} · {editor.dataset}")
    for field in runner.describe(config).fields:
        session.write(f"  {field.label}: {field.value}")
    output_dir = service.artifact_root / "runs" / uuid4().hex
    session.write(text(f"运行目录：{output_dir}", f"Run directory: {output_dir}"))
    decision = session.choose(text("检查通过。", "Validation passed."), [
        (text("返回运行计划", "Back to run plan"), "back"),
        (text("开始训练", "Start training"), "train"),
        (text("退出", "Exit"), "exit"),
    ])
    if decision != "train":
        return decision
    import yaml

    output_dir.mkdir(parents=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8",
    )
    try:
        result = trainer.run(config, output_dir, recipe=editor.recipe.id)
    except (OSError, TypeError, ValueError, RuntimeError, KeyError) as exc:
        session.write(text(f"训练失败，运行目录：{output_dir}", f"Training failed. Run directory: {output_dir}"))
        session.write(str(exc))
        return "failed"
    session.write(text(f"训练完成：{result}", f"Training completed: {result}"))
    return "done"


def _wizard(session, service):
    step = 1
    while True:
        titles = [text("数据集", "Dataset"), text("标签噪声", "Label noise"),
                  text("方法兼容性", "Method compatibility"), text("运行计划", "Run plan"),
                  text("检查与确认", "Review and confirm")]
        session.write(f"\n[{step}/5] {titles[step - 1]}")
        try:
            if step == 1:
                dataset = _dataset(session, service)
                if dataset is None:
                    return 0
                if dataset:
                    session.write(text(f"{dataset.alias} · {dataset.train_size} 训练样本 · {dataset.num_classes} 类",
                                       f"{dataset.alias} · {dataset.train_size} train samples · {dataset.num_classes} classes"))
                    step = 2
            elif step == 2:
                noise, prior = _noise(session, service, dataset)
                step = 3 if noise is not None else 1
            elif step == 3:
                choices = [(text("返回标签噪声", "Back to label noise"), None)]
                methods = {method.paper_id: method for method in service.method_options(dataset.alias, noise)}
                for method in methods.values():
                    label = f"{method.acronym} · {method.title}"
                    if method.status in {"ready", "needs_input"} and method.recipe_id:
                        status = (text("可用", "Compatible") if method.status == "ready"
                                  else text("需补充输入", "Inputs required"))
                        choices.append((f"{label} · {status}", method.paper_id))
                    else:
                        session.write(f"  {label} · {' / '.join(method.reasons) or method.status}")
                paper_id = session.choose(text("选择论文方法", "Choose a paper method"), choices)
                if paper_id is None:
                    step = 2
                    continue
                method = methods[paper_id]
                editor = None
                step = 4
            elif step == 4:
                if editor is None:
                    inputs = _method_inputs(session, method, prior)
                    if inputs is None:
                        step = 3
                        continue
                    plan = service.build_plan(dataset_alias=dataset.alias, noise_selection=noise,
                                              paper_id=method.paper_id, user_inputs=inputs)
                    if plan.status != "ready":
                        session.write("\n".join((plan.summary, *plan.details, *plan.required_user_inputs)))
                        step = 3
                        continue
                    editor = QuickStartEditor(method, service, dataset, noise, inputs)
                step = 5 if editor.edit() else 3
            else:
                result = _review(session, service, editor)
                if result != "back":
                    return 2 if result == "failed" else 0
                step = 4
        except _GoBack:
            step = max(1, step - 1)
        except (OSError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            session.write(text("错误：", "Error: ") + str(exc))
            if step == 5:
                step = 4
            elif step in {3, 4}:
                step -= 1
            try:
                if not session.confirm(text("返回重试？", "Go back and try again?"), default=True):
                    return 2
            except _GoBack:
                continue


def start():
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise ValueError(text("请在交互终端运行 lnl start。", "Run lnl start in an interactive terminal."))
    try:
        return _wizard(_terminal_session(), QuickStartService(artifact_root=find_project_root() / "artifacts" / "cli-quick-start"))
    except (PromptCancelled, EOFError, KeyboardInterrupt):
        return 130
