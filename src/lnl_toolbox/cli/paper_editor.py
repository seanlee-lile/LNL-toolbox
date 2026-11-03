from __future__ import annotations

"""Paper summaries and an opt-in-by-TTY editor using the shared Web contracts."""

from copy import deepcopy
import json
from pathlib import Path
import sys

from lnl_toolbox.catalog import default_paper_config, load_papers, load_recipe_config
from lnl_toolbox.cli import PromptCancelled, PromptSession
from lnl_toolbox.cli.i18n import language, saved_language, save_language, text

SUMMARIES_EN = {
    "binary-risk": "Construct an unbiased binary risk using known class-dependent label-noise rates.",
    "importance-reweighting": "Estimate and apply importance weights under binary asymmetric random classification noise.",
    "loss-correction": "Correct the training loss with a known or estimated class-transition matrix.",
    "coteaching": "Two networks select small-loss examples and exchange them for peer updates.",
    "gce": "Use q to interpolate between cross entropy and mean absolute error.",
    "mentornet": "A learned Mentor network supplies curriculum weights for Student training.",
    "t-revision": "Initialize a pseudo-anchor transition estimate, then jointly revise the transition parameters and classifier using Reweight-R.",
    "apl": "Combine active and passive robust losses to balance learning and resistance to label noise.",
    "dual-t": "Factor the noise-transition matrix into two more easily estimated transitions.",
    "jocor": "Combine peer-prediction agreement with joint small-loss selection.",
    "pdl": "Represent instance-dependent label corruption with sample-dependent part transitions.",
    "cdr": "Restrict updates to critical parameters to reduce late-stage memorization of noisy labels.",
    "cnlcu": "Use two-network loss histories and uncertainty bounds for peer sample selection.",
    "cwd": "Recover a denoised risk from class statistics and train with the configured binary or multiclass protocol.",
    "pcse": "Recover clean per-class statistics from pretrained features and observed-label class statistics.",
    "dss": "Use class- and instance-debiasing evidence for noisy-label sample selection.",
    "fine": "Apply active forgetting and noise suppression to identify and handle suspicious examples.",
}

SUMMARIES_ZH = {
    "l2rw": "通过干净验证集上的元梯度计算训练样本权重。",
    "dividemix": "将逐样本损失拟合为混合分布，把疑似噪声样本作为无标签数据，进行双网络半监督训练。",
    "cal": "通过二阶校正风险训练处理实例依赖的标签噪声。",
    "upm": "为每个稳定索引的训练样本学习混淆概率，描述实例依赖的标签扰动。",
    "volminnet": "不依赖锚点，联合学习干净标签后验分类器与类别转移矩阵。",
    "lend": "使用当前批次特征构建在线图并稀释噪声标签，默认筛选稀释证据与观测标签一致的样本。",
    "mc-ldce": "提供 MC-LDCE 的类别统计与质心风险训练流程。",
    "ca2c": "提供非对称协同学习训练流程。",
    "dld": "构造弱增强和强增强视图的标签预校正，学习方向与随机标签扩散预测器。",
}


def resolve_paper(value):
    key = str(value).strip().casefold().replace("_", "-")
    matches = [p for p in load_papers() if key in {p.id.casefold(), p.acronym.casefold().replace("_", "-"), p.title.casefold()}]
    if len(matches) != 1:
        raise ValueError(text("论文名称不存在或不唯一：", "Unknown or ambiguous paper: ") + str(value))
    return matches[0]


def list_papers(output_format="human"):
    papers = load_papers()
    if output_format in {"json", "tsv"}:
        from lnl_toolbox.catalog import discover_recipes
        recipes = {r.id: r for r in discover_recipes(include_conditional=True)}
        rows = [{
            "id": p.id, "acronym": p.acronym, "title": p.title, "venue": p.venue,
            "year": p.year, "source_url": p.source_url,
            "profiles": sorted({c.profile for c in p.configs}),
            "fidelity": sorted({c.configuration_fidelity for c in p.configs}),
            "runners": sorted({recipes[c.recipe_id].runner for c in p.configs}),
            "recommended": next((c.recipe_id for c in p.configs if c.profile == "reproduction"), None),
        } for p in papers]
        if output_format == "json":
            print(json.dumps(rows, ensure_ascii=False))
        else:
            print("id\tacronym\ttitle\tvenue\tyear\tsource_url")
            for row in rows:
                print("\t".join(str(row[k]) for k in ("id", "acronym", "title", "venue", "year", "source_url")))
        return 0
    for index, paper in enumerate(papers, 1):
        print(f"[{index}/{len(papers)}] {paper.id}  {paper.acronym}")
        print(f"  {paper.title}\n  {paper.venue} {paper.year}\n  {paper.source_url}\n")
    print(text("操作：", "Next steps:") + " lnl show <paper> | lnl validate <source> | lnl run <source>")
    return 0


def summary(paper, write=print):
    write(f"{paper.acronym}: {paper.title}")
    write(f"{paper.venue} {paper.year}\n{paper.source_url}")
    write(SUMMARIES_ZH.get(paper.id, paper.summary) if language() == "zh" else SUMMARIES_EN.get(paper.id, paper.summary))


def show_paper(paper, *, interactive=True, session=None):
    summary(paper, session.write if session else print)
    if not interactive or (session is None and not (sys.stdin.isatty() and sys.stdout.isatty())):
        print(text("编辑参数：", "Edit parameters:") + f" lnl show {paper.id}")
        return 0
    session = session or PromptSession()
    try:
        if saved_language() is None and not __import__('os').environ.get("LNL_LANG"):
            # An explicit --lang already sets the session language; this menu
            # also allows persisting it. No prompt is used by scripts or Web.
            chosen = session.choose("语言 / Language", [("中文", "zh"), ("English", "en")], default=language())
            save_language(chosen)
        return PaperEditor(paper, session).run()
    except PromptCancelled:
        session.write(text("已取消。", "Cancelled."))
        return 130


class PaperEditor:
    def __init__(self, paper, session):
        from lnl_toolbox import parameters
        self.rules = parameters
        self.paper, self.session = paper, session
        _, self.recipe = default_paper_config(paper)
        self.baseline = load_recipe_config(self.recipe)
        self.config = deepcopy(self.baseline)
        self.dataset = ""
        self.advanced = False
        self.schedule_stash = {}

    def schema(self):
        return self.rules.schema_from_payload({
            "config": deepcopy(self.config), "recipe": self.recipe.id,
            "path": self.rules._display_path(self.recipe.config_path), "runner": self.recipe.runner,
            "method": self.recipe.method,
        })

    def schedules(self):
        groups = []
        def visit(value, prefix=""):
            if not isinstance(value, dict):
                return
            if "scheduler" in prefix.split("."):
                if "name" in value or "step_milestones" in value or (self.recipe.runner == "cwd" and prefix == "scheduler"):
                    before, _, after = prefix.partition("scheduler")
                    path = before + "optimizer" + after + ".lr"
                    if not self.rules._config_path_value(self.config, path)[0]:
                        owner = self.rules._config_path_value(self.config, before + "optimizer")[1]
                        if isinstance(owner, dict):
                            path = next((before + "optimizer." + key for key in owner if key.endswith("_lr")), path)
                    if self.rules._config_path_value(self.config, path)[0]:
                        groups.append((prefix, path, value))
            for key, child in value.items():
                if isinstance(child, dict):
                    visit(child, f"{prefix}.{key}" if prefix else key)
        visit(self.config)
        return groups

    @staticmethod
    def enabled(schedule):
        return bool(schedule["step_milestones"]) if "step_milestones" in schedule else schedule.get("name", "multistep") != "none"

    def fields(self):
        fields = self.schema()["fields"]
        moved = {base for _, base, schedule in self.schedules() if self.enabled(schedule)}
        return [f for f in fields if f.get("visible", True)
                and f.get("display_group") != "restricted" and f.get("editable")
                and (self.advanced or f.get("display_group") == "default")
                and f["path"] not in moved
                and not ("scheduler" in f["path"].split(".") and f["path"].split(".")[-1] in {"name", "milestones", "step_milestones", "lr_values", "gamma"})
                and not (f["path"] == "trainer.epochs" and self.config.get("trainer", {}).get("max_steps", 0) > 0)]

    def note(self, field):
        if language() == "zh":
            return field.get("note", "")
        from lnl_toolbox.cli.parameter_help import english_note
        return english_note(field["path"], self.recipe.method)

    def display(self):
        self.session.write("\n" + text("默认参数", "Default parameters"))
        for index, field in enumerate(self.fields(), 1):
            reference = field.get("split_reference")
            suffix = ""
            if reference:
                from lnl_toolbox.data.registry import training_pool_size
                profile = self.config.get("local_dataset", {}).get("profile")
                counts = (profile or {}).get("sample_counts_by_split", {})
                total = training_pool_size(self.config.get("data", {}).get("name", ""), counts) if counts else reference["total"]
                suffix = f" / {total}; " + text("原论文", "Paper") + f" {reference['count']} / {reference['total']}"
            self.session.write(f"  {index}. {field['path']} = {json.dumps(field['value'], ensure_ascii=False)}{suffix}")
            self.session.write("     " + self.note(field))
        for prefix, base, schedule in self.schedules():
            if self.enabled(schedule):
                self.session.write(f"\n{prefix}: {schedule.get('name', 'multistep')}")
                nodes = schedule.get("step_milestones", schedule.get("milestones", []))
                initial = float(self.rules._config_path_value(self.config, base)[1])
                rates = schedule.get("lr_values") or [initial * float(schedule.get("gamma", .1)) ** (i + 1) for i in range(len(nodes))]
                self.session.write("  " + text("区间起点", "Segment start") + " | 0 | " + " | ".join(map(str, nodes)))
                self.session.write(f"  {base} | {initial} | " + " | ".join(map(str, rates)))

    def prepare(self, config):
        import yaml
        # Same coercion, seed synchronization, locked-field checks and typed
        # validation as Web, without a temporary file or HTTP request.
        result, _ = self.rules.prepare_config({
            "recipe": self.recipe.id, "content": yaml.safe_dump(config, allow_unicode=True),
            "acknowledge_paper_impact": True,
        })
        return result

    def update(self, patches):
        candidate = deepcopy(self.config)
        fields = self.rules._field_map(candidate, self.schema()["method"])
        for path, value in patches:
            field = fields.get(path)
            if not field or not field["editable"] or self.rules._internal_seed_path(path):
                raise ValueError(text("此参数禁止修改：", "This parameter is not editable: ") + path)
            if isinstance(value, str) and field["kind"] in {"number", "boolean", "list"}:
                from lnl_toolbox.core.config_overrides import parse_override_value
                value = parse_override_value(value)
            self.rules._set_config_path(candidate, path, self.rules._coerce_patch_value(field, value))
        self.config = self.prepare(candidate)

    def edit(self):
        fields = self.fields()
        if not fields:
            return
        field = self.session.choose(text("选择参数", "Choose a parameter"), [(f["path"], f) for f in fields])
        self.session.write(self.note(field))
        self.update([(field["path"], self.read_value(field))])

    def read_value(self, field):
        rows = field.get("rows")
        if rows:
            value = deepcopy(field["value"])
            index = self.session.choose(text("选择条目", "Choose an item"), [(str(i + 1), i) for i in range(len(rows))])
            child = self.session.choose(text("选择参数", "Choose a parameter"), [(f["key"], f) for f in rows[index] if f.get("editable", True)])
            child = {**child, "path": child.get("path", child["key"]), "value": value[index][child["key"]]}
            value[index][child["key"]] = self.rules._coerce_patch_value(child, self.read_value(child))
            return value
        choices = field.get("choices")
        if field["kind"] == "boolean":
            value = self.session.choose(field["path"], [("true", True), ("false", False)], default=field["value"])
        elif choices:
            value = self.session.choose(field["path"], [(c["value"], c["value"]) for c in choices], default=field["value"])
        else:
            raw = self.session.text(field["path"], default=json.dumps(field["value"]) if field["kind"] in {"list", "number"} else str(field["value"] or ""))
            if field["kind"] in {"list", "number"}:
                from lnl_toolbox.core.config_overrides import parse_override_value
                value = parse_override_value(raw)
            else:
                value = raw
        return value

    def edit_schedule(self):
        groups = self.schedules()
        if not groups:
            self.session.write(text("此配置没有可编辑的学习率阶段。", "This configuration has no editable learning-rate schedule."))
            return
        prefix, base, schedule = self.session.choose(text("学习率阶段", "Learning-rate schedules"), [(g[0], g) for g in groups])
        enabled = self.session.confirm(text("启用学习率阶段", "Enable learning-rate scheduling"), default=self.enabled(schedule))
        candidate = deepcopy(self.config)
        target = self.rules._config_path_value(candidate, prefix)[1]
        if not enabled:
            self.schedule_stash[prefix] = deepcopy(target)
            target.pop("lr_values", None)
            if "step_milestones" in target:
                target["step_milestones"] = []
            target["name"] = "none"
        else:
            if prefix in self.schedule_stash:
                target.update(deepcopy(self.schedule_stash[prefix]))
            choices = self.rules._parameter_choices({"path": prefix + ".name", "kind": "text", "visible": True, "editable": True, "value": target.get("name", "multistep")}, self.recipe.method)
            mode = self.session.choose(text("变化方式", "Schedule mode"), [(c["value"], c["value"]) for c in (choices or [{"value":"multistep"}]) if c["value"] != "none"], default=target.get("name", "multistep"))
            target["name"] = mode
            if mode == "multistep" or "step_milestones" in target:
                key = "step_milestones" if "step_milestones" in target else "milestones"
                nodes = json.loads(self.session.text(text("节点列表（JSON 整数数组）", "Milestones (JSON integer array)"), default=json.dumps(target.get(key, []))))
                if not isinstance(nodes, list) or any(type(n) is not int or n <= 0 for n in nodes) or nodes != sorted(set(nodes)):
                    raise ValueError(text("节点必须为递增、不重复的正整数。", "Milestones must be distinct positive integers in increasing order."))
                initial = float(self.session.text(base, default=str(self.rules._config_path_value(candidate, base)[1])))
                self.rules._set_config_path(candidate, base, initial)
                target[key] = nodes
                old_rates = target.get("lr_values") or []
                target["lr_values"] = [float(self.session.text(f"{base} @ {node}", default=str(old_rates[i] if i < len(old_rates) else initial * target.get("gamma", .1) ** (i + 1)))) for i, node in enumerate(nodes)]
            else:
                self.session.write(text("其余曲线参数可在高级参数中编辑。", "Edit the remaining curve controls under advanced parameters."))
        self.config = self.prepare(candidate)

    def select_dataset(self):
        from lnl_toolbox.training.data_service import DEFAULT_DATA_SERVICE
        from lnl_toolbox.quickstart.templates import adapt_split_counts
        reports = [r for r in DEFAULT_DATA_SERVICE.list_datasets() if r.location]
        if not reports:
            self.session.write(text("请先使用 lnl data register 登记数据集。", "Register a dataset with lnl data register first."))
            return
        alias = self.session.choose(text("选择已登记数据集", "Choose a registered dataset"), [(r.name, r.name) for r in reports])
        candidate = DEFAULT_DATA_SERVICE.apply(self.config, alias)
        report = DEFAULT_DATA_SERVICE.status(alias)
        if report.profile:
            from lnl_toolbox.data.registry import training_pool_size
            total = training_pool_size(report.adapter, dict(report.profile.sample_counts_by_split))
            if total:
                adapt_split_counts(candidate, self.baseline, total)
        self.config = self.prepare(candidate)
        self.dataset = alias

    def verify(self):
        from lnl_toolbox.catalog import find_project_root, resolve_config_paths
        from lnl_toolbox.training.service import ExperimentService
        candidate = self.prepare(self.config)
        candidate = resolve_config_paths(candidate, find_project_root())
        service = ExperimentService()
        runner = service.preflight(candidate, check_data=True)
        self.session.write(text("验证通过。", "Validation passed."))
        for field in runner.describe(candidate).fields:
            self.session.write(f"  {field.label}: {field.value}")
        return service, candidate

    def save(self):
        import yaml
        path = self.session.path(text("另存 YAML", "Save YAML as"), default=Path("configs/experiment") / f"{self.paper.id}-custom.yaml", required=True)
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("YAML path must end in .yaml or .yml")
        if path.resolve() in self.rules._builtin_config_paths():
            raise ValueError(text("不能覆盖论文内置配置。", "Built-in paper configurations cannot be overwritten."))
        if path.exists() and not self.session.confirm(text("覆盖已有文件？", "Overwrite the existing file?"), default=False):
            return
        candidate = self.prepare(self.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True), encoding="utf-8")
        self.session.write(text("已保存：", "Saved: ") + str(path))

    def run(self):
        while True:
            self.display()
            action = self.session.choose(text("操作", "Action"), [
                (text("修改参数", "Edit a parameter"), "edit"),
                (text("展开/收起高级参数", "Show/hide advanced parameters"), "advanced"),
                (text("学习率阶段表", "Learning-rate schedule table"), "schedule"),
                (text("选择数据集", "Choose a dataset"), "dataset"),
                (text("另存 YAML", "Save YAML"), "save"),
                (text("完整验证", "Validate"), "validate"),
                (text("启动训练", "Start training"), "train"),
                ("语言 / Language", "language"),
                (text("退出", "Exit"), "exit"),
            ], default="exit")
            try:
                if action == "exit":
                    return 0
                if action == "advanced":
                    self.advanced = not self.advanced
                elif action == "edit":
                    self.edit()
                elif action == "schedule":
                    self.edit_schedule()
                elif action == "dataset":
                    self.select_dataset()
                elif action == "save":
                    self.save()
                elif action in {"validate", "train"}:
                    service, candidate = self.verify()
                    if action == "train" and self.session.confirm(text("确认启动训练？", "Start training now?"), default=False):
                        result = service.run(candidate, recipe=self.recipe.id)
                        self.session.write(text("运行完成：", "Run completed: ") + str(result))
                elif action == "language":
                    save_language(self.session.choose("语言 / Language", [("中文", "zh"), ("English", "en")], default=language()))
            except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
                self.session.write(text("操作失败（原始错误）：", "Operation failed (original error): ") + text(str(exc)))
