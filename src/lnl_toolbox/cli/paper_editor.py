from __future__ import annotations

"""Paper lookup, summaries and explicitly invoked parameter editing."""

from copy import deepcopy
import json
from pathlib import Path
import sys

from lnl_toolbox.catalog import default_paper_config, load_papers, load_recipe_config
from lnl_toolbox.cli.i18n import language, saved_language, save_language, text

_CLEAR_VALUE = object()

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


def show_paper(paper, *, session=None):
    """Display paper information only; editing is a separate entry point."""
    write = session.write if session else print
    summary(paper, write)
    write(text("编辑参数：", "Edit parameters:") + f" lnl edit {paper.id}")
    return 0


def edit_paper(paper, *, session=None):
    """Enter the keyboard-driven workspace only through the explicit edit command."""
    if session is None and not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise ValueError(text("参数编辑需要交互终端；请在终端运行 lnl edit <paper>。", "Parameter editing requires a terminal; run lnl edit <paper> in a terminal."))
    try:
        return PaperEditor(paper, session).run()
    except (EOFError, KeyboardInterrupt):
        return 130

class PaperEditor:
    def __init__(self, paper, session=None, *, recipe=None, guided=False):
        from lnl_toolbox import parameters
        self.rules = parameters
        self.paper, self.session = paper, session
        self.recipe = recipe if recipe is not None else default_paper_config(paper)[1]
        self.guided = guided
        self.baseline = load_recipe_config(self.recipe)
        self.config = deepcopy(self.baseline)
        self.dataset = ""
        self.advanced = False
        self.schedule_stash = {}
        self.messages = []
        self.saved_config = deepcopy(self.config)
        self.history = []

    def schema(self):
        return self.rules.schema_from_payload({
            "config": deepcopy(self.config), "recipe": self.recipe.id,
            "path": self.rules._display_path(self.recipe.config_path), "runner": self.recipe.runner,
            "method": self.recipe.method,
        }, split_reference_config=self.baseline if self.guided else None)

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

    # UI-independent mutations remain transactional through the shared rules.
    def set_schedule(self, prefix, *, enabled, mode="multistep", nodes=None, initial=None, rates=None):
        candidate = deepcopy(self.config)
        schedule = self.rules._config_path_value(candidate, prefix)[1]
        base = next(base for name, base, _ in self.schedules() if name == prefix)
        previous = deepcopy(schedule)
        if not enabled:
            schedule.pop("lr_values", None)
            if "step_milestones" in schedule:
                schedule["step_milestones"] = []
            schedule["name"] = "none"
        else:
            if prefix in self.schedule_stash:
                schedule.update(deepcopy(self.schedule_stash[prefix]))
            schedule["name"] = mode
            if mode == "multistep" or "step_milestones" in schedule:
                nodes = list(nodes or [])
                if any(type(n) is not int or n <= 0 for n in nodes) or nodes != sorted(set(nodes)):
                    raise ValueError(text("节点必须为递增、不重复的正整数。", "Milestones must be distinct positive integers in increasing order."))
                if len(rates or []) != len(nodes):
                    raise ValueError(text("每个节点都需要一个学习率。", "Each milestone needs a learning rate."))
                key = "step_milestones" if "step_milestones" in schedule else "milestones"
                schedule[key] = nodes
                base_rate = initial if initial is not None else self.rules._config_path_value(candidate, base)[1]
                derived = [float(base_rate) * float(schedule.get("gamma", .1)) ** (i + 1) for i in range(len(nodes))]
                if "lr_values" in schedule or list(rates or []) != derived:
                    schedule["lr_values"] = list(rates or [])
            else:
                schedule.pop("lr_values", None)
            if initial is not None:
                self.rules._set_config_path(candidate, base, initial)
        self.config = self.prepare(candidate)
        if not enabled:
            self.schedule_stash[prefix] = previous

    def select_dataset(self, alias):
        from lnl_toolbox.training.data_service import DEFAULT_DATA_SERVICE
        from lnl_toolbox.quickstart.templates import adapt_split_counts
        from lnl_toolbox.data.registry import training_pool_size
        candidate = DEFAULT_DATA_SERVICE.apply(self.config, alias)
        report = DEFAULT_DATA_SERVICE.status(alias)
        if report.profile:
            total = training_pool_size(report.adapter, dict(report.profile.sample_counts_by_split))
            if total:
                adapt_split_counts(candidate, self.baseline, total)
        self.config = self.prepare(candidate)
        self.dataset = alias

    def write(self, message):
        self.messages.append(str(message))
        if self.session:
            self.session.write(str(message))

    def verify(self):
        from lnl_toolbox.catalog import find_project_root, resolve_config_paths
        from lnl_toolbox.training.service import ExperimentService
        candidate = resolve_config_paths(self.prepare(self.config), find_project_root())
        service = ExperimentService()
        runner = service.preflight(candidate, check_data=True)
        self.write(text("验证通过。", "Validation passed."))
        for field in runner.describe(candidate).fields:
            self.write(f"  {field.label}: {field.value}")
        return service, candidate

    def save_to(self, path, *, overwrite=False):
        import yaml
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(text("文件扩展名必须为 .yaml 或 .yml。", "The file extension must be .yaml or .yml."))
        if path in self.rules._builtin_config_paths():
            raise ValueError(text("不能覆盖论文内置配置。", "Built-in paper configurations cannot be overwritten."))
        candidate = self.prepare(self.config)
        # Exclusive creation also protects against a file appearing after the dialog.
        with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
            yaml.safe_dump(candidate, stream, sort_keys=False, allow_unicode=True)
        self.config = candidate
        self.saved_config = deepcopy(candidate)
        return path

    def changes(self, reference=None):
        """Human-readable leaf differences, excluding generated bookkeeping."""
        def leaves(value, prefix=""):
            result = {}
            for key, item in value.items():
                if key in {"paper_parameter_status", "paper_parameter_changes", "web_parameter_record"}:
                    continue
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(item, dict):
                    result.update(leaves(item, path))
                else:
                    result[path] = item
            return result
        before = leaves(self.baseline if reference is None else reference)
        after = leaves(self.config)
        return [(path, before.get(path), after.get(path)) for path in sorted(before.keys() | after.keys())
                if before.get(path) != after.get(path) and not self.rules._internal_seed_path(path)]

    def workspace(self, *, input=None, output=None):
        return _PaperWorkspace(self, input=input, output=output)

    def run(self, *, input=None, output=None):
        while True:
            workspace = self.workspace(input=input, output=output)
            result = workspace.app.run(pre_run=workspace.start)
            if not isinstance(result, tuple):
                return result
            if result[0] == "language":
                # Apply in the parent context, not a background task's ContextVar.
                save_language(result[1])
                continue
            _, service, candidate = result
            # Training owns the normal terminal and main thread, including Ctrl+C.
            # No background training survives closing an editor application.
            try:
                service.run(candidate, recipe=self.recipe.id)
                self.write(text("训练已完成。", "Training completed."))
            except KeyboardInterrupt:
                self.write(text("训练已中断。", "Training interrupted."))
            except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
                self.write(text("训练失败：", "Training failed: ") + str(exc))


class _PaperWorkspace:
    """Persistent terminal layout. prompt_toolkit is loaded only by lnl edit."""

    def __init__(self, editor, *, input=None, output=None):
        from prompt_toolkit import Application
        from prompt_toolkit.data_structures import Point
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout, HSplit, VSplit, Window, FloatContainer, DynamicContainer
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import TextArea, Frame
        from prompt_toolkit.output.defaults import create_output

        self.editor = editor
        self.output = output or create_output()
        self.selected = 0
        self.schedule_index = 0
        self.modals = []
        self.button_handlers = {}
        self.busy = False
        self.active_task = None
        self.status = editor.messages[-1] if editor.messages else text("草稿仅保存在内存中。", "Your draft is in memory only.")
        self.error = False
        self.entries = []
        self.search = TextArea(height=1, prompt=text(" 搜索 ", " Search "), multiline=False)
        self.search.buffer.on_text_changed += lambda _: self.refresh(reset=True)
        self.list_control = FormattedTextControl(
            self.rows, focusable=True,
            get_cursor_position=lambda: Point(0, self.selected),
        )
        self.list_window = Window(self.list_control, wrap_lines=False)
        self.detail_control = FormattedTextControl(self.details)
        self.detail_window = Window(self.detail_control, wrap_lines=True)
        self.schedule_control = FormattedTextControl(
            self.schedule_rows, focusable=True, get_cursor_position=self.schedule_cursor,
        )
        def detail_pane(fallback, schedule_height=None):
            schedule_window = Window(self.schedule_control, wrap_lines=False, height=schedule_height)
            return Frame(DynamicContainer(lambda: schedule_window if self.selected_schedule() else fallback),
                         title=lambda: text("学习率设置", "Learning rate") if self.selected_schedule()
                         else text("参数说明", "Description"))

        wide = VSplit([Frame(self.list_window, title=lambda: self.group_title()),
                       detail_pane(self.detail_window)], padding=1)
        narrow = HSplit([Frame(self.list_window, title=lambda: self.group_title()),
                         detail_pane(Window(self.detail_control, height=4, wrap_lines=True), schedule_height=8)])
        body = DynamicContainer(lambda: wide if self.output.get_size().columns >= 100 else narrow)
        self.root = FloatContainer(content=HSplit([
            Window(FormattedTextControl(self.header), height=2, style="class:header"),
            self.search,
            body,
            Window(FormattedTextControl(self.status_text), height=3, wrap_lines=True),
            Window(FormattedTextControl(self.footer),
                height=2, wrap_lines=True, style="class:footer"),
        ]), floats=[])
        kb = KeyBindings()
        browsing = Condition(lambda: not self.modals and not self.busy)
        on_list = browsing & Condition(lambda: self.app.layout.has_focus(self.list_control))
        on_schedule = browsing & Condition(lambda: self.app.layout.has_focus(self.schedule_control))
        modal = Condition(lambda: bool(self.modals))
        movable = modal & Condition(lambda: bool(self.modals[-1].get("move")))

        @kb.add("up", filter=browsing | movable)
        def up(event):
            if self.modals:
                self.modals[-1]["move"](-1)
            elif on_schedule():
                self.move_schedule_selection(-1)
            else:
                self.selected = max(0, self.selected - 1)
                self.schedule_index = 0
                self.app.layout.focus(self.list_control)

        @kb.add("down", filter=browsing | movable)
        def down(event):
            if self.modals:
                self.modals[-1]["move"](1)
            elif on_schedule():
                self.move_schedule_selection(1)
            else:
                self.selected = min(max(0, len(self.entries) - 1), self.selected + 1)
                self.schedule_index = 0
                self.app.layout.focus(self.list_control)

        @kb.add("pageup", filter=on_list)
        def page_up(event):
            self.selected = max(0, self.selected - 10)
            self.schedule_index = 0

        @kb.add("pagedown", filter=on_list)
        def page_down(event):
            self.selected = min(max(0, len(self.entries) - 1), self.selected + 10)
            self.schedule_index = 0

        @kb.add("enter", filter=browsing | modal, eager=True)
        def enter(event):
            if self.modals:
                self.modals[-1]["submit"]()
            elif on_schedule():
                self.launch(self.edit_schedule_option())
            elif not self.app.layout.has_focus(self.list_control):
                self.app.layout.focus(self.list_control)
            else:
                self.launch(self.edit_selected())

        @kb.add("right", filter=on_list & Condition(lambda: self.selected_schedule() is not None))
        def focus_schedule(event):
            self.app.layout.focus(self.schedule_control)

        @kb.add("left", filter=on_schedule)
        def focus_parameter_list(event):
            self.app.layout.focus(self.list_control)

        @kb.add("tab", eager=True)
        def tab(event):
            self.focus_next(1)

        @kb.add("s-tab", eager=True)
        def backtab(event):
            self.focus_next(-1)

        @kb.add("c-f", filter=browsing)
        def search(event):
            self.app.layout.focus(self.search)

        @kb.add("/", filter=on_list)
        def actions(event):
            self.launch(self.command_palette())

        @kb.add("c-s", filter=browsing & Condition(lambda: not self.editor.guided))
        def save(event):
            self.launch(self.save())

        @kb.add("f5", filter=browsing & Condition(lambda: self.editor.guided))
        def review(event):
            self.app.exit(result="review")

        @kb.add("left", filter=on_list & Condition(lambda: self.editor.guided))
        def back(event):
            self.launch(self.quit())

        @kb.add("f2", filter=browsing)
        def advanced(event):
            self.toggle_advanced()

        @kb.add("c-z", filter=browsing)
        def undo(event):
            self.undo()

        @kb.add("?", filter=on_list)
        def help_(event):
            self.launch(self.help())

        @kb.add("escape")
        @kb.add("c-c")
        @kb.add("c-d")
        def escape(event):
            if self.modals:
                self.modals[-1]["finish"](None)
            elif self.busy:
                self.set_status(text("正在检查，请稍候。", "A check is running; please wait."))
            elif on_schedule():
                self.app.layout.focus(self.list_control)
            elif self.search.text:
                self.search.text = ""
                self.app.layout.focus(self.list_control)
            else:
                self.launch(self.quit())

        self.app = Application(
            layout=Layout(self.root, focused_element=self.list_control),
            key_bindings=kb, full_screen=True, mouse_support=True,
            input=input, output=self.output,
            style=Style.from_dict({
                "header": "bg:#183447 #ffffff bold", "footer": "bg:#183447 #ffffff",
                "selected": "bg:#31566e #ffffff bold", "changed": "#d7a651",
                "error": "#ff6565 bold", "ok": "#61c2a2", "muted": "#8295a2",
                "dialog": "bg:#23323c #e5edf0", "dialog frame.label": "bold",
                "button.focused": "bg:#3b7895 #ffffff bold",
            }),
        )
        self.app.ttimeoutlen = 0.1
        self.refresh()

    def refresh(self, *, reset=False):
        query = self.search.text.casefold()
        if self.editor.guided:
            self.entries = self.editor.plan_entries()
        else:
            fields = [("field", field) for field in self.editor.fields()]
            schedules = [("schedule", {"path": prefix, "base": base, "value": schedule})
                         for prefix, base, schedule in self.editor.schedules()]
            self.entries = schedules + fields
        if query:
            self.entries = [(kind, f) for kind, f in self.entries
                            if query in f["path"].casefold()
                            or (kind == "field" and query in self.editor.note(f).casefold())]
        self.selected = 0 if reset else min(self.selected, max(0, len(self.entries) - 1))
        if reset:
            self.schedule_index = 0
        if not self.selected_schedule() and hasattr(self, "app") and self.app.layout.has_focus(self.schedule_control):
            self.app.layout.focus(self.list_control)
        if hasattr(self, "app"):
            self.app.invalidate()

    def group_title(self):
        if self.editor.guided and self.editor.restricted:
            return text("专属参数（只读）", "Restricted parameters (read-only)")
        return text("全部可编辑参数", "All editable parameters") if self.editor.advanced else text("默认参数", "Default parameters")

    def header(self):
        dataset = self.editor.dataset or self.editor.config.get("data", {}).get("name", "-")
        if self.editor.guided:
            return f" LNL Start / {self.editor.paper.acronym}    {dataset}\n " + text(
                "[4/5] 运行计划  ·  F5 检查与确认  ·  ← / Esc 返回方法",
                "[4/5] Run plan  ·  F5 Review and confirm  ·  ← / Esc Back to methods")
        dirty = bool(self.editor.changes(self.editor.saved_config))
        return f" LNL Toolbox / {self.editor.paper.acronym}    {dataset}\n " + text(
            f"已修改 {len(self.editor.changes())} 项 / " + ("未保存" if dirty else "无未保存修改"),
            f"{len(self.editor.changes())} changes / " + ("unsaved" if dirty else "no unsaved changes"))

    def footer(self):
        if self.selected_schedule():
            return text(
                " ↑↓ 选择  → / Tab 进入学习率设置  Enter 编辑  ← 返回参数列表",
                " Up/Down Select  Right / Tab Focus schedule  Enter Edit  Left Back to parameters",
            )
        return text(
            " ↑↓ 选择  Enter 编辑  Tab 切换  Ctrl+F 搜索  / 操作  Esc 返回  ? 帮助",
            " Up/Down Select  Enter Edit  Tab Focus  Ctrl+F Search  / Actions  Esc Back  ? Help",
        )

    def rows(self):
        changed = {path for path, _, _ in self.editor.changes()}
        result = []
        for index, (kind, field) in enumerate(self.entries):
            value = field["value"]
            if kind == "schedule":
                if self.editor.guided:
                    value = self.schedule_label(field)
                else:
                    value = text("启用", "On") if self.editor.enabled(value) else text("关闭", "Off")
                label = field["path"] + text("  学习率阶段", "  LR schedule")
            elif kind == "group":
                label = field["path"]
                value = " / ".join(str(child["value"]) for child in field["children"])
            elif kind == "info":
                label = field["path"]
            else:
                label = field["path"]
                value = json.dumps(value, ensure_ascii=False)
                if field.get("split_reference"):
                    value += " / " + str(self.split_total(field))
            marker = "*" if field["path"] in changed or (kind == "schedule" and any(p.startswith(field["path"] + ".") for p in changed)) else " "
            style = "class:selected" if index == self.selected else ("class:changed" if marker == "*" else "")
            def select(event, index=index):
                from prompt_toolkit.mouse_events import MouseEventType
                if event.event_type == MouseEventType.MOUSE_UP:
                    if self.selected != index:
                        self.schedule_index = 0
                    self.selected = index
                    self.app.layout.focus(self.list_control)
                else:
                    return NotImplemented
            result.append((style, f" {'›' if index == self.selected else ' '} {marker} {label}  {value}\n", select))
        return result or [("class:muted", text(" 没有匹配的参数。", " No matching parameters."))]

    def split_total(self, field):
        if self.editor.guided and self.editor.pool_size:
            return self.editor.pool_size
        from lnl_toolbox.data.registry import training_pool_size
        profile = self.editor.config.get("local_dataset", {}).get("profile") or {}
        counts = profile.get("sample_counts_by_split", {})
        return (training_pool_size(self.editor.config.get("data", {}).get("name", ""), counts)
                if counts else field["split_reference"]["total"])

    def details(self):
        if not self.entries:
            return text("清空搜索或按 F2 显示高级参数。", "Clear search or press F2 for advanced parameters.")
        kind, field = self.entries[self.selected]
        if kind == "info":
            return str(field["value"])
        if kind == "group":
            return "\n\n".join(child["path"] + "\n" + self.editor.note(child) for child in field["children"])
        if kind == "schedule":
            if self.editor.guided:
                return self.schedule_label(field) + "\n" + text("按 Enter 查看本阶段控制项。", "Press Enter to inspect stage controls.")
            schedule = field["value"]
            nodes = schedule.get("step_milestones", schedule.get("milestones", []))
            return text("按 Enter 编辑阶段开关、节点和各段学习率。", "Press Enter to edit the toggle, milestones and per-segment rates.") + "\n" + field["base"] + "\n" + str(nodes)
        note = self.editor.note(field)
        reference = field.get("split_reference")
        if reference:
            note += "\n" + text("原论文：", "Paper: ") + f"{reference['count']} / {reference['total']}"
            if self.editor.guided:
                note += "\n" + self.editor.split_summary()
        return field["path"] + "\n\n" + note

    def status_text(self):
        return [("class:error" if self.error else "class:ok", " " + self.status)]

    def set_status(self, message, *, error=False):
        self.status, self.error = str(message), error
        self.app.invalidate()

    def focus_next(self, delta):
        if not self.modals:
            (self.app.layout.focus_next if delta > 0 else self.app.layout.focus_previous)()
            return
        from prompt_toolkit.layout import Window
        from prompt_toolkit.layout.layout import walk
        from prompt_toolkit.layout.containers import to_container
        windows = []
        for container in walk(to_container(self.modals[-1]["container"]), skip_hidden=True):
            if isinstance(container, Window) and container.content.is_focusable() and container not in windows:
                windows.append(container)
        if windows:
            current = self.app.layout.current_window
            index = windows.index(current) if current in windows else 0
            self.app.layout.focus(windows[(index + delta) % len(windows)])

    def button(self, label, handler):
        from prompt_toolkit.widgets import Button
        button = Button(text=label, handler=handler)
        self.button_handlers[button.control] = handler
        return button

    def change(self, operation):
        if self.editor.guided:
            before = self.editor.snapshot()
            try:
                operation()
            except Exception:
                self.editor.restore(before)
                raise
            if before != self.editor.snapshot():
                self.editor.history.append(before)
            self.refresh()
            return
        before = (deepcopy(self.editor.config), self.editor.dataset, deepcopy(self.editor.schedule_stash))
        operation()
        if before[0] != self.editor.config:
            self.editor.history.append(before)
        self.refresh()

    def undo(self):
        if self.editor.history:
            if self.editor.guided:
                self.editor.restore(self.editor.history.pop())
            else:
                self.editor.config, self.editor.dataset, self.editor.schedule_stash = self.editor.history.pop()
            self.refresh()
            self.set_status(text("已撤销上一次修改。", "Last change undone."))

    def toggle_advanced(self):
        if self.editor.guided:
            self.editor.restricted = False
        self.editor.advanced = not self.editor.advanced
        self.refresh(reset=True)

    def launch(self, coroutine):
        if self.active_task and not self.active_task.done():
            coroutine.close()
            return
        async def guarded():
            try:
                await coroutine
            except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
                self.set_status(text("操作失败：", "Operation failed: ") + str(exc), error=True)
        self.active_task = self.app.create_background_task(guarded())

    def start(self):
        import os
        if saved_language() is None and not os.environ.get("LNL_LANG"):
            self.launch(self.choose_language())

    async def dialog(self, body, title, *, focus=None, submit=None, move=None):
        import asyncio
        from prompt_toolkit.layout import Float
        from prompt_toolkit.widgets import Dialog
        future = asyncio.get_running_loop().create_future()
        def finish(value):
            if not future.done():
                future.set_result(value)
        cancel = self.button(text("取消", "Cancel"), lambda: finish(None))
        accept = self.button(text("确认", "Confirm"), lambda: submit(finish) if submit else finish(True))
        dialog = Dialog(body=body, title=title, buttons=[accept, cancel], with_background=True,
                        width=min(100, max(32, self.app.output.get_size().columns - 4)))
        floating = Float(content=dialog)
        previous = self.app.layout.current_control
        def enter():
            self.button_handlers.get(self.app.layout.current_control, accept.handler)()
        state = {"finish": finish, "submit": enter, "move": move, "container": dialog.__pt_container__()}
        self.modals.append(state)
        self.root.floats.append(floating)
        self.app.layout.focus(focus or body)
        self.app.invalidate()
        try:
            return await future
        finally:
            self.root.floats.remove(floating)
            self.modals.remove(state)
            self.app.layout.focus(previous)
            self.app.invalidate()

    async def ask(self, title, value, *, note="", convert=None):
        from prompt_toolkit.widgets import TextArea, Label
        from prompt_toolkit.layout import HSplit
        field = TextArea(text=str(value), height=1, multiline=False)
        field.buffer.cursor_position = len(field.text)
        error = Label("", style="class:error")
        def submit(finish):
            try:
                finish(convert(field.text) if convert else field.text)
            except (ValueError, TypeError, KeyError) as exc:
                error.text = str(exc)
        return await self.dialog(HSplit([Label(note), field, error]), title,
                                 focus=field, submit=submit)

    async def pick(self, title, choices, *, note="", default=None):
        from prompt_toolkit.widgets import TextArea, Label
        from prompt_toolkit.layout import HSplit, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.data_structures import Point
        search = TextArea(height=1, prompt=text("搜索 ", "Search "), multiline=False)
        selected = [next((i for i, (_, value) in enumerate(choices) if value == default), 0)]
        def filtered():
            return [(label, value) for label, value in choices if search.text.casefold() in label.casefold()]
        def lines():
            return [("class:selected" if i == selected[0] else "", " › " + label + "\n")
                    for i, (label, _) in enumerate(filtered())] or [("", text("没有匹配项。", "No matches."))]
        def move(delta):
            selected[0] = min(max(0, len(filtered()) - 1), max(0, selected[0] + delta))
        search.buffer.on_text_changed += lambda _: selected.__setitem__(0, 0)
        control = FormattedTextControl(lines, get_cursor_position=lambda: Point(0, selected[0]))
        def submit(finish):
            rows = filtered()
            if rows:
                finish(rows[selected[0]][1])
        return await self.dialog(HSplit([Label(note), search, Window(control, height=10),
                                        Label(text("↑↓ 选择，Enter 确认，Esc 返回", "Up/Down select, Enter confirm, Esc back"))]),
                                 title, focus=search, submit=submit, move=move)

    async def confirm(self, title, message):
        # Destructive actions default to Cancel, including pressing Enter.
        return await self.pick(title, [
            (text("取消", "Cancel"), False), (text("确认", "Confirm"), True)], note=message)

    async def show_text(self, title, content):
        from prompt_toolkit.widgets import TextArea
        area = TextArea(text=content, read_only=True, scrollbar=True,
                        height=min(18, max(5, self.output.get_size().rows - 9)), wrap_lines=True)
        return await self.dialog(area, title, focus=area)

    async def value(self, field):
        if field.get("rows"):
            index = await self.pick(field["path"], [(str(i + 1), i) for i in range(len(field["rows"]))])
            if index is None:
                return None
            child = await self.pick(field["path"], [(f["key"], f) for f in field["rows"][index] if f.get("editable", True)])
            if child is None:
                return None
            values = deepcopy(field["value"])
            child = {**child, "path": child.get("path", child["key"]), "value": values[index][child["key"]]}
            result = await self.value(child)
            if result is None:
                return None
            values[index][child["key"]] = None if result is _CLEAR_VALUE else result
            return values
        if field["kind"] == "boolean":
            return not field["value"]
        if field.get("choices"):
            return await self.pick(field["path"], [(c["value"], c["value"]) for c in field["choices"]], default=field["value"])
        def convert(raw):
            from lnl_toolbox.core.config_overrides import parse_override_value
            if self.editor.guided and field.get("nullable") and not raw.strip():
                value = None
            elif self.editor.guided and field["kind"] in {"list", "object"}:
                value = json.loads(raw)
            else:
                value = parse_override_value(raw) if field["kind"] in {"number", "list"} else raw
            value = self.editor.rules._coerce_patch_value(field, value)
            return _CLEAR_VALUE if self.editor.guided and value is None else value
        structured = field["kind"] in {"number", "list", "object"}
        initial = json.dumps(field["value"], ensure_ascii=False) if structured else field["value"]
        return await self.ask(field["path"], initial,
                              note=self.editor.note(field) if "." in field["path"] else "", convert=convert)

    async def edit_selected(self):
        if not self.entries:
            return
        kind, field = self.entries[self.selected]
        if kind == "info":
            return
        if kind == "group":
            children = field["children"]
            if self.editor.restricted:
                await self.show_text(field["path"], self.details())
                return
            choices = []
            for child in children:
                options = child.get("choices") or []
                fixed_choice = len(options) == 1 and options[0]["value"] == child["value"]
                if child["editable"] and child.get("display_group") != "restricted" and not fixed_choice:
                    choices.append((f"{child['path']} = {child['value']}", child))
            field = await self.pick(field["path"], choices) if choices else None
            if field is None:
                return
        if kind == "schedule":
            if self.editor.guided:
                self.app.layout.focus(self.schedule_control)
            else:
                await self.schedule(field)
            return
        if not field["editable"] or self.editor.guided and self.editor.restricted:
            return
        value = await self.value(field)
        if value is not None:
            self.change(lambda: self.editor.update([(field["path"], None if value is _CLEAR_VALUE else value)]))
            self.set_status(text("已更新草稿。", "Draft updated."))

    async def choose_language(self):
        chosen = await self.pick("语言 / Language", [("中文", "zh"), ("English", "en")], default=language())
        if chosen:
            self.app.exit(result=("language", chosen))

    async def help(self):
        if self.editor.guided:
            await self.show_text(text("键盘操作", "Keyboard controls"), text(
                "↑↓：选择参数\nEnter：编辑\nCtrl+F：搜索参数\nF2：高级参数\nCtrl+Z：撤销\n/：操作菜单\nF5：检查与确认\n← / Esc：返回方法选择",
                "Up/Down: select a parameter\nEnter: edit\nCtrl+F: search\nF2: advanced parameters\nCtrl+Z: undo\n/: actions\nF5: review and confirm\n← / Esc: back to methods"))
            return
        await self.show_text(text("键盘操作", "Keyboard controls"), text(
            "↑↓ / PageUp / PageDown：选择参数\nEnter：编辑；布尔值直接切换\nTab / Shift+Tab：切换焦点\nCtrl+F：搜索参数\nF2：显示/隐藏高级参数\n/：搜索操作（保存、验证、数据集、改动、训练、语言）\nCtrl+S：另存 YAML\nCtrl+Z：撤销上次修改\nEsc：关闭弹窗、清空搜索或退出\n\n* 表示相对原始配置发生修改。草稿不会自动保存；启动训练需验证和明确确认。",
            "Up/Down / PageUp / PageDown: select a parameter\nEnter: edit; booleans toggle directly\nTab / Shift+Tab: change focus\nCtrl+F: search parameters\nF2: show/hide advanced parameters\n/: search actions (save, validate, dataset, diff, train, language)\nCtrl+S: save YAML as\nCtrl+Z: undo the last change\nEsc: dismiss a dialog, clear search or exit\n\n* marks a change from the initial configuration. Drafts are not saved automatically. Training requires validation and explicit confirmation."))

    def diff_text(self):
        rows = self.editor.changes()
        return "\n".join(f"{path}: {json.dumps(before, ensure_ascii=False)} -> {json.dumps(after, ensure_ascii=False)}"
                         for path, before, after in rows) or text("没有修改。", "No changes.")

    async def command_palette(self):
        actions = [
            (text("选择数据集", "Choose dataset"), "dataset"),
            (text("显示/隐藏高级参数", "Toggle advanced parameters"), "advanced"),
            (text("查看改动", "Review changes"), "diff"),
            (text("另存 YAML", "Save YAML as"), "save"),
            (text("完整验证", "Validate"), "validate"),
            (text("启动训练", "Start training"), "train"),
            (text("撤销上次修改", "Undo last change"), "undo"),
            ("语言 / Language", "language"),
            (text("帮助", "Help"), "help"),
            (text("退出", "Exit"), "quit"),
        ]
        if self.editor.guided:
            actions = [(text("下一步：检查与确认", "Next: Review and confirm"), "review")] + [
                item for item in actions if item[1] not in {"dataset", "save", "validate", "train"}
            ]
            actions.insert(1, (text("查看专属参数 / 返回可编辑参数", "Toggle restricted parameters"), "restricted"))
        choice = await self.pick(text("操作", "Actions"), actions)
        if choice == "restricted":
            self.editor.restricted = not self.editor.restricted
            self.refresh(reset=True)
            return
        if choice == "review":
            self.app.exit(result="review")
            return
        if choice == "advanced":
            self.toggle_advanced()
        elif choice == "undo":
            self.undo()
        elif choice == "diff":
            await self.show_text(text("配置改动", "Configuration changes"), self.diff_text())
        elif choice is not None:
            await getattr(self, "choose_language" if choice == "language" else choice)()

    async def dataset(self):
        from lnl_toolbox.training.data_service import DEFAULT_DATA_SERVICE
        reports = await self.work(DEFAULT_DATA_SERVICE.list_datasets)
        choices = [(r.name, r.name) for r in reports if r.location]
        if not choices:
            self.set_status(text("请先使用 lnl data register 登记数据集。", "Register a dataset with lnl data register first."), error=True)
            return
        alias = await self.pick(text("数据集", "Dataset"), choices)
        if alias:
            # Applying a registered dataset is metadata-only; checks happen in Validate.
            self.change(lambda: self.editor.select_dataset(alias))

    async def save(self):
        path = await self.ask(text("另存 YAML", "Save YAML as"),
                              f"configs/experiment/{self.editor.paper.id}-custom.yaml")
        if path is None:
            return
        path = Path(path).expanduser()
        overwrite = path.exists()
        if overwrite and not await self.confirm(text("覆盖已有文件？", "Overwrite existing file?"), str(path)):
            return
        # Never create directories implicitly until the user has selected a path.
        if not path.parent.exists():
            if not await self.confirm(text("创建目录？", "Create directory?"), str(path.parent)):
                return
            path.parent.mkdir(parents=True, exist_ok=True)
        saved = self.editor.save_to(path, overwrite=overwrite)
        self.refresh()
        self.set_status(text("已保存：", "Saved: ") + str(saved))

    async def work(self, function):
        import asyncio
        self.busy = True
        self.set_status(text("正在检查…", "Checking…"))
        try:
            return await asyncio.to_thread(function)
        finally:
            self.busy = False

    async def validate(self):
        before = len(self.editor.messages)
        await self.work(self.editor.verify)
        self.set_status(text("验证通过。", "Validation passed."))
        await self.show_text(text("运行计划", "Run plan"), "\n".join(self.editor.messages[before:]))

    async def train(self):
        before = len(self.editor.messages)
        service, candidate = await self.work(self.editor.verify)
        message = self.diff_text() + "\n\n" + "\n".join(self.editor.messages[before:])
        if not await self.show_text(text("训练前检查", "Before training"), message):
            return
        if await self.confirm(text("确认启动训练？", "Start training now?"), text(
                "按当前草稿执行；训练日志将在普通终端显示。", "Run this draft; training logs will use the normal terminal.")):
            self.app.exit(result=("train", service, candidate))

    async def quit(self):
        if self.editor.changes(self.editor.saved_config):
            if not await self.confirm(text("放弃未保存的修改？", "Discard unsaved changes?"), ""):
                return
        self.app.exit(result="back" if self.editor.guided else 0)

    def schedule_label(self, group):
        labels = {
            "multistep": text("按节点变化", "Milestones"),
            "cosine": text("余弦下降", "Cosine decay"),
            "linear_after": text("线性下降", "Linear decay"),
            "linear_decay": text("线性下降", "Linear decay"),
            "step": text("固定间隔衰减", "Fixed interval decay"),
        }
        mode = (labels.get(group["value"], labels["multistep"]) if group["enabled"]
                else text("固定学习率", "Constant learning rate"))
        budget = self.editor.schedule_budget(group)
        unit = text("步", "steps") if group["stepped"] else text("轮", "epochs")
        return f"{mode} / {budget} {unit}" if budget is not None else mode

    def selected_schedule(self):
        if self.editor.guided and self.entries:
            kind, field = self.entries[self.selected]
            if kind == "schedule":
                return field
        return None

    def schedule_options(self, group):
        fields = group["fields"]
        options = []
        if "none" in group["choices"]:
            label = text("关闭动态学习率", "Disable dynamic learning rate") if group["enabled"] else text("启用动态学习率", "Enable dynamic learning rate")
            options.append((label, "toggle"))
        if group["enabled"] and not group["stepped"] and len(set(group["choices"]) - {"none"}) > 1:
            options.append((text("变化方式", "Schedule mode"), "mode"))
        base = group["base"]
        has_nodes = group["stepped"] or group["value"] == "multistep"
        if group["enabled"] and group["can_segment"]:
            label = (text("移出学习率分段表", "Move learning rate out of stage table") if group["segmented"]
                     else text("加入学习率分段表", "Move learning rate into stage table"))
            options.append((label, "segment"))
        if group["enabled"] and base and base["editable"] and (not has_nodes or group["segmented"]):
            options.append((f"{text('起始学习率', 'Initial learning rate')} = {base['value']}", base["path"]))
        node = group["node"]
        if group["enabled"] and has_nodes and node is not None:
            options.extend(self.schedule_node_options(group))
        if group["enabled"]:
            active = self.editor.active_schedule_fields(group)
            for suffix, field in fields.items():
                if suffix not in active or suffix in {"milestones", "step_milestones", "lr_values"} or has_nodes and suffix == "gamma":
                    continue
                value = field["value"]
                if suffix == "t_max" and not field["editable"] and self.editor.source_schema["method"] == "fine":
                    warmup = self.editor._visible().get("fine.warmup_epochs", {}).get("value", 0)
                    budget = self.editor.schedule_budget(group)
                    if budget is not None:
                        value = max(1, budget - warmup)
                options.append((f"{field.get('label', field['path'])} = {value}", field["path"]))
        return options

    def schedule_note(self, group):
        note = self.schedule_label(group)
        if "none" not in group["choices"]:
            note += "\n" + text("此阶段的学习率变化方式由训练端固定。", "The training method fixes this schedule mode.")
        if self.editor.source_schema["method"] == "fine" and group["value"] == "cosine":
            warmup = self.editor._visible().get("fine.warmup_epochs", {}).get("value", 0)
            budget = self.editor.schedule_budget(group)
            if budget is not None:
                note += "\n" + text(f"预热 {warmup} 轮，余弦下降 {max(1, budget-warmup)} 轮。",
                                     f"Warmup: {warmup} epochs. Cosine decay: {max(1, budget-warmup)} epochs.")
        return note

    def move_schedule_selection(self, delta=0):
        group = self.selected_schedule()
        options = self.schedule_options(group) if group else []
        self.schedule_index = max(0, min(len(options) - 1, self.schedule_index + delta))
        return options

    def schedule_cursor(self):
        from prompt_toolkit.data_structures import Point

        group = self.selected_schedule()
        if not group:
            return Point(0, 0)
        self.move_schedule_selection()
        return Point(0, len(self.schedule_note(group).splitlines()) + self.schedule_index)

    def schedule_rows(self):
        from prompt_toolkit.mouse_events import MouseEventType

        group = self.selected_schedule()
        if not group:
            return []
        options = self.move_schedule_selection()
        rows = [("class:muted", self.schedule_note(group) + "\n")]
        for index, (label, _) in enumerate(options):
            def select(event, index=index):
                if event.event_type != MouseEventType.MOUSE_UP or self.modals or self.busy:
                    return NotImplemented
                self.schedule_index = index
                self.app.layout.focus(self.schedule_control)
            selected = index == self.schedule_index and self.app.layout.has_focus(self.schedule_control)
            rows.append(("class:selected" if selected else "", f" {'›' if selected else ' '} {label}\n", select))
        return rows

    async def edit_schedule_option(self):
        group = self.selected_schedule()
        options = self.move_schedule_selection()
        if not options:
            return
        prefix = group["path"]
        base = group["base"]
        _, choice = options[self.schedule_index]
        if choice == "toggle":
            self.change(lambda: self.editor.toggle_schedule(prefix))
        elif choice == "segment":
            self.change(lambda: self.editor.segmented.update({base["path"]: not group["segmented"]}))
        elif choice == "mode":
            mode = await self.pick(text("变化方式", "Schedule mode"), [
                (item, item) for item in group["choices"] if item != "none"
            ], default=group["value"])
            if mode is not None:
                self.change(lambda: self.editor.set_schedule_mode(prefix, mode))
        elif isinstance(choice, tuple):
            await self.edit_schedule_node(group, choice)
        elif choice is not None:
            field = self.editor._visible()[choice]
            if not field["editable"]:
                await self.show_text(field["path"], self.editor.note(field))
                return
            value = await self.value(field)
            if value is not None:
                self.change(lambda: self.editor.update([(choice, None if value is _CLEAR_VALUE else value)]))

    @staticmethod
    def schedule_node_values(group):
        fields = group["fields"]
        node = group["node"]
        nodes = list(node["value"] or [])
        rates_field = fields.get("lr_values")
        gamma = fields.get("gamma")
        explicit = rates_field["value"] if rates_field else None
        base = group["base"]["value"] if group["base"] else None
        rates = list(explicit) if explicit is not None else [
            float(f"{base * (gamma['value'] if gamma else .1) ** (i + 1):.7g}") if base is not None else None
            for i in range(len(nodes))
        ]
        return nodes, rates, explicit, base

    def schedule_node_options(self, group):
        node = group["node"]
        nodes, rates, _, _ = self.schedule_node_values(group)
        options = []
        for index, point in enumerate(nodes):
            unit = text("步", "steps") if group["stepped"] else text("轮", "epochs")
            options.append((f"{text('节点', 'Milestone')} {index+1}  {point} {unit}", ("node", index)))
            if group["segmented"] and index < len(rates) and rates[index] is not None:
                options.append((f"  {text('节点后学习率', 'Learning rate after milestone')} = {rates[index]}", ("rate", index)))
            if node["editable"]:
                options.append((f"  {text('移除节点', 'Remove milestone')} {index+1}", ("remove", index)))
        if node["editable"]:
            options.append((text("添加节点", "Add milestone"), ("add", None)))
        return options

    async def edit_schedule_node(self, group, action):
        import math

        fields = group["fields"]
        node = group["node"]
        rates_field = fields.get("lr_values")
        gamma = fields.get("gamma")
        nodes, rates, explicit, base = self.schedule_node_values(group)
        operation, index = action
        if operation == "rate":
            def rate_value(raw):
                value = float(raw)
                if not math.isfinite(value) or value <= 0 or not base or base <= 0:
                    raise ValueError(text("起始和分段学习率必须为正数。", "Initial and segment learning rates must be positive."))
                return value
            rate = await self.ask(text("节点后的学习率", "Learning rate after milestone"), rates[index], convert=rate_value)
            if rate is not None:
                if rates_field and rates_field["editable"]:
                    rates[index] = rate
                    self.change(lambda: self.editor.update([(rates_field["path"], rates)]))
                elif gamma and gamma["editable"]:
                    self.change(lambda: self.editor.update([(gamma["path"], (rate / base) ** (1 / (index + 1)))]))
            return
        if not node["editable"]:
            return
        if operation == "node":
            def milestone(raw):
                value = int(raw)
                if value < 1 or any(value == item for i, item in enumerate(nodes) if i != index):
                    raise ValueError(text("节点须为不重复的正整数。", "Milestones must be distinct positive integers."))
                return value
            point = await self.ask(text("节点位置", "Milestone position"), nodes[index], convert=milestone)
            if point is None:
                return
            nodes[index] = point
        elif operation == "add":
            nodes.append(max(nodes, default=0) + 1)
            if explicit is not None:
                rates.append(rates[-1] if rates else base)
        elif operation == "remove":
            nodes.pop(index)
            if explicit is not None and index < len(rates):
                rates.pop(index)
        order = sorted(range(len(nodes)), key=nodes.__getitem__)
        patches = [(node["path"], [nodes[i] for i in order])]
        if explicit is not None and rates_field["editable"] and len(rates) == len(nodes):
            patches.append((rates_field["path"], [rates[i] for i in order]))
        self.change(lambda: self.editor.update(patches))

    async def schedule(self, field):
        from prompt_toolkit.widgets import TextArea, Label, Checkbox
        from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer
        prefix, base = field["path"], field["base"]
        original = field["value"]
        schedule = self.editor.schedule_stash.get(prefix, original) if not self.editor.enabled(original) else original
        enabled = Checkbox(text=text("启用学习率阶段", "Enable LR scheduling"), checked=self.editor.enabled(original))
        mode = [schedule.get("name", "multistep")]
        if mode[0] == "none":
            mode[0] = "multistep"
        initial = float(self.editor.rules._config_path_value(self.editor.config, base)[1])
        nodes = schedule.get("step_milestones", schedule.get("milestones", []))
        rates = schedule.get("lr_values") or [initial * float(schedule.get("gamma", .1)) ** (i + 1) for i in range(len(nodes))]
        start = TextArea(text=str(initial), height=1, multiline=False)
        start.buffer.cursor_position = len(start.text)
        cells = [(TextArea(text=str(n), height=1, multiline=False),
                  TextArea(text=format(rate, ".12g"), height=1, multiline=False)) for n, rate in zip(nodes, rates)]
        original_rates = {cell[1]: rate for cell, rate in zip(cells, rates)}
        for cell in cells:
            for area in cell:
                area.buffer.cursor_position = len(area.text)
        error = Label("", style="class:error")
        unit = text("步", "step") if "step_milestones" in schedule else text("轮", "epoch")
        grid_cache = []
        page = [0]
        grid_size = [0]
        def capacity():
            return max(1, (self.output.get_size().columns - 30) // 14)
        def remove(cell):
            cells.remove(cell)
            page[0] = min(page[0], max(0, len(cells) - capacity()))
            grid_cache[:] = [grid()]
            self.app.layout.focus(start)
            self.app.invalidate()
        def grid():
            # Keep the same live widgets when resizing or adding/removing stages.
            headings = HSplit([Label(text("阶段", "Segment")), Label(unit), Label(text("学习率", "LR")), Label("")], width=12)
            first = HSplit([Label(text("初始段", "Initial")), Label("0"), start, Label("")])
            columns = [headings, first]
            for index, cell in list(enumerate(cells))[page[0]:page[0] + capacity()]:
                columns.append(HSplit([Label(str(index + 1)), cell[0], cell[1],
                                       self.button(text("移除", "Remove"), lambda cell=cell: remove(cell))]))
            return VSplit(columns, padding=1)
        def add():
            previous = int(cells[-1][0].text) if cells and cells[-1][0].text.isdigit() else 0
            cells.append((TextArea(text=str(previous + 1), height=1, multiline=False),
                          TextArea(text=start.text, height=1, multiline=False)))
            page[0] = max(0, len(cells) - capacity())
            grid_cache[:] = [grid()]
            self.app.invalidate()
        def move_page(delta):
            page[0] = min(max(0, len(cells) - capacity()), max(0, page[0] + delta * capacity()))
            grid_cache[:] = [grid()]
            self.app.invalidate()
        def current_grid():
            if grid_size[0] != self.output.get_size().columns:
                grid_size[0] = self.output.get_size().columns
                grid_cache[:] = [grid()]
            return grid_cache[0]
        async def choose_mode():
            choices = self.editor.rules._parameter_choices(
                {"path": prefix + ".name", "kind": "text", "visible": True, "editable": True,
                 "value": mode[0]}, self.editor.recipe.method)
            chosen = await self.pick(text("变化方式", "Schedule mode"),
                                     [(c["value"], c["value"]) for c in (choices or [{"value": "multistep"}]) if c["value"] != "none"], default=mode[0])
            if chosen:
                mode[0] = chosen
                mode_button.text = chosen
        def select_mode():
            self.app.create_background_task(choose_mode())
        mode_button = self.button(mode[0], select_mode)
        stepped = "step_milestones" in schedule
        def submit(finish):
            try:
                self.change(lambda: self.editor.set_schedule(
                    prefix, enabled=enabled.checked, mode=mode[0],
                    nodes=[int(node.text) for node, _ in cells] if enabled.checked else [],
                    initial=float(start.text) if enabled.checked else None,
                    rates=[original_rates[rate] if rate in original_rates and rate.text == format(original_rates[rate], ".12g")
                           else float(rate.text) for _, rate in cells] if enabled.checked else [],
                ))
                finish(True)
                self.set_status(text("已更新学习率阶段。", "LR schedule updated."))
            except (ValueError, TypeError, RuntimeError, KeyError) as exc:
                error.text = str(exc)
        from prompt_toolkit.layout import ConditionalContainer
        from prompt_toolkit.filters import Condition
        is_piecewise = Condition(lambda: stepped or mode[0] == "multistep")
        grid_cache[:] = [grid()]
        body = HSplit([
            enabled,
            Label(base),
            Label(text("变化方式", "Schedule mode")),
            Label(mode[0]) if stepped else mode_button,
            ConditionalContainer(HSplit([DynamicContainer(current_grid),
                VSplit([self.button(text("上一组", "Previous"), lambda: move_page(-1)),
                        self.button(text("下一组", "Next"), lambda: move_page(1)),
                        self.button(text("添加节点", "Add milestone"), add)])]), filter=is_piecewise),
            ConditionalContainer(HSplit([Label(text("起始学习率", "Initial learning rate")), start,
                Label(text("曲线控制项在高级参数中编辑。", "Edit curve controls under advanced parameters."))]), filter=~is_piecewise),
            Label(text("Enter 应用，Esc 取消；分段表中的学习率可独立填写。", "Enter applies; Esc cancels. Segment rates are independent.")),
            error,
        ])
        await self.dialog(body, prefix, focus=enabled, submit=submit)
