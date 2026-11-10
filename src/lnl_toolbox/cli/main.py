from __future__ import annotations

"""Unified, discoverable command-line interface for toolbox users."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from lnl_toolbox.catalog import (
    PaperSpec,
    RecipeSpec,
    discover_recipes,
    find_project_root,
    load_papers,
    load_yaml,
    paper_by_id,
    recipe_by_id,
    resolve_config_paths,
    load_recipe_config,
    mentornet_preparation_status,
    select_paper_config,
    validate_config,
)
from lnl_toolbox.core.config_overrides import apply_override_assignments
compare_runs = None  # Loaded only by commands that use it.
write_report = None  # Loaded only by commands that use it.
DEFAULT_DATA_SERVICE = None  # Loaded only by commands that use it.
DatasetStatusReport = None  # Loaded only by commands that use it.
from lnl_toolbox.data.profile import NoiseRateInfo, NoiseRateStatus
from lnl_toolbox.training.compatibility import CompatibilityResult, CompatibilityStatus
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner, runner_names
ExperimentReadinessError = None  # Loaded only by commands that use it.
ExperimentService = None  # Loaded only by commands that use it.
plan_sweep = None  # Loaded only by commands that use it.
resolve_planned_config = None  # Loaded only by commands that use it.
run_sweep = None  # Loaded only by commands that use it.
sweep_status = None  # Loaded only by commands that use it.
from lnl_toolbox.cli.ports import cleanup_ports
from lnl_toolbox.cli.i18n import LocalizedParser, language, text, translated_print

print = translated_print


def _validate_with_registry(config: dict[str, Any], *, check_data: bool):
    global ExperimentService
    if ExperimentService is None:
        from lnl_toolbox.training.service import ExperimentService as ExperimentService
    return ExperimentService().preflight(config, check_data=check_data)


def _should_check_data(args: argparse.Namespace) -> bool:
    """Keep run preflight explicit while retaining legacy --check-data syntax."""

    return not bool(getattr(args, "no_check_data", False))


def _source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", nargs="?", help="实验 recipe 名称或 YAML 路径；与 --recipe、--config 三选一")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--recipe", help="使用内置实验配置；不要同时填写 source 或 --config")
    group.add_argument("--config", type=Path, help="读取实验 YAML 文件；不要同时填写 source 或 --recipe")
    parser.add_argument("--project-root", type=Path, help="显式指定项目根目录")
    parser.add_argument(
        "--data",
        dest="local_dataset",
        help="使用 lnl data register 登记的本机数据集别名，覆盖配置中的数据来源",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = LocalizedParser(
        prog="lnl",
        description="LNL Toolbox 统一命令行入口",
        epilog=(
            "lnl papers list -> lnl show <paper> -> lnl validate <source> -> lnl run <source>"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    parser.add_argument("--lang", choices=("zh", "en"), help=text("输出语言", "Output language"))
    show_paper = sub.add_parser("show", help=text("查看论文摘要", "View a paper summary"))
    show_paper.add_argument("paper_id")
    show_paper.add_argument("--no-interactive", action="store_true", help=argparse.SUPPRESS)
    edit_paper = sub.add_parser("edit", help=text("交互编辑论文参数", "Interactively edit paper parameters"))
    edit_paper.add_argument("paper_id")
    lang = sub.add_parser("language", help=text("保存输出语言偏好", "Save the preferred output language"))
    lang.add_argument("value", choices=("zh", "en"))

    doctor = sub.add_parser("doctor", help="检查环境、项目和数据路径")
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--project-root", type=Path)
    doctor.add_argument("--check-data", action="store_true")

    listing = sub.add_parser("list", help="浏览可运行实验或底层组件")
    list_sub = listing.add_subparsers(dest="list_kind", required=True)
    experiments = list_sub.add_parser("experiments", help="列出内置实验配置")
    experiments.add_argument("--profile", choices=("smoke", "reproduction", "experiment"))
    experiments.add_argument("--dataset")
    experiments.add_argument(
        "--format",
        choices=("human", "tsv", "json"),
        default="human",
        help="输出格式；human 适合阅读，tsv 适合脚本处理（默认: human）",
    )
    experiments.add_argument(
        "--include-conditional",
        action="store_true",
        help="同时显示需要外部 artifact 的条件可用配置",
    )
    components = list_sub.add_parser("components", help="列出底层可组合组件")
    experiments.add_argument(
        "--all",
        dest="all_recipes",
        action="store_true",
        help="show advanced and internal recipes in addition to public templates",
    )
    components.add_argument("--kind")
    components.add_argument(
        "--format",
        choices=("human", "tsv", "json"),
        default="human",
        help="输出格式；human 适合阅读，tsv 适合脚本处理（默认: human）",
    )

    validate = sub.add_parser("validate", help=text("检查参数、数据和依赖，展示运行计划", "Check parameters, data and prerequisites, and display the run plan"))
    _source_options(validate)
    validate.add_argument("--check-data", action="store_true", default=True, help=argparse.SUPPRESS)
    validate.add_argument("--epochs", type=int)
    validate.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    validate.add_argument("--output-dir", type=Path)

    run = sub.add_parser("run", help="运行内置 recipe 或自定义配置")
    _source_options(run)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--resume", type=Path, help="从 checkpoint 文件恢复；保持该运行的配置身份")
    run.add_argument("--epochs", type=int, help="覆盖该方法支持的训练阶段轮数；多阶段方法按现有规则处理")
    run.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--review", action="store_true", help=argparse.SUPPRESS)
    data_check = run.add_mutually_exclusive_group()
    data_check.add_argument(
        "--check-data",
        action="store_true",
        help="compatibility flag; run preflight checks data by default",
    )
    data_check.add_argument(
        "--no-check-data",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    run.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=("override a dotted configuration value; repeatable. "
              "Stage early stopping is boolean, e.g. "
              "--set trainer.early_stopping=true or --set upm.stage1.early_stopping=false"),
    )

    resume = sub.add_parser("resume", help="从运行目录自动恢复")
    resume.add_argument("run_dir", type=Path, help="已有运行目录，包含 resolved_config.yaml；不是 checkpoint 文件路径")
    resume.add_argument("--checkpoint", choices=("last", "best"), default="last", help="选择 last.pt 或 best.pt；已完成的运行不重新训练")

    clean_suite = sub.add_parser("clean-suite", help=text("以多个随机种子运行干净标签训练并汇总结果", "Run and summarize clean-label training across seeds"))
    clean_suite.add_argument("--config", required=True, type=Path)
    clean_suite.add_argument("--seeds", required=True, type=int, nargs="+")
    clean_suite.add_argument("--epochs", type=int)
    clean_suite.add_argument("--output-dir", type=Path)

    data = sub.add_parser("data", help="register and verify machine-local datasets")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_register = data_sub.add_parser("register", help="register a local dataset path")
    data_register.add_argument("alias")
    data_register.add_argument("--adapter", required=True)
    data_register.add_argument("--root", type=Path)
    data_register.add_argument("--path", type=Path)
    data_register.add_argument("--labels", type=Path)
    data_register.add_argument("--noise-variant")
    data_list = data_sub.add_parser("list", help="list registration and verification states")
    data_list.add_argument("name", nargs="?")
    data_list.add_argument("--format", choices=("human", "json"), default="human")
    data_path = data_sub.add_parser("path", help="show a registered dataset path")
    data_path.add_argument("name")
    data_show = data_sub.add_parser("show", help="show one local dataset record")
    data_show.add_argument("alias")
    data_inspect = data_sub.add_parser("inspect", help="load and validate train/test layout")
    data_inspect.add_argument("alias", nargs="?")
    data_inspect.add_argument("--format", choices=("human", "json"), default="human")
    data_inspect.add_argument("--adapter", help=text("直接检查指定类型及路径，无需登记", "Inspect a dataset type and path without registration"))
    data_inspect.add_argument("--root", type=Path)
    data_inspect.add_argument("--path", type=Path)
    data_inspect.add_argument("--split", choices=("train", "validation", "test", "all"), default="all")
    data_noise = data_sub.add_parser("make-noise", help=text("从标签文件生成可复用的噪声清单", "Generate a reusable noise manifest from labels"))
    data_noise.add_argument("labels", type=Path)
    data_noise.add_argument("output", type=Path)
    data_noise.add_argument("--kind", choices=("symmetric", "pairflip"), default="symmetric")
    data_noise.add_argument("--rate", type=float, required=True)
    data_noise.add_argument("--classes", type=int, required=True)
    data_noise.add_argument("--seed", type=int, default=1)
    data_noise.add_argument("--dataset", default="unknown")
    data_declare = data_sub.add_parser("declare", help="record explicit dataset knowledge")
    data_declare.add_argument("alias")
    data_declare.add_argument(
        "--clean-train-labels", choices=("available", "unavailable", "unknown")
    )
    data_declare.add_argument("--noise-status", choices=("clean", "noisy", "unknown"))
    data_declare.add_argument("--noise-origin", choices=("synthetic", "native", "unknown"))
    data_declare.add_argument("--noise-rate", type=float)
    data_declare.add_argument(
        "--noise-rate-status", choices=("known", "estimated", "unknown")
    )
    data_declare.add_argument("--noise-rate-provenance")
    data_declare.add_argument(
        "--noise-manifest", choices=("available", "unavailable", "unknown")
    )
    data_declare.add_argument("--clean-labels-location")
    data_declare.add_argument("--clean-labels-provenance")
    data_declare.add_argument("--notes")
    data_remove = data_sub.add_parser("remove", help="remove a local registration")
    data_remove.add_argument("alias")
    data_verify = data_sub.add_parser("verify", help="run one epoch and store evidence")
    data_verify.add_argument("alias")
    data_verify.add_argument("--recipe")
    data_verify.add_argument("--output-dir", type=Path)
    data_verify.add_argument("--project-root", type=Path)

    mentor = sub.add_parser(
        "mentor", help="prepare and inspect the offline MentorNet artifact"
    )
    mentor_sub = mentor.add_subparsers(dest="mentor_command", required=True)
    mentor_status = mentor_sub.add_parser(
        "status", help="show MentorArtifact readiness for a Student recipe"
    )
    mentor_status.add_argument("--recipe", required=True)
    mentor_status.add_argument("--project-root", type=Path)
    mentor_prepare = mentor_sub.add_parser(
        "prepare", help="prepare trusted Mentor feature records"
    )
    mentor_prepare.add_argument("--config", type=Path, required=True)
    mentor_prepare.add_argument("--output-dir", type=Path, required=True)
    mentor_prepare.add_argument("--project-root", type=Path)
    mentor_prepare.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    mentor_train = mentor_sub.add_parser(
        "train", help="train and freeze a reusable MentorArtifact"
    )
    mentor_train.add_argument("--config", type=Path, required=True)
    mentor_train.add_argument("--output", type=Path, required=True)
    mentor_train.add_argument("--project-root", type=Path)
    mentor_train.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")

    methods = sub.add_parser("methods", help="discover dataset/method compatibility")
    methods_sub = methods.add_subparsers(dest="methods_command", required=True)
    methods_compatible = methods_sub.add_parser(
        "compatible", help="compare a registered dataset with all runners"
    )
    methods_compatible.add_argument("--dataset", required=True)
    methods_compatible.add_argument("--noise-rate-prior", type=float)
    methods_compatible.add_argument("--format", choices=("human", "json"), default="human")

    web = sub.add_parser("web", help="start the local Data Management Web UI")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--no-open", action="store_true")
    web.add_argument("--project-root", type=Path)
    web_sub = web.add_subparsers(dest="web_command")
    web_restart = web_sub.add_parser(
        "restart", help="清理当前 Web 端口上的旧 LNL 进程后重新启动 Web UI"
    )
    # Keep `lnl web --host ... --port ... restart` and
    # `lnl web restart --host ... --port ...` equivalent.  Suppressing these
    # defaults prevents the nested parser from overwriting values parsed by
    # the parent command.
    web_restart.add_argument("--host", default=argparse.SUPPRESS)
    web_restart.add_argument("--port", type=int, default=argparse.SUPPRESS)
    web_restart.add_argument("--no-open", action="store_true", default=argparse.SUPPRESS)
    web_restart.add_argument("--project-root", type=Path, default=argparse.SUPPRESS)

    ports = sub.add_parser("ports", help="查看并清理 LNL Web 监听进程")
    ports_sub = ports.add_subparsers(dest="ports_command", required=True)
    ports_cleanup = ports_sub.add_parser(
        "cleanup", help="清理命令行确认属于 LNL 的旧监听进程"
    )
    ports_cleanup.add_argument(
        "--port",
        dest="ports",
        action="append",
        type=int,
        metavar="PORT",
        help="只清理指定端口；可重复传入。默认扫描所有 LNL 监听端口",
    )
    ports_cleanup.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出候选进程，不终止",
    )

    sweep = sub.add_parser("sweep", help="参数组合实验：按笛卡尔积运行多组实验")
    _source_options(sweep)
    sweep.add_argument(
        "status_path",
        nargs="?",
        type=Path,
        help="参数组合实验目录，用于 'lnl sweep status <path>'",
    )
    sweep.add_argument("--seeds", type=int, nargs="+")
    sweep.add_argument(
        "--matrix",
        action="append",
        default=[],
        metavar="PATH=JSON_ARRAY",
        help="add a typed parameter dimension; repeatable",
    )
    sweep.add_argument("--output-dir", type=Path)
    sweep.add_argument("--dry-run", action="store_true")
    sweep.add_argument("--format", choices=("human", "json"), default="human")
    sweep.add_argument(
        "--no-check-data",
        action="store_true",
        help="参数组合实验预演时跳过数据路径/布局检查",
    )
    sweep.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE"
    )

    compare = sub.add_parser("compare", help="compare completed run directories")
    compare.add_argument("path", type=Path)
    compare.add_argument(
        "--group-by",
        action="append",
        help="grouping field; repeat or provide comma-separated fields",
    )
    compare.add_argument(
        "--require-equal",
        action="append",
        help="fairness invariant within comparable groups; repeatable",
    )
    compare.add_argument("--strict", action="store_true")
    compare.add_argument("--format", choices=("human", "json"), default="human")

    report = sub.add_parser("report", help="write Markdown, CSV, and JSON run reports")
    report.add_argument("path", type=Path)
    report.add_argument("--output-dir", type=Path)
    report.add_argument("--group-by", action="append")
    report.add_argument("--require-equal", action="append")
    report.add_argument("--strict", action="store_true")

    papers = sub.add_parser("papers", help="查看有可运行配置的知名论文")
    paper_sub = papers.add_subparsers(dest="paper_command", required=True)
    paper_list = paper_sub.add_parser("list", help="列出有可运行 config 的论文")
    paper_list.add_argument(
        "--format",
        choices=("human", "tsv", "json"),
        default="human",
        help="输出格式；human 适合阅读，tsv 适合脚本处理（默认: human）",
    )
    show = paper_sub.add_parser("show", help=text("查看论文摘要", "View a paper summary"))
    show.add_argument("paper_id")
    show.add_argument("--no-interactive", action="store_true", help=argparse.SUPPRESS)
    config = paper_sub.add_parser("config", help="查看论文 YAML 配置")
    config.add_argument("paper_id")
    config.add_argument("--profile", choices=("smoke", "reproduction"))
    config.add_argument("--variant")
    config.add_argument("--path-only", action="store_true")
    config.add_argument("--resolved", action="store_true")
    config.add_argument("--project-root", type=Path)
    return parser


def _load_source(args: argparse.Namespace) -> tuple[dict[str, Any], Path, RecipeSpec | None, Path]:
    global DEFAULT_DATA_SERVICE
    if DEFAULT_DATA_SERVICE is None:
        from lnl_toolbox.training.data_service import DEFAULT_DATA_SERVICE as DEFAULT_DATA_SERVICE
    root = args.project_root.expanduser().resolve() if args.project_root else None
    source = getattr(args, "source", None)
    recipe_name = getattr(args, "recipe", None)
    config_arg = getattr(args, "config", None)
    if source is not None and (recipe_name is not None or config_arg is not None):
        raise ValueError("positional source cannot be combined with --recipe or --config")
    if source is not None:
        candidate = Path(source).expanduser()
        if candidate.suffix.lower() in {".yaml", ".yml"} or candidate.is_file():
            config_arg = candidate
        else:
            recipe_name = source
    if recipe_name:
        recipe = recipe_by_id(recipe_name, root)
        config_path = recipe.config_path
        config = load_recipe_config(recipe, check_parameters=True)
    elif config_arg is not None:
        recipe = None
        config_path = config_arg.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"configuration does not exist: {config_path}")
        config = load_yaml(config_path, check_parameters=True)
    else:
        raise ValueError("provide a recipe name or YAML path")
    project = find_project_root(None if recipe is not None else config_path, root)
    config = resolve_config_paths(config, project)
    local_dataset = getattr(args, "local_dataset", None)
    if local_dataset:
        config = DEFAULT_DATA_SERVICE.apply(config, local_dataset)
    return config, config_path, recipe, project


def _noise_description(config: dict[str, Any]) -> str:
    noise = config.get("noise", {}) or {}
    if not noise:
        return "干净标签"
    if noise.get("manifest"):
        return f"外部噪声清单：{noise['manifest']}"
    name = noise.get("name", "configured")
    rate = noise.get("rate")
    return f"生成 {name} 噪声" + (f"，比例={rate}" if rate is not None else "")


def _selection_description(config: dict[str, Any]) -> str:
    evaluation = config.get("evaluation", {}) or {}
    noise = config.get("noise", {}) or {}
    split = evaluation.get("selection_split", "validation")
    targets = noise.get("validation_targets", "clean") if split == "validation" else "test"
    primary = evaluation.get("primary", "accuracy")
    return f"{split} 划分，目标标签={targets}，主要指标={primary}"


def _paper_config_value(config: dict[str, Any], key: str) -> str:
    """Describe top-level or stage-local paper settings without opaque dashes."""

    if key == "epochs":
        trainer = config.get("trainer", {}) or {}
        direct = trainer.get("epochs", config.get("epochs"))
    else:
        section = config.get(key, {}) or {}
        direct = section.get("name") if isinstance(section, dict) else None
    if direct is not None:
        return str(direct)

    staged = []
    for stage_name, stage in config.items():
        if not stage_name.endswith("_stage") or not isinstance(stage, dict):
            continue
        if key == "epochs":
            value = stage.get("epochs")
        else:
            section = stage.get(key, {}) or {}
            value = section.get("name") if isinstance(section, dict) else None
        if value is not None:
            staged.append(f"{stage_name}={value}")
    return "；".join(staged) if staged else "由执行器决定"





def _print_failure_guidance(error: Exception) -> None:
    """Present backend facts only; never validate or infer a readiness state."""
    global ExperimentReadinessError
    if ExperimentReadinessError is None:
        from lnl_toolbox.training.service import ExperimentReadinessError as ExperimentReadinessError
    result = error.compatibility if isinstance(error, ExperimentReadinessError) else None
    context = getattr(error, "prerequisite_context", ())
    if result is not None:
        print(f"Compatibility: {result.status.value.upper()}", file=sys.stderr)
        for reason in result.reasons:
            print(f"  {reason.code}: {reason.message}", file=sys.stderr)
        items = [(item.descriptor, item) for item in result.prerequisites if item.status != "ready"]
    else:
        # An early catalog failure has declarations, not validated statuses.
        items = [(entry["descriptor"], None) for entry in context]
    for index, (descriptor, readiness) in enumerate(items, 1):
        label = "Prerequisite" if readiness is not None else "Prerequisite context (not evaluated)"
        print(f"[{index}] {label}: {descriptor.name}", file=sys.stderr)
        print(f"  type: {descriptor.kind}", file=sys.stderr)
        print(f"  role: {descriptor.key}", file=sys.stderr)
        if readiness is not None:
            print(f"  status: {readiness.status.upper()}", file=sys.stderr)
            if readiness.level is not None:
                print(f"  level: {readiness.level.name}", file=sys.stderr)
            print(f"  reason: {readiness.message}", file=sys.stderr)
        source = next((entry["configured"] for entry in context if entry["descriptor"] == descriptor), ())
        applicable = []
        seen_sources = set()
        for entry in source:
            parent = entry.get("source")
            if parent:
                text = json.dumps(parent, ensure_ascii=False, default=str)
                if text not in seen_sources:
                    print(f"  configured source: {text}", file=sys.stderr)
                    seen_sources.add(text)
            value = entry.get("value")
            if value is not None:
                applicable.append(entry["path"])
                print(f"  config field: {entry['path']} = {value}", file=sys.stderr)
            if entry.get("environment_variable"):
                text = entry.get("environment_value")
                print(f"  source env: {entry['environment_variable']}" + (f" = {text}" if text else " (not set)"), file=sys.stderr)
        if descriptor.requirement:
            print(f"  expected: {descriptor.requirement}", file=sys.stderr)
        if descriptor.obtain:
            print(f"  preparation: {descriptor.obtain}", file=sys.stderr)
        # A declared env/run-directory field may not apply to the configured
        # cached-weight adapter.  Do not present it as that adapter's action.
        if applicable and descriptor.provide:
            print(f"  action: {descriptor.provide}", file=sys.stderr)
    if result is not None:
        represented = {"pretrained:" + item.descriptor.key for item in result.prerequisites}
        for guidance in result.input_guidance:
            if guidance.input_id in represented:
                continue
            print(f"Input ({guidance.category}): {guidance.label}", file=sys.stderr)
            for name, value in (("reason", guidance.missing), ("expected", guidance.expected_value),
                                ("action", guidance.provision)):
                if value:
                    print(f"  {name}: {value}", file=sys.stderr)
            for path in guidance.config_paths:
                print(f"  config field: {'.'.join(path)}", file=sys.stderr)
            if guidance.environment_variable:
                print(f"  source env: {guidance.environment_variable}", file=sys.stderr)


def _print_compatibility_result(result: CompatibilityResult | None) -> None:
    print("Compatibility:")
    if result is None:
        print("  NOT_CHECKED")
        return
    print(f"  {result.status.value.upper()}")
    for reason in result.reasons:
        print(f"  - {reason.code}: {reason.message}")
    for required in result.required_user_inputs:
        print(f"  required: {required}")
    for prerequisite in result.prerequisites:
        level = (
            "NOT_CONFIGURED"
            if prerequisite.level is None
            else prerequisite.level.name
        )
        print(
            f"  prerequisite: {prerequisite.descriptor.name}: "
            f"{prerequisite.status.upper()} ({level})"
        )
        print(f"    {prerequisite.message}")


def _print_plan(
    config: dict[str, Any],
    config_path: Path,
    project: Path,
    compatibility: CompatibilityResult | None = None,
) -> None:
    runner = resolve_runner(config)
    plan = runner.describe(config)
    print("Configuration preview")
    print(f"  configuration: {config_path}")
    print(f"  project root: {project}")
    print(f"  runner: {plan.runner}")
    print(f"  method: {plan.method}")
    print(f"  Training budget: {plan.training_budget}")
    for field in plan.fields:
        print(f"  {field.label}: {field.value}")
    _print_compatibility_result(compatibility)


def _doctor(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve() if args.config else None
    root = find_project_root(config_path, args.project_root)
    failures = 0

    def report(ok: bool, name: str, detail: str) -> None:
        nonlocal failures
        print(f"[{'OK' if ok else 'FAIL'}] {text(name)}: {text(detail)}")
        failures += 0 if ok else 1

    report(sys.version_info >= (3, 10), "Python", platform.python_version())
    report((root / "pyproject.toml").is_file(), "项目根目录", str(root))
    for package in ("numpy", "yaml", "torch", "torchvision"):
        distribution = "pyyaml" if package == "yaml" else package
        try:
            version = metadata.version(distribution)
            report(True, package, version)
        except metadata.PackageNotFoundError:
            report(False, package, '缺少依赖；运行 python -m pip install -e ".[train]"')
    try:
        import torch

        cuda = torch.cuda.is_available()
        detail = torch.cuda.get_device_name(0) if cuda else "不可用，将使用 CPU"
        report(True, "CUDA", detail)
    except ImportError:
        pass
    report((root / "configs" / "experiment").is_dir(), "实验配置", str(root / "configs" / "experiment"))
    output = root / "artifacts"
    report(output.exists() or output.parent.is_dir(), "输出位置", str(output))
    if config_path:
        try:
            config = resolve_config_paths(load_yaml(config_path), root)
            runner = _validate_with_registry(config, check_data=args.check_data)
            report(True, "配置", f"{config_path} -> {runner.name}")
        except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
            report(False, "配置", str(exc))
    return 0 if failures == 0 else 1


def _list_experiments(args: argparse.Namespace) -> int:
    recipes = discover_recipes(
        include_conditional=args.include_conditional,
        public_only=not args.all_recipes,
    )
    if args.profile:
        recipes = tuple(item for item in recipes if item.profile == args.profile)
    if args.dataset:
        recipes = tuple(item for item in recipes if item.dataset.lower() == args.dataset.lower())
    if args.format == "json":
        print(json.dumps([
            {
                "recipe": item.id,
                "profile": item.profile,
                "dataset": item.dataset,
                "noise": item.noise,
                "method": item.method,
                "runner": item.runner,
                "epochs": item.epochs,
            }
            for item in recipes
        ], ensure_ascii=False))
        return 0
    if args.format == "tsv":
        print("recipe\tprofile\tdataset\tnoise\tmethod\trunner\tepochs")
        for item in recipes:
            epochs = item.epochs if item.epochs is not None else "-"
            print(
                f"{item.id}\t{item.profile}\t{item.dataset}\t{item.noise}\t"
                f"{item.method}\t{item.runner}\t{epochs}"
            )
        return 0

    filters = []
    if args.profile:
        filters.append(f"规模={args.profile}")
    if args.dataset:
        filters.append(f"数据集={args.dataset}")
    scope = f"（{'，'.join(filters)}）" if filters else ""
    print(f"找到 {len(recipes)} 个可运行实验{scope}")
    print("每个 recipe 都是一份可直接预检和运行的完整配置。")
    if not recipes:
        print("没有符合条件的实验；请调整 --profile 或 --dataset。")
        return 0

    total = len(recipes)
    for index, item in enumerate(recipes, start=1):
        epochs = f"{item.epochs} epochs" if item.epochs is not None else "由执行器决定"
        print()
        print(f"[{index}/{total}] {item.id}")
        print(f"  数据集：{item.dataset}    噪声：{item.noise}")
        print(f"  方法：{item.method}    执行器：{item.runner}")
        print(f"  规模：{item.profile}    训练轮数：{epochs}")
        print(
            f"  IMPLEMENTATION={item.implementation_status} | "
            f"FIDELITY={item.configuration_fidelity} | "
            f"REPRODUCTION={item.reproduction_status} | "
            f"AVAILABILITY={item.availability}"
        )
        print("  先预览：")
        print(f"    lnl validate --recipe {item.id}")
    return 0


def _list_components(args: argparse.Namespace) -> int:
    from lnl_toolbox.plugins.builtin import create_builtin_catalog

    catalog = create_builtin_catalog()
    values = catalog.find(kind=args.kind) if args.kind else catalog.find()
    if not values:
        raise ValueError(f"no components found for kind {args.kind!r}")
    if args.format == "json":
        print(json.dumps([
            {
                "kind": item.kind,
                "name": item.name,
                "capabilities": sorted(item.capabilities),
                "paper": item.metadata.get("paper"),
            }
            for item in values
        ], ensure_ascii=False))
        return 0
    if args.format == "tsv":
        print("kind\tname\tcapabilities\tpaper")
        for item in values:
            paper = item.metadata.get("paper", "-")
            capabilities = ",".join(sorted(item.capabilities)) or "-"
            print(f"{item.kind}\t{item.name}\t{capabilities}\t{paper}")
        return 0

    kinds = sorted({item.kind for item in values})
    scope = f"（类型={args.kind}）" if args.kind else ""
    print(f"找到 {len(values)} 个可组合组件，共 {len(kinds)} 类{scope}")
    print("组件是构建算法的零件，不一定代表一篇可直接运行的完整论文。")
    for kind in kinds:
        members = [item for item in values if item.kind == kind]
        print()
        print(f"【{kind}】{len(members)} 个")
        for index, item in enumerate(members, start=1):
            capabilities = "、".join(sorted(item.capabilities)) or "未声明"
            paper = item.metadata.get("paper", "未关联论文")
            print(f"  {index}. {item.name}")
            print(f"     能力：{capabilities}")
            print(f"     论文：{paper}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    # Keep one preflight/plan implementation, including the Web's legacy path.
    args.dry_run = True
    args.review = False
    args.no_check_data = False
    args.resume = None
    return _run(args)


def _run(args: argparse.Namespace) -> int:
    global ExperimentService
    if ExperimentService is None:
        from lnl_toolbox.training.service import ExperimentService as ExperimentService
    review = bool(getattr(args, "review", False))
    if review and not args.dry_run:
        raise ValueError("--review requires --dry-run")
    if review:
        print("Quick Start phase: validating", flush=True)
    config, path, recipe, project = _load_source(args)
    config = apply_override_assignments(config, args.overrides)
    if args.epochs is not None:
        apply_epoch_override(config, args.epochs)
    if args.no_check_data and not args.dry_run:
        raise ValueError("--no-check-data is only valid together with --dry-run")
    service = ExperimentService()
    if review:
        service.preflight(config, check_data=False)
        print("Quick Start phase: rehearsing", flush=True)
    service.preflight(config, check_data=_should_check_data(args))
    if args.dry_run:
        if args.output_dir is not None:
            output = args.output_dir.expanduser().resolve()
            print("Planned training artifacts (created when training starts)")
            print(f"  artifact directory: {output}")
            if review:
                print(f"  training configuration: {output / 'config.yaml'}")
            print(f"  resolved configuration: {output / 'resolved_config.yaml'}")
        _print_plan(config, path, project, service.last_compatibility)
        return 0
    result = service.run(
        config,
        args.output_dir,
        args.resume,
        recipe=recipe.id if recipe is not None else None,
    )
    print(f"运行完成: {result}")
    return 0


def _resume(args: argparse.Namespace) -> int:
    global ExperimentService
    if ExperimentService is None:
        from lnl_toolbox.training.service import ExperimentService as ExperimentService
    result = ExperimentService().resume(args.run_dir, args.checkpoint)
    print(f"resume complete: {result}")
    return 0


def _clean_suite(args: argparse.Namespace) -> int:
    from lnl_toolbox.training.clean_baseline import run_seed_suite

    path = args.config.expanduser().resolve()
    config = resolve_config_paths(load_yaml(path), find_project_root(path))
    if args.epochs is not None:
        config["trainer"]["epochs"] = args.epochs
    output = args.output_dir or Path(config.get("output_root", "artifacts/runs")) / "seed-suite"
    print(run_seed_suite(config, args.seeds, output))
    return 0


def _sweep(args: argparse.Namespace) -> int:
    global ExperimentService, plan_sweep, resolve_planned_config, run_sweep, sweep_status
    if ExperimentService is None:
        from lnl_toolbox.training.service import ExperimentService as ExperimentService
    if plan_sweep is None:
        from lnl_toolbox.training.sweep import plan_sweep as plan_sweep
    if resolve_planned_config is None:
        from lnl_toolbox.training.sweep import resolve_planned_config as resolve_planned_config
    if run_sweep is None:
        from lnl_toolbox.training.sweep import run_sweep as run_sweep
    if sweep_status is None:
        from lnl_toolbox.training.sweep import sweep_status as sweep_status
    if args.source == "status":
        if args.status_path is None:
            raise ValueError("provide a sweep directory after 'lnl sweep status'")
        if args.seeds or args.matrix or args.overrides or args.output_dir or args.dry_run:
            raise ValueError("sweep status cannot be combined with planning options")
        value = sweep_status(args.status_path)
        if getattr(args, "format", "human") == "json":
            print(json.dumps(value, ensure_ascii=False))
            return 0
        print("参数组合实验")
        print(f"  ID: {value['sweep_id']}")
        print(f"  Path: {value['root']}")
        print("\nStatus")
        for status in ("completed", "running", "failed", "interrupted", "pending"):
            print(f"  {status:<12} {value['counts'][status]}")
        print(f"\nProgress\n  {value['completed']} / {value['total']} completed")
        if value["failed_runs"]:
            print("\nFAILED RUNS")
            for item in value["failed_runs"]:
                overrides = " ".join(
                    f"{path}={entry}" for path, entry in item["overrides"].items()
                ) or "-"
                print(f"  seed={item['seed']} {overrides} reason={item['reason']}")
        return 0
    if args.status_path is not None:
        raise ValueError("unexpected extra sweep path; use 'lnl sweep status <path>'")
    config, _path, recipe, _project, seeds, matrix = _load_sweep_source(args)
    cli_matrix = _parse_sweep_matrix(args.matrix)
    duplicates = sorted(set(matrix).intersection(cli_matrix))
    if duplicates:
        raise ValueError(
            "sweep matrix path is defined in YAML and CLI: " + ", ".join(duplicates)
        )
    matrix = {**matrix, **cli_matrix}
    config = apply_override_assignments(config, args.overrides)
    if args.no_check_data and not args.dry_run:
        raise ValueError("--no-check-data is only valid together with --dry-run")
    plan = plan_sweep(
        config,
        seeds,
        matrix=matrix,
        output_dir=args.output_dir,
        recipe=recipe.id if recipe is not None else None,
    )
    service = ExperimentService()
    for planned in plan.runs:
        service.preflight(
            resolve_planned_config(config, planned),
            check_data=not args.no_check_data,
        )
    if args.dry_run:
        _print_sweep_plan(plan)
        return 0
    result = run_sweep(
        config,
        seeds,
        matrix=matrix,
        output_dir=args.output_dir,
        recipe=recipe.id if recipe is not None else None,
        service=service,
    )
    print(f"sweep: {result.root}")
    print(
        f"completed={result.completed} skipped={result.skipped} failed={result.failed}"
    )
    return 1 if result.failed else 0


def _load_sweep_source(args: argparse.Namespace):
    source = args.source or args.recipe or args.config
    if source is None:
        raise ValueError("provide a recipe name or sweep/experiment YAML path")
    candidate = Path(source).expanduser()
    if candidate.is_file() and candidate.suffix.lower() in {".yaml", ".yml"}:
        value = load_yaml(candidate)
        if "base" in value or "matrix" in value:
            if int(value.get("version", 1)) != 1:
                raise ValueError("sweep spec version must be 1")
            base = value.get("base")
            if not isinstance(base, dict):
                raise ValueError("sweep spec requires a base mapping")
            choices = [key for key in ("recipe", "config") if base.get(key) is not None]
            if len(choices) != 1:
                raise ValueError("sweep base requires exactly one of recipe or config")
            base_source = str(base[choices[0]])
            if choices[0] == "config":
                path = Path(base_source).expanduser()
                if not path.is_absolute():
                    base_source = str((candidate.resolve().parent / path).resolve())
            namespace = argparse.Namespace(
                source=base_source,
                recipe=None,
                config=None,
                project_root=args.project_root,
                local_dataset=args.local_dataset,
            )
            config, path, recipe, project = _load_source(namespace)
            configured_seeds = value.get("seeds")
            seeds = args.seeds if args.seeds is not None else configured_seeds
            if seeds is None:
                seeds = [int(config.get("seed", 1))]
            if not isinstance(seeds, list):
                raise ValueError("sweep seeds must be a list")
            matrix = value.get("matrix", {}) or {}
            if not isinstance(matrix, dict):
                raise ValueError("sweep matrix must be a mapping")
            return config, path, recipe, project, seeds, matrix
    config, path, recipe, project = _load_source(args)
    seeds = args.seeds if args.seeds is not None else [int(config.get("seed", 1))]
    return config, path, recipe, project, seeds, {}


def _parse_sweep_matrix(assignments: list[str]) -> dict[str, list[Any]]:
    matrix: dict[str, list[Any]] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"matrix assignment must use PATH=JSON_ARRAY: {assignment}")
        path, encoded = assignment.split("=", 1)
        path = path.strip()
        if not path:
            raise ValueError("sweep matrix path must not be empty")
        if path in matrix:
            raise ValueError(f"duplicate sweep matrix path: {path}")
        try:
            values = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON array for sweep matrix {path}: {exc.msg}") from exc
        if not isinstance(values, list):
            raise ValueError(f"sweep matrix {path} must be a JSON array")
        if not values:
            raise ValueError(f"sweep matrix {path} must not be empty")
        matrix[path] = values
    return matrix


def _print_sweep_plan(plan) -> None:
    print("参数组合实验计划")
    print(f"\nBase:\n  {plan.recipe}")
    print("\nMatrix:")
    if plan.matrix:
        for path, values in plan.matrix:
            print(f"  {path:<24} {', '.join(map(str, values))}")
    else:
        print("  (none)")
    print(f"\nSeeds:\n  {', '.join(map(str, plan.seeds))}")
    print(f"\nTotal runs:\n  {len(plan.runs)}")
    print(f"\nOutput directory:\n  {plan.root}")
    print("\nRun plan")
    indexed = list(enumerate(plan.runs, start=1))
    preview = indexed if len(indexed) <= 20 else indexed[:10]
    for index, planned in preview:
        overrides = " ".join(
            f"{path}={value}" for path, value in planned.overrides
        )
        suffix = f" {overrides}" if overrides else ""
        print(f"  #{index:02d} seed={planned.seed}{suffix}")
    if len(indexed) > 20:
        print(f"  ... {len(indexed) - 15} runs omitted ...")
        for index, planned in indexed[-5:]:
            overrides = " ".join(
                f"{path}={value}" for path, value in planned.overrides
            )
            suffix = f" {overrides}" if overrides else ""
            print(f"  #{index:02d} seed={planned.seed}{suffix}")


def _compare(args: argparse.Namespace) -> int:
    global compare_runs
    if compare_runs is None:
        from lnl_toolbox.evaluation.run_comparison import compare_runs as compare_runs
    comparison = compare_runs(
        args.path,
        group_by=_comparison_fields(args.group_by),
        require_equal=_comparison_fields(args.require_equal),
        strict=args.strict,
    )
    if args.format == "json":
        print(json.dumps(comparison, ensure_ascii=False))
        return 1 if args.strict and comparison.get("excluded_runs") else 0
    grouping = comparison.get("group_by", ["method", "noise.rate", "primary_metric.name"])
    dimensions = [field for field in grouping if field != "primary_metric.name"]
    labels = {"noise.rate": "NOISE", "method": "METHOD"}
    header = [labels.get(field, field.upper()) for field in dimensions]
    print("\t".join([*header, "METRIC", "N", "MEAN", "STD", "MEDIAN", "MIN", "MAX"]))
    for row in comparison["summaries"]:
        group = row.get("group", {})
        values = [str(group.get(field, row.get(field, "-"))) for field in dimensions]
        values.extend(
            [
                str(row["metric"]),
                str(row["n"]),
                f"{row['mean']:.6f}",
                f"{row['std']:.6f}",
                f"{row['median']:.6f}",
                f"{row['min']:.6f}",
                f"{row['max']:.6f}",
            ]
        )
        print("\t".join(values))
    print("\nCompatibility")
    for field, status in comparison.get("compatibility", {}).items():
        print(f"  {field:<24} {status}")
    for warning in comparison["warnings"]:
        print(warning)
    return 1 if args.strict and comparison.get("excluded_runs") else 0


def _report(args: argparse.Namespace) -> int:
    global compare_runs, write_report
    if compare_runs is None:
        from lnl_toolbox.evaluation.run_comparison import compare_runs as compare_runs
    if write_report is None:
        from lnl_toolbox.evaluation.run_comparison import write_report as write_report
    comparison = compare_runs(
        args.path,
        group_by=_comparison_fields(args.group_by),
        require_equal=_comparison_fields(args.require_equal),
        strict=args.strict,
    )
    output_dir = args.output_dir or args.path
    for path in write_report(comparison, output_dir).values():
        print(path)
    return 1 if args.strict and comparison.get("excluded_runs") else 0


def _comparison_fields(values: list[str] | None):
    if values is None:
        return None
    fields = tuple(
        field.strip()
        for value in values
        for field in value.split(",")
        if field.strip()
    )
    if not fields:
        raise ValueError("comparison field list must not be empty")
    return fields






def _paper_list(args: argparse.Namespace) -> int:
    from lnl_toolbox.cli.paper_editor import list_papers
    return list_papers(args.format)


def _paper_config(args: argparse.Namespace) -> int:
    root = args.project_root.expanduser().resolve() if args.project_root else None
    paper = paper_by_id(args.paper_id, root)
    _selection, recipe = select_paper_config(
        paper, profile=args.profile, variant=args.variant, root=root
    )
    if args.path_only:
        print(recipe.config_path)
        return 0
    config = load_recipe_config(recipe)
    if args.resolved:
        project = find_project_root(None, root)
        config = resolve_config_paths(config, project)
    import yaml

    print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), end="")
    return 0










def _print_local_dataset(record) -> None:
    print(f"{record.alias}: {record.adapter}")
    print(f"  state: {record.effective_state}")
    for key in ("root", "path", "noise_path", "noise_variant"):
        if record.data.get(key) not in {None, ""}:
            print(f"  {key}: {record.data[key]}")
    if record.evidence:
        if record.evidence.get("run_dir"):
            print(f"  verification run: {record.evidence['run_dir']}")
        if record.evidence.get("data_fingerprint"):
            print(f"  data fingerprint: {record.evidence['data_fingerprint']}")
    if record.error:
        print(f"  error: {record.error}")


def _print_dataset_report(report: DatasetStatusReport) -> None:
    print(f"Dataset          {report.name}")
    print(f"Adapter          {report.adapter}")
    print(f"Status           {report.status.upper()}")
    print(f"Location         {report.location or '-'}")
    if report.train_samples is not None:
        print(f"Train samples    {report.train_samples}")
    if report.test_samples is not None:
        print(f"Test samples     {report.test_samples}")
    if report.classes is not None:
        print(f"Classes          {report.classes}")
    if report.fingerprint:
        print(f"Fingerprint      {report.fingerprint}")
    if report.training_evidence:
        print("Training check   VERIFIED")
        if report.training_evidence.get("run_dir"):
            print(f"Verification run {report.training_evidence['run_dir']}")
    if report.error:
        print(f"Error            {report.error}")


def _print_dataset_profile(report: DatasetStatusReport) -> None:
    _print_dataset_report(report)
    profile = report.profile
    if profile is None:
        return
    print(f"Task             {profile.task}")
    print(f"Modality         {profile.modality.value}")
    print(f"Input shape      {profile.input_shape or '-'}")
    print(f"Channels         {profile.channels if profile.channels is not None else '-'}")
    print(f"Available splits {', '.join(profile.available_splits)}")
    print("Sample counts    " + json.dumps(dict(profile.sample_counts_by_split), sort_keys=True))
    print("Class distribution " + json.dumps(dict(profile.class_distribution_by_split), sort_keys=True))
    print(f"Observed labels  {profile.observed_train_labels.value}")
    print(f"Clean labels     {profile.clean_train_labels.value}")
    print(f"Noise status     {profile.noise.status.value}")
    print(f"Noise origin     {profile.noise.origin.value}")
    print(f"Noise rate       {profile.noise.rate.status.value}")
    if profile.noise.rate.value is not None:
        print(f"Noise rate value {profile.noise.rate.value}")
    if profile.noise.rate.provenance:
        print(f"Noise provenance {profile.noise.rate.provenance}")
    print(f"Stable indices   {profile.stable_indices.value}")
    print(f"Profile fingerprint {profile.fingerprint}")


def _print_dataset_table(reports) -> None:
    print(f"{'DATASET':<24} {'ADAPTER':<30} {'STATUS':<12} LOCATION")
    for report in reports:
        print(
            f"{report.name:<24} {report.adapter:<30} "
            f"{report.status:<12} {report.location or '-'}"
        )


def _data_command(args: argparse.Namespace) -> int:
    if args.data_command == "make-noise":
        import numpy as np
        from lnl_toolbox.noise import generate_pairflip, generate_symmetric

        labels = np.load(args.labels)
        generator = generate_symmetric if args.kind == "symmetric" else generate_pairflip
        manifest = generator(labels, args.classes, args.rate, args.seed, args.dataset)
        manifest.save(args.output)
        print(f"saved {args.output}; realized_rate={manifest.realized_rate:.4f}")
        return 0
    global DEFAULT_DATA_SERVICE
    if DEFAULT_DATA_SERVICE is None:
        from lnl_toolbox.training.data_service import DEFAULT_DATA_SERVICE as DEFAULT_DATA_SERVICE
    service = DEFAULT_DATA_SERVICE
    if args.data_command == "list":
        reports = service.list_datasets() if args.name is None else (service.status(args.name),)
        if args.format == "json":
            print(json.dumps([item.to_dict() for item in reports], ensure_ascii=False))
        else:
            if args.name is None:
                _print_dataset_table(reports)
            else:
                _print_dataset_report(reports[0])
        return 0
    if args.data_command == "path":
        path = service.path(args.name)
        print("-" if path is None else path)
        return 0
    if args.data_command == "show":
        _print_local_dataset(service.record(args.alias))
        return 0
    if args.data_command == "register":
        adapter = service.registry.get(args.adapter).name
        data: dict[str, Any] = {"name": adapter}
        if args.root is not None:
            data["root"] = args.root
        if args.path is not None:
            data["path"] = args.path
        if args.labels is not None:
            data["noise_path"] = args.labels
        if args.noise_variant is not None:
            data["noise_variant"] = args.noise_variant
        service.register(args.alias, adapter, data)
        _print_local_dataset(service.record(args.alias))
        print("Registration does not prove trainability; run 'lnl data verify'.")
        return 0
    if args.data_command == "remove":
        service.remove(args.alias)
        print(f"Removed local dataset registration: {args.alias}")
        return 0
    if args.data_command == "inspect":
        if args.adapter:
            spec = service.validate_config({"seed": 0, "data": {"name": args.adapter, "root": args.root, "path": args.path}})
            splits = ("train", "validation", "test") if args.split == "all" else (args.split,)
            summaries = []
            for split in splits:
                try:
                    current = service.registry.load(spec, split, seed=0)
                except ValueError:
                    continue
                summaries.append({"dataset": current.dataset, "split": current.split, "samples": len(current), "classes": current.num_classes, "identity": current.identity.to_dict()})
            print(json.dumps(summaries, ensure_ascii=False, indent=2))
            return 0
        if not args.alias:
            raise ValueError(text("请提供已登记数据集名称或使用 --adapter 指定数据类型", "Provide a registered dataset name or specify --adapter"))
        if args.root is not None or args.path is not None or args.split != "all":
            raise ValueError(text("--root、--path 和 --split 需要与 --adapter 一起使用", "--root, --path and --split require --adapter"))
        report = service.inspect(args.alias)
        if args.format == "json":
            print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
        else:
            _print_dataset_profile(report)
        if report.status != "ready":
            return 2
        if args.format == "human":
            print("Layout validated; training has not yet been verified.")
        return 0
    if args.data_command == "declare":
        updates: dict[str, Any] = {}
        for argument, field in (
            (args.clean_train_labels, "clean_train_labels"),
            (args.noise_status, "noise_status"),
            (args.noise_origin, "noise_origin"),
            (args.noise_manifest, "noise_manifest"),
            (args.clean_labels_location, "clean_labels_location"),
            (args.clean_labels_provenance, "clean_labels_provenance"),
            (args.notes, "semantic_notes"),
        ):
            if argument is not None:
                updates[field] = argument
        if args.noise_rate is not None or args.noise_rate_status is not None:
            status = args.noise_rate_status or "known"
            updates["noise_rate"] = NoiseRateInfo(
                NoiseRateStatus(status),
                args.noise_rate,
                args.noise_rate_provenance,
            ).to_dict()
        elif args.noise_rate_provenance is not None:
            raise ValueError("--noise-rate-provenance requires --noise-rate")
        if not updates:
            raise ValueError("provide at least one dataset declaration")
        capabilities = service.update_declarations(args.alias, updates)
        print(f"Declarations updated: {args.alias}")
        print(f"  clean train labels: {capabilities.clean_train_labels.value}")
        print(f"  noise status: {capabilities.noise_status.value}")
        print(f"  noise origin: {capabilities.noise_origin.value}")
        print(f"  noise rate: {capabilities.noise_rate.status.value}")
        return 0
    if args.data_command == "verify":
        record = service.record(args.alias)
        root = args.project_root.expanduser().resolve() if args.project_root else None
        project = find_project_root(None, root)
        config = None
        recipe = None
        if args.recipe:
            recipe = recipe_by_id(args.recipe, root)
            config = resolve_config_paths(load_recipe_config(recipe), project)
            config = service.apply(config, args.alias)
            runner = resolve_runner(config)
            if runner.budget_path is None and "epochs" in config:
                # Internal verification accepts an unambiguous legacy top-level
                # budget even when the public --epochs shortcut is intentionally
                # disabled for that runner.
                config["epochs"] = 1
            else:
                apply_epoch_override(config, 1)
        destination = args.output_dir or (
            project / "artifacts" / "data-verification"
            / f"{record.alias}-{record.signature[:8]}"
        )
        report, _run_dir = service.verify(
            args.alias,
            config,
            destination,
            recipe=None if recipe is None else recipe.id,
        )
        _print_dataset_report(report)
        profile = "automatic dataset profile" if recipe is None else f"recipe {recipe.id}"
        print(f"Training verified by a completed one-epoch run ({profile}).")
        return 0
    raise ValueError(f"unknown data command: {args.data_command}")


def _methods_command(args: argparse.Namespace) -> int:
    global ExperimentService
    if ExperimentService is None:
        from lnl_toolbox.training.service import ExperimentService as ExperimentService
    if args.methods_command != "compatible":
        raise ValueError(f"unknown methods command: {args.methods_command}")
    results = ExperimentService().list_compatible_methods(
        args.dataset,
        method_noise_rate_prior=args.noise_rate_prior,
    )
    if args.format == "json":
        print(json.dumps([item.to_dict() for item in results], ensure_ascii=False))
        return 0
    print(f"Dataset: {args.dataset}")
    labels = (
        (CompatibilityStatus.COMPATIBLE, "Compatible"),
        (CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS, "Requires additional input"),
        (CompatibilityStatus.INCOMPATIBLE, "Unavailable"),
    )
    for status, label in labels:
        print(f"\n{label}:")
        members = [item for item in results if item.status is status]
        if not members:
            print("  -")
            continue
        for result in members:
            print(f"  {result.method}")
            for reason in result.reasons:
                print(f"    {reason.code}: {reason.message}")
            for required in result.required_user_inputs:
                print(f"    required: {required}")
    return 0


def _mentor_command(args: argparse.Namespace) -> int:
    root = find_project_root(None, args.project_root)
    if args.mentor_command == "status":
        recipe = recipe_by_id(args.recipe, root)
        config = resolve_config_paths(load_recipe_config(recipe), root)
        status = mentornet_preparation_status(
            config, root, student_recipe=recipe.id
        )
        if status is None:
            raise ValueError(f"recipe {recipe.id!r} is not a MentorNet workflow")
        print(f"MentorArtifact: {str(status['status']).upper()}")
        print(f"  artifact: {status['artifact_path']}")
        print(
            "  artifact validation: "
            + ("PASS" if status["artifact_ready"] else "NOT READY")
        )
        print(
            "  Mentor features: "
            + ("READY" if status["feature_ready"] else "NOT READY")
        )
        if status["artifact_error"]:
            print(f"  error: {status['artifact_error']}")
        for step, command in status["commands"].items():
            print(f"  {step}: {command}")
        return 0 if status["artifact_ready"] else 1

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Mentor configuration does not exist: {config_path}")
    config = load_yaml(config_path)
    if args.mentor_command == "prepare":
        config = apply_override_assignments(config, args.overrides)
        from lnl_toolbox.training.mentor_learning import (
            prepare_trusted_mentor_features,
        )

        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = (root / output_dir).resolve()
        feature_path = prepare_trusted_mentor_features(config, output_dir)
        print(f"Mentor features: READY")
        print(feature_path)
        return 0
    if args.mentor_command == "train":
        config = apply_override_assignments(config, args.overrides)
        from lnl_toolbox.training.mentor_learning import train_mentor_artifact

        feature_data = Path(str(config.get("feature_data", ""))).expanduser()
        if not feature_data.is_absolute():
            feature_data = (root / feature_data).resolve()
        config = dict(config)
        config["feature_data"] = str(feature_data)
        output = args.output.expanduser()
        if not output.is_absolute():
            output = (root / output).resolve()
        train_mentor_artifact(config, output)
        print("MentorArtifact: READY")
        print(output)
        return 0
    raise ValueError(f"unknown mentor command: {args.mentor_command}")


def _web(args: argparse.Namespace) -> int:
    if getattr(args, "web_command", None) == "restart":
        # Reuse the same guarded listener discovery as `lnl ports cleanup`.
        # Restrict the cleanup to the port that will be started so a restart
        # cannot terminate an unrelated LNL service on another port.
        cleanup_code = cleanup_ports([args.port])
        if cleanup_code != 0:
            return cleanup_code
    root = find_project_root(None, args.project_root)
    server = root / "web" / "command_console.py"
    # `lnl web` is a user-facing entry point.  When launched outside the
    # checkout, project-root discovery starts at the current directory and
    # cannot find the repository's combined WebUI server; editable installs
    # still expose the source checkout through this module path.
    if not server.is_file():
        source_root = Path(__file__).resolve().parents[3]
        source_server = source_root / "web" / "command_console.py"
        if source_server.is_file():
            root, server = source_root, source_server
    if not server.is_file():
        raise FileNotFoundError(f"Web UI server does not exist: {server}")
    command = [
        sys.executable,
        str(server),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if not args.no_open:
        command.append("--open")
    return int(subprocess.call(command, cwd=root))


def _ports_command(args: argparse.Namespace) -> int:
    if args.ports_command != "cleanup":
        raise ValueError(f"unknown ports command: {args.ports_command}")
    return cleanup_ports(args.ports, dry_run=args.dry_run)


def _dispatch(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "language":
            from lnl_toolbox.cli.i18n import save_language
            save_language(args.value)
            print(text("语言偏好已保存。", "Language preference saved."))
            return 0
        if args.command == "show" or (args.command == "papers" and args.paper_command == "show"):
            from lnl_toolbox.cli.paper_editor import resolve_paper, show_paper
            return show_paper(resolve_paper(args.paper_id))
        if args.command == "edit":
            from lnl_toolbox.cli.paper_editor import edit_paper, resolve_paper
            return edit_paper(resolve_paper(args.paper_id))
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "list":
            return _list_experiments(args) if args.list_kind == "experiments" else _list_components(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "run":
            return _run(args)
        if args.command == "resume":
            return _resume(args)
        if args.command == "clean-suite":
            return _clean_suite(args)
        if args.command == "data":
            return _data_command(args)
        if args.command == "methods":
            return _methods_command(args)
        if args.command == "mentor":
            return _mentor_command(args)
        if args.command == "web":
            return _web(args)
        if args.command == "ports":
            return _ports_command(args)
        if args.command == "sweep":
            return _sweep(args)
        if args.command == "compare":
            return _compare(args)
        if args.command == "report":
            return _report(args)
        if args.command == "papers":
            if args.paper_command == "list":
                return _paper_list(args)
            return _paper_config(args)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(text("错误", "Error") + ": " + text(str(exc)), file=sys.stderr)
        try:
            _print_failure_guidance(exc)
        except Exception:
            # Diagnostic rendering must preserve the original error and exit code.
            pass
        return 2
    parser.error("unknown command")
    return 2


def main(argv: list[str] | None = None) -> int:
    from lnl_toolbox.cli.i18n import reset_language, select_language
    values = list(sys.argv[1:] if argv is None else argv)
    values = [part for value in values for part in (["--lang", value.split("=", 1)[1]] if value.startswith("--lang=") else [value])]
    # Accept --lang before or after subcommands without altering Web argv.
    explicit = None
    while "--lang" in values:
        index = values.index("--lang")
        if index + 1 >= len(values) or values[index + 1] not in {"zh", "en"}:
            build_parser().error("--lang requires zh or en")
        explicit = values[index + 1]
        del values[index:index + 2]
    try:
        token = select_language(explicit)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        return _dispatch(values)
    finally:
        reset_language(token)


if __name__ == "__main__":
    raise SystemExit(main())
