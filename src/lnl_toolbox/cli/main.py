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
    select_paper_config,
    validate_config,
)
from lnl_toolbox.composition import (
    apply_overrides,
    validate_composition,
    write_composed_config,
)
from lnl_toolbox.core.config_overrides import apply_override_assignments
from lnl_toolbox.evaluation.run_comparison import compare_runs, write_report
from lnl_toolbox.training.data_service import DEFAULT_DATA_SERVICE, DatasetStatusReport
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner, runner_names
from lnl_toolbox.training.service import ExperimentService
from lnl_toolbox.training.sweep import (
    plan_sweep,
    resolve_planned_config,
    run_sweep,
    sweep_status,
)


def _validate_with_registry(config: dict[str, Any], *, check_data: bool):
    return ExperimentService().preflight(config, check_data=check_data)


def _should_check_data(args: argparse.Namespace) -> bool:
    """Keep run preflight explicit while retaining legacy --check-data syntax."""

    return not bool(getattr(args, "no_check_data", False))


def _source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", nargs="?", help="recipe name or YAML path")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--recipe", help="内置实验配置名称")
    group.add_argument("--config", type=Path, help="自定义 YAML 配置")
    parser.add_argument("--project-root", type=Path, help="显式指定项目根目录")
    parser.add_argument(
        "--data",
        dest="local_dataset",
        help="machine-local dataset alias registered with 'lnl data register'",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lnl",
        description="LNL Toolbox 统一命令行入口",
        epilog=(
            "推荐顺序: lnl doctor -> lnl list experiments -> "
            "lnl run <source> --dry-run -> lnl run <source>"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

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
        choices=("human", "tsv"),
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
        choices=("human", "tsv"),
        default="human",
        help="输出格式；human 适合阅读，tsv 适合脚本处理（默认: human）",
    )

    validate = sub.add_parser("validate", help="训练前静态检查配置")
    _source_options(validate)
    validate.add_argument("--check-data", action="store_true")

    run = sub.add_parser("run", help="运行内置 recipe 或自定义配置")
    _source_options(run)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--resume", type=Path)
    run.add_argument("--epochs", type=int)
    run.add_argument("--dry-run", action="store_true")
    data_check = run.add_mutually_exclusive_group()
    data_check.add_argument(
        "--check-data",
        action="store_true",
        help="compatibility flag; run preflight checks data by default",
    )
    data_check.add_argument(
        "--no-check-data",
        action="store_true",
        help="skip dataset path/layout checks during dry-run",
    )
    run.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="override an existing dotted configuration value; repeatable",
    )

    resume = sub.add_parser("resume", help="从运行目录自动恢复")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--checkpoint", choices=("last", "best"), default="last")

    data = sub.add_parser("data", help="register and verify machine-local datasets")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_register = data_sub.add_parser("register", help="register a local dataset path")
    data_register.add_argument("alias")
    data_register.add_argument("--adapter", required=True)
    data_register.add_argument("--root", type=Path)
    data_register.add_argument("--path", type=Path)
    data_register.add_argument("--labels", type=Path)
    data_register.add_argument("--noise-variant")
    data_sub.add_parser("list", help="list registration and verification states")
    data_status = data_sub.add_parser("status", help="show dataset readiness")
    data_status.add_argument("name", nargs="?")
    data_path = data_sub.add_parser("path", help="show a registered dataset path")
    data_path.add_argument("name")
    data_show = data_sub.add_parser("show", help="show one local dataset record")
    data_show.add_argument("alias")
    data_inspect = data_sub.add_parser("inspect", help="load and validate train/test layout")
    data_inspect.add_argument("alias")
    data_remove = data_sub.add_parser("remove", help="remove a local registration")
    data_remove.add_argument("alias")
    data_verify = data_sub.add_parser("verify", help="run one epoch and store evidence")
    data_verify.add_argument("alias")
    data_verify.add_argument("--recipe")
    data_verify.add_argument("--output-dir", type=Path)
    data_verify.add_argument("--project-root", type=Path)

    web = sub.add_parser("web", help="start the local Data Management Web UI")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--no-open", action="store_true")
    web.add_argument("--project-root", type=Path)

    sweep = sub.add_parser("sweep", help="run multiple seeds sequentially and resumably")
    _source_options(sweep)
    sweep.add_argument(
        "status_path",
        nargs="?",
        type=Path,
        help="sweep directory when using 'lnl sweep status <path>'",
    )
    sweep.add_argument("--seeds", type=int, nargs="+")
    sweep.add_argument("--output-dir", type=Path)
    sweep.add_argument("--dry-run", action="store_true")
    sweep.add_argument(
        "--no-check-data",
        action="store_true",
        help="skip dataset path/layout checks during sweep dry-run",
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

    report = sub.add_parser("report", help="write Markdown, CSV, and JSON run reports")
    report.add_argument("path", type=Path)
    report.add_argument("--output-dir", type=Path)
    report.add_argument("--group-by", action="append")
    report.add_argument("--require-equal", action="append")
    report.add_argument("--strict", action="store_true")

    compose = sub.add_parser("compose", help="查看兼容组合并生成自定义 YAML")
    compose_sub = compose.add_subparsers(dest="compose_command", required=True)
    compose_list = compose_sub.add_parser("list", help="查看指定 runner 的组合槽位")
    compose_list.add_argument("--runner", choices=runner_names(), default="supervised")
    compose_check = compose_sub.add_parser("check", help="检查自定义配置的组合兼容性")
    compose_check.add_argument("--config", type=Path, required=True)
    compose_check.add_argument("--project-root", type=Path)
    compose_create = compose_sub.add_parser("create", help="从内置 recipe 生成新配置")
    compose_create.add_argument("--base", required=True, help="作为起点的内置 recipe")
    compose_create.add_argument("--output", type=Path, required=True, help="新 YAML 文件路径")
    compose_create.add_argument("--project-root", type=Path)
    compose_create.add_argument("--loss", choices=("ce", "gce", "nce", "mae", "rce", "apl"))
    compose_create.add_argument("--selector", choices=("all", "small_loss"))
    compose_create.add_argument("--keep-rate", type=float)
    compose_create.add_argument(
        "--parameter-update", choices=("standard", "step_milestone", "cdr")
    )
    compose_create.add_argument("--milestones", type=int, nargs="+")
    compose_create.add_argument("--gamma", type=float)
    compose_create.add_argument("--cdr-noise-rate", type=float)
    compose_create.add_argument("--l1-decay", type=float)

    papers = sub.add_parser("papers", help="查看有可运行配置的知名论文")
    paper_sub = papers.add_subparsers(dest="paper_command", required=True)
    paper_list = paper_sub.add_parser("list", help="列出有可运行 config 的论文")
    paper_list.add_argument(
        "--format",
        choices=("human", "tsv"),
        default="human",
        help="输出格式；human 适合阅读，tsv 适合脚本处理（默认: human）",
    )
    show = paper_sub.add_parser("show", help="解释论文在 toolbox 中的详细实现")
    show.add_argument("paper_id")
    config = paper_sub.add_parser("config", help="查看论文 YAML 配置")
    config.add_argument("paper_id")
    config.add_argument("--profile", choices=("smoke", "reproduction"))
    config.add_argument("--variant")
    config.add_argument("--path-only", action="store_true")
    config.add_argument("--resolved", action="store_true")
    config.add_argument("--project-root", type=Path)
    return parser


def _load_source(args: argparse.Namespace) -> tuple[dict[str, Any], Path, RecipeSpec | None, Path]:
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
        config = load_recipe_config(recipe)
    elif config_arg is not None:
        recipe = None
        config_path = config_arg.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"configuration does not exist: {config_path}")
        config = load_yaml(config_path)
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


def _print_plan(config: dict[str, Any], config_path: Path, project: Path) -> None:
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


def _doctor(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve() if args.config else None
    root = find_project_root(config_path, args.project_root)
    failures = 0

    def report(ok: bool, name: str, detail: str) -> None:
        nonlocal failures
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
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
        print(f"    lnl run --recipe {item.id} --dry-run")
    return 0


def _list_components(args: argparse.Namespace) -> int:
    from lnl_toolbox.plugins.builtin import create_builtin_catalog

    catalog = create_builtin_catalog()
    values = catalog.find(kind=args.kind) if args.kind else catalog.find()
    if not values:
        raise ValueError(f"no components found for kind {args.kind!r}")
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
    config, path, _recipe, project = _load_source(args)
    runner = _validate_with_registry(config, check_data=args.check_data)
    print(f"配置有效: {path}")
    print(f"执行器: {runner.name}")
    print(f"项目根目录: {project}")
    return 0


def _run(args: argparse.Namespace) -> int:
    config, path, recipe, project = _load_source(args)
    config = apply_override_assignments(config, args.overrides)
    if args.epochs is not None:
        apply_epoch_override(config, args.epochs)
    if args.no_check_data and not args.dry_run:
        raise ValueError("--no-check-data is only valid together with --dry-run")
    _validate_with_registry(config, check_data=_should_check_data(args))
    if args.dry_run:
        _print_plan(config, path, project)
        return 0
    result = ExperimentService().run(
        config,
        args.output_dir,
        args.resume,
        recipe=recipe.id if recipe is not None else None,
    )
    print(f"运行完成: {result}")
    return 0


def _resume(args: argparse.Namespace) -> int:
    result = ExperimentService().resume(args.run_dir, args.checkpoint)
    print(f"resume complete: {result}")
    return 0


def _sweep(args: argparse.Namespace) -> int:
    if args.source == "status":
        if args.status_path is None:
            raise ValueError("provide a sweep directory after 'lnl sweep status'")
        if args.seeds or args.overrides or args.output_dir or args.dry_run:
            raise ValueError("sweep status cannot be combined with planning options")
        value = sweep_status(args.status_path)
        print("Sweep")
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
    source = args.source
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
            if not isinstance(seeds, list):
                raise ValueError("sweep spec requires a seeds list or CLI --seeds")
            matrix = value.get("matrix", {}) or {}
            if not isinstance(matrix, dict):
                raise ValueError("sweep matrix must be a mapping")
            return config, path, recipe, project, seeds, matrix
    config, path, recipe, project = _load_source(args)
    if args.seeds is None:
        raise ValueError("ordinary experiment sweeps require --seeds")
    return config, path, recipe, project, args.seeds, {}


def _print_sweep_plan(plan) -> None:
    print("Sweep plan")
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
    comparison = compare_runs(
        args.path,
        group_by=_comparison_fields(args.group_by),
        require_equal=_comparison_fields(args.require_equal),
        strict=args.strict,
    )
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
    recipes = {item.id: item for item in discover_recipes(include_conditional=True)}
    papers = load_papers()
    if args.format == "tsv":
        print("id\tacronym\ttitle\tvenue\tyear\tprofiles\tfidelity\trunners\trecommended")
        for paper in papers:
            profiles = ",".join(sorted({item.profile for item in paper.configs}))
            fidelity = ",".join(sorted({item.configuration_fidelity for item in paper.configs}))
            runners = ",".join(sorted({recipes[item.recipe_id].runner for item in paper.configs}))
            recommended = next(
                (item.recipe_id for item in paper.configs if item.profile == "smoke"),
                paper.configs[0].recipe_id if paper.configs else "-",
            )
            print(
                f"{paper.id}\t{paper.acronym}\t{paper.title}\t{paper.venue}\t{paper.year}\t"
                f"{profiles}\t{fidelity}\t{runners}\t{recommended}"
            )
        return 0

    print(f"找到 {len(papers)} 篇具有可运行配置的论文")
    print("建议先查看实现说明，再对推荐的 smoke recipe 执行 dry-run。")
    total = len(papers)
    for index, paper in enumerate(papers, start=1):
        profiles = ",".join(sorted({item.profile for item in paper.configs}))
        fidelity = ",".join(sorted({item.configuration_fidelity for item in paper.configs}))
        runners = ",".join(sorted({recipes[item.recipe_id].runner for item in paper.configs}))
        recommended = next(
            (item.recipe_id for item in paper.configs if item.profile == "smoke"),
            paper.configs[0].recipe_id if paper.configs else "-",
        )
        print()
        print(f"[{index}/{total}] {paper.id} · {paper.acronym}")
        print(f"  标题：{paper.title}")
        print(f"  出处：{paper.venue} {paper.year}")
        print(f"  可用配置：{profiles}")
        print(f"  实现保真度：{fidelity}")
        print(
            f"  实现状态：{paper.implementation_status} | "
            f"复现状态：{paper.reproduction_status} | "
            f"可用性：{paper.availability}"
        )
        print(f"  实际执行器：{runners}")
        print("  建议先看：")
        print(f"    lnl papers show {paper.id}")
        print("  推荐试跑：")
        print(f"    lnl run --recipe {recommended} --dry-run")
    return 0


def _paper_show(paper: PaperSpec) -> int:
    recipes = {item.id: item for item in discover_recipes(include_conditional=True)}
    print(f"{paper.acronym}: {paper.title}")
    print(f"出处: {paper.venue} {paper.year} | {paper.source_url}")
    print(f"问题: {paper.summary}")
    print(f"核心机制: {paper.mechanism}")
    print("\nToolbox 生命周期:")
    for index, stage in enumerate(paper.lifecycle, 1):
        print(f"  {index}. {stage}")
    print("\n论文概念 -> config -> 实现:")
    for item in paper.concept_to_config:
        print(f"  {item['concept']} -> {item['config']} -> {item['implementation']}")
    print("\n可用配置:")
    total = len(paper.configs)
    for index, item in enumerate(paper.configs, start=1):
        recipe = recipes[item.recipe_id]
        config = load_recipe_config(recipe)
        data = config.get("data", {}) or {}
        noise = config.get("noise", {}) or {}
        print()
        print(f"  [{index}/{total}] {item.profile} / {item.variant}")
        print(f"      Recipe：{item.recipe_id}")
        print(f"      实现保真度：{item.configuration_fidelity}")
        print(
            f"      实现状态：{item.implementation_status} | "
            f"复现状态：{item.reproduction_status} | "
            f"可用性：{item.availability}"
        )
        print(f"      执行器：{recipe.runner}")
        print(f"      数据集：{data.get('name', '-')}")
        print(f"      噪声：{noise.get('name', 'clean')}，比例={noise.get('rate', '-')}")
        print(f"      模型：{_paper_config_value(config, 'model')}")
        print(f"      优化器：{_paper_config_value(config, 'optimizer')}")
        print(f"      Scheduler：{_paper_config_value(config, 'scheduler')}")
        print(f"      训练轮数：{_paper_config_value(config, 'epochs')}")
        print(f"      标签来源：{_noise_description(config)}")
        print(f"      模型选择：{_selection_description(config)}")
    print("\nCheckpoint / resume:")
    print("  runner 保存 resolved_config 和阶段状态；支持时用 lnl resume <run_dir> 恢复。")
    print("\n已知差异与限制:")
    for limitation in paper.limitations:
        print(f"  - {limitation}")
    print("\n核心实现路径:")
    for path in paper.implementation_paths:
        print(f"  - {path}")
    print("\n推荐命令:")
    if paper.configs:
        recommended = next(
            (item for item in paper.configs if item.profile == "smoke"),
            paper.configs[0],
        )
        print(f"  lnl validate --recipe {recommended.recipe_id}")
        print(f"  lnl run --recipe {recommended.recipe_id} --dry-run")
        print(f"  lnl run --recipe {recommended.recipe_id}")
    else:
        print("  当前只有组件实现，没有内置可直接运行 recipe。")
    return 0


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


def _print_composition_summary(summary) -> None:
    print(f"  执行器：{summary.runner}")
    print(f"  Loss：{summary.loss}")
    print(f"  Selector：{summary.selector}")
    print(f"  参数更新：{summary.parameter_update}")
    features = "、".join(summary.pipeline_features) or "无"
    print(f"  Pipeline 扩展：{features}")


def _compose_list(args: argparse.Namespace) -> int:
    recipes = tuple(item for item in discover_recipes() if item.runner == args.runner)
    if args.runner != "supervised":
        print(f"{args.runner} 是专用生命周期，不支持自由替换通用组件。")
        print("请从以下完整 recipe 开始，并用 validate / dry-run 检查：")
        if not recipes:
            print("  当前没有内置 recipe。")
        for index, recipe in enumerate(recipes, start=1):
            print(f"  {index}. {recipe.id}")
            print(f"     lnl run --recipe {recipe.id} --dry-run")
        return 0

    from lnl_toolbox.plugins.builtin import create_builtin_catalog

    catalog = create_builtin_catalog()
    names = lambda kind: "、".join(item.name for item in catalog.find(kind=kind))
    print("supervised runner 的可组合结构")
    print("  数据/噪声 → 模型 → Loss → Selector → Pipeline → 参数更新")
    print()
    print("可替换槽位：")
    print(f"  Loss（选择一个）：{names('loss')}")
    print(f"  Selector（选择一个）：{names('batch_selector')}")
    print(f"  参数更新（选择一个）：{names('parameter_update_policy')}")
    print()
    print("Pipeline 规则：")
    print("  1. transition_estimator 必须与 risk_corrector 配对。")
    print("  2. weight_provider 可以与普通 Loss/Selector 共同产生样本贡献。")
    print("  3. objective_consumer 独占训练目标，要求 all selector 和 standard 更新。")
    print("  4. statistic_estimator 必须把产物交给 objective_consumer。")
    print("  5. regularizer 虽已注册，但尚未接入 supervised runner。")
    print()
    print("建议作为 base 的完整模板：")
    preferred = (
        "cifar10-symmetric-ce-smoke",
        "cifar10-foundation-smoke",
        "mentornet-dd-cifar100-symmetric04-smoke",
        "dss-cifar10-symmetric05-smoke",
    )
    known = {item.id for item in recipes}
    for recipe_id in preferred:
        if recipe_id in known:
            print(f"  - {recipe_id}")
    print()
    print("生成示例：")
    print("  lnl compose create --base cifar10-symmetric-ce-smoke `")
    print("    --loss gce --selector small_loss --keep-rate 0.6 `")
    print("    --output configs/experiment/my_gce_small_loss.yaml")
    return 0


def _compose_check(args: argparse.Namespace) -> int:
    root = args.project_root.expanduser().resolve() if args.project_root else None
    path = args.config.expanduser().resolve()
    project = find_project_root(path, root)
    config = resolve_config_paths(load_yaml(path), project)
    summary = validate_composition(config)
    print(f"组合有效：{path}")
    _print_composition_summary(summary)
    print("下一步：")
    print(f"  lnl run --config \"{path}\" --dry-run")
    return 0


def _compose_create(args: argparse.Namespace) -> int:
    root = args.project_root.expanduser().resolve() if args.project_root else None
    recipe = recipe_by_id(args.base, root)
    override_values = (
        args.loss,
        args.selector,
        args.keep_rate,
        args.parameter_update,
        args.milestones,
        args.gamma,
        args.cdr_noise_rate,
        args.l1_decay,
    )
    has_overrides = any(value is not None for value in override_values)
    if recipe.runner != "supervised" and has_overrides:
        raise ValueError(
            f"base recipe uses dedicated runner {recipe.runner!r}; "
            "paper lifecycle recipes can only be copied without component overrides"
        )
    source = load_yaml(recipe.config_path)
    project = find_project_root(recipe.config_path, root)
    if recipe.runner == "supervised":
        composed = apply_overrides(
            source,
            loss=args.loss,
            selector=args.selector,
            keep_rate=args.keep_rate,
            parameter_update=args.parameter_update,
            milestones=args.milestones,
            gamma=args.gamma,
            cdr_noise_rate=args.cdr_noise_rate,
            l1_decay=args.l1_decay,
        )
        summary = validate_composition(resolve_config_paths(composed, project))
    else:
        composed = source
        validate_config(resolve_config_paths(composed, project))
        summary = None
    destination = write_composed_config(composed, args.output)
    print(f"已生成新配置：{destination}")
    print(f"基础 recipe：{recipe.id}")
    if summary is not None:
        _print_composition_summary(summary)
    else:
        print(f"论文生命周期：{recipe.runner}（未修改组件）")
    print("原 recipe 未修改；已有目标文件不会被覆盖。")
    print("下一步：")
    if summary is not None:
        print(f"  lnl compose check --config \"{destination}\"")
    else:
        print(f"  lnl validate --config \"{destination}\"")
    print(f"  lnl run --config \"{destination}\" --dry-run")
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


def _print_dataset_table(reports) -> None:
    print(f"{'DATASET':<24} {'ADAPTER':<30} {'STATUS':<12} LOCATION")
    for report in reports:
        print(
            f"{report.name:<24} {report.adapter:<30} "
            f"{report.status:<12} {report.location or '-'}"
        )


def _data_command(args: argparse.Namespace) -> int:
    service = DEFAULT_DATA_SERVICE
    if args.data_command == "list":
        _print_dataset_table(service.list_datasets())
        return 0
    if args.data_command == "status":
        if args.name is None:
            _print_dataset_table(service.status())
        else:
            _print_dataset_report(service.status(args.name))
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
        report = service.inspect(args.alias)
        _print_dataset_report(report)
        if report.status != "ready":
            return 2
        print("Layout validated; training has not yet been verified.")
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


def _web(args: argparse.Namespace) -> int:
    root = find_project_root(None, args.project_root)
    server = root / "web" / "command_console.py"
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
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
        if args.command == "data":
            return _data_command(args)
        if args.command == "web":
            return _web(args)
        if args.command == "sweep":
            return _sweep(args)
        if args.command == "compare":
            return _compare(args)
        if args.command == "report":
            return _report(args)
        if args.command == "compose":
            if args.compose_command == "list":
                return _compose_list(args)
            if args.compose_command == "check":
                return _compose_check(args)
            return _compose_create(args)
        if args.command == "papers":
            if args.paper_command == "list":
                return _paper_list(args)
            if args.paper_command == "show":
                return _paper_show(paper_by_id(args.paper_id))
            return _paper_config(args)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
