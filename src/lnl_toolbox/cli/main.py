from __future__ import annotations

"""Unified, discoverable command-line interface for toolbox users."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
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
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner, runner_names
from lnl_toolbox.training.reporting import write_run_report, write_toolbox_report


def _source_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--recipe", help="内置实验配置名称")
    group.add_argument("--config", type=Path, help="自定义 YAML 配置")
    parser.add_argument("--project-root", type=Path, help="显式指定项目根目录")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lnl",
        description="LNL Toolbox 统一命令行入口",
        epilog=(
            "推荐顺序: lnl doctor -> lnl list experiments -> "
            "lnl validate --recipe <name> -> lnl run --recipe <name> --dry-run"
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
    run.add_argument("--check-data", action="store_true")

    resume = sub.add_parser("resume", help="从运行目录自动恢复")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--checkpoint", choices=("last", "best"), default="last")

    report = sub.add_parser("report", help="鐢熸垚鍗曟鎴栧叏閮ㄨ繍琛屾姤鍛?")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    report_run = report_sub.add_parser("run", help="鐢熸垚鍗曟 run 鐩綍鎶ュ憡")
    report_run.add_argument("run_dir", type=Path)
    report_toolbox = report_sub.add_parser("toolbox", help="姹囨€昏繍琛屾姤鍛?")
    report_toolbox.add_argument("--runs-root", type=Path, required=True)
    report_toolbox.add_argument("--output", type=Path, required=True)

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
    if args.recipe:
        recipe = recipe_by_id(args.recipe, root)
        config_path = recipe.config_path
        config = load_recipe_config(recipe)
    else:
        recipe = None
        config_path = args.config.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"configuration does not exist: {config_path}")
        config = load_yaml(config_path)
    project = find_project_root(None if recipe is not None else config_path, root)
    config = resolve_config_paths(config, project)
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


def _epoch_description(config: dict[str, Any]) -> str:
    method = config.get("method", "")
    if isinstance(method, dict):
        method = method.get("name", "")
    method = str(method).strip().lower()
    if method == "t_revision":
        values = config.get("t_revision", {}) or {}
        return "/".join(
            str((values.get(stage, {}) or {}).get("epochs", "-"))
            for stage in ("stage1", "classifier_initialization", "revision")
        ) + " (stage1/classifier/revision)"
    if method == "dual_t":
        return "/".join(
            str((config.get(stage, {}) or {}).get("epochs", "-"))
            for stage in ("posterior_stage", "final_stage")
        ) + " (posterior/final)"
    if method == "pcse":
        return "/".join(
            str((config.get(stage, {}) or {}).get("epochs", "-"))
            for stage in ("pretraining_stage", "ensemble_stage")
        ) + " (pretraining/ensemble)"
    if method == "upm":
        values = config.get("upm", {}) or {}
        return "/".join(
            str((values.get(stage, {}) or {}).get("epochs", "-"))
            for stage in ("stage1", "main")
        ) + " (stage1/main)"
    if method == "dld":
        diffusion = (config.get("dld", {}) or {}).get("diffusion", {}) or {}
        return f"pre-correction/{diffusion.get('epochs', '-')} (pre-correction/diffusion)"
    if method == "lend":
        training = (config.get("lend", {}) or {}).get("training", {}) or {}
        return f"{training.get('epochs', '-')} (lend.training)"
    if method == "dividemix":
        values = config.get("dividemix", {}) or {}
        warmup = (values.get("warmup", {}) or {}).get("epochs", "-")
        main = (values.get("training", {}) or {}).get("epochs", "-")
        return f"{warmup}/{main} (warmup/main)"
    if method == "pdl":
        phases = config.get("phases", {}) or {}
        return (
            f"{(config.get('warmup', {}) or {}).get('epochs', '-')}/"
            f"{phases.get('correction_epochs', '-')}/"
            f"{phases.get('revision_epochs', '-')} (warmup/correction/revision)"
        )
    trainer = config.get("trainer", {}) or {}
    return str(trainer.get("epochs", config.get("epochs", "runner default")))


def _print_plan(config: dict[str, Any], config_path: Path, project: Path) -> None:
    runner = resolve_runner(config)
    method = config.get("method", runner.name)
    if isinstance(method, dict):
        method = method.get("name", runner.name)
    data = config.get("data", {}) or {}
    trainer = config.get("trainer", {}) or {}
    print("配置预览")
    print(f"  配置文件: {config_path}")
    print(f"  项目根目录: {project}")
    print(f"  方法: {method}")
    print(f"  执行器: {runner.name}")
    print(f"  数据集: {data.get('name', 'unknown')}")
    print(f"  数据路径: {data.get('root') or data.get('path') or '由数据适配器生成'}")
    print(f"  标签来源: {_noise_description(config)}")
    print(f"  模型: {(config.get('model', {}) or {}).get('name', 'runner default')}")
    print(f"  训练轮数: {_epoch_description(config)}")
    print(f"  设备: {trainer.get('device', 'auto')}")
    print(f"  最佳模型依据: {_selection_description(config)}")
    print(f"  输出根目录: {config.get('output_root', 'artifacts/runs')}")


    normalized_method = str(method).strip().lower()
    if normalized_method == "pdl":
        print("  PDL resume: warmup/correction/revision epoch boundary")
        print("  PDL selection: noisy validation")
    elif normalized_method == "cwd":
        validation_size = int(data.get("validation_size", 0))
        protocol = "independent clean validation" if validation_size > 0 else "fixed budget; test final only"
        print(f"  CWD validation_size: {validation_size}")
        print(f"  CWD evaluation: {protocol}")
    elif normalized_method == "dss":
        objective = ((config.get("pipeline", {}) or {}).get("objective_consumer", {}) or {})
        print(
            "  DSS history-based selection: "
            f"warmup={objective.get('warmup_epochs', '-')}, "
            f"MDA={bool(objective.get('mda', False))}, CCS={bool(objective.get('ccs', False))}"
        )
    elif normalized_method == "fine":
        validation_size = int(data.get("validation_size", 0))
        protocol = "independent clean validation" if validation_size > 0 else "fixed budget; test final only"
        fine = config.get("fine", {}) or {}
        print(
            "  FINE stages: "
            f"warmup={fine.get('warmup_epochs', '-')}, EMA, SCS/SCR"
        )
        print(f"  FINE evaluation: {protocol}")


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
    for package in ("numpy", "yaml", "torch", "torchvision", "randaugment"):
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
            runner = validate_config(config, check_data=args.check_data)
            report(True, "配置", f"{config_path} -> {runner.name}")
        except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
            report(False, "配置", str(exc))
    return 0 if failures == 0 else 1


def _list_experiments(args: argparse.Namespace) -> int:
    recipes = discover_recipes(include_conditional=args.include_conditional)
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
    runner = validate_config(config, check_data=args.check_data)
    print(f"配置有效: {path}")
    print(f"执行器: {runner.name}")
    print(f"项目根目录: {project}")
    return 0


def _run(args: argparse.Namespace) -> int:
    config, path, _recipe, project = _load_source(args)
    if args.epochs is not None:
        apply_epoch_override(config, args.epochs)
    validate_config(config, check_data=args.check_data)
    if args.dry_run:
        _print_plan(config, path, project)
        return 0
    from lnl_toolbox.training.experiment import run_experiment

    result = run_experiment(config, args.output_dir, args.resume)
    print(f"运行完成: {result}")
    return 0


def _resume(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise ValueError(f"run directory is missing resolved_config.yaml: {run_dir}")
    checkpoint = run_dir / f"{args.checkpoint}.pt"
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    config = load_yaml(config_path)
    validate_config(config)
    from lnl_toolbox.training.experiment import run_experiment

    result = run_experiment(config, run_dir, checkpoint)
    print(f"恢复完成: {result}")
    return 0


def _report(args: argparse.Namespace) -> int:
    if args.report_command == "run":
        paths = write_run_report(args.run_dir)
        print("report generated:", paths["markdown"])
        return 0
    paths = write_toolbox_report(args.runs_root, args.output)
    print("toolbox report generated:", paths["markdown"])
    return 0


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
    if recipe.runner != "supervised":
        raise ValueError(
            f"base recipe uses dedicated runner {recipe.runner!r}; "
            "compose create currently supports supervised recipes only"
        )
    source = load_yaml(recipe.config_path)
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
    project = find_project_root(recipe.config_path, root)
    summary = validate_composition(resolve_config_paths(composed, project))
    destination = write_composed_config(composed, args.output)
    print(f"已生成新配置：{destination}")
    print(f"基础 recipe：{recipe.id}")
    _print_composition_summary(summary)
    print("原 recipe 未修改；已有目标文件不会被覆盖。")
    print("下一步：")
    print(f"  lnl compose check --config \"{destination}\"")
    print(f"  lnl run --config \"{destination}\" --dry-run")
    return 0


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
