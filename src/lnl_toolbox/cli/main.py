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
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner


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
        "--include-conditional",
        action="store_true",
        help="同时显示需要外部 artifact 的条件可用配置",
    )
    components = list_sub.add_parser("components", help="列出底层可组合组件")
    components.add_argument("--kind")

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

    papers = sub.add_parser("papers", help="查看有可运行配置的知名论文")
    paper_sub = papers.add_subparsers(dest="paper_command", required=True)
    paper_sub.add_parser("list", help="列出有可运行 config 的论文")
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
        return "clean labels"
    if noise.get("manifest"):
        return f"external manifest: {noise['manifest']}"
    name = noise.get("name", "configured")
    rate = noise.get("rate")
    return f"generated {name}" + (f" rate={rate}" if rate is not None else "")


def _selection_description(config: dict[str, Any]) -> str:
    evaluation = config.get("evaluation", {}) or {}
    noise = config.get("noise", {}) or {}
    split = evaluation.get("selection_split", "validation")
    targets = noise.get("validation_targets", "clean") if split == "validation" else "test"
    primary = evaluation.get("primary", "accuracy")
    return f"{split} / targets={targets} / primary={primary}"


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
        values = config.get("dld", {}) or {}
        return str((values.get("diffusion", {}) or {}).get("epochs", "-")) + " (diffusion)"
    if method == "dividemix":
        values = config.get("dividemix", {}) or {}
        warmup = (values.get("warmup", {}) or {}).get("epochs", "-")
        main = (values.get("training", {}) or {}).get("epochs", "-")
        total = warmup + main if isinstance(warmup, int) and isinstance(main, int) else "-"
        return f"{warmup}/{main}/{total} (warmup/main/total)"
    trainer = config.get("trainer", {}) or {}
    return str(trainer.get("epochs", config.get("epochs", "runner default")))


def _print_plan(config: dict[str, Any], config_path: Path, project: Path) -> None:
    runner = resolve_runner(config)
    data = config.get("data", {}) or {}
    trainer = config.get("trainer", {}) or {}
    print("配置预览")
    print(f"  配置文件: {config_path}")
    print(f"  项目根目录: {project}")
    print(f"  执行器: {runner.name}")
    print(f"  数据集: {data.get('name', 'unknown')}")
    print(f"  数据路径: {data.get('root') or data.get('path') or '由数据适配器生成'}")
    print(f"  标签来源: {_noise_description(config)}")
    print(f"  模型: {(config.get('model', {}) or {}).get('name', 'runner default')}")
    print(f"  训练轮数: {_epoch_description(config)}")
    print(f"  设备: {trainer.get('device', 'auto')}")
    print(f"  最佳模型依据: {_selection_description(config)}")
    print(f"  输出根目录: {config.get('output_root', 'artifacts/runs')}")
    method = config.get("method", "")
    if isinstance(method, dict):
        method = method.get("name", "")
    if str(method).strip().lower() == "upm":
        upm = config.get("upm", {}) or {}
        psi = upm.get("psi", {}) or {}
        eta = upm.get("confusing_probability", {}) or {}
        print(f"  UPM psi source: {psi.get('source', '-')}")
        print(f"  UPM eta initial value: {eta.get('initial_value', '-')}")
        print(f"  UPM eta update start epoch: {eta.get('update_start_epoch', '-')}")
        print(f"  UPM eta update interval: {eta.get('update_interval_epochs', '-')}")
    if str(method).strip().lower() == "dld":
        dld = config.get("dld", {}) or {}
        feature = dld.get("feature_extractor", {}) or {}
        pre = dld.get("precorrection", {}) or {}
        diffusion = dld.get("diffusion", {}) or {}
        inference = dld.get("inference", {}) or {}
        fidelity = dld.get("fidelity", {}) or {}
        print(f"  DLD feature extractor: {feature.get('source', '-')}")
        print(
            "  DLD pre-correction: "
            f"K={pre.get('k_neighbors', '-')} / "
            f"metric={fidelity.get('neighbor_metric', '-')} / "
            f"self={fidelity.get('self_neighbor', '-')} / "
            f"divergence={fidelity.get('divergence', '-')} / GMM"
        )
        print(f"  DLD artifact: dld_precorrection.npz")
        print(f"  DLD timesteps: {diffusion.get('timesteps', '-')}")
        print(f"  DLD inference steps: {inference.get('steps', '-')}")
        print(f"  DLD fidelity: {fidelity.get('name', '-')}")
    if str(method).strip().lower() == "dividemix":
        values = config.get("dividemix", {}) or {}
        gmm = values.get("gmm", {}) or {}; history = gmm.get("loss_history", {}) or {}
        mixmatch = values.get("mixmatch", {}) or {}; objective = values.get("objective", {}) or {}; inference = values.get("inference", {}) or {}
        print("  DivideMix models: 2")
        print(f"  DivideMix GMM threshold/history: {gmm.get('threshold', '-')} / {history.get('name', '-')}")
        print(f"  DivideMix M/T/alpha: {mixmatch.get('augmentations', '-')} / {mixmatch.get('temperature', '-')} / {mixmatch.get('mixup_alpha', '-')}")
        print(f"  DivideMix lambda_u/ramp-up: {objective.get('lambda_u', '-')} / {objective.get('rampup_epochs', '-')}")
        print(f"  DivideMix ensemble: {inference.get('ensemble', '-')}")


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
    print(
        "RECIPE | PROFILE | DATASET | NOISE | METHOD | RUNNER | "
        "IMPLEMENTATION | FIDELITY | REPRODUCTION | AVAILABILITY | EPOCHS"
    )
    for item in recipes:
        print(
            f"{item.id} | {item.profile} | {item.dataset} | {item.noise} | "
            f"{item.method} | {item.runner} | {item.implementation_status} | "
            f"{item.configuration_fidelity} | {item.reproduction_status} | "
            f"{item.availability} | {item.epochs if item.epochs is not None else '-'}"
        )
        print(f"  lnl run --recipe {item.id} --dry-run")
    return 0


def _list_components(args: argparse.Namespace) -> int:
    from lnl_toolbox.plugins.builtin import create_builtin_catalog

    catalog = create_builtin_catalog()
    values = catalog.find(kind=args.kind) if args.kind else catalog.find()
    if not values:
        raise ValueError(f"no components found for kind {args.kind!r}")
    print("KIND | NAME | CAPABILITIES | PAPER")
    for item in values:
        paper = item.metadata.get("paper", "-")
        print(f"{item.kind} | {item.name} | {','.join(sorted(item.capabilities)) or '-'} | {paper}")
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


def _paper_list() -> int:
    print(
        "ID | PAPER | VENUE | IMPLEMENTATION | FIDELITY | REPRODUCTION | "
        "AVAILABILITY | PROFILES | RUNNERS | RECOMMENDED"
    )
    recipes = {item.id: item for item in discover_recipes(include_conditional=True)}
    for paper in load_papers():
        profiles = ",".join(sorted({item.profile for item in paper.configs}))
        fidelity = ",".join(sorted({item.configuration_fidelity for item in paper.configs}))
        runners = ",".join(sorted({recipes[item.recipe_id].runner for item in paper.configs}))
        recommended = next(
            (item.recipe_id for item in paper.configs if item.profile == "smoke"),
            paper.configs[0].recipe_id if paper.configs else "-",
        )
        print(
            f"{paper.id} | {paper.acronym} | {paper.venue} {paper.year} | "
            f"{paper.implementation_status} | {fidelity} | "
            f"{paper.reproduction_status} | {paper.availability} | {profiles} | "
            f"{runners} | {recommended}"
        )
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
    for item in paper.configs:
        recipe = recipes[item.recipe_id]
        config = load_recipe_config(recipe)
        data = config.get("data", {}) or {}
        noise = config.get("noise", {}) or {}
        model = config.get("model", {}) or {}
        optimizer = config.get("optimizer", {}) or {}
        trainer = config.get("trainer", {}) or {}
        print(
            f"  {item.profile}/{item.variant} "
            f"[{item.configuration_fidelity}; {item.availability}] {item.recipe_id}"
        )
        print(
            f"    runner={recipe.runner}; data={data.get('name', '-')}; noise={noise.get('name', 'clean')} "
            f"rate={noise.get('rate', '-')}; model={model.get('name', '-')}; optimizer={optimizer.get('name', '-')}; "
            f"epochs={trainer.get('epochs', config.get('epochs', '-'))}"
        )
        print(f"    labels={_noise_description(config)}; selection={_selection_description(config)}")
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
        if args.command == "papers":
            if args.paper_command == "list":
                return _paper_list()
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
