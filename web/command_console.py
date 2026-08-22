"""Local web command console for the LNL toolbox.

The console intentionally exposes a small, fixed command allowlist. It does
not execute browser-provided shell strings and never starts a shell.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import uuid
import webbrowser
from dataclasses import dataclass, field
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class CommandSpec:
    key: str
    label: str
    description: str
    args: tuple[str, ...]
    output_dir: str | None = None

    @property
    def display_command(self) -> str:
        return "lnl " + " ".join(self.args)


COMMANDS: dict[str, CommandSpec] = {
    "help": CommandSpec("help", "查看帮助", "列出 lnl 的可用命令", ("--help",)),
    "doctor": CommandSpec(
        "doctor", "检查环境", "检查数据和运行环境", ("doctor", "--check-data")
    ),
    "list-smoke": CommandSpec(
        "list-smoke",
        "查看 Smoke 配方",
        "列出可用于短测试的实验配方",
        ("list", "experiments", "--profile", "smoke", "--format", "json"),
    ),
    "validate-clean": CommandSpec(
        "validate-clean",
        "验证 Clean 配置",
        "只验证配置和数据，不训练",
        ("validate", "--recipe", "cifar10-clean-smoke", "--check-data"),
    ),
    "dry-run-clean": CommandSpec(
        "dry-run-clean",
        "预演一次训练",
        "解析配置并检查训练入口，不实际训练",
        (
            "run",
            "--recipe",
            "cifar10-clean-smoke",
            "--dry-run",
            "--check-data",
        ),
    ),
    "train-one": CommandSpec(
        "train-one",
        "训练 1 epoch",
        "执行一个极短的训练示例，输出固定到 artifacts/web-smoke",
        (
            "run",
            "--recipe",
            "cifar10-clean-smoke",
            "--epochs",
            "1",
            "--output-dir",
            "artifacts/web-smoke",
            "--check-data",
        ),
        output_dir="artifacts/web-smoke",
    ),
    "sweep-smoke": CommandSpec(
        "sweep-smoke",
        "多 seed Sweep",
        "按多个随机种子顺序运行同一个 recipe，并支持中断后恢复",
        (
            "sweep",
            "--recipe",
            "cifar10-clean-smoke",
            "--seeds",
            "1",
            "2",
            "3",
            "--output-dir",
            "artifacts/web-sweep",
        ),
        output_dir="artifacts/web-sweep",
    ),
    "resume": CommandSpec(
        "resume",
        "恢复训练",
        "从 web-smoke 的 last checkpoint 恢复",
        ("resume", "artifacts/web-smoke", "--checkpoint", "last"),
        output_dir="artifacts/web-smoke",
    ),
    "report": CommandSpec(
        "report",
        "查看报告",
        "读取 web-smoke 的训练报告",
        ("report", "artifacts/web-smoke"),
        output_dir="artifacts/web-smoke",
    ),
    "compare": CommandSpec(
        "compare",
        "查看曲线",
        "比较 web-smoke 的训练曲线",
        ("compare", "artifacts/web-smoke", "--format", "json"),
        output_dir="artifacts/web-smoke",
    ),
    "compose": CommandSpec(
        "compose",
        "生成自定义 YAML",
        "从已有配方组合一个小损失示例配置",
        (
            "compose",
            "create",
            "--base",
            "cifar10-symmetric-ce-smoke",
            "--output",
            "tmp/web-gce-small-loss.yaml",
            "--loss",
            "gce",
            "--selector",
            "small_loss",
            "--keep-rate",
            "0.8",
        ),
    ),
    "papers": CommandSpec(
        "papers",
        "列出论文方法",
        "列出工具箱内置的论文方法",
        ("papers", "list", "--format", "json"),
    ),
    "paper-config": CommandSpec(
        "paper-config",
        "查看论文配置",
        "查看 Binary Risk 的 reproduction 配置",
        (
            "papers",
            "config",
            "binary-risk",
            "--profile",
            "reproduction",
            "--variant",
            "natarajan-binary",
        ),
    ),
}


TUTORIAL_STEPS: tuple[dict[str, str], ...] = (
    {
        "id": "doctor",
        "label": "检查环境",
        "summary": "确认 Python、PyTorch、CUDA、配置目录和输出目录可用。",
        "why": "先排除环境问题，避免把依赖或 CUDA 错误误认为训练算法错误。",
        "success": "命令退出码为 0，关键项目均显示 OK。",
    },
    {
        "id": "list",
        "label": "选择 Smoke 实验",
        "summary": "浏览公开的短训练 recipe，并选择本教程后续使用的模板。",
        "why": "Smoke 只用于确认训练链路；它不会被误写成论文数值复现。",
        "success": "实验表成功加载，所选 recipe 出现在列表中。",
    },
    {
        "id": "validate",
        "label": "验证配置与数据",
        "summary": "检查 runner、模型、组件、数据路径和条件依赖。",
        "why": "在创建模型和训练产物前，先发现配置或数据问题。",
        "success": "配置验证成功，并显示配置路径和实际 runner。",
    },
    {
        "id": "dry-run",
        "label": "预演训练计划",
        "summary": "查看实际 runner、数据、模型、预算和选模协议，但不训练。",
        "why": "确认即将执行的生命周期与预期一致，同时不生成 checkpoint。",
        "success": "预览显示所选 recipe 的执行计划，输出目录中没有新增训练状态。",
    },
    {
        "id": "run",
        "label": "运行 Smoke",
        "summary": "按所选 recipe 的原始短预算运行，并保存到独立输出目录。",
        "why": "只有实际经过数据、模型、反向传播和 checkpoint，才能证明训练链路可用。",
        "success": "训练退出码为 0，并生成 resolved config、指标和 checkpoint。",
    },
    {
        "id": "resume",
        "label": "检查并恢复",
        "summary": "读取运行阶段、目标轮次、指标和 checkpoint，再判断是否需要恢复。",
        "why": "已达到目标的运行不应盲目重启；未完成的运行才从 last checkpoint 继续。",
        "success": "未完成时恢复成功；已完成时明确显示无需恢复。",
    },
)


def _tutorial_payload() -> dict[str, object]:
    """Return the versioned beginner workflow consumed by the local Web UI."""

    return {
        "version": 1,
        "guide": "docs/LNL-Toolbox-简明操作教程.md",
        "sequence": [step["id"] for step in TUTORIAL_STEPS],
        "steps": [dict(step) for step in TUTORIAL_STEPS],
    }


@dataclass
class Job:
    job_id: str
    key: str
    command: list[str]
    display_command: str
    process: subprocess.Popen[str] | None = None
    lines: list[str] = field(default_factory=list)
    returncode: int | None = None
    error: str | None = None
    structured: object | None = None

    @property
    def done(self) -> bool:
        return self.returncode is not None


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def resolve_lnl_command() -> list[str]:
    """Prefer the installed lnl shortcut, then fall back to the module CLI."""

    executable = shutil.which("lnl")
    if executable:
        return [executable]
    return [sys.executable, "-m", "lnl_toolbox.cli.main"]


def build_command(key: str) -> list[str]:
    """Build one allowlisted argv without shell interpolation."""

    if key not in COMMANDS:
        raise KeyError(f"unknown command key: {key}")
    spec = COMMANDS[key]
    if spec.output_dir:
        (ROOT / spec.output_dir).mkdir(parents=True, exist_ok=True)
    return resolve_lnl_command() + list(spec.args)


def parse_free_command(raw: str) -> list[str]:
    """Parse a beginner-entered lnl command without invoking a shell."""

    text = raw.strip()
    if not text:
        raise ValueError("请输入一条 lnl 指令")
    if any(token in text for token in ("|", "&", ";", "<", ">", chr(96))):
        raise ValueError("为安全起见，不允许管道、重定向或 shell 符号")
    try:
        parts = shlex.split(text, posix=False)
    except ValueError as exc:
        raise ValueError(f"指令引号不完整：{exc}") from exc
    normalized = [
        part[1:-1]
        if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'"
        else part
        for part in parts
    ]
    if not normalized or normalized[0].lower() not in {"lnl", "lnl.exe"}:
        raise ValueError("自由输入必须以 lnl 开头，例如：lnl doctor")
    if len(normalized) == 1:
        raise ValueError("请在 lnl 后面填写子命令，例如：lnl doctor")
    return resolve_lnl_command() + normalized[1:]


def _read_output(job: Job) -> None:
    assert job.process is not None
    stream = job.process.stdout
    if stream is not None:
        for line in stream:
            with JOBS_LOCK:
                job.lines.append(line.rstrip("\r\n"))
    returncode = job.process.wait()
    with JOBS_LOCK:
        job.returncode = returncode
        if returncode == 0 and "--format" in job.command and "json" in job.command:
            try:
                job.structured = json.loads("\n".join(job.lines))
            except json.JSONDecodeError:
                job.structured = None


def _start_process(key: str, command: list[str], display_command: str) -> Job:
    child_env = os.environ.copy()
    # Windows Python processes may otherwise select the GBK console codec
    # when stdout is redirected. The web console always transports UTF-8.
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    job = Job(
        job_id=uuid.uuid4().hex,
        key=key,
        command=command,
        display_command=display_command,
    )
    try:
        job.process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            env=child_env,
        )
    except OSError as exc:
        job.error = str(exc)
        job.returncode = -1
    with JOBS_LOCK:
        JOBS[job.job_id] = job
    if job.process is not None:
        threading.Thread(target=_read_output, args=(job,), daemon=True).start()
    return job


def start_job(key: str) -> Job:
    spec = COMMANDS[key]
    return _start_process(key, build_command(key), spec.display_command)


def start_free_job(raw: str) -> Job:
    command = parse_free_command(raw)
    return _start_process("custom", command, raw.strip())


def _json_response(
    handler: BaseHTTPRequestHandler, payload: object, status: int = 200
) -> None:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _recipe_payload(*, include_all: bool = False) -> list[dict[str, object]]:
    """Return curated templates, or the full catalog for advanced editing."""

    from lnl_toolbox.catalog import discover_recipes

    recipes = discover_recipes(
        ROOT,
        include_conditional=include_all,
        public_only=not include_all,
    )
    return [
        {
            "id": recipe.id,
            "profile": recipe.profile,
            "method": recipe.method,
            "dataset": recipe.dataset,
            "noise": recipe.noise,
            "runner": recipe.runner,
            "epochs": recipe.epochs,
            "availability": recipe.availability,
            "visibility": recipe.visibility,
            "label": recipe.label,
            "description": recipe.description,
        }
        for recipe in recipes
    ]


def _paper_payload() -> list[dict[str, object]]:
    """Return paper metadata and available recipe variants for the UI."""

    from lnl_toolbox.catalog import (
        discover_recipes,
        load_papers,
    )

    recipes = {
        recipe.id: recipe
        for recipe in discover_recipes(ROOT, include_conditional=True)
    }
    payload = []
    for paper in load_papers(ROOT):
        default_config = next(
            config for config in paper.configs if config.profile == "reproduction"
        )
        default_recipe = recipes[default_config.recipe_id]
        payload.append({
            "id": paper.id,
            "acronym": paper.acronym,
            "title": paper.title,
            "venue": paper.venue,
            "year": paper.year,
            "source_url": paper.source_url,
            "summary": paper.summary,
            "mechanism": paper.mechanism,
            "lifecycle": list(paper.lifecycle),
            "limitations": list(paper.limitations),
            "concept_to_config": [dict(item) for item in paper.concept_to_config],
            "implementation_paths": list(paper.implementation_paths),
            "implementation_status": paper.implementation_status,
            "reproduction_status": paper.reproduction_status,
            "default_recipe_id": default_recipe.id,
            "default_variant": default_config.variant,
            "default_fidelity": default_config.configuration_fidelity,
            "configs": [],
        })
        for config in paper.configs:
            recipe = recipes[config.recipe_id]
            profile_label = {
                "reproduction": "正式复现",
                "smoke": "快速检查",
                "experiment": "实验配置",
            }.get(config.profile, config.profile)
            noise_label = recipe.noise
            payload[-1]["configs"].append({
                "recipe_id": config.recipe_id,
                "label": f"{profile_label} · {recipe.dataset} · {noise_label} · {recipe.epochs or '-'} ep",
                "profile": config.profile,
                "variant": config.variant,
                "config_path": recipe.config_path.relative_to(ROOT).as_posix(),
                "runner": recipe.runner,
                "method": recipe.method,
                "data": recipe.dataset,
                "noise": recipe.noise,
                "noise_rate": None,
                "epochs": recipe.epochs,
                "configuration_fidelity": config.configuration_fidelity,
                "implementation_status": config.implementation_status,
                "reproduction_status": config.reproduction_status,
                "availability": config.availability,
            })
    return payload


def _dataset_payload() -> dict[str, object]:
    """Expose adapter-backed readiness through the shared DataService."""

    from lnl_toolbox.training.data_service import DataService

    service = DataService()
    adapters = [
        name for name in service.registry.names() if not name.startswith("synthetic_")
    ]
    return {
        "catalog": str(service.catalog.path),
        "adapters": adapters,
        "datasets": [report.to_dict() for report in service.list_datasets()],
    }


def _dataset_action(payload: object) -> dict[str, object]:
    """Run a non-training data action directly through DataService."""

    if not isinstance(payload, dict):
        raise ValueError("dataset request must be a JSON object")
    action = str(payload.get("action", "")).strip().lower()
    name = str(payload.get("name", "")).strip()
    from lnl_toolbox.training.data_service import DataService

    service = DataService()
    if action == "list":
        return _dataset_payload()
    if action == "status":
        value = service.status(name or None)
        if isinstance(value, tuple):
            return {"datasets": [item.to_dict() for item in value]}
        return {"dataset": value.to_dict()}
    if action == "path":
        if not name:
            raise ValueError("dataset path requires a name")
        value = service.path(name)
        return {"name": name, "path": None if value is None else str(value)}
    if action == "register":
        adapter = str(payload.get("adapter", "")).strip()
        if not name or not adapter:
            raise ValueError("dataset registration requires name and adapter")
        data = {"name": adapter}
        for source, target in (
            ("root", "root"),
            ("path", "path"),
            ("labels", "noise_path"),
            ("noise_variant", "noise_variant"),
        ):
            value = payload.get(source)
            if value not in {None, ""}:
                data[target] = value
        return {
            "dataset": service.register(name, adapter, data).to_dict(),
            "message": "登记已保存。下一步请执行 inspect，实际加载 train/test。",
            "next_action": "inspect",
        }
    if action == "inspect":
        if not name:
            raise ValueError("dataset inspection requires a name")
        report = service.inspect(name)
        if report.status != "ready":
            raise ValueError(report.error or f"dataset is not ready: {name}")
        return {
            "dataset": report.to_dict(),
            "message": "数据检查通过。可继续执行 verify，完成一轮训练验证。",
            "next_action": "verify",
        }
    if action == "remove":
        if not name:
            raise ValueError("dataset removal requires a name")
        service.remove(name)
        return {
            "removed": name,
            "message": f"已删除数据登记：{name}。原始数据文件未被删除。",
            "next_action": "list",
        }
    raise ValueError(f"unsupported dataset action: {action}")


def _dataset_verify_job(payload: object) -> Job:
    """Start the training-backed verify command as a safe background job."""

    if not isinstance(payload, dict):
        raise ValueError("dataset request must be a JSON object")
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError("dataset verification requires a name")
    command = resolve_lnl_command() + ["data", "verify", name]
    recipe = str(payload.get("recipe", "")).strip()
    output_dir = str(payload.get("output_dir", "")).strip()
    if recipe:
        command.extend(("--recipe", recipe))
    if output_dir:
        command.extend(("--output-dir", output_dir))
    display = "lnl " + " ".join(shlex.quote(item) for item in command[len(resolve_lnl_command()):])
    return _start_process("data-verify", command, display)


def _project_path(value: object) -> Path:
    """Resolve a user path while keeping web writes inside this project."""

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("请输入 YAML 文件路径")
    candidate = Path(raw).expanduser()
    destination = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    try:
        destination.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("YAML 路径必须位于项目目录内") from exc
    if destination.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("YAML 文件必须使用 .yaml 或 .yml 扩展名")
    return destination


def _config_payload(
    recipe_id: object = None,
    *,
    path_value: object = None,
) -> dict[str, object]:
    from lnl_toolbox.catalog import load_yaml, recipe_by_id, validate_config

    if str(path_value or "").strip():
        path = _project_path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"YAML 文件不存在：{path.relative_to(ROOT).as_posix()}")
        config = load_yaml(path)
        runner = validate_config(config).name
        method_value = config.get("method", runner)
        method = (
            str(method_value.get("name", runner))
            if isinstance(method_value, dict)
            else str(method_value)
        )
        recipe_name = ""
    else:
        recipe = recipe_by_id(str(recipe_id), ROOT)
        path = recipe.config_path.resolve()
        config = load_yaml(path)
        runner = recipe.runner
        method = recipe.method
        recipe_name = recipe.id
    return {
        "recipe": recipe_name,
        "runner": runner,
        "method": method,
        "path": path.relative_to(ROOT).as_posix(),
        "content": path.read_text(encoding="utf-8"),
        "config": config,
    }


def _builtin_config_paths() -> set[Path]:
    from lnl_toolbox.catalog import discover_recipes

    return {recipe.config_path.resolve() for recipe in discover_recipes(ROOT, include_conditional=True)}


_LEGACY_EDITABLE_SECTIONS = {
    "data",
    "noise",
    "loader",
    "model",
    "loss",
    "selector",
    "parameter_update",
    "optimizer",
    "scheduler",
    "trainer",
    "fine",
    "dividemix",
    "ca2c",
    "l2rw",
    "lend",
    "cnlcu",
    "coteaching",
    "dual_t",
    "dld",
    "pcse",
    "cal",
    "mc_ldce",
    "pdl",
    "upm",
    "volminnet",
    "t_revision",
    "importance_reweighting",
    "meta",
    "evidence",
}
_LEGACY_PROTECTED_LEAF_NAMES = {
    "method",
    "runner",
    "name",
    "type",
    "model",
    "loss",
    "selector",
    "parameter_update",
    "execution",
    "fidelity",
    "ensemble",
    "checkpoint_selection",
    "selection_split",
    "primary",
    "source",
}


def _legacy_editable_config_fields(
    config: object, *, prefix: str = ""
) -> list[dict[str, object]]:
    """Compatibility policy for non-paper recipes without registry metadata."""

    fields: list[dict[str, object]] = []
    if not isinstance(config, dict):
        return fields
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            if prefix or key in _LEGACY_EDITABLE_SECTIONS:
                fields.extend(_legacy_editable_config_fields(value, prefix=path))
            continue
        if not prefix and key not in {"seed", "output_root"}:
            continue
        root = path.split(".", 1)[0]
        if root not in _LEGACY_EDITABLE_SECTIONS and path not in {"seed", "output_root"}:
            continue
        if str(key) in _LEGACY_PROTECTED_LEAF_NAMES and path != "data.name":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            kind = "boolean" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "text"
            fields.append(
                {
                    "path": path,
                    "label": path,
                    "value": value,
                    "kind": kind,
                    "level": "basic",
                    "editable": True,
                    "default_expanded": True,
                    "note": "通用配置参数。",
                }
            )
        elif isinstance(value, list):
            fields.append(
                {
                    "path": path,
                    "label": path,
                    "value": value,
                    "kind": "list",
                    "level": "basic",
                    "editable": True,
                    "default_expanded": True,
                    "note": "通用配置参数。",
                }
            )
    return fields


_PARAMETER_REGISTRY_PATH = WEB_ROOT / "lnl_parameter_metadata_registry_revised.yaml"
_PARAMETER_LEVEL_ORDER = ("basic", "paper", "advanced", "locked")


@lru_cache(maxsize=1)
def _parameter_registry() -> dict[str, Any]:
    """Load and minimally validate the Web parameter authorization registry."""

    try:
        import yaml

        value = yaml.safe_load(_PARAMETER_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"参数元数据 registry 无法加载：{exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("methods"), dict):
        raise ValueError("参数元数据 registry 缺少 methods")
    if not isinstance(value.get("levels"), dict):
        raise ValueError("参数元数据 registry 缺少 levels")
    if not isinstance(value.get("formal_recipe_bindings"), dict):
        raise ValueError("参数元数据 registry 缺少 formal_recipe_bindings")
    return value


def _method_registry_key(method: object) -> str:
    return str(method or "").strip().lower().replace("-", "_")


def _config_method(config: object, fallback: object = "") -> str:
    if not isinstance(config, dict):
        return _method_registry_key(fallback)
    value = config.get("method", fallback)
    if isinstance(value, dict):
        value = value.get("name", fallback)
    method = _method_registry_key(value)
    registry = _parameter_registry()
    if method in registry["methods"]:
        return method
    record = config.get("parameter_record", {})
    formal_recipe = record.get("formal_recipe") if isinstance(record, dict) else ""
    candidate_recipe = str(formal_recipe or fallback or "")
    for candidate_method, recipe_id in registry["formal_recipe_bindings"].items():
        if candidate_recipe == str(recipe_id):
            return str(candidate_method)
    return method


def _config_path_value(config: object, path: str) -> tuple[bool, object]:
    value = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _value_kind(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "text"


@lru_cache(maxsize=1)
def _registry_recipe_paths() -> dict[str, Path]:
    from lnl_toolbox.catalog import discover_recipes

    return {
        recipe.id: recipe.config_path
        for recipe in discover_recipes(ROOT, include_conditional=True)
    }


@lru_cache(maxsize=None)
def _formal_registry_config(method: str) -> tuple[str, dict[str, Any]] | None:
    registry = _parameter_registry()
    recipe_id = registry["formal_recipe_bindings"].get(method)
    if not recipe_id:
        return None
    from lnl_toolbox.catalog import load_yaml

    try:
        recipe_path = _registry_recipe_paths()[str(recipe_id)]
    except KeyError as exc:
        raise ValueError(f"registry formal recipe 不存在：{recipe_id}") from exc
    return str(recipe_id), load_yaml(recipe_path)


def _registry_config_fields(
    config: dict[str, Any], method: str
) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    registry = _parameter_registry()
    method_metadata = registry["methods"].get(method)
    if not isinstance(method_metadata, dict):
        return None
    parameters = method_metadata.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"registry method {method} 缺少 parameters")
    levels = registry["levels"]
    formal = _formal_registry_config(method)
    recipe_id, baseline = formal if formal is not None else ("", config)
    paper_metadata = method_metadata.get("paper", {})
    evidence = paper_metadata.get("evidence", {}) if isinstance(paper_metadata, dict) else {}
    fields: list[dict[str, object]] = []
    for level in _PARAMETER_LEVEL_ORDER:
        level_metadata = levels.get(level, {})
        entries = parameters.get(level, {})
        if not isinstance(entries, dict):
            raise ValueError(f"registry method {method}.{level} 必须是 mapping")
        for path, metadata in entries.items():
            if not isinstance(metadata, dict):
                raise ValueError(f"registry 参数 {method}.{path} 元数据无效")
            found, current = _config_path_value(config, str(path))
            if not found:
                raise ValueError(f"registry 参数路径不存在：{method}.{path}")
            baseline_found, baseline_value = _config_path_value(baseline, str(path))
            evidence_ref = metadata.get("evidence_ref")
            fields.append(
                {
                    "path": str(path),
                    "label": str(path),
                    "value": current,
                    "kind": _value_kind(current),
                    "level": level,
                    "level_label": level_metadata.get("label_zh", level),
                    "editable": bool(level_metadata.get("editable", False)),
                    "default_expanded": bool(
                        level_metadata.get("default_expanded", False)
                    ),
                    "note": str(metadata.get("note", "")),
                    "reproduction_impact": str(
                        metadata.get("reproduction_impact", "")
                    ),
                    "origin": str(metadata.get("origin", "")),
                    "lock_reason": str(metadata.get("lock_reason", "")),
                    "evidence_ref": str(evidence_ref or ""),
                    "evidence": str(evidence.get(evidence_ref, "")),
                    "changed_from": baseline_value if baseline_found else None,
                    "changed_from_paper": bool(
                        level == "paper"
                        and baseline_found
                        and current != baseline_value
                    ),
                }
            )
    paper_changes = [
        field for field in fields if field["level"] == "paper" and field["changed_from_paper"]
    ]
    return fields, {
        "registry_version": str(registry.get("registry_version", "")),
        "formal_recipe": recipe_id,
        "paper_title": str(paper_metadata.get("title", "")),
        "paper_source": str(paper_metadata.get("original_source", "")),
        "modified_from_paper": bool(paper_changes),
        "paper_changes": [field["path"] for field in paper_changes],
        "levels": [
            {
                "id": level,
                "label": str(levels.get(level, {}).get("label_zh", level)),
                "default_expanded": bool(
                    levels.get(level, {}).get("default_expanded", False)
                ),
            }
            for level in _PARAMETER_LEVEL_ORDER
        ],
    }


def _config_schema(
    recipe_id: object = None,
    *,
    path_value: object = None,
) -> dict[str, object]:
    payload = _config_payload(recipe_id, path_value=path_value)
    method = _config_method(
        payload["config"], payload["recipe"] or payload["method"]
    )
    formal_recipe = _parameter_registry()["formal_recipe_bindings"].get(method, "")
    use_registry = not payload["recipe"] or payload["recipe"] == formal_recipe
    registry_fields = (
        _registry_config_fields(payload["config"], method) if use_registry else None
    )
    fields, metadata = (
        registry_fields
        if registry_fields is not None
        else (
            _legacy_editable_config_fields(payload["config"]),
            {
                "registry_version": "",
                "formal_recipe": "",
                "paper_title": "",
                "paper_source": "",
                "modified_from_paper": False,
                "paper_changes": [],
                "levels": [
                    {"id": "basic", "label": "基础参数", "default_expanded": True}
                ],
            },
        )
    )
    return {
        "recipe": payload["recipe"],
        "source_path": payload["path"],
        "runner": payload["runner"],
        "method": method,
        "fields": fields,
        **metadata,
    }


def _field_map(config: dict[str, Any], method: str) -> dict[str, dict[str, object]]:
    registry_fields = _registry_config_fields(config, method)
    fields = (
        registry_fields[0]
        if registry_fields is not None
        else _legacy_editable_config_fields(config)
    )
    return {str(field["path"]): field for field in fields}


def _set_config_path(config: dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    current: dict[str, object] = config
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"不允许新增或重组配置字段：{path}")
        current = child
    if parts[-1] not in current:
        raise ValueError(f"配置字段不存在：{path}")
    current[parts[-1]] = value


def _coerce_patch_value(field: dict[str, object], value: object) -> object:
    kind = field["kind"]
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"参数 {field['path']} 必须是布尔值")
        return value
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"参数 {field['path']} 必须是数字")
        return value
    if kind == "list":
        if not isinstance(value, list):
            raise ValueError(f"参数 {field['path']} 必须是列表")
        return value
    if not isinstance(value, str):
        raise ValueError(f"参数 {field['path']} 必须是文本")
    return value


def _locked_registry_paths(method: str) -> tuple[str, ...]:
    method_metadata = _parameter_registry()["methods"].get(method, {})
    parameters = method_metadata.get("parameters", {}) if isinstance(method_metadata, dict) else {}
    locked = parameters.get("locked", {}) if isinstance(parameters, dict) else {}
    return tuple(str(path) for path in locked) if isinstance(locked, dict) else ()


def _assert_locked_parameters_unchanged(
    before: dict[str, Any], after: dict[str, Any], method: str
) -> None:
    for path in _locked_registry_paths(method):
        before_found, before_value = _config_path_value(before, path)
        after_found, after_value = _config_path_value(after, path)
        if before_found != after_found or before_value != after_value:
            raise ValueError(f"锁定参数不能通过普通 WebUI 修改：{path}")


def _paper_parameter_changes(
    config: dict[str, Any], method: str
) -> tuple[str, list[dict[str, object]]]:
    formal = _formal_registry_config(method)
    if formal is None:
        return "", []
    recipe_id, baseline = formal
    method_metadata = _parameter_registry()["methods"].get(method, {})
    parameters = method_metadata.get("parameters", {}) if isinstance(method_metadata, dict) else {}
    paper = parameters.get("paper", {}) if isinstance(parameters, dict) else {}
    changes: list[dict[str, object]] = []
    if isinstance(paper, dict):
        for path in paper:
            baseline_found, baseline_value = _config_path_value(baseline, str(path))
            current_found, current_value = _config_path_value(config, str(path))
            if not baseline_found or not current_found:
                raise ValueError(f"论文参数路径不存在：{method}.{path}")
            if current_value != baseline_value:
                changes.append(
                    {
                        "path": str(path),
                        "changed_from": baseline_value,
                        "value": current_value,
                    }
                )
    return recipe_id, changes


def _record_paper_parameter_status(
    config: dict[str, Any], method: str, *, acknowledged: bool
) -> None:
    recipe_id, changes = _paper_parameter_changes(config, method)
    if not recipe_id:
        return
    existing = config.get("parameter_record", {})
    record = dict(existing) if isinstance(existing, dict) else {}
    previously_acknowledged = bool(record.get("paper_change_acknowledged", False))
    if changes and not (acknowledged or previously_acknowledged):
        paths = ", ".join(str(change["path"]) for change in changes)
        raise ValueError(f"修改论文参数需要确认复现影响：{paths}")
    record.update(
        {
            "parameter_metadata_registry_version": str(
                _parameter_registry().get("registry_version", "")
            ),
            "formal_recipe": recipe_id,
            "modified_from_paper": bool(changes),
            "paper_change_acknowledged": bool(changes),
            "paper_parameter_changes": changes,
            "effective_reproduction_status": (
                "modified_from_paper" if changes else "formal_recipe"
            ),
        }
    )
    config["parameter_record"] = record


def _save_config(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("请求内容必须是 JSON 对象")
    destination = _project_path(payload.get("path"))
    recipe_id = payload.get("recipe")
    source_path = payload.get("source_path")
    content = payload.get("content")
    patches = payload.get("patches")
    from lnl_toolbox.catalog import load_yaml, recipe_by_id, validate_config

    source_config: dict[str, Any] | None = None
    source_method = ""
    if isinstance(recipe_id, str) and recipe_id.strip():
        source_recipe = recipe_by_id(recipe_id, ROOT)
        source_config = load_yaml(source_recipe.config_path)
        source_method = _config_method(source_config, recipe_id or source_recipe.method)
    elif str(source_path or "").strip():
        source = _project_path(source_path)
        if not source.is_file():
            raise FileNotFoundError("来源 YAML 不存在")
        source_config = load_yaml(source)
        source_method = _config_method(source_config)

    if content is not None:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("YAML 内容不能为空")
        try:
            import yaml

            parsed = yaml.safe_load(content)
        except Exception as exc:
            raise ValueError(f"YAML 解析失败：{exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("YAML 顶层必须是 mapping")
    else:
        if source_config is None:
            raise ValueError("保存 YAML 必须指定来源 recipe 或项目 YAML")
        parsed = load_yaml(recipe_by_id(recipe_id, ROOT).config_path) if (
            isinstance(recipe_id, str) and recipe_id.strip()
        ) else load_yaml(_project_path(source_path))
        if not isinstance(patches, list):
            raise ValueError("只能通过参数菜单提交 YAML 修改")
        method = source_method or _config_method(parsed, recipe_id)
        fields = _field_map(parsed, method)
        for patch in patches:
            if not isinstance(patch, dict) or not isinstance(patch.get("path"), str):
                raise ValueError("YAML 参数修改格式错误")
            path = patch["path"]
            field = fields.get(path)
            if field is None:
                raise ValueError(f"不允许修改配置结构或组件：{path}")
            if not field.get("editable", False):
                raise ValueError(f"锁定参数不能通过普通 WebUI 修改：{path}")
            _set_config_path(parsed, path, _coerce_patch_value(field, patch.get("value")))
    method = source_method or _config_method(source_config or parsed, _config_method(parsed))
    if source_config is None:
        formal = _formal_registry_config(method)
        source_config = formal[1] if formal is not None else parsed
    _assert_locked_parameters_unchanged(source_config, parsed, method)
    _record_paper_parameter_status(
        parsed,
        method,
        acknowledged=bool(payload.get("acknowledge_paper_impact", False)),
    )
    validate_config(parsed)
    try:
        import yaml

        content = yaml.safe_dump(parsed, sort_keys=False, allow_unicode=True)
    except Exception as exc:
        raise ValueError(f"YAML 生成失败：{exc}") from exc
    overwrite = bool(payload.get("overwrite", False))
    if destination.resolve() in _builtin_config_paths():
        raise ValueError("不能直接覆盖内置 recipe，请另存为新的 YAML")
    if destination.exists() and not overwrite:
        raise FileExistsError("目标文件已存在；如需修改请使用覆盖保存")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return {"path": destination.relative_to(ROOT).as_posix(), "bytes": destination.stat().st_size, "content": content}


def _job_payload(job: Job) -> dict[str, object]:
    with JOBS_LOCK:
        return {
            "id": job.job_id,
            "key": job.key,
            "command": job.display_command,
            "lines": list(job.lines),
            "returncode": job.returncode,
            "error": job.error,
            "running": not job.done,
            "structured": job.structured,
        }


def _picker_payload(payload: object) -> dict[str, object]:
    """Open a Windows-native path dialog without executing a shell."""

    if os.name != "nt":
        raise OSError("Windows native path selection is only available on Windows")
    if not isinstance(payload, dict):
        raise TypeError("picker request must be an object")
    mode = str(payload.get("mode", "folder"))
    if mode not in {"folder", "open_file", "save_file"}:
        raise ValueError(f"unknown picker mode: {mode}")
    kind = str(payload.get("kind", "all"))
    filters = {
        "yaml": "YAML files (*.yaml;*.yml)|*.yaml;*.yml|All files (*.*)|*.*",
        "checkpoint": "PyTorch checkpoints (*.pt;*.pth)|*.pt;*.pth|All files (*.*)|*.*",
        "labels": "Label files (*.json;*.txt;*.csv)|*.json;*.txt;*.csv|All files (*.*)|*.*",
        "all": "All files (*.*)|*.*",
    }
    if kind not in filters:
        raise ValueError(f"unknown picker file kind: {kind}")
    initial = str(payload.get("initial", "")).strip()
    initial_path = Path(initial).expanduser() if initial else ROOT
    if not initial or not initial_path.is_absolute() or not initial_path.exists():
        initial_path = ROOT
    initial = str(initial_path.resolve())
    environment = os.environ.copy()
    environment["LNL_PICKER_INITIAL"] = initial
    environment["LNL_PICKER_FILTER"] = filters[kind]
    environment["LNL_PICKER_MODE"] = mode
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$mode=$env:LNL_PICKER_MODE; $initial=$env:LNL_PICKER_INITIAL; "
        "if($mode -eq 'folder'){ $d=New-Object System.Windows.Forms.FolderBrowserDialog; "
        "if($initial -and (Test-Path -LiteralPath $initial)){ $d.SelectedPath=$initial }; "
        "if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){[Console]::Write($d.SelectedPath)} } "
        "else { if($mode -eq 'save_file'){ $d=New-Object System.Windows.Forms.SaveFileDialog } "
        "else { $d=New-Object System.Windows.Forms.OpenFileDialog }; "
        "$d.Filter=$env:LNL_PICKER_FILTER; "
        "if($initial){ if(Test-Path -LiteralPath $initial -PathType Container){$d.InitialDirectory=$initial} "
        "elseif(Test-Path -LiteralPath $initial){$d.InitialDirectory=(Split-Path -Parent $initial);$d.FileName=(Split-Path -Leaf $initial)} }; "
        "if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){[Console]::Write($d.FileName)} }"
    )
    executable = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    result = subprocess.run(
        [str(executable), "-NoProfile", "-STA", "-Command", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise OSError(result.stderr.strip() or "Windows path dialog failed")
    selected = result.stdout.strip()
    return {"cancelled": not bool(selected), "path": selected or None}


def _sweep_plan_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("sweep plan request must be an object")
    from lnl_toolbox.catalog import load_recipe_config, recipe_by_id, resolve_config_paths
    from lnl_toolbox.training.service import ExperimentService
    from lnl_toolbox.training.sweep import plan_sweep, resolve_planned_config

    recipe = recipe_by_id(str(payload.get("recipe", "")), ROOT)
    config = resolve_config_paths(load_recipe_config(recipe), ROOT)
    matrix = payload.get("matrix", {}) or {}
    if not isinstance(matrix, dict):
        raise TypeError("sweep matrix must be an object")
    raw_seeds = payload.get("seeds")
    seeds = [int(config.get("seed", 1))] if raw_seeds in (None, []) else raw_seeds
    if not isinstance(seeds, list):
        raise TypeError("sweep seeds must be a list")
    plan = plan_sweep(
        config,
        seeds,
        matrix=matrix,
        output_dir=payload.get("output_dir") or None,
        recipe=recipe.id,
    )
    service = ExperimentService()
    for run in plan.runs:
        service.preflight(resolve_planned_config(config, run), check_data=False)
    return {
        "root": str(plan.root),
        "seeds": list(plan.seeds),
        "matrix": {path: list(values) for path, values in plan.matrix},
        "total": len(plan.runs),
        "runs": [
            {"run_id": run.run_id, "seed": run.seed, "overrides": run.override_mapping}
            for run in plan.runs
        ],
    }


def _results_payload(path: str) -> dict[str, object]:
    from lnl_toolbox.training.results import discover_run_results

    return {"root": str(Path(path).expanduser().resolve()), "runs": discover_run_results(path)}


def _resume_payload(path: str, checkpoint: str) -> dict[str, object]:
    from lnl_toolbox.training.results import inspect_resume_run

    return inspect_resume_run(path, checkpoint)


class ConsoleHandler(BaseHTTPRequestHandler):
    server_version = "LNLWebConsole/1.0"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/recipe", "/recipe/"}:
            self._serve_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if path == "/api/commands":
            _json_response(
                self,
                [
                    {
                        "key": spec.key,
                        "label": spec.label,
                        "description": spec.description,
                        "command": spec.display_command,
                    }
                    for spec in COMMANDS.values()
                ],
            )
            return
        if path == "/api/tutorial":
            _json_response(self, _tutorial_payload())
            return
        if path == "/api/recipes":
            try:
                query = parse_qs(urlparse(self.path).query)
                include_all = query.get("all", [""])[0].lower() in {
                    "1", "true", "yes"
                }
                _json_response(self, _recipe_payload(include_all=include_all))
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, 500)
            return
        if path == "/api/papers":
            try:
                _json_response(self, _paper_payload())
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, 500)
            return
        if path == "/api/datasets":
            try:
                _json_response(self, _dataset_payload())
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, 500)
            return
        if path == "/api/configs":
            try:
                query = parse_qs(urlparse(self.path).query)
                recipe_id = query.get("recipe", [""])[0]
                path_value = query.get("path", [""])[0]
                _json_response(
                    self,
                    _config_payload(recipe_id, path_value=path_value),
                )
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, 400)
            return
        if path == "/api/config-schema":
            try:
                query = parse_qs(urlparse(self.path).query)
                recipe_id = query.get("recipe", [""])[0]
                path_value = query.get("path", [""])[0]
                _json_response(
                    self,
                    _config_schema(recipe_id, path_value=path_value),
                )
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, 400)
            return
        if path == "/api/results":
            try:
                query = parse_qs(urlparse(self.path).query)
                selected = query.get("path", [""])[0].strip()
                if not selected:
                    raise ValueError("provide a run directory")
                _json_response(self, _results_payload(selected))
            except (OSError, TypeError, ValueError) as exc:
                _json_response(self, {"error": str(exc)}, 400)
            return
        if path == "/api/resume-inspect":
            try:
                query = parse_qs(urlparse(self.path).query)
                selected = query.get("path", [""])[0].strip()
                checkpoint = query.get("checkpoint", ["last"])[0].strip()
                if not selected:
                    raise ValueError("provide a run directory")
                _json_response(self, _resume_payload(selected, checkpoint))
            except (OSError, TypeError, ValueError) as exc:
                _json_response(self, {"error": str(exc)}, 400)
            return
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                _json_response(self, {"error": "job not found"}, 404)
            else:
                _json_response(self, _job_payload(job))
            return
        _json_response(self, {"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path in {"/api/picker", "/api/sweep/plan"}:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if path == "/api/picker":
                    if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                        raise PermissionError("native path selection is limited to localhost")
                    value = _picker_payload(payload)
                else:
                    value = _sweep_plan_payload(payload)
                _json_response(self, value)
            except (
                json.JSONDecodeError,
                OSError,
                PermissionError,
                TypeError,
                ValueError,
            ) as exc:
                _json_response(self, {"error": str(exc)}, 400)
            return
        if path == "/api/datasets":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if isinstance(payload, dict) and payload.get("action") == "verify":
                    _json_response(self, _job_payload(_dataset_verify_job(payload)), 202)
                else:
                    _json_response(self, _dataset_action(payload))
            except (
                KeyError,
                TypeError,
                ValueError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                _json_response(self, {"error": str(exc)}, 400)
            except Exception as exc:  # Keep the Web client responsive on backend faults.
                _json_response(self, {"error": f"数据操作失败：{exc}"}, 500)
            return
        if path == "/api/configs":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                _json_response(self, _save_config(payload), 201)
            except (TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                _json_response(self, {"error": str(exc)}, 400)
            return
        if path != "/api/run":
            _json_response(self, {"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if "command" in payload:
                raw_command = payload["command"]
                if not isinstance(raw_command, str):
                    raise ValueError("command must be a string")
                job = start_free_job(raw_command)
            else:
                key = payload["key"]
                if not isinstance(key, str):
                    raise ValueError("key must be a string")
                job = start_job(key)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            _json_response(self, {"error": str(exc)}, 400)
            return
        _json_response(self, _job_payload(job), 202)

    def _serve_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            _json_response(self, {"error": "web asset missing"}, 500)
            return
        if content_type.startswith("text/html"):
            body = b"\xef\xbb\xbf" + body
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    open_browser: bool = False,
) -> None:
    server = ThreadingHTTPServer((host, port), ConsoleHandler)
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{browser_host}:{port}/"
    print(f"LNL web: {url}", flush=True)
    if open_browser:
        threading.Timer(0.25, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LNL local web command console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args(argv)
    serve(args.host, args.port, open_browser=args.open_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
