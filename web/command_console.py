"""Local web command console for the LNL toolbox.

The console intentionally exposes a small, fixed command allowlist. It does
not execute browser-provided shell strings and never starts a shell.
"""

from __future__ import annotations

import argparse
import errno
import ipaddress
import json
import math
import os
import socket
import shlex
import subprocess
import sys
import threading
import tempfile
import uuid
import webbrowser
from dataclasses import dataclass, field
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
SCRATCH_ROOT = SRC_ROOT / "lnl_toolbox" / "scratch"
SCRATCH_WEB_ROOT = SCRATCH_ROOT / "web"
SCRATCH_RECIPE_ROOT = SCRATCH_ROOT / "recipes"
_SCRATCH_FORMULA_RELOAD_LOCK = threading.RLock()
_DATASET_RECIPE_COMPATIBILITY_LOCK = threading.RLock()
_DATASET_RECIPE_COMPATIBILITY_CACHE: dict[tuple[str, float | None], dict[str, object]] = {}
STATIC_ASSETS = {
    "/assets/quick_start.js": (WEB_ROOT / "assets" / "quick_start.js", "application/javascript; charset=utf-8"),
    "/assets/quick_start.css": (WEB_ROOT / "assets" / "quick_start.css", "text/css; charset=utf-8"),
    "/assets/run_output.js": (WEB_ROOT / "assets" / "run_output.js", "application/javascript; charset=utf-8"),
}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lnl_toolbox.quickstart.templates import SPLIT_COUNT_PATHS, reference_train_size


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
        "多 seed 参数组合实验",
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
    cancel_requested: bool = False
    training_context: object | None = None
    output_complete: bool = False

    @property
    def done(self) -> bool:
        return self.returncode is not None


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
REVIEW_FILES: dict[str, tempfile.TemporaryDirectory] = {}


def _cleanup_review_file(path: str) -> None:
    with JOBS_LOCK:
        directory = REVIEW_FILES.pop(path, None)
    if directory is not None:
        directory.cleanup()


def resolve_lnl_command() -> list[str]:
    """Run the CLI with the interpreter serving the current Web process."""

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
    if "--review" in job.command and "--dry-run" in job.command and "--config" in job.command:
        _cleanup_review_file(job.command[job.command.index("--config") + 1])
    with JOBS_LOCK:
        job.returncode = returncode
        job.output_complete = True
        if returncode == 0 and "--format" in job.command and "json" in job.command:
            try:
                job.structured = json.loads("\n".join(job.lines))
            except json.JSONDecodeError:
                job.structured = None


def _reconcile_job_process(job: Job) -> None:
    """Publish a child exit before the stdout reader has drained its pipe."""

    process = job.process
    if process is None or job.returncode is not None:
        return
    returncode = process.poll()
    if returncode is None:
        return
    with JOBS_LOCK:
        if job.returncode is None:
            job.returncode = returncode


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
    from web.training_status import infer_training_context

    context = infer_training_context(command, ROOT)
    if (
        context is not None
        and context.command_kind == "run"
        and context.run_dir is None
        and "--dry-run" not in command
    ):
        # A recipe's output_root can produce a runner-defined subdirectory.
        # Give Web-owned formal runs a unique explicit location instead, so
        # polling never guesses from the newest artifact directory.
        web_output = ROOT / "artifacts" / "web-runs" / job.job_id
        command = [*command, "--output-dir", str(web_output.relative_to(ROOT))]
        job.command = command
        job.display_command = f"{display_command} --output-dir {web_output.relative_to(ROOT)}"
        context = infer_training_context(command, ROOT)
    job.training_context = context
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
        job.output_complete = True
        if "--review" in command and "--dry-run" in command and "--config" in command:
            _cleanup_review_file(command[command.index("--config") + 1])
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


def cancel_job(job_id: str) -> Job:
    """Request termination of one WebUI-owned child process."""

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise KeyError("job not found")
        if job.done:
            return job
        process = job.process
        job.cancel_requested = True
    if process is not None and process.poll() is None:
        # ``lnl`` may be a Windows console wrapper which starts the actual
        # Python worker as a child.  Terminating only the wrapper leaks that
        # worker into the next batch case, so cancel the complete process tree.
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.terminate()
    return job


def _json_response(
    handler: BaseHTTPRequestHandler, payload: object, status: int = 200
) -> None:
    def json_safe(value: object) -> object:
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    body = json.dumps(json_safe(payload), ensure_ascii=True, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _scratch_recipe_path(name: str) -> Path:
    """Resolve a Scratch recipe below the repository's Scratch recipe root."""

    from lnl_toolbox.scratch import recipe_workspace_root

    requested = Path(name)
    # Paper templates are packaged read-only assets.  Do not make opening one
    # depend on the optional user recipe directory being writable (which can
    # fail with PermissionError on locked-down Windows profiles).
    try:
        user_root = recipe_workspace_root()
    except OSError:
        user_root = None
    candidates = ([
        (user_root, user_root / requested),
    ] if user_root is not None else []) + [
        (SCRATCH_RECIPE_ROOT, SCRATCH_RECIPE_ROOT / requested),
        (SCRATCH_RECIPE_ROOT, SCRATCH_RECIPE_ROOT / "examples" / requested.name),
        (SCRATCH_RECIPE_ROOT, SCRATCH_RECIPE_ROOT / "papers" / requested.name),
    ]
    for root, candidate in candidates:
        resolved = candidate.resolve()
        root = root.resolve()
        if root not in resolved.parents:
            continue
        if candidate.suffix in {".yaml", ".yml"} and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Scratch recipe not found: {name}")


def _scratch_error(exc: Exception) -> dict[str, object]:
    payload: dict[str, object] = {"ok": False, "error": str(exc)}
    try:
        from lnl_toolbox.scratch.web.server import _job_error_code

        payload["code"] = _job_error_code(str(exc))
    except Exception:
        pass
    path = getattr(exc, "path", None)
    block_id = getattr(exc, "block_id", None)
    if path is not None:
        payload["path"] = list(path)
    if block_id is not None:
        payload["block_id"] = block_id
    return payload


def _scratch_recipe_api_payload() -> list[str]:
    from lnl_toolbox.scratch import recipe_workspace_root

    try:
        root = recipe_workspace_root()
    except OSError:
        return []
    return sorted(str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*.y*ml"))


def _scratch_template_payload() -> list[dict[str, str]]:
    formula_ready = {"gce", "coteaching"}
    display_names = {"gce": "GCE", "coteaching": "Co-teaching"}
    paper_root = SCRATCH_RECIPE_ROOT / "papers"
    if not paper_root.exists():
        return []
    return [
        {
            "id": path.stem,
            "name": display_names.get(path.stem, path.stem.replace("_", " ").title()),
            "path": f"papers/{path.name}",
            "status": "formula-ready" if path.stem in formula_ready else "template-ready",
        }
        for path in sorted(paper_root.glob("*.y*ml"))
    ]


def _scratch_examples_payload() -> list[dict[str, str]]:
    return _scratch_template_payload()


def _scratch_formula_payload(spec: object) -> dict[str, object]:
    from lnl_toolbox.scratch.formula.runtime import formula_hash

    payload = spec.to_dict()
    payload.update({"formula_hash": formula_hash(spec), "block_id": "formula/" + spec.id})
    return payload


def _scratch_reload_formulas() -> None:
    from lnl_toolbox.scratch.formula.registry import reload_formulas

    # The Scratch editor requests blocks and formulas in parallel.  Both
    # endpoints refresh the process-wide registry; serialize the unregister /
    # register cycle so a user formula cannot be registered twice by racing
    # requests.
    with _SCRATCH_FORMULA_RELOAD_LOCK:
        reload_formulas()


def _scratch_request_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    length = int(handler.headers.get("Content-Length", "0"))
    payload = json.loads(handler.rfile.read(length) or b"{}")
    if not isinstance(payload, dict):
        raise TypeError("Scratch request body must be an object")
    return payload


def _scratch_recipe_from_body(payload: dict[str, object]) -> dict[str, object]:
    from lnl_toolbox.scratch import validate_recipe

    recipe = payload.get("recipe", payload)
    if not isinstance(recipe, dict):
        raise TypeError("Scratch recipe must be an object")
    return validate_recipe(recipe)


def _scratch_dataset_preflight(recipe: dict[str, object], runtime_limits: object = None) -> dict[str, object]:
    """Apply the same filesystem gate used by the standalone Scratch server."""

    from lnl_toolbox.scratch.web.server import _dataset_preflight

    return _dataset_preflight(recipe, runtime_limits=runtime_limits)


def _scratch_run(recipe: dict[str, object]) -> dict[str, object]:
    from lnl_toolbox.scratch import execute_recipe, resolve_recipe, save_recipe
    from lnl_toolbox.scratch.formula.registry import collect_formula_provenance, merge_formula_provenance

    output = ROOT / "artifacts" / "scratch" / Path(str(recipe["name"])).name
    output.mkdir(parents=True, exist_ok=True)
    save_recipe(recipe, output / "recipe.yaml")
    save_recipe(resolve_recipe(recipe), output / "resolved_recipe.yaml")
    static_provenance = collect_formula_provenance(recipe)
    context = execute_recipe(recipe, {"artifact_dir": str(output)})
    provenance = merge_formula_provenance(static_provenance, context.get("_formula_provenance", []))
    (output / "formula_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = context.get("metrics", [])
    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metrics),
        encoding="utf-8",
    )
    (output / "stdout.log").write_text(
        json.dumps({"name": recipe["name"], "metrics": metrics}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"ok": True, "metrics": metrics, "artifact_dir": str(output), "formula_provenance": provenance}


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


_WEB_RECIPE_CATALOG: dict[str, object] | None = None


def _web_recipe_catalog() -> dict[str, object]:
    """Load the recipe manifest once per console process.

    The compatibility page used to call ``recipe_by_id`` once per paper.  That
    helper performs a fresh full ``discover_recipes`` scan, so one page load
    caused every recipe YAML to be parsed dozens of times.  The Web console is
    a long-lived process; a process-local catalog is the correct cache scope.
    """

    global _WEB_RECIPE_CATALOG
    if _WEB_RECIPE_CATALOG is not None:
        return _WEB_RECIPE_CATALOG
    from lnl_toolbox.catalog import discover_recipes

    discovered = {
        recipe.id: recipe
        for recipe in discover_recipes(ROOT, include_conditional=True)
    }
    # Do not poison the process cache when a unit test (or a partially
    # installed package) supplies a stub YAML loader and discovery yields no
    # runnable recipes. A real checkout has substantially more than this.
    if len(discovered) >= 20:
        _WEB_RECIPE_CATALOG = discovered
    return discovered


@lru_cache(maxsize=1)
def _web_paper_specs() -> tuple[object, ...]:
    from lnl_toolbox.catalog import load_papers

    return load_papers(ROOT)


@lru_cache(maxsize=None)
def _web_recipe_config(recipe_id: str) -> dict[str, Any]:
    """Load one recipe YAML through the cached manifest.

    The fallback keeps isolated unit tests that provide a synthetic
    ``recipe_by_id`` response working without reintroducing the production
    full-catalog scan.
    """

    from copy import deepcopy
    from lnl_toolbox.catalog import load_yaml, recipe_by_id

    recipe = _web_recipe_catalog().get(str(recipe_id))
    if recipe is None:
        recipe = recipe_by_id(str(recipe_id), ROOT)
    return deepcopy(load_yaml(recipe.config_path))


def _external_resource_payload(recipe_id: str, config_path: str = "", dataset: str = "", paper_id: str = "", noise: str = "") -> dict[str, object]:
    from web.external_resources import resources
    from lnl_toolbox.catalog import load_yaml

    config = _web_recipe_config(recipe_id)
    if dataset and not config_path:
        from web.quick_start_api import SERVICE, _selection
        option = next((item for item in SERVICE.method_options(dataset, _selection({"noise": json.loads(noise)}))
                       if item.paper_id == paper_id), None)
        if option is None or option.candidate_config is None:
            raise ValueError("当前论文的运行配置尚未生成")
        config = option.candidate_config
    if config_path:
        source = Path(config_path)
        if not source.is_absolute():
            source = ROOT / source
        source = source.resolve()
        if not source.is_relative_to(ROOT.resolve()) or source.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("运行配置必须位于项目目录中")
        config = load_yaml(source)
    items = resources(config, ROOT)
    # Only fixed server-selected targets can be written by the downloader.
    for item in items:
        if item["id"] != "dld-resnet34" and not Path(item["path"]).is_relative_to(ROOT.resolve()):
            item["downloadable"] = False
            item["note"] += " 此路径位于项目外，请自行放置文件。"
    return {"resources": items}


def _paper_payload() -> list[dict[str, object]]:
    """Return paper metadata and available recipe variants for the UI."""

    from lnl_toolbox.catalog import (
        load_recipe_config,
        mentornet_preparation_status,
        resolve_config_paths,
    )

    recipes = _web_recipe_catalog()
    payload = []
    for paper in _web_paper_specs():
        default_config = next(
            (config for config in paper.configs if config.profile == "reproduction"),
            None,
        )
        default_recipe = recipes[default_config.recipe_id] if default_config else None
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
            "default_recipe_id": default_recipe.id if default_recipe else "",
            "default_variant": default_config.variant if default_config else "",
            "default_fidelity": default_config.configuration_fidelity if default_config else "",
            "configs": [],
        })
        if default_config is None:
            payload[-1].update(availability="unavailable", metadata_only=True)
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
                "external_resources": _external_resource_payload(config.recipe_id)["resources"],
            })
            if paper.id == "mentornet":
                resolved = resolve_config_paths(load_recipe_config(recipe), ROOT)
                payload[-1]["configs"][-1]["preparation"] = (
                    mentornet_preparation_status(
                        resolved,
                        ROOT,
                        student_recipe=config.recipe_id,
                    )
                )
    return payload


def _dataset_payload() -> dict[str, object]:
    """Expose adapter-backed readiness through the shared DataService."""

    from lnl_toolbox.training.data_service import DataService

    service = DataService()
    from lnl_toolbox.data.profile import resolve_dataset_capabilities

    adapters = [
        name for name in service.registry.names() if not name.startswith("synthetic_")
    ]
    datasets = []
    for report in service.list_datasets():
        item = report.to_dict()
        if report.status == "ready" and report.profile is not None and report.location:
            declared = service.declarations(report.name)
            capabilities = resolve_dataset_capabilities(report.profile, declared)
            item["noise_facts"] = {
                "status": capabilities.noise_status.value,
                "rate": capabilities.noise_rate.to_dict(),
                "inspected_status": report.profile.noise.status.value,
                "inspected_rate": report.profile.noise.rate.to_dict(),
                "declared": declared.to_dict(),
            }
        datasets.append(item)
    return {
        "catalog": str(service.catalog.path),
        "adapters": adapters,
        "datasets": datasets,
    }


def _dataset_profile_context(name: object) -> tuple[dict[str, object], object]:
    """Return the public profile payload and its inspected capabilities."""

    from lnl_toolbox.training.service import ExperimentService

    alias = str(name).strip()
    if not alias:
        raise ValueError("dataset profile requires a dataset alias")
    service = ExperimentService()
    report = service.inspect_dataset(alias)
    if report.status != "ready" or report.profile is None:
        raise ValueError(report.error or f"dataset profile is unavailable: {alias}")
    profile = report.profile.to_dict()
    declared = service.data_service.declarations(alias).to_dict()
    capabilities = service.data_service.capabilities(alias, persist=False)
    unresolved = []
    for field, value in (
        ("clean_train_labels", capabilities.clean_train_labels.value),
        ("noise_status", capabilities.noise_status.value),
        ("noise_origin", capabilities.noise_origin.value),
        ("dataset_noise_rate", capabilities.noise_rate.status.value),
    ):
        if value == "unknown":
            unresolved.append(field)
    payload = {
        "dataset": alias,
        "adapter": report.adapter,
        "source": report.location,
        "profile": profile,
        "detected": profile,
        "capabilities": capabilities.to_dict(),
        "unresolved_dataset_facts": unresolved,
        "declared": declared,
    }
    return payload, capabilities


def _dataset_compatibility_payload(
    name: object,
    *,
    method_noise_rate_prior: float | None = None,
) -> dict[str, object]:
    """Expose the exact ExperimentService compatibility result to Web."""

    from lnl_toolbox.training.service import ExperimentService

    alias = str(name).strip()
    if not alias:
        raise ValueError("compatibility query requires a dataset alias")
    service = ExperimentService()
    profile = _dataset_profile_payload(alias)
    methods = service.list_compatible_methods(
        alias,
        method_noise_rate_prior=method_noise_rate_prior,
    )
    return {
        "dataset": alias,
        "profile": profile["profile"],
        "detected": profile["detected"],
        "capabilities": profile.get("capabilities", {}),
        "unresolved_dataset_facts": profile.get("unresolved_dataset_facts", []),
        "declared": profile["declared"],
        "method_noise_rate_prior": method_noise_rate_prior,
        "methods": [result.to_dict() for result in methods],
    }


def _dataset_recipe_compatibility_payload(
    name: object,
    *,
    method_noise_rate_prior: float | None = None,
) -> dict[str, object]:
    """Resolve each paper's concrete formal recipe for one local dataset."""

    alias = str(name).strip()
    if not alias:
        raise ValueError("recipe compatibility query requires a dataset alias")
    key = (alias, method_noise_rate_prior)
    # Compatibility is a read-heavy page operation. Serialize the first
    # computation for a key so concurrent browser requests do not all reload
    # the same CIFAR source and run the same 26-recipe check.
    with _DATASET_RECIPE_COMPATIBILITY_LOCK:
        cached = _DATASET_RECIPE_COMPATIBILITY_CACHE.get(key)
        if cached is not None:
            from copy import deepcopy

            return deepcopy(cached)
        value = _compute_dataset_recipe_compatibility_payload(
            alias,
            method_noise_rate_prior=method_noise_rate_prior,
        )
        _DATASET_RECIPE_COMPATIBILITY_CACHE[key] = value
        from copy import deepcopy

        return deepcopy(value)


def _compute_dataset_recipe_compatibility_payload(
    alias: str,
    *,
    method_noise_rate_prior: float | None = None,
) -> dict[str, object]:
    """Compute one compatibility result; callers provide cache coordination."""

    from lnl_toolbox.training.service import ExperimentService

    papers = _paper_payload()
    recipe_meta: dict[str, dict[str, object]] = {}
    for paper in papers:
        if not paper.get("default_recipe_id"):
            continue
        paper_configs = paper.get("configs") or [{
            "recipe_id": paper["default_recipe_id"],
            "profile": "reproduction",
            "configuration_fidelity": paper["default_fidelity"],
        }]
        formal_configs = [item for item in paper_configs if item.get("profile") == "reproduction"]
        if not formal_configs:
            formal_configs = [next(item for item in paper_configs if item.get("recipe_id") == paper["default_recipe_id"])]
        for config_meta in formal_configs:
            recipe_id = str(config_meta["recipe_id"])
            recipe_meta[recipe_id] = {
                "paper_id": paper["id"],
                "acronym": paper["acronym"],
                "title": paper["title"],
                "fidelity": config_meta.get("configuration_fidelity", paper["default_fidelity"]),
            }
    # Use the process-local recipe catalog/config cache.  Calling
    # ``recipe_by_id`` here once per paper used to rediscover and parse the
    # complete recipe directory for every entry (over 2,000 YAML reads on a
    # single compatibility page).
    configs = {
        recipe_id: _web_recipe_config(recipe_id)
        for recipe_id in recipe_meta
    }
    results = {}
    profile = None
    if configs:
        service = ExperimentService()
        profile, capabilities = _dataset_profile_context(alias)
        results = dict(service.list_config_compatibility(
            alias,
            configs,
            method_noise_rate_prior=method_noise_rate_prior,
            dataset_capabilities=capabilities,
        ))
    recipes = []
    for recipe_id, meta in recipe_meta.items():
        result = results[recipe_id].to_dict()
        recipes.append({
            "paper_id": meta["paper_id"],
            "acronym": meta["acronym"],
            "title": meta["title"],
            "recipe_id": recipe_id,
            "profile": "reproduction",
            "fidelity": meta["fidelity"],
            **result,
        })
    grouped: dict[str, dict[str, object]] = {}
    for item in recipes:
        group = grouped.setdefault(str(item["paper_id"]), {
            "paper_id": item["paper_id"],
            "acronym": item["acronym"],
            "title": item["title"],
            "recipes": [],
        })
        group["recipes"].append(item)
    payload = {
        "dataset": alias,
        "profile": profile,
        "methods": list(grouped.values()),
        "recipes": recipes,
    }
    return payload


def _dataset_profile_payload(name: object) -> dict[str, object]:
    """Return the Phase-2 profile contract for one registered dataset."""

    payload, _ = _dataset_profile_context(name)
    return payload


def _invalidate_dataset_recipe_compatibility(name: object | None = None) -> None:
    with _DATASET_RECIPE_COMPATIBILITY_LOCK:
        if name is None:
            _DATASET_RECIPE_COMPATIBILITY_CACHE.clear()
            return
        alias = str(name).strip()
        for key in tuple(_DATASET_RECIPE_COMPATIBILITY_CACHE):
            if key[0] == alias:
                _DATASET_RECIPE_COMPATIBILITY_CACHE.pop(key, None)


def _dataset_declarations_payload(name: object, payload: object) -> dict[str, object]:
    """Persist dataset facts only; method priors remain experiment-local."""

    if not isinstance(payload, dict):
        raise ValueError("dataset declarations request must be a JSON object")
    declarations = payload.get("declarations", {})
    if not isinstance(declarations, dict):
        raise ValueError("declarations must be a JSON object")
    if "method_noise_rate_prior" in declarations or "pretrained_roles" in declarations:
        raise ValueError("method noise-rate prior and pretrained roles are experiment-specific")
    if "method_noise_rate_prior" in payload:
        raise ValueError("method noise-rate prior is an experiment input, not a dataset declaration")
    from lnl_toolbox.training.data_service import DataService

    service = DataService()
    service.update_declarations(name, declarations)
    _invalidate_dataset_recipe_compatibility(name)
    return _dataset_compatibility_payload(name)


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
        result = {
            "dataset": service.register(name, adapter, data).to_dict(),
            "message": "登记已保存。下一步请执行 inspect，实际加载 train/test。",
            "next_action": "inspect",
        }
        _invalidate_dataset_recipe_compatibility(name)
        return result
    if action == "inspect":
        if not name:
            raise ValueError("dataset inspection requires a name")
        report = service.inspect(name)
        if report.status != "ready":
            raise ValueError(report.error or f"dataset is not ready: {name}")
        result = {
            "dataset": report.to_dict(),
            "message": "数据检查通过。可继续执行 verify，完成一轮训练验证。",
            "next_action": "verify",
        }
        _invalidate_dataset_recipe_compatibility(name)
        return result
    if action == "remove":
        if not name:
            raise ValueError("dataset removal requires a name")
        service.remove(name)
        result = {
            "removed": name,
            "message": f"已删除数据登记：{name}。原始数据文件未被删除。",
            "next_action": "list",
        }
        _invalidate_dataset_recipe_compatibility(name)
        return result
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
    # Verification can update the catalog/profile facts. Never reuse a
    # compatibility result produced before this background job.
    _invalidate_dataset_recipe_compatibility(name)
    display = "lnl " + " ".join(shlex.quote(item) for item in command[len(resolve_lnl_command()):])
    return _start_process("data-verify", command, display)


# Backwards-compatible names; the shared module owns all parameter rules.
from lnl_toolbox.parameters import (
    _LEGACY_EDITABLE_SECTIONS,
    _LEGACY_PROTECTED_LEAF_NAMES,
    _PARAMETER_REGISTRY_PATH,
    _PARAMETER_LEVEL_ORDER,
    _project_path,
    _config_payload,
    _builtin_config_paths,
    _legacy_editable_config_fields,
    _parameter_registry,
    _parameter_display_policy,
    _internal_seed_path,
    _parameter_presentation,
    _method_registry_key,
    _config_method,
    _config_path_value,
    _value_kind,
    _metadata_field_kind,
    _registry_recipe_paths,
    _formal_registry_config,
    _research_parameter_category,
    _parameter_display_group,
    _parameter_context,
    _parameter_user_note,
    _parameter_base_note,
    _parameter_ui_spec,
    _parameter_choices,
    _feature_layer_choices,
    _parameter_list_rows,
    _parameter_user_label,
    _registry_config_fields,
    _config_schema,
    _field_map,
    _stage_early_stopping_fields,
    _schedule_editor_fields,
    _validate_schedule_editor_rates,
    _set_config_path,
    _normalize_registry_numeric_fields,
    _coerce_patch_value,
    _parameter_edit_rule,
    _synchronize_web_derived_parameters,
    _validate_changed_model_configs,
    _validate_changed_data_noise_selectors,
    _validate_changed_training_selectors,
    _assert_locked_parameters_unchanged,
    _paper_parameter_changes,
    _record_paper_parameter_status,
    _save_config,
)


def _quick_start_config(payload: dict[str, object], *, review: bool) -> dict[str, object]:
    body = {key: payload.get(key) for key in ("recipe", "source_path", "recipe_hint", "patches", "dataset_alias")}
    body["acknowledge_paper_impact"] = True
    body["automatic_validation_split"] = True
    run_id = uuid.uuid4().hex if review else str(payload.get("run_id") or uuid.uuid4().hex)
    if uuid.UUID(run_id).hex != run_id:
        raise ValueError("invalid run identifier")
    run_dir = ROOT / "artifacts" / "web-runs" / run_id
    locations = {"run_id": run_id, "output_dir": str(run_dir), "config_path": str(run_dir / "config.yaml")}
    if not review:
        body.update(content=payload.get("content"), path=str(run_dir / "config.yaml"))
        return {**_save_config(body), **locations}
    directory = tempfile.TemporaryDirectory(prefix="lnl-review-")
    path = str(Path(directory.name) / "config.yaml")
    try:
        saved = _save_config(body, for_review=True, destination=Path(path))
    except Exception:
        directory.cleanup()
        raise
    with JOBS_LOCK:
        REVIEW_FILES[path] = directory
    # Clean abandoned requests too; an active CLI owns its file until exit.
    def expire() -> None:
        with JOBS_LOCK:
            active = any(path in job.command and not job.output_complete for job in JOBS.values())
        if not active:
            _cleanup_review_file(path)
    timer = threading.Timer(600, expire)
    timer.daemon = True
    timer.start()
    return {**saved, **locations}


def _job_payload(job: Job) -> dict[str, object]:
    _reconcile_job_process(job)
    with JOBS_LOCK:
        payload: dict[str, object] = {
            "id": job.job_id,
            "key": job.key,
            "command": job.display_command,
            "lines": list(job.lines),
            "returncode": job.returncode,
            "error": job.error,
            "running": not job.done,
            "output_complete": job.output_complete,
            "cancel_requested": job.cancel_requested,
            "structured": job.structured,
        }
        context = job.training_context
        lines = list(job.lines)
        running = not job.done
        returncode = job.returncode
        cancel_requested = job.cancel_requested
    if context is not None:
        from web.training_status import training_snapshot

        snapshot = training_snapshot(
            context,
            lines=lines,
            running=running,
            returncode=returncode,
            cancel_requested=cancel_requested,
        )
        payload["training"] = None if snapshot is None else snapshot.to_dict()
    else:
        payload["training"] = None
    return payload


def job_list_payload() -> dict[str, object]:
    """Return current and recent main-console jobs for the process panel."""

    with JOBS_LOCK:
        jobs = list(reversed(list(JOBS.values())))[:30]
    return {"jobs": [_job_payload(job) for job in jobs]}


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
        "$owner=New-Object System.Windows.Forms.Form; $d=$null; "
        "$owner.ShowInTaskbar=$false; $owner.TopMost=$true; "
        "$owner.FormBorderStyle=[System.Windows.Forms.FormBorderStyle]::None; "
        "$owner.StartPosition=[System.Windows.Forms.FormStartPosition]::CenterScreen; "
        "$owner.Width=1; $owner.Height=1; $owner.Opacity=0; "
        "try { [void]$owner.Show(); [void]$owner.Activate(); "
        "if($mode -eq 'folder'){ $d=New-Object System.Windows.Forms.FolderBrowserDialog; "
        "if($initial -and (Test-Path -LiteralPath $initial)){ $d.SelectedPath=$initial }; "
        "if($d.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK){[Console]::Write($d.SelectedPath)} } "
        "else { if($mode -eq 'save_file'){ $d=New-Object System.Windows.Forms.SaveFileDialog } "
        "else { $d=New-Object System.Windows.Forms.OpenFileDialog }; "
        "$d.Filter=$env:LNL_PICKER_FILTER; "
        "if($initial){ if(Test-Path -LiteralPath $initial -PathType Container){$d.InitialDirectory=$initial} "
        "elseif(Test-Path -LiteralPath $initial){$d.InitialDirectory=(Split-Path -Parent $initial);$d.FileName=(Split-Path -Leaf $initial)} }; "
        "if($d.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK){[Console]::Write($d.FileName)} } } "
        "finally { if($null -ne $d){$d.Dispose()}; $owner.Close(); $owner.Dispose() }"
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
    from lnl_toolbox.catalog import resolve_config_paths
    from lnl_toolbox.training.service import ExperimentService
    from lnl_toolbox.training.sweep import plan_sweep, resolve_planned_config

    recipe_value = str(payload.get("recipe") or "").strip()
    path_value = str(payload.get("path") or "").strip()
    if bool(recipe_value) == bool(path_value):
        raise ValueError("sweep source requires exactly one of recipe or path")
    source_payload = _config_payload(
        recipe_value or None,
        path_value=path_value or None,
    )
    schema = _config_schema(
        recipe_value or None,
        path_value=path_value or None,
    )
    config = resolve_config_paths(source_payload["config"], ROOT)
    dataset_alias = str(payload.get("dataset_alias") or "").strip()
    if dataset_alias:
        config = ExperimentService().data_service.apply(config, dataset_alias)
    matrix = payload.get("matrix", {}) or {}
    if not isinstance(matrix, dict):
        raise TypeError("sweep matrix must be an object")
    fields = {str(field["path"]): field for field in schema.get("fields", [])}
    normalized_matrix: dict[str, list[object]] = {}
    for raw_path, raw_values in matrix.items():
        path = str(raw_path)
        field = fields.get(path)
        if field is None:
            raise ValueError(f"sweep parameter is not registered for this config: {path}")
        if not field.get("editable", False):
            raise ValueError(f"locked parameter cannot be swept from WebUI: {path}")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"sweep matrix {path!r} must be a non-empty list")
        normalized_matrix[path] = [
            _coerce_patch_value(field, value) for value in raw_values
        ]
    raw_seeds = payload.get("seeds")
    seeds = [int(config.get("seed", 1))] if raw_seeds in (None, []) else raw_seeds
    if not isinstance(seeds, list):
        raise TypeError("sweep seeds must be a list")
    recipe_id = str(source_payload["recipe"] or "")
    plan = plan_sweep(
        config,
        seeds,
        matrix=normalized_matrix,
        output_dir=payload.get("output_dir") or None,
        recipe=recipe_id or None,
    )
    service = ExperimentService()
    for run in plan.runs:
        service.preflight(resolve_planned_config(config, run), check_data=False)
    matrix_fields = [fields[path] for path in normalized_matrix]
    base_paper_changes = set(schema.get("paper_changes", []))
    paper_matrix_paths = {
        str(field["path"])
        for field in matrix_fields
        if field.get("level") == "paper"
        and any(
            value != field.get("changed_from")
            for value in normalized_matrix[str(field["path"])]
        )
    }
    paper_deviation_paths = sorted(base_paper_changes | paper_matrix_paths)
    return {
        "root": str(plan.root),
        "source": {
            "kind": "recipe" if recipe_id else "path",
            "value": recipe_id or str(source_payload["path"]),
            "label": recipe_id or str(source_payload["path"]),
        },
        "seeds": list(plan.seeds),
        "matrix": {path: list(values) for path, values in plan.matrix},
        "matrix_fields": matrix_fields,
        "levels": list(schema.get("levels", [])),
        "paper_deviation_paths": paper_deviation_paths,
        "modified_from_paper": bool(paper_deviation_paths),
        "total": len(plan.runs),
        "runs": [
            {
                "run_id": run.run_id,
                "seed": run.seed,
                "overrides": run.override_mapping,
                "paper_deviation_paths": sorted(
                    base_paper_changes
                    | {
                        path
                        for path, value in run.overrides
                        if fields[path].get("level") == "paper"
                        and value != fields[path].get("changed_from")
                    }
                ),
            }
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
        if path in STATIC_ASSETS:
            asset, content_type = STATIC_ASSETS[path]
            self._serve_file(asset, content_type)
            return
        if path in {"/", "/recipe", "/recipe/"}:
            self._serve_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if path in {"/scratch", "/scratch/"}:
            self._serve_file(SCRATCH_WEB_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if path in {"/scratch.js", "/scratch.css"}:
            content_type = "application/javascript; charset=utf-8" if path.endswith(".js") else "text/css; charset=utf-8"
            self._serve_file(SCRATCH_WEB_ROOT / path.lstrip("/"), content_type)
            return
        if path == "/api/scratch/blocks":
            try:
                _scratch_reload_formulas()
                from lnl_toolbox.scratch import list_blocks

                _json_response(self, [definition.describe() for definition in list_blocks()])
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 500)
            return
        if path == "/api/scratch/formulas":
            try:
                _scratch_reload_formulas()
                from lnl_toolbox.scratch.formula.registry import list_formulas

                _json_response(self, [_scratch_formula_payload(spec) for spec in list_formulas()])
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 400)
            return
        if path.startswith("/api/scratch/formula/") and not path.endswith("/export"):
            try:
                from lnl_toolbox.scratch.formula.registry import get_formula

                formula_id = unquote(path.removeprefix("/api/scratch/formula/"))
                _json_response(self, _scratch_formula_payload(get_formula(formula_id)))
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 404)
            return
        if path.startswith("/api/scratch/formula/") and path.endswith("/export"):
            try:
                from lnl_toolbox.scratch.formula.storage import export_formula

                formula_id = unquote(path.removeprefix("/api/scratch/formula/").removesuffix("/export"))
                _json_response(self, export_formula(formula_id))
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 404)
            return
        if path == "/api/scratch/default-recipe":
            try:
                from lnl_toolbox.scratch import load_recipe

                _json_response(self, load_recipe(SCRATCH_RECIPE_ROOT / "examples" / "default_supervised.yaml"))
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 400)
            return
        if path == "/api/scratch/entry-recipe":
            _json_response(self, {
                "schema_version": 1,
                "name": "空白 Scratch 算法",
                "description": "选择单模型、双模型、论文模板或从空白开始。",
                "settings": {},
                "steps": [],
            })
            return
        if path == "/api/scratch/templates":
            try:
                _json_response(self, _scratch_template_payload())
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 400)
            return
        if path == "/api/scratch/examples":
            try:
                _json_response(self, _scratch_examples_payload())
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 400)
            return
        if path == "/api/scratch/datasets":
            try:
                from lnl_toolbox.scratch.web.data_bridge import dataset_catalog_payload

                _json_response(self, dataset_catalog_payload())
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 400)
            return
        if path.startswith("/api/scratch/dataset/"):
            try:
                from lnl_toolbox.scratch.web.data_bridge import dataset_fact_payload

                alias = unquote(path.removeprefix("/api/scratch/dataset/"))
                _json_response(self, dataset_fact_payload(alias))
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 400)
            return
        if path == "/api/scratch/recipes":
            try:
                _json_response(self, _scratch_recipe_api_payload())
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 500)
            return
        if path.startswith("/api/scratch/recipe/"):
            try:
                from lnl_toolbox.scratch import load_recipe

                name = unquote(path.removeprefix("/api/scratch/recipe/"))
                _json_response(self, load_recipe(_scratch_recipe_path(name)))
            except FileNotFoundError as exc:
                _json_response(self, _scratch_error(exc), 404)
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 400)
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
        if path == "/api/quick-start/noises":
            try:
                query = parse_qs(urlparse(self.path).query)
                alias = query.get("dataset", [""])[0].strip()
                if not alias:
                    raise ValueError("provide a dataset alias")
                from web.quick_start_api import noise_options_payload

                _json_response(self, noise_options_payload(alias))
            except (KeyError, OSError, TypeError, ValueError) as exc:
                _json_response(self, {"error": str(exc)}, 400)
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
        if path == "/api/external-resources":
            try:
                query = parse_qs(urlparse(self.path).query)
                _json_response(self, _external_resource_payload(
                    *(query.get(key, [""])[0] for key in ("recipe", "config_path", "dataset", "paper_id", "noise"))))
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, 400)
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
        if path.startswith("/api/datasets/"):
            try:
                parts = path.split("/")
                if len(parts) != 5:
                    raise ValueError("unknown dataset API endpoint")
                alias = unquote(parts[3])
                action = parts[4]
                if action == "profile":
                    value = _dataset_profile_payload(alias)
                elif action == "compatible-methods":
                    query = parse_qs(urlparse(self.path).query)
                    raw_prior = query.get("method_noise_rate_prior", [""])[0].strip()
                    value = _dataset_compatibility_payload(
                        alias,
                        method_noise_rate_prior=(
                            None if not raw_prior else float(raw_prior)
                        ),
                    )
                elif action == "compatible-recipes":
                    query = parse_qs(urlparse(self.path).query)
                    raw_prior = query.get("method_noise_rate_prior", [""])[0].strip()
                    value = _dataset_recipe_compatibility_payload(
                        alias,
                        method_noise_rate_prior=(
                            None if not raw_prior else float(raw_prior)
                        ),
                    )
                else:
                    raise ValueError("unknown dataset API endpoint")
                _json_response(self, value)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                _json_response(self, {"error": str(exc)}, 400)
            except Exception as exc:
                _json_response(self, {"error": f"dataset backend error: {exc}"}, 500)
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
                recipe_hint = query.get("recipe_hint", [""])[0]
                _json_response(
                    self,
                    _config_schema(recipe_id, path_value=path_value, recipe_hint=recipe_hint),
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
        if path == "/api/scratch/jobs":
            try:
                from lnl_toolbox.scratch.web.server import scratch_job_list_payload

                _json_response(self, scratch_job_list_payload())
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, 400)
            return
        if path == "/api/jobs":
            _json_response(self, job_list_payload())
            return
        if path.startswith("/api/scratch/jobs/"):
            try:
                from lnl_toolbox.scratch.web.server import SCRATCH_JOBS, SCRATCH_JOBS_LOCK, _scratch_job_payload

                job_id = path.removeprefix("/api/scratch/jobs/").strip("/")
                with SCRATCH_JOBS_LOCK:
                    job = SCRATCH_JOBS.get(job_id)
                if job is None:
                    _json_response(self, {"error": "Scratch job not found"}, 404)
                else:
                    _json_response(self, _scratch_job_payload(job))
            except Exception as exc:
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
        if path == "/api/external-resources/download":
            try:
                from web.external_resources import start_download
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                payload = _external_resource_payload(*(str(body.get(key, "")) for key in ("recipe", "config_path", "dataset", "paper_id", "noise")))
                item = next((item for item in payload["resources"] if item["id"] == body.get("resource")), None)
                if item is None:
                    raise ValueError("当前配置不需要该资源")
                start_download(item)
                _json_response(self, {"started": True}, 202)
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, 400)
            return
        if path.startswith("/api/quick-start/"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                from web import quick_start_api

                handlers = {
                    "/api/quick-start/review-config": lambda body: _quick_start_config(body, review=True),
                    "/api/quick-start/training-config": lambda body: _quick_start_config(body, review=False),
                    "/api/quick-start/probe": quick_start_api.probe_payload,
                    "/api/quick-start/register": quick_start_api.register_payload,
                    "/api/quick-start/methods": quick_start_api.method_options_payload,
                    "/api/quick-start/plan": quick_start_api.plan_payload,
                }
                handler = handlers.get(path)
                if handler is None:
                    raise ValueError("unknown Quick Start API endpoint")
                _json_response(self, handler(payload))
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                _json_response(self, {"error": str(exc)}, 400)
            except Exception as exc:
                _json_response(self, {"error": f"Quick Start backend error: {exc}"}, 500)
            return
        if path in {"/api/scratch/validate", "/api/scratch/save", "/api/scratch/run"}:
            try:
                payload = _scratch_request_body(self)
                recipe = _scratch_recipe_from_body(payload)
                if path == "/api/scratch/validate":
                    preflight = _scratch_dataset_preflight(recipe, payload.get("runtime_limits"))
                    if not preflight.get("ok", False):
                        _json_response(self, preflight, 400)
                        return
                    _json_response(self, {"ok": True, "recipe": recipe, "dataset_preflight": preflight})
                elif path == "/api/scratch/save":
                    from lnl_toolbox.scratch import recipe_workspace_root, save_recipe

                    name = Path(str(recipe["name"])).name
                    destination = recipe_workspace_root() / f"{name}.yaml"
                    save_recipe(recipe, destination)
                    _json_response(
                        self,
                        {"ok": True, "path": str(destination)},
                        201,
                    )
                else:
                    from lnl_toolbox.scratch.web.server import _scratch_job_payload, start_scratch_job

                    preflight = _scratch_dataset_preflight(recipe, payload.get("runtime_limits"))
                    if not preflight.get("ok", False):
                        _json_response(self, preflight, 400)
                        return
                    job = start_scratch_job(recipe, payload.get("runtime_limits", {}))
                    _json_response(self, _scratch_job_payload(job), 202)
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 400)
            return
        if path.startswith("/api/scratch/jobs/") and path.endswith("/cancel"):
            try:
                from lnl_toolbox.scratch.web.server import _scratch_job_payload, cancel_scratch_job

                job_id = path.removeprefix("/api/scratch/jobs/").removesuffix("/cancel").strip("/")
                _json_response(self, _scratch_job_payload(cancel_scratch_job(job_id)), 202)
            except KeyError:
                _json_response(self, {"error": "Scratch job not found"}, 404)
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, 400)
            return
        if path == "/api/scratch/formulas":
            try:
                payload = _scratch_request_body(self)
                from lnl_toolbox.scratch.formula.registry import validate_and_register_formula
                from lnl_toolbox.scratch.formula.storage import save_formula

                spec = validate_and_register_formula(payload.get("formula", payload), replace=bool(payload.get("replace", False)))
                destination = save_formula(spec, overwrite=True)
                _json_response(self, {"ok": True, "formula": _scratch_formula_payload(spec), "path": str(destination)}, 201)
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 400)
            return
        if path == "/api/scratch/formula/delete":
            try:
                payload = _scratch_request_body(self)
                from lnl_toolbox.scratch.formula.registry import unregister_formula
                from lnl_toolbox.scratch.formula.storage import delete_formula, find_formula_references

                formula_id = str(payload.get("id", ""))
                from lnl_toolbox.scratch import recipe_workspace_root

                references = find_formula_references(SCRATCH_RECIPE_ROOT, formula_id)
                references.extend(find_formula_references(recipe_workspace_root(), formula_id))
                spec = delete_formula(formula_id, referenced_by=tuple(references))
                unregister_formula(spec.id)
                _json_response(self, {"ok": True, "id": spec.id})
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 400)
            return
        if path == "/api/scratch/formula/import":
            try:
                payload = _scratch_request_body(self)
                from lnl_toolbox.scratch.formula.registry import validate_and_register_formula
                from lnl_toolbox.scratch.formula.storage import import_formula

                spec = import_formula(payload.get("formula", payload.get("yaml", "")), overwrite=bool(payload.get("replace", False)))
                validate_and_register_formula(spec, replace=True)
                _json_response(self, {"ok": True, "formula": _scratch_formula_payload(spec)}, 201)
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 400)
            return
        if path == "/api/scratch/formula/rename":
            try:
                payload = _scratch_request_body(self)
                from lnl_toolbox.scratch.formula.registry import register_formula, unregister_formula
                from lnl_toolbox.scratch.formula.storage import rename_formula

                old_id, new_id = str(payload.get("old_id", "")), str(payload.get("new_id", ""))
                spec = rename_formula(old_id, new_id)
                try:
                    unregister_formula(old_id)
                except KeyError:
                    pass
                register_formula(spec, replace=True)
                _json_response(self, {"ok": True, "formula": _scratch_formula_payload(spec)}, 201)
            except Exception as exc:
                _json_response(self, _scratch_error(exc), 400)
            return
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job_id = path.split("/")[3]
            try:
                _json_response(self, _job_payload(cancel_job(job_id)), 202)
            except KeyError as exc:
                _json_response(self, {"error": str(exc)}, 404)
            except OSError as exc:
                _json_response(self, {"error": str(exc)}, 409)
            return
        if path.startswith("/api/datasets/"):
            try:
                parts = path.split("/")
                if len(parts) != 5 or parts[4] != "declarations":
                    raise ValueError("unknown dataset API endpoint")
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                _json_response(
                    self,
                    _dataset_declarations_payload(unquote(parts[3]), payload),
                )
            except (
                KeyError,
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                _json_response(self, {"error": str(exc)}, 400)
            except Exception as exc:
                _json_response(self, {"error": f"dataset backend error: {exc}"}, 500)
            return
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


class SingleInstanceHTTPServer(ThreadingHTTPServer):
    """Refuse another listener on the same host/port, including on Windows."""

    allow_reuse_address = False

    def server_bind(self) -> None:
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    open_browser: bool = False,
) -> None:
    # Reserve the port before the expensive warm-up.  A second launch must
    # fail immediately instead of starting another request-serving process.
    server = SingleInstanceHTTPServer((host, port), ConsoleHandler)
    # Warm imports/catalogs before accepting browser requests.  Without this,
    # the first compatibility request pays the cost of importing torch/
    # torchvision and discovering every recipe while the UI displays a
    # generic loading card.  Fail-soft so the server can still start when an
    # optional training dependency is unavailable.
    try:
        try:
            from lnl_toolbox.training.service import ExperimentService  # noqa: F401

            _web_recipe_catalog()
            _web_paper_specs()
        except Exception:
            pass
        browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        url = f"http://{browser_host}:{port}/"
        print(f"LNL web: {url}", flush=True)
        if open_browser:
            threading.Timer(0.25, webbrowser.open, args=(url,)).start()
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
    try:
        serve(args.host, args.port, open_browser=args.open_browser)
    except OSError as exc:
        if exc.errno not in {errno.EADDRINUSE, 10048} and getattr(exc, "winerror", None) != 10048:
            raise
        print(
            f"Web port {args.port} is already in use. Use the running service or stop the old instance first.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
