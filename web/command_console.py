"""Local web command console for the LNL toolbox.

The console intentionally exposes a small, fixed command allowlist. It does
not execute browser-provided shell strings and never starts a shell.
"""

from __future__ import annotations

import argparse
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
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
        ("list", "experiments", "--profile", "smoke"),
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
        ("compare", "artifacts/web-smoke"),
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
        ("papers", "list", "--format", "tsv"),
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


def _recipe_payload() -> list[dict[str, object]]:
    """Return all repository recipes for the beginner menu."""

    from lnl_toolbox.catalog import discover_recipes

    recipes = discover_recipes(ROOT, include_conditional=True)
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
        }
        for recipe in recipes
    ]


def _paper_payload() -> list[dict[str, object]]:
    """Return paper metadata and available recipe variants for the UI."""

    from lnl_toolbox.catalog import load_papers

    return [
        {
            "id": paper.id,
            "acronym": paper.acronym,
            "title": paper.title,
            "venue": paper.venue,
            "year": paper.year,
            "implementation_status": paper.implementation_status,
            "reproduction_status": paper.reproduction_status,
            "configs": [
                {
                    "recipe_id": config.recipe_id,
                    "profile": config.profile,
                    "variant": config.variant,
                    "availability": config.availability,
                }
                for config in paper.configs
            ],
        }
        for paper in load_papers(ROOT)
    ]


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


def _config_payload(recipe_id: object) -> dict[str, object]:
    from lnl_toolbox.catalog import load_yaml, recipe_by_id

    recipe = recipe_by_id(str(recipe_id), ROOT)
    path = recipe.config_path.resolve()
    return {
        "recipe": recipe.id,
        "runner": recipe.runner,
        "method": recipe.method,
        "path": path.relative_to(ROOT).as_posix(),
        "content": path.read_text(encoding="utf-8"),
        "config": load_yaml(path),
    }


def _builtin_config_paths() -> set[Path]:
    from lnl_toolbox.catalog import discover_recipes

    return {recipe.config_path.resolve() for recipe in discover_recipes(ROOT, include_conditional=True)}


_EDITABLE_SECTIONS = {
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
_PROTECTED_LEAF_NAMES = {
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


def _editable_config_fields(config: object, *, prefix: str = "") -> list[dict[str, object]]:
    """Expose scalar hyperparameters without exposing component wiring."""

    fields: list[dict[str, object]] = []
    if not isinstance(config, dict):
        return fields
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            if prefix or key in _EDITABLE_SECTIONS:
                fields.extend(_editable_config_fields(value, prefix=path))
            continue
        if not prefix and key not in {"seed", "output_root"}:
            continue
        root = path.split(".", 1)[0]
        if root not in _EDITABLE_SECTIONS and path not in {"seed", "output_root"}:
            continue
        if str(key) in _PROTECTED_LEAF_NAMES and path != "data.name":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            kind = "boolean" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "text"
            fields.append({"path": path, "label": path, "value": value, "kind": kind})
        elif isinstance(value, list):
            fields.append({"path": path, "label": path, "value": value, "kind": "list"})
    return fields


def _config_schema(recipe_id: object) -> dict[str, object]:
    payload = _config_payload(recipe_id)
    return {
        "recipe": payload["recipe"],
        "runner": payload["runner"],
        "method": payload["method"],
        "fields": _editable_config_fields(payload["config"]),
    }


def _field_map(config: object) -> dict[str, dict[str, object]]:
    return {field["path"]: field for field in _editable_config_fields(config)}


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


def _save_config(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("请求内容必须是 JSON 对象")
    destination = _project_path(payload.get("path"))
    recipe_id = payload.get("recipe")
    patches = payload.get("patches")
    if not isinstance(recipe_id, str) or not recipe_id.strip():
        raise ValueError("保存 YAML 必须指定来源 recipe")
    if not isinstance(patches, list):
        raise ValueError("只能通过参数菜单提交 YAML 修改")
    from lnl_toolbox.catalog import load_yaml, recipe_by_id

    recipe = recipe_by_id(recipe_id, ROOT)
    parsed = load_yaml(recipe.config_path)
    fields = _field_map(parsed)
    for patch in patches:
        if not isinstance(patch, dict) or not isinstance(patch.get("path"), str):
            raise ValueError("YAML 参数修改格式错误")
        path = patch["path"]
        field = fields.get(path)
        if field is None:
            raise ValueError(f"不允许修改配置结构或组件：{path}")
        _set_config_path(parsed, path, _coerce_patch_value(field, patch.get("value")))
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
        }


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
        if path == "/api/recipes":
            try:
                _json_response(self, _recipe_payload())
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
                _json_response(self, _config_payload(recipe_id))
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, 400)
            return
        if path == "/api/config-schema":
            try:
                query = parse_qs(urlparse(self.path).query)
                recipe_id = query.get("recipe", [""])[0]
                _json_response(self, _config_schema(recipe_id))
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
            except (TypeError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
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
