"""Minimal standard-library Web server for the Scratch recipe editor."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from collections.abc import Mapping
from urllib.parse import unquote, urlparse

from .. import ScratchExecutionError, execute_recipe, list_blocks, load_recipe, recipe_workspace_root, resolve_recipe, save_recipe, validate_recipe


ROOT = Path(__file__).resolve().parent
RECIPE_ROOT = ROOT.parent / "recipes"
REPO_ROOT = ROOT.parents[3]
_FORMULA_RELOAD_LOCK = threading.RLock()


@dataclass
class ScratchJob:
    """A WebUI-owned Scratch process with observable progress and cancellation."""

    job_id: str
    recipe_name: str
    output_dir: Path
    command: list[str]
    process: subprocess.Popen[str] | None = None
    lines: list[str] = field(default_factory=list)
    returncode: int | None = None
    error: str | None = None
    structured: object | None = None
    cancel_requested: bool = False
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def done(self) -> bool:
        return self.returncode is not None


SCRATCH_JOBS: dict[str, ScratchJob] = {}
SCRATCH_JOBS_LOCK = threading.Lock()


def _read_progress(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _job_error_code(message: str | None) -> str | None:
    """Map common runtime failures to the same guidance codes as the UI."""

    text = str(message or "").lower()
    # Runtime diagnostics include the available Context slots (for example
    # ``_progress_path``).  Inspect the actual error tail before classifying
    # dataset failures so an incidental slot name cannot turn a model/data
    # mismatch into a misleading "missing path" hint.
    diagnostic = text.split("original error:", 1)[-1]
    if "num_classes" in diagnostic and any(token in diagnostic for token in ("match", "mismatch", "匹配", "类别")):
        return "dataset-model-class-mismatch"
    if "dataset" in text or "数据集" in text:
        if "requires a dataset name" in diagnostic or "dataset name" in diagnostic:
            return "missing-dataset-selection"
        if any(token in diagnostic for token in ("not found", "unavailable", "cannot materialize", "不存在", "不可用")):
            return "dataset-unavailable"
        if any(token in diagnostic for token in ("path", "root", "file", "目录", "路径", "dataset path")):
            return "missing-dataset-path"
        return "missing-dataset-selection"
    if "manifest" in text or "artifact" in text or "resource" in text:
        return "resource"
    if "missing context" in text or "requires" in text or "slot" in text:
        return "missing-slot"
    if "parameter" in text or "must be" in text or "invalid" in text:
        return "parameter"
    return None


def _scratch_job_payload(job: ScratchJob) -> dict[str, object]:
    with SCRATCH_JOBS_LOCK:
        lines = list(job.lines)
        returncode = job.returncode
        error = job.error
        running = not job.done
        cancel_requested = job.cancel_requested
        structured = job.structured
    progress = _read_progress(job.output_dir / "progress.json")
    if progress is None:
        progress = {"state": "starting" if running else ("cancelled" if cancel_requested else "failed")}
    if running:
        status = "stopping" if cancel_requested else "running"
    elif cancel_requested:
        status = "cancelled"
    elif returncode == 0:
        status = "completed"
    else:
        status = "failed"
    if error is None and returncode not in (None, 0) and lines:
        error_lines = [line for line in lines if "Scratch error:" in line]
        error = error_lines[-1] if error_lines else lines[-1]
    return {
        "id": job.job_id,
        "recipe": job.recipe_name,
        "command": " ".join(job.command),
        "lines": lines,
        "returncode": returncode,
        "error": error,
        "error_code": _job_error_code(error),
        "block_id": progress.get("block") if isinstance(progress, dict) else None,
        "running": running,
        "status": status,
        "cancel_requested": cancel_requested,
        "structured": structured,
        "artifact_dir": str(job.output_dir),
        "progress": progress,
        "elapsed_seconds": max(0.0, (job.finished_at or time.time()) - job.started_at),
    }


def _read_scratch_job_output(job: ScratchJob) -> None:
    process = job.process
    if process is None:
        return
    stream = process.stdout
    if stream is not None:
        for line in stream:
            with SCRATCH_JOBS_LOCK:
                job.lines.append(line.rstrip("\r\n"))
    returncode = process.wait()
    with SCRATCH_JOBS_LOCK:
        job.returncode = returncode
        job.finished_at = time.time()
        for line in reversed(job.lines):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "artifact_dir" in parsed:
                job.structured = parsed
                break
    progress_path = job.output_dir / "progress.json"
    progress = _read_progress(progress_path) or {}
    progress["state"] = "cancelled" if job.cancel_requested else ("completed" if returncode == 0 else "failed")
    if returncode != 0 and job.lines:
        error_lines = [line for line in job.lines if "Scratch error:" in line]
        if error_lines:
            progress["error"] = error_lines[-1]
    try:
        progress_path.write_text(json.dumps(progress, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def start_scratch_job(recipe: dict[str, object], runtime_limits: object = None) -> ScratchJob:
    """Persist one recipe and launch it outside the HTTP request thread."""

    if not isinstance(runtime_limits, dict):
        runtime_limits = {}
    job_id = uuid.uuid4().hex
    safe_name = Path(str(recipe.get("name") or "scratch_recipe")).name or "scratch_recipe"
    output = REPO_ROOT / "artifacts" / "scratch" / safe_name / job_id
    output.mkdir(parents=True, exist_ok=True)
    recipe_path = output / "recipe.yaml"
    save_recipe(recipe, recipe_path)
    save_recipe(resolve_recipe(recipe), output / "resolved_recipe.yaml")
    try:
        from ..formula.registry import collect_formula_provenance

        (output / "formula_provenance.json").write_text(
            json.dumps(collect_formula_provenance(recipe), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    command = [sys.executable, "-m", "lnl_toolbox.scratch.cli", "run", str(recipe_path), "--output", str(output)]
    if runtime_limits.get("max_epochs") is not None:
        command.extend(["--max-epochs", str(int(runtime_limits["max_epochs"]))])
    if runtime_limits.get("max_batches") is not None:
        command.extend(["--max-batches", str(int(runtime_limits["max_batches"]))])
    if runtime_limits.get("skip_final_test"):
        command.append("--skip-final-test")
    if runtime_limits.get("fixture"):
        if runtime_limits.get("max_epochs") is None:
            command.extend(["--max-epochs", "1"])
        if runtime_limits.get("max_batches") is None:
            command.extend(["--max-batches", "1"])
    environment = os.environ.copy()
    src_root = str(REPO_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [src_root, environment.get("PYTHONPATH", "")]))
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    job = ScratchJob(job_id, safe_name, output, command)
    try:
        job.process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            env=environment,
        )
    except OSError as exc:
        job.error = str(exc)
        job.returncode = -1
        job.finished_at = time.time()
    with SCRATCH_JOBS_LOCK:
        SCRATCH_JOBS[job.job_id] = job
    if job.process is not None:
        threading.Thread(target=_read_scratch_job_output, args=(job,), daemon=True).start()
    return job


def cancel_scratch_job(job_id: str) -> ScratchJob:
    with SCRATCH_JOBS_LOCK:
        job = SCRATCH_JOBS.get(job_id)
        if job is None:
            raise KeyError("Scratch job not found")
        if job.done:
            return job
        job.cancel_requested = True
        process = job.process
    if process is not None and process.poll() is None:
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
            # Some Windows environments deny taskkill for a child process
            # even though the owning Popen handle can terminate it.  Fall
            # back to that handle so Stop is never left in "stopping".
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
        else:
            process.terminate()
        # Do not report cancellation before the child has actually exited.
        # Recipe navigation can immediately start another run, so returning
        # while the old process is still alive would permit two Scratch jobs
        # to overlap briefly.
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    return job


def _recipe_path(name: str) -> Path:
    requested = Path(name)
    # Loading a read-only paper template must not depend on the optional user
    # recipe directory being writable.  On locked-down Windows profiles,
    # recipe_workspace_root() can raise PermissionError while the packaged
    # paper files are still perfectly readable.
    try:
        user_root = recipe_workspace_root()
    except OSError:
        user_root = None
    candidates = ([
        (user_root, user_root / requested),
    ] if user_root is not None else []) + [
        (RECIPE_ROOT, RECIPE_ROOT / requested),
        (RECIPE_ROOT, RECIPE_ROOT / "examples" / requested.name),
        (RECIPE_ROOT, RECIPE_ROOT / "papers" / requested.name),
    ]
    for root, candidate in candidates:
        resolved = candidate.resolve()
        root = root.resolve()
        if resolved != root and root not in resolved.parents:
            continue
        if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
            return candidate
    raise FileNotFoundError(name)


def _load_dataset_params(recipe: object) -> Mapping[str, object] | None:
    """Find the canonical load_dataset parameters, including nested steps."""

    def visit(steps: object) -> Mapping[str, object] | None:
        if not isinstance(steps, list):
            return None
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            if step.get("block") == "load_dataset" and isinstance(step.get("params"), Mapping):
                return step["params"]
            nested = visit(step.get("steps"))
            if nested is not None:
                return nested
        return None

    if isinstance(recipe, Mapping):
        return visit(recipe.get("steps"))
    return None


def _recipe_steps(recipe: object) -> list[Mapping[str, object]]:
    """Flatten recipe steps for server-side data capability checks."""

    result: list[Mapping[str, object]] = []

    def visit(steps: object) -> None:
        if not isinstance(steps, list):
            return
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            result.append(step)
            visit(step.get("steps"))

    if isinstance(recipe, Mapping):
        visit(recipe.get("steps"))
    return result


def _recipe_data_requirements(recipe: object) -> dict[str, object]:
    """Extract explicit data facts required by the current recipe.

    This intentionally reads only declared Scratch data blocks.  It does not
    infer method compatibility from a paper name or from a dataset alias.
    """

    roles: set[str] = set()
    noise_names: list[str] = []
    labels: dict[str, str] = {}
    split: dict[str, object] = {}
    models: list[Mapping[str, object]] = []
    for step in _recipe_steps(recipe):
        block_id = str(step.get("block", ""))
        params = step.get("params") if isinstance(step.get("params"), Mapping) else {}
        if block_id == "assign_data_roles":
            declared = params.get("roles", [])
            if isinstance(declared, (list, tuple)):
                aliases = {"validation": "clean_validation", "trusted": "trusted_validation", "train-eval": "train_eval"}
                roles.update(aliases.get(str(value).strip().lower(), str(value).strip().lower()) for value in declared)
        elif block_id == "apply_noise":
            noise_names.append(str(params.get("name", "none")).strip().lower())
        elif block_id == "select_label_source":
            for key in ("train", "validation", "test", "trusted"):
                if key in params:
                    labels[key] = str(params[key]).strip().lower()
        elif block_id == "create_dataset_split":
            split = dict(params)
        elif block_id == "create_model":
            models.append(params)
    return {"roles": roles, "noise_names": noise_names, "labels": labels,
            "split": split, "models": models}


def _dataset_capability_preflight(recipe: object, params: Mapping[str, object]) -> dict[str, object]:
    """Reject declared recipe/data mismatches before launching a job."""

    from .data_bridge import dataset_fact_payload, inspect_source_capabilities

    requirements = _recipe_data_requirements(recipe)
    roles = requirements["roles"]
    labels = requirements["labels"]
    alias = str(params.get("dataset") or params.get("name") or "").strip().lower()
    mode = str(params.get("source_mode") or "registered").strip().lower()
    custom_mode = mode in {"custom", "custom_path", "local", "path", "folder"} or bool(
        params.get("path") or params.get("root")
    )
    facts = dataset_fact_payload(alias)

    # A custom source with a supported adapter has no persisted profile yet;
    # use the adapter's declared static facts where available.  The path/layout
    # gate still remains authoritative for the filesystem itself.
    if custom_mode:
        options = params.get("options") if isinstance(params.get("options"), Mapping) else {}
        adapter = str(params.get("adapter") or options.get("adapter") or alias).strip().lower().replace("-", "_")
        static = dataset_fact_payload(adapter)
        if static.get("status") != "unregistered":
            facts = static

    capabilities = {
        key: facts.get(key)
        for key in ("dataset", "adapter", "num_classes", "input_shape",
                    "has_clean_target", "has_test_clean_target", "has_validation_clean_target", "has_noisy_target", "has_sample_index",
                    "has_validation_source",
                    "layout_validated", "training_verified", "noise_methods")
    }
    capabilities["dataset"] = facts.get("alias") or alias
    if facts.get("status") in {"unregistered", "unavailable"} and not custom_mode:
        return {"ok": False, "code": "dataset-unavailable",
                "error": f"数据集“{alias}”当前没有可用的 Scratch 数据源。",
                "capabilities": capabilities}
    if capabilities["has_sample_index"] is False:
        return {"ok": False, "code": "dataset-capability-missing",
                "error": "当前数据源没有稳定 sample index，不能进入 Scratch 运行。",
                "capabilities": capabilities}

    clean_keys = set()
    if "trusted_validation" in roles:
        clean_keys.add("has_clean_target")
    if "clean_validation" in roles:
        clean_keys.add("has_validation_clean_target" if requirements["split"].get("split_strategy") == "official" else "has_clean_target")
    if "test" in roles and labels.get("test") == "clean":
        clean_keys.add("has_test_clean_target")
    # Custom paths and incomplete registration facts require actual source
    # inspection, rather than adapter-name guesses or a nonempty directory.
    source_options = params.get("options") if isinstance(params.get("options"), Mapping) else {}
    if custom_mode or any(key in source_options for key in ("classes", "num_classes", "binary_classes")) or not facts.get("layout_validated") or capabilities["has_sample_index"] is None or any(capabilities.get(key) is None for key in clean_keys):
        try:
            inspected = inspect_source_capabilities(params, base_dir=REPO_ROOT)
        except Exception as exc:
            return {"ok": False, "code": "dataset-inspection-failed",
                    "error": f"数据源检查失败：{exc}。请检查 adapter、路径及 train/test 文件布局。"}
        facts = {**facts, **inspected}
        capabilities.update(inspected)
    for model in requirements["models"]:
        if model.get("num_classes") is not None and capabilities.get("num_classes") is not None:
            if int(model["num_classes"]) != int(capabilities["num_classes"]):
                return {"ok": False, "code": "dataset-model-class-mismatch",
                        "error": f"数据源为 {capabilities['num_classes']} 类，但模型配置为 {model['num_classes']} 类。"}
    if any(capabilities.get(key) is not True for key in clean_keys):
        return {"ok": False, "code": "dataset-clean-target-missing",
                "error": "当前 Recipe 要求 clean target，但所选数据源没有可用的真实 clean label。",
                "capabilities": capabilities}

    supported_noise = {str(value).lower() for value in (capabilities.get("noise_methods") or [])}
    for noise_name in requirements["noise_names"]:
        if noise_name and noise_name not in supported_noise:
            return {"ok": False, "code": "noise-method-unsupported",
                    "error": f"数据源不声明支持噪声方法：{noise_name}。",
                    "capabilities": capabilities}

    split = requirements["split"]
    validation_size = int(split.get("validation_size", 0) or 0) if isinstance(split, Mapping) else 0
    strategy = str(split.get("split_strategy", "random")).lower() if isinstance(split, Mapping) else "random"
    validation_roles = {"clean_validation", "noisy_validation"} & set(roles)
    if validation_roles and validation_size <= 0:
        return {"ok": False, "code": "validation-role-without-split",
                "error": f"Recipe 请求 {', '.join(sorted(validation_roles))}，但 validation_size=0。不能把 test split 冒充 validation。",
                "capabilities": capabilities}
    if "trusted_validation" in roles:
        options = params.get("options") if isinstance(params.get("options"), Mapping) else {}
        trusted_size = int(options.get("num_clean", options.get("trusted_size", 0)) or 0)
        if trusted_size <= 0:
            return {"ok": False, "code": "trusted-size-missing",
                    "error": "trusted_validation 需要正数 num_clean/trusted_size，以及真实 clean labels。"}
        train_count = facts.get("train_samples")
        if train_count is not None and trusted_size >= int(train_count) - validation_size:
            return {"ok": False, "code": "trusted-size-too-large",
                    "error": "trusted subset 太大，划分后必须保留训练样本。"}
    if strategy == "official" and validation_size > 0 and not facts.get("has_validation_source", False):
        return {"ok": False, "code": "official-validation-unavailable",
                "error": "当前数据源没有声明 official validation source，不能执行 official split。",
                "capabilities": capabilities}

    return {"ok": True, "capabilities": capabilities,
            "requirements": {"roles": sorted(roles), "noise": list(requirements["noise_names"]),
                              "validation_size": validation_size}}


def _external_resource_preflight(recipe: object, *, runtime_limits: object = None) -> dict[str, object]:
    """Check external noise artifacts before a Scratch child is spawned."""

    if isinstance(runtime_limits, Mapping) and runtime_limits.get("fixture"):
        return {"ok": True, "skipped": True, "reason": "fixture runtime"}
    for step in _recipe_steps(recipe):
        if str(step.get("block", "")) != "apply_noise":
            continue
        params = step.get("params") if isinstance(step.get("params"), Mapping) else {}
        if str(params.get("name", "none")).strip().lower() not in {"external", "external_torch"}:
            continue
        options = params.get("options") if isinstance(params.get("options"), Mapping) else {}
        effective = dict(params)
        effective.update(options)
        explicit = [effective.get(key) for key in ("path", "artifact_path", "external_path") if effective.get(key)]
        env_name = str(effective.get("source_env") or "").strip()
        env_value = os.environ.get(env_name) if env_name else None
        if len({str(value) for value in explicit}) > 1 or (env_value and explicit and str(env_value) not in {str(value) for value in explicit}):
            return {"ok": False, "code": "external-source-conflict", "error": "外部噪声 source 同时指定了冲突的 path/artifact_path/source_env。"}
        value = explicit[0] if explicit else env_value
        if not value:
            return {"ok": False, "code": "missing-external-source", "error": "当前 Recipe 需要外部噪声 artifact，但没有可用的路径或 source_env。"}
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        path = path.resolve()
        if path.is_dir():
            candidates = [path / "noise_manifest.npz", path / "noise.pt", path / "artifact.pt"]
            path = next((candidate for candidate in candidates if candidate.exists()), path)
        if not path.is_file():
            return {"ok": False, "code": "external-source-not-found", "error": f"外部噪声 artifact 不存在：{path}", "path": str(path)}
        try:
            if path.stat().st_size <= 0:
                return {"ok": False, "code": "external-source-empty", "error": f"外部噪声 artifact 为空：{path}", "path": str(path)}
        except OSError as exc:
            return {"ok": False, "code": "external-source-inaccessible", "error": f"无法访问外部噪声 artifact：{path}（{exc}）", "path": str(path)}
    return {"ok": True, "skipped": True}


def _dataset_preflight(recipe: object, *, runtime_limits: object = None) -> dict[str, object]:
    """Run the server-side filesystem gate for a recipe's data source."""

    params = _load_dataset_params(recipe)
    if params is None:
        return {"ok": True, "skipped": True}
    from .data_bridge import dataset_preflight_payload

    filesystem = dataset_preflight_payload(params, base_dir=REPO_ROOT)
    if not filesystem.get("ok", False):
        return filesystem
    capability = _dataset_capability_preflight(recipe, params)
    if not capability.get("ok", False):
        return capability
    external = _external_resource_preflight(recipe, runtime_limits=runtime_limits)
    if not external.get("ok", False):
        return external
    return {**filesystem, **capability, "external": external}


def _error_payload(exc: Exception) -> dict[str, object]:
    payload: dict[str, object] = {"ok": False, "error": str(exc)}
    payload["code"] = _job_error_code(str(exc))
    if isinstance(exc, ScratchExecutionError):
        payload.update({"path": list(exc.path), "block_id": exc.block_id, "params": exc.params})
    return payload


def _json_safe(value: object) -> object:
    """Convert non-finite numeric metadata to JSON-compatible nulls."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _template_catalog() -> list[dict[str, str]]:
    formula_ready = {"gce", "coteaching"}
    display_names = {"gce": "GCE", "coteaching": "Co-teaching"}
    templates = []
    for path in sorted((RECIPE_ROOT / "papers").glob("*.y*ml")) if (RECIPE_ROOT / "papers").exists() else []:
        template_id = path.stem
        templates.append({
            "id": template_id,
            "name": display_names.get(template_id, template_id.replace("_", " ").title()),
            "path": f"papers/{path.name}",
            "status": "formula-ready" if template_id in formula_ready else "template-ready",
        })
    return templates


def _paper_examples() -> list[dict[str, str]]:
    return _template_catalog()


def _user_recipe_catalog() -> list[str]:
    try:
        root = recipe_workspace_root()
    except OSError:
        return []
    return sorted(str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*.y*ml"))


def _formula_payload(spec) -> dict[str, object]:
    from ..formula.runtime import formula_hash

    payload = spec.to_dict()
    payload.update({"formula_hash": formula_hash(spec), "block_id": "formula/" + spec.id})
    return payload


def _reload_user_formulas() -> None:
    from ..formula.registry import reload_formulas

    # The editor loads blocks and formulas in parallel.  Both endpoints refresh
    # the same process-wide Formula registry, so unregister/register must be one
    # atomic operation or the two requests can observe a duplicate user ID.
    with _FORMULA_RELOAD_LOCK:
        reload_formulas()


class ScratchHandler(BaseHTTPRequestHandler):
    server_version = "LNL-Scratch/1.0"

    def _send(self, payload: object, status: int = 200, content_type: str = "application/json") -> None:
        data = payload if isinstance(payload, bytes) else (json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False).encode("utf-8") if content_type == "application/json" else str(payload).encode("utf-8"))
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/scratch", "/scratch/"}:
                # The standalone service serves the editor at `/`.  Keep the
                # mounted WebUI URL friendly as well instead of returning a
                # confusing 404 when both launch modes are tried.
                self.send_response(302)
                self.send_header("Location", "/")
                self.end_headers()
            elif parsed.path == "/api/blocks":
                _reload_user_formulas()
                self._send([definition.describe() for definition in list_blocks()])
            elif parsed.path == "/api/formulas":
                _reload_user_formulas()
                from ..formula.registry import list_formulas

                self._send([_formula_payload(spec) for spec in list_formulas()])
            elif parsed.path.startswith("/api/formula/") and not parsed.path.endswith("/export"):
                formula_id = unquote(parsed.path.removeprefix("/api/formula/"))
                from ..formula.registry import get_formula

                self._send(_formula_payload(get_formula(formula_id)))
            elif parsed.path.startswith("/api/formula/") and parsed.path.endswith("/export"):
                formula_id = unquote(parsed.path.removeprefix("/api/formula/").removesuffix("/export"))
                from ..formula.storage import export_formula

                self._send(export_formula(formula_id))
            elif parsed.path == "/api/default-recipe":
                self._send(load_recipe(RECIPE_ROOT / "examples" / "default_supervised.yaml"))
            elif parsed.path == "/api/entry-recipe":
                self._send({
                    "schema_version": 1,
                    "name": "空白 Scratch 算法",
                    "description": "选择单模型、双模型、论文模板或从空白开始。",
                    "settings": {},
                    "steps": [],
                })
            elif parsed.path == "/api/templates":
                self._send(_template_catalog())
            elif parsed.path == "/api/examples":
                self._send(_paper_examples())
            elif parsed.path == "/api/datasets":
                from .data_bridge import dataset_catalog_payload

                self._send(dataset_catalog_payload())
            elif parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.removeprefix("/api/jobs/").strip("/")
                if not job_id:
                    self._send({"error": "Scratch job id is required"}, 400)
                else:
                    with SCRATCH_JOBS_LOCK:
                        job = SCRATCH_JOBS.get(job_id)
                    if job is None:
                        self._send({"error": "Scratch job not found"}, 404)
                    else:
                        self._send(_scratch_job_payload(job))
            elif parsed.path.startswith("/api/dataset/"):
                from .data_bridge import dataset_fact_payload

                self._send(dataset_fact_payload(unquote(parsed.path.removeprefix("/api/dataset/"))))
            elif parsed.path == "/api/recipes":
                self._send(_user_recipe_catalog())
            elif parsed.path.startswith("/api/recipe/"):
                recipe = load_recipe(_recipe_path(unquote(parsed.path.removeprefix("/api/recipe/"))))
                self._send(recipe)
            elif parsed.path in {"/", "/index.html"}:
                self._send((ROOT / "index.html").read_bytes(), content_type="text/html")
            elif parsed.path in {"/scratch.js", "/scratch.css"}:
                content_type = "application/javascript" if parsed.path.endswith(".js") else "text/css"
                self._send((ROOT / parsed.path.lstrip("/")).read_bytes(), content_type=content_type)
            else:
                self._send({"error": "not found"}, 404)
        except FileNotFoundError:
            self._send({"error": "recipe not found"}, 404)
        except Exception as exc:
            self._send({"error": str(exc)}, 400)

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._body()
            if self.path == "/api/validate":
                recipe = validate_recipe(body.get("recipe", body))
                preflight = _dataset_preflight(recipe, runtime_limits=body.get("runtime_limits"))
                if not preflight.get("ok", False):
                    self._send(preflight, 400)
                    return
                self._send({"ok": True, "recipe": recipe, "dataset_preflight": preflight})
            elif self.path == "/api/formulas":
                from ..formula.registry import validate_and_register_formula
                from ..formula.storage import save_formula

                raw = body.get("formula", body)
                spec = validate_and_register_formula(raw, replace=bool(body.get("replace", False)))
                path = save_formula(spec, overwrite=True)
                self._send({"ok": True, "formula": _formula_payload(spec), "path": str(path)}, 201)
            elif self.path == "/api/formula/delete":
                from ..formula.registry import unregister_formula
                from ..formula.storage import delete_formula, find_formula_references

                formula_id = str(body.get("id", ""))
                references = find_formula_references(RECIPE_ROOT, formula_id)
                references.extend(find_formula_references(recipe_workspace_root(), formula_id))
                spec = delete_formula(formula_id, referenced_by=tuple(references))
                unregister_formula(spec.id)
                self._send({"ok": True, "id": spec.id})
            elif self.path == "/api/formula/import":
                from ..formula.registry import validate_and_register_formula
                from ..formula.storage import import_formula

                raw = body.get("formula", body.get("yaml", ""))
                spec = import_formula(raw, overwrite=bool(body.get("replace", False)))
                validate_and_register_formula(spec, replace=True)
                self._send({"ok": True, "formula": _formula_payload(spec)}, 201)
            elif self.path == "/api/formula/rename":
                from ..formula.registry import register_formula, unregister_formula
                from ..formula.storage import rename_formula

                old_id, new_id = str(body.get("old_id", "")), str(body.get("new_id", ""))
                spec = rename_formula(old_id, new_id)
                try:
                    unregister_formula(old_id)
                except KeyError:
                    pass
                register_formula(spec, replace=True)
                self._send({"ok": True, "formula": _formula_payload(spec)}, 201)
            elif self.path == "/api/save":
                recipe = validate_recipe(body["recipe"])
                name = Path(str(recipe["name"])).name
                destination = recipe_workspace_root() / f"{name}.yaml"
                save_recipe(recipe, destination)
                self._send({"ok": True, "path": str(destination)})
            elif self.path == "/api/run":
                recipe = validate_recipe(body["recipe"])
                preflight = _dataset_preflight(recipe, runtime_limits=body.get("runtime_limits"))
                if not preflight.get("ok", False):
                    self._send(preflight, 400)
                    return
                limits = body.get("runtime_limits", {})
                job = start_scratch_job(recipe, limits)
                self._send(_scratch_job_payload(job), 202)
            elif self.path.startswith("/api/jobs/") and self.path.endswith("/cancel"):
                job_id = self.path.removeprefix("/api/jobs/").removesuffix("/cancel").strip("/")
                try:
                    self._send(_scratch_job_payload(cancel_scratch_job(job_id)), 202)
                except KeyError:
                    self._send({"error": "Scratch job not found"}, 404)
            else:
                self._send({"error": "not found"}, 404)
        except Exception as exc:
            self._send(_error_payload(exc), 400)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LNL Scratch Web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), ScratchHandler)
    print(f"LNL Scratch running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
