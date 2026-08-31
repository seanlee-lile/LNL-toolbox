"""Minimal standard-library Web server for the Scratch recipe editor."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from .. import ScratchExecutionError, execute_recipe, list_blocks, load_recipe, resolve_recipe, save_recipe, validate_recipe


ROOT = Path(__file__).resolve().parent
RECIPE_ROOT = ROOT.parent / "recipes"


def _recipe_path(name: str) -> Path:
    requested = Path(name)
    candidates = [RECIPE_ROOT / requested, RECIPE_ROOT / "examples" / requested.name, RECIPE_ROOT / "papers" / requested.name]
    root = RECIPE_ROOT.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved != root and root not in resolved.parents:
            continue
        if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
            return candidate
    raise FileNotFoundError(name)


def _error_payload(exc: Exception) -> dict[str, object]:
    payload: dict[str, object] = {"ok": False, "error": str(exc)}
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


def _formula_payload(spec) -> dict[str, object]:
    from ..formula.runtime import formula_hash

    payload = spec.to_dict()
    payload.update({"formula_hash": formula_hash(spec), "block_id": "formula/" + spec.id})
    return payload


def _reload_user_formulas() -> None:
    from ..formula.registry import reload_formulas

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
                self._send(load_recipe(RECIPE_ROOT / "papers" / "gce.yaml"))
            elif parsed.path == "/api/templates":
                self._send(_template_catalog())
            elif parsed.path == "/api/examples":
                self._send(_paper_examples())
            elif parsed.path == "/api/datasets":
                from .data_bridge import dataset_catalog_payload

                self._send(dataset_catalog_payload())
            elif parsed.path.startswith("/api/dataset/"):
                from .data_bridge import dataset_fact_payload

                self._send(dataset_fact_payload(unquote(parsed.path.removeprefix("/api/dataset/"))))
            elif parsed.path == "/api/recipes":
                names = sorted(str(path.relative_to(RECIPE_ROOT)) for path in RECIPE_ROOT.rglob("*.y*ml")) if RECIPE_ROOT.exists() else []
                self._send(names)
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
                self._send({"ok": True, "recipe": validate_recipe(body.get("recipe", body))})
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
                destination = RECIPE_ROOT / "examples" / f"{name}.yaml"
                save_recipe(recipe, destination)
                self._send({"ok": True, "path": str(destination.relative_to(RECIPE_ROOT))})
            elif self.path == "/api/run":
                recipe = validate_recipe(body["recipe"])
                output = Path("artifacts") / "scratch" / Path(str(recipe["name"])).name
                output.mkdir(parents=True, exist_ok=True)
                save_recipe(recipe, output / "recipe.yaml")
                save_recipe(resolve_recipe(recipe), output / "resolved_recipe.yaml")
                from ..formula.registry import collect_formula_provenance

                provenance = collect_formula_provenance(recipe)
                (output / "formula_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
                limits = body.get("runtime_limits", {})
                context = execute_recipe(recipe, {"artifact_dir": str(output)}, runtime_limits=limits)
                metrics = context.get("metrics", [])
                (output / "stdout.log").write_text(json.dumps({"name": recipe["name"], "metrics": metrics}, ensure_ascii=False) + "\n", encoding="utf-8")
                self._send({"ok": True, "metrics": metrics, "artifact_dir": str(output), "runtime_limits": limits, "formula_provenance": provenance})
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
