"""Minimal standard-library Web server for the Scratch recipe editor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from .. import execute_recipe, list_blocks, load_recipe, resolve_recipe, save_recipe, validate_recipe


ROOT = Path(__file__).resolve().parent
RECIPE_ROOT = ROOT.parent / "recipes"


def _recipe_path(name: str) -> Path:
    safe = Path(name).name
    candidates = [RECIPE_ROOT / safe, RECIPE_ROOT / "examples" / safe, RECIPE_ROOT / "papers" / safe]
    for candidate in candidates:
        if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
            return candidate
    raise FileNotFoundError(name)


class ScratchHandler(BaseHTTPRequestHandler):
    server_version = "LNL-Scratch/1.0"

    def _send(self, payload: object, status: int = 200, content_type: str = "application/json") -> None:
        data = payload if isinstance(payload, bytes) else (json.dumps(payload, ensure_ascii=False).encode("utf-8") if content_type == "application/json" else str(payload).encode("utf-8"))
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
            if parsed.path == "/api/blocks":
                self._send([definition.describe() for definition in list_blocks()])
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
                context = execute_recipe(recipe, {"artifact_dir": str(output)})
                metrics = context.get("metrics", [])
                (output / "stdout.log").write_text(json.dumps({"name": recipe["name"], "metrics": metrics}, ensure_ascii=False) + "\n", encoding="utf-8")
                self._send({"ok": True, "metrics": metrics, "artifact_dir": str(output)})
            else:
                self._send({"error": "not found"}, 404)
        except Exception as exc:
            self._send({"ok": False, "error": str(exc)}, 400)


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
