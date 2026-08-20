from __future__ import annotations

"""Machine-local dataset locations and verification evidence.

The catalog stores paths, never dataset contents.  Registration, layout
validation, and successful training are deliberately separate states so the
toolbox never describes an untrained local source as training verified.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


CATALOG_VERSION = 1
_SOURCE_KEYS = {"root", "path", "noise_path", "labels_path", "annotation_root"}


def default_catalog_path() -> Path:
    configured = os.environ.get("LNL_DATA_CATALOG")
    if configured:
        return Path(configured).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / ".lnl-toolbox"
    return (base / "lnl-toolbox" / "datasets.json").resolve() if local else base / "datasets.json"


def normalize_alias(value: object) -> str:
    alias = str(value).strip().lower().replace(" ", "-")
    if not alias or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in alias):
        raise ValueError("dataset alias must contain only letters, digits, '-' and '_'")
    return alias


def _path_state(value: object) -> dict[str, Any]:
    path = Path(str(value)).expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    state: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "directory": path.is_dir(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if path.is_dir():
        children = []
        for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
            child_stat = child.stat()
            children.append((child.name, child.is_dir(), int(child_stat.st_size), int(child_stat.st_mtime_ns)))
        state["children"] = children
    return state


def source_signature(data: Mapping[str, Any]) -> str:
    payload = {
        key: _path_state(value)
        for key, value in sorted(data.items())
        if key in _SOURCE_KEYS and value not in {None, ""}
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@dataclass(frozen=True, slots=True)
class LocalDatasetRecord:
    alias: str
    adapter: str
    data: Mapping[str, Any]
    state: str = "registered"
    evidence: Mapping[str, Any] | None = None
    error: str | None = None

    @property
    def signature(self) -> str:
        return source_signature(self.data)

    @property
    def effective_state(self) -> str:
        if self.state == "training_verified" and self.evidence:
            if self.evidence.get("source_signature") != self.signature:
                return "verification_stale"
        return self.state

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "adapter": self.adapter,
            "data": deepcopy(dict(self.data)),
            "state": self.effective_state,
            "evidence": None if self.evidence is None else deepcopy(dict(self.evidence)),
            "error": self.error,
        }


class LocalDatasetCatalog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser().resolve() if path is not None else default_catalog_path()

    def _load_raw(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"version": CATALOG_VERSION, "datasets": {}}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("version") != CATALOG_VERSION:
            raise ValueError(f"unsupported local dataset catalog: {self.path}")
        if not isinstance(value.get("datasets"), dict):
            raise ValueError(f"local dataset catalog has invalid datasets mapping: {self.path}")
        return value

    def _write_raw(self, value: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    @staticmethod
    def _record(alias: str, value: Mapping[str, Any]) -> LocalDatasetRecord:
        return LocalDatasetRecord(
            alias=alias,
            adapter=str(value["adapter"]),
            data=dict(value["data"]),
            state=str(value.get("state", "registered")),
            evidence=value.get("evidence"),
            error=value.get("error"),
        )

    def records(self) -> tuple[LocalDatasetRecord, ...]:
        raw = self._load_raw()["datasets"]
        return tuple(self._record(alias, raw[alias]) for alias in sorted(raw))

    def get(self, alias: object) -> LocalDatasetRecord:
        key = normalize_alias(alias)
        raw = self._load_raw()["datasets"]
        if key not in raw:
            raise KeyError(f"local dataset is not registered: {key}")
        return self._record(key, raw[key])

    def register(self, alias: object, adapter: object, data: Mapping[str, Any]) -> LocalDatasetRecord:
        key = normalize_alias(alias)
        adapter_name = str(adapter).strip().lower().replace("-", "_")
        payload = deepcopy(dict(data))
        payload["name"] = adapter_name
        for path_key in _SOURCE_KEYS:
            if payload.get(path_key) not in {None, ""}:
                payload[path_key] = str(Path(str(payload[path_key])).expanduser().resolve())
        raw = self._load_raw()
        raw["datasets"][key] = {
            "adapter": adapter_name,
            "data": payload,
            "state": "registered",
            "evidence": None,
            "error": None,
        }
        self._write_raw(raw)
        return self.get(key)

    def remove(self, alias: object) -> None:
        key = normalize_alias(alias)
        raw = self._load_raw()
        if key not in raw["datasets"]:
            raise KeyError(f"local dataset is not registered: {key}")
        del raw["datasets"][key]
        self._write_raw(raw)

    def _set_state(
        self,
        alias: object,
        state: str,
        *,
        evidence: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> LocalDatasetRecord:
        key = normalize_alias(alias)
        raw = self._load_raw()
        if key not in raw["datasets"]:
            raise KeyError(f"local dataset is not registered: {key}")
        raw["datasets"][key]["state"] = state
        raw["datasets"][key]["evidence"] = None if evidence is None else dict(evidence)
        raw["datasets"][key]["error"] = error
        self._write_raw(raw)
        return self.get(key)

    def mark_layout_validated(self, alias: object) -> LocalDatasetRecord:
        return self._set_state(
            alias,
            "layout_validated",
            evidence={"source_signature": self.get(alias).signature},
        )

    def mark_training_verified(self, alias: object, evidence: Mapping[str, Any]) -> LocalDatasetRecord:
        value = dict(evidence)
        value["source_signature"] = self.get(alias).signature
        return self._set_state(alias, "training_verified", evidence=value)

    def mark_failed(self, alias: object, error: object) -> LocalDatasetRecord:
        return self._set_state(alias, "failed", error=str(error))

    def apply(self, config: Mapping[str, Any], alias: object) -> dict[str, Any]:
        record = self.get(alias)
        result = deepcopy(dict(config))
        current = dict(result.get("data", {}) or {})
        for key in _SOURCE_KEYS:
            current.pop(key, None)
        result["data"] = _deep_merge(current, record.data)
        result.setdefault("local_dataset", {})
        result["local_dataset"] = {
            "alias": record.alias,
            "adapter": record.adapter,
            "source_signature": record.signature,
        }
        return result


__all__ = [
    "LocalDatasetCatalog",
    "LocalDatasetRecord",
    "default_catalog_path",
    "normalize_alias",
    "source_signature",
]
