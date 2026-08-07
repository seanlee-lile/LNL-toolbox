from __future__ import annotations

"""Generic versioned artifact references used by staged training pipelines."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    path: str
    artifact_hash: str
    metadata: Mapping[str, Any]


class ArtifactStore:
    """Persist artifacts through their own ``save``/``load`` contracts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, artifact: Any, *, filename: str | None = None) -> ArtifactRef:
        if not hasattr(artifact, "save"):
            raise TypeError("artifact must expose save()")
        artifact_hash = getattr(artifact, "artifact_hash", getattr(artifact, "snapshot_hash", None))
        if artifact_hash is None:
            raise TypeError("artifact must expose artifact_hash or snapshot_hash")
        safe_name = str(name).strip()
        if not safe_name:
            raise ValueError("artifact name must not be empty")
        path = self.root / (filename or f"{safe_name}.npz")
        artifact.save(path)
        metadata = {"name": safe_name, "artifact_hash": str(artifact_hash)}
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(metadata, sort_keys=True), encoding="utf-8"
        )
        return ArtifactRef(safe_name, str(path), str(artifact_hash), metadata)

    def read_metadata(self, name: str, *, filename: str | None = None) -> Mapping[str, Any]:
        path = self.root / (filename or f"{name}.npz")
        metadata_path = path.with_suffix(path.suffix + ".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("artifact metadata must be a mapping")
        return value
