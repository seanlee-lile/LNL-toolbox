from __future__ import annotations

"""Hash-verified frozen curriculum-model artifacts."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


def _artifact_digest(
    architecture: Mapping[str, Any],
    feature_schema: Mapping[str, Any],
    source: Mapping[str, Any],
    model_state: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256(b"mentor-artifact-v2\0")
    metadata = {
        "architecture": dict(architecture),
        "feature_schema": dict(feature_schema),
        "source": dict(source),
    }
    encoded_metadata = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest.update(len(encoded_metadata).to_bytes(8, "big"))
    digest.update(encoded_metadata)
    for name, value in sorted(model_state.items(), key=lambda item: str(item[0])):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Mentor model state entry {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        descriptor = json.dumps(
            {"name": str(name), "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _legacy_artifact_digest(
    architecture: Mapping[str, Any],
    feature_schema: Mapping[str, Any],
    source: Mapping[str, Any],
    model_state: Mapping[str, Any],
) -> str:
    """Read artifacts written before the portable content hash was introduced."""
    import io

    buffer = io.BytesIO()
    torch.save(
        {
            "architecture": dict(architecture),
            "feature_schema": dict(feature_schema),
            "source": dict(source),
            "model_state": dict(model_state),
        },
        buffer,
    )
    return hashlib.sha256(buffer.getvalue()).hexdigest()


@dataclass(frozen=True)
class MentorArtifact:
    architecture: Mapping[str, Any]
    feature_schema: Mapping[str, Any]
    source: Mapping[str, Any]
    model_state: Mapping[str, Any]
    artifact_hash: str

    @classmethod
    def create(
        cls,
        *,
        architecture: Mapping[str, Any],
        feature_schema: Mapping[str, Any],
        source: Mapping[str, Any],
        model_state: Mapping[str, Any],
    ) -> "MentorArtifact":
        digest = _artifact_digest(
            architecture, feature_schema, source, model_state
        )
        return cls(
            dict(architecture),
            dict(feature_schema),
            dict(source),
            dict(model_state),
            digest,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": dict(self.architecture),
                "feature_schema": dict(self.feature_schema),
                "source": dict(self.source),
                "model_state": dict(self.model_state),
                "artifact_hash": self.artifact_hash,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MentorArtifact":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        artifact = cls.create(
            architecture=payload["architecture"],
            feature_schema=payload["feature_schema"],
            source=payload["source"],
            model_state=payload["model_state"],
        )
        stored_hash = payload.get("artifact_hash")
        if stored_hash != artifact.artifact_hash and stored_hash != _legacy_artifact_digest(
            artifact.architecture, artifact.feature_schema, artifact.source, artifact.model_state
        ):
            raise ValueError("MentorArtifact hash mismatch")
        return artifact
