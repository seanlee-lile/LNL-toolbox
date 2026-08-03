from __future__ import annotations

"""Hash-verified frozen curriculum-model artifacts."""

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from typing import Any, Mapping

import torch


def _artifact_digest(
    architecture: Mapping[str, Any],
    feature_schema: Mapping[str, Any],
    source: Mapping[str, Any],
    model_state: Mapping[str, Any],
) -> str:
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
        if payload.get("artifact_hash") != artifact.artifact_hash:
            raise ValueError("MentorArtifact hash mismatch")
        return artifact
