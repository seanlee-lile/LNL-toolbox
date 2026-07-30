from __future__ import annotations

"""Versioned raw-min noise-rate artifact bound to one posterior snapshot."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class NoiseRateArtifact:
    rho_positive: float
    rho_negative: float
    positive_extreme_global_index: int
    negative_extreme_global_index: int
    source_snapshot_hash: str
    dataset: str
    split: str
    estimator: str = "paper_raw_min"
    method: str = "importance_reweighting"
    num_classes: int = 2
    label_convention: str = "zero_one"
    version: str = "1.0"

    def __post_init__(self) -> None:
        for name, value in (
            ("rho_positive", self.rho_positive),
            ("rho_negative", self.rho_negative),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1)")
        if self.rho_positive + self.rho_negative >= 1.0:
            raise ValueError("estimated noise rates must sum to less than 1")
        if self.positive_extreme_global_index < 0:
            raise ValueError("positive extreme global index must be non-negative")
        if self.negative_extreme_global_index < 0:
            raise ValueError("negative extreme global index must be non-negative")
        if len(self.source_snapshot_hash) != 64:
            raise ValueError("rate artifact requires a 64-character snapshot hash")
        if self.method != "importance_reweighting":
            raise ValueError("rate artifact method identity mismatch")
        if self.estimator != "paper_raw_min":
            raise ValueError("rate artifact estimator must be paper_raw_min")
        if self.num_classes != 2:
            raise ValueError("rate artifact num_classes must be 2")
        if self.label_convention != "zero_one":
            raise ValueError("rate artifact label_convention must be zero_one")
        if not self.dataset.strip() or not self.split.strip():
            raise ValueError("rate artifact dataset and split must not be empty")

    def _payload(self) -> dict[str, Any]:
        return {
            "rho_positive": float(self.rho_positive),
            "rho_negative": float(self.rho_negative),
            "positive_extreme_global_index": int(
                self.positive_extreme_global_index
            ),
            "negative_extreme_global_index": int(
                self.negative_extreme_global_index
            ),
            "source_snapshot_hash": self.source_snapshot_hash,
            "dataset": self.dataset,
            "split": self.split,
            "estimator": self.estimator,
            "method": self.method,
            "num_classes": self.num_classes,
            "label_convention": self.label_convention,
            "version": self.version,
        }

    @property
    def artifact_hash(self) -> str:
        payload = json.dumps(
            self._payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self._payload(), "artifact_hash": self.artifact_hash}
        np.savez_compressed(
            destination,
            metadata_json=np.array(json.dumps(payload, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NoiseRateArtifact":
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"metadata_json"}:
                raise ValueError("noise-rate artifact contains unexpected fields")
            try:
                payload = json.loads(str(data["metadata_json"].item()))
            except (AttributeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError("noise-rate artifact metadata is invalid") from exc
        if not isinstance(payload, dict):
            raise ValueError("noise-rate artifact metadata must be a mapping")
        artifact = cls(
            rho_positive=float(payload["rho_positive"]),
            rho_negative=float(payload["rho_negative"]),
            positive_extreme_global_index=int(
                payload["positive_extreme_global_index"]
            ),
            negative_extreme_global_index=int(
                payload["negative_extreme_global_index"]
            ),
            source_snapshot_hash=str(payload["source_snapshot_hash"]),
            dataset=str(payload["dataset"]),
            split=str(payload["split"]),
            estimator=str(payload.get("estimator", "")),
            method=str(payload.get("method", "")),
            num_classes=int(payload.get("num_classes", 0)),
            label_convention=str(payload.get("label_convention", "")),
            version=str(payload.get("version", "")),
        )
        if payload.get("artifact_hash") != artifact.artifact_hash:
            raise ValueError("noise-rate artifact hash does not match its contents")
        return artifact
