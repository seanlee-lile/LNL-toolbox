from __future__ import annotations

"""Strict, method-local adapters for immutable PCSE backbone sources."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.training.checkpoint import read_checkpoint
from lnl_toolbox.training.noisy_labels import file_sha256


@dataclass(frozen=True)
class PCSESourceFileIdentity:
    path: str
    sha256: str
    size: int
    mtime_ns: int

    @classmethod
    def capture(cls, path: str | Path) -> "PCSESourceFileIdentity":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"PCSE source file does not exist: {source}")
        stat = source.stat()
        return cls(
            path=str(source),
            sha256=file_sha256(source),
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }

    def assert_unchanged(self) -> None:
        if PCSESourceFileIdentity.capture(self.path) != self:
            raise RuntimeError(f"PCSE immutable source changed: {self.path}")


@dataclass(frozen=True)
class PCSEUPMMainBestSource:
    run_dir: Path
    checkpoint: PCSESourceFileIdentity
    manifest: PCSESourceFileIdentity
    checkpoint_payload: Mapping[str, Any]
    noise_manifest: NoiseManifest
    state_dict: Mapping[str, Any]
    source_completed_epochs: int
    source_global_step: int
    best_epoch: int
    best_validation_accuracy: float

    @property
    def provenance(self) -> dict[str, Any]:
        noise = self.checkpoint_payload["noise"]
        config = self.checkpoint_payload["config"]
        return {
            "adapter": "upm_main_best",
            "source_method": "upm",
            "source_checkpoint_role": "main_best",
            "source_run_dir": str(self.run_dir),
            "checkpoint": self.checkpoint.state_dict(),
            "manifest": self.manifest.state_dict(),
            "mapping_hash": str(noise["mapping_hash"]),
            "dataset_fingerprint": str(noise["dataset_fingerprint"]),
            "dataset": str(noise["dataset"]),
            "num_classes": int(noise["num_classes"]),
            "model": dict(config["upm"]["main"]["model"]),
        }

    def assert_unchanged(self) -> None:
        self.checkpoint.assert_unchanged()
        self.manifest.assert_unchanged()


def _expected_digest(value: Any, *, owner: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{owner} must be a 64-character SHA-256 digest")
    return result


def load_upm_main_best_source(
    source_config: Mapping[str, Any],
    model: nn.Module,
    *,
    num_classes: int,
) -> PCSEUPMMainBestSource:
    """Load the one supported source schema without state-dict guessing."""

    if str(source_config.get("adapter", "")).strip().lower() != "upm_main_best":
        raise ValueError("PCSE external source adapter must be upm_main_best")
    environment_name = str(source_config.get("run_directory_env", "")).strip()
    if not environment_name:
        raise ValueError("PCSE source run_directory_env must not be empty")
    raw_run_dir = os.environ.get(environment_name, "").strip()
    if not raw_run_dir:
        raise ValueError(
            f"PCSE source environment variable is not set: {environment_name}"
        )
    run_dir = Path(raw_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"PCSE source run does not exist: {run_dir}")
    checkpoint = PCSESourceFileIdentity.capture(run_dir / "best.pt")
    manifest_identity = PCSESourceFileIdentity.capture(
        run_dir / "noise_manifest.npz"
    )
    expected_checkpoint = _expected_digest(
        source_config.get("checkpoint_sha256"), owner="source checkpoint_sha256"
    )
    expected_manifest = _expected_digest(
        source_config.get("manifest_sha256"), owner="source manifest_sha256"
    )
    if checkpoint.sha256 != expected_checkpoint:
        raise ValueError("PCSE source checkpoint SHA-256 mismatch")
    if manifest_identity.sha256 != expected_manifest:
        raise ValueError("PCSE source manifest SHA-256 mismatch")

    payload = read_checkpoint(checkpoint.path, "cpu")
    if payload.get("method") != "upm" or payload.get("checkpoint_role") != "main_best":
        raise ValueError("PCSE source must be an UPM main_best checkpoint")
    config = payload.get("config")
    noise = payload.get("noise")
    upm_state = payload.get("upm_state")
    state_dict = payload.get("best_main_model_state")
    if not all(isinstance(item, Mapping) for item in (config, noise, upm_state, state_dict)):
        raise ValueError("PCSE UPM source checkpoint schema is incomplete")
    expected_model = dict(source_config.get("model", {}))
    actual_model = dict(config.get("upm", {}).get("main", {}).get("model", {}))
    if actual_model != expected_model:
        raise ValueError("PCSE source model architecture mismatch")
    if str(noise.get("dataset")) != "cifar10":
        raise ValueError("PCSE source dataset must be cifar10")
    if int(noise.get("num_classes", -1)) != int(num_classes) or num_classes != 10:
        raise ValueError("PCSE source num_classes mismatch")
    expected_mapping = _expected_digest(
        source_config.get("mapping_hash"), owner="source mapping_hash"
    )
    expected_fingerprint = _expected_digest(
        source_config.get("dataset_fingerprint"),
        owner="source dataset_fingerprint",
    )
    if str(noise.get("mapping_hash", "")).lower() != expected_mapping:
        raise ValueError("PCSE source noise mapping hash mismatch")
    if str(noise.get("dataset_fingerprint", "")).lower() != expected_fingerprint:
        raise ValueError("PCSE source dataset fingerprint mismatch")
    if str(noise.get("manifest_sha256", "")).lower() != manifest_identity.sha256:
        raise ValueError("PCSE checkpoint manifest identity mismatch")

    manifest = NoiseManifest.load(manifest_identity.path)
    if manifest.mapping_hash != expected_mapping:
        raise ValueError("PCSE source manifest mapping hash mismatch")
    if manifest.dataset_fingerprint != expected_fingerprint:
        raise ValueError("PCSE source manifest dataset fingerprint mismatch")
    if manifest.dataset != "cifar10" or manifest.num_classes != 10:
        raise ValueError("PCSE source manifest dataset contract mismatch")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("PCSE source state_dict is incompatible") from error

    result = PCSEUPMMainBestSource(
        run_dir=run_dir,
        checkpoint=checkpoint,
        manifest=manifest_identity,
        checkpoint_payload=payload,
        noise_manifest=manifest,
        state_dict=state_dict,
        source_completed_epochs=int(upm_state["main_completed_epochs"]),
        source_global_step=int(upm_state["main_global_step"]),
        best_epoch=int(upm_state["main_best_epoch"]),
        best_validation_accuracy=float(upm_state["main_best_validation_accuracy"]),
    )
    result.assert_unchanged()
    return result


__all__ = [
    "PCSESourceFileIdentity",
    "PCSEUPMMainBestSource",
    "load_upm_main_best_source",
]
