from __future__ import annotations

"""Strict, read-only pretrained feature source for real-data DLD runs."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

from torch import nn

from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.training.checkpoint import read_checkpoint
from lnl_toolbox.training.noisy_labels import file_sha256


def _digest(value: Any, owner: str) -> str:
    parsed = str(value).strip().lower()
    if len(parsed) != 64 or any(char not in "0123456789abcdef" for char in parsed):
        raise ValueError(f"{owner} must be a 64-character SHA-256 digest")
    return parsed


@dataclass(frozen=True)
class DLDSourceFileIdentity:
    path: str
    sha256: str
    size: int
    mtime_ns: int

    @classmethod
    def capture(cls, path: str | Path) -> "DLDSourceFileIdentity":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"DLD pretrained source file does not exist: {source}")
        stat = source.stat()
        return cls(
            str(source), file_sha256(source), int(stat.st_size), int(stat.st_mtime_ns)
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }

    def assert_unchanged(self) -> None:
        if DLDSourceFileIdentity.capture(self.path) != self:
            raise RuntimeError(f"DLD pretrained source changed: {self.path}")


@dataclass(frozen=True)
class DLDUPMMainBestSource:
    run_dir: Path
    checkpoint: DLDSourceFileIdentity
    manifest: DLDSourceFileIdentity
    checkpoint_payload: Mapping[str, Any]
    noise_manifest: NoiseManifest

    @property
    def provenance(self) -> dict[str, Any]:
        noise = self.checkpoint_payload["noise"]
        config = self.checkpoint_payload["config"]
        state = self.checkpoint_payload["upm_state"]
        return {
            "adapter": "upm_main_best",
            "method": "upm",
            "checkpoint_role": "main_best",
            "run_dir": str(self.run_dir),
            "checkpoint": self.checkpoint.state_dict(),
            "manifest": self.manifest.state_dict(),
            "mapping_hash": str(noise["mapping_hash"]),
            "dataset_fingerprint": str(noise["dataset_fingerprint"]),
            "model": dict(config["upm"]["main"]["model"]),
            "completed_epochs": int(state["main_completed_epochs"]),
            "global_step": int(state["main_global_step"]),
            "best_epoch": int(state["main_best_epoch"]),
            "best_validation_accuracy": float(state["main_best_validation_accuracy"]),
        }

    def assert_unchanged(self) -> None:
        self.checkpoint.assert_unchanged()
        self.manifest.assert_unchanged()


def load_upm_main_best_feature_source(
    source_config: Mapping[str, Any],
    model: nn.Module,
    *,
    num_classes: int,
) -> DLDUPMMainBestSource:
    """Load the one explicitly supported source schema without guessing."""

    if str(source_config.get("adapter", "")).strip().lower() != "upm_main_best":
        raise ValueError("DLD external feature adapter must be upm_main_best")
    environment_name = str(source_config.get("run_directory_env", "")).strip()
    run_value = os.environ.get(environment_name, "").strip()
    if not environment_name or not run_value:
        raise ValueError(f"DLD source environment variable is not set: {environment_name}")
    run_dir = Path(run_value).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"DLD pretrained source run does not exist: {run_dir}")
    checkpoint = DLDSourceFileIdentity.capture(run_dir / "best.pt")
    manifest_identity = DLDSourceFileIdentity.capture(run_dir / "noise_manifest.npz")
    if checkpoint.sha256 != _digest(source_config.get("checkpoint_sha256"), "checkpoint_sha256"):
        raise ValueError("DLD source checkpoint SHA-256 mismatch")
    if manifest_identity.sha256 != _digest(source_config.get("manifest_sha256"), "manifest_sha256"):
        raise ValueError("DLD source manifest SHA-256 mismatch")

    payload = read_checkpoint(checkpoint.path, "cpu")
    if payload.get("method") != "upm" or payload.get("checkpoint_role") != "main_best":
        raise ValueError("DLD source must be an UPM main_best checkpoint")
    config = payload.get("config")
    noise = payload.get("noise")
    state = payload.get("upm_state")
    model_state = payload.get("best_main_model_state")
    if not all(isinstance(item, Mapping) for item in (config, noise, state, model_state)):
        raise ValueError("DLD UPM source checkpoint schema is incomplete")
    expected_model = dict(source_config.get("model", {}))
    actual_model = dict(config.get("upm", {}).get("main", {}).get("model", {}))
    if expected_model != actual_model:
        raise ValueError("DLD source model architecture mismatch")
    if str(noise.get("dataset")) != "cifar10" or int(noise.get("num_classes", -1)) != num_classes:
        raise ValueError("DLD source dataset or class count mismatch")
    expected_mapping = _digest(source_config.get("mapping_hash"), "mapping_hash")
    expected_fingerprint = _digest(
        source_config.get("dataset_fingerprint"), "dataset_fingerprint"
    )
    if str(noise.get("mapping_hash", "")).lower() != expected_mapping:
        raise ValueError("DLD source mapping hash mismatch")
    if str(noise.get("dataset_fingerprint", "")).lower() != expected_fingerprint:
        raise ValueError("DLD source dataset fingerprint mismatch")
    if str(noise.get("manifest_sha256", "")).lower() != manifest_identity.sha256:
        raise ValueError("DLD source checkpoint manifest identity mismatch")
    manifest = NoiseManifest.load(manifest_identity.path)
    if manifest.mapping_hash != expected_mapping or manifest.dataset_fingerprint != expected_fingerprint:
        raise ValueError("DLD source manifest provenance mismatch")
    if manifest.dataset != "cifar10" or manifest.num_classes != num_classes:
        raise ValueError("DLD source manifest dataset contract mismatch")
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise ValueError("DLD source state_dict is incompatible") from error
    result = DLDUPMMainBestSource(
        run_dir, checkpoint, manifest_identity, payload, manifest
    )
    result.assert_unchanged()
    return result


__all__ = [
    "DLDSourceFileIdentity",
    "DLDUPMMainBestSource",
    "load_upm_main_best_feature_source",
]
