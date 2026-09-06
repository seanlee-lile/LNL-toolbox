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
class PCSEPretrainedClassifierSource:
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
    adapter: str
    source_method: str
    checkpoint_role: str
    model_config: Mapping[str, Any]

    @property
    def provenance(self) -> dict[str, Any]:
        noise = self.checkpoint_payload["noise"]
        return {
            "adapter": self.adapter,
            "source_method": self.source_method,
            "source_checkpoint_role": self.checkpoint_role,
            "source_run_dir": str(self.run_dir),
            "checkpoint": self.checkpoint.state_dict(),
            "manifest": self.manifest.state_dict(),
            "mapping_hash": str(noise["mapping_hash"]),
            "dataset_fingerprint": str(noise["dataset_fingerprint"]),
            "dataset": str(noise["dataset"]),
            "num_classes": int(noise["num_classes"]),
            "model": dict(self.model_config),
        }

    def assert_unchanged(self) -> None:
        self.checkpoint.assert_unchanged()
        self.manifest.assert_unchanged()


def _expected_digest(value: Any, *, owner: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{owner} must be a 64-character SHA-256 digest")
    return result


def _source_files(
    source_config: Mapping[str, Any],
) -> tuple[Path, PCSESourceFileIdentity, PCSESourceFileIdentity]:
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
    manifest = PCSESourceFileIdentity.capture(run_dir / "noise_manifest.npz")
    if checkpoint.sha256 != _expected_digest(
        source_config.get("checkpoint_sha256"), owner="source checkpoint_sha256"
    ):
        raise ValueError("PCSE source checkpoint SHA-256 mismatch")
    if manifest.sha256 != _expected_digest(
        source_config.get("manifest_sha256"), owner="source manifest_sha256"
    ):
        raise ValueError("PCSE source manifest SHA-256 mismatch")
    return run_dir, checkpoint, manifest


def _validated_manifest(
    source_config: Mapping[str, Any],
    payload: Mapping[str, Any],
    identity: PCSESourceFileIdentity,
    *,
    num_classes: int,
) -> NoiseManifest:
    noise = payload.get("noise")
    if not isinstance(noise, Mapping):
        raise ValueError("PCSE source checkpoint noise schema is incomplete")
    expected_mapping = _expected_digest(
        source_config.get("mapping_hash"), owner="source mapping_hash"
    )
    expected_fingerprint = _expected_digest(
        source_config.get("dataset_fingerprint"),
        owner="source dataset_fingerprint",
    )
    if str(noise.get("dataset")) != "cifar10":
        raise ValueError("PCSE source dataset must be cifar10")
    if int(noise.get("num_classes", -1)) != num_classes or num_classes != 10:
        raise ValueError("PCSE source num_classes mismatch")
    if str(noise.get("mapping_hash", "")).lower() != expected_mapping:
        raise ValueError("PCSE source noise mapping hash mismatch")
    if str(noise.get("dataset_fingerprint", "")).lower() != expected_fingerprint:
        raise ValueError("PCSE source dataset fingerprint mismatch")
    if str(noise.get("manifest_sha256", "")).lower() != identity.sha256:
        raise ValueError("PCSE checkpoint manifest identity mismatch")
    manifest = NoiseManifest.load(identity.path)
    if manifest.mapping_hash != expected_mapping:
        raise ValueError("PCSE source manifest mapping hash mismatch")
    if manifest.dataset_fingerprint != expected_fingerprint:
        raise ValueError("PCSE source manifest dataset fingerprint mismatch")
    if manifest.dataset != "cifar10" or manifest.num_classes != 10:
        raise ValueError("PCSE source manifest dataset contract mismatch")
    return manifest


def _load_model_state(model: nn.Module, state: Any) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError("PCSE classifier source state_dict is missing")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("PCSE source state_dict is incompatible") from error
    return state


def load_upm_main_best_source(
    source_config: Mapping[str, Any],
    model: nn.Module,
    *,
    num_classes: int,
) -> PCSEPretrainedClassifierSource:
    """Load the one supported source schema without state-dict guessing."""

    if str(source_config.get("adapter", "")).strip().lower() != "upm_main_best":
        raise ValueError("PCSE external source adapter must be upm_main_best")
    run_dir, checkpoint, manifest_identity = _source_files(source_config)

    payload = read_checkpoint(checkpoint.path, "cpu")
    if payload.get("method") != "upm" or payload.get("checkpoint_role") != "main_best":
        raise ValueError("PCSE source must be an UPM main_best checkpoint")
    config = payload.get("config")
    upm_state = payload.get("upm_state")
    state_dict = payload.get("best_main_model_state")
    if not all(isinstance(item, Mapping) for item in (config, upm_state, state_dict)):
        raise ValueError("PCSE UPM source checkpoint schema is incomplete")
    expected_model = dict(source_config.get("model", {}))
    actual_model = dict(config.get("upm", {}).get("main", {}).get("model", {}))
    if actual_model != expected_model:
        raise ValueError("PCSE source model architecture mismatch")
    manifest = _validated_manifest(
        source_config, payload, manifest_identity, num_classes=num_classes
    )
    state_dict = _load_model_state(model, state_dict)

    result = PCSEPretrainedClassifierSource(
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
        adapter="upm_main_best",
        source_method="upm",
        checkpoint_role="main_best",
        model_config=actual_model,
    )
    result.assert_unchanged()
    return result


def _load_standard_source(
    source_config: Mapping[str, Any],
    model: nn.Module,
    *,
    num_classes: int,
) -> PCSEPretrainedClassifierSource:
    adapter = str(source_config.get("adapter", "")).strip().lower()
    run_dir, checkpoint, manifest_identity = _source_files(source_config)
    payload = read_checkpoint(checkpoint.path, "cpu")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("PCSE source checkpoint is missing resolved config")
    expected_model = dict(source_config.get("model", {}))
    actual_model = dict(config.get("model", {}))
    if expected_model != actual_model:
        raise ValueError("PCSE source model architecture mismatch")

    if adapter == "supervised_best":
        execution = config.get("execution", {})
        loss = config.get("loss", {})
        if not isinstance(execution, Mapping) or str(execution.get("runner")) not in {
            "supervised", "clean"
        }:
            raise ValueError("PCSE supervised source runner identity mismatch")
        if not isinstance(loss, Mapping) or str(loss.get("name", "")).lower() != "ce":
            raise ValueError("PCSE supervised source must use CE")
        source_method = "ce"
        checkpoint_role = "best"
        state = payload.get("model")
    elif adapter == "coteaching_peer_a_best":
        method = config.get("method")
        method_name = (
            str(method.get("name", "")) if isinstance(method, Mapping) else str(method)
        ).lower()
        if method_name != "coteaching":
            raise ValueError("PCSE Co-teaching source method identity mismatch")
        models = payload.get("model")
        if not isinstance(models, Mapping) or not isinstance(models.get("a"), Mapping):
            raise ValueError("PCSE Co-teaching peer A state is missing")
        source_method = "coteaching"
        checkpoint_role = "peer_a_best"
        state = models["a"]
    else:
        raise ValueError(f"unsupported PCSE classifier source adapter: {adapter}")

    manifest = _validated_manifest(
        source_config, payload, manifest_identity, num_classes=num_classes
    )
    state_dict = _load_model_state(model, state)
    run_state = payload.get("run_state", {})
    if not isinstance(run_state, Mapping):
        raise ValueError("PCSE source run_state is missing")
    result = PCSEPretrainedClassifierSource(
        run_dir=run_dir,
        checkpoint=checkpoint,
        manifest=manifest_identity,
        checkpoint_payload=payload,
        noise_manifest=manifest,
        state_dict=state_dict,
        source_completed_epochs=int(payload.get("completed_epoch", -1)) + 1,
        source_global_step=int(run_state.get("step", 0)),
        best_epoch=int(payload.get("best_epoch", -1)),
        best_validation_accuracy=float(
            payload.get("best_validation_accuracy", payload.get("best_selection_accuracy", 0.0))
        ),
        adapter=adapter,
        source_method=source_method,
        checkpoint_role=checkpoint_role,
        model_config=actual_model,
    )
    result.assert_unchanged()
    return result


def load_pretrained_classifier_source(
    source_config: Mapping[str, Any],
    model: nn.Module,
    *,
    num_classes: int,
) -> PCSEPretrainedClassifierSource:
    adapter = str(source_config.get("adapter", "")).strip().lower()
    if adapter == "upm_main_best":
        return load_upm_main_best_source(
            source_config, model, num_classes=num_classes
        )
    return _load_standard_source(source_config, model, num_classes=num_classes)


PCSEUPMMainBestSource = PCSEPretrainedClassifierSource


__all__ = [
    "PCSESourceFileIdentity",
    "PCSEUPMMainBestSource",
    "PCSEPretrainedClassifierSource",
    "load_pretrained_classifier_source",
    "load_upm_main_best_source",
]
