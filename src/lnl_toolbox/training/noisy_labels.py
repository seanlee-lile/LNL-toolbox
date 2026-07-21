from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lnl_toolbox.noise.generators import generate_pairflip, generate_symmetric
from lnl_toolbox.noise.manifest import NoiseManifest, fingerprint_labels


def validate_ce_algorithm(config: Mapping[str, Any]) -> None:
    name = str(config.get("algorithm", {}).get("name", "ce")).lower()
    if name != "ce":
        raise ValueError(f"Unsupported algorithm for this baseline: {name}; expected ce")


def noise_enabled(config: Mapping[str, Any]) -> bool:
    noise = config.get("noise")
    return bool(noise) and str(noise.get("name", "clean")).lower() != "clean"


def _expected_noise(config: Mapping[str, Any]) -> tuple[str, float, int, str]:
    noise = config["noise"]
    name = str(noise["name"]).lower()
    if name not in {"symmetric", "pairflip"}:
        raise ValueError(f"Unsupported noise type for CE baseline: {name}")
    rate = float(noise["rate"])
    seed = int(noise.get("seed", config.get("seed", 1)))
    filename = str(noise.get("manifest_filename", "noise_manifest.npz"))
    if Path(filename).name != filename:
        raise ValueError("noise manifest_filename must be a filename within the run directory")
    return name, rate, seed, filename


def _validate_manifest(
    manifest: NoiseManifest,
    *,
    dataset: str,
    clean_targets: np.ndarray,
    global_indices: np.ndarray,
    num_classes: int,
    noise_type: str,
    rate: float,
    seed: int,
) -> None:
    expected_fingerprint = fingerprint_labels(clean_targets)
    checks = {
        "dataset": manifest.dataset == dataset,
        "split": manifest.split == "train",
        "dataset fingerprint": manifest.dataset_fingerprint == expected_fingerprint,
        "noise type": manifest.noise_type == noise_type,
        "requested rate": np.isclose(manifest.requested_rate, rate, rtol=0.0, atol=1e-12),
        "seed": manifest.seed == seed,
        "num classes": manifest.num_classes == num_classes,
        "global indices": np.array_equal(manifest.global_indices, global_indices),
        "clean targets": np.array_equal(manifest.clean_targets, clean_targets),
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"Noise manifest validation failed: {', '.join(failed)}")


def prepare_noise_manifest(
    config: Mapping[str, Any],
    *,
    dataset: str,
    clean_targets: np.ndarray,
    global_indices: np.ndarray,
    num_classes: int,
    run_dir: Path,
    checkpoint_payload: Mapping[str, Any] | None = None,
) -> tuple[NoiseManifest | None, Path | None]:
    """Generate a new manifest or strictly restore the checkpoint-associated one."""

    validate_ce_algorithm(config)
    if not noise_enabled(config):
        if checkpoint_payload is not None and checkpoint_payload.get("noise") is not None:
            raise ValueError("Checkpoint contains noisy-label metadata but the resume config is clean")
        return None, None

    noise_type, rate, seed, filename = _expected_noise(config)
    clean_targets = np.asarray(clean_targets, dtype=np.int64)
    global_indices = np.asarray(global_indices, dtype=np.int64)
    if clean_targets.shape != global_indices.shape:
        raise ValueError("clean_targets and global_indices must have the same shape")

    if checkpoint_payload is not None:
        checkpoint_noise = checkpoint_payload.get("noise")
        if not isinstance(checkpoint_noise, Mapping):
            raise ValueError("Noisy-label checkpoint is missing manifest association metadata")
        relative_path = checkpoint_noise.get("manifest_path")
        if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
            raise ValueError("Checkpoint noise manifest_path must be relative to the run directory")
        manifest_path = (run_dir / relative_path).resolve()
        if manifest_path.parent != run_dir.resolve():
            raise ValueError("Checkpoint noise manifest_path escapes the run directory")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Checkpoint-associated noise manifest does not exist: {manifest_path}")
        manifest = NoiseManifest.load(manifest_path)
        _validate_manifest(
            manifest, dataset=dataset, clean_targets=clean_targets, global_indices=global_indices,
            num_classes=num_classes, noise_type=noise_type, rate=rate, seed=seed,
        )
        if checkpoint_noise.get("mapping_hash") != manifest.mapping_hash:
            raise ValueError("Checkpoint mapping hash does not match the noise manifest")
        if checkpoint_noise.get("manifest_path") != filename:
            raise ValueError("Resume config manifest filename does not match the checkpoint")
        checkpoint_checks = {
            "dataset": checkpoint_noise.get("dataset") == manifest.dataset,
            "split": checkpoint_noise.get("split") == manifest.split,
            "dataset fingerprint": checkpoint_noise.get("dataset_fingerprint") == manifest.dataset_fingerprint,
            "noise type": checkpoint_noise.get("noise_type") == manifest.noise_type,
            "requested rate": np.isclose(
                float(checkpoint_noise.get("requested_rate", -1.0)), manifest.requested_rate,
                rtol=0.0, atol=1e-12,
            ),
            "seed": checkpoint_noise.get("seed") == manifest.seed,
            "num classes": checkpoint_noise.get("num_classes") == manifest.num_classes,
            "manifest actual rate": np.isclose(
                float(checkpoint_noise.get("manifest_actual_rate", -1.0)), manifest.actual_rate,
                rtol=0.0, atol=1e-12,
            ),
        }
        failed = [name for name, valid in checkpoint_checks.items() if not valid]
        if failed:
            raise ValueError(f"Checkpoint noise metadata validation failed: {', '.join(failed)}")
        return manifest, manifest_path

    generator = generate_symmetric if noise_type == "symmetric" else generate_pairflip
    manifest = generator(clean_targets, num_classes, rate, seed, dataset)
    manifest.split = "train"
    manifest.num_classes = num_classes
    manifest.global_indices = global_indices
    manifest.dataset_fingerprint = fingerprint_labels(clean_targets)
    manifest.version = "2.0"
    manifest_path = run_dir / filename
    manifest.save(manifest_path)
    return manifest, manifest_path


def effective_subset_actual_rate(manifest: NoiseManifest, subset_indices: np.ndarray) -> float:
    mapping = {
        int(index): bool(corrupted)
        for index, corrupted in zip(manifest.global_indices, manifest.corruption_mask)
    }
    try:
        values = np.asarray([mapping[int(index)] for index in subset_indices], dtype=np.bool_)
    except KeyError as error:
        raise ValueError(f"Training subset index is absent from noise manifest: {error.args[0]}") from error
    return float(values.mean()) if values.size else 0.0


def checkpoint_noise_metadata(
    manifest: NoiseManifest, manifest_path: Path, run_dir: Path, effective_rate: float
) -> dict[str, Any]:
    return {
        "manifest_path": manifest_path.relative_to(run_dir).as_posix(),
        "manifest_version": manifest.version,
        "mapping_hash": manifest.mapping_hash,
        "dataset": manifest.dataset,
        "split": manifest.split,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "noise_type": manifest.noise_type,
        "requested_rate": manifest.requested_rate,
        "seed": manifest.seed,
        "num_classes": manifest.num_classes,
        "manifest_actual_rate": manifest.actual_rate,
        "effective_train_subset_actual_rate": effective_rate,
    }
