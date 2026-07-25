from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lnl_toolbox.noise.generators import generate_pairflip, generate_symmetric
from lnl_toolbox.noise.manifest import NoiseManifest, fingerprint_labels


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def noise_mode(config: Mapping[str, Any]) -> str:
    noise = config.get("noise")
    if not noise:
        return "clean"
    if not isinstance(noise, Mapping):
        raise TypeError("noise configuration must be a mapping")
    has_manifest = bool(noise.get("manifest"))
    name = str(noise.get("name", "clean")).lower()
    generated = name not in {"", "clean", "external"}
    if has_manifest and (generated or "rate" in noise or "seed" in noise):
        raise ValueError("noise.manifest cannot be combined with generated-noise parameters")
    if has_manifest:
        return "external"
    if generated:
        return "generated"
    return "clean"


def noise_enabled(config: Mapping[str, Any]) -> bool:
    return noise_mode(config) != "clean"


def _manifest_filename(noise: Mapping[str, Any]) -> str:
    filename = str(noise.get("manifest_filename", "noise_manifest.npz"))
    if Path(filename).name != filename:
        raise ValueError("noise manifest_filename must be a filename within the run directory")
    return filename


def _generated_spec(config: Mapping[str, Any]) -> tuple[str, float, int]:
    noise = config["noise"]
    name = str(noise["name"]).lower()
    if name not in {"symmetric", "pairflip"}:
        raise ValueError(f"Unsupported generated noise type: {name}")
    rate = float(noise["rate"])
    if not 0.0 <= rate <= 1.0:
        raise ValueError("noise rate must be in [0, 1]")
    return name, rate, int(noise.get("seed", config.get("seed", 1)))


def _validate_manifest(
    manifest: NoiseManifest,
    *,
    dataset: str,
    dataset_targets: np.ndarray,
    required_indices: np.ndarray,
    num_classes: int,
    mode: str,
    generated_spec: tuple[str, float, int] | None,
) -> None:
    manifest.validate_for(
        dataset_targets,
        dataset,
        num_classes,
        required_indices=required_indices,
    )
    if manifest.split != "train":
        raise ValueError("Noise manifest split must be 'train'")
    if mode == "generated":
        assert generated_spec is not None
        noise_type, rate, seed = generated_spec
        checks = {
            "noise type": manifest.noise_type == noise_type,
            "requested rate": np.isclose(
                manifest.requested_rate, rate, rtol=0.0, atol=1e-12
            ),
            "seed": manifest.seed == seed,
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
    dataset_targets: np.ndarray | None = None,
) -> tuple[NoiseManifest | None, Path | None]:
    """Prepare one immutable run-local manifest for generated or external noise."""

    mode = noise_mode(config)
    if mode == "clean":
        if checkpoint_payload is not None and checkpoint_payload.get("noise") is not None:
            raise ValueError("Checkpoint contains noisy-label metadata but the resume config is clean")
        return None, None

    noise = config["noise"]
    filename = _manifest_filename(noise)
    run_dir = run_dir.resolve()
    manifest_path = run_dir / filename
    clean_targets = np.asarray(clean_targets, dtype=np.int64)
    global_indices = np.asarray(global_indices, dtype=np.int64)
    if clean_targets.shape != global_indices.shape:
        raise ValueError("clean_targets and global_indices must have the same shape")
    if dataset_targets is None:
        if not np.array_equal(global_indices, np.arange(clean_targets.size)):
            raise ValueError("dataset_targets is required for non-contiguous global indices")
        dataset_targets = clean_targets
    dataset_targets = np.asarray(dataset_targets, dtype=np.int64)
    generated_spec = _generated_spec(config) if mode == "generated" else None

    if checkpoint_payload is not None:
        checkpoint_noise = checkpoint_payload.get("noise")
        if not isinstance(checkpoint_noise, Mapping):
            raise ValueError("Noisy-label checkpoint is missing manifest association metadata")
        relative_path = checkpoint_noise.get("manifest_path")
        if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
            raise ValueError("Checkpoint noise manifest_path must be relative to the run directory")
        manifest_path = (run_dir / relative_path).resolve()
        if manifest_path.parent != run_dir:
            raise ValueError("Checkpoint noise manifest_path escapes the run directory")
        if checkpoint_noise.get("manifest_path") != filename:
            raise ValueError("Resume config manifest filename does not match the checkpoint")
        if checkpoint_noise.get("mode", mode) != mode:
            raise ValueError("Resume config noise mode does not match the checkpoint")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint-associated noise manifest does not exist: {manifest_path}"
            )
        manifest = NoiseManifest.load(manifest_path)
        _validate_manifest(
            manifest,
            dataset=dataset,
            dataset_targets=dataset_targets,
            required_indices=global_indices,
            num_classes=num_classes,
            mode=mode,
            generated_spec=generated_spec,
        )
        checks = {
            "mapping hash": checkpoint_noise.get("mapping_hash") == manifest.mapping_hash,
            "manifest SHA-256": checkpoint_noise.get("manifest_sha256") in {
                None,
                file_sha256(manifest_path),
            },
            "dataset": checkpoint_noise.get("dataset") == manifest.dataset,
            "dataset fingerprint": checkpoint_noise.get("dataset_fingerprint")
            == manifest.dataset_fingerprint,
            "noise type": checkpoint_noise.get("noise_type") == manifest.noise_type,
            "seed": checkpoint_noise.get("seed") == manifest.seed,
            "num classes": checkpoint_noise.get("num_classes") == manifest.num_classes,
        }
        failed = [name for name, valid in checks.items() if not valid]
        if failed:
            raise ValueError(f"Checkpoint noise metadata validation failed: {', '.join(failed)}")
        return manifest, manifest_path

    if mode == "generated":
        assert generated_spec is not None
        noise_type, rate, seed = generated_spec
        generator = generate_symmetric if noise_type == "symmetric" else generate_pairflip
        manifest = generator(clean_targets, num_classes, rate, seed, dataset)
        manifest.global_indices = global_indices.copy()
        manifest.split = "train"
        manifest.num_classes = num_classes
        manifest.dataset_fingerprint = fingerprint_labels(clean_targets)
        manifest.version = "2.0"
        manifest.metadata = {**manifest.metadata, "source": "generated"}
    else:
        source = Path(str(noise["manifest"])).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Noise manifest does not exist: {source}")
        source_sha = file_sha256(source)
        expected_sha = noise.get("manifest_sha256")
        if expected_sha and str(expected_sha).lower() != source_sha:
            raise ValueError("Noise manifest SHA-256 does not match the configuration")
        manifest = NoiseManifest.load(source)
        _validate_manifest(
            manifest,
            dataset=dataset,
            dataset_targets=dataset_targets,
            required_indices=global_indices,
            num_classes=num_classes,
            mode=mode,
            generated_spec=None,
        )
        manifest.version = "2.0"
        manifest.metadata = {
            **manifest.metadata,
            "source": "external",
            "source_manifest": str(source),
            "source_manifest_sha256": source_sha,
        }

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
        raise ValueError(
            f"Training subset index is absent from noise manifest: {error.args[0]}"
        ) from error
    return float(values.mean()) if values.size else 0.0


def checkpoint_noise_metadata(
    manifest: NoiseManifest,
    manifest_path: Path,
    run_dir: Path,
    effective_rate: float,
    mode: str | None = None,
    *,
    validation_targets: str = "clean",
    effective_validation_rate: float | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "mode": mode or str(manifest.metadata.get("source", "generated")),
        "manifest_path": manifest_path.relative_to(run_dir).as_posix(),
        "manifest_version": manifest.version,
        "manifest_sha256": file_sha256(manifest_path),
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
        "validation_targets": validation_targets,
        "effective_validation_subset_actual_rate": effective_validation_rate,
        "has_transition_matrix": manifest.transition_matrix is not None,
        "has_per_sample_transition": manifest.per_sample_transition is not None,
    }
    source_sha = manifest.metadata.get("source_manifest_sha256")
    if source_sha:
        metadata["source_manifest_sha256"] = str(source_sha)
    return metadata
