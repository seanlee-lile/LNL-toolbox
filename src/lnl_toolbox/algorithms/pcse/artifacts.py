from __future__ import annotations

"""Hashed, versioned NPZ artifacts for PCSE stages."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .gda import GDALayer
from .statistics import PCSELayerStatistics, PCSEStatistics


def _hash_payload(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            dict(metadata), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _write_npz(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    artifact_hash: str,
) -> None:
    payload = {**dict(metadata), "artifact_hash": artifact_hash}
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.array(json.dumps(payload, sort_keys=True)),
    )


def persist_npz_atomically(
    artifact: Any,
    destination: Path,
    loader: Callable[[Path], Any],
) -> Any:
    """Write, reload and validate before atomically publishing one artifact."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    if temporary.exists():
        raise FileExistsError(f"PCSE temporary artifact already exists: {temporary}")
    try:
        artifact.save(temporary)
        loaded = loader(temporary)
        if loaded.artifact_hash != artifact.artifact_hash:
            raise ValueError("PCSE temporary artifact hash verification failed")
        temporary.replace(destination)
        return loaded
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class PCSEStatisticsArtifact:
    statistics: PCSEStatistics
    provenance: Mapping[str, Any]
    version: str = "1.0"

    def _parts(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        metadata: dict[str, Any] = {
            "version": self.version,
            "kind": "pcse_statistics",
            "provenance": dict(self.provenance),
            "layer_names": [layer.name for layer in self.statistics.layers],
            "transition_condition": self.statistics.transition_condition,
            "coefficient_condition": self.statistics.coefficient_condition,
        }
        arrays = {
            "noisy_priors": self.statistics.noisy_priors,
            "clean_priors": self.statistics.clean_priors,
            "coefficient_matrix": self.statistics.coefficient_matrix,
        }
        for index, layer in enumerate(self.statistics.layers):
            prefix = f"layer_{index}"
            arrays[f"{prefix}_noisy_means"] = layer.noisy_means
            arrays[f"{prefix}_noisy_second_moments"] = layer.noisy_second_moments
            arrays[f"{prefix}_clean_means"] = layer.clean_means
            arrays[f"{prefix}_clean_second_moments"] = layer.clean_second_moments
            arrays[f"{prefix}_clean_covariances"] = layer.clean_covariances
        return metadata, arrays

    @property
    def artifact_hash(self) -> str:
        metadata, arrays = self._parts()
        return _hash_payload(metadata, arrays)

    def save(self, path: str | Path) -> None:
        metadata, arrays = self._parts()
        _write_npz(
            Path(path),
            metadata=metadata,
            arrays=arrays,
            artifact_hash=self.artifact_hash,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PCSEStatisticsArtifact":
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["metadata_json"].item()))
            names = payload["layer_names"]
            layers = tuple(
                PCSELayerStatistics(
                    name=name,
                    noisy_means=data[f"layer_{index}_noisy_means"],
                    noisy_second_moments=data[
                        f"layer_{index}_noisy_second_moments"
                    ],
                    clean_means=data[f"layer_{index}_clean_means"],
                    clean_second_moments=data[
                        f"layer_{index}_clean_second_moments"
                    ],
                    clean_covariances=data[
                        f"layer_{index}_clean_covariances"
                    ],
                )
                for index, name in enumerate(names)
            )
            artifact = cls(
                PCSEStatistics(
                    noisy_priors=data["noisy_priors"],
                    clean_priors=data["clean_priors"],
                    coefficient_matrix=data["coefficient_matrix"],
                    transition_condition=float(payload["transition_condition"]),
                    coefficient_condition=float(payload["coefficient_condition"]),
                    layers=layers,
                ),
                provenance=payload["provenance"],
                version=payload["version"],
            )
            if payload.get("artifact_hash") != artifact.artifact_hash:
                raise ValueError("PCSE statistics artifact hash mismatch")
            return artifact


@dataclass(frozen=True)
class PCSEGDAArtifact:
    layers: tuple[GDALayer, ...]
    provenance: Mapping[str, Any]
    version: str = "1.0"

    def _parts(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        metadata = {
            "version": self.version,
            "kind": "pcse_gda",
            "provenance": dict(self.provenance),
            "layer_names": [layer.name for layer in self.layers],
            "covariance_ridges": [
                float(layer.covariance_ridge) for layer in self.layers
            ],
        }
        arrays: dict[str, np.ndarray] = {}
        for index, layer in enumerate(self.layers):
            arrays[f"layer_{index}_priors"] = layer.clean_priors
            arrays[f"layer_{index}_means"] = layer.means
            arrays[f"layer_{index}_covariance"] = layer.shared_covariance
        return metadata, arrays

    @property
    def artifact_hash(self) -> str:
        metadata, arrays = self._parts()
        return _hash_payload(metadata, arrays)

    def save(self, path: str | Path) -> None:
        metadata, arrays = self._parts()
        _write_npz(
            Path(path),
            metadata=metadata,
            arrays=arrays,
            artifact_hash=self.artifact_hash,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PCSEGDAArtifact":
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["metadata_json"].item()))
            layers = tuple(
                GDALayer(
                    name=name,
                    clean_priors=data[f"layer_{index}_priors"],
                    means=data[f"layer_{index}_means"],
                    shared_covariance=data[f"layer_{index}_covariance"],
                    covariance_ridge=float(
                        payload["covariance_ridges"][index]
                    ),
                )
                for index, name in enumerate(payload["layer_names"])
            )
            artifact = cls(
                layers=layers,
                provenance=payload["provenance"],
                version=payload["version"],
            )
            if payload.get("artifact_hash") != artifact.artifact_hash:
                raise ValueError("PCSE GDA artifact hash mismatch")
            return artifact


@dataclass(frozen=True)
class PCSEEnsembleArtifact:
    layer_names: tuple[str, ...]
    weights: np.ndarray
    provenance: Mapping[str, Any]
    version: str = "1.0"

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        if len(self.layer_names) < 2 or weights.shape != (len(self.layer_names),):
            raise ValueError("PCSE ensemble weights must align at least two layers")
        if (
            not np.isfinite(weights).all()
            or (weights <= 0.0).any()
            or not np.isclose(weights.sum(), 1.0)
        ):
            raise ValueError(
                "PCSE ensemble weights must be positive, finite and sum to one"
            )
        object.__setattr__(self, "weights", weights.copy())

    def _parts(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        return (
            {
                "version": self.version,
                "kind": "pcse_ensemble",
                "provenance": dict(self.provenance),
                "layer_names": list(self.layer_names),
            },
            {"weights": self.weights},
        )

    @property
    def artifact_hash(self) -> str:
        metadata, arrays = self._parts()
        return _hash_payload(metadata, arrays)

    def save(self, path: str | Path) -> None:
        metadata, arrays = self._parts()
        _write_npz(
            Path(path),
            metadata=metadata,
            arrays=arrays,
            artifact_hash=self.artifact_hash,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PCSEEnsembleArtifact":
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["metadata_json"].item()))
            artifact = cls(
                layer_names=tuple(payload["layer_names"]),
                weights=data["weights"],
                provenance=payload["provenance"],
                version=payload["version"],
            )
            if payload.get("artifact_hash") != artifact.artifact_hash:
                raise ValueError("PCSE ensemble artifact hash mismatch")
            return artifact
