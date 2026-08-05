from __future__ import annotations

"""Explicit, auditable clean supervision for bilevel methods such as L2RW."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True, slots=True)
class TrustedSupervisionManifest:
    global_indices: np.ndarray
    targets: np.ndarray
    dataset: str
    split: str
    source: str
    balanced: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        indices = np.asarray(self.global_indices, dtype=np.int64)
        targets = np.asarray(self.targets, dtype=np.int64)
        if indices.ndim != 1 or targets.shape != indices.shape or indices.size == 0:
            raise ValueError("trusted manifest arrays must be aligned and non-empty")
        if np.unique(indices).size != indices.size or indices.min() < 0 or targets.min() < 0:
            raise ValueError("trusted manifest indices/targets are invalid")
        dataset = str(self.dataset).strip()
        split = str(self.split).strip().lower()
        source = str(self.source).strip().lower()
        if not dataset or split != "trusted_validation":
            raise ValueError("trusted supervision split must be 'trusted_validation'")
        if source not in {"audited_manifest", "synthetic_fixture"}:
            raise ValueError("trusted supervision source must be explicitly audited")
        order = np.argsort(indices, kind="stable")
        object.__setattr__(self, "global_indices", indices[order].copy())
        object.__setattr__(self, "targets", targets[order].copy())
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        context = {
            "dataset": self.dataset,
            "split": self.split,
            "source": self.source,
            "balanced": self.balanced,
            "metadata": dict(self.metadata),
        }
        digest.update(json.dumps(context, sort_keys=True, separators=(",", ":")).encode())
        digest.update(self.global_indices.astype("<i8", copy=False).tobytes())
        digest.update(self.targets.astype("<i8", copy=False).tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        metadata = {
            "dataset": self.dataset,
            "split": self.split,
            "source": self.source,
            "balanced": self.balanced,
            "metadata": dict(self.metadata),
            "fingerprint": self.fingerprint,
        }
        np.savez_compressed(
            path,
            global_indices=self.global_indices,
            targets=self.targets,
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrustedSupervisionManifest":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            manifest = cls(
                data["global_indices"], data["targets"], metadata["dataset"],
                metadata["split"], metadata["source"], bool(metadata["balanced"]),
                metadata.get("metadata", {}),
            )
        if metadata.get("fingerprint") != manifest.fingerprint:
            raise ValueError("trusted supervision fingerprint mismatch")
        return manifest


class _TrustedDataset(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset, manifest: TrustedSupervisionManifest) -> None:
        self.dataset = dataset
        self._targets = {
            int(index): int(target)
            for index, target in zip(manifest.global_indices, manifest.targets)
        }
        observed = []
        for position in range(len(dataset)):
            sample = dataset[position]
            if not isinstance(sample, Mapping) or "index" not in sample or "input" not in sample:
                raise TypeError("trusted base dataset must return input/index mappings")
            observed.append(int(sample["index"]))
        if set(observed) != set(self._targets):
            raise ValueError("trusted manifest and dataset sample identities differ")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = self.dataset[item]
        index = int(sample["index"])
        return {"input": sample["input"], "target": self._targets[index], "index": index}


class TrustedValidationProvider:
    """Own a dedicated trusted mini-batch source and its provenance."""

    def __init__(self, dataset: Dataset, manifest: TrustedSupervisionManifest) -> None:
        if not isinstance(manifest, TrustedSupervisionManifest):
            raise TypeError("TrustedValidationProvider requires an explicit manifest")
        self.manifest = manifest
        self.dataset = _TrustedDataset(dataset, manifest)

    @property
    def fingerprint(self) -> str:
        return self.manifest.fingerprint

    def loader(self, *, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0) -> DataLoader:
        if batch_size <= 0:
            raise ValueError("trusted batch_size must be positive")
        import torch

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed),
            drop_last=False,
        )


__all__ = ["TrustedSupervisionManifest", "TrustedValidationProvider"]
