from __future__ import annotations

"""Lightweight dataset-path recognition for the Quick Start flow.

This module only recognizes layouts.  It deliberately does not register a
dataset, load samples, or inspect labels; those operations belong to
``DataService`` after the user has selected an unambiguous candidate.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from lnl_toolbox.training.data_service import DataService


_CONFIDENCE = {"high", "medium", "low"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True, slots=True)
class ProbeCandidate:
    adapter: str
    confidence: str
    reason: str
    data: dict[str, Any]

    def __post_init__(self) -> None:
        if self.confidence not in _CONFIDENCE:
            raise ValueError(f"unsupported probe confidence: {self.confidence}")


@dataclass(frozen=True, slots=True)
class DatasetProbeResult:
    path: str
    status: str
    candidates: tuple[ProbeCandidate, ...] = ()
    existing_alias: str | None = None


def _cifar10_root(path: Path) -> Path | None:
    required = {*(f"data_batch_{index}" for index in range(1, 6)), "test_batch", "batches.meta"}
    for root in (path, path / "cifar-10-batches-py"):
        if root.is_dir() and required.issubset({item.name for item in root.iterdir()}):
            return root
    return None


def _cifar100_root(path: Path) -> Path | None:
    for root in (path, path / "cifar-100-python"):
        if root.is_dir() and {"train", "test", "meta"}.issubset({item.name for item in root.iterdir()}):
            return root
    return None


def _cifar_n_file(path: Path, classes: int) -> Path | None:
    filename = "CIFAR-10_human.pt" if classes == 10 else "CIFAR-100_human.pt"
    candidates = (path / filename, path.parent / filename)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _idx_signature(path: Path) -> tuple[str, Path] | None:
    names = {
        "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
        "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
    }
    roots = (path, path / "raw", path / "MNIST" / "raw", path / "FashionMNIST" / "raw")
    for root in roots:
        if not root.is_dir():
            continue
        for suffix in ("", ".gz"):
            if all((root / f"{name}{suffix}").is_file() for pair in names.values() for name in pair):
                return ("idx.gz" if suffix else "idx", root)
    return None


def _clothing_signature(path: Path) -> bool:
    required = {
        "noisy_train_key_list.txt",
        "clean_val_key_list.txt",
        "clean_test_key_list.txt",
        "noisy_label_kv.txt",
        "clean_label_kv.txt",
    }
    return path.is_dir() and required.issubset({item.name for item in path.iterdir()})


def _animal_signature(path: Path) -> bool:
    binary = bool(tuple(path.glob("data_batch_*.bin")) and (path / "test_batch.bin").is_file())
    folders = any((path / train).is_dir() for train in ("train", "training")) and any(
        (path / test).is_dir() for test in ("test", "testing")
    )
    if binary or not folders:
        return binary
    train = next(path / name for name in ("train", "training") if (path / name).is_dir())
    test = next(path / name for name in ("test", "testing") if (path / name).is_dir())
    return any(item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES for item in train.rglob("*")) and any(
        item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES for item in test.rglob("*")
    )


def _looks_like_uci(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        rows = [line.split(",") if "," in line else line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][:8]
    except (OSError, UnicodeError):
        return False
    if len(rows) < 2 or len({len(row) for row in rows}) != 1 or len(rows[0]) < 3:
        return False
    return all(len(re.findall(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$", value.strip())) == 1 for row in rows for value in row)


def _registered_alias(path: Path, data_service: "DataService") -> str | None:
    resolved = path.expanduser().resolve()
    for record in data_service.catalog.records():
        for key in ("root", "path", "noise_path", "labels_path", "annotation_root"):
            value = record.data.get(key)
            if value and Path(str(value)).expanduser().resolve() == resolved:
                return record.alias
    return None


def _candidate(adapter: str, confidence: str, reason: str, **data: Any) -> ProbeCandidate:
    return ProbeCandidate(adapter, confidence, reason, data)


def probe_dataset_path(
    path: str | Path,
    *,
    data_service: "DataService | None" = None,
) -> DatasetProbeResult:
    """Recognize a local path using adapter-specific layout signatures."""

    from lnl_toolbox.training.data_service import DEFAULT_DATA_SERVICE

    service = data_service or DEFAULT_DATA_SERVICE
    original = Path(path).expanduser()
    resolved = original.resolve()
    if not resolved.exists():
        return DatasetProbeResult(str(resolved), "unsupported")
    existing = _registered_alias(resolved, service)
    if existing is not None:
        return DatasetProbeResult(str(resolved), "already_registered", existing_alias=existing)

    candidates: list[ProbeCandidate] = []
    cifar10 = _cifar10_root(resolved) if resolved.is_dir() else None
    cifar100 = _cifar100_root(resolved) if resolved.is_dir() else None
    if cifar10 is not None:
        annotation = _cifar_n_file(cifar10, 10)
        if annotation is not None:
            candidates.append(_candidate(
                "cifar10n", "high", "official CIFAR-10 batches and CIFAR-10_human.pt", root=str(cifar10), noise_path=str(annotation)
            ))
        candidates.append(_candidate("cifar10", "high", "official CIFAR-10 batches", root=str(cifar10)))
    if cifar100 is not None:
        annotation = _cifar_n_file(cifar100, 100)
        if annotation is not None:
            candidates.append(_candidate(
                "cifar100n", "high", "official CIFAR-100 files and CIFAR-100_human.pt", root=str(cifar100), noise_path=str(annotation)
            ))
        candidates.append(_candidate("cifar100", "high", "official CIFAR-100 files", root=str(cifar100)))
    idx = _idx_signature(resolved) if resolved.is_dir() else None
    if idx is not None:
        kind, root = idx
        if (root / "FashionMNIST").exists() or "fashion" in root.as_posix().lower():
            candidates.append(_candidate("fashion_mnist", "high", f"official Fashion-MNIST {kind} files", root=str(resolved)))
        elif (root / "MNIST").exists() or "mnist" in root.as_posix().lower():
            candidates.append(_candidate("mnist", "high", f"official MNIST {kind} files", root=str(resolved)))
        else:
            candidates.extend((
                _candidate("mnist", "medium", f"official MNIST-family {kind} files", root=str(resolved)),
                _candidate("fashion_mnist", "medium", f"official MNIST-family {kind} files", root=str(resolved)),
            ))
    if _clothing_signature(resolved):
        candidates.append(_candidate("clothing1m", "high", "Clothing1M key lists and label mappings", root=str(resolved)))
    if _animal_signature(resolved):
        candidates.append(_candidate("animal10n", "high", "Animal-10N binary or train/test image layout", root=str(resolved)))
    if _looks_like_uci(resolved):
        candidates.append(_candidate(
            "uci_binary", "medium", "single numeric tabular file with a candidate target column", path=str(resolved),
            preprocessing={"format": "delimited", "target_column": -1, "has_header": False},
        ))

    # Prefer exact high-confidence candidates, but never silently choose between
    # a clean CIFAR source and its CIFAR-N annotation source.
    unique: dict[tuple[str, str], ProbeCandidate] = {(item.adapter, str(item.data.get("root") or item.data.get("path"))): item for item in candidates}
    candidates = list(unique.values())
    if not candidates:
        return DatasetProbeResult(str(resolved), "unsupported")
    if len(candidates) == 1:
        return DatasetProbeResult(str(resolved), "detected", tuple(candidates))
    return DatasetProbeResult(str(resolved), "ambiguous", tuple(candidates))


def suggest_dataset_alias(
    adapter: str,
    path: str | Path,
    existing_names: Iterable[str],
) -> str:
    """Return a stable, collision-free local alias for a probe candidate."""

    del path
    normalized = re.sub(r"[^a-z0-9]+", "-", str(adapter).strip().lower()).strip("-")
    base = f"{normalized}-local"
    used = {str(value).strip().lower() for value in existing_names}
    if base not in used:
        return base
    index = 2
    while f"{base}-{index}" in used:
        index += 1
    return f"{base}-{index}"


__all__ = ["DatasetProbeResult", "ProbeCandidate", "probe_dataset_path", "suggest_dataset_alias"]
