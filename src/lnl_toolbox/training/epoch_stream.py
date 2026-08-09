from __future__ import annotations

"""Deterministic epoch-scoped data streams for resumable training."""

import hashlib
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader


_STREAM_VERSION = 1


def derive_epoch_seed(base_seed: int, namespace: str, epoch: int) -> int:
    """Derive a stable PyTorch-compatible seed without Python ``hash()``."""

    name = str(namespace).strip()
    if not name:
        raise ValueError("epoch stream namespace must not be empty")
    if int(epoch) < 0:
        raise ValueError("epoch stream epoch must be non-negative")
    encoded = f"lnl-epoch-stream-v{_STREAM_VERSION}\0{int(base_seed)}\0{name}\0{int(epoch)}"
    value = int.from_bytes(hashlib.sha256(encoded.encode("utf-8")).digest()[:8], "big")
    return value % (2**63 - 1)


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seed_epoch_process(base_seed: int, namespace: str, epoch: int) -> int:
    """Seed main-process stochastic operations for one deterministic epoch."""

    seed = derive_epoch_seed(base_seed, namespace, epoch)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def build_epoch_loader(
    dataset: Any,
    config: Mapping[str, Any],
    *,
    base_seed: int,
    namespace: str,
    epoch: int,
    shuffle: bool = True,
    batch_size: int | None = None,
    drop_last: bool | None = None,
) -> DataLoader:
    """Rebuild workers and sampler from an epoch-derived deterministic seed."""

    seed = seed_epoch_process(base_seed, namespace, epoch)
    workers = int(config.get("num_workers", 0))
    if workers < 0:
        raise ValueError("loader num_workers must be non-negative")
    effective_drop_last = (
        bool(config.get("drop_last", False)) if drop_last is None else bool(drop_last)
    )
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"] if batch_size is None else batch_size),
        shuffle=bool(shuffle),
        num_workers=workers,
        pin_memory=bool(config.get("pin_memory", True)),
        persistent_workers=False,
        worker_init_fn=_seed_worker if workers else None,
        generator=torch.Generator().manual_seed(seed),
        drop_last=effective_drop_last if shuffle else False,
    )


def loader_stream_metadata(
    *, base_seed: int, namespace: str, next_epoch: int
) -> dict[str, Any]:
    """Return checkpoint metadata identifying the next epoch data stream."""

    if int(next_epoch) < 0:
        raise ValueError("loader stream next_epoch must be non-negative")
    # Validate the namespace through the same canonical seed path.
    derive_epoch_seed(base_seed, namespace, next_epoch)
    return {
        "version": _STREAM_VERSION,
        "base_seed": int(base_seed),
        "namespace": str(namespace).strip(),
        "next_epoch": int(next_epoch),
    }


def validate_loader_stream(
    value: Any,
    *,
    base_seed: int,
    namespace: str,
    next_epoch: int,
) -> bool:
    """Validate new metadata; return ``False`` for a legacy missing value."""

    if value is None:
        return False
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint loader_stream must be a mapping")
    expected = loader_stream_metadata(
        base_seed=base_seed, namespace=namespace, next_epoch=next_epoch
    )
    actual = dict(value)
    if actual != expected:
        raise ValueError(
            "checkpoint loader stream identity mismatch: "
            f"expected {expected!r}, found {actual!r}"
        )
    return True


__all__ = [
    "build_epoch_loader",
    "derive_epoch_seed",
    "loader_stream_metadata",
    "seed_epoch_process",
    "validate_loader_stream",
]
