"""Runtime setup blocks with optional numerical backends loaded lazily."""

from __future__ import annotations

import random

from ..context import ScratchContext
from ..registry import block


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("This block requires PyTorch; install the `train` extra.") from exc
    return torch


@block(
    id="set_seed",
    name="Set Seed",
    category="Runtime",
    description="Set Python, NumPy, and PyTorch random seeds when available.",
    params={"seed": {"type": "int", "default": 1, "min": 0}},
    provides=("seed",),
    placement=("top",), stage="setup", ui_group="② 初始化",
)
def set_seed(ctx: ScratchContext, seed: int = 1) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - NumPy is a project dependency
        pass
    try:
        torch = _torch()
    except RuntimeError:
        ctx["seed"] = int(seed)
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ctx["seed"] = int(seed)


@block(
    id="select_device",
    name="Select Device",
    category="Runtime",
    description="Select CUDA when requested and available, otherwise use CPU.",
    params={"device": {"type": "str", "default": "auto"}, "save_as": {"type": "slot", "default": "device"}},
    provides=("save_as",),
    placement=("top",), stage="setup", ui_group="② 初始化",
)
def select_device(ctx: ScratchContext, device: str = "auto", save_as: str = "device") -> None:
    requested = str(device).strip().lower()
    try:
        torch = _torch()
        resolved = "cuda" if requested in {"auto", "cuda", "gpu"} and torch.cuda.is_available() else "cpu"
        if requested == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            resolved = "mps"
        if requested not in {"auto", "cuda", "gpu", "cpu", "mps"}:
            resolved = device
    except RuntimeError:
        resolved = "cpu" if requested == "auto" else device
    ctx[save_as] = resolved
