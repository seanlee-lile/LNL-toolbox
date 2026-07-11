from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from lnl_toolbox.core import RunState


def save_checkpoint(path: str | Path, algorithm, run_state: RunState,
                    completed_epoch: int, config: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "algorithm": algorithm.state_dict(),
        "run_state": asdict(run_state),
        "completed_epoch": completed_epoch,
        "config": dict(config),
    }, path)


def load_checkpoint(path: str | Path, algorithm, device: torch.device) -> tuple[RunState, int, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    algorithm.load_state_dict(payload["algorithm"])
    state = RunState(**payload["run_state"])
    return state, int(payload["completed_epoch"]), payload
