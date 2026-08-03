from __future__ import annotations

"""Isolated trusted curriculum records for offline weight-model learning."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MentorFeatureRecord:
    loss: float
    loss_difference: float
    label: int
    epoch_percentage: int
    curriculum_target: float


class MentorFeatureDataset(Dataset):
    """Load trusted MentorNet features without exposing them to Student runs."""

    def __init__(self, records: list[MentorFeatureRecord]) -> None:
        if not records:
            raise ValueError("Mentor feature records must not be empty")
        self.records = tuple(records)

    @classmethod
    def from_npz(cls, path: str | Path) -> "MentorFeatureDataset":
        with np.load(Path(path), allow_pickle=False) as payload:
            required = {
                "losses",
                "loss_differences",
                "labels",
                "epoch_percentages",
                "curriculum_targets",
            }
            if set(payload.files) != required:
                raise ValueError("trusted curriculum NPZ schema mismatch")
            arrays = {name: np.asarray(payload[name]) for name in required}
        sizes = {array.shape for array in arrays.values()}
        if len(sizes) != 1 or next(iter(sizes))[1:] != ():
            raise ValueError("trusted curriculum arrays must be aligned vectors")
        records = [
            MentorFeatureRecord(
                float(loss),
                float(difference),
                int(label),
                int(epoch),
                float(target),
            )
            for loss, difference, label, epoch, target in zip(
                arrays["losses"],
                arrays["loss_differences"],
                arrays["labels"],
                arrays["epoch_percentages"],
                arrays["curriculum_targets"],
            )
        ]
        return cls(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        return {
            "loss": torch.tensor(record.loss, dtype=torch.float32),
            "loss_difference": torch.tensor(
                record.loss_difference, dtype=torch.float32
            ),
            "label": torch.tensor(record.label, dtype=torch.long),
            "epoch_percentage": torch.tensor(
                record.epoch_percentage, dtype=torch.long
            ),
            "curriculum_target": torch.tensor(
                record.curriculum_target, dtype=torch.float32
            ),
        }
