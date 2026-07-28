from __future__ import annotations

"""Batch contracts for methods that split noisy data into labeled/unlabeled views."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SemiSupervisedBatch:
    labeled: Mapping[str, Any]
    unlabeled: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.labeled, Mapping) or not isinstance(self.unlabeled, Mapping):
            raise TypeError("labeled and unlabeled batches must be mappings")
        for name, batch in (("labeled", self.labeled), ("unlabeled", self.unlabeled)):
            if "input" not in batch or "index" not in batch:
                raise ValueError(f"{name} batch must contain input and index")

    @property
    def labeled_indices(self) -> Any:
        return self.labeled["index"]

    @property
    def unlabeled_indices(self) -> Any:
        return self.unlabeled["index"]
