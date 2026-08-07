from __future__ import annotations

"""Per-example state used by the Universal Probabilistic Model (UPM).

The state is deliberately independent of a trainer.  ``psi`` is frozen from
the noisy-label warm-up posterior, while ``eta`` is the only state updated by
the alternating UPM procedure.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from .estimators import PosteriorSnapshot


@dataclass
class UPMNoiseState:
    global_indices: Tensor
    noisy_label_probability: Tensor
    confusion_probability: Tensor
    num_classes: int

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PosteriorSnapshot,
        noisy_targets: Tensor,
        *,
        eta_init: float = 0.01,
        eps: float = 1e-8,
    ) -> "UPMNoiseState":
        posterior = torch.as_tensor(snapshot.noisy_probabilities.copy()).detach().cpu()
        indices = torch.as_tensor(snapshot.global_indices.copy(), dtype=torch.long).cpu()
        labels = torch.as_tensor(noisy_targets, dtype=torch.long).cpu()
        if posterior.ndim != 2 or labels.ndim != 1 or posterior.shape[0] != labels.numel():
            raise ValueError("UPM snapshot and noisy targets must align as [N,C] and [N]")
        if indices.shape != labels.shape or torch.unique(indices).numel() != indices.numel():
            raise ValueError("UPM global indices must be unique and aligned")
        if not 0.0 <= float(eta_init) <= 1.0:
            raise ValueError("UPM eta_init must be in [0, 1]")
        if eps <= 0.0:
            raise ValueError("UPM eps must be positive")
        posterior = posterior / posterior.sum(dim=1, keepdim=True).clamp_min(eps)
        if int(labels.min()) < 0 or int(labels.max()) >= posterior.shape[1]:
            raise ValueError("UPM noisy targets are outside posterior classes")
        psi = posterior[torch.arange(labels.numel()), labels].clamp(0.0, 1.0)
        return cls(indices, psi, torch.full_like(psi, float(eta_init)), posterior.shape[1])

    def _positions(self, global_indices: Tensor) -> Tensor:
        query = torch.as_tensor(global_indices, dtype=torch.long).detach().cpu()
        positions = {int(value): i for i, value in enumerate(self.global_indices.tolist())}
        try:
            return torch.tensor([positions[int(value)] for value in query.tolist()], dtype=torch.long)
        except KeyError as exc:
            raise KeyError("UPM lookup contains an unknown global index") from exc

    def lookup(self, global_indices: Tensor) -> tuple[Tensor, Tensor]:
        positions = self._positions(global_indices)
        device = global_indices.device if torch.is_tensor(global_indices) else None
        return (
            self.noisy_label_probability[positions].to(device=device),
            self.confusion_probability[positions].to(device=device),
        )

    def update_eta(self, global_indices: Tensor, values: Tensor) -> None:
        positions = self._positions(global_indices)
        values = torch.as_tensor(values, dtype=self.confusion_probability.dtype).detach().cpu()
        if values.shape != positions.shape or not bool(torch.isfinite(values).all()):
            raise ValueError("UPM eta updates must be finite and aligned")
        self.confusion_probability[positions] = values.clamp(0.0, 1.0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_indices": self.global_indices.clone(),
            "noisy_label_probability": self.noisy_label_probability.clone(),
            "confusion_probability": self.confusion_probability.clone(),
            "num_classes": self.num_classes,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = (self.global_indices.shape, self.num_classes)
        indices = torch.as_tensor(state.get("global_indices"), dtype=torch.long)
        psi = torch.as_tensor(state.get("noisy_label_probability"), dtype=torch.float32)
        eta = torch.as_tensor(state.get("confusion_probability"), dtype=torch.float32)
        if indices.shape != expected[0] or psi.shape != indices.shape or eta.shape != indices.shape:
            raise ValueError("UPM state shapes do not match")
        if int(state.get("num_classes", -1)) != expected[1] or not torch.equal(indices, self.global_indices):
            raise ValueError("UPM state dataset identity does not match")
        if not bool(torch.isfinite(psi).all() and torch.isfinite(eta).all()):
            raise ValueError("UPM state contains non-finite values")
        self.noisy_label_probability.copy_(psi.clamp(0.0, 1.0))
        self.confusion_probability.copy_(eta.clamp(0.0, 1.0))


__all__ = ["UPMNoiseState"]
