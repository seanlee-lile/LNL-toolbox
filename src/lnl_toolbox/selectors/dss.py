"""Stateful MDA, prediction matching, and CCS for DSS."""

from __future__ import annotations

from collections.abc import Mapping
from statistics import NormalDist
from typing import Any

import torch
from torch import Tensor

from .history import IndexedTensorHistory


class DSSSelectorState:
    """Dataset-indexed DSS state updated once per observed sample and epoch."""

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        total_epochs: int,
        *,
        warmup_epochs: int = 30,
        alpha: float = 0.10,
        prior_decay: float = 0.99,
        mda: bool = True,
        ccs: bool = True,
    ) -> None:
        if int(num_samples) <= 0 or int(num_classes) < 2:
            raise ValueError("DSS requires positive samples and at least two classes")
        if int(total_epochs) <= 0:
            raise ValueError("total_epochs must be positive")
        if not 0 <= int(warmup_epochs) <= int(total_epochs):
            raise ValueError("warmup_epochs must be within total_epochs")
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 <= float(prior_decay) < 1.0:
            raise ValueError("prior_decay must be in [0, 1)")
        self.num_samples = int(num_samples)
        self.num_classes = int(num_classes)
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.alpha = float(alpha)
        self.prior_decay = float(prior_decay)
        self.mda = bool(mda)
        self.ccs = bool(ccs)
        self.history = IndexedTensorHistory(
            self.num_samples, self.total_epochs, self.num_classes
        )
        self.labels = torch.full((self.num_samples,), -1, dtype=torch.long)
        self.current_prediction = torch.full(
            (self.num_samples, self.num_classes),
            1.0 / self.num_classes,
            dtype=torch.float32,
        )
        self.trend_score = torch.zeros(
            self.num_samples, self.num_classes, dtype=torch.float32
        )
        self.marginal = torch.full(
            (self.num_classes,), 1.0 / self.num_classes, dtype=torch.float32
        )
        self.selected = torch.ones(self.num_samples, dtype=torch.bool)
        self.excluded = torch.zeros(
            self.num_samples, self.num_classes, dtype=torch.bool
        )
        self.current_epoch = -1

    def _indices_targets(
        self, sample_indices: Tensor, noisy_targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        if not torch.is_tensor(sample_indices) or sample_indices.ndim != 1:
            raise ValueError("DSS sample indices must have shape [B]")
        if sample_indices.dtype not in {
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        }:
            raise ValueError("DSS sample indices must use an integer dtype")
        indices = sample_indices.detach().cpu().to(torch.long)
        targets = torch.as_tensor(noisy_targets).detach().cpu()
        if targets.ndim != 1 or targets.shape != indices.shape:
            raise ValueError("DSS indices and targets must align as [B]")
        if targets.dtype not in {
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        }:
            raise ValueError("DSS noisy targets must use an integer dtype")
        targets = targets.to(torch.long)
        if indices.numel() == 0:
            raise ValueError("DSS batch must not be empty")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("DSS batch sample indices must be unique")
        if int(indices.min()) < 0 or int(indices.max()) >= self.num_samples:
            raise IndexError("DSS sample index exceeds num_samples")
        if int(targets.min()) < 0 or int(targets.max()) >= self.num_classes:
            raise ValueError("DSS noisy target is outside the class range")
        known = self.labels[indices]
        if bool(((known >= 0) & (known != targets)).any()):
            raise ValueError("DSS noisy target changed for a stable sample index")
        return indices, targets

    def masks(
        self, sample_indices: Tensor, noisy_targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        indices, targets = self._indices_targets(sample_indices, noisy_targets)
        excluded = self.excluded[indices].clone()
        excluded[torch.arange(indices.numel()), targets] = False
        return self.selected[indices].clone(), excluded

    def observe(
        self,
        sample_indices: Tensor,
        noisy_targets: Tensor,
        probabilities: Tensor,
        epoch: int,
    ) -> None:
        indices, targets = self._indices_targets(sample_indices, noisy_targets)
        epoch = int(epoch)
        if epoch != self.current_epoch:
            raise ValueError("DSS observation epoch does not match lifecycle state")
        values = torch.as_tensor(probabilities).detach().cpu().to(torch.float32)
        if values.shape != (indices.numel(), self.num_classes):
            raise ValueError("DSS probabilities must have shape [B, C]")
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("DSS probabilities must be finite and non-negative")
        if not torch.allclose(
            values.sum(dim=1), torch.ones(indices.numel()), rtol=1e-5, atol=1e-6
        ):
            raise ValueError("DSS probabilities must sum to one")
        if self.mda:
            self.marginal.mul_(self.prior_decay).add_(
                values.mean(dim=0), alpha=1.0 - self.prior_decay
            )
            if not bool((self.marginal > 0).all()):
                raise ValueError("DSS marginal distribution became non-positive")
            values = values / (self.num_classes * self.marginal)
            values = values / values.sum(dim=1, keepdim=True)
        if epoch > 0 and not bool(self.history.observed[indices, :epoch].all()):
            raise ValueError("DSS requires one observation per prior epoch")
        previous = self.history.previous(indices, epoch)
        if epoch:
            self.trend_score[indices] += (
                (values[:, None, :] > previous).sum(dim=1)
                - (values[:, None, :] < previous).sum(dim=1)
            ).to(torch.float32)
        self.labels[indices] = targets
        self.current_prediction[indices] = values
        self.history.update(indices, epoch, values)

    def on_cycle_start(self, epoch: int) -> None:
        epoch = int(epoch)
        if not 0 <= epoch < self.total_epochs:
            raise ValueError("DSS epoch is outside total_epochs")
        if self.current_epoch not in {-1, epoch - 1, epoch}:
            raise ValueError("DSS epochs must be processed sequentially")
        self.current_epoch = epoch

    def on_cycle_end(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch != self.current_epoch:
            raise ValueError("DSS cycle end does not match current epoch")
        observed = self.labels >= 0
        if not bool(self.history.observed[observed, epoch].all()):
            raise ValueError("DSS did not observe every known sample this epoch")
        if epoch + 1 < self.warmup_epochs:
            return
        self.selected[observed] = (
            self.current_prediction[observed].argmax(dim=1)
            == self.labels[observed]
        )
        if not self.ccs:
            self.excluded.zero_()
            return
        n = epoch + 1
        variance = n * (n - 1) * (2 * n + 5) / 18
        score = self.trend_score[observed]
        if variance == 0:
            z_score = torch.zeros_like(score)
        else:
            z_score = (score - score.sign()) / variance**0.5
            z_score[score == 0] = 0
        observed_labels = self.labels[observed]
        z_score[torch.arange(observed_labels.numel()), observed_labels] = float("-inf")
        threshold = NormalDist().inv_cdf(1.0 - self.alpha)
        self.excluded[observed] = z_score > threshold

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "num_samples": self.num_samples,
                "num_classes": self.num_classes,
                "total_epochs": self.total_epochs,
                "warmup_epochs": self.warmup_epochs,
                "alpha": self.alpha,
                "prior_decay": self.prior_decay,
                "mda": self.mda,
                "ccs": self.ccs,
            },
            "history": self.history.state_dict(),
            "labels": self.labels.clone(),
            "current_prediction": self.current_prediction.clone(),
            "trend_score": self.trend_score.clone(),
            "marginal": self.marginal.clone(),
            "selected": self.selected.clone(),
            "excluded": self.excluded.clone(),
            "current_epoch": self.current_epoch,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "num_samples": self.num_samples,
            "num_classes": self.num_classes,
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "alpha": self.alpha,
            "prior_decay": self.prior_decay,
            "mda": self.mda,
            "ccs": self.ccs,
        }
        if dict(state.get("config", {})) != expected:
            raise ValueError("DSS selector configuration mismatch")
        self.history.load_state_dict(state["history"])
        tensor_fields = {
            "labels": self.labels,
            "current_prediction": self.current_prediction,
            "trend_score": self.trend_score,
            "marginal": self.marginal,
            "selected": self.selected,
            "excluded": self.excluded,
        }
        for name, destination in tensor_fields.items():
            value = torch.as_tensor(state.get(name))
            if value.shape != destination.shape:
                raise ValueError(f"DSS state shape mismatch for {name}")
            if value.dtype != destination.dtype:
                raise ValueError(f"DSS state dtype mismatch for {name}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"DSS state contains non-finite values for {name}")
            destination.copy_(value)
        self.current_epoch = int(state.get("current_epoch", -1))
        if not -1 <= self.current_epoch < self.total_epochs:
            raise ValueError("DSS current epoch is outside the configured range")


__all__ = ["DSSSelectorState"]
