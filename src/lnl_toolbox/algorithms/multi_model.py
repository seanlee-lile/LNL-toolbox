from __future__ import annotations

"""Method-neutral multi-network and consistency contracts."""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@dataclass
class ModelGroup:
    models: dict[str, nn.Module]

    def __post_init__(self) -> None:
        if len(self.models) < 2 or any(not name for name in self.models):
            raise ValueError("ModelGroup requires at least two named models")

    def state_dict(self) -> dict[str, Any]:
        return {name: model.state_dict() for name, model in self.models.items()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != set(self.models):
            raise ValueError("model group members do not match checkpoint")
        for name, model in self.models.items():
            model.load_state_dict(state[name])


@dataclass(frozen=True)
class PeerExchangeResult:
    masks: Mapping[str, Tensor]
    metrics: Mapping[str, float]


@runtime_checkable
class PeerExchange(Protocol):
    def exchange(self, losses: Mapping[str, Tensor], indices: Tensor) -> PeerExchangeResult: ...


class SmallLossPeerExchange:
    """Exchange each model's small-loss mask with its peer."""

    def __init__(self, keep_rate: float) -> None:
        if not 0.0 < float(keep_rate) <= 1.0:
            raise ValueError("keep_rate must be in (0, 1]")
        self.keep_rate = float(keep_rate)

    def exchange(self, losses: Mapping[str, Tensor], indices: Tensor) -> PeerExchangeResult:
        if len(losses) < 2:
            raise ValueError("peer exchange requires at least two models")
        masks: dict[str, Tensor] = {}
        for name, loss in losses.items():
            if loss.ndim != 1 or loss.shape != indices.shape:
                raise ValueError("peer losses and indices must align as [B]")
            keep = max(1, int(torch.ceil(torch.tensor(loss.numel() * self.keep_rate)).item()))
            order = torch.argsort(loss.detach(), stable=True)
            masks[name] = torch.zeros_like(loss, dtype=torch.bool).scatter_(0, order[:keep], True)
        names = list(masks)
        exchanged = {names[i]: masks[names[(i + 1) % len(names)]] for i in range(len(names))}
        return PeerExchangeResult(exchanged, {"peer_keep_rate": self.keep_rate})


def consistency_loss(logits_a: Tensor, logits_b: Tensor) -> Tensor:
    if logits_a.shape != logits_b.shape or logits_a.ndim != 2:
        raise ValueError("consistency logits must have matching shape [B, C]")
    return torch.nn.functional.kl_div(
        torch.log_softmax(logits_a, dim=1),
        torch.softmax(logits_b.detach(), dim=1),
        reduction="batchmean",
    )
