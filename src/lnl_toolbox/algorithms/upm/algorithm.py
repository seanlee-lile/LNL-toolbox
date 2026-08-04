from __future__ import annotations

"""UPM Stage-2 supervision provider and batch algorithm composition."""

from typing import Any, Mapping

import torch
from torch import nn

from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.core import SoftTargetResult, TargetInput
from lnl_toolbox.noise.estimators import PosteriorSnapshot

from .config import UPMConfig
from .confusing import ConfusingProbabilityState
from .objective import predict_true_posterior
from .posterior import ObservedNoisyProbabilityLookup


class UPMTargetProvider:
    """Compute fixed q and perform the explicit eta PGA before model update."""

    def __init__(
        self,
        *,
        snapshot: PosteriorSnapshot,
        eta_state: ConfusingProbabilityState,
        config: UPMConfig,
        device: torch.device,
    ) -> None:
        self.lookup = ObservedNoisyProbabilityLookup(snapshot, device)
        self.eta_state = eta_state
        self.config = config
        self.last_q: torch.Tensor | None = None
        self.last_psi: torch.Tensor | None = None
        self.last_eta_before: torch.Tensor | None = None

    def resolve(self, target_input: TargetInput) -> SoftTargetResult:
        epoch = int(target_input.metadata.get("epoch", -1))
        if epoch < 0:
            raise ValueError("UPM target provider requires a non-negative epoch")
        clean_probability = torch.softmax(target_input.logits.detach(), dim=1)
        psi = self.lookup.resolve(
            target_input.sample_indices, target_input.noisy_targets,
            dtype=clean_probability.dtype,
        )
        eta = self.eta_state.gather(
            target_input.sample_indices, dtype=clean_probability.dtype
        ).to(clean_probability.device)
        q = predict_true_posterior(
            clean_probability, target_input.noisy_targets, psi, eta
        )
        self.last_q = q.clone()
        self.last_psi = psi.clone()
        self.last_eta_before = eta.clone()
        eta_config = self.config.confusing_probability
        if eta_config.updates_at(epoch):
            self.eta_state.update(
                target_input.sample_indices, q, target_input.noisy_targets, psi,
                learning_rate=eta_config.learning_rate,
                epsilon=eta_config.epsilon,
            )
        return SoftTargetResult(q, target_input.sample_indices)


class UPMAlgorithm(SupervisedClassificationAlgorithm):
    """Stage-2 UPM classifier using the shared supervised optimization step."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module,
        device: torch.device,
        *,
        snapshot: PosteriorSnapshot,
        eta_state: ConfusingProbabilityState,
        config: Mapping[str, Any],
    ) -> None:
        self.method_config = UPMConfig.from_mapping(config)
        self.upm_target_provider = UPMTargetProvider(
            snapshot=snapshot, eta_state=eta_state,
            config=self.method_config, device=device,
        )
        super().__init__(
            model, optimizer, loss, device,
            target_provider=self.upm_target_provider,
        )


__all__ = ["UPMAlgorithm", "UPMTargetProvider"]
