from __future__ import annotations

"""Paper-oriented VolMinNet transition parameterization."""

import math

import torch
from torch import Tensor, nn


class VolMinTransition(nn.Module):
    """Train only off-diagonal ``w`` in ``A_ij=sigmoid(w_ij), A_ii=1``."""

    convention = "clean_to_noisy_row"
    parameterization = "fixed_diagonal_sigmoid_offdiag"
    normalization_axis = "row"
    initialization = "paper"

    def __init__(self, num_classes: int, *, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        if num_classes < 3:
            raise ValueError("VolMinNet paper initialization requires num_classes >= 3")
        self.num_classes = int(num_classes)
        rows, columns = torch.where(~torch.eye(num_classes, dtype=torch.bool))
        self.register_buffer("off_diagonal_rows", rows)
        self.register_buffer("off_diagonal_columns", columns)
        initial = math.log(1.0 / (num_classes - 2))
        self.off_diagonal_logits = nn.Parameter(
            torch.full((num_classes * (num_classes - 1),), initial, dtype=dtype)
        )

    @property
    def initial_raw_value(self) -> float:
        return math.log(1.0 / (self.num_classes - 2))

    def matrix(self, *, dtype: torch.dtype | None = None) -> Tensor:
        flat = torch.zeros(
            self.num_classes * self.num_classes,
            dtype=self.off_diagonal_logits.dtype,
            device=self.off_diagonal_logits.device,
        )
        positions = self.off_diagonal_rows * self.num_classes + self.off_diagonal_columns
        flat = flat.scatter(0, positions, torch.sigmoid(self.off_diagonal_logits))
        unnormalized = flat.reshape(self.num_classes, self.num_classes)
        unnormalized = unnormalized + torch.eye(
            self.num_classes,
            dtype=unnormalized.dtype,
            device=unnormalized.device,
        )
        transition = unnormalized / unnormalized.sum(dim=1, keepdim=True)
        return transition if dtype is None else transition.to(dtype=dtype)

    def forward(self) -> Tensor:
        return self.matrix()

    def diagnostics(self) -> dict[str, float]:
        transition = self.matrix().detach()
        sign, logabsdet = torch.linalg.slogdet(transition)
        singular_values = torch.linalg.svdvals(transition)
        diagonal = torch.diagonal(transition)
        mask = ~torch.eye(self.num_classes, dtype=torch.bool, device=transition.device)
        off_diagonal = transition[mask]
        return {
            "determinant": float((sign * torch.exp(logabsdet)).cpu().item()),
            "slogdet_sign": float(sign.cpu().item()),
            "logabsdet": float(logabsdet.cpu().item()),
            "minimum_singular_value": float(singular_values[-1].cpu().item()),
            "condition_number": float((singular_values[0] / singular_values[-1]).cpu().item()),
            "diagonal_min": float(diagonal.min().cpu().item()),
            "diagonal_max": float(diagonal.max().cpu().item()),
            "off_diagonal_max": float(off_diagonal.max().cpu().item()),
        }
