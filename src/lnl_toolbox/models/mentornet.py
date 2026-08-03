from __future__ import annotations

"""PyTorch MentorNet used as a frozen curriculum model."""

import torch
from torch import nn


class MentorNet(nn.Module):
    """Encode loss feedback, label, and epoch features into sample weights."""

    def __init__(
        self,
        num_labels: int = 1,
        *,
        hidden_size: int = 10,
        sequence_length: int = 10,
        label_embedding_dim: int = 2,
        epoch_embedding_dim: int = 5,
        dense_size: int = 20,
    ) -> None:
        super().__init__()
        if min(num_labels, hidden_size, sequence_length) <= 0:
            raise ValueError("MentorNet dimensions must be positive")
        self.num_labels = int(num_labels)
        self.hidden_size = int(hidden_size)
        self.sequence_length = int(sequence_length)
        self.label_embedding_dim = int(label_embedding_dim)
        self.epoch_embedding_dim = int(epoch_embedding_dim)
        self.dense_size = int(dense_size)
        self.loss_encoder = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.label_embedding = nn.Embedding(
            self.num_labels, self.label_embedding_dim
        )
        self.epoch_embedding = nn.Embedding(100, self.epoch_embedding_dim)
        combined = (
            2 * self.hidden_size
            + self.label_embedding_dim
            + self.epoch_embedding_dim
        )
        self.hidden = nn.Linear(combined, self.dense_size)
        self.output = nn.Linear(self.dense_size, 1)

    def architecture(self) -> dict[str, int]:
        return {
            "num_labels": self.num_labels,
            "hidden_size": self.hidden_size,
            "sequence_length": self.sequence_length,
            "label_embedding_dim": self.label_embedding_dim,
            "epoch_embedding_dim": self.epoch_embedding_dim,
            "dense_size": self.dense_size,
        }

    def forward(
        self,
        losses: torch.Tensor,
        loss_differences: torch.Tensor,
        labels: torch.Tensor,
        epoch_percentages: torch.Tensor,
    ) -> torch.Tensor:
        count = int(losses.numel())
        if losses.shape != (count,) or loss_differences.shape != (count,):
            raise ValueError("loss features must be aligned vectors")
        if labels.shape != (count,) or epoch_percentages.shape != (count,):
            raise ValueError("discrete MentorNet features must have shape [B]")
        if count == 0:
            raise ValueError("MentorNet batch must not be empty")
        numeric = torch.stack((losses, loss_differences), dim=1)
        padding = (-count) % self.sequence_length
        if padding:
            numeric = torch.cat(
                (numeric, numeric.new_zeros((padding, 2))), dim=0
            )
        encoded, _ = self.loss_encoder(
            numeric.reshape(-1, self.sequence_length, 2)
        )
        encoded = encoded.reshape(-1, 2 * self.hidden_size)[:count]
        combined = torch.cat(
            (
                encoded,
                self.label_embedding(labels.long()),
                self.epoch_embedding(epoch_percentages.long().clamp(0, 99)),
            ),
            dim=1,
        )
        return torch.sigmoid(self.output(torch.tanh(self.hidden(combined)))).squeeze(1)
