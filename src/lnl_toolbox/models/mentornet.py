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


class OfficialMentorNet(nn.Module):
    """PyTorch equivalent of the official MentorNet CIFAR graph.

    The released TensorFlow graph feeds one ``[loss, loss_difference]``
    timestep to two ``BasicLSTMCell(1)`` instances.  It does not treat
    neighbouring samples in a minibatch as a temporal sequence.
    """

    implementation = "official"

    def __init__(
        self,
        *,
        label_embedding_dim: int = 2,
        epoch_embedding_dim: int = 5,
        dense_size: int = 20,
    ) -> None:
        super().__init__()
        if min(label_embedding_dim, epoch_embedding_dim, dense_size) <= 0:
            raise ValueError("official MentorNet dimensions must be positive")
        self.label_embedding_dim = int(label_embedding_dim)
        self.epoch_embedding_dim = int(epoch_embedding_dim)
        self.dense_size = int(dense_size)
        self.loss_encoder = nn.LSTM(
            input_size=2,
            hidden_size=1,
            batch_first=True,
            bidirectional=True,
        )
        self.label_embedding = nn.Embedding(2, self.label_embedding_dim)
        self.epoch_embedding = nn.Embedding(
            100, self.epoch_embedding_dim, _freeze=True
        )
        feature_size = 2 + self.label_embedding_dim + self.epoch_embedding_dim
        self.hidden = nn.Linear(feature_size, self.dense_size)
        self.output = nn.Linear(self.dense_size, 1)

    def architecture(self) -> dict[str, int | str]:
        return {
            "implementation": self.implementation,
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
        expected = (count,)
        if any(value.shape != expected for value in (
            losses, loss_differences, labels, epoch_percentages
        )):
            raise ValueError("official MentorNet features must have shape [B]")
        if count == 0:
            raise ValueError("official MentorNet batch must not be empty")
        sequence = torch.stack((losses, loss_differences), dim=-1).unsqueeze(1)
        _, (hidden, _) = self.loss_encoder(sequence)
        encoded = torch.cat((hidden[0], hidden[1]), dim=1)
        label_features = self.label_embedding(labels.long().clamp(0, 1))
        epoch_features = self.epoch_embedding(
            epoch_percentages.long().clamp(0, 99)
        )
        features = torch.cat((encoded, label_features, epoch_features), dim=1)
        return torch.sigmoid(self.output(torch.tanh(self.hidden(features)))).squeeze(1)


def build_mentor_model(architecture: dict[str, object] | None = None) -> nn.Module:
    """Build a legacy or official MentorNet from artifact metadata."""

    values = dict(architecture or {})
    implementation = str(values.pop("implementation", "legacy")).strip().lower()
    if implementation == "official":
        allowed = {"label_embedding_dim", "epoch_embedding_dim", "dense_size"}
        return OfficialMentorNet(**{
            key: int(value) for key, value in values.items() if key in allowed
        })
    allowed = {
        "num_labels", "hidden_size", "sequence_length",
        "label_embedding_dim", "epoch_embedding_dim", "dense_size",
    }
    return MentorNet(**{
        key: int(value) for key, value in values.items() if key in allowed
    })


__all__ = ["MentorNet", "OfficialMentorNet", "build_mentor_model"]
