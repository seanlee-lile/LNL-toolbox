from __future__ import annotations

"""Official SevenCNN classifier used by the standalone SED+FINE runner."""

import torch
from torch import nn
from .feature_output import FeatureOutput


class _MLPHead(nn.Module):
    """The MLPHead used by the official FINE implementation."""

    def __init__(
        self,
        input_size: int,
        scale: float,
        output_size: int,
        *,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden_size = round(float(scale) * input_size)
        if activation == "relu":
            nonlinear: nn.Module = nn.ReLU(inplace=True)
        elif activation == "tanh":
            nonlinear = nn.Tanh()
        else:
            raise ValueError(f"unsupported FINE head activation: {activation}")
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nonlinear,
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class FineSevenCNN(nn.Module):
    """Official FINE SevenCNN with a reusable feature-output contract.

    At ``base_width=64`` the channels are exactly ``64, 128, 196, 16`` as in
    the authors' ``model/SevenCNN.py``. Smaller widths are retained for CPU
    tests and scale the same architecture without changing the default.
    """

    def __init__(
        self,
        num_classes: int = 100,
        *,
        base_width: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if num_classes < 2 or base_width <= 0:
            raise ValueError("num_classes and base_width must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        del dropout  # retained for configuration/API compatibility
        scale = float(base_width) / 64.0
        c1 = max(1, int(round(64 * scale)))
        c2 = max(1, int(round(128 * scale)))
        c3 = max(1, int(round(196 * scale)))
        c4 = max(1, int(round(16 * scale)))

        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels, momentum=0.1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels, momentum=0.1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

        self.block1 = block(3, c1)
        self.block2 = block(c1, c2)
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(c3, c4, 3, padding=1),
            nn.BatchNorm2d(c4, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.feature_size = c4 * 16
        self.classifier = _MLPHead(self.feature_size, 2.0, num_classes)
        self.probability_head = nn.Sequential(
            _MLPHead(
                self.feature_size,
                1.0,
                num_classes,
                activation="tanh",
            ),
            nn.Sigmoid(),
        )
        # The official SevenCNN leaves convolution/BatchNorm modules at their
        # PyTorch defaults.  Its two MLP heads, however, call ``init_weights``
        # with the He option, which applies Kaiming-normal initialization to
        # their Linear layers and zeros their biases.  Keep that distinction
        # explicit so the reusable model remains source-faithful.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_with_features(self, inputs: torch.Tensor) -> FeatureOutput:
        values = self.block3(self.block2(self.block1(inputs)))
        features = values.flatten(1)
        return FeatureOutput(self.classifier(features), features)

    def forward_outputs(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self.block3(self.block2(self.block1(inputs)))
        features = values.flatten(1)
        return {
            "logits": self.classifier(features),
            "prob": self.probability_head(features),
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(inputs).logits


__all__ = ["FineSevenCNN"]
