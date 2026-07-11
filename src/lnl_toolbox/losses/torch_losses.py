from torch import nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """PyTorch cross-entropy exposed as a toolbox loss plugin."""
