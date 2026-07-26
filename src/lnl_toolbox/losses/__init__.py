from .numpy_losses import cross_entropy, generalized_cross_entropy

__all__ = ["cross_entropy", "generalized_cross_entropy"]
try:
    from .torch_losses import (
        ActivePassiveLoss,
        CrossEntropyLoss,
        GeneralizedCrossEntropyLoss,
        MeanAbsoluteErrorLoss,
        NormalizedCrossEntropyLoss,
        ReverseCrossEntropyLoss,
        loss_for_all_targets,
        validate_per_sample_loss,
    )
except ImportError:  # Torch remains an optional dependency of the generic core.
    CrossEntropyLoss = None  # type: ignore[assignment]
    GeneralizedCrossEntropyLoss = None  # type: ignore[assignment]
    NormalizedCrossEntropyLoss = None  # type: ignore[assignment]
    MeanAbsoluteErrorLoss = None  # type: ignore[assignment]
    ReverseCrossEntropyLoss = None  # type: ignore[assignment]
    ActivePassiveLoss = None  # type: ignore[assignment]
    validate_per_sample_loss = None  # type: ignore[assignment]
    loss_for_all_targets = None  # type: ignore[assignment]
else:
    __all__ += [
        "CrossEntropyLoss",
        "GeneralizedCrossEntropyLoss",
        "NormalizedCrossEntropyLoss",
        "MeanAbsoluteErrorLoss",
        "ReverseCrossEntropyLoss",
        "ActivePassiveLoss",
        "validate_per_sample_loss",
        "loss_for_all_targets",
    ]
