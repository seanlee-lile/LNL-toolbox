from .numpy_losses import cross_entropy, generalized_cross_entropy

__all__ = ["cross_entropy", "generalized_cross_entropy"]
try:
    from .torch_losses import CrossEntropyLoss
except ImportError:  # Torch remains an optional dependency of the generic core.
    CrossEntropyLoss = None  # type: ignore[assignment]
