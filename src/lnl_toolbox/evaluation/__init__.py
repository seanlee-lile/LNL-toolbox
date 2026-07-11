from .metrics import accuracy, selection_precision_recall

__all__ = ["accuracy", "selection_precision_recall"]
try:
    from .classification import evaluate_classification
except ImportError:
    evaluate_classification = None  # type: ignore[assignment]
