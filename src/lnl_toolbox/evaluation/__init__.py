from .metrics import accuracy, selection_precision_recall
from .run_comparison import collect_run_results, compare_runs, write_report

__all__ = [
    "accuracy",
    "selection_precision_recall",
    "collect_run_results",
    "compare_runs",
    "write_report",
]
try:
    from .classification import evaluate_classification
except ImportError:
    evaluate_classification = None  # type: ignore[assignment]
