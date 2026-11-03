"""Extensible research toolbox with optional noisy-label-learning plugins."""

from .registry import Registry


def __getattr__(name):
    if name not in {"Batch", "ExperimentContext", "RunState", "StepResult"}:
        raise AttributeError(name)
    from . import core
    value = getattr(core, name)
    globals()[name] = value
    return value

__all__ = ["Batch", "ExperimentContext", "Registry", "RunState", "StepResult"]
__version__ = "0.1.0"
