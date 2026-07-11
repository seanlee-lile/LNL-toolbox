from __future__ import annotations

from lnl_toolbox.core.algorithm import Algorithm
from lnl_toolbox.core.state import RunState

# Compatibility alias for early experiments. New code should use RunState.
TrainState = RunState

__all__ = ["Algorithm", "RunState", "TrainState"]
