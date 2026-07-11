import tempfile
import unittest
from pathlib import Path

from lnl_toolbox.core import Batch, ExperimentContext, StepResult
from lnl_toolbox.engine import run_cycles


class DummyAlgorithm:
    def __init__(self) -> None:
        self.closed = False
        self.total = 0

    def setup(self, context: ExperimentContext) -> None:
        self.context = context

    def on_run_start(self, state) -> None:
        state.metadata["started"] = True

    def on_cycle_start(self, state) -> None:
        pass

    def step(self, batch: Batch, state) -> StepResult:
        self.total += batch.payload
        return StepResult(outputs=self.total, metrics={"last_total": float(self.total)})

    def on_cycle_end(self, state) -> StepResult:
        return StepResult(metadata={"cycle": state.cycle})

    def on_run_end(self, state) -> StepResult:
        return StepResult(metrics={"final_total": float(self.total)})

    def state_dict(self):
        return {"total": self.total}

    def load_state_dict(self, state) -> None:
        self.total = state["total"]

    def close(self) -> None:
        self.closed = True


class CoreLifecycleTest(unittest.TestCase):
    def test_runner_only_depends_on_generic_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = ExperimentContext(Path(directory), config={"demo": True})
            algorithm = DummyAlgorithm()
            state = run_cycles(algorithm, [Batch(2), Batch(3)], 2, context)
        self.assertEqual(state.step, 4)
        self.assertEqual(state.metrics["final_total"], 10.0)
        self.assertTrue(state.metadata["started"])
        self.assertTrue(algorithm.closed)


if __name__ == "__main__":
    unittest.main()

