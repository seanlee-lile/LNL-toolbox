from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from lnl_toolbox.training.adapters import LegacyRunnerAdapter
from lnl_toolbox.training.interfaces import RunResult
from lnl_toolbox.training.runners import resolve_runner


class UnifiedCheckpointContractTest(unittest.TestCase):
    def test_checkpoint_boundary_is_explicit(self) -> None:
        spec = resolve_runner({"method": "gce", "data": {"name": "synthetic"}})
        adapter = LegacyRunnerAdapter(spec, method="gce")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "last.pt").write_bytes(b"last")
            (root / "best.pt").write_bytes(b"best")
            result = RunResult.from_run_dir(root)
            self.assertEqual(adapter.save_checkpoint(result, "last"), root / "last.pt")
            self.assertEqual(adapter.save_checkpoint(result, "best"), root / "best.pt")
            with self.assertRaises(ValueError):
                adapter.save_checkpoint(result, "phase")

    def test_missing_checkpoint_is_reported_by_the_common_contract(self) -> None:
        spec = resolve_runner({"method": "gce", "data": {"name": "synthetic"}})
        adapter = LegacyRunnerAdapter(spec, method="gce")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                adapter.save_checkpoint(RunResult.from_run_dir(directory))


if __name__ == "__main__":
    unittest.main()
