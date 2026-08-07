from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from lnl_toolbox.training.adapters import LegacyRunnerAdapter
from lnl_toolbox.training.runners import resolve_runner


class UnifiedResumeTest(unittest.TestCase):
    def test_context_and_resume_checkpoint_are_shared_by_all_adapters(self) -> None:
        spec = resolve_runner({"method": "gce", "data": {"name": "synthetic"}})
        adapter = LegacyRunnerAdapter(spec, method="gce")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pt"
            checkpoint.write_bytes(b"checkpoint")
            context = adapter.prepare(
                config={"method": "gce", "data": {"name": "synthetic"}},
                output_dir=directory,
            )
            adapter.load_checkpoint(context, checkpoint)
            self.assertEqual(context.state["resume_checkpoint"], checkpoint.resolve())

    def test_missing_resume_checkpoint_fails_before_training(self) -> None:
        spec = resolve_runner({"method": "gce", "data": {"name": "synthetic"}})
        adapter = LegacyRunnerAdapter(spec, method="gce")
        with tempfile.TemporaryDirectory() as directory:
            context = adapter.prepare(
                config={"method": "gce", "data": {"name": "synthetic"}},
                output_dir=directory,
            )
            with self.assertRaises(FileNotFoundError):
                adapter.load_checkpoint(context, Path(directory) / "missing.pt")


if __name__ == "__main__":
    unittest.main()
