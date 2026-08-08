from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from lnl_toolbox import toolbox
from lnl_toolbox.catalog import load_papers
from lnl_toolbox.training.interfaces import RunResult
from lnl_toolbox.training.adapters import (
    AplRunner,
    CdrRunner,
    GceRunner,
    LossCorrectionRunner,
    NativeMultiModelRunner,
    NativeSingleStageRunner,
    NativeStagedRunner,
    AlternatingRunner,
    DiffusionRunner,
    GraphStateRunner,
)


class UnifiedInterfaceTest(unittest.TestCase):
    def test_all_catalogued_papers_resolve_to_the_same_public_methods(self) -> None:
        required = {"prepare", "fit", "evaluate", "save_checkpoint", "load_checkpoint"}
        for paper in load_papers():
            runner = toolbox.get(paper.id)
            self.assertTrue(required.issubset(set(dir(runner))), paper.id)
            self.assertTrue(hasattr(runner, "spec"), paper.id)

    def test_run_result_has_stable_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = RunResult.from_run_dir(Path(directory))
            self.assertEqual(result.run_dir, Path(directory).resolve())
            self.assertIsInstance(result.final_metrics, dict)
            self.assertIsNone(result.last_checkpoint)

    def test_first_single_stage_methods_use_native_lifecycle_runners(self) -> None:
        expected = {
            "gce": GceRunner,
            "apl": AplRunner,
            "cdr": CdrRunner,
            "loss_correction": LossCorrectionRunner,
        }
        for method, runner_class in expected.items():
            self.assertIsInstance(toolbox.get(method), runner_class)

    def test_jocor_uses_native_multi_model_lifecycle_runner(self) -> None:
        self.assertIsInstance(toolbox.get("jocor"), NativeMultiModelRunner)

    def test_peer_methods_use_the_same_native_multi_model_contract(self) -> None:
        for method in ("coteaching", "cnlcu"):
            self.assertIsInstance(toolbox.get(method), NativeMultiModelRunner)

    def test_second_native_migration_group_uses_shared_lifecycle_runners(self) -> None:
        for method in ("fine", "ca2c", "importance_reweighting"):
            self.assertIsInstance(toolbox.get(method), NativeStagedRunner)
        self.assertIsInstance(toolbox.get("binary-risk"), NativeSingleStageRunner)

    def test_specialized_lifecycle_methods_use_their_native_adapters(self) -> None:
        self.assertIsInstance(toolbox.get("dld"), DiffusionRunner)
        self.assertIsInstance(toolbox.get("upm"), AlternatingRunner)
        self.assertIsInstance(toolbox.get("lend"), GraphStateRunner)

    def test_custom_runner_registration_uses_same_lookup(self) -> None:
        class DummyRunner:
            def __init__(self, *, name, metadata):
                self.name = name
                self.metadata = metadata

        toolbox.register("dummy-unified", DummyRunner, lifecycle="step", checkpoint_unit="step")
        runner = toolbox.get("dummy_unified")
        self.assertEqual(runner.name, "dummy_unified")
        self.assertEqual(runner.metadata["checkpoint_unit"], "step")


if __name__ == "__main__":
    unittest.main()
