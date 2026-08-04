from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import unittest

import yaml

from lnl_toolbox.catalog import paper_by_id, recipe_by_id, validate_config
from lnl_toolbox.cli.main import main
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner


ROOT = Path(__file__).resolve().parents[1]


class DLDCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = ROOT / "configs" / "experiment" / "cifar10_dld_smoke.yaml"
        cls.config = yaml.safe_load(cls.path.read_text(encoding="utf-8"))

    def test_registry_recipe_paper_and_preflight(self) -> None:
        self.assertEqual(resolve_runner(self.config).name, "dld")
        self.assertEqual(validate_config(self.config).name, "dld")
        self.assertEqual(recipe_by_id("cifar10-dld-smoke").runner, "dld")
        self.assertEqual(paper_by_id("dld").implementation_status, "user_ready")

    def test_epoch_override_only_changes_diffusion(self) -> None:
        config = deepcopy(self.config)
        before = deepcopy(config["trainer"])
        apply_epoch_override(config, 7)
        self.assertEqual(config["dld"]["diffusion"]["epochs"], 7)
        self.assertEqual(config["trainer"], before)

    def test_dry_run_discloses_fidelity_and_plan(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["run", "--config", str(self.path), "--dry-run"]), 0)
        text = output.getvalue()
        for value in ("dld", "2 (diffusion)", "K=10", "cosine_distance", "dld_precorrection.npz", "paper_oriented_v1", "DLD inference steps: 5"):
            self.assertIn(value, text)

    def test_protected_local_yaml_is_not_catalogued(self) -> None:
        with self.assertRaises(ValueError):
            recipe_by_id("cifar10-symmetric40-all-e5")


if __name__ == "__main__":
    unittest.main()
