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


class LENDCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "configs/experiment/cifar10_lend_smoke.yaml"
        cls.config = yaml.safe_load(cls.path.read_text(encoding="utf-8"))

    def test_registry_recipe_paper_and_preflight(self):
        self.assertEqual(resolve_runner(self.config).name, "lend")
        self.assertEqual(validate_config(self.config).name, "lend")
        self.assertEqual(recipe_by_id("cifar10-lend-smoke").runner, "lend")
        paper = paper_by_id("lend")
        self.assertEqual(paper.implementation_status, "user_ready")
        self.assertIn("paper-oriented", " ".join(paper.limitations))

    def test_epoch_override_only_changes_lend_training(self):
        config = deepcopy(self.config)
        trainer = deepcopy(config["trainer"])
        apply_epoch_override(config, 7)
        self.assertEqual(config["lend"]["training"]["epochs"], 7)
        self.assertEqual(config["trainer"], trainer)

    def test_dry_run_discloses_all_fidelity_choices(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["run", "--config", str(self.path), "--dry-run"]), 0)
        text = output.getvalue()
        for value in ("lend", "2 (LEND)", "k=15", "gamma=1.0",
                      "inner_product", "normalize_features=False", "alpha=0.99",
                      "fixed_steps", "steps=3", "beta=0.9", "paper_sum",
                      "skip_update"):
            self.assertIn(value, text)

    def test_preflight_rejects_unresolved_and_small_final_batch(self):
        bad = deepcopy(self.config); bad["lend"]["graph"]["gamma"] = None
        with self.assertRaises((TypeError, ValueError)):
            validate_config(bad)
        bad = deepcopy(self.config); bad["data"]["max_train_samples"] = 68
        with self.assertRaisesRegex(ValueError, "final partial batch"):
            validate_config(bad)
        bad = deepcopy(self.config); bad["model"]["name"] = "opaque_model"
        with self.assertRaisesRegex(ValueError, "feature-aware"):
            validate_config(bad)


if __name__ == "__main__": unittest.main()
