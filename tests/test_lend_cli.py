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
        cls.reproduction_path = ROOT / "configs/experiment/lend_cifar10_reproduction.yaml"
        cls.reproduction_config = yaml.safe_load(
            cls.reproduction_path.read_text(encoding="utf-8")
        )

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
                      "fixed_steps", "steps=3", "beta=0.9", "batch_mean",
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

    def test_full_budget_recipe_is_public_and_valid(self):
        recipe = recipe_by_id("lend-cifar10-reproduction")
        self.assertEqual(recipe.runner, "lend")
        self.assertEqual(recipe.profile, "reproduction")
        self.assertEqual(recipe.configuration_fidelity, "paper_oriented")
        self.assertEqual(recipe.reproduction_status, "not_run")
        self.assertEqual(validate_config(self.reproduction_config).name, "lend")
        paper = paper_by_id("lend")
        formal = [item for item in paper.configs if item.recipe_id == recipe.id]
        self.assertEqual(len(formal), 1)
        self.assertEqual(formal[0].configuration_fidelity, "paper_oriented")
        self.assertEqual(formal[0].reproduction_status, "not_run")

    def test_full_budget_config_and_dry_run_disclose_formal_setting(self):
        config = self.reproduction_config
        self.assertEqual(config["method"], "lend")
        self.assertEqual(config["execution"]["runner"], "lend")
        self.assertEqual(config["data"]["name"], "cifar10")
        for key in ("max_train_samples", "max_validation_samples", "max_test_samples"):
            self.assertNotIn(key, config["data"])
        self.assertEqual(config["noise"]["validation_targets"], "noisy")
        self.assertEqual(config["model"], {"name": "resnet18", "base_width": 64})
        self.assertEqual(config["lend"]["training"]["epochs"], 200)
        self.assertEqual(config["loader"]["batch_size"], 256)
        self.assertEqual(config["optimizer"], {
            "name": "sgd", "lr": 0.05, "momentum": 0.9,
            "weight_decay": 0.0005,
        })
        self.assertEqual(config["scheduler"], {
            "name": "multistep", "milestones": [100], "gamma": 0.1,
        })
        self.assertEqual(config["lend"]["graph"]["k"], 8)
        self.assertEqual(config["lend"]["graph"]["gamma"], 1.0)
        self.assertEqual(config["lend"]["dilution"], {
            "alpha": 0.99, "policy": "fixed_steps", "steps": 10,
        })
        self.assertEqual(config["lend"]["history"]["beta"], 0.9)

        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main([
                "run", "--recipe", "lend-cifar10-reproduction", "--dry-run"
            ]), 0)
        text = output.getvalue()
        for value in (
            "lend", "200 (LEND)", "resnet18", "LEND batch size: 256",
            "k=8", "gamma=1.0", "alpha=0.99", "steps=10", "beta=0.9",
        ):
            self.assertIn(value, text)


if __name__ == "__main__": unittest.main()
