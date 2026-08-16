from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import unittest

from lnl_toolbox.catalog import (
    load_recipe_config,
    paper_by_id,
    recipe_by_id,
    validate_config,
)
from lnl_toolbox.cli import main as cli_main
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner


def _config():
    stage = {
        "epochs": 2, "model": {"name": "tiny_cnn"},
        "optimizer": {"name": "sgd", "lr": 0.01},
        "scheduler": {"name": "none"},
    }
    return {
        "method": "upm", "execution": {"runner": "upm"},
        "data": {"name": "cifar10"}, "loader": {"batch_size": 2},
        "noise": {"name": "symmetric", "rate": 0.2, "validation_targets": "noisy"},
        "evaluation": {"selection_split": "validation"},
        "trainer": {"device": "cpu"},
        "upm": {
            "stage1": {**stage, "best_metric": "noisy_validation_accuracy"},
            "psi": {"source": "stage1_best", "split": "train", "augmentation": False},
            "main": {**stage, "initialization": "fresh"},
            "confusing_probability": {
                "initial_value": 0.01, "learning_rate": 0.1, "epsilon": 1e-4,
                "update_start_epoch": 0, "update_interval_epochs": 1,
            },
        },
    }


class UPMCliTest(unittest.TestCase):
    def test_registry_recipe_paper_and_validation(self) -> None:
        self.assertEqual(resolve_runner(_config()).name, "upm")
        self.assertEqual(validate_config(_config()).name, "upm")
        recipe = recipe_by_id("cifar10-upm-smoke")
        self.assertEqual(recipe.runner, "upm")
        paper = paper_by_id("upm")
        self.assertEqual(paper.implementation_status, "user_ready")

    def test_full_run_recipe_uses_current_upm_schema(self) -> None:
        recipe = recipe_by_id("upm-cifar10-reproduction")
        self.assertEqual(recipe.profile, "reproduction")
        self.assertEqual(recipe.runner, "upm")
        self.assertEqual(recipe.configuration_fidelity, "engineering")
        self.assertEqual(
            recipe_by_id("cifar10-upm-smoke").configuration_fidelity,
            "smoke",
        )
        config = load_recipe_config(recipe)
        self.assertEqual(validate_config(config).name, "upm")
        self.assertEqual(config["method"], "upm")
        self.assertEqual(config["execution"]["runner"], "upm")
        self.assertEqual(config["data"]["name"], "cifar10")
        self.assertNotIn("max_train_samples", config["data"])
        self.assertEqual(config["noise"]["name"], "symmetric")
        self.assertEqual(config["noise"]["rate"], 0.4)
        self.assertEqual(config["noise"]["validation_targets"], "noisy")
        self.assertEqual(config["upm"]["stage1"]["epochs"], 160)
        self.assertEqual(config["upm"]["main"]["epochs"], 160)
        self.assertEqual(config["upm"]["main"]["model"], {
            "name": "resnet18", "base_width": 16,
        })

    def test_epoch_override_only_changes_main(self) -> None:
        config = _config()
        stage1 = deepcopy(config["upm"]["stage1"])
        apply_epoch_override(config, 7)
        self.assertEqual(config["upm"]["main"]["epochs"], 7)
        self.assertEqual(config["upm"]["stage1"], stage1)
        self.assertNotIn("epochs", config["trainer"])

    def test_dry_run_reports_stages_psi_and_eta(self) -> None:
        config_path = Path("configs/experiment/cifar10_upm_smoke.yaml").resolve()
        output = io.StringIO()
        with redirect_stdout(output):
            result = cli_main.main([
                "run", "--config", str(config_path), "--dry-run"
            ])
        self.assertEqual(result, 0)
        text = output.getvalue()
        for expected in (
            "upm", "2/2 (stage1/main)", "UPM psi source: stage1_best",
            "UPM eta initial value: 0.01",
            "UPM eta update start epoch: 0",
            "UPM eta update interval: 1",
        ):
            self.assertIn(expected, text)

    def test_protected_yaml_is_not_a_recipe(self) -> None:
        with self.assertRaises(ValueError):
            recipe_by_id("cifar10-symmetric40-all-e5")


if __name__ == "__main__":
    unittest.main()
