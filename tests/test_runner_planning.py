from __future__ import annotations

import inspect
import json
from pathlib import Path
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from lnl_toolbox.cli.main import _print_plan
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner


class RunnerPlanningTest(unittest.TestCase):
    def test_supervised_runner_owns_plan_and_budget_override(self) -> None:
        config = {
            "execution": {"runner": "supervised"},
            "data": {"name": "cifar10"},
            "trainer": {"epochs": 10},
            "loss": {"name": "gce"},
        }
        runner = resolve_runner(config)
        self.assertEqual(runner.describe(config).training_budget, "10 epochs")
        apply_epoch_override(config, 3)
        self.assertEqual(config["trainer"]["epochs"], 3)

    def test_multistage_runner_uses_registered_budget_path(self) -> None:
        config = {
            "execution": {"runner": "upm"},
            "method": "upm",
            "data": {"name": "cifar10"},
            "upm": {"main": {"epochs": 100}},
        }
        apply_epoch_override(config, 4)
        self.assertEqual(config["upm"]["main"]["epochs"], 4)

    def test_cli_preview_contains_no_method_specific_dispatch(self) -> None:
        source = inspect.getsource(_print_plan)
        for method in ("upm", "coteaching", "dld", "dividemix", "lend"):
            self.assertNotIn(f'== "{method}"', source)

    def test_coteaching_plan_uses_nested_remember_schedule(self) -> None:
        config = {
            "method": "coteaching",
            "execution": {"runner": "coteaching"},
            "data": {"name": "cifar10"},
            "model": {"name": "cifar_cnn8"},
            "loader": {"batch_size": 128},
            "optimizer": {"name": "adam", "lr": 0.001},
            "noise": {"rate": 0.2},
            "coteaching": {
                "model_count": 2,
                "noise_rate": 0.2,
                "gradual_epochs": 999,
                "remember_schedule": {"gradual_epochs": 10},
            },
            "trainer": {"epochs": 200},
        }
        fields = {
            field.label: field.value
            for field in resolve_runner(config).describe(config).fields
        }
        self.assertEqual(fields["Co-teaching networks"], "2")
        self.assertEqual(fields["Co-teaching batch size"], "128")
        self.assertEqual(fields["Co-teaching optimizer"], "adam")
        self.assertEqual(fields["Co-teaching learning rate"], "0.001")
        self.assertEqual(fields["Co-teaching Tk / gradual epochs"], "10")
        self.assertEqual(fields["Co-teaching tau / noise rate"], "0.2")

    def test_dld_plan_exposes_current_fidelity_and_provenance(self) -> None:
        config = {
            "method": "dld",
            "execution": {"runner": "dld"},
            "data": {"name": "cifar10"},
            "loader": {"batch_size": 256},
            "dld": {
                "fidelity": {
                    "name": "paper_oriented_v2_cosine_similarity",
                    "neighbor_metric": "cosine_similarity",
                    "self_neighbor": "include",
                    "divergence": "kl_ps_to_pw",
                },
                "feature_extractor": {
                    "source": "external_checkpoint",
                    "model": {"name": "resnet18"},
                    "external": {
                        "adapter": "upm_main_best",
                        "checkpoint_sha256": "a" * 64,
                    },
                },
                "precorrection": {"k_neighbors": 50},
                "diffusion": {"epochs": 15, "timesteps": 100},
                "inference": {"steps": 5},
            },
        }
        plan = resolve_runner(config).describe(config)
        fields = {field.label: field.value for field in plan.fields}
        self.assertEqual(plan.training_budget, "15 (diffusion)")
        self.assertEqual(fields["DLD feature source"], "external_checkpoint")
        self.assertEqual(fields["DLD feature model"], "resnet18")
        self.assertEqual(fields["DLD source adapter"], "upm_main_best")
        self.assertEqual(fields["DLD checkpoint identity"], "a" * 64)
        self.assertEqual(fields["DLD neighbors"], "K=50")
        self.assertEqual(fields["DLD neighbor metric"], "cosine_similarity")
        self.assertEqual(fields["DLD self-neighbor"], "include")
        self.assertEqual(fields["DLD divergence"], "kl_ps_to_pw")
        self.assertEqual(fields["DLD timesteps"], "100")
        self.assertEqual(fields["DLD inference steps"], "5")

    def test_dividemix_and_lend_plans_expose_training_contracts(self) -> None:
        dividemix = {
            "method": "dividemix",
            "execution": {"runner": "dividemix"},
            "data": {"name": "cifar10"},
            "loader": {"batch_size": 128},
            "optimizer": {"name": "sgd", "lr": 0.02},
            "dividemix": {
                "warmup": {"epochs": 10},
                "training": {"epochs": 300},
                "gmm": {
                    "threshold": 0.5,
                    "loss_history": {"name": "official_auto"},
                },
                "mixmatch": {"temperature": 0.5, "mixup_alpha": 4.0},
                "objective": {"lambda_u": 25.0, "rampup_epochs": 16},
                "inference": {"ensemble": "official_logits_sum"},
            },
        }
        plan = resolve_runner(dividemix).describe(dividemix)
        fields = {field.label: field.value for field in plan.fields}
        self.assertEqual(plan.training_budget, "10/300/310 (warmup/main/total)")
        self.assertEqual(fields["DivideMix loss history"], "official_auto")
        self.assertEqual(fields["DivideMix lambda_u"], "25.0")
        self.assertEqual(fields["DivideMix ramp-up epochs"], "16")
        self.assertEqual(fields["DivideMix ensemble"], "official_logits_sum")

        lend = {
            "method": "lend",
            "execution": {"runner": "lend"},
            "data": {"name": "cifar10"},
            "loader": {"batch_size": 256},
            "optimizer": {"name": "sgd", "lr": 0.05},
            "lend": {
                "graph": {
                    "k": 8,
                    "gamma": 1.0,
                    "metric": "inner_product",
                    "normalize_features": False,
                    "zero_degree_policy": "self_loop",
                },
                "dilution": {"alpha": 0.99, "policy": "fixed_steps", "steps": 10},
                "history": {"beta": 0.9, "first_observation": "current"},
                "selection": {
                    "rule": "noisy_equals_diluted_argmax",
                    "reduction": "batch_mean",
                    "empty_batch": "skip_update",
                },
                "training": {"epochs": 200},
            },
        }
        plan = resolve_runner(lend).describe(lend)
        fields = {field.label: field.value for field in plan.fields}
        self.assertEqual(plan.training_budget, "200 (LEND)")
        self.assertEqual(fields["LEND batch size"], "256")
        self.assertEqual(fields["LEND optimizer"], "sgd")
        self.assertEqual(fields["LEND learning rate"], "0.05")
        self.assertIn("zero_degree=self_loop", fields["LEND graph"])
        self.assertIn("first=current", fields["LEND history"])

    def test_public_workflow_configs_are_declared_as_package_data(self) -> None:
        root = Path(__file__).resolve().parents[1]
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        data_files = pyproject["tool"]["setuptools"]["data-files"]
        declared = {
            path
            for paths in data_files.values()
            for path in paths
        }
        manifest = json.loads(
            (root / "src/lnl_toolbox/cli/data/recipe_catalog.json").read_text(
                encoding="utf-8"
            )
        )
        catalog_recipes = set(manifest["recipes"]) | set(
            manifest.get("conditional", [])
        )
        self.assertEqual(catalog_recipes - declared, set())
        for path in catalog_recipes:
            self.assertTrue((root / path).is_file(), path)


if __name__ == "__main__":
    unittest.main()
