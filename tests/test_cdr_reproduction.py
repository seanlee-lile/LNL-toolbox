import unittest
from pathlib import Path

import torch
import yaml

from lnl_toolbox.plugins.builtin import build_builtin_parameter_update_policy
from lnl_toolbox.training.experiment import (
    _validate_supervised_config,
    build_model,
    build_optimizer,
)


class CDRReproductionTest(unittest.TestCase):
    def test_resnet50_builder_has_expected_structure_and_options(self) -> None:
        model = build_model({
            "name": "resnet50",
            "base_width": 8,
            "stem_padding": 0,
            "initialization": "torch_default",
        }, 10)
        self.assertEqual(
            tuple(len(layer) for layer in (
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )),
            (3, 4, 6, 3),
        )
        self.assertEqual(model.stem[0].padding, (0, 0))
        self.assertEqual(
            tuple(model(torch.randn(1, 3, 32, 32)).shape),
            (1, 10),
        )
        featured = model.forward_with_features(
            torch.randn(1, 3, 32, 32)
        )
        self.assertEqual(tuple(featured.logits.shape), (1, 10))
        self.assertEqual(tuple(featured.features.shape), (1, 256))

    def test_existing_resnet18_defaults_are_not_changed(self) -> None:
        model = build_model(
            {"name": "resnet18", "base_width": 8},
            10,
        )
        self.assertEqual(model.stem[0].padding, (1, 1))
        self.assertEqual(
            tuple(len(layer) for layer in (
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )),
            (2, 2, 2, 2),
        )

    def test_reproduction_config_records_audited_paper_and_data_path(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "experiment"
            / "cifar10_symmetric_cdr_reproduction.yaml"
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["trainer"]["epochs"], 100)
        self.assertEqual(config["data"]["validation_size"], 5000)
        self.assertEqual(config["data"]["validation_split"], {
            "strategy": "random",
            "rng": "numpy_legacy",
        })
        self.assertEqual(config["noise"]["validation_targets"], "noisy")
        self.assertEqual(config["noise"]["sampling"], "transition")
        self.assertEqual(config["noise"]["seed"], config["seed"])
        self.assertEqual(
            config["parameter_update"]["compatibility_mode"],
            "paper",
        )
        self.assertEqual(
            config["parameter_update"]["critical_scope"],
            "all_trainable",
        )
        self.assertEqual(config["optimizer"]["weight_decay"], 0.0)
        self.assertEqual(config["optimizer"]["momentum"], 0.0)
        self.assertEqual(config["model"]["name"], "resnet50")
        self.assertEqual(config["model"]["stem_padding"], 0)
        self.assertEqual(
            config["model"]["initialization"],
            "torch_default",
        )
        _validate_supervised_config(config)
        model = build_model(config["model"], 10)
        optimizer = build_optimizer(model, config["optimizer"])
        policy = build_builtin_parameter_update_policy(
            config["parameter_update"]
        )
        self.assertEqual(policy.compatibility_mode, "paper")
        self.assertEqual(optimizer.param_groups[0]["momentum"], 0.0)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)

    def test_smoke_config_uses_strict_paper_optimizer_contract(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "experiment"
            / "cifar10_symmetric_cdr_smoke.yaml"
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(
            config["parameter_update"]["compatibility_mode"],
            "paper",
        )
        self.assertEqual(config["optimizer"]["momentum"], 0.0)
        self.assertEqual(config["optimizer"]["weight_decay"], 0.0)
        self.assertEqual(
            config["noise"]["rate"],
            config["parameter_update"]["noise_rate"],
        )
        _validate_supervised_config(config)


if __name__ == "__main__":
    unittest.main()
