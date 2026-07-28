import unittest
from pathlib import Path

import torch
import yaml

from lnl_toolbox.models.cifar_resnet import cifar_resnet50
from lnl_toolbox.training.experiment import build_model


class CDRReproductionModelTest(unittest.TestCase):
    def test_cifar_resnet50_shape_and_parameter_scope(self) -> None:
        model = cifar_resnet50(num_classes=10, base_width=8)
        output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 10))
        self.assertEqual(model.classifier.in_features, 8 * 8 * 4)
        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 0)

    def test_experiment_builder_registers_generic_resnet50(self) -> None:
        model = build_model({"name": "resnet50", "base_width": 8}, 10)
        self.assertEqual(tuple(model(torch.randn(1, 3, 32, 32)).shape), (1, 10))

    def test_official_structure_options_preserve_generic_defaults(self) -> None:
        default = cifar_resnet50(num_classes=10, base_width=8)
        official = cifar_resnet50(
            num_classes=10,
            base_width=8,
            stem_padding=0,
            initialization="torch_default",
        )
        self.assertEqual(default.stem[0].padding, (1, 1))
        self.assertEqual(official.stem[0].padding, (0, 0))
        self.assertEqual(
            tuple(official(torch.randn(1, 3, 32, 32)).shape),
            (1, 10),
        )

    def test_builder_forwards_official_structure_options(self) -> None:
        model = build_model({
            "name": "resnet50",
            "base_width": 8,
            "stem_padding": 0,
            "initialization": "torch_default",
        }, 10)
        self.assertEqual(model.stem[0].padding, (0, 0))

    def test_reproduction_config_records_paper_horizon_and_official_data_path(self) -> None:
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
        self.assertEqual(config["parameter_update"]["compatibility_mode"], "paper")
        self.assertEqual(config["model"]["stem_padding"], 0)
        self.assertEqual(config["model"]["initialization"], "torch_default")


if __name__ == "__main__":
    unittest.main()
