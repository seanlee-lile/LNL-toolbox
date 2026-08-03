import unittest
from pathlib import Path

import yaml

import torch

from lnl_toolbox.algorithms.transition_risk import BackwardRiskCorrector, ForwardRiskCorrector
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.models.cifar_resnet import cifar_resnet14, cifar_resnet32
from lnl_toolbox.noise.transition import KnownTransition
from lnl_toolbox.plugins.builtin import build_builtin_pipeline


class LossCorrectionReproductionTest(unittest.TestCase):
    def test_formal_cifar10_configuration_uses_paper_schedule(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (root / "configs" / "experiment" / "loss_correction_cifar10_asymmetric04.yaml")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(config["trainer"]["epochs"], 120)
        self.assertEqual(config["scheduler"]["milestones"], [40, 80])
        self.assertEqual(config["loader"]["batch_size"], 128)
        self.assertEqual(config["model"]["name"], "resnet32")
        self.assertEqual(config["noise"]["name"], "class_conditional")
        self.assertEqual(
            config["pipeline"]["transition_estimator"]["name"], "known"
        )

    def test_known_transition_forward_and_backward_paths(self) -> None:
        logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
        targets = torch.tensor([0, 1])
        transition = KnownTransition(torch.eye(2).numpy())
        criterion = CrossEntropyLoss()
        forward = ForwardRiskCorrector().per_sample_risk(
            logits=logits, noisy_targets=targets, base_loss=criterion, transition=transition
        )
        backward = BackwardRiskCorrector().per_sample_risk(
            logits=logits, noisy_targets=targets, base_loss=criterion, transition=transition
        )
        self.assertTrue(torch.allclose(forward, backward, atol=1e-6))

    def test_paper_depth_constructors_return_class_logits(self) -> None:
        inputs = torch.randn(2, 3, 32, 32)
        self.assertEqual(cifar_resnet14()(inputs).shape, (2, 10))
        self.assertEqual(cifar_resnet32()(inputs).shape, (2, 10))

    def test_full_experiment_config_routes_correction_through_pipeline(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (root / "configs" / "experiment" / "cifar10_foundation_smoke.yaml").read_text(
                encoding="utf-8"
            )
        )
        pipeline = build_builtin_pipeline(config["pipeline"])
        self.assertIsInstance(pipeline.risk_corrector, ForwardRiskCorrector)
        self.assertIsNotNone(pipeline.transition_estimator)
        self.assertIsInstance(
            build_builtin_pipeline({
                "name": "standard_noisy_erm",
                "transition_estimator": {"name": "anchor"},
                "risk_corrector": {"name": "backward"},
            }).risk_corrector,
            BackwardRiskCorrector,
        )
        fragment = yaml.safe_load(
            (root / "configs" / "algorithm" / "loss_correction.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(fragment["name"], "forward")
        self.assertNotIn("pipeline", fragment)


if __name__ == "__main__":
    unittest.main()
