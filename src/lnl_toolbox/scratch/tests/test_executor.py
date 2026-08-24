from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path

import yaml

from lnl_toolbox.scratch import ScratchContext, ScratchExecutionError, execute_recipe, load_recipe
from lnl_toolbox.scratch.blocks.data import prepare_apl_cifar10, prepare_binary_risk_data, prepare_gce_cifar10
from lnl_toolbox.scratch.blocks.evaluation import evaluate_accuracy
from lnl_toolbox.scratch.blocks.forward import softmax_probability
from lnl_toolbox.scratch.blocks.losses import (
    active_passive_composition,
    binary_risk,
    forward_correction,
    gather_target_probability,
    gce_q_formula,
    nce_loss,
    rce_loss,
)
from lnl_toolbox.scratch.blocks.paper_specific.research import jocor_joint_composition, jocor_symmetric_kl
from lnl_toolbox.algorithms.jocor import jocor_joint_scores, symmetric_kl_per_sample
from lnl_toolbox.scratch.blocks.models import create_model
from lnl_toolbox.losses.torch_losses import GeneralizedCrossEntropyLoss
from lnl_toolbox.algorithms.transition_risk import ForwardRiskCorrector
from lnl_toolbox.noise.transition import KnownTransition
from lnl_toolbox.scratch.registry import block


@block(id="test_explode", name="Explode", category="Test")
def _explode(ctx):
    raise RuntimeError("boom")


@block(id="test_increment", name="Increment", category="Test")
def _increment(ctx):
    ctx["count"] = int(ctx.get("count", 0)) + 1


class _PreparedData:
    num_classes = 10

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def loader(self, role, *, epoch: int = 0, batch_size: int = 128, shuffle: bool = True):
        self.calls.append((str(getattr(role, "value", role)), epoch, batch_size))
        return []

    def dataset_for(self, role):
        return ("dataset", str(getattr(role, "value", role)))


class ScratchExecutorTest(unittest.TestCase):
    def test_sequence_loop_and_condition_share_context(self) -> None:
        recipe = {
            "schema_version": 1,
            "name": "toy",
            "steps": [
                {"block": "set_value", "params": {"value": True, "save_as": "enabled"}},
                {
                    "block": "repeat_n",
                    "params": {"count": 3, "index_as": "iteration"},
                    "steps": [{
                        "block": "if_context",
                        "params": {"flag": "enabled"},
                        "steps": [{
                            "block": "set_value",
                            "params": {"value": "ran", "save_as": "result"},
                        }],
                    }],
                },
            ],
        }
        context = execute_recipe(recipe)
        self.assertEqual(context["iteration"], 2)
        self.assertEqual(context["result"], "ran")

    def test_runtime_error_identifies_block(self) -> None:
        recipe = {
            "schema_version": 1,
            "name": "runtime-error",
            "steps": [{"block": "test_explode"}],
        }
        with self.assertRaisesRegex(ScratchExecutionError, "Explode.*boom"):
            execute_recipe(recipe, ScratchContext())

    def test_runtime_limits_cap_formal_loop_without_editing_recipe(self) -> None:
        recipe = {
            "schema_version": 1,
            "name": "limited-loop",
            "settings": {"loader": [(1, 1), (2, 2), (3, 3)]},
            "steps": [{
                "block": "epoch_loop", "params": {"epochs": 3}, "steps": [{
                    "block": "batch_loop", "params": {"loader": "loader"}, "steps": [
                        {"block": "test_increment"},
                    ],
                }],
            }],
        }
        result = execute_recipe(recipe, runtime_limits={"max_epochs": 2, "max_batches": 1})
        self.assertEqual(result["count"], 2)
        self.assertEqual(recipe["steps"][0]["params"]["epochs"], 3)

    def test_refresh_epoch_loader_uses_epoch_seeded_prepared_data(self) -> None:
        prepared = _PreparedData()
        recipe = {
            "schema_version": 1,
            "name": "seeded-loader",
            "steps": [{
                "block": "epoch_loop", "params": {"epochs": 2}, "steps": [{
                    "block": "refresh_epoch_loader",
                    "params": {"data": "prepared", "role": "train", "batch_size": 128, "save_as": "train_loader"},
                }],
            }],
        }
        execute_recipe(recipe, {"prepared": prepared})
        self.assertEqual(prepared.calls, [("train", 0, 128), ("train", 1, 128)])

    def test_gce_data_block_uses_formal_manifest_pipeline(self) -> None:
        prepared = _PreparedData()
        with patch("lnl_toolbox.training.data_service.prepare_experiment_data", return_value=prepared) as prepare:
            context = ScratchContext({"seed": 1, "artifact_dir": "artifacts/scratch/gce-test"})
            prepare_gce_cifar10(context)
        config = prepare.call_args.args[0]
        requirements = prepare.call_args.kwargs["requirements"]
        self.assertEqual(config["data"]["preprocessing"], "gce2018")
        self.assertEqual(config["noise"]["manifest_filename"], "noise_manifest.npz")
        self.assertEqual(requirements.validation_targets, "noisy")
        self.assertTrue(requirements.needs_noise_manifest)
        self.assertEqual(prepared.calls, [("noisy_validation", 0, 128), ("test", 0, 128)])

    def test_apl_data_block_uses_formal_manifest_pipeline(self) -> None:
        prepared = _PreparedData()
        with patch("lnl_toolbox.training.data_service.prepare_experiment_data", return_value=prepared) as prepare:
            context = ScratchContext({"seed": 1, "artifact_dir": "artifacts/scratch/apl-test"})
            prepare_apl_cifar10(context)
        config = prepare.call_args.args[0]
        requirements = prepare.call_args.kwargs["requirements"]
        self.assertEqual(config["data"], {"name": "cifar10", "validation_size": 0, "augment": True, "preprocessing": "standard"})
        self.assertEqual(config["noise"]["sampling"], "per_class")
        self.assertEqual(config["noise"]["validation_targets"], "clean")
        self.assertEqual(requirements.validation_targets, "clean")
        self.assertTrue(requirements.needs_noise_manifest)
        self.assertEqual(prepared.calls, [("test", 0, 128)])

    def test_binary_risk_data_block_uses_formal_manifest_pipeline(self) -> None:
        prepared = _PreparedData()
        with patch("lnl_toolbox.training.data_service.prepare_experiment_data", return_value=prepared) as prepare:
            context = ScratchContext({"artifact_dir": "artifacts/scratch/binary-risk-test"})
            prepare_binary_risk_data(context)
        config = prepare.call_args.args[0]
        requirements = prepare.call_args.kwargs["requirements"]
        self.assertEqual(config["data"], {"name": "synthetic_binary_2d", "train_size": 512, "test_size": 2048, "seed": 2013})
        self.assertEqual(config["noise"]["name"], "binary_asymmetric_rcn")
        self.assertEqual((config["noise"]["rho_positive"], config["noise"]["rho_negative"]), (0.4, 0.4))
        self.assertEqual({role.value for role in requirements.roles}, {"train", "test"})
        self.assertTrue(requirements.needs_noise_manifest)
        self.assertEqual(prepared.calls, [("train", 0, 64), ("test", 0, 64)])

    def test_resnet34_uses_the_formal_cifar_model(self) -> None:
        context = ScratchContext({"device": "cpu"})
        create_model(context, model="resnet34", num_classes=10, base_width=4)
        self.assertEqual(context["model"].__class__.__module__, "lnl_toolbox.models.cifar_resnet")

    def test_evaluation_moves_batches_to_the_model_device(self) -> None:
        import torch

        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
            model.bias.zero_()
        context = ScratchContext({
            "model": model,
            "loader": [(torch.tensor([[3.0, 1.0]]), torch.tensor([0]))],
        })
        evaluate_accuracy(context, model="model", loader="loader", save_as="accuracy")
        self.assertEqual(context["accuracy"], 1.0)

    def test_gce_recipe_matches_its_reproduction_protocol(self) -> None:
        root = Path(__file__).resolve().parents[4]
        recipe = load_recipe(root / "src" / "lnl_toolbox" / "scratch" / "recipes" / "papers" / "gce.yaml")
        config = yaml.safe_load((root / "configs" / "experiment" / "gce_cifar10_noise02_reproduction.yaml").read_text(encoding="utf-8"))
        top = {step["block"]: step.get("params", {}) for step in recipe["steps"]}
        data = top["prepare_gce_cifar10"]
        self.assertEqual((data["validation_size"], data["augment"], data["noise_method"], data["noise_rate"], data["noise_seed"], data["batch_size"]), (config["data"]["validation_size"], config["data"]["augment"], config["noise"]["name"], config["noise"]["rate"], config["noise"]["seed"], config["loader"]["batch_size"]))
        self.assertEqual(top["create_model"], {"model": config["model"]["name"], "num_classes": 10, "base_width": config["model"]["base_width"], "device": "device"})
        self.assertEqual({key: top["create_optimizer"][key] for key in ("optimizer", "lr", "momentum", "nesterov", "weight_decay")}, {"optimizer": config["optimizer"]["name"], "lr": config["optimizer"]["lr"], "momentum": config["optimizer"]["momentum"], "nesterov": config["optimizer"]["nesterov"], "weight_decay": config["optimizer"]["weight_decay"]})
        self.assertEqual({key: top["create_scheduler"][key] for key in ("scheduler", "milestones", "gamma")}, {"scheduler": config["scheduler"]["name"], "milestones": config["scheduler"]["milestones"], "gamma": config["scheduler"]["gamma"]})
        epoch = next(step for step in recipe["steps"] if step["block"] == "epoch_loop")
        self.assertEqual(epoch["params"]["epochs"], config["trainer"]["epochs"])
        self.assertEqual([step["block"] for step in epoch["steps"]], ["refresh_epoch_loader", "batch_loop", "evaluate_accuracy", "track_best_model", "scheduler_step"])

    def test_gce_formula_blocks_match_the_legacy_lq_loss(self) -> None:
        import torch

        logits = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 3.0, 0.0]])
        labels = torch.tensor([2, 1])
        context = ScratchContext({"logits": logits, "labels": labels})
        softmax_probability(context)
        gather_target_probability(context)
        gce_q_formula(context, q=0.7)
        expected = GeneralizedCrossEntropyLoss(q=0.7)(logits, labels)
        self.assertTrue(torch.allclose(context["loss_per_sample"], expected))

    def test_apl_formula_blocks_match_legacy_nce_rce_composition(self) -> None:
        import torch
        from lnl_toolbox.losses.torch_losses import (
            ActivePassiveLoss,
            NormalizedCrossEntropyLoss,
            ReverseCrossEntropyLoss,
        )

        logits = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 3.0, 0.0]])
        labels = torch.tensor([2, 1])
        context = ScratchContext({"logits": logits, "labels": labels})
        nce_loss(context, save_as="active")
        rce_loss(context, log_zero=-9.210340371976184, save_as="passive")
        active_passive_composition(context, active="active", passive="passive", save_as="loss_per_sample")
        expected = ActivePassiveLoss(
            NormalizedCrossEntropyLoss(),
            ReverseCrossEntropyLoss(log_zero=-9.210340371976184),
        )(logits, labels)
        self.assertTrue(torch.allclose(context["loss_per_sample"], expected))

    def test_binary_risk_formula_matches_legacy_natarajan_risk(self) -> None:
        import torch
        from lnl_toolbox.algorithms.binary_risk import NatarajanUnbiasedRisk
        from lnl_toolbox.losses.torch_losses import CrossEntropyLoss

        logits = torch.tensor([[1.0, -2.0], [-1.0, 3.0], [0.5, 0.2]])
        labels = torch.tensor([0, 1, 1])
        context = ScratchContext({"logits": logits, "labels": labels})
        binary_risk(context, rho_positive=0.4, rho_negative=0.4)
        expected = NatarajanUnbiasedRisk(0.4, 0.4).per_sample_risk(
            logits=logits, noisy_targets=labels, base_loss=CrossEntropyLoss()
        )
        self.assertTrue(torch.allclose(context["loss_per_sample"], expected))

    def test_forward_correction_formula_matches_legacy_risk_corrector(self) -> None:
        import torch
        from lnl_toolbox.losses.torch_losses import CrossEntropyLoss

        logits = torch.tensor([[2.0, -1.0, 0.5], [-1.0, 3.0, 0.0]])
        labels = torch.tensor([2, 1])
        transition = torch.tensor([
            [0.8, 0.2, 0.0],
            [0.0, 0.9, 0.1],
            [0.1, 0.0, 0.9],
        ])
        context = ScratchContext({"logits": logits, "labels": labels, "transition": transition})
        forward_correction(context)
        expected = ForwardRiskCorrector().per_sample_risk(
            logits=logits,
            noisy_targets=labels,
            base_loss=CrossEntropyLoss(),
            transition=KnownTransition(transition.numpy()),
        )
        self.assertTrue(torch.allclose(context["loss_per_sample"], expected))

    def test_jocor_formula_blocks_match_legacy_algorithm(self) -> None:
        import torch

        logits_a = torch.tensor([[2.0, -1.0, 0.5], [-1.0, 3.0, 0.0]])
        logits_b = torch.tensor([[1.5, -0.5, 0.1], [-0.4, 2.5, 0.2]])
        labels = torch.tensor([2, 1])
        context = ScratchContext({"logits_a": logits_a, "logits_b": logits_b, "labels": labels})
        from lnl_toolbox.scratch.blocks.losses import per_sample_ce
        per_sample_ce(context, logits="logits_a", labels="labels", save_as="loss_a")
        per_sample_ce(context, logits="logits_b", labels="labels", save_as="loss_b")
        jocor_symmetric_kl(context)
        jocor_joint_composition(context, loss_a="loss_a", loss_b="loss_b", lambda_=0.9)
        expected_agreement = symmetric_kl_per_sample(logits_a, logits_b)
        expected_joint = jocor_joint_scores(context["loss_a"], context["loss_b"], logits_a, logits_b, 0.9)
        self.assertTrue(torch.allclose(context["agreement_per_sample"], expected_agreement))
        self.assertTrue(torch.allclose(context["joint_loss_per_sample"], expected_joint))

    def test_coteaching_small_loss_selection_uses_stable_sample_indices(self) -> None:
        import torch
        from lnl_toolbox.scratch.blocks.selection import small_loss_indices

        context = ScratchContext({
            "losses": torch.tensor([1.0, 1.0, 0.5]),
            "remember_rate": 2 / 3,
            "indices": torch.tensor([9, 3, 7]),
        })
        small_loss_indices(context, input="losses", remember_rate="remember_rate", sample_indices="indices", save_as="selected")
        self.assertEqual(context["selected"].tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
