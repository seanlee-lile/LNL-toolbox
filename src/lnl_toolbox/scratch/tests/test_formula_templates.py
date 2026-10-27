from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from lnl_toolbox.scratch import execute_recipe, load_recipe, validate_recipe
from lnl_toolbox.scratch.registry import get_block


ROOT = Path(__file__).resolve().parents[1]


def _batch_blocks(recipe: dict) -> list[str]:
    epoch = next(step for step in recipe["steps"] if step["block"] == "epoch_loop")
    batch = next(step for step in epoch["steps"] if step["block"] == "batch_loop")
    return [step["block"] for step in batch["steps"]]


class ScratchFormulaTemplateTest(unittest.TestCase):
    def test_gce_is_expanded_to_formula_blocks(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "gce.yaml")
        validate_recipe(recipe)
        epoch = next(step for step in recipe["steps"] if step["block"] == "epoch_loop")
        self.assertEqual(epoch["params"]["epochs"], 120)
        self.assertEqual(
            _batch_blocks(recipe),
            [
                "get_batch",
                "move_batch_to_device",
                "zero_grad",
                "forward",
                "softmax",
                "gather_by_label",
                "clamp_min",
                "elementwise_power",
                "affine_transform",
                "mean_loss",
                "backward",
                "optimizer_step",
            ],
        )
        self.assertNotIn("gce_loss", _batch_blocks(recipe))

    def test_coteaching_exposes_peer_selection_and_cross_update(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "coteaching.yaml")
        validate_recipe(recipe)
        epoch = next(step for step in recipe["steps"] if step["block"] == "epoch_loop")
        self.assertEqual(epoch["steps"][0]["block"], "refresh_epoch_loader")
        self.assertEqual(epoch["steps"][1]["block"], "linear_rate_schedule")
        self.assertEqual(
            _batch_blocks(recipe)[8:],
            [
                "select_lowest_scores",
                "select_lowest_scores",
                "select_by_indices",
                "mean_loss",
                "select_by_indices",
                "mean_loss",
                "backward",
                "optimizer_step",
                "backward",
                "optimizer_step",
            ],
        )

    def test_coteaching_recipe_preserves_peer_wiring_with_custom_parameters(self) -> None:
        from copy import deepcopy

        recipe = load_recipe(ROOT / "recipes" / "papers" / "coteaching.yaml")
        original = deepcopy(recipe)
        validate_recipe(recipe)
        steps = recipe["steps"]
        models = [step for step in steps if step["block"] == "create_model"]
        optimizers = [step for step in steps if step["block"] == "create_optimizer"]
        slots = {step["params"]["save_as"] for step in models}
        self.assertEqual(len(models), 2)
        self.assertEqual(len(slots), 2)
        self.assertEqual({step["params"]["model"] for step in optimizers}, slots)
        for model in ("cnn8", "cnn9"):
            changed = deepcopy(recipe)
            for step in changed["steps"]:
                if step["block"] == "create_model":
                    step["params"]["model"] = model
                elif step["block"] == "create_optimizer":
                    step["params"]["lr"] = 0.003
                elif step["block"] == "epoch_loop":
                    step["params"]["epochs"] = 3
            validate_recipe(changed)
        self.assertEqual(recipe, original)
        epoch = next(step for step in steps if step["block"] == "epoch_loop")
        self.assertEqual([step["block"] for step in epoch["steps"][-2:]], ["evaluate_peer_ensemble", "track_best_state"])

    def test_cnns_match_shared_model_initialization_and_outputs(self) -> None:
        import torch
        from lnl_toolbox.models.cifar_cnn import CNN8, CNN9
        from lnl_toolbox.scratch.blocks.models import create_model
        from lnl_toolbox.scratch.context import ScratchContext

        for name, factory in (("cnn8", CNN8), ("cnn9", CNN9)):
            with self.subTest(model=name):
                torch.manual_seed(17)
                context = ScratchContext(device="cpu")
                create_model(context, model=name, num_classes=3, device="device")
                actual = context["model"].eval()
                torch.manual_seed(17)
                expected = factory(num_classes=3).eval()
                self.assertEqual(set(actual.state_dict()), set(expected.state_dict()))
                for key, value in expected.state_dict().items():
                    torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
                inputs = torch.randn(2, 3, 32, 32)
                with torch.no_grad():
                    torch.testing.assert_close(actual(inputs), expected(inputs), rtol=0, atol=0)

    def test_classwise_fixture_keeps_complete_class_groups(self) -> None:
        from copy import deepcopy
        from lnl_toolbox.scratch.data_runtime import load_sources, split_source

        for classes in (2, 3, 10):
            plan = {"seed": 5, "data": {"name": "synthetic", "train_size": 50000},
                    "semantics": {"num_classes": classes},
                    "split": {"strategy": "classwise_legacy", "rng": "numpy_legacy", "validation_size": 5000}}
            original = deepcopy(plan)
            effective, train, validation, _ = load_sources(plan, {"_runtime_limits": {"fixture": True}})
            train, validation = split_source(train, validation, effective)
            self.assertEqual({sample.observed_target for sample in train.samples}, set(range(classes)))
            self.assertEqual({sample.observed_target for sample in validation.samples}, set(range(classes)))
            self.assertEqual(len(validation.samples), effective["split"]["validation_size"])
            self.assertEqual(plan, original)

    def test_late_classwise_split_works_only_with_explicit_fixture_adaptation(self) -> None:
        import torch
        from lnl_toolbox.scratch.data_runtime import ScratchSample, ScratchSplit, split_source

        recipe = {"schema_version": 1, "name": "classwise_fixture", "steps": [
            {"block": "set_seed", "params": {"seed": 1}},
            {"block": "load_dataset", "params": {"dataset": "cifar10"}},
            {"block": "inspect_dataset_semantics"},
            {"block": "create_dataset_split", "params": {
                "validation_size": 5000, "split_strategy": "classwise_legacy", "split_rng": "numpy_legacy"}},
        ]}
        context = execute_recipe(recipe, runtime_limits={"fixture": True})
        self.assertEqual(len(context["validation_split"].samples), context["num_classes"])
        self.assertEqual(context["data_plan"]["split"]["validation_size"], len(context["validation_split"].samples))
        source = ScratchSplit("real", "train", tuple(
            ScratchSample(torch.zeros(2), i, i % 3, i % 3) for i in range(6)
        ), 3)
        with self.assertRaisesRegex(ValueError, "class counts compatible"):
            split_source(source, None, {"split": {
                "strategy": "classwise_legacy", "rng": "numpy_legacy", "validation_size": 1}})

    def test_cnlcu_persists_peer_selection_counts(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "cnlcu.yaml")
        validate_recipe(recipe)
        blocks = _batch_blocks(recipe)
        first = blocks.index("select_lowest_scores")
        self.assertEqual(
            blocks[first:first + 12],
            [
                "select_lowest_scores",
                "select_lowest_scores",
                "indices_to_mask",
                "indices_to_mask",
                "mask_to_indices",
                "mask_to_indices",
                "ones_like",
                "ones_like",
                "indexed_increment",
                "indexed_increment",
                "indexed_read",
                "indexed_read",
            ],
        )


    def test_lend_recipe_selects_and_restores_noisy_validation_best(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "lend.yaml")
        validate_recipe(recipe)
        config_path = ROOT.parents[2] / "configs" / "experiment" / "lend_cifar10_reproduction.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        epoch = next(step for step in recipe["steps"] if step["block"] == "epoch_loop")
        self.assertEqual(epoch["params"]["epochs"], config["lend"]["training"]["epochs"])
        self.assertEqual(
            [step["block"] for step in epoch["steps"][-3:]],
            ["evaluate_accuracy", "track_best_model", "scheduler_step"],
        )
        self.assertEqual(epoch["steps"][-2]["params"]["metric_name"], "validation_accuracy")
        self.assertEqual(recipe["steps"][-2], {"block": "restore_best_model", "params": {"model": "model", "state": "best_model_state"}})
        self.assertEqual(recipe["steps"][-1]["params"]["loader"], "test_loader")
        batch = next(step for step in epoch["steps"] if step["block"] == "batch_loop")
        blocks = [step["block"] for step in batch["steps"]]
        graph = blocks.index("pairwise_similarity")
        self.assertEqual(blocks[graph:graph + 5], ["pairwise_similarity", "topk_neighborhood", "neighbor_edge_weights", "normalize_graph", "propagate_labels"])
        self.assertNotIn("lend_build_neighbor_graph", blocks)







    def test_formula_metadata_is_complete(self) -> None:
        common = (
            "softmax",
            "gather_by_label",
            "clamp_min",
            "elementwise_power",
            "affine_transform",
            "symmetric_kl",
            "linear_rate_schedule",
            "select_lowest_scores",
            "select_by_indices",
            "mean_by_indices",
        )
        for block_id in common:
            definition = get_block(block_id)
            self.assertTrue(definition.formula, block_id)
            self.assertTrue(definition.formula_ref, block_id)
        for block_id in (
            "nce_loss",
            "rce_loss",
            "binary_risk",
        ):
            definition = get_block(block_id)
            self.assertTrue(definition.formula, block_id)
            self.assertTrue(definition.formula_ref, block_id)
            self.assertTrue(definition.paper, block_id)

    def test_formula_templates_execute_smoke(self) -> None:
        for name, context_key in (
            (ROOT / "recipes" / "examples" / "gce_formula_smoke.yaml", "model"),
        ):
            recipe = load_recipe(name)
            context = execute_recipe(recipe)
            self.assertIn(context_key, context)


if __name__ == "__main__":
    unittest.main()
