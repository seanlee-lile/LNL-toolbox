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

    def test_coteaching_recipe_matches_formal_protocol(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "coteaching.yaml")
        config_path = ROOT.parents[2] / "configs" / "experiment" / "cifar10_coteaching_reproduction.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        steps = recipe["steps"]
        data = {step["block"]: step.get("params", {}) for step in steps[:14]}
        self.assertEqual(data["create_dataset_split"]["validation_size"], config["data"]["validation_size"])
        self.assertEqual(data["apply_noise"]["rate"], config["noise"]["rate"])
        self.assertEqual(data["apply_noise"]["seed"], config["noise"]["seed"])
        self.assertEqual(data["configure_loader"]["batch_size"], config["loader"]["batch_size"])
        self.assertEqual(data["configure_loader"]["num_workers"], config["loader"]["num_workers"])
        models = [step for step in steps if step["block"] == "create_model"]
        self.assertEqual([step["params"]["model"] for step in models], [config["model"]["name"]] * 2)
        optimizers = [step for step in steps if step["block"] == "create_optimizer"]
        self.assertEqual([step["params"]["optimizer"] for step in optimizers], [config["optimizer"]["name"]] * 2)
        epoch = next(step for step in steps if step["block"] == "epoch_loop")
        self.assertEqual(epoch["params"]["epochs"], config["trainer"]["epochs"])
        self.assertEqual([step["block"] for step in epoch["steps"][-2:]], ["evaluate_peer_ensemble", "track_best_state"])

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

    def test_fine_recipe_matches_warmup_and_cosine_protocol(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "fine.yaml")
        validate_recipe(recipe)
        config_path = ROOT.parents[2] / "configs" / "experiment" / "fine_cifar100n_reproduction.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        steps = recipe["steps"]
        optimizer = next(step for step in steps if step["block"] == "create_optimizer")["params"]
        self.assertEqual(optimizer["lr"], config["fine"]["warmup_lr"])
        scheduler = next(step for step in steps if step["block"] == "create_scheduler")["params"]
        self.assertEqual(
            (scheduler["scheduler"], scheduler["t_max"], scheduler["eta_min"]),
            (
                config["scheduler"]["name"],
                config["trainer"]["epochs"] - config["fine"]["warmup_epochs"],
                config["scheduler"]["eta_min"],
            ),
        )
        epoch = next(step for step in steps if step["block"] == "epoch_loop")
        warmup_epochs = config["fine"]["warmup_epochs"]
        self.assertEqual(epoch["params"]["epochs"], config["trainer"]["epochs"])
        self.assertEqual(epoch["steps"][0]["block"], "if_epoch_eq")
        self.assertEqual(epoch["steps"][0]["params"]["epoch"], warmup_epochs)
        self.assertEqual(epoch["steps"][0]["steps"], [{"block": "set_optimizer_learning_rate", "params": {"optimizer": "optimizer", "learning_rate": config["optimizer"]["lr"]}}])
        self.assertEqual(epoch["steps"][-1]["block"], "if_epoch_ge")
        self.assertEqual(epoch["steps"][-1]["params"]["epoch"], warmup_epochs)
        self.assertEqual(epoch["steps"][-1]["steps"], [{"block": "scheduler_step"}])
        batch = next(step for step in epoch["steps"] if step["block"] == "batch_loop")
        robust = next(step for step in batch["steps"] if step["block"] == "if_epoch_ge")
        self.assertEqual(
            [step["block"] for step in robust["steps"][:3]],
            ["indexed_read", "indexed_read", "indexed_read"],
        )
        self.assertNotIn("fine_" + "prepare_batch_targets", [step["block"] for step in robust["steps"]])
        self.assertEqual(
            [step["block"] for step in robust["steps"][-4:]],
            ["elementwise_multiply", "mean_loss", "sed_rejected_regularizer", "weighted_sum"],
        )
        regularizer = next(step for step in robust["steps"] if step["block"] == "sed_rejected_regularizer")
        self.assertNotIn("pseudo_labels", regularizer.get("params", {}))
        self.assertNotIn("fine_robust_loss", [step["block"] for step in robust["steps"]])

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

    def test_l2rw_recipe_enforces_official_global_step_schedule(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "l2rw.yaml")
        validate_recipe(recipe)
        config_path = ROOT.parents[2] / "configs" / "experiment" / "l2rw_cifar10_reproduction.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        epoch = next(step for step in recipe["steps"] if step["block"] == "epoch_loop")
        self.assertEqual(epoch["params"]["epochs"], config["trainer"]["epochs"])
        batch = next(step for step in epoch["steps"] if step["block"] == "batch_loop")
        self.assertEqual(batch["params"]["max_steps"], config["trainer"]["max_steps"])
        self.assertEqual(batch["params"]["global_step_as"], "l2rw_global_step")
        schedule = batch["steps"][2]
        self.assertEqual(schedule["block"], "step_milestone_update")
        self.assertEqual(
            (schedule["params"]["milestones"], schedule["params"]["gamma"], schedule["params"]["global_step"]),
            (config["scheduler"]["step_milestones"], config["scheduler"]["gamma"], "l2rw_global_step"),
        )
        blocks = [step["block"] for step in batch["steps"]]
        chain = blocks.index("zeros_like")
        self.assertEqual(
            blocks[chain:chain + 7],
            ["zeros_like", "elementwise_multiply", "sum_values", "virtual_parameter_update", "functional_forward_with_state", "per_sample_ce", "mean_loss"],
        )
        self.assertIn("gradient_wrt", blocks)
        self.assertNotIn("l2rw_virtual_update", blocks)
        self.assertNotIn("l2rw_trusted_meta_loss", blocks)
        self.assertNotIn("l2rw_epsilon_gradient", blocks)

    def test_cal_recipe_keeps_external_proxy_stage_separate(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "cal.yaml")
        validate_recipe(recipe)
        config_path = ROOT.parents[2] / "configs" / "experiment" / "cal_cifar10_reproduction.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        steps = recipe["steps"]
        data = {step["block"]: step.get("params", {}) for step in steps[:16]}
        self.assertEqual(data["apply_noise"]["options"]["artifact_path"], config["noise"]["path"])
        warmup = next(step for step in steps if step["block"] == "epoch_loop")
        self.assertEqual(warmup["params"]["epochs"], config["warmup"]["epochs"])
        self.assertEqual(next(step for step in steps if step["block"] == "cal_materialize_proxy_artifact")["params"]["model"], "warmup_model")
        main = [step for step in steps if step["block"] == "epoch_loop"][1]
        self.assertEqual(main["params"]["epochs"], config["trainer"]["epochs"])
        batch = next(step for step in main["steps"] if step["block"] == "batch_loop")
        blocks = [step["block"] for step in batch["steps"]]
        top_blocks = [step["block"] for step in steps]
        removed = {
            "create_" + "cal_state",
            "cal_" + "prepare_proxy_batch",
            "cal_" + "warmup_objective",
            "cal_" + "reset_reference_accumulator",
            "cal_" + "accumulate_reference_losses",
            "cal_" + "finalize_reference_losses",
        }
        self.assertTrue(removed.isdisjoint(blocks))
        self.assertIn("estimate_class_prior", top_blocks)
        self.assertIn("cal_cores2_adjusted_risk", blocks)
        self.assertIn("cal_covariance_correction", blocks)
        self.assertIn("grouped_accumulate", blocks)
        self.assertIn("finalize_grouped_accumulator", [step["block"] for step in main["steps"]])
        self.assertNotIn("cal_second_order_objective", blocks)

    def test_apl_recipe_matches_formal_protocol(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "apl.yaml")
        config_path = ROOT.parents[2] / "configs" / "experiment" / "apl_cifar10_noise02_reproduction.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        steps = recipe["steps"]
        data = {step["block"]: step.get("params", {}) for step in steps[:19]}
        self.assertEqual(data["create_dataset_split"]["validation_size"], config["data"]["validation_size"])
        self.assertEqual(data["configure_preprocessing"]["augment"], config["data"]["augment"])
        self.assertEqual(data["apply_noise"]["name"], config["noise"]["name"])
        self.assertEqual(data["apply_noise"]["rate"], config["noise"]["rate"])
        self.assertEqual(data["apply_noise"]["seed"], config["noise"]["seed"])
        self.assertEqual(data["configure_loader"]["batch_size"], config["loader"]["batch_size"])
        self.assertEqual(data["configure_loader"]["num_workers"], config["loader"]["num_workers"])
        rce = next(step for step in steps if step["block"] == "epoch_loop")["steps"]
        rce = next(step for step in rce if step["block"] == "batch_loop")
        rce = next(step for step in rce["steps"] if step["block"] == "rce_loss")
        self.assertEqual(rce["params"]["log_zero"], -4.0)
        self.assertEqual(next(step for step in steps if step["block"] == "create_model")["params"], {"model": config["model"]["name"], "num_classes": 10, "device": "device"})
        optimizer = next(step for step in steps if step["block"] == "create_optimizer")["params"]
        self.assertEqual({key: optimizer[key] for key in ("optimizer", "lr", "momentum", "nesterov", "weight_decay")}, {"optimizer": config["optimizer"]["name"], "lr": config["optimizer"]["lr"], "momentum": config["optimizer"]["momentum"], "nesterov": config["optimizer"]["nesterov"], "weight_decay": config["optimizer"]["weight_decay"]})
        scheduler = next(step for step in steps if step["block"] == "create_scheduler")["params"]
        self.assertEqual((scheduler["scheduler"], scheduler["t_max"], scheduler["eta_min"]), (config["scheduler"]["name"], config["scheduler"]["t_max"], config["scheduler"]["eta_min"]))
        epoch = next(step for step in steps if step["block"] == "epoch_loop")
        self.assertEqual(epoch["params"]["epochs"], config["trainer"]["epochs"])
        self.assertEqual(_batch_blocks(recipe), ["get_batch", "move_batch_to_device", "zero_grad", "forward", "nce_loss", "rce_loss", "weighted_sum", "mean_loss", "backward", "optimizer_step"])
        self.assertNotIn("clip_grad_norm", _batch_blocks(recipe))
        self.assertEqual([step["block"] for step in epoch["steps"][-3:]], ["evaluate_accuracy", "track_best_model", "scheduler_step"])
        self.assertEqual(epoch["steps"][-2]["params"]["metric_name"], "selection_accuracy")

    def test_binary_risk_recipe_matches_formal_protocol(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "binary_risk.yaml")
        config_path = ROOT.parents[2] / "configs" / "experiment" / "binary_risk_natarajan_reproduction.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        steps = recipe["steps"]
        data = {step["block"]: step.get("params", {}) for step in steps[:20]}
        options = data["load_dataset"]["options"]
        self.assertEqual((options["train_size"], options["test_size"], options["data_seed"]), (config["data"]["train_size"], config["data"]["test_size"], config["data"]["seed"]))
        self.assertEqual((data["apply_noise"]["options"]["rho_positive"], data["apply_noise"]["options"]["rho_negative"], data["apply_noise"]["seed"], data["configure_loader"]["batch_size"]), (config["risk"]["rho_positive"], config["risk"]["rho_negative"], config["noise"]["seed"], config["loader"]["batch_size"]))
        self.assertEqual(next(step for step in steps if step["block"] == "create_model")["params"], {"model": config["model"]["name"], "input_dim": 2, "num_classes": 2, "device": "device"})
        optimizer = next(step for step in steps if step["block"] == "create_optimizer")["params"]
        self.assertEqual((optimizer["optimizer"], optimizer["lr"], optimizer["momentum"]), (config["optimizer"]["name"], config["optimizer"]["lr"], config["optimizer"]["momentum"]))
        epoch = next(step for step in steps if step["block"] == "epoch_loop")
        self.assertEqual(epoch["params"]["epochs"], config["trainer"]["epochs"])
        self.assertEqual(_batch_blocks(recipe), ["get_batch", "move_batch_to_device", "zero_grad", "forward", "binary_risk", "mean_loss", "backward", "optimizer_step"])
        self.assertEqual(epoch["steps"][-1]["block"], "evaluate_accuracy")

    def test_loss_correction_recipe_matches_formal_protocol(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "loss_correction.yaml")
        config_path = ROOT.parents[2] / "configs" / "experiment" / "loss_correction_cifar10_asymmetric04.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        steps = recipe["steps"]
        data = {step["block"]: step.get("params", {}) for step in steps[:19]}
        self.assertEqual(
            (data["create_dataset_split"]["validation_size"], data["configure_preprocessing"]["augment"], data["apply_noise"]["seed"], data["configure_loader"]["batch_size"]),
            (config["data"]["validation_size"], config["data"]["augment"], config["noise"]["seed"], config["loader"]["batch_size"]),
        )
        self.assertEqual(
            next(step for step in steps if step["block"] == "create_model")["params"],
            {"model": config["model"]["name"], "num_classes": 10, "base_width": config["model"]["base_width"], "device": "device"},
        )
        optimizer = next(step for step in steps if step["block"] == "create_optimizer")["params"]
        self.assertEqual(
            {key: optimizer[key] for key in ("optimizer", "lr", "momentum", "nesterov", "weight_decay")},
            {"optimizer": config["optimizer"]["name"], "lr": config["optimizer"]["lr"], "momentum": config["optimizer"]["momentum"], "nesterov": config["optimizer"]["nesterov"], "weight_decay": config["optimizer"]["weight_decay"]},
        )
        scheduler = next(step for step in steps if step["block"] == "create_scheduler")["params"]
        self.assertEqual((scheduler["scheduler"], scheduler["milestones"], scheduler["gamma"]), (config["scheduler"]["name"], config["scheduler"]["milestones"], config["scheduler"]["gamma"]))
        epoch = next(step for step in steps if step["block"] == "epoch_loop")
        self.assertEqual(epoch["params"]["epochs"], config["trainer"]["epochs"])
        self.assertEqual(_batch_blocks(recipe), ["get_batch", "move_batch_to_device", "zero_grad", "forward", "softmax", "apply_transition", "gather_by_label", "negative_log", "mean_loss", "backward", "optimizer_step"])
        correction = next(step for step in epoch["steps"][1]["steps"] if step["block"] == "apply_transition")
        self.assertEqual(correction["params"]["transition"], "transition")

    def test_jocor_recipe_matches_formal_protocol(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "jocor.yaml")
        config_path = ROOT.parents[2] / "configs" / "experiment" / "jocor_cifar10_symmetric05_reproduction.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        steps = recipe["steps"]
        data = {step["block"]: step.get("params", {}) for step in steps[:19]}
        self.assertEqual((data["apply_noise"]["rate"], data["apply_noise"]["seed"], data["configure_loader"]["batch_size"], data["configure_loader"]["num_workers"]), (config["noise"]["rate"], config["noise"]["seed"], config["loader"]["batch_size"], config["loader"]["num_workers"]))
        models = [step for step in steps if step["block"] == "create_model"]
        self.assertEqual([step["params"]["model"] for step in models], [config_model["name"] for config_model in config["models"]])
        optimizer = next(step for step in steps if step["block"] == "create_parameter_group_optimizer")["params"]
        self.assertEqual(optimizer["optimizer_spec"]["name"], config["optimizer"]["name"])
        self.assertEqual(optimizer["optimizer_spec"]["betas"], config["optimizer"]["betas"])
        self.assertEqual(optimizer["optimizer_spec"]["weight_decay"], config["optimizer"]["weight_decay"])
        self.assertEqual([group["lr"] for group in optimizer["groups"]], [config["optimizer"]["lr"]] * 2)
        scheduler = next(step for step in steps if step["block"] == "create_scheduler")["params"]
        self.assertEqual((scheduler["scheduler"], scheduler["start_epoch"], scheduler["end_epoch"], scheduler["initial_lr"], scheduler["final_lr"], scheduler["beta1_before"], scheduler["beta1_after"]), (config["scheduler"]["name"], config["scheduler"]["start_epoch"], config["scheduler"]["end_epoch"], config["scheduler"]["initial_lr"], config["scheduler"]["final_lr"], config["scheduler"]["beta1_before"], config["scheduler"]["beta1_after"]))
        epoch = next(step for step in steps if step["block"] == "epoch_loop")
        self.assertEqual(epoch["params"]["epochs"], config["trainer"]["epochs"])
        self.assertEqual(_batch_blocks(recipe), ["get_batch", "move_batch_to_device", "zero_grad", "forward", "forward", "per_sample_ce", "per_sample_ce", "symmetric_kl", "weighted_sum", "select_lowest_scores", "mean_by_indices", "backward", "optimizer_step"])
        self.assertEqual([step["block"] for step in epoch["steps"][-2:]], ["track_best_state", "scheduler_step"])

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
