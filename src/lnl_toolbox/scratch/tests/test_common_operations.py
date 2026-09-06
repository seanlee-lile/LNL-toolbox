from __future__ import annotations

import unittest

import torch

import lnl_toolbox.scratch.blocks  # noqa: F401
from lnl_toolbox.scratch.context import ScratchContext
from lnl_toolbox.scratch.registry import BLOCKS


def run(block_id: str, context: ScratchContext, **params):
    BLOCKS[block_id].execute(context, **params)


class CommonOperationTest(unittest.TestCase):
    def test_model_lifecycle_and_module_forward(self):
        model = torch.nn.Linear(2, 2)
        context = ScratchContext(model=model, x=torch.ones(3, 2), y=torch.ones(3, 2))
        run("attach_trainable_head", context, model="model", input_dim=2, output_dim=2, head_name="head")
        self.assertTrue(hasattr(model, "head"))
        run("module_forward", context, module="model", inputs=["x"], save_as="out")
        self.assertEqual(tuple(context["out"].shape), (3, 2))
        run("zero_module_parameters", context, module="model", submodule="head")
        self.assertTrue(torch.equal(model.head.weight, torch.zeros_like(model.head.weight)))
        run("set_module_trainability", context, module="model", submodule="head", trainable=False)
        self.assertFalse(model.head.weight.requires_grad)

    def test_tensor_and_label_operations_are_atomic(self):
        context = ScratchContext(values=torch.tensor([[1.0, 2.0], [2.0, 2.0]]), logits=torch.tensor([[1.0, 0.0], [0.0, 1.0]]), mask=torch.tensor([[True, False], [False, True]]))
        run("row_normalize", context, input="values", save_as="normalized")
        self.assertTrue(torch.allclose(context["normalized"].sum(-1), torch.ones(2)))
        run("positive_logdet", ScratchContext(matrix=torch.eye(2)), matrix="matrix", save_as="logdet")
        labels = torch.tensor([0, 1])
        context["labels"] = labels
        run("gather_by_label", context, values="logits", labels="labels", save_as="gathered")
        self.assertTrue(torch.equal(context["gathered"], torch.tensor([1.0, 1.0])))
        context["left"] = torch.tensor([1.0, 2.0])
        context["right"] = torch.tensor([3.0, 4.0])
        run("add", context, left="left", right="right", save_as="sum")
        self.assertTrue(torch.equal(context["sum"], torch.tensor([4.0, 6.0])))
        run("partial_label_loss", context, logits="logits", candidates="mask", save_as="partial")
        run("complementary_negative_loss", context, logits="logits", complements="mask", save_as="negative")
        self.assertTrue(torch.isfinite(context["partial"]))
        self.assertTrue(torch.isfinite(context["negative"]))

    def test_selection_state_and_statistics_contracts(self):
        context = ScratchContext(scores=torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3]]), indices=torch.tensor([20, 5, 10]), labels=torch.tensor([0, 1, 0]))
        run("classwise_percentile_anchor_candidates", context, scores="scores", percentiles=[0.5], stable_sample_indices="indices", save_as="anchors")
        self.assertEqual(tuple(context["anchors"].shape), (2, 1))
        run("create_indexed_state", context, size=3, width=1, dtype="bool", save_as="mask_state")
        context["rows"] = torch.tensor([0, 2]); context["increments"] = torch.tensor([1, 1])
        run("indexed_increment", context, state="mask_state", indices="rows", values="increments")
        self.assertTrue(torch.equal(context["mask_state"]["values"].flatten()[[0, 2]], torch.tensor([True, True])))
        self.assertFalse(bool(context["mask_state"]["values"].flatten()[1]))
        context["window_values"] = torch.tensor([[1.0, 3.0], [2.0, 2.0]])
        context["window_observed"] = torch.tensor([[True, False], [True, True]])
        run("robust_window_mean", context, values="window_values", observed="window_observed", save_as="mean")
        self.assertTrue(torch.allclose(context["mean"], torch.tensor([1.0, 2.0])))
        run("estimate_class_prior", context, labels="labels", num_classes=2, save_as="prior")
        self.assertTrue(torch.allclose(context["prior"], torch.tensor([2 / 3, 1 / 3])))

    def test_transition_statistics_and_callable_revision(self):
        base = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        context = ScratchContext(noisy_prior=torch.tensor([0.45, 0.55]), transition=base, matrix=base)
        run("recover_clean_prior", context, noisy_prior="noisy_prior", transition="transition", save_as="clean")
        self.assertAlmostEqual(float(context["clean"].sum()), 1.0, places=5)
        run("matrix_pseudoinverse", context, matrix="matrix", save_as="pinv")
        self.assertTrue(torch.allclose(context["pinv"] @ base, torch.eye(2), atol=1e-5))
        run("create_additive_transition_revision", context, transition="transition", save_as="revision")
        run("materialize_transition", context, artifact="revision", save_as="materialized")
        self.assertEqual(tuple(context["materialized"].shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
