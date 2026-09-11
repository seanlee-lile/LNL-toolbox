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
        run("divide", context, numerator="right", denominator="left", save_as="quotient")
        self.assertTrue(torch.allclose(context["quotient"], torch.tensor([3.0, 2.0])))
        run("maximum", context, left="left", right="right", save_as="maximum")
        run("minimum", context, left="left", right="right", save_as="minimum")
        self.assertTrue(torch.equal(context["maximum"], torch.tensor([3.0, 4.0])))
        self.assertTrue(torch.equal(context["minimum"], torch.tensor([1.0, 2.0])))
        run("exp", context, input="left", save_as="exponential")
        run("log", context, input="left", save_as="log_values")
        run("sqrt", context, input="left", save_as="square_root")
        context["signed"] = torch.tensor([-2.0, 3.0])
        run("abs", context, input="signed", save_as="absolute")
        run("sign", context, input="signed", save_as="signs")
        self.assertTrue(torch.allclose(context["log_values"], context["left"].log()))
        self.assertTrue(torch.equal(context["signs"], torch.tensor([-1.0, 1.0])))
        context["matrix_left"] = torch.eye(2)
        context["matrix_right"] = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        run("matrix_multiply", context, left="matrix_left", right="matrix_right", save_as="matrix_product")
        self.assertTrue(torch.equal(context["matrix_product"], context["matrix_right"]))
        run("log_softmax", context, logits="logits", save_as="log_probabilities")
        self.assertTrue(torch.allclose(context["log_probabilities"].exp().sum(-1), torch.ones(2)))
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

    def test_formula_primitives_support_broadcast_reduction_shape_and_selection(self):
        context = ScratchContext(
            values=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            bias=torch.tensor([10.0, 20.0]),
            mask=torch.tensor([True, False]),
            true_values=torch.tensor([1.0, 2.0]),
            false_values=torch.tensor([3.0, 4.0]),
            indices=torch.tensor([1, 0]),
        )
        run("elementwise_multiply", context, left="values", right="bias", save_as="broadcast_product")
        self.assertTrue(torch.equal(context["broadcast_product"], torch.tensor([[10.0, 40.0], [30.0, 80.0]])))
        run("reduce_sum", context, input="values", dim=1, save_as="sum_rows")
        run("reduce_mean", context, input="values", dim=0, save_as="mean_columns")
        run("reduce_max", context, input="values", dim=None, save_as="max_all")
        run("reduce_min", context, input="values", dim=None, save_as="min_all")
        self.assertTrue(torch.equal(context["sum_rows"], torch.tensor([3.0, 7.0])))
        self.assertTrue(torch.equal(context["mean_columns"], torch.tensor([2.0, 3.0])))
        self.assertEqual(float(context["max_all"]), 4.0)
        self.assertEqual(float(context["min_all"]), 1.0)
        run("reshape_tensor", context, input="values", shape=[4], save_as="flat")
        run("unsqueeze", context, input="flat", dim=0, save_as="expanded")
        run("squeeze", context, input="expanded", dim=0, save_as="squeezed")
        run("transpose_dims", context, input="values", dim0=0, dim1=1, save_as="transposed")
        self.assertEqual(tuple(context["squeezed"].shape), (4,))
        self.assertTrue(torch.equal(context["transposed"], context["values"].t()))
        run("where", context, condition="mask", when_true="true_values", when_false="false_values", save_as="chosen")
        run("logical_not", context, input="mask", save_as="not_mask")
        run("logical_and", context, left="mask", right="not_mask", save_as="and_mask")
        self.assertTrue(torch.equal(context["chosen"], torch.tensor([1.0, 4.0])))
        self.assertTrue(torch.equal(context["and_mask"], torch.tensor([False, False])))
        run("argmax", context, input="values", dim=1, keepdim=True, save_as="argmax_indices")
        run("gather", context, input="values", indices="argmax_indices", dim=1, save_as="gathered")
        run("index_select", context, input="values", indices="indices", dim=0, save_as="selected_rows")
        self.assertTrue(torch.equal(context["gathered"], torch.tensor([[2.0], [4.0]])))
        self.assertTrue(torch.equal(context["selected_rows"], torch.tensor([[3.0, 4.0], [1.0, 2.0]])))

    def test_batched_matmul_and_formula_kind_metadata(self):
        context = ScratchContext(
            left=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            right=torch.tensor([[[1.0], [2.0]], [[2.0], [1.0]]]),
        )
        run("batched_matmul", context, left="left", right="right", save_as="product")
        self.assertTrue(torch.equal(context["product"], torch.tensor([[5.0], [10.0]])))
        self.assertEqual(BLOCKS["reduce_sum"].formula_kind, "primitive")
        self.assertEqual(BLOCKS["gather_by_label"].formula_kind, "composite")


if __name__ == "__main__":
    unittest.main()
