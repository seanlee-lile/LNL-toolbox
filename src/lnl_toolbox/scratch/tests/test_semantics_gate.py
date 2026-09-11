"""Formal-semantics gates for Scratch models, state, DSS, LEND and Meta.

Legacy implementations are imported only from this test module as numerical
oracles.  Scratch production modules remain isolated from those packages.
"""

from __future__ import annotations

import unittest

import torch

import lnl_toolbox.scratch.blocks  # noqa: F401 - register the palette
from lnl_toolbox.algorithms.lend.graph import build_lend_similarity, normalize_lend_graph
from lnl_toolbox.selectors.dss import DSSSelectorState
from lnl_toolbox.scratch.blocks.graph import (
    neighbor_edge_weights,
    normalize_graph,
    pairwise_similarity,
    topk_neighborhood,
)
from lnl_toolbox.scratch.blocks.models import create_model
from lnl_toolbox.scratch.blocks.paper_specific.dss import _ScratchDSSState
from lnl_toolbox.scratch.blocks.state import (
    append_indexed_window,
    create_grouped_accumulator,
    create_indexed_state,
    create_indexed_window,
    finalize_grouped_accumulator,
    indexed_ema,
    indexed_read,
    indexed_write,
    read_indexed_window,
)
from lnl_toolbox.scratch.blocks.meta import (
    functional_forward_with_state,
    gradient_wrt,
    virtual_parameter_update,
)
from lnl_toolbox.scratch.context import ScratchContext


class ScratchSemanticsGateTest(unittest.TestCase):
    def test_resnet_topologies_match_legacy_parameter_counts(self) -> None:
        from lnl_toolbox.models.cifar_resnet import (
            cifar_resnet14,
            cifar_resnet18,
            cifar_resnet32,
            cifar_resnet34,
            cifar_resnet50,
            cifar_resnet101,
            preact_resnet18,
        )

        factories = {
            "resnet18": cifar_resnet18,
            "resnet34": cifar_resnet34,
            "resnet50": cifar_resnet50,
            "resnet101": cifar_resnet101,
            "resnet14": cifar_resnet14,
            "resnet32": cifar_resnet32,
            "preact_resnet18": preact_resnet18,
        }
        for name, factory in factories.items():
            scratch = ScratchContext(device="cpu")
            create_model(scratch, model=name, num_classes=3, base_width=4, device="device")
            legacy = factory(num_classes=3, base_width=4)
            self.assertEqual(
                sum(parameter.numel() for parameter in scratch["model"].parameters()),
                sum(parameter.numel() for parameter in legacy.parameters()),
                name,
            )
            self.assertEqual(
                set(scratch["model"].state_dict()),
                set(legacy.state_dict()),
                name,
            )
            with torch.no_grad():
                output = scratch["model"](torch.randn(2, 3, 32, 32))
            self.assertEqual(tuple(output.shape), (2, 3), name)

    def test_fine_seven_cnn_module_layout_matches_legacy(self) -> None:
        """Formal FINE topology includes stable module paths, not only counts."""
        from lnl_toolbox.models.fine_cnn import FineSevenCNN

        scratch = ScratchContext(device="cpu")
        create_model(scratch, model="fine_seven_cnn", num_classes=3,
                     base_width=16, device="device")
        legacy = FineSevenCNN(num_classes=3, base_width=16)
        self.assertEqual(
            set(scratch["model"].state_dict()),
            set(legacy.state_dict()),
        )
        inputs = torch.randn(2, 3, 32, 32)
        scratch_output = scratch["model"].forward_with_features(inputs)
        legacy_output = legacy.forward_with_features(inputs)
        self.assertEqual(tuple(scratch_output.logits.shape), (2, 3))
        self.assertEqual(tuple(scratch_output.features.shape),
                         tuple(legacy_output.features.shape))

    def test_formal_cnn_module_layouts_match_legacy(self) -> None:
        """Non-ResNet formal models must retain their layer topology too."""
        from lnl_toolbox.models.ca2c_cnn import CA2CSevenCNN
        from lnl_toolbox.models.cifar_cnn import CifarCnn8, CnlcuCnn9
        from lnl_toolbox.models.cifar_resnet import L2RWResNet32
        from lnl_toolbox.models.mc_ldce_cnn import MCLDCECifarCNN
        from lnl_toolbox.models.mentor_wide_resnet import MentorWideResNet101

        cases = (
            ("cifar_cnn8", CifarCnn8, {}, {}),
            ("cnlcu_cnn9", CnlcuCnn9, {}, {}),
            ("mentor_wide_resnet", MentorWideResNet101,
             {"num_residual_units": 2, "width_multiplier": 0.25},
             {"num_residual_units": 2, "width_multiplier": 0.25}),
            ("mc_ldce_cnn", MCLDCECifarCNN, {}, {}),
            ("ca2c_seven_cnn", CA2CSevenCNN, {}, {}),
            ("l2rw_resnet32", L2RWResNet32, {"base_width": 4},
             {"base_width": 4}),
        )
        for name, factory, scratch_options, legacy_options in cases:
            context = ScratchContext(device="cpu")
            create_model(
                context,
                model=name,
                num_classes=3,
                base_width=int(scratch_options.get("base_width", 64)),
                num_residual_units=int(scratch_options.get("num_residual_units", 9)),
                width_multiplier=float(scratch_options.get("width_multiplier", 1.0)),
                device="device",
            )
            scratch_model = context["model"]
            legacy_model = factory(num_classes=3, **legacy_options)
            self.assertEqual(
                sum(parameter.numel() for parameter in scratch_model.parameters()),
                sum(parameter.numel() for parameter in legacy_model.parameters()),
                name,
            )
            self.assertEqual(
                set(scratch_model.state_dict()), set(legacy_model.state_dict()), name
            )

    def test_dss_lifecycle_matches_legacy_oracle(self) -> None:
        torch.manual_seed(23)
        scratch = _ScratchDSSState(5, 3, 4, warmup_epochs=0, alpha=0.1,
                                   prior_decay=0.99, strict=True, mda=True, ccs=True)
        legacy = DSSSelectorState(5, 3, 4, warmup_epochs=0, alpha=0.1,
                                  prior_decay=0.99, mda=True, ccs=True)
        indices = torch.arange(5)
        labels = torch.tensor([0, 1, 2, 0, 1])
        for epoch in range(3):
            probabilities = torch.softmax(torch.randn(5, 3), dim=1)
            scratch.on_cycle_start(epoch)
            legacy.on_cycle_start(epoch)
            scratch.observe(indices, labels, probabilities, epoch)
            legacy.observe(indices, labels, probabilities, epoch)
            scratch.on_cycle_end(epoch)
            legacy.on_cycle_end(epoch)
        self.assertTrue(torch.equal(scratch.selected, legacy.selected))
        self.assertTrue(torch.equal(scratch.excluded, legacy.excluded))
        self.assertTrue(torch.allclose(scratch.current_prediction, legacy.current_prediction))
        self.assertTrue(torch.allclose(scratch.marginal, legacy.marginal))
        self.assertTrue(torch.allclose(scratch.trend_score, legacy.trend_score))

    def test_lend_graph_layers_match_legacy_for_metrics_and_stable_ids(self) -> None:
        torch.manual_seed(31)
        features = torch.rand(6, 4) + 0.25
        sample_indices = torch.tensor([50, 2, 40, 7, 11, 3])
        for metric, normalize in (("inner_product", False), ("cosine", False), ("euclidean", False), ("euclidean", True), ("inner_product", True)):
            reference = build_lend_similarity(features, sample_indices, k=2, gamma=1.2,
                                              metric=metric, normalize_features=normalize)
            context = ScratchContext(features=features, indices=sample_indices)
            pairwise_similarity(context, metric=metric, normalize_features=normalize)
            topk_neighborhood(context, k=2)
            neighbor_edge_weights(context, gamma=1.2, normalize_features=normalize)
            normalize_graph(context)
            self.assertTrue(torch.allclose(context["adjacency"], reference), metric)
            self.assertTrue(torch.allclose(context["graph"], normalize_lend_graph(reference)), metric)
            # Positive features make all selected inner-product edges nonzero;
            # compare the selected neighbour sets with the oracle adjacency.
            for row, selected in enumerate(context["neighbor_indices"]):
                self.assertEqual(set(selected.tolist()), set(reference[row].nonzero(as_tuple=False).flatten().tolist()), metric)

        # Equal ranking scores must be resolved by sample identity, not row
        # position.  The smallest IDs are selected first for every row.
        ties = torch.ones(4, 3)
        ids = torch.tensor([40, 4, 30, 2])
        context = ScratchContext(features=ties, indices=ids)
        pairwise_similarity(context, metric="inner_product")
        topk_neighborhood(context, k=2)
        self.assertEqual(context["neighbor_indices"][0].tolist(), [3, 1])

    def test_generic_indexed_and_window_state_contract(self) -> None:
        context = ScratchContext(indices=torch.tensor([0, 2]), values=torch.tensor([2.0, 4.0]))
        create_indexed_state(context, size=3, width=1, save_as="table")
        indexed_write(context, state="table", indices="indices", values="values")
        indexed_read(context, state="table", indices="indices", save_as="read")
        self.assertTrue(torch.equal(context["read"].flatten(), torch.tensor([2.0, 4.0])))
        context["epoch"] = 0
        indexed_ema(context, state="table", indices="indices", values="values", beta=0.5, epoch="epoch")
        context["epoch"] = 1
        context["values2"] = torch.tensor([4.0, 8.0])
        indexed_ema(context, state="table", indices="indices", values="values2", beta=0.5, epoch="epoch")
        create_indexed_window(context, size=3, window_size=2, save_as="window")
        append_indexed_window(context, state="window", indices="indices", values="values")
        context["values2"] = torch.tensor([3.0, 5.0])
        append_indexed_window(context, state="window", indices="indices", values="values2")
        read_indexed_window(context, state="window", indices="indices")
        self.assertTrue(torch.equal(context["window_counts"], torch.tensor([2, 2])))

    def test_grouped_accumulator_and_meta_chain_are_composable(self) -> None:
        context = ScratchContext(groups=torch.tensor([0, 1, 0]), values=torch.tensor([1.0, 2.0, 3.0]))
        create_grouped_accumulator(context, groups=2, save_as="grouped")
        from lnl_toolbox.scratch.blocks.state import grouped_accumulate
        grouped_accumulate(context, state="grouped", groups="groups", values="values")
        finalize_grouped_accumulator(context, state="grouped", save_as="means")
        self.assertTrue(torch.allclose(context["means"].flatten(), torch.tensor([2.0, 2.0])))

        model = torch.nn.Linear(2, 2)
        epsilon = torch.zeros(2, requires_grad=True)
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        logits = model(inputs + epsilon)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        context.update(model=model, virtual_loss=loss, images=inputs + epsilon, epsilon=epsilon)
        virtual_parameter_update(context, model="model", loss="virtual_loss", learning_rate=0.1, create_graph=True)
        functional_forward_with_state(context, model="model", state="virtual_state", inputs="images")
        meta_loss = torch.nn.functional.cross_entropy(context["logits"], labels)
        context["meta_loss"] = meta_loss
        gradient_wrt(context, loss="meta_loss", input="epsilon", create_graph=False)
        self.assertEqual(tuple(context["gradient"].shape), tuple(epsilon.shape))
        self.assertTrue(torch.isfinite(context["gradient"]).all())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
