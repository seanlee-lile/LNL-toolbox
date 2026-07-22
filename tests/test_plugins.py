import sys
import unittest
from unittest.mock import patch

import numpy as np

from lnl_toolbox.losses import (
    ActivePassiveLoss,
    CrossEntropyLoss,
    GeneralizedCrossEntropyLoss,
    MeanAbsoluteErrorLoss,
    NormalizedCrossEntropyLoss,
)
from lnl_toolbox.plugins import PluginCatalog
from lnl_toolbox.plugins.builtin import (
    build_builtin_loss,
    build_builtin_selector,
    create_builtin_catalog,
)
from lnl_toolbox.selectors import AllSelector, SmallLossSelector


class PluginCatalogTest(unittest.TestCase):
    def test_capability_discovery(self) -> None:
        catalog = create_builtin_catalog()
        selectors = catalog.find(capability="sample_selection")
        self.assertEqual([(item.kind, item.name) for item in selectors], [("selector", "coteaching_exchange")])

    def test_custom_plugin_does_not_need_lnl_types(self) -> None:
        catalog = PluginCatalog()
        catalog.add("transform", "uppercase", lambda value: value.upper(), capabilities=("text",))
        self.assertEqual(catalog.build("transform", "uppercase", value="hello"), "HELLO")

    def test_trainable_and_numpy_losses_use_distinct_plugin_kinds(self) -> None:
        catalog = create_builtin_catalog()
        self.assertEqual(
            [item.name for item in catalog.find(kind="loss")],
            ["apl", "ce", "gce", "mae", "nce", "rce"],
        )
        self.assertEqual([item.name for item in catalog.find(kind="numpy_loss")], ["ce", "gce"])
        self.assertIsInstance(build_builtin_loss({"name": "gce", "q": 0.5}, catalog),
                              GeneralizedCrossEntropyLoss)

    def test_batch_selectors_use_a_distinct_plugin_kind(self) -> None:
        catalog = create_builtin_catalog()
        self.assertEqual(
            [item.name for item in catalog.find(kind="batch_selector")],
            ["all", "small_loss"],
        )
        self.assertIsInstance(build_builtin_selector(None, catalog), AllSelector)
        selector = build_builtin_selector(
            {"name": "small_loss", "keep_rate": 0.5}, catalog
        )
        self.assertIsInstance(selector, SmallLossSelector)
        self.assertEqual(selector.keep_rate, 0.5)
        constant = build_builtin_selector({
            "name": "small_loss",
            "keep_rate": {"name": "constant", "value": 0.8},
        }, catalog)
        linear = build_builtin_selector({
            "name": "small_loss",
            "keep_rate": {
                "name": "linear", "start": 1.0, "end": 0.6,
                "warmup_epochs": 10,
            },
        }, catalog)
        self.assertEqual(constant.schedule.rate_at(7), 0.8)
        self.assertEqual(linear.schedule.rate_at(5), 0.8)
        with self.assertRaises(ValueError):
            build_builtin_selector({"name": "unknown"}, catalog)
        with self.assertRaises(TypeError):
            build_builtin_selector("all", catalog)  # type: ignore[arg-type]

    def test_selector_import_failure_does_not_change_loss_catalog(self) -> None:
        catalog_with_selectors = create_builtin_catalog()
        expected_losses = [
            (item.name, item.factory, item.capabilities)
            for item in catalog_with_selectors.find(kind="loss")
        ]
        with patch.dict(sys.modules, {"lnl_toolbox.selectors": None}):
            catalog = create_builtin_catalog()

        self.assertEqual(
            [
                (item.name, item.factory, item.capabilities)
                for item in catalog.find(kind="loss")
            ],
            expected_losses,
        )
        self.assertEqual(catalog.find(kind="batch_selector"), ())

    def test_legacy_coteaching_exchange_registration_and_behavior_are_unchanged(self) -> None:
        catalog = create_builtin_catalog()
        spec = catalog.get("selector", "coteaching_exchange")
        self.assertEqual(spec.kind, "selector")
        self.assertEqual(spec.name, "coteaching_exchange")
        self.assertEqual(
            spec.capabilities,
            frozenset({"multi_model", "sample_selection"}),
        )

        selected_for_a, selected_for_b = catalog.build(
            "selector",
            "coteaching_exchange",
            losses_a=np.array([0.4, 0.1, 0.3, 0.2]),
            losses_b=np.array([0.2, 0.4, 0.1, 0.3]),
            keep_rate=0.5,
        )
        np.testing.assert_array_equal(selected_for_a, np.array([2, 0]))
        np.testing.assert_array_equal(selected_for_b, np.array([1, 3]))
        with self.assertRaises(ValueError):
            build_builtin_selector({"name": "coteaching_exchange"}, catalog)

    def test_apl_is_built_recursively_and_rejects_unsupported_children(self) -> None:
        loss = build_builtin_loss({
            "name": "apl", "alpha": 2.0, "beta": 0.5,
            "active": {"name": "nce"},
            "passive": {"name": "mae", "scale": 1.0},
        })
        self.assertIsInstance(loss, ActivePassiveLoss)
        with self.assertRaises(ValueError):
            build_builtin_loss({"name": "apl", "active": {"name": "ce"}})
        with self.assertRaises(ValueError):
            build_builtin_loss({"name": "apl", "alpha": 0.0})
        with self.assertRaises(ValueError):
            build_builtin_loss({"name": "gce", "q": 0.7, "eps": 1e-8})
        with self.assertRaises(ValueError):
            build_builtin_loss({"name": "unknown"})
        with self.assertRaises(TypeError):
            build_builtin_loss("gce")  # type: ignore[arg-type]

    def test_direct_catalog_apl_build_enforces_paper_constraints(self) -> None:
        catalog = create_builtin_catalog()
        valid = catalog.build(
            "loss", "apl",
            active=NormalizedCrossEntropyLoss(),
            passive=MeanAbsoluteErrorLoss(),
        )
        self.assertIsInstance(valid, ActivePassiveLoss)
        with self.assertRaises(TypeError):
            catalog.build(
                "loss", "apl",
                active=CrossEntropyLoss(),
                passive=MeanAbsoluteErrorLoss(),
            )
        with self.assertRaises(ValueError):
            catalog.build(
                "loss", "apl",
                active=NormalizedCrossEntropyLoss(),
                passive=MeanAbsoluteErrorLoss(),
                alpha=0.0,
            )


if __name__ == "__main__":
    unittest.main()

