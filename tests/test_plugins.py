import unittest

from lnl_toolbox.losses import (
    ActivePassiveLoss,
    CrossEntropyLoss,
    GeneralizedCrossEntropyLoss,
    MeanAbsoluteErrorLoss,
    NormalizedCrossEntropyLoss,
)
from lnl_toolbox.noise import AnchorTransitionEstimator, DualTransitionEstimator
from lnl_toolbox.plugins import PluginCatalog
from lnl_toolbox.plugins.builtin import (
    build_builtin_loss,
    build_builtin_transition_estimator,
    create_builtin_catalog,
)


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

    def test_transition_estimator_registry_and_builder(self) -> None:
        catalog = create_builtin_catalog()
        self.assertEqual(
            [item.name for item in catalog.find(kind="transition_estimator")],
            ["anchor", "dual_t"],
        )
        self.assertIsInstance(
            build_builtin_transition_estimator({"name": "anchor"}, catalog),
            AnchorTransitionEstimator,
        )
        self.assertIsInstance(
            build_builtin_transition_estimator({"name": "dual_t"}, catalog),
            DualTransitionEstimator,
        )
        with self.assertRaises(ValueError):
            build_builtin_transition_estimator({"name": "unknown"}, catalog)
        with self.assertRaises(TypeError):
            build_builtin_transition_estimator("anchor")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

