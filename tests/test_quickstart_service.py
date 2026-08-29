from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lnl_toolbox.data.contracts import DataSpec, RawDatasetSplit
from lnl_toolbox.data.probe import DatasetProbeResult
from lnl_toolbox.data.local_catalog import LocalDatasetCatalog
from lnl_toolbox.data.registry import DatasetRegistry
from lnl_toolbox.catalog import load_papers
from lnl_toolbox.training.data_service import DataService
from lnl_toolbox.quickstart.models import QuickStartNoiseSelection
from lnl_toolbox.quickstart.service import QuickStartService
from lnl_toolbox.quickstart.templates import adapt_method_template, method_template_for_paper
from lnl_toolbox.quickstart.templates import find_exact_reproduction
from lnl_toolbox.training.compatibility import CompatibilityStatus


class _FakeImageAdapter:
    name = "cifar10"
    aliases = ()

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None or not spec.root.is_dir():
            raise FileNotFoundError(spec.root)

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del seed
        count = 12 if split == "train" else 6
        images = np.zeros((count, 32, 32, 3), dtype=np.uint8)
        labels = np.arange(count, dtype=np.int64) % 2
        return RawDatasetSplit(
            images, labels, np.arange(count, dtype=np.int64), self.name, split, 2,
            clean_targets=labels, class_names=("zero", "one"), source="test-fixture",
        )


class _FakeCifar100Adapter(_FakeImageAdapter):
    name = "cifar100"

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del spec, seed
        count = 200 if split == "train" else 100
        images = np.zeros((count, 32, 32, 3), dtype=np.uint8)
        labels = np.arange(count, dtype=np.int64) % 100
        return RawDatasetSplit(
            images, labels, np.arange(count, dtype=np.int64), self.name, split, 100,
            clean_targets=labels, source="test-fixture",
        )


class _FakeFashionMnistAdapter(_FakeImageAdapter):
    name = "fashion_mnist"


class QuickStartServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.data_root = root / "data"
        self.data_root.mkdir()
        registry = DatasetRegistry((_FakeImageAdapter(),))
        catalog = LocalDatasetCatalog(root / "catalog.json")
        self.data_service = DataService(registry=registry, catalog=catalog)
        self.service = QuickStartService(self.data_service, artifact_root=root / "artifacts")
        self.registered = self.data_service.register(
            "local-cifar10", "cifar10", {"root": str(self.data_root)}
        )
        self.data_service.inspect("local-cifar10")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_registered_dataset_is_reused(self) -> None:
        result = self.service.register_and_inspect(str(self.data_root))
        self.assertEqual(result.alias, "local-cifar10")
        self.assertEqual(len(self.data_service.catalog.records()), 1)

    def test_noise_options_are_capability_based(self) -> None:
        result = self.service.noise_options("local-cifar10")
        keys = {item["key"] for item in result["options"]}
        self.assertIn("symmetric", keys)
        self.assertNotIn("external_torch", keys)

    def test_method_options_are_paper_catalog_driven(self) -> None:
        result = self.service.method_options(
            "local-cifar10", QuickStartNoiseSelection("clean", "clean")
        )
        self.assertGreater(len(result), 5)
        self.assertTrue(all(item.acronym for item in result))
        fine = next(item for item in result if item.paper_id == "fine")
        self.assertEqual(fine.status, "ready")

    def test_method_options_batches_compatibility_and_reuses_cache(self) -> None:
        selection = QuickStartNoiseSelection("clean", "clean")
        original = self.service.experiment_service.list_config_compatibility
        with patch.object(
            self.service.experiment_service,
            "list_config_compatibility",
            wraps=original,
        ) as checker:
            first = self.service.method_options("local-cifar10", selection)
            second = self.service.method_options("local-cifar10", selection)
        self.assertEqual(first, second)
        self.assertEqual(checker.call_count, 1)
        self.assertGreater(len(first), 5)

    def test_method_options_cache_skips_second_dataset_inspection(self) -> None:
        selection = QuickStartNoiseSelection("clean", "clean")
        original = self.service.data_service.inspect
        with patch.object(self.service.data_service, "inspect", wraps=original) as inspector:
            self.service.method_options("local-cifar10", selection)
            first_calls = inspector.call_count
            self.service.method_options("local-cifar10", selection)
        self.assertGreaterEqual(first_calls, 1)
        self.assertEqual(inspector.call_count, first_calls)

    def test_unknown_path_is_not_marked_ready(self) -> None:
        result = self.service.register_and_inspect(str(Path(self.temp.name) / "missing"))
        self.assertEqual(result.status, "unsupported")

    def test_inspect_failure_stops_registration_flow(self) -> None:
        self.data_service.inspect = lambda *args, **kwargs: SimpleNamespace(
            status="incomplete", error="fixture inspect failed", profile=None
        )
        with patch.object(self.service, "probe", return_value=DatasetProbeResult(
            str(self.data_root), "already_registered", existing_alias="local-cifar10"
        )):
            with self.assertRaisesRegex(ValueError, "fixture inspect failed"):
                self.service.register_and_inspect(str(self.data_root))

    def test_method_options_rejects_an_incomplete_profile(self) -> None:
        self.data_service.inspect = lambda *args, **kwargs: SimpleNamespace(
            status="incomplete", error="fixture inspect failed", profile=None
        )
        with self.assertRaisesRegex(ValueError, "fixture inspect failed"):
            self.service.method_options(
                "local-cifar10", QuickStartNoiseSelection("clean", "clean")
            )

    def test_exact_reproduction_cannot_bypass_preflight(self) -> None:
        compatible = SimpleNamespace(
            status=CompatibilityStatus.COMPATIBLE,
            reasons=(),
            required_user_inputs=(),
        )
        paper = next(item for item in __import__("lnl_toolbox.catalog", fromlist=["load_papers"]).load_papers() if item.id == "gce")
        with patch("lnl_toolbox.catalog.paper_by_id", return_value=paper), \
             patch("lnl_toolbox.quickstart.service.find_exact_reproduction", return_value="fake-recipe"), \
             patch("lnl_toolbox.quickstart.service.recipe_by_id", return_value=SimpleNamespace()), \
             patch("lnl_toolbox.quickstart.service.load_recipe_config", return_value={"data": {"name": "cifar10"}}), \
             patch.object(self.data_service, "apply", return_value={"data": {"name": "cifar10"}}), \
             patch.object(self.service.experiment_service, "list_config_compatibility", return_value=(("gce", compatible),)), \
             patch.object(self.service.experiment_service, "preflight", side_effect=ValueError("preflight failed")):
            plan = self.service.build_plan(
                dataset_alias="local-cifar10",
                noise_selection=QuickStartNoiseSelection("clean", "clean"),
                paper_id="gce",
            )
        self.assertEqual(plan.status, "unsupported")
        self.assertIsNone(plan.command)
        self.assertIn("preflight failed", plan.details)

    def test_exact_reproduction_preserves_native_noise_and_returns_recipe_id(self) -> None:
        clean_paper = SimpleNamespace(configs=(SimpleNamespace(profile="reproduction", recipe_id="formal-clean"),))
        native_paper = SimpleNamespace(configs=(SimpleNamespace(profile="reproduction", recipe_id="formal-native"),))
        def config_for(recipe_id):
            return {"data": {"name": "cifar10"}} if recipe_id == "formal-clean" else {
                "data": {"name": "cifar10"}, "noise": {"name": "native"},
            }
        with patch("lnl_toolbox.quickstart.templates._cached_recipe_config", side_effect=config_for):
            self.assertIsNone(find_exact_reproduction(
                clean_paper, dataset_adapter="cifar10",
                noise_selection=QuickStartNoiseSelection("native", "native"),
            ))
            self.assertEqual(find_exact_reproduction(
                native_paper, dataset_adapter="cifar10",
                noise_selection=QuickStartNoiseSelection("synthetic", "native"),
            ), "formal-native")
            self.assertEqual(find_exact_reproduction(
                clean_paper, dataset_adapter="cifar10",
                noise_selection=QuickStartNoiseSelection("clean", "clean"),
            ), "formal-clean")


    def test_all_ready_options_build_ready_plans_across_registered_class_spaces(self) -> None:
        root = Path(self.temp.name)
        cifar100_service = DataService(
            registry=DatasetRegistry((_FakeCifar100Adapter(),)),
            catalog=LocalDatasetCatalog(root / "cifar100-catalog.json"),
        )
        cifar100_service.register(
            "local-cifar100", "cifar100", {"root": str(self.data_root)}
        )
        fashion_service = DataService(
            registry=DatasetRegistry((_FakeFashionMnistAdapter(),)),
            catalog=LocalDatasetCatalog(root / "fashion-catalog.json"),
        )
        fashion_service.register(
            "fashion-mnist", "fashion_mnist", {"root": str(self.data_root)}
        )
        scenarios = (
            ("local-cifar10", self.service),
            ("local-cifar100", QuickStartService(
                cifar100_service, artifact_root=root / "cifar100-artifacts"
            )),
            ("fashion-mnist", QuickStartService(
                fashion_service, artifact_root=root / "fashion-artifacts"
            )),
        )
        noises = (
            QuickStartNoiseSelection("clean", "clean"),
            QuickStartNoiseSelection("synthetic", "symmetric", rate=0.2, seed=1),
            QuickStartNoiseSelection("synthetic", "pairflip", rate=0.2, seed=1),
            QuickStartNoiseSelection("synthetic", "pdl", rate=0.2, seed=1),
        )
        for noise in noises:
            for alias, service in scenarios:
                options = service.method_options(alias, noise)
                self.assertEqual(len(options), 26)
                for option in options:
                    if option.status != "ready":
                        continue
                    with self.subTest(
                        dataset=alias, noise=noise.key, paper=option.paper_id,
                    ):
                        plan = service.build_plan(
                            dataset_alias=alias,
                            noise_selection=noise,
                            paper_id=option.paper_id,
                        )
                        self.assertEqual(plan.status, "ready", plan.details)

    def test_generic_templates_rebind_classes_and_discard_stale_transition_matrix(self) -> None:
        root = Path(self.temp.name)
        service = DataService(
            registry=DatasetRegistry((_FakeCifar100Adapter(),)),
            catalog=LocalDatasetCatalog(root / "class-binding-catalog.json"),
        )
        service.register("local-cifar100", "cifar100", {"root": str(self.data_root)})
        profile = service.inspect("local-cifar100").profile.to_dict()
        noise = QuickStartNoiseSelection("synthetic", "symmetric", rate=0.2, seed=1)
        candidates = {}
        for paper_id in ("mc-ldce", "dss", "loss-correction"):
            paper = next(item for item in load_papers() if item.id == paper_id)
            template = method_template_for_paper(paper)
            candidates[paper_id] = adapt_method_template(
                template.config,
                dataset_alias="local-cifar100",
                dataset_profile=profile,
                noise_selection=noise,
                data_service=service,
            )
        self.assertTrue(all(
            candidate["data"]["num_classes"] == 100
            for candidate in candidates.values()
        ))
        self.assertEqual(candidates["mc-ldce"]["transition"]["num_classes"], 100)
        self.assertEqual(
            candidates["dss"]["pipeline"]["objective_consumer"]["num_classes"], 100
        )
        estimator = candidates["loss-correction"]["pipeline"]["transition_estimator"]
        self.assertNotIn("matrix", estimator)

    def test_clean_upm_needs_input_and_symmetric_upm_builds_ready_plan(self) -> None:
        clean = QuickStartNoiseSelection("clean", "clean")
        clean_option = next(
            item for item in self.service.method_options("local-cifar10", clean)
            if item.paper_id == "upm"
        )
        self.assertEqual(clean_option.status, "needs_input")
        self.assertIn("config:requires_noisy_training_labels", clean_option.required_user_inputs)

        symmetric = QuickStartNoiseSelection(
            "synthetic", "symmetric", rate=0.2, seed=1
        )
        option = next(
            item for item in self.service.method_options("local-cifar10", symmetric)
            if item.paper_id == "upm"
        )
        self.assertEqual(option.status, "ready")
        plan = self.service.build_plan(
            dataset_alias="local-cifar10",
            noise_selection=symmetric,
            paper_id="upm",
        )
        self.assertEqual(plan.status, "ready", plan.details)


if __name__ == "__main__":
    unittest.main()
