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
from lnl_toolbox.quickstart.service import QuickStartService, _class_space_problems
from lnl_toolbox.quickstart.templates import adapt_method_template, method_template_for_paper
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
        self.assertEqual(fine.status, "unsupported")

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

    def test_all_visible_noise_choices_respect_registered_class_spaces(self) -> None:
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
            ("local-cifar10", self.data_service, "cifar10", 10,
             {"importance-reweighting"}),
            ("local-cifar100", cifar100_service, "cifar100", 100,
             {"importance-reweighting", "loss-correction", "mc-ldce", "pcse", "dss"}),
            ("fashion-mnist", fashion_service, "fashion_mnist", 10,
             {"importance-reweighting"}),
        )
        noises = (
            QuickStartNoiseSelection("clean", "clean"),
            QuickStartNoiseSelection("synthetic", "symmetric", rate=0.2, seed=1),
            QuickStartNoiseSelection("synthetic", "pairflip", rate=0.2, seed=1),
            QuickStartNoiseSelection("synthetic", "pdl", rate=0.2, seed=1),
        )
        no_requirements_runner = SimpleNamespace(requirements=lambda config: None)
        with patch(
            "lnl_toolbox.training.runners.resolve_runner",
            return_value=no_requirements_runner,
        ):
            for noise in noises:
                for alias, data_service, adapter, num_classes, fixed_templates in scenarios:
                    for paper in load_papers():
                        template = method_template_for_paper(paper)
                        candidate = adapt_method_template(
                            template.config,
                            dataset_alias=alias,
                            dataset_profile={"adapter": adapter},
                            noise_selection=noise,
                            data_service=data_service,
                        )
                        problems = _class_space_problems(candidate, num_classes=num_classes)
                        with self.subTest(
                            dataset=alias, noise=noise.key, paper=paper.id,
                        ):
                            self.assertEqual(
                                bool(problems), paper.id in fixed_templates,
                            )

        loss_correction = next(item for item in load_papers() if item.id == "loss-correction")
        service = QuickStartService(cifar100_service)
        report = SimpleNamespace(
            status="ready",
            error=None,
            profile=SimpleNamespace(num_classes=100, to_dict=lambda: {"adapter": "cifar100"}),
        )
        with patch.object(cifar100_service, "inspect", return_value=report), \
             patch("lnl_toolbox.quickstart.service.load_papers", return_value=(loss_correction,)):
            option = service.method_options(
                "local-cifar100", QuickStartNoiseSelection("clean", "clean")
            )[0]
        self.assertEqual(option.status, "unsupported")
        self.assertIn("100 类", option.reasons[0])
        with patch.object(cifar100_service, "inspect", return_value=report):
            plan = service.build_plan(
                dataset_alias="local-cifar100",
                noise_selection=QuickStartNoiseSelection("clean", "clean"),
                paper_id="loss-correction",
            )
        self.assertEqual(plan.status, "unsupported")
        self.assertIsNone(plan.command)


if __name__ == "__main__":
    unittest.main()
