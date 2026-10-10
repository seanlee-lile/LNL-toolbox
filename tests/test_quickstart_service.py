from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lnl_toolbox.data.contracts import DataSpec, RawDatasetSplit, UnsupportedDatasetSplitError
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
        if split == "validation":
            raise UnsupportedDatasetSplitError("fixture has no native validation split")
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


class _FakeUnverifiedLabelsAdapter(_FakeImageAdapter):
    name = "custom_labels"

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del spec, seed
        if split == "validation":
            raise UnsupportedDatasetSplitError("fixture has no native validation split")
        count = 12 if split == "train" else 6
        labels = np.arange(count, dtype=np.int64) % 2
        return RawDatasetSplit(
            np.zeros((count, 32, 32, 3), dtype=np.uint8), labels,
            np.arange(count, dtype=np.int64), self.name, split, 2,
        )


class _FakeNoisyImageAdapter(_FakeImageAdapter):
    name = "noisy_image"

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        clean = super().load(spec, split, seed=seed)
        observed = np.array(clean.observed_targets, copy=True)
        if split == "train":
            observed[0] = 1 - observed[0]
        return RawDatasetSplit(
            clean.inputs, observed, clean.global_indices, self.name, split, 2,
            clean_targets=clean.clean_targets,
        )


class QuickStartServiceTests(unittest.TestCase):
    def test_all_papers_keep_original_counts_and_one_web_validation_source(self) -> None:
        from copy import deepcopy
        from lnl_toolbox.quickstart.templates import configure_validation_split, SPLIT_COUNT_PATHS

        covered = []
        for paper in load_papers():
            original = method_template_for_paper(paper).config
            before = deepcopy(original)
            candidate = configure_validation_split(deepcopy(original))
            covered.append(paper.id)
            if paper.id == "cwd":
                self.assertEqual(candidate, before)
                continue
            self.assertEqual(candidate["data"]["validation_split"]["source"], "training_pool", paper.id)
            paths = []
            for path in SPLIT_COUNT_PATHS:
                if path[-1] == "num_clean":
                    continue
                old, new = before, candidate
                for key in path:
                    old = old.get(key) if isinstance(old, dict) else None
                    new = new.get(key) if isinstance(new, dict) else None
                if old is not None:
                    self.assertEqual(new, old, (paper.id, path))
                if new is not None:
                    paths.append(path)
            self.assertEqual(len(paths), 1, paper.id)
            self.assertEqual(original, before, paper.id)
        self.assertEqual(len(covered), 26)

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
        self.assertEqual(result["dataset_state"], "clean")
        self.assertEqual(result["status_source"], "inspected")
        self.assertEqual(result["clean_train_labels"], "available")
        keys = {item["key"] for item in result["options"]}
        self.assertIn("symmetric", keys)
        self.assertNotIn("external_torch", keys)
        self.assertIn("binary_asymmetric_rcn", keys)

    def test_binary_asymmetric_selection_reaches_plan(self) -> None:
        selection = QuickStartNoiseSelection(
            "synthetic", "binary_asymmetric_rcn", seed=1,
            rho_positive=0.3, rho_negative=0.05,
        )
        profile = self.data_service.status("local-cifar10").profile.to_dict()
        paper = next(item for item in load_papers() if item.id == "importance-reweighting")
        config = adapt_method_template(
            method_template_for_paper(paper).config,
            dataset_alias="local-cifar10", dataset_profile=profile,
            noise_selection=selection, data_service=self.data_service,
        )
        self.assertEqual(config["noise"]["rho_positive"], 0.3)
        self.assertEqual(config["noise"]["rho_negative"], 0.05)
        plan = self.service.build_plan(
            dataset_alias="local-cifar10", noise_selection=selection,
            paper_id="importance-reweighting",
        )
        self.assertEqual(plan.status, "ready", plan.details)
        from lnl_toolbox.catalog import load_yaml
        saved = load_yaml(Path(plan.generated_config_path))
        self.assertEqual(saved["noise"]["rho_positive"], 0.3)
        self.assertEqual(saved["noise"]["rho_negative"], 0.05)

    def test_binary_risk_rates_follow_selected_noise(self) -> None:
        paper = next(item for item in load_papers() if item.id == "binary-risk")
        profile = self.data_service.status("local-cifar10").profile.to_dict()
        config = adapt_method_template(
            method_template_for_paper(paper).config,
            dataset_alias="local-cifar10", dataset_profile=profile,
            noise_selection=QuickStartNoiseSelection(
                "synthetic", "binary_asymmetric_rcn", seed=1,
                rho_positive=0.2, rho_negative=0.1,
            ),
            data_service=self.data_service,
        )
        self.assertEqual(config["noise"]["rho_positive"], 0.2)
        self.assertEqual(config["noise"]["rho_negative"], 0.1)
        self.assertEqual(config["risk"]["rho_positive"], 0.2)
        self.assertEqual(config["risk"]["rho_negative"], 0.1)

        symmetric = adapt_method_template(
            method_template_for_paper(paper).config,
            dataset_alias="local-cifar10", dataset_profile=profile,
            noise_selection=QuickStartNoiseSelection(
                "synthetic", "symmetric", rate=0.25, seed=1,
            ),
            data_service=self.data_service,
        )
        self.assertEqual(symmetric["risk"]["rho_positive"], 0.25)
        self.assertEqual(symmetric["risk"]["rho_negative"], 0.25)

        clean = adapt_method_template(
            method_template_for_paper(paper).config,
            dataset_alias="local-cifar10", dataset_profile=profile,
            noise_selection=QuickStartNoiseSelection("clean", "clean", seed=1),
            data_service=self.data_service,
        )
        self.assertNotIn("rho_positive", clean["risk"])
        self.assertNotIn("rho_negative", clean["risk"])

    def test_known_transition_follows_selected_synthetic_noise(self) -> None:
        paper = next(item for item in load_papers() if item.id == "loss-correction")
        profile = self.data_service.status("local-cifar10").profile.to_dict()
        profile["num_classes"] = 3
        for noise_name, expected in (
            ("symmetric", [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
            ("pairflip", [[0.8, 0.2, 0.0], [0.0, 0.8, 0.2], [0.2, 0.0, 0.8]]),
        ):
            with self.subTest(noise_name=noise_name):
                config = adapt_method_template(
                    method_template_for_paper(paper).config,
                    dataset_alias="local-cifar10", dataset_profile=profile,
                    noise_selection=QuickStartNoiseSelection(
                        "synthetic", noise_name, rate=0.2, seed=1,
                    ),
                    data_service=self.data_service,
                )
                self.assertEqual(
                    config["pipeline"]["transition_estimator"]["matrix"], expected,
                )

    def test_method_noise_prior_does_not_keep_old_recipe_rate(self) -> None:
        paper = next(item for item in load_papers() if item.id == "coteaching")
        profile = self.data_service.status("local-cifar10").profile.to_dict()
        base = method_template_for_paper(paper).config
        selected = adapt_method_template(
            base, dataset_alias="local-cifar10", dataset_profile=profile,
            noise_selection=QuickStartNoiseSelection(
                "synthetic", "symmetric", rate=0.3, seed=1,
            ), data_service=self.data_service,
        )
        self.assertEqual(selected["coteaching"]["noise_rate"], 0.3)
        clean = adapt_method_template(
            base, dataset_alias="local-cifar10", dataset_profile=profile,
            noise_selection=QuickStartNoiseSelection("clean", "clean", seed=1),
            data_service=self.data_service,
        )
        self.assertNotIn("noise_rate", clean["coteaching"])

    def test_unknown_labels_are_not_treated_as_clean_and_noisy_rate_can_be_declared(self) -> None:
        root = Path(self.temp.name)
        data_service = DataService(
            registry=DatasetRegistry((_FakeUnverifiedLabelsAdapter(),)),
            catalog=LocalDatasetCatalog(root / "unknown-catalog.json"),
        )
        data_service.register("unverified", "custom_labels", {"root": str(self.data_root)})
        data_service.inspect("unverified")
        service = QuickStartService(data_service, artifact_root=root / "unknown-artifacts")
        self.assertEqual(service.noise_options("unverified")["dataset_state"], "unknown")
        data_service.update_declarations("unverified", {"noise_status":"noisy", "noise_origin":"native"})
        unknown_rate = service.noise_options("unverified")
        self.assertEqual(unknown_rate["dataset_state"], "native")
        self.assertEqual(unknown_rate["noise_rate"]["status"], "unknown")
        self.assertEqual([item["key"] for item in unknown_rate["options"]], ["native"])
        data_service.update_declarations("unverified", {"noise_rate":{
            "status":"estimated", "value":0.25, "provenance":"local_measurement",
        }})
        self.assertEqual(service.noise_options("unverified")["noise_rate"]["value"], 0.25)

    def test_declared_noise_status_is_reflected_in_quick_start(self) -> None:
        root = Path(self.temp.name)
        data_service = DataService(
            registry=DatasetRegistry((_FakeUnverifiedLabelsAdapter(),)),
            catalog=LocalDatasetCatalog(root / "declared-catalog.json"),
        )
        data_service.register("unverified", "custom_labels", {"root": str(self.data_root)})
        data_service.inspect("unverified")
        service = QuickStartService(data_service, artifact_root=root / "declared-artifacts")
        self.assertEqual(service.noise_options("unverified")["dataset_state"], "unknown")
        data_service.update_declarations("unverified", {"noise_status": "clean"})
        clean = service.noise_options("unverified")
        self.assertEqual(clean["dataset_state"], "clean")
        self.assertEqual(clean["status_source"], "declared")
        self.assertIn("symmetric", {item["key"] for item in clean["options"]})
        data_service.update_declarations("unverified", {"noise_status": "noisy", "noise_origin": "native"})
        noisy = service.noise_options("unverified")
        self.assertEqual(noisy["dataset_state"], "native")
        self.assertEqual([item["key"] for item in noisy["options"]], ["native"])

    def test_user_noise_settings_override_inspection_and_persist(self) -> None:
        self.data_service.update_declarations("local-cifar10", {
            "noise_status": "noisy", "noise_origin": "native",
            "noise_rate": {"status": "known", "value": 0.2},
        })
        result = self.service.noise_options("local-cifar10")
        self.assertEqual(result["dataset_state"], "native")
        self.assertEqual(result["noise_rate"]["value"], 0.2)
        self.assertEqual(result["status_source"], "declared")
        self.assertEqual(self.data_service.record("local-cifar10").profile["noise"]["status"], "clean")
        reloaded = DataService(registry=self.data_service.registry, catalog=self.data_service.catalog)
        fresh = QuickStartService(reloaded, artifact_root=Path(self.temp.name) / "reopened")
        self.assertEqual(fresh.noise_options("local-cifar10")["noise_rate"]["value"], 0.2)

    def test_user_can_clear_noise_rate_and_change_back_to_unknown_or_clean(self) -> None:
        self.data_service.update_declarations("local-cifar10", {
            "noise_status": "noisy", "noise_rate": {"status": "known", "value": 0.3},
        })
        self.data_service.update_declarations("local-cifar10", {"noise_rate": {"status": "unknown"}})
        self.assertEqual(self.service.noise_options("local-cifar10")["noise_rate"]["status"], "unknown")
        self.data_service.update_declarations("local-cifar10", {"noise_status": "unknown"})
        self.assertEqual(self.service.noise_options("local-cifar10")["dataset_state"], "unknown")
        self.data_service.update_declarations("local-cifar10", {"noise_status": "clean"})
        clean = self.service.noise_options("local-cifar10")
        self.assertEqual(clean["dataset_state"], "clean")
        self.assertEqual(clean["noise_rate"]["status"], "not_applicable")
        self.assertIsNone(clean["noise_rate"]["value"])

    def test_measured_noise_rate_can_be_edited_and_cleared(self) -> None:
        root = Path(self.temp.name)
        data_service = DataService(registry=DatasetRegistry((_FakeNoisyImageAdapter(),)),
                                   catalog=LocalDatasetCatalog(root / "edited-noisy.json"))
        data_service.register("noisy", "noisy_image", {"root": str(self.data_root)})
        data_service.inspect("noisy")
        service = QuickStartService(data_service, artifact_root=root / "edited-noisy-artifacts")
        data_service.update_declarations("noisy", {"noise_rate": {
            "status": "estimated", "value": 0.4, "provenance": "user measurement",
        }})
        result = service.noise_options("noisy")
        self.assertEqual(result["dataset_state"], "native")
        self.assertEqual(result["noise_rate"]["value"], 0.4)
        self.assertEqual(result["status_source"], "declared")
        self.assertAlmostEqual(data_service.record("noisy").profile["noise"]["rate"]["value"], 1 / 12)
        data_service.update_declarations("noisy", {"noise_rate": {"status": "unknown"}})
        result = service.noise_options("noisy")
        self.assertEqual(result["noise_rate"]["status"], "unknown")
        self.assertIsNone(result["noise_rate"]["value"])

    def test_inspected_noisy_labels_publish_measured_original_rate(self) -> None:
        root = Path(self.temp.name)
        data_service = DataService(
            registry=DatasetRegistry((_FakeNoisyImageAdapter(),)),
            catalog=LocalDatasetCatalog(root / "noisy-catalog.json"),
        )
        data_service.register("noisy", "noisy_image", {"root": str(self.data_root)})
        data_service.inspect("noisy")
        result = QuickStartService(data_service, artifact_root=root / "noisy-artifacts").noise_options("noisy")
        self.assertEqual(result["dataset_state"], "native")
        self.assertEqual(result["status_source"], "inspected")
        self.assertAlmostEqual(result["noise_rate"]["value"], 1 / 12)
        self.assertEqual([item["key"] for item in result["options"]], ["native"])

    def test_method_options_are_paper_catalog_driven(self) -> None:
        result = self.service.method_options(
            "local-cifar10", QuickStartNoiseSelection("clean", "clean")
        )
        self.assertGreater(len(result), 5)
        self.assertTrue(all(item.acronym for item in result))
        fine = next(item for item in result if item.paper_id == "fine")
        self.assertEqual(fine.status, "ready")

    def test_internal_pcse_does_not_require_upm_environment(self) -> None:
        root = Path(self.temp.name)
        service = DataService(
            registry=DatasetRegistry((_FakeCifar100Adapter(),)),
            catalog=LocalDatasetCatalog(root / "pcse-catalog.json"),
        )
        service.register(
            "local-cifar100", "cifar100", {"root": str(self.data_root)}
        )
        service.inspect("local-cifar100")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LNL_PCSE_SOURCE_RUN", None)
            option = next(
                item
                for item in QuickStartService(
                    service, artifact_root=root / "pcse-artifacts"
                ).method_options(
                    "local-cifar100",
                    QuickStartNoiseSelection("synthetic", "symmetric", rate=0.2, seed=1),
                )
                if item.paper_id == "pcse"
            )
        self.assertNotEqual(option.status, "needs_input")
        self.assertNotIn("LNL_PCSE_SOURCE_RUN", option.required_user_inputs)

    def test_method_options_batches_compatibility_and_reuses_cache(self) -> None:
        selection = QuickStartNoiseSelection("clean", "clean")
        original = self.service.experiment_service.list_config_compatibility
        with patch.object(
            self.service.experiment_service,
            "list_config_compatibility",
            wraps=original,
        ) as checker, patch(
            "lnl_toolbox.quickstart.templates._cached_recipe_config",
            side_effect=AssertionError("method scan must reuse discovered recipes"),
        ):
            first = self.service.method_options("local-cifar10", selection)
            second = self.service.method_options("local-cifar10", selection)
        self.assertEqual(first, second)
        self.assertEqual(checker.call_count, 1)
        self.assertGreater(len(first), 5)

    def test_method_options_use_accepted_profile_without_dataset_inspection(self) -> None:
        selection = QuickStartNoiseSelection("clean", "clean")
        with patch.object(self.service.data_service, "inspect", side_effect=AssertionError("unexpected reinspection")):
            first = self.service.method_options("local-cifar10", selection)
            second = self.service.method_options("local-cifar10", selection)
        self.assertEqual(first, second)

    def test_accepted_quick_start_flow_does_not_reload_dataset(self) -> None:
        selection = QuickStartNoiseSelection("synthetic", "symmetric", rate=0.2, seed=1)
        with patch.object(self.data_service.registry, "load", side_effect=AssertionError("unexpected sample load")):
            self.assertEqual(self.service.register_and_inspect(str(self.data_root)).alias, "local-cifar10")
            self.assertIn("symmetric", {item["key"] for item in self.service.noise_options("local-cifar10")["options"]})
            methods = self.service.method_options("local-cifar10", selection)
            self.assertTrue(any(item.paper_id == "gce" for item in methods))
            plan = self.service.build_plan(
                dataset_alias="local-cifar10", noise_selection=selection, paper_id="gce",
            )
        self.assertEqual(plan.status, "ready", plan.details)

    def test_dividemix_does_not_request_a_separate_noise_rate_prior(self) -> None:
        clean = QuickStartNoiseSelection("clean", "clean")
        option = next(
            item for item in self.service.method_options("local-cifar10", clean)
            if item.paper_id == "dividemix"
        )
        self.assertIn("config:requires_noisy_training_labels", option.required_user_inputs)
        self.assertNotIn("noise_rate_prior", option.required_user_inputs)

        synthetic = QuickStartNoiseSelection("synthetic", "symmetric", rate=0.2, seed=1)
        option = next(
            item for item in self.service.method_options("local-cifar10", synthetic)
            if item.paper_id == "dividemix"
        )
        self.assertNotIn("noise_rate_prior", option.required_user_inputs)

    def test_unknown_path_is_not_marked_ready(self) -> None:
        result = self.service.register_and_inspect(str(Path(self.temp.name) / "missing"))
        self.assertEqual(result.status, "unsupported")

    def test_inspect_failure_stops_registration_flow(self) -> None:
        self.data_service.status = lambda *args, **kwargs: SimpleNamespace(
            status="incomplete", error="fixture status failed", profile=None
        )
        self.data_service.inspect = lambda *args, **kwargs: SimpleNamespace(
            status="incomplete", error="fixture inspect failed", profile=None
        )
        with patch.object(self.service, "probe", return_value=DatasetProbeResult(
            str(self.data_root), "already_registered", existing_alias="local-cifar10"
        )):
            with self.assertRaisesRegex(ValueError, "fixture inspect failed"):
                self.service.register_and_inspect(str(self.data_root))

    def test_method_options_rejects_an_incomplete_profile(self) -> None:
        self.data_service.status = lambda *args, **kwargs: SimpleNamespace(
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
             patch("lnl_toolbox.quickstart.service.reference_train_size", return_value=12), \
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

    def test_adapted_split_counts_follow_paper_fraction_on_mini_dataset(self) -> None:
        profile = self.data_service.status("local-cifar10").profile.to_dict()
        profile["sample_counts_by_split"]["train"] = 3000
        noise = QuickStartNoiseSelection("synthetic", "symmetric", rate=0.4, seed=1234)
        expected = {
            "l2rw": ("data", "num_val", 300),
            "gce": ("data", "validation_size", 300),
            "pdl": ("warmup", "noisy_validation_size", 300),
        }
        for paper in load_papers():
            if paper.id not in expected:
                continue
            section, key, count = expected[paper.id]
            candidate = adapt_method_template(
                method_template_for_paper(paper).config,
                dataset_alias="local-cifar10",
                dataset_profile=profile,
                noise_selection=noise,
                data_service=self.data_service,
            )
            self.assertEqual(candidate[section][key], count, paper.id)
            if paper.id == "l2rw":
                self.assertEqual(candidate["data"]["num_clean"], 6)
            profile["sample_counts_by_split"].update(train=2700, validation=300)
            restored = adapt_method_template(
                method_template_for_paper(paper).config,
                dataset_alias="local-cifar10", dataset_profile=profile,
                noise_selection=noise, data_service=self.data_service,
            )
            self.assertEqual(restored[section][key], count, paper.id)
            profile["sample_counts_by_split"].update(train=3000, validation=0)

    def test_small_dataset_uses_adapted_plan_instead_of_full_size_recipe(self) -> None:
        from lnl_toolbox.catalog import load_yaml

        plan = self.service.build_plan(
            dataset_alias="local-cifar10",
            noise_selection=QuickStartNoiseSelection("synthetic", "symmetric", rate=0.2, seed=1),
            paper_id="gce",
        )
        self.assertEqual(plan.config_kind, "toolbox_adapted")
        self.assertEqual(plan.status, "ready", plan.details)
        self.assertEqual(load_yaml(Path(plan.generated_config_path))["data"]["validation_size"], 1)

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

    def test_exact_reproduction_reuses_discovered_recipe(self) -> None:
        recipe = SimpleNamespace(id="formal-clean")
        paper = SimpleNamespace(configs=(SimpleNamespace(profile="reproduction", recipe_id=recipe.id),))
        with patch("lnl_toolbox.quickstart.templates._cached_recipe_config", side_effect=AssertionError("rescan")), \
             patch("lnl_toolbox.quickstart.templates.load_recipe_config", return_value={"data": {"name": "cifar10"}}) as load:
            result = find_exact_reproduction(
                paper, dataset_adapter="cifar10",
                noise_selection=QuickStartNoiseSelection("clean", "clean"),
                recipes={recipe.id: recipe},
            )
        self.assertEqual(result, recipe.id)
        load.assert_called_once_with(recipe)

    def test_changed_experiment_seed_is_not_mistaken_for_exact_recipe(self) -> None:
        paper = SimpleNamespace(configs=(SimpleNamespace(profile="reproduction", recipe_id="formal-gce"),))
        with patch("lnl_toolbox.quickstart.templates._cached_recipe_config", return_value={
            "seed": 1, "data": {"name": "cifar10"},
            "noise": {"name": "symmetric", "rate": 0.2, "seed": 1},
        }):
            self.assertIsNone(find_exact_reproduction(
                paper, dataset_adapter="cifar10",
                noise_selection=QuickStartNoiseSelection("synthetic", "symmetric", rate=0.2, seed=7),
            ))


    def test_all_ready_options_build_ready_plans_across_registered_class_spaces(self) -> None:
        root = Path(self.temp.name)
        cifar100_service = DataService(
            registry=DatasetRegistry((_FakeCifar100Adapter(),)),
            catalog=LocalDatasetCatalog(root / "cifar100-catalog.json"),
        )
        cifar100_service.register(
            "local-cifar100", "cifar100", {"root": str(self.data_root)}
        )
        cifar100_service.inspect("local-cifar100")
        fashion_service = DataService(
            registry=DatasetRegistry((_FakeFashionMnistAdapter(),)),
            catalog=LocalDatasetCatalog(root / "fashion-catalog.json"),
        )
        fashion_service.register(
            "fashion-mnist", "fashion_mnist", {"root": str(self.data_root)}
        )
        fashion_service.inspect("fashion-mnist")
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
                        if option.paper_id == "pcse" and noise.kind == "clean":
                            self.assertEqual(plan.status, "unsupported")
                            self.assertFalse(plan.command)
                        elif option.paper_id == "mentornet":
                            if plan.status == "ready":
                                self.assertTrue(plan.command)
                            else:
                                self.assertIn(plan.status, {"needs_input", "unsupported"})
                                self.assertFalse(plan.command)
                        else:
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
