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


class _FakeTenClassImageAdapter(_FakeImageAdapter):
    def __init__(self, name: str = "cifar10") -> None:
        self.name = name

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        if split == "validation":
            raise UnsupportedDatasetSplitError("fixture has no native validation split")
        count = 100 if split == "train" else 20
        labels = np.arange(count, dtype=np.int64) % 10
        shape = (28, 28, 1) if self.name == "fashion_mnist" else (32, 32, 3)
        return RawDatasetSplit(
            np.zeros((count, *shape), dtype=np.uint8), labels,
            np.arange(count, dtype=np.int64), self.name, split, 10,
            clean_targets=labels, source="ten-class-fixture",
        )


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
    def test_plan_template_cache_tracks_recipe_file_revision(self) -> None:
        from lnl_toolbox.quickstart import templates
        from lnl_toolbox.catalog import recipe_by_id
        from dataclasses import replace
        import yaml

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.yaml"
            path.write_text("lend: {selection: {reduction: sum}}", encoding="utf-8")
            recipe = replace(recipe_by_id("lend-cifar10-reproduction"), config_path=path)
            templates._recipe_config_revision.cache_clear()
            with patch.object(templates, "_cached_recipe_by_id", return_value=recipe), \
                 patch.object(templates, "load_recipe_config", side_effect=lambda recipe: yaml.safe_load(path.read_text(encoding="utf-8"))) as load:
                first = templates._cached_recipe_config(recipe.id)
                templates._cached_recipe_config(recipe.id)
                self.assertEqual(load.call_count, 1)
                path.write_text("lend: {selection: {reduction: batch_mean}}", encoding="utf-8")
                second = templates._cached_recipe_config(recipe.id)
                self.assertEqual(load.call_count, 2)
                self.assertEqual(first["lend"]["selection"]["reduction"], "sum")
                self.assertEqual(second["lend"]["selection"]["reduction"], "batch_mean")

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
        self.assertEqual(set(covered), {paper.id for paper in load_papers()})

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
        self.assertTrue({"external_torch", "class_conditional", "official_uniform_flip"} <= keys)
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

    def test_class_conditional_matrix_survives_plan_and_noise_generation(self) -> None:
        from lnl_toolbox.catalog import load_yaml
        from lnl_toolbox.training.noisy_labels import prepare_noise_manifest

        matrix = ((0.7, 0.3), (0.1, 0.9))
        selection = QuickStartNoiseSelection(
            "synthetic", "class_conditional", rate=0.2, seed=1,
            transition_matrix=matrix,
        )
        plan = self.service.build_plan(
            dataset_alias="local-cifar10", noise_selection=selection, paper_id="gce",
        )
        self.assertEqual(plan.status, "ready", plan.details)
        saved = load_yaml(Path(plan.generated_config_path))
        np.testing.assert_array_equal(saved["noise"]["transition_matrix"], matrix)
        labels = np.arange(12, dtype=np.int64) % 2
        manifest, _ = prepare_noise_manifest(
            saved, dataset="cifar10", clean_targets=labels,
            global_indices=np.arange(labels.size), num_classes=2,
            run_dir=Path(self.temp.name),
        )
        np.testing.assert_array_equal(manifest.transition_matrix, matrix)

    def test_replacing_noise_matrix_or_path_invalidates_only_stale_matrices(self) -> None:
        from copy import deepcopy

        paper = next(item for item in load_papers() if item.id == "mc-ldce")
        service = DataService(
            registry=DatasetRegistry((_FakeTenClassImageAdapter(),)),
            catalog=LocalDatasetCatalog(Path(self.temp.name) / "matrix-catalog.json"),
        )
        service.register("ten-class", "cifar10", {"root": str(self.data_root)})
        profile = service.inspect("ten-class").profile.to_dict()
        original_matrix = (0.8 * np.eye(10) + (0.2 / 9) * (1 - np.eye(10))).tolist()
        for key in ("class_conditional", "external_torch"):
            for changed in (False, True):
                with self.subTest(key=key, changed=changed):
                    base = deepcopy(method_template_for_paper(paper).config)
                    base["noise"] = {
                        "name": key, "rate": 0.2, "rng": "numpy_legacy",
                        "transition_matrix": original_matrix, "path": "old-labels.pt",
                    }
                    base["transition"]["matrix"] = deepcopy(original_matrix)
                    matrix = tuple(map(tuple, 0.8 * np.eye(10) + 0.2 * np.roll(np.eye(10), 1, axis=1))) if changed else tuple(map(tuple, original_matrix))
                    path = "new-labels.pt" if changed else "old-labels.pt"
                    config = adapt_method_template(
                        base, dataset_alias="ten-class", dataset_profile=profile,
                        noise_selection=QuickStartNoiseSelection(
                            "synthetic", key, rate=0.2, seed=1,
                            transition_matrix=matrix if key == "class_conditional" else None,
                            path=path if key == "external_torch" else None,
                        ), data_service=service,
                    )
                    self.assertEqual("matrix" in config["transition"], not changed)
                    if key == "class_conditional":
                        np.testing.assert_array_equal(config["noise"]["transition_matrix"], matrix)
                    else:
                        self.assertEqual(config["noise"]["path"], path)
                        if changed:
                            self.assertNotIn("transition_matrix", config["noise"])
                    self.assertEqual(base["transition"]["matrix"], original_matrix)

    def test_class_conditional_matrix_must_match_registered_classes(self) -> None:
        paper = next(item for item in load_papers() if item.id == "gce")
        with self.assertRaisesRegex(ValueError, "类别数"):
            adapt_method_template(
                method_template_for_paper(paper).config,
                dataset_alias="local-cifar10",
                dataset_profile=self.data_service.status("local-cifar10").profile.to_dict(),
                noise_selection=QuickStartNoiseSelection(
                    "synthetic", "class_conditional", rate=0.2, seed=1,
                    transition_matrix=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                ), data_service=self.data_service,
            )

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

    def test_loss_correction_automatically_estimates_unknown_T(self) -> None:
        paper = next(item for item in load_papers() if item.id == "loss-correction")
        profile = self.data_service.status("local-cifar10").profile.to_dict()
        options = dict(dataset_alias="local-cifar10", dataset_profile=profile,
                       noise_selection=QuickStartNoiseSelection("native", "native", rate=None, seed=1),
                       data_service=self.data_service)
        config = adapt_method_template(method_template_for_paper(paper).config,
            method_inputs={"pipeline.warmup_epochs": 2}, **options)
        self.assertEqual(config["pipeline"]["transition_estimator"], {"name": "anchor", "percentile": 97.0})
        self.assertEqual(config["pipeline"]["warmup_epochs"], 2)
        self.assertEqual(config["pipeline"]["pretraining_protocol"], "loss_correction")
        with self.assertRaises(ValueError):
            adapt_method_template(method_template_for_paper(paper).config,
                method_inputs={"pipeline.warmup_epochs": 0}, **options)

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

    def test_loss_correction_automatic_T_source_for_clean_and_instance_noise(self) -> None:
        paper = next(item for item in load_papers() if item.id == "loss-correction")
        profile = self.data_service.status("local-cifar10").profile.to_dict()
        for selection, expected in ((QuickStartNoiseSelection("clean", "clean"), "known"),
                                    (QuickStartNoiseSelection("synthetic", "pdl", rate=.2, seed=1), "anchor")):
            config = adapt_method_template(method_template_for_paper(paper).config,
                dataset_alias="local-cifar10", dataset_profile=profile, noise_selection=selection,
                data_service=self.data_service)
            self.assertEqual(config['pipeline']['transition_estimator']['name'], expected)
            if expected == 'known':
                np.testing.assert_array_equal(config['pipeline']['transition_estimator']['matrix'], np.eye(2))
                self.assertEqual(config['pipeline']['warmup_epochs'], 0)
            else:
                self.assertNotIn('matrix', config['pipeline']['transition_estimator'])
                self.assertEqual(config['pipeline']['warmup_epochs'], 120)
        profile['adapter'] = 'cifar100'
        config = adapt_method_template(method_template_for_paper(paper).config,
            dataset_alias="local-cifar10", dataset_profile=profile,
            noise_selection=QuickStartNoiseSelection('native', 'native'), data_service=self.data_service)
        self.assertIsNone(config['pipeline']['transition_estimator']['percentile'])


    def test_prior_adaptation_updates_keep_rate_for_both_peer_methods(self) -> None:
        from lnl_toolbox.catalog import validate_config
        profile = self.data_service.status("local-cifar10").profile.to_dict()
        for method in ("coteaching", "cnlcu"):
            paper = next(item for item in load_papers() if item.id == method)
            base = method_template_for_paper(paper).config
            for kind, key, rate in (("clean", "clean", None), ("synthetic", "symmetric", 0.3)):
                with self.subTest(method=method, kind=kind):
                    adapted = adapt_method_template(
                        base, dataset_alias="local-cifar10", dataset_profile=profile,
                        noise_selection=QuickStartNoiseSelection(kind, key, rate=rate, seed=1),
                        data_service=self.data_service,
                    )
                    expected_rate = 0.0 if kind == "clean" else rate
                    self.assertEqual(adapted[method]["noise_rate"], expected_rate)
                    self.assertAlmostEqual(adapted[method]["remember_schedule"]["end"], 1.0 - expected_rate)
                    self.assertEqual(validate_config(adapted).name, method)

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

    def test_clean_pcse_plan_preserves_labels_and_observed_validation(self) -> None:
        from lnl_toolbox.catalog import load_yaml
        from lnl_toolbox.algorithms.pcse.config import PCSEConfig

        root = Path(self.temp.name)
        data_service = DataService(
            registry=DatasetRegistry((_FakeCifar100Adapter(),)),
            catalog=LocalDatasetCatalog(root / "pcse-clean-catalog.json"),
        )
        data_service.register("pcse-clean", "cifar100", {"root": str(self.data_root)})
        data_service.inspect("pcse-clean")
        service = QuickStartService(data_service, artifact_root=root / "pcse-clean-artifacts")
        plan = service.build_plan(
            dataset_alias="pcse-clean",
            noise_selection=QuickStartNoiseSelection("clean", "clean", seed=1),
            paper_id="pcse",
        )
        self.assertEqual(plan.status, "ready", plan.details)
        self.assertTrue(plan.command)
        config = load_yaml(Path(plan.generated_config_path))
        self.assertEqual(config["noise"], {"name": "clean", "validation_targets": "noisy"})
        PCSEConfig.from_mapping(config)

    def test_no_corruption_preserves_label_contract_for_every_paper(self) -> None:
        profile = self.data_service.inspect("local-cifar10").profile.to_dict()
        for paper in load_papers():
            template = method_template_for_paper(paper)
            for kind in ("clean", "native"):
                with self.subTest(paper=paper.id, kind=kind):
                    candidate = adapt_method_template(
                        template.config, dataset_alias="local-cifar10",
                        dataset_profile=profile,
                        noise_selection=QuickStartNoiseSelection(kind, kind, rate=None if kind == "clean" else 0.2),
                        data_service=self.data_service,
                    )
                    original = template.config.get("noise", {}) or {}
                    if kind == "native" and paper.id == "loss-correction":
                        self.assertEqual(candidate["pipeline"]["transition_estimator"], {"name": "anchor", "percentile": 97.0})
                        self.assertGreater(candidate["pipeline"]["warmup_epochs"], 0)
                    self.assertIsInstance(candidate["noise"], dict)
                    self.assertEqual(candidate["noise"].get("validation_targets"),
                                     original.get("validation_targets"))
                    self.assertFalse(set(candidate["noise"]) &
                                     {"seed", "manifest", "transition_matrix", "sampling"})
                    if "rate" in candidate["noise"]:
                        from lnl_toolbox.training.runners import resolve_runner
                        requirements = resolve_runner(candidate).requirements(candidate)
                        self.assertIn("noise.rate", requirements.method_noise_prior_paths)
                        self.assertEqual(candidate["noise"]["rate"], 0.0 if kind == "clean" else 0.2)
                    if kind == "native" and paper.id in {
                        "coteaching", "dividemix", "dual-t", "upm", "volminnet", "cnlcu", "pcse"
                    }:
                        self.service.experiment_service.preflight(candidate, check_data=False)

    def test_jocor_estimate_defaults_and_native_unknown_input(self) -> None:
        from lnl_toolbox.catalog import load_yaml
        for kind, key, rate, expected in (
            ("clean", "clean", None, 0.0),
            ("synthetic", "symmetric", 0.3, 0.3),
        ):
            with self.subTest(kind=kind):
                plan = self.service.build_plan(
                    dataset_alias="local-cifar10", paper_id="jocor",
                    noise_selection=QuickStartNoiseSelection(kind, key, rate=rate, seed=1),
                )
                self.assertEqual(plan.status, "ready", plan.details)
                config = load_yaml(Path(plan.generated_config_path))
                self.assertEqual(config["algorithm"]["estimated_noise_rate"], expected)
                if kind == "synthetic":
                    self.assertEqual(config["noise"]["rate"], rate)
                else:
                    self.assertNotIn("rate", config["noise"])
        self.data_service.update_declarations("local-cifar10", {
            "noise_status": "noisy", "noise_origin": "native",
            "noise_rate": {"status": "known", "value": 0.2},
        })
        native = QuickStartNoiseSelection("native", "native")
        known = self.service.build_plan(dataset_alias="local-cifar10", paper_id="jocor", noise_selection=native)
        self.assertEqual(known.status, "ready", known.details)
        self.assertEqual(load_yaml(Path(known.generated_config_path))["algorithm"]["estimated_noise_rate"], 0.2)
        self.data_service.update_declarations("local-cifar10", {"noise_rate": {"status": "unknown"}})
        unknown = self.service.build_plan(dataset_alias="local-cifar10", paper_id="jocor", noise_selection=native)
        self.assertEqual(unknown.status, "needs_input")
        self.assertIn("noise_rate_prior", unknown.required_user_inputs)
        supplied = self.service.build_plan(dataset_alias="local-cifar10", paper_id="jocor", noise_selection=native,
                                           user_inputs={"noise_rate_prior": 0.4})
        self.assertEqual(supplied.status, "ready", supplied.details)
        config = load_yaml(Path(supplied.generated_config_path))
        self.assertEqual(config["algorithm"]["estimated_noise_rate"], 0.4)
        self.assertNotIn("rate", config["noise"])

    def test_invalid_class_dependent_parameter_is_rejected_in_method_options(self) -> None:
        selection = QuickStartNoiseSelection("synthetic", "symmetric", rate=0.2, seed=1)
        option = next(item for item in self.service.method_options("local-cifar10", selection)
                      if item.paper_id == "ca2c")
        self.assertEqual(option.status, "unsupported")
        self.assertTrue(any("candidate_k" in reason for reason in option.reasons))

    def test_ca2c_supports_ten_classes_without_changing_candidate_count(self) -> None:
        from lnl_toolbox.catalog import load_yaml
        root = Path(self.temp.name)
        for name in ("cifar10", "fashion_mnist"):
            adapter = _FakeTenClassImageAdapter(name)
            data_service = DataService(
                registry=DatasetRegistry((adapter,)),
                catalog=LocalDatasetCatalog(root / f"{name}-ten-class-catalog.json"),
            )
            data_service.register("ten-classes", adapter.name, {"root": str(self.data_root)})
            data_service.inspect("ten-classes")
            service = QuickStartService(data_service, artifact_root=root / f"{name}-ten-class-artifacts")
            for key in ("clean", "symmetric", "pairflip", "pdl"):
                with self.subTest(dataset=name, noise=key):
                    selection = QuickStartNoiseSelection(
                        "clean" if key == "clean" else "synthetic", key,
                        rate=None if key == "clean" else 0.2, seed=1,
                    )
                    option = next(item for item in service.method_options("ten-classes", selection)
                                  if item.paper_id == "ca2c")
                    self.assertEqual(option.status, "ready", option.reasons)
                    plan = service.build_plan(dataset_alias="ten-classes", noise_selection=selection, paper_id="ca2c")
                    self.assertEqual(plan.status, "ready", plan.details)
                    config = load_yaml(Path(plan.generated_config_path))
                    self.assertEqual(config["data"]["num_classes"], 10)
                    self.assertEqual(config["ca2c"]["candidate_k"], 2)

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
        self.assertEqual(option.status, "ready", option.reasons)
        self.assertNotIn("config:requires_noisy_training_labels", option.required_user_inputs)
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

    def test_gce_dataset_defaults_and_user_overrides(self) -> None:
        paper = next(item for item in load_papers() if item.id == 'gce')
        template = method_template_for_paper(paper).config
        profile = self.data_service.status('local-cifar10').profile.to_dict()
        noise = QuickStartNoiseSelection('synthetic', 'symmetric', rate=0.2, seed=1)
        for adapter, epochs, milestones in (('cifar10', 120, [40, 80]), ('cifar100', 150, [80, 120])):
            with self.subTest(adapter=adapter):
                profile.update(adapter=adapter, num_classes=10 if adapter == 'cifar10' else 100)
                profile['sample_counts_by_split'] = {'train': 50000, 'test': 10000}
                candidate = adapt_method_template(template, dataset_alias='local-cifar10',
                    dataset_profile=profile, noise_selection=noise, data_service=self.data_service)
                self.assertEqual(candidate['trainer']['epochs'], epochs)
                self.assertEqual(candidate['scheduler']['milestones'], milestones)
                self.assertEqual(candidate['scheduler']['gamma'], 0.1)
                self.assertEqual(candidate['scheduler']['name'], 'multistep')
                self.assertEqual(candidate['optimizer']['lr'], 0.01)
                self.assertEqual(candidate['optimizer']['momentum'], 0.9)
                self.assertEqual(candidate['optimizer']['weight_decay'], 1e-4)
                self.assertEqual(candidate['loader']['batch_size'], 128)
                self.assertEqual(candidate['loss']['q'], 0.7)
                self.assertEqual(candidate['model']['name'], 'resnet34')
                overridden = adapt_method_template(template, dataset_alias='local-cifar10',
                    dataset_profile=profile, noise_selection=noise, data_service=self.data_service,
                    method_inputs={'trainer.epochs': 3, 'scheduler.milestones': [1, 2]})
                self.assertEqual(overridden['trainer']['epochs'], 3)
                self.assertEqual(overridden['scheduler']['milestones'], [1, 2])
        self.assertEqual(template['trainer']['epochs'], 120)
        self.assertEqual(template['scheduler']['milestones'], [40, 80])

    def test_volminnet_cifar100_defaults_and_user_overrides(self) -> None:
        from lnl_toolbox.training.experiment import build_optimizer, build_model
        from lnl_toolbox.algorithms.volminnet import VolMinNetConfig, VolMinTransition
        paper = next(item for item in load_papers() if item.id == 'volminnet')
        template = method_template_for_paper(paper).config
        profile = self.data_service.status('local-cifar10').profile.to_dict()
        profile.update(adapter='cifar100', num_classes=100)
        profile['sample_counts_by_split'] = {'train': 50000, 'test': 10000}
        noise = QuickStartNoiseSelection('synthetic', 'symmetric', rate=0.2, seed=1)
        config = adapt_method_template(template, dataset_alias='local-cifar10',
            dataset_profile=profile, noise_selection=noise, data_service=self.data_service)
        method = config['volminnet']
        self.assertEqual(method['model'], {'name': 'resnet32', 'base_width': 16})
        self.assertEqual(method['optimizer']['transition'], {'name': 'adam', 'lr': 0.001, 'weight_decay': 0.0})
        self.assertEqual(method['optimizer']['classifier'], template['volminnet']['optimizer']['classifier'])
        self.assertEqual(config['trainer']['epochs'], 150)
        for role in ('classifier', 'transition'):
            self.assertEqual(method['scheduler'][role]['milestones'], [30, 60])
        parsed = VolMinNetConfig.from_mapping(config)
        optimizer = build_optimizer(VolMinTransition(100), parsed.transition_optimizer)
        self.assertEqual(optimizer.defaults['betas'], (0.9, 0.999))
        model = build_model(method['model'], 100)
        self.assertEqual(model.classifier.out_features, 100)
        overridden = adapt_method_template(template, dataset_alias='local-cifar10',
            dataset_profile=profile, noise_selection=noise, data_service=self.data_service,
            method_inputs={'volminnet.optimizer.transition.lr': 0.02, 'trainer.epochs': 3})
        self.assertEqual(overridden['volminnet']['optimizer']['transition']['lr'], 0.02)
        self.assertEqual(overridden['trainer']['epochs'], 3)
        self.assertEqual(template['volminnet']['model']['name'], 'resnet18')

    def test_t_revision_dataset_defaults_and_user_overrides(self) -> None:
        from lnl_toolbox.algorithms.t_revision import TRevisionConfig
        from lnl_toolbox.training.experiment import build_model
        paper = next(item for item in load_papers() if item.id == 't-revision')
        template = method_template_for_paper(paper).config
        profile = self.data_service.status('local-cifar10').profile.to_dict()
        noise = QuickStartNoiseSelection('synthetic', 'symmetric', rate=0.2, seed=1)
        for adapter, name, epochs, blocks in (
                ('cifar10', 'resnet18', 20, [2, 2, 2, 2]),
                ('cifar100', 'resnet34', 15, [3, 4, 6, 3])):
            with self.subTest(adapter=adapter):
                classes = 10 if adapter == 'cifar10' else 100
                profile.update(adapter=adapter, num_classes=classes)
                profile['sample_counts_by_split'] = {'train': 50000, 'test': 10000}
                config = adapt_method_template(template, dataset_alias='local-cifar10',
                    dataset_profile=profile, noise_selection=noise, data_service=self.data_service)
                method = config['t_revision']
                self.assertEqual(method['stage1']['epochs'], epochs)
                self.assertEqual(method['stage1']['model']['name'], name)
                for stage in ('classifier_initialization', 'revision'):
                    self.assertEqual(method[stage]['epochs'], 200)
                self.assertEqual(config['data']['normalization'], template['data']['normalization'])
                parsed = TRevisionConfig.from_mapping(config)
                model = build_model(parsed.model, classes).eval()
                self.assertEqual(model.stem[0].padding, (0, 0))
                self.assertEqual(model.classifier.out_features, classes)
                self.assertEqual([len(getattr(model, f'layer{i}')) for i in range(1, 5)], blocks)
                overridden = adapt_method_template(template, dataset_alias='local-cifar10',
                    dataset_profile=profile, noise_selection=noise, data_service=self.data_service,
                    method_inputs={'t_revision.stage1.epochs': 3,
                        't_revision.stage1.model.name': 'resnet18',
                        't_revision.stage1.optimizer.lr': 0.02})
                self.assertEqual(overridden['t_revision']['stage1']['epochs'], 3)
                self.assertEqual(overridden['t_revision']['stage1']['model']['name'], 'resnet18')
                self.assertEqual(overridden['t_revision']['stage1']['optimizer']['lr'], 0.02)
        self.assertEqual(template['t_revision']['stage1']['epochs'], 20)
        self.assertEqual(template['t_revision']['stage1']['model']['name'], 'resnet18')

    def test_dual_t_dataset_defaults_and_user_overrides(self) -> None:
        from lnl_toolbox.algorithms.dual_t import DualTConfig
        from lnl_toolbox.training.experiment import build_model
        paper = next(item for item in load_papers() if item.id == 'dual-t')
        template = method_template_for_paper(paper).config
        profile = self.data_service.status('local-cifar10').profile.to_dict()
        noise = QuickStartNoiseSelection('synthetic', 'symmetric', rate=0.2, seed=1)
        for adapter, name, classes in (('cifar10', 'resnet18', 10), ('cifar100', 'resnet34', 100)):
            with self.subTest(adapter=adapter):
                profile.update(adapter=adapter, num_classes=classes)
                profile['sample_counts_by_split'] = {'train': 50000, 'test': 10000}
                config = adapt_method_template(template, dataset_alias='local-cifar10',
                    dataset_profile=profile, noise_selection=noise, data_service=self.data_service)
                parsed = DualTConfig.from_mapping(config)
                self.assertEqual(config['data']['validation_size'], 10000)
                for stage in ('posterior_stage', 'final_stage'):
                    self.assertEqual(config[stage]['model']['name'], name)
                    self.assertEqual(config[stage]['model']['variant'], 'imagenet_standard')
                    self.assertEqual(config[stage]['optimizer']['momentum'], 0)
                    self.assertEqual(config[stage]['optimizer']['weight_decay'], 1e-4)
                model = build_model(parsed.final_stage.model, classes)
                self.assertEqual(model.conv1.kernel_size, (7, 7))
                self.assertEqual(model.fc.out_features, classes)
                self.assertEqual(config['posterior_stage']['epochs'], 100)
                self.assertEqual(config['final_stage']['epochs'], 200)
                self.assertEqual(config['final_stage']['scheduler']['milestones'], [50, 100, 150])
                overridden = adapt_method_template(template, dataset_alias='local-cifar10',
                    dataset_profile=profile, noise_selection=noise, data_service=self.data_service,
                    method_inputs={'final_stage.epochs': 3, 'final_stage.model.name': 'resnet18',
                        'final_stage.optimizer.lr': 0.02})
                self.assertEqual(overridden['final_stage']['epochs'], 3)
                self.assertEqual(overridden['final_stage']['model']['name'], 'resnet18')
                self.assertEqual(overridden['final_stage']['optimizer']['lr'], 0.02)
        self.assertEqual(template['final_stage']['model']['name'], 'resnet18')
        self.assertEqual(template['final_stage']['epochs'], 200)

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
                self.assertEqual(
                    {option.paper_id for option in options},
                    {paper.id for paper in load_papers()},
                )
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
                        if option.paper_id == "mentornet":
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
        matrix = np.asarray(estimator["matrix"])
        self.assertEqual(matrix.shape, (100, 100))
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
        np.testing.assert_allclose(np.diag(matrix), 0.8)
        np.testing.assert_allclose(matrix[0, 1:], 0.2 / 99)
        self.assertEqual(estimator["name"], "known")
        self.assertEqual(method_template_for_paper(paper).config["data"]["name"], "cifar10")

        # A different protocol must replace the matrix even when C is unchanged.
        template = method_template_for_paper(paper)
        pairflip = adapt_method_template(
            template.config, dataset_alias="local-cifar100", dataset_profile=profile,
            noise_selection=QuickStartNoiseSelection("synthetic", "pairflip", rate=0.3, seed=1),
            data_service=service,
        )
        changed_matrix = np.asarray(pairflip["pipeline"]["transition_estimator"]["matrix"])
        np.testing.assert_allclose(np.diag(changed_matrix), 0.7)
        self.assertAlmostEqual(changed_matrix[0, 1], 0.3)
        self.assertEqual(np.count_nonzero(changed_matrix[0]), 2)
        unknown = adapt_method_template(
            template.config, dataset_alias="local-cifar100", dataset_profile=profile,
            noise_selection=QuickStartNoiseSelection("synthetic", "pdl", rate=0.2, seed=1),
            data_service=service,
        )
        self.assertNotIn("matrix", unknown["pipeline"]["transition_estimator"])

    def test_clean_and_symmetric_upm_build_ready_plans(self) -> None:
        clean = QuickStartNoiseSelection("clean", "clean")
        clean_option = next(
            item for item in self.service.method_options("local-cifar10", clean)
            if item.paper_id == "upm"
        )
        self.assertEqual(clean_option.status, "ready", clean_option.reasons)
        self.assertNotIn("config:requires_noisy_training_labels", clean_option.required_user_inputs)
        clean_plan = self.service.build_plan(dataset_alias="local-cifar10", noise_selection=clean, paper_id="upm")
        self.assertEqual(clean_plan.status, "ready", clean_plan.details)

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
