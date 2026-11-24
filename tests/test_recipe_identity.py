"""Release identity contract frozen before the recipe identity migration."""
from __future__ import annotations

from dataclasses import asdict
from contextlib import redirect_stdout
import hashlib
from importlib import metadata
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import yaml

from lnl_toolbox import catalog
from lnl_toolbox.cli.main import _load_source, build_parser, main
from lnl_toolbox.training.results import config_hash
from lnl_toolbox.training.runners import resolve_runner
from lnl_toolbox.training.sweep import _new_manifest, _validate_existing_manifest, plan_sweep


ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).with_name("fixtures") / "recipe_identity_baseline.json"
BASE_SHA = "c9a3234a03db7311c0dd70e6700ab5ae5971303a"
CONFIG_DEST = "share/lnl-toolbox/configs/"
VIEWS = ((False, False), (False, True), (True, False), (True, True))
PAPER_VARIANT_BATCH = {
    "apl-cifar10-noise04-reproduction": "apl",
    "cifar10-cnlcu-hard-smoke": "cnlcu",
    "importance-reweighting-binary-high-dim": "importance-reweighting",
}
DEDICATED_BATCH = (
    "binary-risk-natarajan-1epoch",
    "importance-reweighting-binary-high-dim-smoke",
    "cifar10-t-revision-sym20-short",
)
COMPOSE_BATCH_RECIPE = "cifar10-foundation-smoke"
PAPER_SMOKE_BATCH = {
    "apl-cifar10-noise02-smoke": "apl",
    "cifar10-cnlcu-soft-smoke": "cnlcu",
    "cifar10-coteaching-smoke": "coteaching",
    "cifar10-symmetric-cdr-smoke": "cdr",
}
STANDARD_PAPER_SMOKE_BATCH = {
    "ca2c-cifar10-smoke": "ca2c",
    "cifar10-dividemix-smoke": "dividemix",
    "jocor-cifar10-symmetric05-smoke": "jocor",
    "l2rw-cifar10-smoke": "l2rw",
    "cifar10-t-revision-smoke": "t-revision",
}
FINAL_STANDARD_PAPER_SMOKE_BATCH = {
    "cwd-cifar10-smoke": "cwd",
    "fine-cifar100n-smoke": "fine",
    "cifar10-lend-smoke": "lend",
}
INTERNAL_LIFECYCLE_SMOKE_BATCH = {
    "pcse-multiclass-smoke": "pcse",
    "mc-ldce-cifar10-smoke": "mc-ldce",
    "pdl-cifar10-smoke": "pdl",
    "cifar10-dual-t-smoke": "dual-t",
}
HISTORICAL_SUMMARIES = {
    "recipe": "ac7b430c213e331fab7ef8c044d9ff384473524e4fb4f1ae35d918790133eb49",
    "paper": "ef2af29c25c75641f5ec4222de69e774ac5d5092d45c5b50cfcc16087896d09c",
    "source_discovery": "02cfa7cc14878f7c10ad6080be9ae6d5ea1411e8b512b0bc9f1bf1b202711499",
    "packaging_discovery": "6b63060f14b8b281bc6ba6d3f244c8efd08d39b2ff53508cee833cfb9c7c3ea5",
}


def _digest(value):
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _view_name(conditional, public):
    return f"{conditional}/{public}"


def _package_map(root):
    with (root / "pyproject.toml").open("rb") as stream:
        files = tomllib.load(stream)["tool"]["setuptools"]["data-files"]
    pairs = [
        (path, f"{destination}/{PurePosixPath(path).name}")
        for destination, paths in files.items()
        if destination.startswith(CONFIG_DEST)
        for path in paths
    ]
    _unique(pairs, "packaged source path", key=lambda pair: pair[0])
    _unique(pairs, "package resource", key=lambda pair: pair[1])
    return dict(pairs)


def _unique(values, label, key=lambda item: item):
    seen = set()
    for value in values:
        name = key(value)
        if name in seen:
            raise AssertionError(f"duplicate {label}: {name}")
        seen.add(name)
    return seen


def verify_candidate_manifest(raw, known_paths, paper_recipe_ids=()):
    """Test-side gate, not a production manifest parser."""
    entries = raw["recipes"]
    paths = [entry if isinstance(entry, str) else entry["path"] for entry in entries]
    registered = _unique(paths, "manifest path")
    for kind, refs, targets in (
        ("normal", paths, set(known_paths)),
        ("public", [item["path"] for item in raw.get("public", ())], registered),
        ("conditional", raw.get("conditional", ()), registered),
    ):
        _unique(refs, f"{kind} reference")
        if missing := set(refs) - targets:
            raise AssertionError(f"dangling {kind} reference: {sorted(missing)}")

    # Model both the legacy fallback and its second collision hazard; forbid
    # raw stem ambiguity even when the final fallback happens to be unique.
    stems = set()
    allocated = set()
    ambiguous = []
    for path in (entry for entry in entries if isinstance(entry, str)):
        p = PurePosixPath(path)
        stem = catalog._recipe_id(p)
        candidate = stem
        if stem in allocated:
            candidate = f"{p.parent.name}-{stem}"
        if candidate in allocated:
            raise AssertionError(f"duplicate identity after fallback: {candidate}")
        allocated.add(candidate)
        if stem in stems:
            ambiguous.append(stem)
        stems.add(stem)
    if ambiguous:
        raise AssertionError(f"ambiguous canonical stem: {ambiguous}")
    for entry in (entry for entry in entries if not isinstance(entry, str)):
        if entry["id"] in allocated:
            raise AssertionError(f"duplicate canonical identity: {entry['id']}")
        allocated.add(entry["id"])
    if missing := set(paper_recipe_ids) - allocated:
        raise AssertionError(f"dangling Paper recipe reference: {sorted(missing)}")


def verify_view_identities(views):
    """Each row is (canonical ID, source-relative or install-relative resource)."""
    by_id = {}
    for view, rows in views.items():
        _unique(rows, f"ID in view {view}", key=lambda row: row[0])
        for recipe_id, resource in rows:
            previous = by_id.setdefault(recipe_id, resource)
            if previous != resource:
                raise AssertionError(
                    f"{view}: recipe {recipe_id} moved from {previous} to {resource}"
                )


def _sweep_snapshot(recipes, configs):
    recipe_id = "gce-cifar10-noise02-smoke"
    config = configs[recipe_id]
    result = {}
    for name, matrix in (("seed_only", None), ("matrix", {"trainer.epochs": [1, 2]})):
        plan = plan_sweep(config, [8, 3], matrix=matrix, recipe=recipes[recipe_id].id)
        manifest = _new_manifest(plan)
        del manifest["created_at"]
        relative_root = plan.root.relative_to(Path.cwd().resolve()).as_posix()
        for row in manifest["runs"]:
            row["run_dir"] = Path(row["run_dir"]).relative_to(Path.cwd().resolve()).as_posix()
        # Normalize tuples to JSON lists without erasing ordering.
        result[name] = json.loads(json.dumps({
            "default_root": relative_root,
            "planned_runs": [asdict(run) for run in plan.runs],
            "manifest": manifest,
        }))
    return result


def make_snapshot(root, package_map, resolve_resource, web_bindings):
    all_recipes = catalog.discover_recipes(root, include_conditional=True)
    by_id = {recipe.id: recipe for recipe in all_recipes}
    _unique(all_recipes, "canonical ID", key=lambda recipe: recipe.id)
    configs = {item.id: catalog.load_recipe_config(item) for item in all_recipes}
    papers = catalog.load_papers(root)
    associations = {item.id: [] for item in all_recipes}
    paper_rows = []
    for paper in papers:
        default = catalog.default_paper_config(paper, root=root)[1].id
        links = [asdict(item) for item in paper.configs]
        paper_rows.append({
            "id": paper.id, "default": default, "associations": links,
            "cli_recommended": next(
                (item.recipe_id for item in paper.configs if item.profile == "smoke"),
                paper.configs[0].recipe_id,
            ),
        })
        for order, item in enumerate(paper.configs):
            associations[item.recipe_id].append({
                "paper": paper.id, "order": order, "formal_default": item.recipe_id == default,
            })

    rows = []
    for item in all_recipes:
        path = resolve_resource(item)
        rows.append({
            "id": item.id, "path": path, "package_resource": package_map[path],
            "profile": item.profile, "runner": item.runner, "method": item.method,
            "dataset": item.dataset, "noise": item.noise, "epochs": item.epochs,
            "visibility": item.visibility, "availability": item.availability,
            "label": item.label, "description": item.description,
            "configuration_fidelity": item.configuration_fidelity,
            "implementation_status": item.implementation_status,
            "normalized_config_hash": config_hash(configs[item.id]),
            "paper_associations": associations[item.id],
            "web_formal_methods": [method for method, key in web_bindings.items() if key == item.id],
        })
    source_discovery = {}
    for conditional, public in VIEWS:
        view = catalog.discover_recipes(root, include_conditional=conditional, public_only=public)
        source_discovery[_view_name(conditional, public)] = [
            (x.id, resolve_resource(x), x.profile, x.runner, x.visibility, x.availability)
            for x in view
        ]
    verify_view_identities({
        name: [(row[0], row[1]) for row in entries] for name, entries in source_discovery.items()
    })
    return {
        "baseline_commit": BASE_SHA,
        "recipes": rows, "papers": paper_rows, "web_formal_bindings": web_bindings,
        "views": {name: [row[0] for row in entries] for name, entries in source_discovery.items()},
        "packaged_only": {path: package_map[path] for path in sorted(
            set(package_map) - {row["path"] for row in rows}
        )},
        "sweep": _sweep_snapshot(by_id, configs),
        "summaries": {
            "recipe": _digest([(x.id, x.profile, x.runner, configs[x.id]) for x in all_recipes]),
            "paper": _digest([(p["id"], p["associations"], p["default"]) for p in paper_rows]),
            "source_discovery": _digest(source_discovery),
            "packaging_discovery": HISTORICAL_SUMMARIES["packaging_discovery"],
        },
    }, source_discovery


def source_snapshot(root=ROOT):
    bindings_file = root / "web" / "lnl_parameter_metadata_registry_revised.yaml"
    bindings = yaml.safe_load(bindings_file.read_text(encoding="utf-8"))["formal_recipe_bindings"]
    return make_snapshot(
        root, _package_map(root), lambda item: item.config_path.relative_to(root).as_posix(), bindings
    )


class RecipeIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads(BASELINE.read_text(encoding="utf-8"))
        cls.actual, _ = source_snapshot()

    def test_frozen_identity_and_paper_associations(self):
        self.assertEqual(self.expected["baseline_commit"], BASE_SHA)
        self.assertEqual(len(self.actual["recipes"]), 68)
        self.assertEqual(len(self.actual["papers"]), 26)
        self.assertEqual(sum(len(p["associations"]) for p in self.actual["papers"]), 56)
        self.assertEqual(sum(bool(p["default"]) for p in self.actual["papers"]), 26)
        self.assertEqual(len(self.actual["packaged_only"]), 3)
        for section in ("recipes", "papers"):
            self.assertEqual(
                [row["id"] for row in self.actual[section]],
                [row["id"] for row in self.expected[section]], section,
            )
            for actual, expected in zip(self.actual[section], self.expected[section]):
                with self.subTest(section=section, key=expected["id"]):
                    self.assertEqual(actual, expected)
        self.assertEqual(self.actual["web_formal_bindings"], self.expected["web_formal_bindings"])
        self.assertEqual(self.actual["packaged_only"], self.expected["packaged_only"])
        self.assertEqual(self.actual["summaries"], HISTORICAL_SUMMARIES)
        self.assertEqual(self.expected["summaries"], HISTORICAL_SUMMARIES)

    def test_discovery_order_and_filter_identity(self):
        self.assertEqual([len(v) for v in self.actual["views"].values()], [63, 4, 68, 4])
        for view, expected in self.expected["views"].items():
            with self.subTest(view=view):
                self.assertEqual(self.actual["views"][view], expected)

    def test_cli_normalizes_old_lookup_and_keeps_custom_yaml_unregistered(self):
        parser = build_parser()
        canonical = "gce-cifar10-noise02-smoke"
        expected = next(r for r in self.expected["recipes"] if r["id"] == canonical)
        for value in (canonical, " GCE_CIFAR10_NOISE02_SMOKE "):
            with self.subTest(value=value):
                _, path, recipe, _ = _load_source(parser.parse_args(["run", "--recipe", value]))
                self.assertEqual(recipe.id, canonical)
                self.assertEqual(path.relative_to(ROOT).as_posix(), expected["path"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "unregistered.yaml"
            path.write_bytes((ROOT / expected["path"]).read_bytes())
            config, actual, recipe, project = _load_source(
                parser.parse_args(["run", "--config", str(path)])
            )
            self.assertEqual(actual, path)
            self.assertIsNone(recipe)
            normalized = catalog.load_yaml(path)
            self.assertEqual(config_hash(normalized), expected["normalized_config_hash"])
            self.assertEqual(config, catalog.resolve_config_paths(normalized, project))

    def test_sweep_legacy_manifest_and_default_root_match_baseline(self):
        canonical = "gce-cifar10-noise02-smoke"
        config = catalog.load_recipe_config(catalog.recipe_by_id(canonical))
        for name, matrix in (("seed_only", None), ("matrix", {"trainer.epochs": [1, 2]})):
            with self.subTest(plan=name):
                expected = self.expected["sweep"][name]
                self.assertEqual(self.actual["sweep"][name], expected)
                current = plan_sweep(config, [8, 3], matrix=matrix, recipe=canonical)
                _validate_existing_manifest(expected["manifest"], current)

    def test_current_manifest_passes_test_only_gate(self):
        raw = json.loads(
            (ROOT / "src/lnl_toolbox/cli/data/recipe_catalog.json").read_text(encoding="utf-8")
        )
        verify_candidate_manifest(
            raw,
            [path for path in _package_map(ROOT) if (ROOT / path).is_file()],
            [a["recipe_id"] for paper in self.actual["papers"] for a in paper["associations"]],
        )

    def test_pilot_recipe_changes_representation_not_behavior(self):
        recipe_id = "cifar10-symmetric-small-loss-linear-smoke"
        expected = next(row for row in self.expected["recipes"] if row["id"] == recipe_id)
        manifest_path = ROOT / "src/lnl_toolbox/cli/data/recipe_catalog.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        pilot = {"id": recipe_id, "path": expected["path"], "profile": expected["profile"]}
        self.assertIn(pilot, raw["recipes"])
        actual = catalog.recipe_by_id(" CIFAR10_SYMMETRIC_SMALL_LOSS_LINEAR_SMOKE ", ROOT)
        original = [entry["path"] if entry == pilot else entry for entry in raw["recipes"]]
        with patch.object(catalog.resources, "files") as files:
            files.return_value.joinpath.return_value.read_text.return_value = json.dumps({
                **raw, "recipes": original,
            })
            legacy = catalog.recipe_by_id(recipe_id, ROOT)
        self.assertEqual(actual, legacy)
        self.assertEqual(config_hash(catalog.load_recipe_config(actual)), expected["normalized_config_hash"])

    def test_first_low_risk_batch_preserves_cli_path_and_sweep_identity(self):
        batch = ("cifar10-clean", "cifar10-symmetric-small-loss-smoke")
        pilot = "cifar10-symmetric-small-loss-linear-smoke"
        expected = {row["id"]: row for row in self.expected["recipes"]}
        raw = json.loads(
            (ROOT / "src/lnl_toolbox/cli/data/recipe_catalog.json").read_text(encoding="utf-8")
        )
        entries = raw["recipes"]
        explicit_ids = {
            *batch, pilot, *PAPER_VARIANT_BATCH, *DEDICATED_BATCH,
            COMPOSE_BATCH_RECIPE, *PAPER_SMOKE_BATCH, *STANDARD_PAPER_SMOKE_BATCH,
            *FINAL_STANDARD_PAPER_SMOKE_BATCH, *INTERNAL_LIFECYCLE_SMOKE_BATCH,
        }
        self.assertEqual(
            [item for item in entries if isinstance(item, dict)],
            [{key: row[key] for key in ("id", "path", "profile")}
             for row in self.expected["recipes"] if row["id"] in explicit_ids],
        )
        self.assertEqual(sum(isinstance(item, str) for item in entries), 42)
        self.assertEqual(
            [item if isinstance(item, str) else item["path"] for item in entries],
            [row["path"] for row in self.expected["recipes"]],
        )
        old_entries = [
            item["path"] if isinstance(item, dict) and item["id"] in batch else item
            for item in entries
        ]
        parser = build_parser()
        direct_path = str(ROOT / expected["cifar10-clean"]["path"])

        def cli_load(*args):
            return _load_source(parser.parse_args(["run", *args]))

        explicit = {
            key: cli_load("--recipe", f" {key.upper().replace('-', '_')} ") for key in batch
        }
        direct = cli_load("--config", direct_path)
        self.assertEqual(direct, cli_load(direct_path))
        with patch.object(catalog.resources, "files") as files:
            files.return_value.joinpath.return_value.read_text.return_value = json.dumps({
                **raw, "recipes": old_entries,
            })
            legacy = {key: cli_load("--recipe", key) for key in batch}
            self.assertEqual(direct, cli_load("--config", direct_path))
            self.assertEqual(direct, cli_load(direct_path))
        self.assertIsNone(direct[2])
        self.assertEqual(direct[:2], explicit["cifar10-clean"][:2])
        self.assertEqual(
            explicit[batch[1]][0]["selector"], {"name": "small_loss", "keep_rate": 0.5}
        )

        for key in batch:
            with self.subTest(recipe=key):
                config, path, recipe, _ = explicit[key]
                self.assertEqual(explicit[key], legacy[key])
                self.assertEqual(path.relative_to(ROOT).as_posix(), expected[key]["path"])
                self.assertEqual(
                    (recipe.id, recipe.profile, recipe.runner, recipe.method),
                    (key, *(expected[key][field] for field in ("profile", "runner", "method"))),
                )
                self.assertEqual(
                    config_hash(catalog.load_recipe_config(recipe)),
                    expected[key]["normalized_config_hash"],
                )
                old_config, _, old_recipe, _ = legacy[key]
                for matrix in (None, {"trainer.epochs": [1, 2]}):
                    with self.subTest(matrix=matrix):
                        current = plan_sweep(config, [8, 3], matrix=matrix, recipe=recipe.id)
                        previous = plan_sweep(old_config, [8, 3], matrix=matrix, recipe=old_recipe.id)
                        self.assertEqual(current, previous)
                        _validate_existing_manifest(_new_manifest(previous), current)

    def test_paper_variant_batch_preserves_selection_path_and_sweep(self):
        raw = json.loads(
            (ROOT / "src/lnl_toolbox/cli/data/recipe_catalog.json").read_text(encoding="utf-8")
        )
        legacy_manifest = json.dumps({
            **raw, "recipes": [
                item["path"] if isinstance(item, dict) and item["id"] in PAPER_VARIANT_BATCH else item
                for item in raw["recipes"]
            ],
        })
        expected_papers = {paper["id"]: paper for paper in self.expected["papers"]}
        parser = build_parser()

        def observe():
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(main(["papers", "list", "--format", "json"]), 0)
            self.assertEqual(
                [(item["id"], item["recommended"]) for item in json.loads(out.getvalue())],
                [(paper["id"], paper["cli_recommended"]) for paper in self.expected["papers"]],
            )
            papers = {paper.id: paper for paper in catalog.load_papers(ROOT)}
            selected = {}
            for key, paper_id in PAPER_VARIANT_BATCH.items():
                with self.subTest(recipe=key):
                    paper, expected = papers[paper_id], expected_papers[paper_id]
                    self.assertEqual([asdict(item) for item in paper.configs], expected["associations"])
                    association = next(item for item in paper.configs if item.recipe_id == key)
                    paper_config, recipe = catalog.select_paper_config(
                        paper, profile=association.profile, variant=association.variant, root=ROOT
                    )
                    self.assertEqual(paper_config, association)
                    self.assertEqual(recipe.id, key)
                    default = catalog.default_paper_config(paper, root=ROOT)[1].id
                    self.assertEqual(default, expected["default"])
                    self.assertNotEqual(default, key)
                    config, path, bound, _ = _load_source(parser.parse_args(["run", "--recipe", key]))
                    direct, direct_path, unbound, _ = _load_source(
                        parser.parse_args(["run", "--config", str(path)])
                    )
                    self.assertEqual(bound, recipe)
                    self.assertEqual((direct, direct_path, unbound), (config, path, None))
                    selected[key] = (recipe, [
                        plan_sweep(config, [8, 3], recipe=key, matrix=matrix)
                        for matrix in (None, {"trainer.epochs": [1, 2]})
                    ])
            return selected

        explicit = observe()
        real_files = catalog.resources.files
        with patch.object(catalog.resources, "files") as files:
            fake = files.return_value
            fake.joinpath.return_value.read_text.return_value = legacy_manifest
            files.side_effect = lambda package: fake if package == "lnl_toolbox.cli" else real_files(package)
            legacy = observe()
        self.assertEqual(explicit, legacy)
        for key, (_, plans) in explicit.items():
            for old, new in zip(legacy[key][1], plans):
                _validate_existing_manifest(_new_manifest(old), new)

    def test_paper_smoke_batch_preserves_order_and_cli_recommendations(self):
        self._assert_paper_smoke_batch(PAPER_SMOKE_BATCH)

    def test_standard_paper_smoke_batch_preserves_order_and_cli_recommendations(self):
        self._assert_paper_smoke_batch(STANDARD_PAPER_SMOKE_BATCH)

    def test_final_standard_smoke_batch_preserves_paper_and_runtime_contracts(self):
        self._assert_paper_smoke_batch(FINAL_STANDARD_PAPER_SMOKE_BATCH)

    def test_internal_lifecycle_smoke_batch_preserves_paper_and_runtime_contracts(self):
        self._assert_paper_smoke_batch(INTERNAL_LIFECYCLE_SMOKE_BATCH)

    def _observe_internal_lifecycle(self, recipe, config, command):
        from lnl_toolbox.training.service import ExperimentService

        runner = ExperimentService().preflight(config, check_data=False)
        self.assertEqual(runner.name, recipe.runner)
        requirements = runner.requirements(config)
        self.assertFalse(requirements.prerequisites)
        artifacts = ()
        if recipe.runner == "pcse":
            self.assertEqual(config["data"]["num_classes"], 3)
            self.assertEqual(config["pretraining_stage"]["mode"], "train")
            self.assertNotIn("source", config["pretraining_stage"])
        elif recipe.runner == "mc_ldce":
            self.assertEqual(config["transition"]["estimator"], "known_smoke")
            self.assertIn("matrix", config["transition"])
            self.assertNotIn("artifact", config["transition"])
            # The bundled tiny synthetic smoke needs no external dataset.
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                run_dir = runner.invoke(config, output_dir=Path(directory) / "run")
                artifacts = tuple(sorted(path.name for path in run_dir.glob("*.npz")))
            self.assertTrue({"feature_snapshot.npz", "transition_artifact.npz",
                             "statistic_artifact.npz"}.issubset(artifacts))
        elif recipe.runner == "instance_transition":
            self.assertEqual(config["instance_transition"]["name"], "pdl")
            self.assertNotIn("artifact", config["instance_transition"])
            self.assertEqual(config["warmup"]["epochs"], 1)
        elif recipe.runner == "dual_t":
            self.assertNotIn("evidence", config)
            self.assertTrue(config["final_stage"]["fresh_model"])
            self.assertEqual(config["transition_stage"], {})

        sweep = plan_sweep(config, [8, 3], recipe=recipe.id)
        _validate_existing_manifest(_new_manifest(sweep), sweep)
        return (
            runner, requirements, runner.describe(config), sweep, artifacts,
            command("validate", "--recipe", recipe.id),
            command("run", "--recipe", recipe.id, "--dry-run", "--no-check-data"),
        )

    def _assert_final_smoke_contract(self, recipe, parsed):
        requirements = resolve_runner(parsed).requirements(parsed)
        self.assertFalse(requirements.prerequisites)
        if recipe.id == "cwd-cifar10-smoke":
            self.assertEqual(parsed["data"]["name"], "cifar10_airplane_automobile")
            self.assertEqual(requirements.implemented_variant, "binary_protocol")
            self.assertEqual(requirements.exact_classes, frozenset({2}))
            self.assertNotIn("transition_artifact", parsed["cwd"])
        elif recipe.id == "fine-cifar100n-smoke":
            from web.external_resources import resources

            self.assertEqual(parsed["data"]["name"], "cifar100")
            self.assertEqual(parsed["noise"]["name"], "symmetric")
            self.assertNotIn("path", parsed["noise"])
            self.assertEqual(resources(parsed, ROOT), [])
        elif recipe.id == "cifar10-lend-smoke":
            from lnl_toolbox.algorithms.lend.config import LENDConfig

            self.assertEqual(recipe.profile, "smoke")
            self.assertEqual(LENDConfig.from_mapping(parsed).implementation_variant, "paper_equation")

    def _assert_paper_smoke_batch(self, batch):
        raw = json.loads(
            (ROOT / "src/lnl_toolbox/cli/data/recipe_catalog.json").read_text(encoding="utf-8")
        )
        legacy_manifest = json.dumps({
            **raw, "recipes": [
                item["path"] if isinstance(item, dict) and item["id"] in batch else item
                for item in raw["recipes"]
            ],
        })
        expected_papers = {row["id"]: row for row in self.expected["papers"]}
        expected_recipes = {row["id"]: row for row in self.expected["recipes"]}
        parser = build_parser()

        def command(*args):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(main(list(args)), 0, out.getvalue())
            return out.getvalue()

        def cli(*args):
            return command("papers", *args)

        def observe():
            listed = cli("list", "--format", "json")
            recommendations = {row["id"]: row["recommended"] for row in json.loads(listed)}
            self.assertEqual(recommendations,
                             {key: row["cli_recommended"] for key, row in expected_papers.items()})
            papers = {paper.id: paper for paper in catalog.load_papers(ROOT)}
            selections = {}
            for recommended, paper_id in batch.items():
                with self.subTest(paper=paper_id):
                    paper, expected = papers[paper_id], expected_papers[paper_id]
                    ordered = [item.recipe_id for item in paper.configs]
                    self.assertEqual(ordered, [item["recipe_id"] for item in expected["associations"]])
                    self.assertEqual([asdict(item) for item in paper.configs], expected["associations"])
                    self.assertEqual(catalog.default_paper_config(paper, root=ROOT)[1].id, expected["default"])
                    self.assertNotEqual(expected["default"], recommended)
                    self.assertEqual(recommendations[paper_id], recommended)
                    self.assertEqual(ordered[0], recommended)
                    shown = cli("show", paper_id)
                    self.assertIn(f"lnl run --recipe {recommended} --dry-run", shown)
                    configs = []
                    lifecycle = None
                    for association in paper.configs:
                        args = ("config", paper_id, "--profile", association.profile,
                                "--variant", association.variant)
                        selection, recipe = catalog.select_paper_config(
                            paper, profile=association.profile, variant=association.variant, root=ROOT
                        )
                        self.assertEqual(selection, association)
                        self.assertEqual(recipe.id, association.recipe_id)
                        row = expected_recipes[recipe.id]
                        self.assertEqual((recipe.profile, recipe.runner, recipe.method),
                                         (row["profile"], row["runner"], row["method"]))
                        self.assertEqual(recipe.config_path, ROOT / row["path"])
                        self.assertEqual(Path(cli(*args, "--path-only").strip()), recipe.config_path)
                        config = cli(*args)
                        parsed = yaml.safe_load(config)
                        self.assertEqual(config_hash(parsed), row["normalized_config_hash"])
                        if recipe.id == recommended and recommended in STANDARD_PAPER_SMOKE_BATCH:
                            runner = resolve_runner(parsed)
                            self.assertEqual(runner.name, row["runner"])
                            self.assertFalse(getattr(runner.requirements(parsed), "prerequisites", ()))
                        if recipe.id == recommended and recommended in FINAL_STANDARD_PAPER_SMOKE_BATCH:
                            self._assert_final_smoke_contract(recipe, parsed)
                        if recipe.id == recommended and recommended in INTERNAL_LIFECYCLE_SMOKE_BATCH:
                            lifecycle = self._observe_internal_lifecycle(recipe, parsed, command)
                        if recipe.id == "l2rw-cifar10-smoke":
                            from_path = _load_source(parser.parse_args(
                                ["run", "--config", str(recipe.config_path)]
                            ))
                            from_id = _load_source(parser.parse_args(["run", "--recipe", recipe.id]))
                            self.assertEqual(from_path[:3], (*from_id[:2], None))
                        configs.append((recipe, config))
                    if paper_id == "t-revision":
                        short = catalog.recipe_by_id("cifar10-t-revision-sym20-short", ROOT)
                        row = expected_recipes[short.id]
                        self.assertNotIn(short.id, ordered)
                        self.assertEqual((short.runner, short.profile, short.config_path),
                                         ("t_revision", row["profile"], ROOT / row["path"]))
                        self.assertEqual(config_hash(catalog.load_recipe_config(short)),
                                         row["normalized_config_hash"])
                        configs.append((short, None))
                    if paper_id == "dual-t":
                        evidence = catalog.recipe_by_id("cifar10-dual-t-evidence-smoke", ROOT)
                        self.assertNotIn(evidence.id, ordered)
                        self.assertIn("evidence", catalog.load_recipe_config(evidence))
                        configs.append((evidence, config_hash(catalog.load_recipe_config(evidence))))
                    selections[paper_id] = (ordered, shown, configs, lifecycle)
            return listed, selections

        explicit = observe()
        real_files = catalog.resources.files
        with patch.object(catalog.resources, "files") as files:
            fake = files.return_value
            fake.joinpath.return_value.read_text.return_value = legacy_manifest
            files.side_effect = lambda package: fake if package == "lnl_toolbox.cli" else real_files(package)
            self.assertEqual(explicit, observe())

    def test_dedicated_and_compose_batch_preserves_cli_and_execution_projection(self):
        batch = (*DEDICATED_BATCH, COMPOSE_BATCH_RECIPE)
        raw = json.loads(
            (ROOT / "src/lnl_toolbox/cli/data/recipe_catalog.json").read_text(encoding="utf-8")
        )
        legacy_manifest = json.dumps({
            **raw, "recipes": [
                item["path"] if isinstance(item, dict) and item["id"] in batch else item
                for item in raw["recipes"]
            ],
        })
        expected = {row["id"]: row for row in self.expected["recipes"]}
        parser = build_parser()

        def cli(*args):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(main(list(args)), 0, out.getvalue())
            return out.getvalue()

        def observe():
            listed = cli("compose", "list", "--runner", "supervised")
            self.assertIn(f"  - {COMPOSE_BATCH_RECIPE}\n", listed)
            with tempfile.TemporaryDirectory() as folder:
                destination = Path(folder) / "composed.yaml"
                # Capture the YAML selected by the real Compose create handler.
                with patch("lnl_toolbox.cli.main.load_yaml", wraps=catalog.load_yaml) as load:
                    composed = cli("compose", "create", "--base", COMPOSE_BATCH_RECIPE,
                                   "--output", str(destination))
                load.assert_called_once_with(ROOT / expected[COMPOSE_BATCH_RECIPE]["path"])
                generated = catalog.load_yaml(destination)
            self.assertIn(str(destination), composed)
            self.assertIn(COMPOSE_BATCH_RECIPE, composed)

            observed = {}
            for key in batch:
                with self.subTest(recipe=key):
                    row = expected[key]
                    config, path, recipe, _ = _load_source(parser.parse_args(["run", "--recipe", key]))
                    self.assertEqual((recipe.id, recipe.profile, recipe.runner, recipe.method),
                                     (key, row["profile"], row["runner"], row["method"]))
                    self.assertEqual(path, ROOT / row["path"])
                    self.assertEqual(config_hash(catalog.load_recipe_config(recipe)),
                                     row["normalized_config_hash"])
                    direct = _load_source(parser.parse_args(["run", "--config", str(path)]))
                    self.assertEqual(direct[:3], (config, path, None))
                    self.assertEqual(row["paper_associations"], [])
                    self.assertEqual(row["web_formal_methods"], [])
                    budget_path = ("t_revision.revision.epochs" if recipe.runner == "t_revision"
                                   else "trainer.epochs")
                    observed[key] = (
                        recipe, config, cli("validate", "--recipe", key),
                        cli("run", "--recipe", key, "--dry-run", "--no-check-data"),
                        [plan_sweep(config, [8, 3], recipe=key, matrix=m)
                         for m in (None, {budget_path: [1, 2]})],
                    )
            return listed, composed.replace(str(destination), "<destination>"), generated, observed

        explicit = observe()
        real_files = catalog.resources.files
        with patch.object(catalog.resources, "files") as files:
            fake = files.return_value
            fake.joinpath.return_value.read_text.return_value = legacy_manifest
            files.side_effect = lambda package: fake if package == "lnl_toolbox.cli" else real_files(package)
            legacy = observe()
        self.assertEqual(explicit, legacy)

    def test_installed_catalog_reads_distribution_once_per_build_and_revalidates(self):
        package_map = _package_map(ROOT)
        paths = {
            PurePosixPath("prefix", resource): (ROOT / source).resolve()
            for source, resource in package_map.items()
        }
        entries = list(paths)
        expected = self.expected["recipes"]
        conditional = next(item for item in expected if item["id"] == "cifar10-pcse-reproduction")
        removed = PurePosixPath("prefix", conditional["package_resource"])
        file_reads = []

        class Distribution:
            @property
            def files(self):
                file_reads.append(None)
                return tuple(entries)

            def locate_file(self, entry):
                return paths[entry]

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(catalog.metadata, "distribution", return_value=Distribution()) as lookup, \
             patch.object(catalog, "load_yaml", wraps=catalog.load_yaml) as load:
            project = Path(directory)
            installed = catalog.discover_recipes(project, include_conditional=True)
            lookup.assert_called_once_with("lnl-toolbox")
            self.assertEqual(len(file_reads), 1)
            self.assertEqual(
                [(item.id, item.config_path.relative_to(ROOT).as_posix()) for item in installed],
                [(item["id"], item["path"]) for item in expected],
            )
            self.assertEqual(len(package_map), 71)
            self.assertEqual(
                [call.args[0] for call in load.call_args_list],
                [item.config_path for item in installed],
            )
            self.assertEqual(
                installed, catalog.discover_recipes(ROOT, include_conditional=True)
            )
            lookup.assert_called_once_with("lnl-toolbox")
            self.assertEqual(len(file_reads), 1)

            # Changing the installation record must affect the next build,
            # even when the missing conditional is excluded from the view.
            entries.remove(removed)
            with self.assertRaisesRegex(FileNotFoundError, "missing YAML.*cifar10-pcse-reproduction"):
                catalog.discover_recipes(project, public_only=True)
            self.assertEqual(lookup.call_count, 2)
            self.assertEqual(len(file_reads), 2)


class CandidateIdentityGateTest(unittest.TestCase):
    def test_candidate_failures_cannot_be_silently_overwritten(self):
        a, b, c = "configs/a/shared.yaml", "configs/b/shared.yaml", "configs/b/b-shared.yaml"
        # The third path repeats the fallback ID used for the second path.
        for raw, real, paper, error in (
            ({"recipes": [a, a]}, [a], [], "duplicate manifest path"),
            ({"recipes": [a, b]}, [a, b], [], "ambiguous canonical stem"),
            ({"recipes": [c, a, b]}, [a, b, c], [], "duplicate identity after fallback"),
            ({"recipes": [a, "configs/b/SHARED.yaml"]}, [a, "configs/b/SHARED.yaml"], [],
             "ambiguous canonical stem"),
            ({"recipes": [a]}, [], [], "dangling normal"),
            ({"recipes": [a], "public": [{"path": b}]}, [a], [], "dangling public"),
            ({"recipes": [a], "conditional": [b]}, [a], [], "dangling conditional"),
            ({"recipes": [a]}, [a], ["removed"], "dangling Paper"),
        ):
            with self.subTest(error=error, candidate=raw):
                with self.assertRaisesRegex(AssertionError, error):
                    verify_candidate_manifest(raw, real, paper)

    def test_discovery_gate_rejects_cross_view_resource_drift(self):
        with self.assertRaisesRegex(AssertionError, "shared moved from"):
            verify_view_identities({"first": [("shared", "a")], "second": [("shared", "b")]})


class ProductionRecipeIdentityTest(unittest.TestCase):
    """Exercise the real resolver; only the manifest and temporary resources are substituted."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        files = patch.object(catalog.resources, "files")
        self.resource_files = files.start()
        self.addCleanup(files.stop)

    def write(self, relative, source="kind: experiment\ndata: {name: cifar10}\n"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return path

    def manifest(self, paths, *, conditional=(), public=()):
        self.resource_files.return_value.joinpath.return_value.read_text.return_value = json.dumps({
            "recipes": paths, "conditional": conditional,
            "public": [{"path": p, "label": p, "description": ""} for p in public],
        })

    def assert_catalog_rejected(self, error, *details, exception=ValueError):
        for conditional, public in VIEWS:
            with self.subTest(conditional=conditional, public=public):
                with self.assertRaisesRegex(exception, error) as raised:
                    catalog.discover_recipes(
                        self.root, include_conditional=conditional, public_only=public
                    )
                for detail in details:
                    self.assertIn(detail, str(raised.exception))

    def test_explicit_identity_overrides_filename_and_preserves_path_projections(self):
        path = "configs/experiment/totally_unrelated_smoke.yaml"
        legacy = "configs/experiment/legacy_smoke.yaml"
        source = self.write(path)
        self.write(legacy)
        self.manifest(
            [legacy, {"id": "stable-recipe-id", "path": path, "profile": "reproduction"}],
            conditional=[path], public=[path],
        )
        recipe = catalog.recipe_by_id(" STABLE_RECIPE_ID ", self.root)
        self.assertEqual((recipe.id, recipe.profile, recipe.config_path),
                         ("stable-recipe-id", "reproduction", source))
        with self.assertRaisesRegex(ValueError, "unknown recipe"):
            catalog.recipe_by_id("totally-unrelated-smoke", self.root)
        expected_ids = (
            ["legacy-smoke"], [], ["legacy-smoke", "stable-recipe-id"], ["stable-recipe-id"],
        )
        for (conditional, public), expected in zip(VIEWS, expected_ids):
            with self.subTest(conditional=conditional, public=public):
                rows = catalog.discover_recipes(self.root, include_conditional=conditional, public_only=public)
                self.assertEqual([row.id for row in rows], expected)
                for row in rows:
                    if row.id == recipe.id:
                        self.assertEqual(row, recipe)
                    else:
                        self.assertEqual(row.profile, "smoke")
        self.assertEqual((recipe.availability, recipe.visibility), ("conditional", "public"))

    def test_invalid_explicit_metadata_cannot_fall_back_to_path_inference(self):
        path = "configs/reproduction/fallback.yaml"
        self.write(path)
        valid = {"id": "stable-id", "path": path, "profile": "smoke"}
        invalid = [
            ({key: value for key, value in valid.items() if key != missing}, f"missing=.*{missing}")
            for missing in valid
        ]
        invalid += [
            ({**valid, "runner": "supervised"}, "unknown=.*runner"),
            ({**valid, "id": "STABLE_ID"}, "invalid explicit recipe id"),
            ({**valid, "id": 1}, "invalid explicit recipe id"),
            ({**valid, "path": ""}, "invalid explicit path"),
            ({**valid, "path": 1}, "invalid explicit path"),
            ({**valid, "path": "../outside.yaml"}, "invalid explicit path"),
            ({**valid, "path": "/absolute.yaml"}, "invalid explicit path"),
            ({**valid, "path": "C:/absolute.yaml"}, "invalid explicit path"),
            ({**valid, "path": "configs/recipe.json"}, "invalid explicit path"),
            ({**valid, "profile": "formal"}, "invalid explicit profile"),
            ({**valid, "profile": []}, "invalid explicit profile"),
        ]
        for entry, error in invalid:
            with self.subTest(entry=entry):
                self.manifest([entry])
                self.assert_catalog_rejected(error, "recipes[0]")

    def test_explicit_and_legacy_ids_share_one_namespace(self):
        a, b = "configs/one/shared_id.yaml", "configs/two/unrelated.yaml"
        for path in (a, b):
            self.write(path)
        explicit_a = {"id": "shared-id", "path": a, "profile": "smoke"}
        explicit_b = {"id": "shared-id", "path": b, "profile": "experiment"}
        for first, second in ((a, explicit_b), (explicit_b, a), (explicit_a, explicit_b)):
            with self.subTest(entries=(first, second)):
                self.manifest([first, second], conditional=[b], public=[a])
                self.assert_catalog_rejected("duplicate canonical recipe ID", "shared-id", a, b)

    def test_explicit_entries_preserve_resource_and_projection_safety(self):
        path = "configs/experiment/actual.yaml"
        alias = "configs/experiment/./actual.yaml"
        self.write(path)
        entry = {"id": "stable-id", "path": path, "profile": "smoke"}
        self.manifest([alias, entry])
        self.assert_catalog_rejected("duplicate resource", "stable-id", path, alias)
        self.manifest([path, entry])
        self.assert_catalog_rejected("duplicate recipes path", path)
        for section in ("public", "conditional"):
            self.manifest([entry], **{section: [path, path]})
            self.assert_catalog_rejected(f"duplicate {section} path", path)
            self.manifest([entry], **{section: [alias]})
            self.assert_catalog_rejected(f"{section} recipe manifest references unknown paths", alias)
        self.write(path, "broken: [")
        self.manifest([entry], conditional=[path])
        self.assert_catalog_rejected("invalid YAML for built-in recipe", "stable-id", path)
        missing = "configs/experiment/other.yaml"
        self.manifest([{**entry, "path": missing}], conditional=[missing])
        with patch.object(catalog.metadata, "distribution", return_value=SimpleNamespace(files=[])):
            self.assert_catalog_rejected("missing YAML for built-in recipe", "stable-id", missing,
                                         exception=FileNotFoundError)

    def test_shared_canonical_id_rejected_before_all_projections_and_lookup(self):
        a, b = "configs/one/shared_smoke.yaml", "configs/two/shared-smoke.yaml"
        self.write(a)
        self.write(b)
        self.manifest([a, b], conditional=[a], public=[b])
        self.assert_catalog_rejected("duplicate canonical recipe ID", "shared-smoke", a, b)
        with self.assertRaisesRegex(ValueError, "duplicate canonical recipe ID.*shared-smoke"):
            catalog.recipe_by_id(" SHARED_SMOKE ", self.root)
        with self.assertRaisesRegex(ValueError, "duplicate canonical recipe ID.*shared-smoke"):
            catalog.load_papers(self.root)

    def test_old_parent_fallback_collision_rejected_in_either_order(self):
        a, b = "configs/one/shared.yaml", "configs/two/shared.yaml"
        c = "configs/two/two-shared.yaml"
        for path in (a, b, c):
            self.write(path)
        # The old fallback for the second shared.yaml collided with this
        # independently named recipe in this ordering.
        for order in ((c, a, b), (b, a, c)):
            with self.subTest(order=order):
                self.manifest(order, conditional=[a], public=[c])
                self.assert_catalog_rejected("duplicate canonical recipe ID", "'shared'", a, b)

    def test_duplicate_resource_rejected_even_with_distinct_manifest_spelling(self):
        a, b = "configs/one/shared.yaml", "configs/one/./shared.yaml"
        self.write(a)
        self.manifest([a, b])
        self.assert_catalog_rejected("duplicate resource", "shared", a, b)

    def test_dangling_conditional_and_duplicate_projection_references_rejected(self):
        a, b = "configs/one/normal.yaml", "configs/one/unknown.yaml"
        self.write(a)
        for options, error, detail in (
            ({"conditional": [b]}, "conditional recipe manifest references unknown paths", b),
            ({"conditional": [a, a]}, "duplicate conditional path", a),
            ({"public": [a, a]}, "duplicate public path", a),
        ):
            with self.subTest(options=options):
                self.manifest([a], **options)
                self.assert_catalog_rejected(error, detail)

    def test_hidden_declared_yaml_must_exist_and_parse(self):
        a, b = "configs/phase1a/visible.yaml", "configs/phase1a/hidden.yaml"
        self.write(a)
        self.manifest([a, b], conditional=[b], public=[a])
        with patch.object(catalog.metadata, "distribution",
                          side_effect=metadata.PackageNotFoundError("lnl-toolbox")):
            self.assert_catalog_rejected("missing YAML for built-in recipe", "hidden", b,
                                         exception=FileNotFoundError)
        for value in ("data: [", "kind: fragment\n", "kind: experiment\ndata: {name: cifar10}\ntrainer: []"):
            with self.subTest(value=value):
                self.write(b, value)
                self.assert_catalog_rejected("invalid YAML for built-in recipe", "hidden", b)

    def test_installed_identity_rejects_multiple_matching_distribution_records(self):
        relative = "configs/phase1a/installed.yaml"
        candidates = tuple(PurePosixPath(f"{name}/share/lnl-toolbox/{relative}") for name in ("a", "b"))
        for entry in candidates:
            self.write(entry.as_posix())
        self.manifest([relative])
        dist = SimpleNamespace(files=candidates, locate_file=lambda entry: self.root / entry)
        with patch.object(catalog.metadata, "distribution", return_value=dist):
            self.assert_catalog_rejected(
                "ambiguous installed resource", "installed", *(entry.as_posix() for entry in candidates)
            )

    def test_installed_record_without_file_is_rejected(self):
        relative = "configs/installed/absent.yaml"
        record = PurePosixPath("prefix/share/lnl-toolbox", relative)
        self.manifest([relative], conditional=[relative])
        dist = SimpleNamespace(files=[record], locate_file=lambda entry: self.root / entry)
        with patch.object(catalog.metadata, "distribution", return_value=dist):
            self.assert_catalog_rejected(
                "missing YAML for built-in recipe", "absent", relative,
                exception=FileNotFoundError,
            )

    def test_source_precedes_installed_candidates_even_in_mixed_index(self):
        a, b = "configs/experiment/source.yaml", "configs/reproduction/installed.yaml"
        source = self.write(a)
        candidates = [PurePosixPath(f"{p}/share/lnl-toolbox/{a}") for p in ("x", "y")]
        candidates.append(PurePosixPath("x/share/lnl-toolbox", b))
        for entry in candidates:
            self.write(entry.as_posix())
        dist = SimpleNamespace(files=candidates, locate_file=lambda entry: self.root / entry)
        with patch.object(catalog.metadata, "distribution", return_value=dist) as lookup:
            self.manifest([a])
            self.assertEqual(catalog.recipe_by_id("source", self.root).config_path, source)
            lookup.assert_not_called()
            self.manifest([a, b])
            rows = catalog.discover_recipes(self.root)
            self.assertEqual(
                [item.config_path for item in rows], [source, self.root / candidates[-1]]
            )
            lookup.assert_called_once_with("lnl-toolbox")

    def test_valid_conditional_resource_does_not_require_runtime_inputs(self):
        a, b = "configs/phase1a/visible.yaml", "configs/phase1a/conditional.yaml"
        self.write(a)
        config = self.write(b, "kind: experiment\ndata: {name: cifar10, root: not-present}\n")
        self.manifest([a, b], conditional=[b], public=[a, b])
        with patch.object(catalog, "validate_config", side_effect=AssertionError("runtime validation")):
            for conditional, public in VIEWS:
                with self.subTest(conditional=conditional, public=public):
                    rows = catalog.discover_recipes(
                        self.root, include_conditional=conditional, public_only=public
                    )
                    self.assertEqual([r.id for r in rows],
                                     ["visible", "conditional"] if conditional else ["visible"])
                    if conditional:
                        self.assertEqual(rows[-1].config_path, config)
            self.assertEqual(catalog.recipe_by_id("conditional", self.root).config_path, config)


def verify_installed(prefix, wheel=None):
    """Run from an external cwd, with an explicitly isolated install prefix."""
    import lnl_toolbox

    prefix = prefix.resolve()
    assert Path(lnl_toolbox.__file__).resolve().is_relative_to(prefix), lnl_toolbox.__file__
    assert not Path.cwd().resolve().is_relative_to(ROOT), Path.cwd()
    root = catalog.repository_root()
    assert not (root / "configs").exists(), f"source config fallback under {root}"
    assert not (Path.cwd() / "configs").exists()
    dist = metadata.distribution("lnl-toolbox")
    files = {}
    for entry in dist.files or ():
        name = entry.as_posix()
        start = name.find(CONFIG_DEST)
        if start < 0 or not name.endswith((".yaml", ".yml")):
            continue
        resource = name[start:]
        assert resource not in files, f"duplicate installed resource: {resource}"
        resolved = Path(dist.locate_file(entry)).resolve()
        assert resolved.is_relative_to(prefix) and resolved.is_file(), resolved
        files[resource] = resolved

    expected = json.loads(BASELINE.read_text(encoding="utf-8"))
    package_map = dict(expected["packaged_only"])
    package_map.update({r["path"]: r["package_resource"] for r in expected["recipes"]})
    unittest.TestCase().assertEqual(set(files), set(package_map.values()))
    path_to_source = {files[resource]: path for path, resource in package_map.items()}
    actual, views = make_snapshot(
        root, package_map, lambda item: path_to_source[item.config_path], expected["web_formal_bindings"]
    )
    case = unittest.TestCase()
    case.maxDiff = None
    for key in expected:
        case.assertEqual(actual[key], expected[key], key)
    if wheel is not None:
        with ZipFile(wheel) as archive:
            names = sorted(archive.namelist())
            hashes = {
                name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in names if f"/{CONFIG_DEST}" in name and name.endswith((".yaml", ".yml"))
            }
        case.assertEqual(len(hashes), 71)
        for name, digest in hashes.items():
            resource = name[name.index(CONFIG_DEST):]
            case.assertEqual(hashlib.sha256(files[resource].read_bytes()).hexdigest(), digest, resource)
        case.assertEqual(_digest((names, hashes, views)), HISTORICAL_SUMMARIES["packaging_discovery"])
    print(f"PASS: isolated resources={len(files)}, recipes=68, Papers=26, associations=56, defaults=26")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--emit-source"]:
        print(json.dumps(source_snapshot(Path(sys.argv[2]).resolve())[0], ensure_ascii=True, indent=2))
    elif sys.argv[1:2] == ["--verify-installed"]:
        verify_installed(Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        unittest.main()
