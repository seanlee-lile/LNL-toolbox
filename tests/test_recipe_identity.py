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
# Approved moves only. Keep the on-disk baseline in its original layout.
RELOCATIONS = {
    f"configs/experiment/{filename}": f"configs/methods/{family}/{filename}"
    for family, files in {
        "apl": (
            "apl_cifar10_noise02_reproduction.yaml",
            "apl_cifar10_noise02_smoke.yaml",
            "apl_cifar10_noise04_reproduction.yaml",
        ),
        "cdr": (
            "cifar10_symmetric_cdr_reproduction.yaml",
            "cifar10_symmetric_cdr_smoke.yaml",
        ),
        "dss": (
            "dss_cifar10_symmetric05_reproduction.yaml",
            "dss_cifar10_symmetric05_smoke.yaml",
        ),
        "gce": (
            "gce_cifar10_noise02_reproduction.yaml",
            "gce_cifar10_noise02_smoke.yaml",
        ),
        "binary-risk": (
            "binary_risk_natarajan_1epoch.yaml",
            "binary_risk_natarajan_reproduction.yaml",
        ),
        "ca2c": (
            "ca2c_cifar100_reproduction.yaml",
            "ca2c_cifar10_smoke.yaml",
        ),
        "clean": (
            "cifar10_clean.yaml",
            "cifar10_clean_baseline.yaml",
            "cifar10_clean_smoke.yaml",
        ),
        "cnlcu": (
            "cifar10_cnlcu_hard_smoke.yaml",
            "cifar10_cnlcu_soft_smoke.yaml",
            "cnlcu_cifar10_reproduction.yaml",
        ),
        "coteaching": (
            "cifar10_coteaching_reproduction.yaml",
            "cifar10_coteaching_smoke.yaml",
        ),
        "cwd": (
            "cwd_cifar10_reproduction.yaml",
            "cwd_cifar10_smoke.yaml",
        ),
        "dividemix": (
            "cifar10_dividemix_smoke.yaml",
        ),
        "dual-t": (
            "cifar10_dual_t_smoke.yaml",
        ),
        "dual-t/evidence": (
            "cifar10_dual_t_evidence_smoke.yaml",
        ),
        "fine": (
            "fine_cifar100n_reproduction.yaml",
            "fine_cifar100n_smoke.yaml",
        ),
        "importance-reweighting": (
            "importance_reweighting_binary_high_dim_smoke.yaml",
            "importance_reweighting_binary_smoke.yaml",
        ),
        "jocor": (
            "jocor_cifar10_symmetric05_reproduction.yaml",
            "jocor_cifar10_symmetric05_smoke.yaml",
        ),
        "l2rw": (
            "l2rw_cifar10_reproduction.yaml",
            "l2rw_cifar10_smoke.yaml",
        ),
        "lend": (
            "cifar10_lend_smoke.yaml",
            "lend_cifar10_reproduction.yaml",
        ),
        "loss-correction": (
            "loss_correction_cifar10_asymmetric04.yaml",
        ),
        "mc-ldce": (
            "mc_ldce_cifar10_reproduction.yaml",
            "mc_ldce_cifar10_smoke.yaml",
        ),
        "pdl": (
            "pdl_cifar10_reproduction.yaml",
            "pdl_cifar10_smoke.yaml",
        ),
        "supervised-baselines": (
            "cifar10_foundation_smoke.yaml",
            "cifar10_smoke.yaml",
            "cifar10_symmetric_ce_smoke.yaml",
            "cifar10_symmetric_small_loss_linear_smoke.yaml",
            "cifar10_symmetric_small_loss_smoke.yaml",
        ),
        "t-revision": (
            "cifar10_t_revision_smoke.yaml",
        ),
    }.items()
    for filename in files
} | {
    f"configs/reproduction/{filename}": f"configs/methods/{family}/{filename}"
    for family, files in {
        "cnlcu/support": (
            "cifar10_cnlcu_soft_sym20_short.yaml",
        ),
        "dividemix": (
            "cifar10_dividemix_sym20.yaml",
        ),
        "dual-t": (
            "cifar10_dual_t_sym20.yaml",
        ),
        "dual-t/evidence": (
            "cifar10_dual_t_sym20_evidence.yaml",
        ),
        "importance-reweighting": (
            "importance_reweighting_binary_high_dim.yaml",
            "importance_reweighting_binary_low_dim.yaml",
            "uci_heart_importance_reweighting.yaml",
        ),
        "t-revision": (
            "cifar10_t_revision_sym20_short.yaml",
            "cifar10_t_revision_sym20_reproduction.yaml",
        ),
    }.items()
    for filename in files
}
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
ADDITIONAL_CONSUMER_SMOKE_BATCH = {
    "dss-cifar10-symmetric05-smoke": "dss",
    "gce-cifar10-noise02-smoke": "gce",
    "cal-cifar10-smoke": "cal",
    "cifar10-dld-smoke": "dld",
}
FIRST_FORMAL_BATCH = {
    "ca2c-cifar100-reproduction": "ca2c",
    "dss-cifar10-symmetric05-reproduction": "dss",
    "gce-cifar10-noise02-reproduction": "gce",
    "jocor-cifar10-symmetric05-reproduction": "jocor",
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
    paths = [entry["path"] for entry in entries]
    registered = _unique(paths, "manifest path")
    for kind, refs, targets in (
        ("normal", paths, set(known_paths)),
        ("public", [item["path"] for item in raw.get("public", ())], registered),
        ("conditional", raw.get("conditional", ()), registered),
    ):
        _unique(refs, f"{kind} reference")
        if missing := set(refs) - targets:
            raise AssertionError(f"dangling {kind} reference: {sorted(missing)}")

    allocated = _unique(entries, "canonical identity", key=lambda entry: entry["id"])
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


def _current_path(frozen_path):
    return RELOCATIONS.get(frozen_path, frozen_path)


def _current_resource(resource):
    prefix = "share/lnl-toolbox/"
    return prefix + _current_path(resource.removeprefix(prefix))


def _frozen_layout():
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def _package_layout(frozen):
    package = dict(frozen["packaged_only"])
    package.update({r["path"]: r["package_resource"] for r in frozen["recipes"]})
    return package


def source_snapshot(root=ROOT):
    bindings_file = root / "web" / "lnl_parameter_metadata_registry_revised.yaml"
    bindings = yaml.safe_load(bindings_file.read_text(encoding="utf-8"))["formal_recipe_bindings"]
    frozen_package = _package_layout(_frozen_layout())
    current_to_frozen = {_current_path(path): path for path in frozen_package}
    unittest.TestCase().assertEqual(_package_map(root), {
        _current_path(path): _current_resource(resource)
        for path, resource in frozen_package.items()
    })
    return make_snapshot(
        root, frozen_package,
        lambda item: current_to_frozen[item.config_path.relative_to(root).as_posix()], bindings,
    )


class RecipeIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = _frozen_layout()
        cls.expected = {
            **cls.frozen,
            "recipes": [
                dict(row, path=_current_path(row["path"]),
                     package_resource=_current_resource(row["package_resource"]))
                for row in cls.frozen["recipes"]
            ],
        }
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
                [row["id"] for row in self.frozen[section]], section,
            )
            for actual, expected in zip(self.actual[section], self.frozen[section]):
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

    def test_pilot_recipe_preserves_canonical_alias_and_config(self):
        recipe_id = "cifar10-symmetric-small-loss-linear-smoke"
        expected = next(row for row in self.expected["recipes"] if row["id"] == recipe_id)
        manifest_path = ROOT / "src/lnl_toolbox/cli/data/recipe_catalog.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        pilot = {"id": recipe_id, "path": expected["path"], "profile": expected["profile"]}
        self.assertIn(pilot, raw["recipes"])
        actual = catalog.recipe_by_id(" CIFAR10_SYMMETRIC_SMALL_LOSS_LINEAR_SMOKE ", ROOT)
        self.assertEqual(actual, catalog.recipe_by_id(recipe_id, ROOT))
        self.assertEqual(config_hash(catalog.load_recipe_config(actual)), expected["normalized_config_hash"])

    def test_manifest_fully_explicit_and_in_frozen_order(self):
        raw = json.loads(
            (ROOT / "src/lnl_toolbox/cli/data/recipe_catalog.json").read_text(encoding="utf-8")
        )
        entries = raw["recipes"]
        fields = ("id", "path", "profile")
        self.assertEqual(len(entries), 68)
        for index, entry in enumerate(entries, start=1):
            self.assertIsInstance(entry, dict, f"manifest entry {index}")
            self.assertEqual(set(entry), set(fields), f"manifest entry {index}")
        for field in ("id", "path"):
            self.assertEqual(len({entry[field] for entry in entries}), 68, field)
        self.assertEqual(
            entries, [{key: row[key] for key in fields} for row in self.expected["recipes"]]
        )

    def test_first_low_risk_batch_preserves_cli_path_and_sweep_identity(self):
        batch = ("cifar10-clean", "cifar10-symmetric-small-loss-smoke")
        expected = {row["id"]: row for row in self.expected["recipes"]}
        raw = json.loads(
            (ROOT / "src/lnl_toolbox/cli/data/recipe_catalog.json").read_text(encoding="utf-8")
        )
        invalid_entries = [item["path"] if item["id"] in batch else item for item in raw["recipes"]]
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
                **raw, "recipes": invalid_entries,
            })
            with self.assertRaisesRegex(ValueError, "requires explicit recipe objects"):
                cli_load("--recipe", batch[0])
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
                canonical = cli_load("--recipe", key)
                self.assertEqual(explicit[key], canonical)
                self.assertEqual(path.relative_to(ROOT).as_posix(), expected[key]["path"])
                self.assertEqual(
                    (recipe.id, recipe.profile, recipe.runner, recipe.method),
                    (key, *(expected[key][field] for field in ("profile", "runner", "method"))),
                )
                self.assertEqual(
                    config_hash(catalog.load_recipe_config(recipe)),
                    expected[key]["normalized_config_hash"],
                )
                old_config, _, old_recipe, _ = canonical
                for matrix in (None, {"trainer.epochs": [1, 2]}):
                    with self.subTest(matrix=matrix):
                        current = plan_sweep(config, [8, 3], matrix=matrix, recipe=recipe.id)
                        previous = plan_sweep(old_config, [8, 3], matrix=matrix, recipe=old_recipe.id)
                        self.assertEqual(current, previous)
                        _validate_existing_manifest(_new_manifest(previous), current)

    def test_paper_variant_batch_preserves_selection_path_and_sweep(self):
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

        for _, plans in observe().values():
            for plan in plans:
                _validate_existing_manifest(_new_manifest(plan), plan)

    def test_paper_smoke_batch_preserves_order_and_cli_recommendations(self):
        self._assert_paper_smoke_batch(PAPER_SMOKE_BATCH)

    def test_standard_paper_smoke_batch_preserves_order_and_cli_recommendations(self):
        self._assert_paper_smoke_batch(STANDARD_PAPER_SMOKE_BATCH)

    def test_final_standard_smoke_batch_preserves_paper_and_runtime_contracts(self):
        self._assert_paper_smoke_batch(FINAL_STANDARD_PAPER_SMOKE_BATCH)

    def test_internal_lifecycle_smoke_batch_preserves_paper_and_runtime_contracts(self):
        self._assert_paper_smoke_batch(INTERNAL_LIFECYCLE_SMOKE_BATCH)

    def test_additional_consumer_smoke_batch_preserves_paper_and_secondary_consumers(self):
        self._assert_paper_smoke_batch(ADDITIONAL_CONSUMER_SMOKE_BATCH)

    def test_first_formal_batch_preserves_defaults_and_consumer_bindings(self):
        self._assert_paper_smoke_batch(FIRST_FORMAL_BATCH, formal=True)

    def _observe_additional_smoke_consumer(self, recipe, config, command):
        if recipe.id == "dss-cifar10-symmetric05-smoke":
            listed = command("compose", "list", "--runner", "supervised")
            self.assertIn(f"  - {recipe.id}\n", listed)
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "dss.yaml"
                with patch("lnl_toolbox.cli.main.load_yaml", wraps=catalog.load_yaml) as load:
                    created = command("compose", "create", "--base", recipe.id, "--output", str(target))
                load.assert_called_once_with(recipe.config_path)
                self.assertIn(recipe.id, created)
                self.assertEqual(config_hash(catalog.load_yaml(target)), config_hash(config))
            return
        if recipe.id == "gce-cifar10-noise02-smoke":
            snapshot = _sweep_snapshot({recipe.id: recipe}, {recipe.id: config})
            self.assertEqual(snapshot, self.expected["sweep"])
            return

        from lnl_toolbox.training.service import ExperimentService
        from web.external_resources import resources

        runner = ExperimentService().preflight(config, check_data=False)
        self.assertEqual(runner.name, recipe.runner)
        requirements = runner.requirements(config)
        self.assertFalse(requirements.prerequisites)
        self.assertEqual(resources(config, ROOT), [])
        if recipe.runner == "cal":
            self.assertEqual(config["data"]["name"], "synthetic_multiclass")
            self.assertEqual(config["model"]["name"], "feature_mlp")
            self.assertEqual(config["noise"]["name"], "symmetric")
            self.assertFalse(requirements.required_config_inputs)
        else:
            from lnl_toolbox.training import dld_experiment
            from tests.test_dld import _dld_workflow__data

            self.assertEqual(config["dld"]["feature_extractor"]["source"], "repository_frozen_model")
            train, test = _dld_workflow__data("train", 20), _dld_workflow__data("test", 8)

            class ExtractorReady(Exception):
                pass

            def inspect_extractor(model, *_args, **_kwargs):
                self.assertFalse(model.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
                raise ExtractorReady(dld_experiment._model_identity(model))

            # Halt at feature extraction; no diffusion training or network access.
            with tempfile.TemporaryDirectory() as directory, \
                 patch("lnl_toolbox.data.sources.load_cifar10",
                       side_effect=lambda root, split: train if split == "train" else test), \
                 patch.object(dld_experiment, "load_dld_feature_source") as external, \
                 patch.object(dld_experiment, "_feature_snapshot", side_effect=inspect_extractor), \
                 self.assertRaises(ExtractorReady) as ready:
                runtime = {**config, "data": {**config["data"], "root": directory}}
                runner.invoke(runtime, output_dir=Path(directory) / "run")
            external.assert_not_called()
            self.assertTrue(str(ready.exception))
        command("validate", "--recipe", recipe.id)
        command("run", "--recipe", recipe.id, "--dry-run", "--no-check-data")

    def _observe_internal_lifecycle(self, recipe, config, command):
        from lnl_toolbox.training.service import ExperimentService

        runner = ExperimentService().preflight(config, check_data=False)
        self.assertEqual(runner.name, recipe.runner)
        requirements = runner.requirements(config)
        self.assertFalse(requirements.prerequisites)
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
        command("validate", "--recipe", recipe.id)
        command("run", "--recipe", recipe.id, "--dry-run", "--no-check-data")

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

    def _assert_paper_smoke_batch(self, batch, *, formal=False):
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

        def formal_consumers():
            from lnl_toolbox.quickstart.service import QuickStartService
            from web import command_console

            # Use fresh production catalog reads even if another test warmed the cache.
            caches = (command_console._web_paper_specs, command_console._web_recipe_config,
                      command_console._registry_recipe_paths, command_console._formal_registry_config)
            for cached in caches:
                cached.cache_clear()
                self.addCleanup(cached.cache_clear)
            with patch.object(command_console, "_WEB_RECIPE_CATALOG", None):
                payload = {row["id"]: row for row in command_console._paper_payload()}
                quick_start = QuickStartService()
                recipes = quick_start._recipes()
                papers = {paper.id: paper for paper in catalog.load_papers(ROOT)}
                for default_id, paper_id in batch.items():
                    with self.subTest(paper=paper_id, consumer="formal"):
                        row = payload[paper_id]
                        expected = expected_papers[paper_id]
                        self.assertEqual(row["default_recipe_id"], default_id)
                        self.assertEqual([item["recipe_id"] for item in row["configs"]],
                                         [item["recipe_id"] for item in expected["associations"]])
                        selected = next(item for item in row["configs"] if item["recipe_id"] == default_id)
                        spec = expected_recipes[default_id]
                        self.assertEqual((selected["config_path"], selected["runner"], selected["method"]),
                                         (spec["path"], spec["runner"], spec["method"]))
                        web_id, _ = command_console._formal_registry_config(paper_id)
                        self.assertEqual(web_id, default_id)
                        self.assertEqual(command_console._registry_recipe_paths()[web_id], ROOT / spec["path"])
                        template = quick_start._template(papers[paper_id], recipes)
                        self.assertEqual(template.recipe_id, default_id)
                        self.assertEqual(template.recipe, recipes[default_id])
                        self.assertEqual(config_hash(template.config), spec["normalized_config_hash"])

        def observe():
            listed = cli("list", "--format", "json")
            recommendations = {row["id"]: row["recommended"] for row in json.loads(listed)}
            self.assertEqual(recommendations,
                             {key: row["cli_recommended"] for key, row in expected_papers.items()})
            papers = {paper.id: paper for paper in catalog.load_papers(ROOT)}
            for recipe_id, paper_id in batch.items():
                with self.subTest(paper=paper_id):
                    paper, expected = papers[paper_id], expected_papers[paper_id]
                    recommended = expected["cli_recommended"] if formal else recipe_id
                    ordered = [item.recipe_id for item in paper.configs]
                    self.assertEqual(ordered, [item["recipe_id"] for item in expected["associations"]])
                    self.assertEqual([asdict(item) for item in paper.configs], expected["associations"])
                    self.assertEqual(catalog.default_paper_config(paper, root=ROOT)[1].id, expected["default"])
                    if formal:
                        self.assertEqual(expected["default"], recipe_id)
                    self.assertNotEqual(expected["default"], recommended)
                    self.assertEqual(recommendations[paper_id], recommended)
                    self.assertEqual(ordered[0], recommended)
                    shown = cli("show", paper_id)
                    self.assertIn(f"lnl run --recipe {recommended} --dry-run", shown)
                    configs_by_id = {}
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
                            self._observe_internal_lifecycle(recipe, parsed, command)
                        if recipe.id == recommended and recommended in ADDITIONAL_CONSUMER_SMOKE_BATCH:
                            self._observe_additional_smoke_consumer(recipe, parsed, command)
                        if formal and recipe.id == recipe_id:
                            self.assertEqual(recipe.profile, "reproduction")
                            self.assertEqual(resolve_runner(parsed).name, row["runner"])
                            self.assertEqual(cli("config", paper_id, "--profile", "reproduction"), config)
                        if recipe.id == "l2rw-cifar10-smoke":
                            from_path = _load_source(parser.parse_args(
                                ["run", "--config", str(recipe.config_path)]
                            ))
                            from_id = _load_source(parser.parse_args(["run", "--recipe", recipe.id]))
                            self.assertEqual(from_path[:3], (*from_id[:2], None))
                        configs_by_id[recipe.id] = config
                    if formal:
                        # A variant can span smoke and reproduction. Variant-only selection
                        # uses the first ordered match, not necessarily the formal recipe.
                        seen_variants = set()
                        for item in expected["associations"]:
                            variant = item["variant"]
                            if variant not in seen_variants:
                                chosen = cli("config", paper_id, "--variant", variant)
                                self.assertEqual(chosen, configs_by_id[item["recipe_id"]])
                                seen_variants.add(variant)
                    if paper_id == "t-revision":
                        short = catalog.recipe_by_id("cifar10-t-revision-sym20-short", ROOT)
                        row = expected_recipes[short.id]
                        self.assertNotIn(short.id, ordered)
                        self.assertEqual((short.runner, short.profile, short.config_path),
                                         ("t_revision", row["profile"], ROOT / row["path"]))
                        self.assertEqual(config_hash(catalog.load_recipe_config(short)),
                                         row["normalized_config_hash"])
                    if paper_id == "dual-t":
                        evidence = catalog.recipe_by_id("cifar10-dual-t-evidence-smoke", ROOT)
                        self.assertNotIn(evidence.id, ordered)
                        self.assertIn("evidence", catalog.load_recipe_config(evidence))
            if formal:
                formal_consumers()

        observe()

    def test_dedicated_and_compose_batch_preserves_cli_and_execution_projection(self):
        batch = (*DEDICATED_BATCH, COMPOSE_BATCH_RECIPE)
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

            self.assertEqual(config_hash(generated),
                             expected[COMPOSE_BATCH_RECIPE]["normalized_config_hash"])
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
                    cli("validate", "--recipe", key)
                    cli("run", "--recipe", key, "--dry-run", "--no-check-data")
                    for matrix in (None, {budget_path: [1, 2]}):
                        plan = plan_sweep(config, [8, 3], recipe=key, matrix=matrix)
                        _validate_existing_manifest(_new_manifest(plan), plan)

        observe()

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
        a, b = "configs/a/shared.yaml", "configs/b/shared.yaml"
        first = _manifest_entry(a, "shared")
        second = _manifest_entry(b, "shared")
        for raw, real, paper, error in (
            ({"recipes": [first, first]}, [a], [], "duplicate manifest path"),
            ({"recipes": [first, second]}, [a, b], [], "duplicate canonical identity"),
            ({"recipes": [first]}, [], [], "dangling normal"),
            ({"recipes": [first], "public": [{"path": b}]}, [a], [], "dangling public"),
            ({"recipes": [first], "conditional": [b]}, [a], [], "dangling conditional"),
            ({"recipes": [first]}, [a], ["removed"], "dangling Paper"),
        ):
            with self.subTest(error=error, candidate=raw):
                with self.assertRaisesRegex(AssertionError, error):
                    verify_candidate_manifest(raw, real, paper)

    def test_discovery_gate_rejects_cross_view_resource_drift(self):
        with self.assertRaisesRegex(AssertionError, "shared moved from"):
            verify_view_identities({"first": [("shared", "a")], "second": [("shared", "b")]})


def _manifest_entry(path, recipe_id, profile="experiment"):
    return {"id": recipe_id, "path": path, "profile": profile}


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

    def manifest(self, entries, *, conditional=(), public=()):
        self.resource_files.return_value.joinpath.return_value.read_text.return_value = json.dumps({
            "recipes": entries, "conditional": conditional,
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
        other = "configs/reproduction/totally_unrelated_smoke.yaml"
        source = self.write(path)
        self.write(other)
        self.manifest(
            [_manifest_entry(other, "other-smoke", "smoke"),
             _manifest_entry(path, "stable-recipe-id", "reproduction")],
            conditional=[path], public=[path],
        )
        recipe = catalog.recipe_by_id(" STABLE_RECIPE_ID ", self.root)
        self.assertEqual((recipe.id, recipe.profile, recipe.config_path),
                         ("stable-recipe-id", "reproduction", source))
        with self.assertRaisesRegex(ValueError, "unknown recipe"):
            catalog.recipe_by_id("totally-unrelated-smoke", self.root)
        expected_ids = (
            ["other-smoke"], [], ["other-smoke", "stable-recipe-id"], ["stable-recipe-id"],
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

    def test_configuration_fidelity_follows_profile_not_resource_path(self):
        recipe_id = "stable-fidelity"
        original = "configs/reproduction/budget_reproduction.yaml"
        moved = "configs/methods/example/budget_smoke.yaml"
        for profile, default in (
            ("experiment", "engineering"),
            ("smoke", "smoke"),
            ("reproduction", "paper_oriented"),
        ):
            for override in (None, "paper_protocol"):
                for path in (original, moved):
                    with self.subTest(profile=profile, override=override, path=path):
                        source = "kind: experiment\ndata: {name: cifar10}\n"
                        if override is not None:
                            source += f"configuration_fidelity: {override}\n"
                        resource = self.write(path, source)
                        self.manifest([_manifest_entry(path, recipe_id, profile)])
                        recipe = catalog.recipe_by_id(recipe_id, self.root)
                        self.assertEqual((recipe.id, recipe.profile, recipe.config_path),
                                         (recipe_id, profile, resource))
                        self.assertEqual(recipe.configuration_fidelity, override or default)

    def test_invalid_explicit_metadata_cannot_fall_back_to_path_inference(self):
        path = "configs/reproduction/fallback.yaml"
        self.write(path)
        valid = {"id": "stable-id", "path": path, "profile": "smoke"}
        invalid = [
            ({key: value for key, value in valid.items() if key != missing}, f"missing=.*{missing}")
            for missing in valid
        ]
        invalid += [
            (path, "requires explicit recipe objects"),
            (None, "requires explicit recipe objects"),
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

    def test_string_entry_is_rejected_in_an_otherwise_explicit_manifest(self):
        a, b = "configs/one/shared_id.yaml", "configs/two/unrelated.yaml"
        explicit_b = _manifest_entry(b, "unrelated")
        for first, second, position in ((a, explicit_b, 0), (explicit_b, a, 1)):
            with self.subTest(entries=(first, second)):
                self.manifest([first, second], conditional=[b], public=[a])
                self.assert_catalog_rejected("requires explicit recipe objects", f"recipes[{position}]")

    def test_explicit_entries_preserve_resource_and_projection_safety(self):
        path = "configs/experiment/actual.yaml"
        alias = "configs/experiment/./actual.yaml"
        self.write(path)
        entry = {"id": "stable-id", "path": path, "profile": "smoke"}
        self.manifest([_manifest_entry(alias, "alias"), entry])
        self.assert_catalog_rejected("invalid explicit path", alias)
        self.manifest([_manifest_entry(path, "alias"), entry])
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
        self.manifest([_manifest_entry(a, "shared-smoke"), _manifest_entry(b, "shared-smoke")],
                      conditional=[a], public=[b])
        self.assert_catalog_rejected("duplicate canonical recipe ID", "shared-smoke", a, b)
        with self.assertRaisesRegex(ValueError, "duplicate canonical recipe ID.*shared-smoke"):
            catalog.recipe_by_id(" SHARED_SMOKE ", self.root)
        with self.assertRaisesRegex(ValueError, "duplicate canonical recipe ID.*shared-smoke"):
            catalog.load_papers(self.root)

    def test_duplicate_explicit_id_rejected_in_either_order(self):
        a, b = "configs/one/shared.yaml", "configs/two/shared.yaml"
        c = "configs/two/two-shared.yaml"
        for path in (a, b, c):
            self.write(path)
        ids = {a: "shared", b: "shared", c: "two-shared"}
        for order in ((c, a, b), (b, a, c)):
            with self.subTest(order=order):
                self.manifest([_manifest_entry(p, ids[p]) for p in order], conditional=[a], public=[c])
                self.assert_catalog_rejected("duplicate canonical recipe ID", "'shared'", a, b)

    def test_duplicate_physical_resource_rejected_with_distinct_valid_paths(self):
        a, b = "configs/one/shared.yaml", "configs/two/shared.yaml"
        shared = self.write("shared.yaml")
        records = [PurePosixPath("prefix/share/lnl-toolbox", path) for path in (a, b)]
        self.manifest([_manifest_entry(a, "first"), _manifest_entry(b, "second")])
        dist = SimpleNamespace(files=records, locate_file=lambda entry: shared)
        with patch.object(catalog.metadata, "distribution", return_value=dist):
            self.assert_catalog_rejected("duplicate resource", "second", a, b)

    def test_dangling_conditional_and_duplicate_projection_references_rejected(self):
        a, b = "configs/one/normal.yaml", "configs/one/unknown.yaml"
        self.write(a)
        for options, error, detail in (
            ({"conditional": [b]}, "conditional recipe manifest references unknown paths", b),
            ({"conditional": [a, a]}, "duplicate conditional path", a),
            ({"public": [a, a]}, "duplicate public path", a),
        ):
            with self.subTest(options=options):
                self.manifest([_manifest_entry(a, "normal")], **options)
                self.assert_catalog_rejected(error, detail)

    def test_hidden_declared_yaml_must_exist_and_parse(self):
        a, b = "configs/phase1a/visible.yaml", "configs/phase1a/hidden.yaml"
        self.write(a)
        self.manifest([_manifest_entry(a, "visible"), _manifest_entry(b, "hidden")],
                      conditional=[b], public=[a])
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
        self.manifest([_manifest_entry(relative, "declared-id")])
        dist = SimpleNamespace(files=candidates, locate_file=lambda entry: self.root / entry)
        with patch.object(catalog.metadata, "distribution", return_value=dist):
            self.assert_catalog_rejected(
                "ambiguous installed resource", "declared-id", *(entry.as_posix() for entry in candidates)
            )

    def test_installed_record_without_file_is_rejected(self):
        relative = "configs/installed/absent.yaml"
        record = PurePosixPath("prefix/share/lnl-toolbox", relative)
        self.manifest([_manifest_entry(relative, "absent")], conditional=[relative])
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
            self.manifest([_manifest_entry(a, "source")])
            self.assertEqual(catalog.recipe_by_id("source", self.root).config_path, source)
            lookup.assert_not_called()
            self.manifest([_manifest_entry(a, "source"), _manifest_entry(b, "installed")])
            rows = catalog.discover_recipes(self.root)
            self.assertEqual(
                [item.config_path for item in rows], [source, self.root / candidates[-1]]
            )
            lookup.assert_called_once_with("lnl-toolbox")

    def test_valid_conditional_resource_does_not_require_runtime_inputs(self):
        a, b = "configs/phase1a/visible.yaml", "configs/phase1a/conditional.yaml"
        self.write(a)
        config = self.write(b, "kind: experiment\ndata: {name: cifar10, root: not-present}\n")
        self.manifest([_manifest_entry(a, "visible"), _manifest_entry(b, "conditional")],
                      conditional=[b], public=[a, b])
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

    expected = _frozen_layout()
    package_map = _package_layout(expected)
    resource_to_frozen = {_current_resource(resource): resource for resource in package_map.values()}
    unittest.TestCase().assertEqual(set(files), set(resource_to_frozen))
    path_to_source = {
        files[_current_resource(resource)]: path for path, resource in package_map.items()
    }
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
        def historical_archive_name(name):
            start = name.find(CONFIG_DEST)
            if start < 0 or name[start:] not in resource_to_frozen:
                return name
            return name[:start] + resource_to_frozen[name[start:]]

        frozen_names = sorted(historical_archive_name(name) for name in names)
        frozen_hashes = {historical_archive_name(name): digest for name, digest in hashes.items()}
        case.assertEqual(
            _digest((frozen_names, frozen_hashes, views)), HISTORICAL_SUMMARIES["packaging_discovery"]
        )
    print(f"PASS: isolated resources={len(files)}, recipes=68, Papers=26, associations=56, defaults=26")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--emit-source"]:
        print(json.dumps(source_snapshot(Path(sys.argv[2]).resolve())[0], ensure_ascii=True, indent=2))
    elif sys.argv[1:2] == ["--verify-installed"]:
        verify_installed(Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        unittest.main()
