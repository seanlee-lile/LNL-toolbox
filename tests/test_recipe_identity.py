"""Release identity contract frozen before the recipe identity migration."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib import metadata
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
from lnl_toolbox.cli.main import _load_source, build_parser
from lnl_toolbox.training.results import config_hash
from lnl_toolbox.training.sweep import _new_manifest, _validate_existing_manifest, plan_sweep


ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).with_name("fixtures") / "recipe_identity_baseline.json"
BASE_SHA = "c9a3234a03db7311c0dd70e6700ab5ae5971303a"
CONFIG_DEST = "share/lnl-toolbox/configs/"
VIEWS = ((False, False), (False, True), (True, False), (True, True))
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
    """Test-side gate for the legacy candidate format, not a production parser."""
    paths = raw["recipes"]
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
    for path in paths:
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
