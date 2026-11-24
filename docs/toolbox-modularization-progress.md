# Toolbox modularization progress

## Recipe Identity Migration Phase 1C — self-contained lifecycle smoke batch

Base: `code@be8c13b3b940d1888f8339584286edd3877604c2` (pushed Batch 6).
Converted only the PCSE, MC-LDCE, PDL and Dual-T smoke recommendations in
place: 26 explicit and 42 legacy entries. YAML, registry order, Paper/Web
metadata, and the frozen fixture are unchanged.

- The before/after test compares ordered Paper associations, CLI smoke
  recommendations, formal defaults and real `papers list/show/config`
  output. It also compares static preflight, runner plans, no-data dry-run,
  and Sweep identity. PCSE still pretrains internally and uses no external
  checkpoint. MC-LDCE still uses a configured matrix and a short synthetic
  test checks its generated feature, transition and statistic file names.
- PDL still runs the warmup/instance-transition workflow with no external
  input artifact. A no-training probe intercepted the real runner's resume
  lookup and confirmed the same run-local `instance_transition_artifact.npz`
  in both representations. Dual-T remains separate from the evidence recipe.
  Existing method tests cover artifact loading, validation and resume.
- Deferred recipes are untouched. Raw JSON consumers still need to handle
  both path strings and explicit objects.
- Focused: 293 passed / 1 Node skip, including PCSE, MC-LDCE, PDL and Dual-T
  method tests. Isolated wheel and sdist-derived installs matched the
  frozen snapshot and CLI discovery; each installed index build still
  reads the distribution file list once.
- Full suites: Main 1178 passed / 21 skipped; Web 110 passed / 31 skipped;
  Scratch 196 passed / 10 skipped. Zero failures or errors. Only the
  three approved existing files changed. Await human review; no stage,
  commit, push, or next batch.

## Recipe Identity Migration Phase 1C — final standard Paper smoke batch

Base: `code@86c3d547c8237399ccef524d5885887690e16319` (pushed Batch 5).
Converted only `cwd-cifar10-smoke`, `fine-cifar100n-smoke`, and
`cifar10-lend-smoke` in place. The registry has 22 explicit and 46 legacy
entries. No YAML, path, ordering, Paper/Web metadata, or frozen fixture
changes.

- The before/after test compares ordered Paper associations, CLI smoke
  recommendations, formal defaults and real `papers list/show/config`
  output for all three Papers. CWD remains two-class with no transition
  artifact input; FINE still uses CIFAR-100 synthetic symmetric noise
  without an external label resource; LEND smoke remains distinct from
  the formal default and uses the paper-equation implementation.
- Deferred recipes are untouched. Raw manifest consumers still need to
  accept both path strings and explicit identity objects.
- Focused: 172 passed / 1 Node skip. Isolated wheel and sdist-derived
  install checks match the frozen snapshot and installed CLI output.
  Each installed index build still reads the distribution file list once.
- Full suites: Main 1177 passed / 21 skipped; Web 110 passed / 31 skipped;
  Scratch 196 passed / 10 skipped. Zero failures and errors. Only the
  three approved existing files changed. Await human review; no stage,
  commit, push, or next batch.

## Recipe Identity Migration Phase 1C — standard Paper smoke recommendations

Base: `code@9fef04a020c496483f30684467651bba7d9aecb2` (pushed Batch 4).
Only the CA2C, DivideMix, JoCoR, L2RW and T-Revision standard smoke
recommendations were converted in place. The manifest now contains 19
explicit and 49 legacy entries. YAML, resource paths, manifest order,
Paper/Web metadata, and the frozen fixture are unchanged.

- The shared before/after test checks the real `papers list/show/config`
  commands, ordered associations, formal defaults and all published
  variants for these five Papers. It also checks L2RW direct YAML loading,
  DivideMix and the other smoke runners' prerequisite sets, and
  coexistence with the already-explicit T-Revision short recipe.
- Deferred recipes were not changed. Direct consumers of the raw JSON
  still need to support both strings and explicit objects.
- Focused: 171 passed / 1 Node skip. Both isolated wheel and sdist installs
  match the frozen snapshot and installed CLI output, with one distribution
  file-list read per index build.
- Full suites: Main 1176 passed / 21 skipped; Web 110 passed / 31 skipped;
  Scratch 196 passed / 10 skipped; zero failures or errors. Only the three
  approved files changed. Await review; no stage, commit, push, or next batch.

## Recipe Identity Migration Phase 1C — first Paper smoke recommendation batch

Base: `code@07fb9dc38207b7585b1d09aa5fef1ffd4199cb98` (pushed Batch 3).
Only the APL, CNLCU-S, Co-teaching and CDR smoke recommendations were
converted in place; the manifest contains 14 explicit and 54 legacy
entries. YAML, registry order, Paper/Web metadata, and the frozen fixture
are unchanged.

- The before/after test calls the real CLI `papers list/show/config`
  commands. It compares all ordered associations, formal defaults and
  per-variant selections for the four Papers, including CNLCU-S and
  CNLCU-H together and the other APL variants. Selected paths, runner,
  method and normalized configuration hashes are checked.
- Paper consumes canonical IDs through catalog APIs, not raw manifest
  entry types. External raw JSON consumers still need to accept both
  path strings and explicit objects. Deferred recipes are unchanged.
- Focused: 170 passed / 1 Node skip. Isolated wheel and sdist-derived
  installs matched the frozen snapshot, ordered CLI discovery and Paper
  recommendations. Each installed index build still reads the distribution
  file list once.
- Full suites: Main 1175 passed / 21 skipped; Web 110 passed / 31 skipped;
  Scratch 196 passed / 10 skipped. Zero failures and errors. The three
  approved files are the only changes. Await review; no stage, commit,
  push, or next batch.

## Recipe Identity Migration Phase 1C — third batch

Base: `code@208d9d0dbb57a98ab9b2b8777dfa3a02f32d4464` (pushed Batch 2).
This batch converts four entries in place: `binary-risk-natarajan-1epoch`,
`importance-reweighting-binary-high-dim-smoke`, `cifar10-t-revision-sym20-short`,
and `cifar10-foundation-smoke`. The manifest now has 10 explicit and 58
legacy entries. Paths, order, YAML, Paper/Web bindings and the frozen fixture
are unchanged. DSS and the higher-coupling entries are deferred.

- The real Compose list and create commands are checked against both forms.
  Foundation smoke uses its existing canonical ID and source path; the
  recommendation list, output guidance and generated YAML are compared.
- Dedicated-runner checks compare CLI validation, no-data dry-run, direct
  YAML lookup and seed/matrix Sweep plans. T-Revision uses its existing
  `t_revision.revision.epochs` budget field, not `trainer.epochs`.
- No new raw manifest consumer was found. External consumers still need to
  support both path strings and explicit objects.
- Focused: 169 passed / 1 Node skip. Main: 1174 passed / 21 skipped. Web:
  110 passed / 31 skipped. Scratch: 196 passed / 10 skipped. No failures or errors.
- External wheel and sdist installs matched the frozen snapshot: 68 unique
  IDs and resources, 71 packaged YAMLs, discovery 63/4/68/4, 26 Papers,
  56 ordered associations and 26 defaults. Installed CLI passed; both
  installs read the distribution file list once per index build.
- Scope check: three approved files, no additions. Wait for human review
  before staging, committing, pushing, or migrating further.

## Recipe Identity Migration Phase 1C — second batch (2026-11-23)

Base: `code@868e4235d39fa2b187db5a278b6f8e0ab9712b48` (pushed Batch 1).
The frozen identity fixture is unchanged.

- In-place conversions: `apl-cifar10-noise04-reproduction`, `cifar10-cnlcu-hard-smoke`,
  and `importance-reweighting-binary-high-dim`. Registry: 6 explicit / 62
  legacy. No YAML, path, order, visibility, or runtime metadata changes.
- Production selection preserves APL sym40 / paper-aligned, CNLCU-H (with
  CNLCU-S still the CLI smoke recommendation), and the synthetic high-dim
  Importance Reweighting variant. Tests compare ordered Paper associations,
  formal defaults, CLI recommendations, direct YAML loads, and Sweep plans
  against the legacy form. UCI Heart and its Web external-resource contract
  are outside this batch and untouched.
- Raw JSON consumers must still accept both path strings and explicit objects.
  There is no new schema change or new external-resource dependency.
- Focused: 168 passed / 1 Node skip. Source and isolated wheel/sdist
  installs matched the frozen identities, 68 unique resources and 71
  packaged YAMLs, the 63/4/68/4 discovery views, 26 Papers, 56 ordered
  associations and 26 defaults. Installed CLI recommendations matched;
  both installs read the distribution file list once per index build.
- Full regressions: Main 1173 passed / 21 skipped, Web 110 passed / 31
  skipped, Scratch 196 passed / 10 skipped; zero failures or errors.
  Scope check: three approved existing files only; no additions.
  Next step: human review. No stage, commit, push, or next batch.

## Recipe Identity Migration Phase 1C — first low-risk batch (2026-11-22)

Base: `code@e8f6f1c554ae05e657ff6a692f98f4b8457acc49` (pushed Phase 1B).
The frozen experiment contract remains
`c9a3234a03db7311c0dd70e6700ab5ae5971303a`.

- Converted only `cifar10-clean` (`experiment`, clean runner) and
  `cifar10-symmetric-small-loss-smoke` (`smoke`, supervised runner, fixed
  small-loss keep rate 0.5). Both are internal, nonconditional, and lack
  Paper associations, Web formal bindings, and external artifact requirements.
  The original pilot remains explicit: 3 explicit and 65 legacy entries.
- Compared the manifest to the Phase 1B commit: only the two approved entries
  changed representation, at their original positions. Paths, projections,
  YAML, normalized config hashes, runtime metadata, and package layout did
  not change. The production resolver and packaging implementation are unchanged.
- The focused suite passed 167 tests with one Node skip. It retains the Phase
  1B decoupling and mixed-format fail-closed assertions, tests the two recipes
  against their legacy entries and the frozen fixture, and compares CLI
  lookup and seed-only/matrix Sweep plans, roots, and member identities.
  Both direct `--config` and positional path usage for the clean YAML still
  bypass built-in recipe lookup.
- Direct wheel and sdist-derived installs from an external directory match
  the frozen 68 recipes/resources, 71 YAMLs, 63/4/68/4 ordered views, and
  26 Papers / 56 ordered associations / 26 defaults. Installed CLI discovery
  and Paper recommendations also matched.
- Stage 2 is intact: wheel and sdist-derived installs still read the
  distribution file list once per index build; all views parse 68 YAMLs.
  Discovery medians in milliseconds, Phase 1B to this batch: wheel
  default/full 386.3/417.4 to 418.6/436.1; sdist install 494.7/487.1 to
  503.5/485.9. Separate source checkouts read 294.7/302.8 versus
  519.7/481.1; alternating the two manifests with the same resolver and
  source directory measured 529.0/532.4 versus 520.6/521.5.
  The timings are local observations, not regression thresholds.
- The official catalog API and CLI continue to accept the same IDs. External
  tooling that assumes every raw `recipes[]` item is a string still needs
  adaptation; no new schema or external-consumer claim is made in this batch.
- Full suites: Main 1172 passed / 21 skipped, Web 110 passed / 31 skipped,
  Scratch 196 passed / 10 skipped. All have zero failures and errors; the
  existing Node/CUDA skip policy is unchanged.
- Scope: manifest, recipe identity tests, and this record. Keep the other
  65 recipes unchanged, especially public/default/fallback, conditional,
  artifact handoff, and special-profile entries. All 6/6 checks passed:
  manifest scope, focused equivalence and fail-closed tests, frozen baseline,
  isolated packaging, installed lookup, and full regressions.
  Next step: human review; no next batch, stage, commit, or push.

## Recipe Identity Migration Phase 1B pilot (2026-11-22)

Base: `code@ec40cc07f1635871f996ff0830ba6e8453d2de4d` (pushed Stage 2).
The frozen experiment contract remains
`c9a3234a03db7311c0dd70e6700ab5ae5971303a`.

- The ordered built-in `recipes` array accepts legacy path strings and
  explicit objects with exactly `id`, `path`, and `profile`. Legacy entries
  retain the existing ID/profile inference. Explicit entries use a lowercase
  canonical ID, a repository-relative YAML path, and one of `smoke`,
  `reproduction`, or `experiment`. Incomplete or invalid objects are rejected,
  never interpreted as legacy entries.
- Both forms share one fail-closed identity index. Public and conditional
  references are still paths. Source resources take precedence, and installed
  fallback still indexes the distribution file list once per build.
  Paper/Web continue to consume the catalog by canonical ID.
- Only `cifar10-symmetric-small-loss-linear-smoke` was converted in place;
  67 entries remain legacy. This internal, nonconditional smoke recipe has
  no Paper association or formal Web binding. Its legacy and explicit
  catalog/CLI results, normalized configuration and seed-only/matrix
  Sweep plans are equal.
- The official catalog API preserves recipe behavior. Tools outside this
  repository that parse the raw manifest and assume every array element is
  a string may need an adapter; their compatibility was not verified.
- Scope: catalog resolver, built-in manifest, identity tests, the existing
  pipeline package-coverage assertion, and this record. No YAML,
  packaging layout, migration baseline, runner, or resume changes.
- Focused: 166 passed / 1 Node skip. The frozen source snapshot and both
  external wheel and sdist-derived installations match the 68 recipes,
  71 YAMLs, 63/4/68/4 views, 26 Papers, 56 ordered associations, 26 defaults,
  and normalized configuration/package fingerprints. Isolated CLI discovery
  and Paper recommendations also match.
- Measured discovery medians in milliseconds (Stage 2 to Phase 1B, excluding
  interpreter startup): source default/full 347.1/364.4 to 318.3/306.3;
  wheel 388.7/405.6 to 392.4/404.6; sdist install 380.5/396.3 to 415.9/422.6.
  Both installed paths still read the distribution file list once per
  build, and all environments parse 68 YAMLs. These are samples, not bounds.
- Full suites: main 1171 passed / 21 skipped, Web 110 passed / 31 skipped,
  Scratch 196 passed / 10 skipped, all with zero failures and errors.
  Existing Node/CUDA skips are not runtime verification of those paths.
- Progress: 6/6 gates complete (dual-format resolver and negative tests;
  single-pilot equivalence and focused consumers; isolated packaging;
  performance; full suites; Git scope and whitespace checks). Five approved
  existing files changed, no files added, no Phase 1B staging, commit or push.
  Next step: human review. Keep the remaining 67 entries on the legacy format
  until an additional batch is approved.

## Recipe Identity Migration Phase 1A (2026-11-21)

Base: `code@7d889b75bb4c7b149404f042cecf928128af4ff4` (pushed Phase 0).
The frozen experiment contract is still `c9a3234a03db7311c0dd70e6700ab5ae5971303a`.
The scope is built-in registry validation, its tests, and this record. Phase 1B
and all source YAML, Paper mapping, packaging configuration, and training
or resume changes are out of scope.

- The old resolver filtered first and tried a parent-directory prefix on a
  repeated stem. A second collision could overwrite an earlier recipe in the
  lookup/Paper dictionary; a dangling conditional reference was also silently
  ignored. Literal duplicate primary paths and unknown public or Paper
  references were already rejected.
- Catalog discovery, direct ID lookup and Paper lookup now use one ordered
  built-in index per call. The index rejects repeated canonical IDs, repeated
  resolved resources, and bad registered files before projection. Public and
  conditional references must be unique and registered. Installed built-in
  lookup rejects multiple matching distribution records instead of selecting
  the first. Conditional still describes catalog availability; index creation
  does not perform data access, readiness checks, artifact loading or training.
- Seven direct production resolver tests passed; they exercise synthetic
  manifest references with real temporary YAML resources. Phase 0 source
  snapshots, CLI input compatibility and formal Web/Quick Start, sweep and
  resume focused checks passed without regenerating the fixture. The 68
  recipes have distinct IDs and resources, so none needs the removed prefix
  fallback. Custom YAML and MentorNet producer lookup are outside the index.
- Windows sample medians for the public catalog, Web recipe menu and CLI
  default list rose from about 23, 24 and 76 ms to 401, 428 and 442 ms, respectively
  (seven repeated invocations, excluding interpreter startup). Public
  discovery reads 68 YAMLs instead of 4. The isolated installed public view
  rose from about 72 ms to 1.59 s in a five-run sample; the installed full view
  was about 1.40 s before and 1.49 s afterward. These are local measurements,
  not CI bounds. No persistent cache was added.
- A pre-existing Web unit-test fixture replaces the global YAML loader with
  an incomplete dummy config for every file. It passes on the old resolver,
  which silently drops those entries, but fails closed on a cold Phase 1A
  catalog. The replacement fixture should stub only the Web config lookup for
  its `fine-formal` scenario. That Web test file is outside the approved
  Phase 1A allowlist; it was not changed. No production bypass is appropriate.
- The unrelated CWD transition-artifact late failure, sdist dev-test/fixture
  distribution, and Node/CUDA validation gaps remain deferred. Review the
  allowlisted diff; do not stage or start Phase 1B with a failing regression.
- Verification: direct resolver 7/7 passed; focused identity/CLI/Web
  binding/Quick Start/sweep/resume 138 passed and 1 Node skip. Full main
  suite: 1163 passed / 21 CUDA skips; Scratch: 196 passed / 10 Node skips;
  Web: 109 passed / 31 Node skips / 1 fixture error as described above.
  The Web test passed on Phase 0 and its original assertions passed on the
  Phase 1A checkout when only the local config lookup was stubbed in memory.
  No such test-file change was made. Direct wheel and the wheel built from
  the actual sdist passed isolated resource and CLI verification, including
  all 71 YAMLs, 68 unique recipes, 63/4/68/4 projections, 26 Papers, 56 ordered
  links, and 26 defaults. All Phase 0 summaries matched.
- Status: 5/6 verification gates complete (resolver negatives, frozen source
  baseline and focused consumers, main, Scratch, external packages);
  the Web full-suite gate is blocked on approval to fix the fixture, followed
  by a Web re-run. No stage, commit, push or Phase 1B work.

## Recipe Identity Migration Phase 0 (2026-11-21)

Baseline: `code@c9a3234a03db7311c0dd70e6700ab5ae5971303a`. Scope is tests,
the readable identity fixture, and this record; production resolution is unchanged.

- `tests/fixtures/recipe_identity_baseline.json` freezes the ordered four discovery
  views (conditional/public-only), 68 recipe records, 26 Paper records with 56
  ordered associations and 26 formal defaults, the 26 Web editing baselines,
  71 package resource mappings (including three packaged-only YAMLs), and two
  representative sweep plans. The old aggregate digests are additional checks,
  not a replacement for the individual records. Do not regenerate the fixture
  to make an unexplained difference pass.
- Authoritative inputs are the existing manifest (membership, visibility and
  conditional flags), runtime Paper catalog (associations, roles and ordering),
  YAML plus runner resolution (executed parameters and dispatch), Web metadata
  (editor bindings), and packaging declaration (installed resource targets).
  IDs and discovery profiles are currently derived from paths. Loss Correction
  intentionally retains a Paper/discovery profile difference. Normalized config
  hashes and sweep/member hashes are observations, not registry fields.
- `python -m unittest tests.test_recipe_identity -v` compares the source
  contract. For package verification, start an isolated Python from an external
  empty working directory, explicitly prepend only the wheel's installed
  site-packages path, and execute this test module by its file path with
  `--verify-installed PREFIX WHEEL`. Verify both a direct wheel and a wheel
  built from the actual sdist, using the existing setuptools backend. The
  verifier checks import and actual mapped resource locations before comparing
  against the source fixture. Run the development suites from a checkout:
  setuptools includes the new test module in the sdist by default, but not
  the checkout-only JSON fixture. This does not change installed runtime assets
  or the existing CI contract, which installs the sdist and exercises its CLI.
- The candidate gate in tests rejects duplicate paths and canonical collisions
  (including collisions after parent-prefix fallback), dangling normal,
  public or conditional references, broken Paper keys and cross-view identity
  drift. This is **not** a production guarantee: the current resolver can
  overwrite a repeated final ID when building its index and can silently
  ignore a dangling conditional entry. Duplicate declared normal paths and
  unknown public paths are already rejected by production; Paper lookup also
  rejects a dangling recipe key.
- Existing runtime gap, not an identity migration: take the built-in CWD
  reproduction config in memory, set `cwd.variant` to `multiclass`, and omit
  `cwd.transition_artifact`. Static preflight can pass and the multiclass
  requirements do not advertise that input. The runner subsequently indexes
  the field and loads the VolMin artifact after data preparation. No runner,
  artifact, schema, or checkpoint contract was changed to cover this case.
- Phase 1 should preserve the public identities with dual-format reads; single
  global identity resolution must fail closed before new entries can route
  ambiguously. Keep history-based resume independent of catalog availability.
  Environment-dependent checks (browser/Node, accelerator-specific execution
  and real external artifacts) are not proved by matching the fixture alone.
- Verification (6/6: snapshot, candidate negatives, Web/Quick Start,
  resume/sweep, isolated packages, full regression): focused identity and
  integration run 26 passed / 1 Node skip; `python -m unittest discover -s
  tests -v` 1156 passed / 21 CUDA skips; `python -m unittest discover -s web
  -p "test_*.py" -v` 110 passed / 31 Node skips; `python -m unittest discover -s
  src/lnl_toolbox/scratch/tests -v` 196 passed / 10 Node skips. No failures or
  errors. Source, direct-wheel, and sdist-built installs matched all identity
  records, all four CLI views, Paper recommendations, and the prior digests.
  Both installed wheels retained the baseline member contents (excluding
  installation bookkeeping). No Phase 0 stage, commit, or push; next step is
  human diff review and separate Phase 1 approval, not regeneration of the
  fixture or remediation of unrelated runtime gaps.

## CI input-validation and resume repair (2026-10-15)

- Task: resolve the failures in quality run `38672394362`, without freezing configurable experiment values to historical defaults. Branch/base: `scratch` / `8f673918f2246864022a6db093ae2ac27216665e`. No branch change, commit or push is authorized.
- Approved modification allowlist: `src/lnl_toolbox/catalog.py`, `src/lnl_toolbox/training/cwd_experiment.py`, `configs/experiment/mc_ldce_cifar10_smoke.yaml`, `tests/test_cli.py`, `tests/test_registry.py`, `tests/test_cwd.py`, `tests/test_training.py`, and this progress document. No new files or dependencies.
- Checklist: (1) repair model-family/nested input validation; (2) replace fixed-default assertions with supported-custom-configuration checks and correct CE fixture facts; (3) migrate the existing MC-LDCE short fixture to the implemented lifecycle; (4) preserve CWD loader RNG state across epoch-boundary resume; (5) run and close focused, full-unit, Web, Scratch and lint regressions. Completed: 5 / 5 (100%); the approved second-stage repairs and final local regressions are complete. Earlier failed runs below are diagnostic history, not the final status.
- Input validation: model variants are checked against the selected model family at any stage nesting depth; model initialization options, risk-correction names and LEND dilution policy/tolerance/iteration limits are checked before normalization. Supported choices are implementation domains, not prescribed experiment defaults.
- Configurable controls: LEND, UPM and CNLCU checks accept modified model/budget/optimizer/scheduler settings and verify that validation does not rewrite source configurations. CE compatibility uses explicitly declared clean validation data; its runtime cleanliness requirement was not disabled.
- CWD diagnosis: `PreparedData.loader()` uses an independent Torch generator. Global checkpoint RNG restoration alone resets each loader's sample-order state after restart. Checkpoints now retain and restore the train, snapshot and test loader generators. A strengthened multi-batch test failed before this repair, then passed with zero tolerance on model tensors, exact metric equality and exact loader-state equality. Older incomplete checkpoints without loader RNG states are rejected when more epochs remain, rather than silently promising identical continuation.
- MC-LDCE: only the old synthetic smoke YAML is moved from lifecycle 2/fixed features to the existing lifecycle 3/joint-feature training. Formal configurations and algorithm equations are unchanged. The fixture completes two real optimization epochs and checkpoint loading.
- Focused command (passed, 31 tests): `python -m unittest test_cli.ConfigParameterGateTest test_cli._lend_cli_LENDCliTest test_cli._upm_cli_UPMCliTest test_cli._unified_cli_UnifiedCliTest.test_plain_ce_validate_check_data_reports_compatible test_registry._cnlcu_readiness_CNLCUReadinessTest test_registry._compatibility_RunnerRequirementsTest.test_cal_mc_ldce_and_ca2c_preserve_their_synthetic_smokes test_cwd._cwd_training_CWDTrainingTest.test_interrupted_fold_resumes_same_plan_without_validation_metrics test_training._new_paper_training_NewPaperTrainingTest.test_all_new_workflows_smoke_and_resume -v` (`PYTHONPATH=src;.;tests`).
- Additional focused command (passed after the legacy-state guard): `python -m unittest test_cwd._cwd_training_CWDTrainingTest.test_interrupted_fold_resumes_same_plan_without_validation_metrics -v`. Scoped Ruff and `git diff --check` passed.
- Files inspected: the allowlist plus model factories, LEND/UPM/CNLCU parsers, shared data-loader/checkpoint/snapshot contracts, CLI planning and the CI workflow. Collaborator warning: the worktree already contained 28 modified files, including `catalog.py`, `test_registry.py` and `test_training.py`; these existing changes are preserved. Other pre-existing Web/model/algorithm/recipe work is outside this task.
- Limitations: local verification is Windows with its existing Python/Torch installation, not the hosted Linux/Python 3.12 runner. CI has not been rerun; no Web training acceptance is claimed from unit tests.
- Regression commands (`PYTHONPATH=src;.;tests`, existing local Python runtime):
  - `python -m unittest discover -s tests -q`: 1179 tests in 478.884s; one error remains. The all-built-in-config check previously stopped on Loss Correction; it now reaches `pcse-multiclass-volmin-smoke`, whose old `transition_stage.parameterization` contains unsupported `initial_flip_mass`, `max_flip_mass` and `temperature`. A separate read-only validation of all 63 non-conditional built-in recipes identified exactly this one invalid recipe. Its YAML is outside the approved allowlist.
  - `python -m unittest discover -s web -p 'test_*.py' -q`: 163 tests in 422.226s; six failure records: parameter-order subtests for Co-teaching/CNLCU/PDL, Loss Correction presentation expectation, noise-prior derivation expectation, and CDR permission expectation. No Web file was modified by this task.
  - `python -m unittest discover -s src/lnl_toolbox/scratch/tests -q`: 249 tests in 15.470s; three failures and one error: synthetic fixture incompatible with classwise validation splitting, cross-config Co-teaching model assertion, pure-helper import isolation assertions for LEND/CDR, and Scratch CNN8 missing the classifier BatchNorm1d layer present in the shared CNN8. No Scratch file was modified by this task.
  - `python -m ruff check src tests web`: passed. `git diff --check`: passed (line-ending warnings only).
- Final scope review for this stage: all eight approved files were changed, no files were created, and the 28 pre-existing dirty paths were preserved. High-conflict shared-file changes are restricted to the input-gate section of `catalog.py`, the CNLCU test in `test_registry.py`, the new-workflow test in `test_training.py`, and this new progress section.
- The first-stage failures above are retained as diagnostic history. The user subsequently approved a second modification allowlist: `configs/experiment/pcse_multiclass_volmin_smoke.yaml`, `web/command_console.py`, `web/lnl_parameter_metadata_registry_revised.yaml`, `web/test_command_console.py`, `src/lnl_toolbox/scratch/data_runtime.py`, `src/lnl_toolbox/scratch/blocks/models.py`, `src/lnl_toolbox/scratch/tests/test_formula_templates.py`, and `src/lnl_toolbox/scratch/tests/test_runtime_isolation.py`. No new files or dependencies were approved or added.
- Second-stage file changes:
  - `configs/experiment/pcse_multiclass_volmin_smoke.yaml`: migrate obsolete transition parameterization to the existing sigmoid-off-diagonal implementation, preserving the initial transition matrix (diagonal 0.95, off-diagonal 0.025). Formal PCSE defaults are unchanged.
  - `tests/test_training.py`: add direct validation, real pipeline training and completed-checkpoint resume of the unmodified registered PCSE short fixture; the test does not replace its parameterization.
  - `web/lnl_parameter_metadata_registry_revised.yaml`: remove the stale hidden-path entry for the editable Loss Correction risk corrector, matching the existing method-aware presentation policy.
  - `web/test_command_console.py`: check ordering of explicitly ordered controls without forbidding additional supported controls; resolve method-aware presentation precedence; test known-matrix warmup visibility, mode-dependent CDR permissions, and independence of dataset noise and the editable method noise prior. Wrong derived values remain rejected.
  - `src/lnl_toolbox/scratch/blocks/models.py`: align Scratch CNN8/CNN9 topology, BatchNorm, dropout and initialization with the existing shared implementations; preserve the pre-existing model-name migration.
  - `src/lnl_toolbox/scratch/data_runtime.py`: construct bounded balanced synthetic fixtures and complete class groups for classwise splitting, including splits configured after source loading. Real-data splitting remains fail-closed and unchanged.
  - `src/lnl_toolbox/scratch/tests/test_formula_templates.py`: test cross-peer recipe wiring with customizable models/budgets, exact shared/Scratch model tensors and outputs, and early/late classwise fixture splitting with original plans unchanged.
  - `src/lnl_toolbox/scratch/tests/test_runtime_isolation.py`: explicitly allow only the existing LEND/CDR pure-math helper imports; retain rejection of other algorithm/trainer imports.
  - `web/command_console.py`: approved for inspection/possible repair but no edits made by this task; existing presentation/permission behavior was correct for the failed assertions.
- Second-stage regression: `python -m unittest discover -s tests -q` passed all 1180 tests in 852.422s; `python -m unittest discover -s src/lnl_toolbox/scratch/tests -q` passed all 252 tests in 58.189s; the final fresh `python -m unittest discover -s web -p 'test_*.py' -q` passed all 164 tests in 797.529s. `python -m ruff check src tests web` and `git diff --check` passed (line-ending warnings only). The 24-test focused batch and a previously started full Web process initially had only the stale Loss Correction warmup-group assertion remaining; after changing it to check actual known-matrix visibility, its focused recheck passed (one comprehensive all-paper test, 55.436s), followed by the fully green fresh Web run above.
- Execution environment: Windows, existing `F:/Miniconda/envs/pytorch/python.exe` (Python 3.11.15); `PYTHONIOENCODING=utf-8`, `PYTHONPATH=src;.;tests` for unit tests, `src;.` for discovered Web/Scratch tests and Ruff, and `src;.;web` for the focused Web recheck. Full-suite stdout was redirected to `$null`; unittest failure/summary stderr remained visible. No new runtime or dependencies were installed.
- Scope: this task has modified 15 of the 16 approved paths, with no added files. The other 28 initially dirty paths/overlapping prior edits remain preserved. Shared catalog, Web metadata/tests, Scratch model factory and progress-document changes need joint diff review by collaborators.
- Final scope/diff: 15 approved paths modified by this task; the sixteenth approved path (`web/command_console.py`) retains only pre-existing changes. No new or unplanned files were produced, no staging/checkpoint commits/Git operations altering state were performed, and the original 28 dirty paths were preserved. Final `git diff --check` passed. Formal reproduction YAMLs, shared training algorithms/data adapters, CI workflow and dependencies were intentionally left unchanged by this task.
- Known limitations: hosted Linux/Python 3.12 CI was not rerun, and local regression success does not claim 26-paper browser training acceptance. Legacy incomplete CWD checkpoints without loader RNG state cannot provide exact continuation and now report that limitation explicitly.
- Exact next step / integration: review the approved 15-file incremental changes alongside the pre-existing model-name migration; use the commands above for local reproduction; request separate approval before staging, committing or publishing. History cleanup is not needed (no task commits); local implementation is ready for review, but publication remains unapproved and hosted CI is unverified.

## Quick Start method–dataset portability closeout (2026-08-28)

- Current task: make Quick Start select datasets from the same `MethodRequirements × DatasetCapabilities` contract used by real preflight, while preserving formal reproduction recipes and paper training stages.
- Branch/base: `new-cli` at `62892f5`; no branch creation, commit, or push was authorized. The pre-existing `training/runners.py`/`tests/test_jocor.py` working-tree changes and five unrelated untracked files were preserved.
- Checklist: remove the private class-space gate; bind generic class-dependent config; publish noisy/artifact requirements; prove READY→plan/preflight; run real-data/focused/full acceptance. Completed: 5 / 5 (100%).
- Public behavior: formal parity still uses the exact recipe. Generic Quick Start binds the selected dataset's class count, discards stale transition matrices, and delegates all READY/INCOMPATIBLE/NEEDS_INPUT decisions to `ExperimentService` and the runner requirement provider. Importance Reweighting no longer rejects a binary dataset by name; missing binary rates remain explicit user inputs. Noisy-only methods require noisy labels, T-Revision rejects incompatible non-class-dependent noise, and external PCSE retains its exact implemented-variant boundary.
- Input binding: existing CIFAR ResNet/PreActResNet families now consume `input_channels` from `PreparedData.input_spec`, defaulting to three channels. This removes the hidden Fashion-MNIST failure without changing CIFAR formal behavior or any paper objective.
- Real compatibility matrix: CIFAR-10, CIFAR-100 and Fashion-MNIST were checked with clean and symmetric-0.2 selections across all 26 papers. Every menu item marked READY produced a READY plan; other entries remained typed INCOMPATIBLE or NEEDS_INPUT. Real `supervised → GCE → ResNet-34` runs completed one optimization epoch on all three datasets (global steps 1, 2 and 1 respectively) using temporary artifacts.
- Files inspected: Quick Start templates/service/models, runner registry, ExperimentService, ResNet builders, Importance Reweighting config, local dataset catalog, affected tests, and the shared data-flow/progress documents.
- Files modified: `src/lnl_toolbox/quickstart/templates.py`, `src/lnl_toolbox/quickstart/service.py`, `src/lnl_toolbox/training/runners.py`, `src/lnl_toolbox/training/service.py`, `src/lnl_toolbox/training/experiment.py`, `src/lnl_toolbox/models/cifar_resnet.py`, `src/lnl_toolbox/algorithms/importance_reweighting/config.py`, `tests/test_quickstart_service.py`, `tests/test_registry.py`, `tests/test_training.py`, `tests/test_cli.py`, `tests/test_importance_reweighting.py`, `web/test_quick_start_integration.py`, `docs/data-flow-guide.md`, and this progress record. Files added: none.
- Tests executed: Quick Start `13/13`; registry `53/53`; Web integration `2/2`; training `75/75`; CLI `89/89`; Importance Reweighting `50/50`; nine affected method suites `257/257`; full unittest `969/969`; Python compile and `git diff --check` passed.
- Blockers: none. Assumption: shortened real runs validate data portability and first-stage execution, not paper numerical reproduction.
- Local checkpoint commits: none. History cleanup: not needed. Push readiness: implementation is reviewable, but commit and publication remain unapproved.
- Collaborator integration: `training/runners.py`, `training/service.py`, `training/experiment.py`, Quick Start files and both shared documents are high-conflict. `tests/test_jocor.py` is pre-existing collaborator work and was not modified by this task.
- Exact next step: human review of the scoped diff; request separate authorization before staging or committing.

## Five startup-boundary fixes (2026-08-25)

- Current task: repair only the five confirmed startup/data-role defects reported by the Web formal-run path.
- Branch/base: `new-cli` at `3a7582d`; no commit or push performed.
- Completed: 5 / 5 code fixes and focused regression checks.
- JoCoR now derives `validation_size` from its data configuration before choosing the validation loader. PDL now derives `official_pdl` from the configured correction/phase variant before loading revision artifacts. The paper algorithms are unchanged.
- CWD's folded CIFAR adapter is correctly treated as a train/test-only source during inspection; the absence of a native validation split no longer aborts preflight.
- MC-LDCE and CA2C requirements and runners no longer request or materialize a validation loader when `validation_size` is zero. They retain validation metrics when an explicit validation role exists.
- MentorNet remains fail-closed: a missing or hash-mismatched MentorArtifact is an external resource readiness failure, not a reason to bypass validation.
- Validation: Ruff passed; data adapters `54/54`; registry `52/52`; training `74/74` with one CUDA skip; CWD data inspection against the existing F-drive CIFAR source returned `ready`; zero-validation role inspection showed no validation role for MC-LDCE/CA2C. Full unittest ran `957` tests: `954` passed, `1` skipped, and the same two baseline failures remain (95-vs-94 YAML count and the pre-existing NoiseManifest error-text mismatch).
- Exact next step: run the formal Web paths for JoCoR, PDL, CWD, MC-LDCE and CA2C with available data/resources, then separately prepare the pinned MentorArtifact/PCSE resources for the remaining external-resource cases.

## Unified data connector final semantic closeout (2026-08-25)

- Current task: close the last validation/TEST and compatibility-origin semantics without changing DataService or paper algorithms.
- Branch/base: `new-cli` at `3a7582d`; no branch creation, commit, or push is authorized.
- Checklist: FINE role declaration; CAL role declaration; train-only progress; final-only TEST; compatibility origin table; focused/full acceptance. Completed: 5 / 6 (83%); code regression is complete, but the all-paper formal-runtime acceptance is not.
- Result: FINE/CAL request validation only when `data.validation_size` or `data.num_val` is positive. Their runners never alias TEST as validation. Epoch rows are train-only or use a real independent validation role; TEST is evaluated once after training and appears only in one final event with leakage flags.
- Compatibility: class-count limits for CWD, Importance Reweighting, PCSE, VolMinNet and unresolved MC-LDCE are reported as `implemented_variant_limit`; FINE/DLD modality and MentorArtifact interface limits use the same origin. Binary Risk and L2RW trusted supervision remain algorithm requirements.
- Files added: none. DataService, algorithms, YAML, CLI/Web/Scratch, taxonomy/guideline, data sources and the four pre-existing untracked files remain unchanged.
- Focused tests: FINE `14/14`; CAL `13/13`; registry `52/52`; training `74/74` with one CUDA skip. Both validation modes, resume final-row replacement and train-only/train+validation SVGs are covered. Ruff and `git diff --check` pass.
- Full regression: `957` tests executed; `954` passed, `1` skipped, and only the two already-recorded baseline assertions failed (95 vs 94 YAML count; manifest namespace error text vs `length`).
- Formal preflight: `24/26` formal recipes are compatible after injecting the existing MentorArtifact. CWD is blocked before training because the folded CIFAR adapter's validation probe raises `folded CIFAR binary view exposes train and test`; PCSE is blocked because no discovered UPM run matches its pinned checkpoint SHA-256. These are pre-existing acceptance-resource/data-adapter blockers outside this task's allowlist. Formal FINE/CAL also cannot be reduced to one stage solely through their current public `trainer.epochs` budget because their fixed warm-up budgets would then exceed the total budget; their unchanged formal protocols were therefore not rewritten for this check.
- Exact next step: separately approve the CWD validation-probe fix and provide the pinned PCSE source run (or approve a formal acceptance protocol that records stage-specific budget overrides). Then run all 26 first real stages and compatible alternate-data cases before marking 6/6.
- Ready for history cleanup/push: no; neither commit nor publication is authorized.

## 26-paper unified data boundary (2026-08-25)

- Current task: separate method data requirements from reproduction protocols, then migrate 26 paper methods without changing their training engines.
- Branch/base: `new-cli` at `3b1675b`; four pre-existing untracked audit/WebUI files are intentionally preserved.
- Checklist: repository/audit baseline; neutral PreparedData contracts; shared-runner migration; specialized role/view migration; remaining dedicated-runner migration; 26-paper formal/alternate-data acceptance.
- Completed: 4 / 6 (67%); baseline audit、backward-compatible contract foundation、shared-runner migration 和 specialized role/view migration 已完成；remaining dedicated-runner migration 已完成 Phase 5A，但该 checklist 项尚未整体完成。
- Contract result: `InputSpec`, `NoiseDescriptor`, `DataProtocol`, `UNLABELED` and `CURRICULUM` are public vocabulary. `PreparedData` exposes input/noise/available-role contracts and fail-closed validation selection. Explicit protocols participate in the data fingerprint; legacy callers retain their prior manifest fingerprint shape.
- Files inspected: data contracts/profile/service, reproduction compatibility layer, all dedicated experiment entry points, runner compatibility requirements, data tests, shared data-flow/file-map documents, taxonomy coverage, and the 26-method data-requirements audit.
- Files modified: phase 1 changed `src/lnl_toolbox/data/contracts.py`, `src/lnl_toolbox/data/__init__.py`, `src/lnl_toolbox/training/data_service.py`, `tests/test_data_adapters.py`; phase 2 changed `src/lnl_toolbox/training/runners.py`, `src/lnl_toolbox/training/experiment.py`, `tests/test_registry.py`, `tests/test_training.py`; phase 3 changed `src/lnl_toolbox/data/views.py`, the data export/service, and the data-entry sections of `dld_experiment.py`, `dividemix_experiment.py`, `l2rw_experiment.py`, with contract coverage in `tests/test_data_adapters.py`; phase 5A changed capability declarations, PCSE train-mode data construction, VolMinNet class-count parsing, and their focused tests. All phases update `docs/data-flow-guide.md`, `docs/file-map.md`, and this progress record. Files added: none.
- Shared-runner result: CE/GCE/APL consume only TRAIN, the selected validation role and TEST. GCE/APL accept compatible IMAGE/TABULAR data, native observed-only noise does not require clean labels or a private manifest, while formal generated-noise recipes retain manifest identity. `feature_mlp` consumes tabular `input_spec.feature_dim`; loss/selector/optimizer/epoch loop remain unchanged.
- Specialized role/view result: `PreparedData.view_loader()` projects one named view while preserving target/index, and `subset_loader()` constructs validated dynamic training partitions. DLD now obtains aligned weak/strong loaders from the service; DivideMix obtains per-epoch labeled/unlabeled loaders from the service; L2RW selects its validation loader through the declared requirements. Their feature extraction, diffusion, GMM, MixMatch, meta gradient and checkpoint state are unchanged. Dynamic DivideMix membership deliberately remains a subset operation rather than a static UNLABELED role.
- Tests: focused data contracts pass 57/57; registry passes 46/46; training passes 71/71 with 1 CUDA skip. The added generic tabular GCE and APL cases each completed a real CPU epoch with finite test loss. The phase 2 full unittest ran 942 tests with 2 failures, 2 errors and 1 skip; all four failures are the same pre-existing failures observed before phase 2: stale YAML count, stale noise error text, and two PCSE validation-index/manifest mismatches. Official-format image and tabular fixtures each completed a real one-epoch path in phase 1. Ruff and `git diff --check` pass.
- Blockers: none for the contract foundation. The four pre-existing full-suite failures must remain visible and are outside this phase's allowlist.
- Phase 3 tests: data adapters/service pass 53/53; DLD passes 27/27; DivideMix passes 30/30; L2RW passes 16/16. Ruff and `git diff --check` pass. Full unittest reran 942 tests with the unchanged 2 failures, 2 errors and 1 skip: stale YAML count, stale noise error text, and two PCSE validation-index/manifest mismatches. No phase 3 regression was found.
- Phase 5A result: CDR、DSS、CA2C 和 VolMinNet 的能力声明不再包含无论文依据的 IMAGE/CIFAR/10-or-100-class 限制；PCSE train mode 从 `PreparedData` 获取 dataset、类别数和 feature dimension，external UPM reproduction 路径保持不变。CDR/DSS 在非内置名称的四分类 tabular adapter 上完成真实 CPU epoch；CA2C generic tabular smoke 继续通过；VolMinNet 在四分类 tabular 数据上完成完整 epoch；PCSE generic adapter 完成第一个真实预训练 epoch。算法公式、正式 YAML、CLI、Web、CAL、FINE、MentorNet 和 MC-LDCE `C>=3` 未修改。
- Phase 5A tests: registry 46/46；training 72/72 with 1 CIFAR-data skip；VolMinNet/estimators 49/49；PCSE generic first-stage 1/1；Ruff 通过。完整 unittest 运行 945 项，结果仍为实施前相同的 2 failures、2 errors、1 skip：stale YAML count、stale noise error text 和两项 PCSE independent-validation index/manifest mismatch；没有新增失败。
- Exact next step: propose the next separately approved dedicated-runner batch. Do not mark item 5 complete until all remaining runners are migrated.
- Ready for history cleanup: no; the 26-paper migration is in progress. Ready to push: no.
- Collaborator note: contracts, DataService, runner registry and shared documents are high-conflict surfaces. Phase 3 changes shared `data_service.py` and only the data-entry sections of `dld_experiment.py`, `dividemix_experiment.py`, and `l2rw_experiment.py`. Phase 5A changes only CA2C's capability declaration, not its algorithm/runner; FINE/UPM and all loss, selector, recipe, CLI and Web files remain unchanged.

## Web configuration workflow and semantic Sweep (2026-08-22)

- Current task: carry one recipe/project-YAML context through paper guidance, YAML editing, execution, parameter Sweep, and result management.
- Branch: `codex/cli`; base commit: `7a8c9be`; the committed beginner-tutorial baseline was preserved.
- Checklist: workflow audit; project-YAML Sweep API; Registry-enforced matrix validation; cross-module UI links; focused/browser acceptance; full regression and documentation.
- Completed: 6 / 6 (100%); implementation, focused tests, real browser workflow, runtime Sweep evidence, documentation, full regression, and scope audit are complete.
- Behavior: paper recipes can be prepared for formal execution, copied into a safe project YAML, or passed directly to Sweep. Saved project YAMLs can validate, prepare a run, enter Sweep, and pass the Sweep output to run management without losing source context.
- Sweep semantics: both recipe and project-YAML sources use the revised four-level Registry. Locked/unknown/type-incompatible paths fail server-side; paper parameters show provenance and reproduction impact and are reported as deviations in every affected planned run.
- Safety correction: Web reproduction metadata is stored under `meta.web_parameter_record`. The training core's top-level `parameter_record` remains reserved for hyperparameter-sampling records, preventing Web-saved Binary Risk configurations from failing in `ParameterRecord.from_dict()`.
- Runtime evidence: a project-YAML batch/LR 2×2 Sweep completed 4/4 runs; after the metadata correction, a Binary Risk 1×2 paper-parameter Sweep completed 2/2 runs and reported `risk.rho_positive` as the paper-deviation path.
- Validation: Web 39/39 and full unittest 845/845 passed; Ruff, embedded JavaScript syntax, and `git diff --check` passed.
- Files changed: Web backend/page/test and four approved shared documents. No CLI, Sweep engine, recipe, Registry, algorithm, runner, model, dataset adapter, or training default changed.
- Exact next step: human diff review; commit only after separate authorization.
- Local checkpoint commits: none; commit and push are not authorized.
- Collaborator note: Web files and shared documents are high-conflict integration surfaces; changes are scoped to configuration context, Sweep authorization, and documentation.

## Web beginner tutorial workflow (2026-08-22)

- Current task: implement `docs/LNL-Toolbox-简明操作教程.md` as an executable Web workflow rather than a single action dropdown.
- Branch: `codex/cli`; base commit: `f99ed98`; the worktree was clean at task start.
- Checklist: tutorial audit; versioned six-step contract; guided UI and quick-mode compatibility; focused tests; real browser execution; full regression and documentation.
- Completed: 7 / 7 (100%); implementation, focused tests, JavaScript/Ruff checks, real browser execution, documentation, full regression, and final scope review are complete.
- Behavior: the default page advances only after successful `doctor → list → validate → dry-run → run`; the final step inspects resolved config, metrics, files, phase, epoch target and `last.pt` before enabling resume. A completed target is marked `not-needed` instead of launching a no-op resume.
- Runtime evidence: the real Web page used registered `local-cifar10`, ran `cifar10-symmetric-ce-smoke` for its configured 2 epochs, and wrote `artifacts/runs/web-tutorial-acceptance`. Final evidence was epoch `2/2`, validation accuracy `0.16796875`, test accuracy `0.16015625`, complete config/metric/checkpoint files, and no browser errors.
- Files changed: Web backend/page/test, Web README, concise tutorial, file map, and this progress record. No CLI, runner, algorithm, data service, recipe, or parameter registry changed.
- Validation: Web 34/34 and full unittest 845/845 passed; Ruff, JavaScript syntax, and `git diff --check` passed. Browser acceptance completed all six tutorial steps and correctly treated resume as unnecessary after reaching epoch `2/2`.
- Exact next step: human diff review; commit only after separate authorization.
- Local checkpoint commits: none; commit and push are not authorized.
- Collaborator note: Web files and shared documents are high-conflict integration surfaces; edits are limited to the beginner tutorial contract and presentation.

## Revised Web parameter permission policy (2026-08-21)

- Current task: make `lnl_parameter_metadata_registry_revised.yaml` the only active Web parameter-selection policy without changing recipes, training code, or the page layout.
- Branch: `codex/cli`; base commit: `840ace1`; the revised registry was the only untracked file at task start.
- Checklist: compare registries; switch authority; test editable/locked boundaries; focused regression; full regression and documentation.
- Completed: 6 / 6 (100%); registry comparison, authority switch, boundary tests, focused/full regression, browser acceptance, and documentation are complete.
- Registry scope: all 26 methods and all 1,231 existing paths are retained. The revised levels contain 120 basic, 137 paper, 809 advanced, and 165 locked parameters.
- Safety boundary: supported experimental choices are editable under the revised policy, while structural/runtime/integrity fields such as `execution.runner` remain server-enforced locks.
- Files changed: revised registry, Web backend/test, and three shared documents. No page layout, recipe, algorithm, runner, model, dataset adapter, or training default changed.
- Validation: Web 32/32 and full unittest 845/845 passed; Ruff passed. Browser acceptance confirmed Binary Risk exposes 6 basic, 6 paper, 6 advanced, and 4 locked fields; `model.name` and both noise rates are editable while `execution.runner` remains read-only.
- Exact next step: human diff review; commit only after separate authorization.
- Local checkpoint commits: none; commit and push are not authorized.
- Collaborator note: `web/command_console.py`, Web tests, and shared documents are high-conflict integration surfaces; edits are limited to the registry authority and permission assertions.

## Web parameter metadata registry (2026-08-21)

- Current task: make the 26-paper parameter registry the Web authorization, grouping, provenance, and reproduction-deviation source.
- Branch: `codex/cli`; base commit: `4ace09c`; the supplied registry was the only untracked file at task start.
- Checklist: registry audit; path/binding repair; backend schema and enforcement; four-level UI; focused tests; full regression and documentation.
- Completed: 7 / 7 (100%); implementation, focused Web tests, full regression, lint, JavaScript syntax, and browser acceptance are complete.
- Registry audit: 26 methods and 99 paper parameters were retained; stale T-Revision, DivideMix, split, scheduler, model-list, PCSE, DSS and DLD paths were repaired and each method is bound to its catalog formal recipe.
- Safety: locked values are visible but cannot be changed by parameter patches or complete-YAML API requests. Paper-level changes require acknowledgement and persist `modified_from_paper`, the formal recipe, and changed-from/value evidence in `meta.web_parameter_record`; the top-level `parameter_record` remains reserved for training hyperparameter sampling.
- Files changed: the supplied registry, Web backend/page/test, and three shared documents. No recipe, algorithm, runner, model, dataset adapter, catalog, or training default changed.
- Validation: Web 31/31 and full unittest 845/845 passed; Ruff, JavaScript syntax, and `git diff --check` passed. Browser acceptance confirmed four groups, read-only locked fields, paper provenance, and the paper-deviation warning.
- Exact next step: human diff review; commit only after separate authorization.
- Local checkpoint commits: none; commit and push are not authorized.
- Collaborator note: Web files and shared documents are high-conflict integration surfaces; changes are limited to parameter metadata presentation and authorization.

## Web matrix sweep and result visualization (2026-08-21)

- Current task: expose typed Cartesian parameter sweeps, native Windows path selection, structured CLI tables, and run metrics/curves in the local Web console.
- Branch: `codex/cli`; base: clean working tree at task start.
- Checklist: Sweep CLI; result contract; Web APIs; Web controls; focused tests; full regression and documentation.
- Completed: 6 / 6 (100%); implementation, browser acceptance, focused tests, documentation, and final regression are complete.
- Behavior: `--matrix PATH=JSON_ARRAY` preserves types and defaults to the recipe seed; Web preflights and lists every combination before execution. Run management reads complete and partial artifacts, uses Sweep manifests for authoritative status, and overlays up to five curves by default.
- Follow-up UI: long run lists can be filtered by name, method, status, or seed and collapsed without clearing selected curves. Native path dialogs fall back to the project root when the current value is empty, relative, or invalid.
- Paper guidance follow-up: result lists are compare-only; the paper page now explains the method lifecycle and its mapping to YAML fields and implementation modules, with a direct YAML editor shortcut instead of exposing profile/variant identifiers.
- Resume inspection follow-up: Web resume now requires a read-only inspection of the selected run/checkpoint and shows resolved configuration, file/checkpoint inventory, current epoch/phase, stage schedule, best metrics, and explicit readiness blockers before enabling the command.
- Browser evidence: the real page generated the two-dimensional batch/LR command, the backend planned exactly four runs, the run manager loaded 144 existing run directories, and structured Smoke recipes rendered as a table.
- Files changed: Sweep/results contracts, unified CLI, Web backend/page, three focused tests, and three shared documents. No algorithm, runner, model, adapter, YAML, or training default changed.
- Validation: Sweep 7/7, Web 26/26, unified CLI 43/43, and full unittest 845/845 passed. Ruff, JavaScript syntax, `git diff --check`, and a no-training four-combination Sweep dry-run passed.
- Exact next step: human diff review; commit only after separate authorization.
- Local checkpoint commits: none; commit and push are not authorized.
- Collaborator note: `cli/main.py`, Web files, and shared documents are high-conflict integration surfaces; edits are scoped to CLI/Web presentation and result inspection.

## Curated public recipe surface (2026-08-21)

- Current task: keep all reproducibility YAMLs while exposing only useful templates to ordinary users.
- Branch: `codex/cli`; base commit: `6ed461f` with the approved schema-v1 working tree preserved.
- Checklist: public manifest; catalog/CLI filtering; Web filtering; tests; docs; final regression.
- Completed: 6 / 6 (100%); implementation and validation are complete.
- Public behavior: `lnl list experiments` and the Web beginner page expose four curated templates. `--all`, direct recipe lookup, the paper catalog, and `/recipe` retain access to the complete catalog.
- Configuration safety: no experiment YAML, runner, algorithm, dataset adapter, or training default was changed.
- Validation: unified CLI 38/38, Web 21/21, and full unittest 833/833 passed. Ruff, Web JavaScript syntax, and `git diff --check` passed.
- Runtime evidence: the public `cifar10-clean-smoke` template completed one Fashion-MNIST GPU epoch through the registered alias. Browser inspection showed exactly four beginner templates, while `/recipe` retained all 64 configuration sources.
- Exact next step: human diff review; commit only after separate authorization.
- Local checkpoint commits: none; no commit or push is authorized.
- Collaborator note: the recipe manifest, `catalog.py`, CLI and Web files are shared integration surfaces.

## Versioned YAML contract migration (2026-08-21)

- Current task: audit every YAML parameter category, preserve the old files, and make active recipes portable and directly runnable.
- Branch: `codex/cli`; base commit: `6ed461f`.
- Checklist: archive; schema/normalization; 92-file migration; Binary Risk/MentorNet adaptation; tests/docs; final regression.
- Completed: 6 / 6 (100%); implementation and validation are complete.
- Configuration result: 92 / 92 active YAMLs use schema v1; 76 are complete experiment/auxiliary configurations and 16 are explicit fragments; all 64 built-in recipes remain internally discoverable and four curated templates are public by default.
- Data result: machine-local CIFAR/MNIST roots are removed from active recipes and resolved through a unique local registration; the UCI Heart engineering recipe retains its single-file fallback path required by its strict source contract.
- Legacy recovery: `archive/configs-legacy-2026-08-21/manifest.json` records 92 matching SHA-256 snapshots.
- Tests executed: config schema 4/4, Binary Risk 5/5, MentorNet 9/9, DataService 12/12, ExperimentService 3/3, runner planning 7/7, unified CLI 37/37, and the full unittest suite 832/832. Ruff and `git diff --check` passed.
- Runtime evidence: rootless `cifar10-clean-smoke` resolved the unique `local-cifar10` registration, loaded 50,000 train / 10,000 test samples, and completed one GPU epoch. Its persisted resolved configuration remains canonical and does not contain runtime-only aliases.
- Exact next step: human diff review; commit only after separate authorization.
- Local checkpoint commits: none; no commit or push is authorized.
- Collaborator note: `catalog.py`, `training/service.py`, `training/data_service.py`, all active YAMLs, and shared documents are high-conflict files.

## Dataset-independent training selection (2026-08-21)

- Current task: remove the Web/CLI assumption that every non-tabular dataset verifies through a CIFAR recipe.
- Branch: `codex/cli`; base commit: `2f3ee42` with pre-existing CI/fixture working-tree edits preserved.
- Checklist: automatic verification profile; CLI compatibility; separate Web training-template/data selectors; real Fashion-MNIST verification; switched-template training; tests and documentation.
- Completed: 6 / 6 (100%). Implementation, real-data validation, browser validation, and final regression are complete.
- Real evidence: registered Fashion-MNIST loaded 60,000 train / 10,000 test samples and completed one automatic GPU epoch without a recipe. `cifar10-clean-smoke --data fashion-mnist` also completed one epoch, proving that template and data selection are independent.
- Browser evidence: selecting Fashion-MNIST in the beginner page generated `lnl validate --recipe cifar10-clean-smoke --data fashion-mnist --check-data`; data verify generated `lnl data verify fashion-mnist` and hid the recipe field.
- Files added: none.
- Tests executed: DataService 10/10, unified CLI 37/37, Web 21/21, full unittest 826/826; Ruff and `git diff --check` passed.
- Exact next step: human diff review; no commit or push is authorized.
- Local checkpoint commits: none. No commit or push is authorized.
- Collaborator note: `cli/main.py`, `training/data_service.py`, Web UI, and shared documents are high-conflict; no paper runner, dataset adapter, model, or recipe was changed.

- Current task: unified experiment service, result contract, overrides, sweep, comparison, runner introspection, and CI.
- Branch: `codex/cli`
- Base commit: `5e9c19b`
- Checklist: 6 infrastructure items.
- Completed: 6 / 6 (100%).
- In progress: none; final handoff pending.
- Files inspected: CLI, runner registry, experiment entry, evaluation exports, project metadata, architecture/data-flow/file-map documents, and related tests.
- Files modified: `pyproject.toml`, CLI/training/evaluation integration modules, tests, and three existing documents.
- Files added: quality workflow; config override, planning, results, service, sweep, comparison modules; six focused test files.
- Local checkpoint commits: none.
- Tests executed: focused tests passed; unified CLI 30/30; full unittest 734/734; coverage full suite passed at 74%; Ruff passed; `git diff --check` passed; wheel built, installed in a temporary environment, imported from that environment, and exposed the unified `lnl --help` commands.
- Blockers: none.
- Assumptions: sweeps are sequential; completed legacy result files with a recognized test metric remain resume-compatible.
- Exact next step: human diff review, then commit only the approved files if requested.
- Ready for history cleanup: yes; no checkpoint cleanup is needed.
- Ready to push: ready after an explicit commit and publication request.
- Collaborator note: `cli/main.py`, `training/runners.py`, `docs/file-map.md`, and `pyproject.toml` are shared/high-conflict files; changes are scoped to the approved infrastructure contracts.

## Unified data protocol migration (2026-08-18)

- Current task: one dataset registry, one preparation service, and one data entry for all paper runners.
- Branch: `codex/cli`.
- Base commit: `1ed54c4`.
- Checklist: protocol/registry; runner migration; real-noise adapters; manifest/checkpoint identity; tests; documentation.
- Completed: 6 / 6 (100%); implementation and validation are complete.
- Implemented datasets: CIFAR-10/100, CIFAR binary view, CIFAR-10N/100N, MNIST/Fashion-MNIST, Clothing1M, Animal-10N, UCI binary, synthetic binary/multiclass.
- Implemented safety: no automatic downloads, no clean target in train batches, stable global indices, deterministic epoch loaders, run-local data manifest, checkpoint fingerprint validation.
- Runner status: all listed experiment runners call `prepare_experiment_data()`; static gate rejects direct concrete loader imports and local `DataLoader` construction.
- Tests: unified data tests and affected workflow suites pass; full unittest passed 797 / 797; 63 directly available built-in recipes validate and construct their runner successfully. The conditional PCSE reproduction recipe correctly refuses construction without `LNL_PCSE_SOURCE_RUN`.
- Exact next step: human diff review, then commit only if requested.
- Collaborator note: `training/experiment.py`, `training/checkpoint.py`, `cli/main.py`, and the four documentation files are high-conflict shared files; edits are limited to generic data construction and identity validation.

## Official dataset-format correction (2026-08-19)

- Current task: verify installed and deferred datasets against primary dataset releases and official paper repositories.
- Branch: `codex/cli`; base commit: `1ed54c4`.
- Checklist: CIFAR-N safe loading; MNIST IDX compatibility; Clothing1M official metadata; ANIMAL-10N official layouts; focused tests; full regression.
- Completed: 6 / 6 (100%); implementation and validation are complete.
- Real-data evidence: local CIFAR-10N and CIFAR-100N each load 50,000 train / 10,000 test samples; local Fashion-MNIST IDX.GZ loads 60,000 train / 10,000 test samples.
- Focused evidence: official-format adapter and unified-service tests pass 13 / 13; affected data/CLI tests pass 47 / 47; full unittest passes 802 / 802.
- Exact next step: human diff review, then commit only if requested.
- Collaborator note: no runner, algorithm, model, configuration, checkpoint, or registry interface changed.

## CLI experiment workflow v2 (2026-08-20)

- Current task: align the documented CLI workflow, full dry-run preflight, matrix sweep/status, and grouping-aware comparison/reporting.
- Branch: `codex/cli`; base commit: `ccc9dc5`.
- Checklist: documentation; shared preflight; Sweep v2; Compare v2; final CLI/regression validation.
- Completed: 5 / 5 (100%); implementation and final validation are complete.
- Files inspected: README, development/data-flow guidance, CLI, experiment service, sweep, comparison, override engine, Result Contract, and focused tests.
- Files modified: README, development/file-map/progress documents, CLI, experiment service, sweep/comparison modules, and four existing test files.
- Files added: none.
- Local checkpoint commits: none.
- Tests executed: Task 1 full suite `802/802`; Task 2 focused tests and full suite `804/804`; Task 3 focused tests and full suite `808/808`; Task 4 focused tests and full suite `812/812`; final focused tests, Ruff, CLI help/dry-run checks, `git diff --check`, and final full suite `812/812` all passed.
- Blockers: none.
- Assumptions: sweeps remain sequential; duplicate seeds and duplicate resolved matrix configurations fail; strict compare excludes leakage-marked runs.
- Exact next step: human diff review, then commit only the approved files if requested.
- Ready for history cleanup: no checkpoint cleanup is needed.
- Ready to push: ready after explicit commit and publication approval; neither operation has been performed.
- Collaborator note: `README.md`, `cli/main.py`, and shared progress documents are high-conflict files; changes are limited to CLI experiment infrastructure.

## Quality workflow dry-run preflight correction (2026-08-20)

- Current task: keep default dry-run data preflight while making planning-only tests and CI smoke independent of local CIFAR files.
- Branch: `codex/cli`; base commit: `ee88b26`.
- Checklist: enumerate dry-run calls; reproduce clean-runner failures; add explicit planning-only bypasses; run full validation.
- Completed: 4 / 4 (100%).
- Pre-change evidence: a clean Git archive without `data/cifar10` reproduced 10 failures across unified CLI, Co-teaching, DLD, LEND, and UPM planning tests.
- Implementation: planning-only CIFAR dry-runs now pass `--no-check-data`; the missing-data semantic test remains unchanged; the quality workflow uses the same explicit bypass for its final smoke.
- Tests executed: Ruff passed; clean-runner focused tests passed; coverage full suite passed `812/812` with 75% coverage; unified CLI passed `33/33`; `lnl --help` and the explicit no-data dry-run passed; sdist and wheel builds passed.
- Files added: none.
- Local checkpoint commits: none.
- Blockers: none.
- Exact next step: human diff review, then commit only the approved files if requested.
- Ready for history cleanup: no cleanup needed.
- Ready to push: ready after explicit commit and publication approval.
- Collaborator note: `.github/workflows/quality.yml` is shared; no source, algorithm, data-preflight, Result Contract, or checkpoint behavior changed.

## Local dataset catalog and executable format evidence (2026-08-20)

- Current task: machine-local dataset registration, Web controls, and training-backed format verification.
- Branch: `codex/cli`; base commit: `3942558`.
- Checklist: local catalog/CLI; grayscale and real-noise compatibility; Web controls; official-format fixtures and real-data runs; final regression/documentation.
- Completed: 5 / 5 (100%); implementation and validation are complete.
- Files added: `data/local_catalog.py`, `tests/test_local_data_catalog.py`, `tests/test_dataset_training_fixtures.py`.
- Training evidence: official-format temporary fixtures completed 1 epoch for MNIST, Fashion-MNIST, Clothing1M, Animal-10N, and UCI Heart. Real F-drive data completed 1 epoch for CIFAR-10, CIFAR-100, CIFAR-10N, CIFAR-100N, Fashion-MNIST, and the CIFAR airplane/automobile view. Synthetic binary and multiclass runners produced epoch metrics.
- State semantics: registration and layout inspection never imply trainability; only a completed epoch plus data manifest is recorded as `training_verified`; a changed catalog source signature produces `verification_stale`.
- Tests executed: local catalog `2/2`, Web `15/15`, dataset training fixtures `5/5`, unified CLI `35/35`, noisy CE `11/11`, clean baseline `5/5`, torch training `39/39`; Ruff passed; full unittest passed `821/821`; `git diff --check` passed.
- Blockers: none.
- Exact next step: human diff review; commit only after explicit authorization.
- Local checkpoint commits: none. No commit or push is authorized.

## Web dataset registration usability pass (2026-08-20)

- Current task: validate and improve the full Web lifecycle for local dataset registration.
- Branch: `codex/cli`; base commit: `77cd274` plus the existing DataService working-tree changes.
- Checklist: real register/inspect/verify/remove walkthrough; operation-specific fields; actionable status and summaries; backend error JSON; two-step removal; regression and documentation.
- Completed: 6 / 6 (100%). Implementation, real-browser lifecycle validation, cleanup, and final regression are complete.
- Real evidence: a temporary CIFAR-10 alias loaded 50,000 train and 10,000 test samples, completed a one-epoch verification, and was removed without deleting source data.
- Tests executed: Web tests passed 20/20; full unittest passed 824/824; Ruff on `src tests web`, JavaScript syntax validation, and `git diff --check` passed.
- Files added: none.
- Exact next step: human diff review; no commit or push is authorized.
- Local checkpoint commits: none. No commit or push is authorized.
- Collaborator note: `README.md`, `cli/main.py`, Web files, and shared progress/file-map documents are high-conflict; changes are scoped to local data source selection and evidence status.

## Unified data management facade and Web API (2026-08-20)

- Current task: Data Phase 1/5/6 — one management facade for CLI, Web and experiment preflight.
- Branch: `codex/cli`; base commit: `77cd274`.
- Checklist: status contract; CLI list/status/path/inspect/verify; shared preflight; direct Web data API; CI gate; documentation and regression.
- Completed: 6 / 6 (100%).
- Public behavior: `missing/incomplete/ready` describes adapter-backed readiness; one-epoch `training_verified` evidence remains separate. `inspect` loads train/test, while `verify` performs the same inspection before training.
- Integration: doctor, validate, dry-run, run and sweep reach data validation through `ExperimentService` and the injected `DataService`; existing paper runners retain the compatibility `prepare_experiment_data()` entry.
- Tests executed: affected suites passed 76/76; Web focused tests passed 17/17 including a real local HTTP API request; full unittest passed 823/823; Ruff on `src tests web`, embedded Web JavaScript syntax, and `git diff --check` passed.
- Blockers: none. Conda emits a pre-existing missing OpenCL vendor temp-file warning before commands, but the `pytorch` environment and all tests complete successfully.
- Exact next step: human diff review; commit only after explicit authorization.
- Local checkpoint commits: none. No commit or push is authorized.
- Collaborator note: `cli/main.py`, `training/service.py`, `.github/workflows/quality.yml`, Web files, and shared documents are high-conflict; edits are limited to the approved data-management integration.

## Original Web console and Recipe subpage (2026-08-20)

- Current task: retain the original single-page Web console and expose the existing Recipe/YAML editor at `/recipe`.
- Branch: `codex/cli`; base commit: `77cd274` plus the completed DataService working-tree changes above.
- Checklist: restore original layout; remove unapproved workspace tabs; `/recipe` entry; `lnl web`; HTTP/CLI tests; documentation.
- Completed: 6 / 6 checklist items (100%).
- Public behavior: `lnl web` starts `127.0.0.1:8765` and opens the original console at `/`; `lnl web --no-open` suppresses browser launch; `/recipe` opens the Recipe/YAML editor directly.
- Files added: none. One route-aware HTML asset is intentionally reused to avoid duplicated UI logic.
- Validation: Web tests 18/18; unified CLI tests 36/36; full unittest 824/824; Ruff, JavaScript syntax and diff checks passed; `/` and `/recipe` were inspected in the app browser.
- Exact next step: human diff review, then commit only with separate authorization.
- Local checkpoint commits: none. No commit or push is authorized.

## CI NumPy compatibility and workflow cleanup (2026-08-21)

- Current task: diagnose the latest `quality` failure and remove redundant CI work without weakening required gates.
- Branch: `codex/cli`; base commit: `2f3ee42`.
- Checklist: inspect remote jobs and logs; reproduce the failure across NumPy versions; correct the fixture; audit every workflow step; run focused, full, Web, lint, packaging, and artifact-install checks.
- Completed: 5 / 5 (100%).
- Root cause: the Animal-10N test fixture passed pixel value `716` directly to a `uint8` array. NumPy 1.26 silently wrapped it to `204`, while NumPy 2.x correctly raised `OverflowError` in both Python 3.10 and 3.12 CI jobs.
- Implementation: fixture values now use explicit modulo-256 arithmetic; GitHub Actions use Node-24-based `checkout@v6` and `setup-python@v6`; redundant pip upgrades, duplicate unified-CLI execution, and duplicate editable-install help smoke were removed. The wheel job keeps required training dependencies but no longer installs unrelated test/coverage/lint dependencies.
- Validation: NumPy 2.3.5 boundary check passed; dataset fixtures passed 5/5; ordinary full unittest passed 824/824; Web tests passed 20/20; Ruff and workflow YAML checks passed; wheel and sdist both built, installed into isolated environments, resolved the installed package, and passed help/list/validate CLI checks.
- Local limitation: coverage-instrumented tests on Windows intermittently hit an unrelated `PermissionError` while atomically replacing a sweep manifest. The same tests pass without coverage and passed in the failed Ubuntu CI run; no training or sweep code was changed under this task.
- Files modified by this task: `.github/workflows/quality.yml`, `tests/test_dataset_training_fixtures.py`, and this progress record. Files added: none.
- Exact next step: review the three-file diff, then commit only with separate authorization.

## 26-paper formal YAML exposure (2026-08-21)

- Current task: expose exactly one formal training YAML per paper in Web “New YAML”, while retaining four beginner templates and the full Recipe editor.
- Branch: `codex/cli`; base commit: `6ed461f` plus the pre-existing approved working-tree changes.
- Checklist: official protocol audit; missing formal YAMLs; paper-to-recipe defaults; safe dedicated-runner clone; Web mode split; focused/browser/full regression.
- Completed: 6 / 6 (100%).
- Public behavior: paper mode lists 26/26 papers and clones the selected formal configuration; generic mode keeps the reusable supervised loss/selector composer.
- Fidelity boundary: CNLCU and VolMinNet new defaults are `paper_protocol`; DLD remains `paper_oriented` because the official ViT-L/14 feature path is not present in the toolbox.
- Modular changes: CNLCU architecture is a reusable model, its decay is a generic scheduler, and no paper-name branch was added to the common experiment runner.
- Focused validation: CNLCU 2/2, DLD 4/4, VolMinNet 6/6, schema 4/4, unified CLI 40/40, Torch training 41/41, Web 21/21; browser interaction verified all 26 options and both creation modes.
- Final validation: full unittest 840/840; Web 21/21; runner package-data 7/7; Ruff passed; CNLCU, DLD and VolMinNet formal-config dry-runs passed; app-browser interaction verified all 26 options and both creation modes.
- Exact next step: human diff review and optional commit under separate authorization; no commit or push is authorized.

## Web YAML creation/editor closeout (2026-08-21)

- Current task: close the broken gap between “New YAML” and the Recipe editor.
- Branch: `codex/cli`; base commit: `6ed461f` plus the approved working-tree changes.
- Checklist: path-based load API; validated full-text save; 4+26 editor sources; automatic open after create; project-YAML reload; browser and regression verification.
- Completed: 6 / 6 (100%).
- Public behavior: a paper YAML created from the main page opens immediately in the editor; the textarea is editable; save, validate and run target the generated project YAML. The editor exposes four beginner templates and all 26 paper defaults without making project files built-in recipes.
- Safety: paths remain restricted to the repository, built-in recipes cannot be overwritten, and full YAML text is parsed and passed through `validate_config()` before writing.
- Browser evidence: created a CNLCU project YAML, automatically opened it, changed seed 1 to 23/24, saved it, reloaded it by project path, and completed `lnl validate`; the editor contained 31 options (4 templates + 26 paper defaults + the current project YAML).
- Final validation: Web 23/23, full unittest 840/840, Ruff, embedded JavaScript syntax and `git diff --check` passed.
- Cleanup: the browser-created project YAML was removed after validation; no generated training or configuration artifact remains.
- Exact next step: human diff review and optional commit under separate authorization; no commit or push is authorized.

## 26-paper noise compatibility preflight (2026-08-22)

- Current task: add the missing compatibility requirements for 14 papers without changing algorithms, experiment runners, adapters, YAML, CLI or Web.
- Branch/base: `codex/cli` at `7eaa452`.
- Checklist: declarative config-input contract; component-driven shared-runner providers; dedicated-runner providers; 26-paper/native-noise regression; full validation.
- Completed: 5 / 5 (100%).
- Public behavior: all 26 formal paper recipes now publish config-specific requirements during preflight. Missing rate pairs, known-T matrix, trusted supervision or external artifacts produce stable requirement codes before runner invocation.
- Safety boundary: compatibility describes the current executable path. Observed-only native noise is rejected when that path requires a manifest or aligned clean/noisy evidence; no formula, lifecycle, data adapter or clean-label boundary changed.
- Tests executed: compatibility 23/23, experiment service 3/3, unified CLI 47/47, full unittest 878/878; focused Ruff and `git diff --check` passed.
- Files added: none. Blockers: none. Local checkpoint commits: none.
- Exact next step: human review of the eight-file diff; commit or push requires separate authorization.
- Collaborator note: `training/runners.py`, `training/service.py` and shared documents are high-conflict; changes are limited to requirements providers, preflight validation and documentation.

## Web dataset-aware YAML workflow (2026-08-22)

- Current task: remove the duplicate YAML destination, make the editor contextual, and surface dataset/method compatibility while constructing YAML.
- Branch/base: `codex/cli` at `7eaa452`, layered on the approved uncommitted 26-paper compatibility work.
- Checklist: concrete-recipe batch query; server-side save gate; one-path editor; draft-preserving module switch; dataset-to-YAML link; focused/browser/full regression; documentation.
- Completed: 6 / 7 (86%); focused service 4/4 and Web 47/47 tests pass, and browser checks confirm CIFAR disables Binary Risk/FINE while enabling Loss Correction, a single YAML path, automatic editor collapse, and draft restoration.
- Safety boundary: dataset selection is a runtime `--data` override and is labeled as migration training; locked paper data/noise protocol fields, algorithms, runners, adapters and formal YAML remain unchanged.
- Files added: none. Local checkpoint commits: none. Push is not authorized.
- Exact next step: run full unittest/Ruff/final browser check, then update this record to 7/7 and hand off the reviewed diff.

## Dataset-first workflow 输入重构（2026-08-23）

- Current task: 按 Dataset-first 计划区分适配器事实、UNKNOWN 数据集声明和具体 recipe 实验输入。
- Branch: `codex/cli`; base: 当前工作区既有兼容性与 Web 修改；未提交、未推送。
- Checklist: adapter semantic hints；UNKNOWN-only 声明；具体 formal recipe；方法先验配置注入；实际预训练资源；Web 三段式 UI；测试与文档。
- Completed: 7 / 7（100%）。
- Public behavior: Web 不再允许把 method prior 或 pretrained role 写入 dataset catalog；正式 recipe 必须显式选择；兼容结果返回真实 input paths；原生 Clothing1M/Animal-10N 语义由 adapter hints 提供。
- Validation: Web `test_command_console.py` 47/47；数据适配器 46/46；Python 编译检查通过。完整 registry 测试当前受仓库既有缺失模块 `test_t_revision_workflow` 阻塞，未归因于本任务。
- Files modified in this task: `README.md`, `web/README.md`, `web/index.html`, `web/command_console.py`, `docs/data-flow-guide.md`, `docs/development-guide.md`, `docs/file-map.md`, `docs/toolbox-modularization-progress.md`, `src/lnl_toolbox/data/profile.py`, `src/lnl_toolbox/data/real_noise.py`, `src/lnl_toolbox/training/data_service.py`, `src/lnl_toolbox/training/compatibility.py`, `src/lnl_toolbox/training/service.py`, `tests/test_data_adapters.py`, `tests/test_registry.py`, `web/test_command_console.py`。其中多份文件已有协作者工作，本次只做局部增量。
- Exact next step: review final diff and resolve the pre-existing merged-test import issue before considering commit/push.

## 26 篇论文统一数据接头收口（2026-08-25）

- Current task: 以 `RunnerSpec` 为唯一 method requirement 真源，让 compatibility 与真实 runner 共享同一个 `DataRequirements`，并仅解除不改变现有数学流程的数据限制。
- Branch/base: `new-cli` at `3382934`; no branch creation, commit, or push is authorized.
- Checklist: canonical requirement source；RunnerSpec→runner identity；第一批六方法；其余二十方法；label/role boundary；tests/docs。Completed: 6 / 6（100%）。
- Public contract: `RunnerSpec.data_requirements(config)` 同时供 compatibility 与 `RunnerSpec.invoke()` 使用；`prepare_experiment_data(..., requirements=...)` 保持显式必填，DataService 不认识 method/runner。
- Safe removals: 可由 role/view/model input adapter 支持的 dataset-name、CIFAR、IMAGE、固定类别数和手工 feature snapshot 限制已解除。CWD binary、VolMinNet `C>=3`、Binary Risk/Importance binary、DLD multi-view、FINE robust strong-view、L2RW trusted supervision 等仍按 `implemented_variant_limit` 或算法 requirement 保留；没有新增论文 variant 或数学流程。
- Label boundary: noisy TRAIN 始终使用 observed target 且 batch 不暴露 clean label；clean/trusted labels 只在显式 evaluation、trusted/meta/curriculum role 使用。协议 TEST source 可提供 clean evaluation ground truth，TRAIN 派生 clean validation 仍须真实 clean target。
- Focused validation: registry `49/49`；data adapters `54/54`；CLI `89/89`；Quick Start `10/10`；training `72/72`（1 skipped）；estimators `49/49`；各 dedicated method suites已通过。
- Full validation: `949` tests executed; `946` passed, `1` skipped, and the two known baseline assertions below failed. All focused suites affected by this task pass.
- Known baseline failures outside this task: `test_config.py` still hard-codes 94 YAMLs although 95 are tracked; `test_noise.py` expects a generic length error while manifest validation raises the more specific out-of-namespace index error. Neither file nor noise/config implementation was modified.
- Files added: none. Four pre-existing untracked WebUI audit files remain untouched.
- Local checkpoint commits: none.
- Exact next step: human review of the final diff and the two explicitly recorded baseline failures; commit or publication requires separate authorization.
- Collaborator note: `training/runners.py`, `training/compatibility.py`, `training/data_service.py`, `training/experiment.py`, and shared docs are high-conflict; algorithms, losses, selectors, estimators, reproduction YAML, CLI/Web/Scratch, and legacy evidence runners remain unchanged.

## Quick Start 与 validation 语义收口（2026-08-28）

- Current task: 修复 Quick Start exact formal recipe、typed unsupported split、TEST-as-validation、JoCoR validation-size 和 PDL variant predicate 问题；禁用不可选方法卡片。
- Branch/base: `new-cli` at `3a7582d`; 不创建分支、不提交、不推送；工作区原有未提交内容保持不动。
- Completed: 7 / 7（100%）：exact matcher；typed split contract；无 validation 时不伪造 TEST；JoCoR；PDL predicate；Web 状态保护；回归验证（仅剩一项既有断言文本不一致）。
- Public behavior: Quick Start 的 `clean/native` 语义不再混同，formal 命中返回真实 recipe id；validation 仅来自 native validation 或 TRAIN 派生 split，零 validation 时 `PreparedData.validation_split=None` 且 manifest 省略 validation。
- Focused validation: data adapters 57/57；Quick Start service 11/11；PDL 20/20；JoCoR 7/7；Web assets/integration 6/6；CLI 89/89；data-contract 及 training focused suites通过。
- Full validation: 963/963 tests 通过；`test_noise` 的错误文本断言已更新为匹配当前更具体的 index-namespace 错误。未放宽生产代码的 typed-exception 边界。
- Files modified in this task: `src/lnl_toolbox/quickstart/templates.py`, `src/lnl_toolbox/quickstart/service.py`, `src/lnl_toolbox/data/contracts.py`, `src/lnl_toolbox/data/sources.py`, `src/lnl_toolbox/data/cifar_n.py`, `src/lnl_toolbox/data/real_noise.py`, `src/lnl_toolbox/training/data_service.py`, `src/lnl_toolbox/training/multi_model_experiment.py`, `src/lnl_toolbox/training/runners.py`, `src/lnl_toolbox/training/instance_transition_experiment.py`, `web/assets/quick_start.js`, `tests/test_data_adapters.py`, `tests/test_data_capability_contract.py`, `tests/test_experiment_service_data_contract.py`, `tests/test_pdl.py`, `tests/test_quickstart_service.py`, `tests/test_training.py`, `web/test_quick_start_assets.py`, `web/test_quick_start_integration.py`, and this progress record. No new files.
- Known collaborator scope: shared `training/data_service.py`, `training/runners.py`, Quick Start assets and progress documentation are high-conflict; no algorithm/loss/selector implementation was changed.
- Exact next step: human review；本轮功能和回归均已收口。无 commit 或 push 授权。
