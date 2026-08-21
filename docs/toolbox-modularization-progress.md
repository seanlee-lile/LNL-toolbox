# Toolbox modularization progress

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
