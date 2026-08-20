# Toolbox modularization progress

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
