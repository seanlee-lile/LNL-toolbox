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
