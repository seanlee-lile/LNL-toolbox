# Toolbox modularization progress

## Current task

Selective integration of collaborator branch `new-cli` into the unified `fixes`
baseline. The integration preserves the existing public runner protocol and
does not merge the collaborator branch wholesale.

## Baseline and scope

- Branch: `fixes`
- Base commit: `7743620` (`Unify paper runner interfaces and lifecycle reporting`)
- Collaborator review ref: `review/new-cli` at `0110d81`
- Approved scope: add non-conflicting collaborator workflows, preserve all
  existing papers, names, YAML behavior, unified adapters, reports, and tests.
- Intentionally unchanged: collaborator untracked directories `cnlcu-run/`,
  `coteaching-run/`, `fixture/`, and `docs/paper-code-flowcharts.md`.

## Completed in this task

1. Added DivideMix as a registered unified workflow.
2. Added standalone VolMinNet as a registered workflow while preserving the
   existing `volmin` runner and catalog identity.
3. Added method-specific smoke recipes, paper catalog entries/variants, docs,
   algorithm modules, workflow tests, and packaging data-file entries.
4. Adapted the shared lifecycle adapter to accept the new workflow context and
   to make completed-run resume a true no-op without rewriting artifacts.
5. Updated registry contract tests for the new runner names.

## Validation

- DivideMix-focused tests: 30/30 passed.
- VolMinNet-focused tests: 18/18 passed after no-op correction.
- Runner and unified contract tests: 5/5 and 33/33 passed.
- Compatibility workflow registry tests: 4/4 passed.
- Full suite: 655 tests ran; the only failure was the environment-dependent
  `test_repository_data_path` because this worktree has no local CIFAR files.
  The original F-drive CIFAR-10 directory was present and used for smoke runs.
- F-drive DivideMix smoke: completed at 1 epoch, resumed to 2 epochs.
- F-drive VolMinNet smoke: completed at 1 epoch.

## Status

- Interface integration: completed for the two non-conflicting collaborator
  workflows in this pass.
- Smoke status: verified for DivideMix and VolMinNet only.
- Formal training: not run.
- Paper fidelity: not audited or claimed.
- Remaining collaborator work: DLD, UPM, and LEND have same-name or ownership
  conflicts with current implementations and require per-method equivalence
  review before any replacement or adapter migration.

## Next step

Perform a file-level mathematical/data-flow comparison for DLD, UPM, and LEND;
only then decide whether to port isolated modules, add compatibility aliases,
or leave the current implementations unchanged.
