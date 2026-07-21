# Repository Guidelines

## Project Structure & Module Organization

Python source lives under `src/lnl_toolbox/`. Keep task-neutral contracts in `core/`, lifecycle orchestration in `engine/` or `training/`, and concrete implementations in `data/`, `models/`, `losses/`, `noise/`, `algorithms/`, and `evaluation/`. Command-line entry points belong in `cli/`; reusable component registration belongs in `plugins/`.

Tests are in `tests/` and mirror source responsibilities (`test_noise.py`, `test_clean_baseline.py`). YAML configurations live in `configs/{experiment,algorithm,noise}/`. Documentation is in `docs/`, research papers in `papers/`, and helper scripts in `scripts/`. Local datasets under `data/` and generated runs under `artifacts/` are ignored and must not be committed.

## Build, Test, and Development Commands

Use Python 3.10 or newer. Install the package and training dependencies:

```powershell
python -m pip install -e ".[train]"
$env:PYTHONPATH = "src"
```

Run the full test suite:

```powershell
python -m unittest discover -s tests -v
```

Run a short GPU/CPU integration check:

```powershell
python -m lnl_toolbox.cli.clean_train --config configs/experiment/cifar10_clean_smoke.yaml
```

Use `cifar10_clean_baseline.yaml` for formal clean-label runs. Resume with `--resume <run>/last.pt`; run repeated seeds with `--seeds 1 2 3`.

## Coding Style & Naming Conventions

Follow standard Python style: four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and descriptive type hints on public interfaces. Prefer small dataclasses and protocols over framework-wide inheritance. Keep `core/` independent of PyTorch and LNL-specific assumptions. Use docstrings where lifecycle behavior, state ownership, or tensor shapes are not obvious. No formatter or linter is currently enforced; match surrounding code and keep imports grouped.

Do not add new files unless necessary; prefer extending an existing, appropriately scoped module or document.

## Testing Guidelines

Tests use the standard-library `unittest` framework. Name files `test_<feature>.py`, classes `<Feature>Test`, and methods `test_<behavior>`. Add deterministic tests for seeds, checkpoint round trips, tensor shapes, and failure cases. GPU-dependent assertions must skip or fall back cleanly when CUDA is unavailable. Before submitting, run all tests plus the relevant smoke configuration.

## Multi-machine Collaboration Guidelines

This repository is developed by two contributors on separate machines. Each feature should normally be developed on a dedicated local feature branch created from the latest stable version of the shared main branch. Use short, descriptive names such as `feature/gce-loss`, `feature/coteaching`, or `feature/noise-transition`.

Local commits and remote publication are separate operations. A feature branch may contain several local checkpoint commits for experimentation, recovery, and staged development. Intermediate experiments, failed attempts, temporary debugging work, and checkpoint commits remain local by default. Publishing work with `git push` is a deliberate release step performed only after testing, diff review, history cleanup or an explicit decision to retain the history, and user approval.

The normal collaboration flow is:

1. Start from an up-to-date stable baseline and create a local feature branch.
2. Develop and test the feature in coherent, reversible stages, using authorized local checkpoint commits when helpful.
3. Run focused tests, affected-module tests, the full test suite, and the relevant smoke test before publication.
4. Review the branch status and final diff, then consolidate temporary checkpoints into a small number of clear commits when appropriate.
5. After explicit user approval, push the feature branch, never an unreviewed experiment directly to `main` or `master`.
6. Have a person review and merge the feature branch into the shared main branch.

Each contributor should own a clear module and file scope wherever practical. Put algorithm implementations in separate modules, unit tests in separate `test_<feature>.py` files, and algorithm configurations in separate YAML files. Registries, factories, `README` files, dependency files, shared configurations, and shared progress documents are high-conflict files; a designated contributor should integrate changes to them whenever possible. Coordinate before changing public interfaces or high-conflict files, and record the exact shared-file locations changed in the final handoff.

## Commit & Pull Request Guidelines

Local feature development may use multiple checkpoint commits, but those checkpoints remain local by default. After testing, shape the publishable history into a small number of clear feature commits and keep unrelated changes separate. Final commit subjects should be concise and imperative, such as `Add PreActResNet clean baseline`.

Pull requests and merge handoffs must explain experiment-facing behavior, configuration or API changes, tests and results, and relevant issues or papers. Include useful metric snippets for training changes. Never commit datasets, model or training checkpoints, large generated outputs, or environment-specific files. Do not push unreviewed experimental code directly to the shared `main` or `master` branch.

## Reproducibility & Safety

Never expose clean labels to a noisy-label training algorithm; reserve them for evaluation. Preserve stable global sample indices, resolved configurations, seeds, environment metadata, and complete optimizer/scheduler state in checkpoints.

## Codex Local Development Protocol

Codex-assisted changes are developed and tested on local feature branches, then prepared for human review and controlled publication to the shared repository. Local checkpoint commits are recovery points; a remote push publishes work to collaborators and therefore has a higher approval and readiness threshold.

### Git Operations and Authorization

Read-only inspection commands may be used when relevant:

- `git status`
- `git diff`
- `git log`
- `git show`
- `git branch`

After the user explicitly authorizes the current task to use the corresponding local Git operation, Codex may run:

- `git switch -c <feature-branch>`
- `git add <explicit-file-paths>`
- `git commit -m "<message>"`

Before creating or switching to a branch, Codex must report the current branch, `git status`, and all uncommitted changes. Creating a feature branch requires explicit user authorization for the current task. Codex must not switch branches while unrelated modifications remain unexplained.

Before staging or committing, Codex must report the plan, run `git status` and `git diff`, list the exact files to be committed, and verify that every file belongs to the current task. Use explicit file paths with `git add`; do not default to `git add -A`, stage the entire workspace, or include a collaborator's unrelated changes.

Codex may create a local checkpoint commit on the current feature branch only when the user has explicitly authorized checkpoints for the current task. Messages must identify the stage, for example `checkpoint: before GCE implementation`, `checkpoint: add initial GCE loss`, or `checkpoint: add GCE tests`. Each checkpoint must represent an understandable, reversible development stage and remains local unless the user later approves publication.

Without explicit user confirmation for the specific operation, Codex must not run:

- `git push` or any force push;
- `git pull` or a fetch workflow that automatically merges;
- `git merge` or `git rebase`;
- `git reset`, `git restore`, `git clean`, or `git stash`;
- branch deletion;
- remote URL modification; or
- any operation that rewrites shared main-branch history.

Never automatically push an intermediate checkpoint, push directly to `main` or `master`, squash or rewrite history, use `git push --force`, or run `reset --hard` or `clean` without first confirming the exact branch and worktree state. Even after a branch is ready, `git push -u origin <feature-branch>` requires explicit approval for that command, remote, and branch.

### Change Approval Boundary

Codex must distinguish four primary approval layers: inspection, modification, commit, and publication. History rewriting is an additional operation-specific approval boundary between commit and publication. Permission to inspect does not authorize modification; modification approval does not imply commit approval; commit approval does not imply history-rewrite approval; and publication approval must always be granted separately for the exact remote and feature branch.

Before modifying any business code, configuration, test, or documentation, Codex must present:

- its understanding of the task;
- the files it plans to inspect;
- the exact files proposed for modification;
- the exact files proposed for creation;
- important related files intentionally left unchanged;
- the planned changes for each file;
- the tests or validation commands it plans to run; and
- possible collaborator conflicts.

For every non-trivial task, Codex must present this plan, stop, and wait for explicit user approval before editing. It must not present a plan and begin editing in the same response. A user's approved modify-and-create file list is the modification allowlist for that task. Codex may modify or create only files on that allowlist and must not interpret general relevance to the task as permission to change other related files.

If implementation reveals that an additional file is required, Codex must stop before changing it, explain why the file is needed, describe the exact proposed change, and request explicit approval to expand the allowlist. Until approval is granted, the file must remain unchanged.

Read-only file inspection, code search, non-mutating tests, and read-only Git commands such as `git status`, `git diff`, `git log`, and `git show` do not require separate modification approval. Before running a command that generates, overwrites, formats, migrates, or otherwise changes repository files, Codex must disclose the expected repository changes and obtain modification approval for the affected paths.

At completion, Codex must compare the actual modified and created files with the approved allowlist, summarize the changes to each file, state whether any unplanned files were produced, report test results, and identify potential collaborator conflicts. If any actual change falls outside the allowlist, Codex must report it and must not stage, commit, rewrite history, or publish the changes before user review.

### Before Modifying Code

For every non-trivial task:

1. Read the relevant architecture, development, data-flow, and method documentation.
2. Inspect the existing implementation and relevant tests.
3. Check whether the requested functionality already exists.
4. Reuse existing interfaces and modules where possible.
5. Present the complete file-level plan required by `Change Approval Boundary`.
6. Stop and wait for explicit user approval of the modification allowlist before editing.
7. Identify assumptions, ambiguities, and potential conflicts with collaborator work.

Do not begin implementation or a large refactor until the plan and modification allowlist are explicitly approved.

### Scope Control

Modify only files included in the approved modification allowlist for the current task.

Do not:

- overwrite unrelated collaborator work;
- delete code merely because its purpose is not immediately clear;
- perform unrelated refactoring;
- run repository-wide formatting;
- rename public interfaces without explicit approval;
- move large directory trees without explicit approval;
- change experiment defaults without documenting the reason;
- add dependencies without explaining why they are required;
- silently expand the task after discovering additional problems.

Record newly discovered but out-of-scope work as follow-up items instead of implementing it automatically.

### Incremental Implementation

Implement one coherent and testable step at a time.

The recommended local feature workflow is:

1. Inspect the current branch, `git status`, recent commits, and any existing progress record.
2. With explicit authorization, create a feature branch from the stable baseline.
3. If all relevant worktree content belongs to the task and checkpoints are authorized, create a pre-change checkpoint.
4. Implement the smallest testable step.
5. Run the focused tests for that step.
6. Create an authorized local checkpoint representing the completed stage.
7. Record the completed work, changed files, actual test results, blockers or assumptions, and exact next step; then continue iteratively.
8. When implementation is complete, run all required focused, affected-module, full-suite, and smoke tests.
9. Inspect the final status and diff for scope, generated files, and collaborator conflicts.
10. Prepare the final commit history, subject to the history-cleanup approval rules below.
11. After explicit user approval, push the feature branch to the named remote.
12. Hand the branch off for human review and merge.

Do not claim that a feature is complete until its relevant tests and integration path have been checked.

### Final History Cleanup

After the feature is complete and its required tests pass, temporary checkpoint commits may be consolidated into a small number of clear commits before push. Codex must not automatically run interactive rebase, squash, soft reset, or any other history-rewriting operation.

Before proposing a cleanup operation, Codex must:

1. Show the feature branch's commits relative to its baseline.
2. Recommend which commits, if any, should be combined.
3. Provide the expected final commit messages.
4. Wait for explicit user confirmation.
5. Only then run the specifically approved cleanup command.

Each final remote commit should represent one clear feature and exclude failed experiments, temporary debugging information, datasets, model checkpoints, large artifacts, local environment files, and unrelated changes.

### Persistent Progress Handoff

Before starting a new task, check whether the repository already has a progress, status, roadmap, project-management, or task-tracking document.

Prefer updating an existing suitable document instead of creating a duplicate progress file. If no suitable progress document exists, report this to the user before creating one.

The persistent progress record should contain:

- current task;
- current feature branch;
- base branch or base commit;
- precise goal;
- task checklist;
- completed item count;
- total item count;
- progress percentage;
- files inspected;
- files modified;
- files added;
- local checkpoint commits;
- tests executed and results;
- blockers;
- important assumptions;
- exact next step;
- whether the branch is ready for final history cleanup;
- whether the branch is ready to push; and
- collaborator integration notes.

Progress must be calculated from completed checklist items:

`progress = completed items / total items`

Do not estimate progress only from intuition.

At the start of a new Codex session:

1. Inspect the current branch and `git status`.
2. Review recent local commits.
3. Read the latest progress record.
4. Inspect the actual implementation and tests.
5. Compare the recorded state with Git, code, and test state and correct stale progress information when necessary.
6. Continue from the recorded exact next step without asking the user to repeat recorded context.

Git and the actual code and test state take precedence over stale documentation.

### Testing Rules

Run the narrowest relevant tests first, then broader tests when appropriate.

Before Codex recommends pushing a feature branch, it must actually run and record:

1. Focused unit tests for the feature.
2. Tests for affected modules.
3. The full `unittest` suite.
4. The relevant smoke configuration.
5. Checkpoint save/resume checks when applicable.
6. CPU/CUDA fallback checks when applicable.

Use the repository's existing `unittest` commands and smoke configurations unless the task requires something else.

For every test, record:

- the exact command;
- whether it passed, failed, or could not run;
- the relevant error or limitation.

Never claim that a test passed unless it was actually executed.

Classify every failure accurately as an implementation failure, test failure, environment failure, dependency failure, or data/network failure. Do not recommend publication while an unexplained failure remains. If a known limitation prevents a required test, the user must explicitly accept that limitation before the branch can be considered for push.

### Collaborator Protection

Assume that existing files may contain work prepared by another collaborator.

Codex must:

1. Preserve unrelated existing functionality and never stage or commit a collaborator's unrelated work.
2. Keep feature ownership to a clear module and file scope wherever practical.
3. Put algorithms, feature tests, and feature configurations in separate files when practical.
4. Treat registries, factories, `README` files, dependency files, shared configurations, and shared progress documents as high-conflict files, preferably modified by a designated integrator.
5. Check whether collaborator modules depend on a public interface before changing it.
6. Avoid unrelated shared-interface refactors for an individual feature.
7. Report a likely shared-file conflict before editing.
8. Avoid whole-file replacement when a scoped edit is sufficient, and never use whole-file replacement to combine collaborator changes.
9. Avoid changing unfamiliar modules without first tracing their use.
10. List every modified and added file at task completion.
11. Identify exact changed locations in shared files and explain every public API, configuration, dependency, or data-flow change collaborators must integrate.

### Push Readiness

A push is publication, not a checkpoint operation. Codex may recommend pushing only when all of the following are true:

- implementation is complete;
- required tests passed, or the user accepted the remaining documented limitations;
- `git status` and the final diff were reviewed;
- checkpoint history was cleaned up or the user explicitly chose to retain it;
- no unrelated files are included;
- the target remote and feature branch are explicit; and
- the user explicitly approved this push operation.

Codex must never push directly to `main` or `master`. The default publication form is `git push -u origin <feature-branch>`, and it still requires approval for the current operation.

### Completion and Integration Report

At the end of every implementation task, report using this structure:

```text
Completed
- ...

Current branch
- ...

Base branch or commit
- ...

Files modified
- ...

Files added
- ...

Approved modification allowlist
- modified files: ...
- created files: ...

Allowlist comparison
- actual changes match allowlist: yes / no
- unplanned files produced: none / ...

Changes by file
- path: ...
  summary: ...

Local checkpoint commits
- ...

Tests executed
- command: ...
  result: ...

Final diff status
- ...

History cleanup status
- not needed / pending approval / completed

Push readiness
- not ready / ready pending approval / pushed with approval

Potential collaborator conflicts
- ...

Integration instructions
1. ...
2. ...
3. ...

Known limitations
- ...

Progress record
- file: ...
- completed / total: ...
- exact next step: ...
```

The report must let the other contributor review, transfer if necessary, publish with approval, and verify the branch without reading the entire Codex conversation. Include relevant collaborator files intentionally left unchanged and the exact locations changed in any high-conflict shared file.
