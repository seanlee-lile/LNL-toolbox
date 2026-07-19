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

## Commit & Pull Request Guidelines

History is small and has no strict convention. Use concise, imperative subjects such as `Add PreActResNet clean baseline`; keep unrelated changes separate. Pull requests should explain the experiment-facing behavior, list configuration/API changes, report tests run, and link relevant issues or papers. Include metric snippets for training changes, but never commit datasets, checkpoints, or large generated artifacts.

## Reproducibility & Safety

Never expose clean labels to a noisy-label training algorithm; reserve them for evaluation. Preserve stable global sample indices, resolved configurations, seeds, environment metadata, and complete optimizer/scheduler state in checkpoints.
