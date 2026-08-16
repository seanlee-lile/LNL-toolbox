# Directional Label Diffusion: reproduction status

## Fidelity

The implemented `paper_oriented_v2_cosine_similarity` policy fixes averaged
weak/strong `y0`, an estimated `yn`, direction `yn - y0`, included
self-neighbors, `KL(p_s || p_w)` without a second softmax, average diffusion
schedules, and five deterministic reverse steps. For the cosine backend it
matches the released implementation: select the largest cosine similarities
and use `1 / (similarity + delta)` before row normalization. Invalid non-positive
denominators fail rather than being clipped. A zero initial inference vector is
documented as a toolbox engineering choice.

This version is user-ready as a workflow smoke. It is not a paper-exact numerical
reproduction. In particular, the repository-frozen-model backend substitutes
for the paper/released representation setup, and the bundled CIFAR smoke uses
synthetic symmetric noise.

## Commands

```powershell
python -m lnl_toolbox.cli.main papers show dld
python -m lnl_toolbox.cli.main validate --recipe cifar10-dld-smoke
python -m lnl_toolbox.cli.main run --recipe cifar10-dld-smoke --dry-run
python -m lnl_toolbox.cli.main run --recipe cifar10-dld-smoke
python -m lnl_toolbox.cli.main resume <run-directory>
```

The pre-correction artifact records stable indices, noisy targets, both neighbor
distributions, partition evidence, `y0`, `yn`, `yd`, condition features, manifest
and mapping hashes, feature/transform identities, KNN/GMM settings, and fidelity.
It is written to a sibling temporary NPZ, reloaded and validated, then atomically
replaced before phase/checkpoint advancement.

## Deferred reproduction work

- paper/released pretrained representation backends;
- the paper's exact noise/data protocols and full training budgets;
- multi-seed accuracy aggregation and paper-table comparison.
