# Dual-T CIFAR-10 Reproduction Guide

This guide fixes the repository configuration for the CIFAR-10 symmetric
20% noise setting from Yao et al., *Dual T: Reducing Estimation Error for
Transition Matrix in Label-noise Learning* (NeurIPS 2020). It distinguishes
parameters stated by the paper from choices required to make the local
training workflow fully specified.

The implementation provides a runnable `method: dual_t` workflow:

```text
noisy-label posterior training
-> best noisy-validation checkpoint
-> stable-index posterior snapshot
-> Dual-T transition estimation
-> fresh Forward-corrected classifier
-> clean-test evaluation
```

The evidence workflow reuses one posterior checkpoint and one persisted
posterior snapshot for ordinary anchor-based T and Dual-T. It then compares
matrix error and trains three independently reconstructed final arms:
Noisy CE, T-Forward, and DT-Forward.

## Commands

Run the production Dual-T workflow:

```powershell
$env:PYTHONPATH = "src"
python -m lnl_toolbox.cli.train `
  --config configs/reproduction/cifar10_dual_t_sym20.yaml `
  --output-dir artifacts/reproductions/cifar10-dual-t-sym20-seed1
```

Run the standalone evidence chain:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_dual_t_evidence.py `
  --config configs/reproduction/cifar10_dual_t_sym20_evidence.yaml `
  --output-dir artifacts/reproductions/cifar10-dual-t-sym20-evidence-seed1
```

Generated checkpoints, manifests, metrics, and other run artifacts remain
under `artifacts/` and must not be committed.

## Paper-Specified Settings

Section 4.1 of the paper specifies the following settings for transition
matrix estimation on CIFAR-10:

- symmetric label noise at rate 20% is an evaluated setting;
- 20% of the original training examples are held out as noisy validation;
- the best noisy-validation-accuracy checkpoint is used for anchor
  estimation;
- anchors and posteriors are collected from the remaining training split;
- the CIFAR-10 posterior classifier is ResNet-18;
- the posterior classifier is trained for 100 epochs with SGD;
- the initial learning rate is 0.01 and is reduced by a factor of ten after
  epoch 50.

The configurations encode the 20% holdout as `validation_size: 10000`.
Neither formal configuration limits the train, validation, or test sample
count. Both training and validation labels are corrupted through the same
synthetic-noise configuration, while the test split remains clean.

The paper defines the Dual-T estimator and evaluates DT-Forward, but its main
text does not fully specify every optimizer, preprocessing, loader, or
final-classifier training parameter needed by this repository.

## Implementation Choices

The following values make the experiment executable and reproducible, but
must not be described as original Dual-T paper hyperparameters:

| Setting | Repository choice |
|---|---|
| batch size | 128 |
| SGD momentum | 0.9 |
| weight decay | 5e-4 |
| Nesterov | disabled |
| preprocessing | repository `standard` CIFAR preprocessing |
| augmentation | enabled for training |
| experiment/noise seed | 1 |
| DataLoader workers | 0 for deterministic Windows-compatible execution |
| pinned memory | enabled |
| device | `auto` |
| final classifier | a fresh CIFAR ResNet-18 with Forward correction |
| final training budget | 100 epochs, SGD at LR 0.01, 10x decay at epoch 50 |
| evidence sampler seed | 1001 |

The final-stage schedule deliberately mirrors the paper-specified posterior
schedule as a conservative implementation choice. It is not evidence that
the paper prescribed this exact DT-Forward optimization setup.

## Smoke Versus Formal Configurations

`configs/experiment/cifar10_dual_t_smoke.yaml` and
`configs/experiment/cifar10_dual_t_evidence_smoke.yaml` verify lifecycle,
artifact, fairness, and resume behavior. They use TinyCNN, small dataset
subsets, CPU execution, and short epoch counts. Their accuracy and matrix
errors are pipeline diagnostics only.

The two files under `configs/reproduction/` use the full CIFAR-10 split,
ResNet-18, and formal training budgets. They are intended for GPU sanity and
long-running experiments, not routine unit tests.

## Experiment Plan and Claim Boundary

Run the validation sequence in this order:

1. Run the existing production and evidence smoke configurations.
2. Run a short GPU sanity using the smoke configuration on CUDA and inspect
   checkpoint, snapshot, transition artifact, and resume behavior.
3. Run one full seed with `cifar10_dual_t_sym20.yaml`.
4. Run the formal evidence configuration and compare ordinary T versus
   Dual-T matrix error and T-Forward versus DT-Forward clean-test accuracy.
5. Repeat the formal configurations for five explicitly recorded seeds and
   report mean and standard deviation.

At the time this document was added, production and evidence-chain smoke
tests had passed. No full formal run had been completed. The repository may
claim a tested Dual-T + Forward workflow and a tested evidence-chain
implementation; it must not yet claim reproduction of the paper's reported
classification accuracy or five-seed statistics.

## Source

- Yu Yao, Tongliang Liu, Bo Han, Mingming Gong, Jiankang Deng, Gang Niu,
  and Masashi Sugiyama. *Dual T: Reducing Estimation Error for Transition
  Matrix in Label-noise Learning*. NeurIPS 2020.
- Local paper:
  `D:/ABhomework/科研/label-noise/papers/04_statistic_estimation/dual-t-reducing-estimation-error-for-transition-matrix-in-label-noise-learning.pdf`
- Proceedings:
  <https://proceedings.neurips.cc/paper/2020/hash/512c5cad6c37edb98ae91c8a76c3a291-Abstract.html>
