# LEND reproduction and maintenance

The toolbox implements a **论文方法对齐（paper-oriented）工程流程（engineering workflow）**.
It is not described as paper-exact, released-code exact, or a numerical
reproduction because no
verifiable author implementation was found and the paper leaves several runtime
choices unspecified.

## Implemented flow

```text
current noisy mini-batch
-> one model forward producing logits and penultimate embeddings
-> directed batch-local kNN and Eq. (1) adjacency
-> A^T A and symmetric degree normalization
-> Eq. (3) label dilution from noisy one-hot labels until convergence
-> stable-index Eq. (5) cross-epoch history
-> Eq. (6) agreement selection
-> summed selected noisy-label CE
```

Diluted vectors are detached selection evidence, not probability targets. Empty
selected batches commit the successfully observed label history but skip
backward and the optimizer step.

The catalog default and Scratch paper recipe stop when the maximum absolute
change between successive label matrices is at most `1e-5`, with a safety cap
of 1000 iterations. The paper requests convergence but supplies neither number;
both are explicit toolbox numerical settings. Exhausting the cap raises an
error, not a successful selection result. Iteration count, residual and
convergence status are reported by the formal batch algorithm.

`fixed_steps` is retained as an explicitly selected engineering variant (and
in the bounded smoke configuration). Scratch reuses the same dilution function,
uses inner-product neighbor ranking, and skips backward/SGD for empty selection
while still updating label history. The historical `selectors/lend.py` module
now exposes `KNNLabelPropagationSelector` and `propagate_knn_labels`, not LEND
aliases: its softmax-cosine propagation is a separate, stateless algorithm.

The default validation split, validation-best checkpoint reporting and missing
five-run Best/Last-10 aggregate remain fidelity limits; this change does not
claim paper numerical reproduction or verified author-code parity.

## Fidelity policy

The paper explicitly supplies ResNet-18, 200 epochs, batch size 256, SGD with
learning rate 0.05, momentum 0.9 and weight decay 0.0005, a ten-fold learning
rate decrease after epoch 100, alpha 0.99, beta 0.9, and recommends k=8.

The smoke profile uses explicit engineering choices: k=15 for its batch size 16
to exercise the dense toy-graph path, gamma=1, inner-product neighbor ranking,
no feature L2 normalization, three fixed dilution rounds,
current-value first history initialization, batch-mean selected loss, skipped
empty updates, and disabled augmentation. These are not asserted to be paper
defaults.
Batch size affects the batch-local graph and therefore the algorithm itself.
The paper defines `W'=A^T A` and `W=D^-1/2 W' D^-1/2`, but does not specify
zero-degree behavior.  As an explicit toolbox boundary-handling choice, a
positive degree uses its ordinary inverse square root while a zero degree uses
inverse square root zero.  This adds no self-loop or epsilon, changes neither
the directed neighbor graph nor k, leaves the corresponding normalized row and
column zero, and leaves every positive-degree result unchanged.

## Unified commands

```powershell
lnl doctor
lnl list experiments --profile smoke
lnl papers show lend
lnl papers config lend --profile smoke --path-only
lnl validate --recipe cifar10-lend-smoke
lnl run --recipe cifar10-lend-smoke --dry-run
lnl run --recipe cifar10-lend-smoke
lnl resume <run-directory>
```

`--epochs N` overrides only `lend.training.epochs`. Resume validates the method,
graph, dilution, history and selection policies; model/optimizer/scheduler;
stable sample mapping; and noise manifest. A completed resume with the same
epoch target is a no-op. Increasing the epoch target is supported.

`python -m lnl_toolbox.cli.main ...` remains a developer-equivalent entrypoint;
the `lnl ...` commands above are the current recommended user entrypoint.

## Outputs and limits

Inspect `resolved_config.yaml`, `environment.json`, `noise_manifest.npz`,
`metrics.jsonl`, `last.pt`, `best.pt`, and `final_metrics.json`. Synthetic-noise
oracle selection metrics are diagnostics only and never control training or
checkpoint selection.

## Full-budget CIFAR-10 profile

The public `lend-cifar10-reproduction` recipe is a real CIFAR-10, single-seed,
full-budget **论文方法对齐（paper-oriented）工程流程**. It is not a paper-exact
numerical reproduction and has not yet been run to completion; 尚未完成数值复现验证.

Paper-specified or paper-recommended settings retained by this profile are:

- CIFAR-10 with symmetric 40% noise;
- ResNet-18 with its standard toolbox width of 64;
- 200 epochs and batch size 256;
- SGD with learning rate 0.05, momentum 0.9 and weight decay 0.0005;
- one ten-fold learning-rate decrease after epoch 100;
- dilution alpha 0.99, history momentum beta 0.9, and recommended k=8.

The following are explicit toolbox implementation choices rather than claimed
paper hyperparameters or protocol details:

- gamma=1.0 and ten fixed dilution steps (the paper does not provide an exact
  gamma value or executable convergence/iteration setting);
- inner-product neighbor ranking without feature normalization;
- a stratified 45,000/5,000 train/noisy-validation split;
- best-checkpoint selection by noisy-validation accuracy;
- transition-sampled synthetic noise, standard CIFAR augmentation, seed 1, and
  a single-run result rather than the paper's five-trial aggregate.

Equation (7) formally writes a sum over selected sample losses, but the paper
does not specify the executable mini-batch reduction.  The toolbox uses the
paper-compatible engineering choice
`sum(selected per-sample CE) / current_batch_size`.  The denominator is the
actual batch size, not the selected count.  This preserves the paper's batch
size 256 and learning rate 0.05 without amplifying the gradient scale by
approximately the batch size, while leaving graph construction, dilution,
history, and selection unchanged.  This reduction is not claimed to be a
paper-exact implementation detail.

Run the formal profile through the same unified lifecycle:

```powershell
lnl list experiments --profile reproduction
lnl validate --recipe lend-cifar10-reproduction
lnl run --recipe lend-cifar10-reproduction --dry-run
lnl run --recipe lend-cifar10-reproduction
lnl resume <run-directory>
```

Before a 200-epoch run, perform a separately authorized short GPU sanity at
batch size 256. Formal CIFAR-100, Animal-10N, additional symmetric/asymmetric
noise recipes, multi-seed aggregation, and reproduction of paper tables remain
pending. `paper-exact numerical reproduction = NOT VERIFIED`.
