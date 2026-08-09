# LEND reproduction and maintenance

The toolbox implements a **paper-oriented LEND workflow**. It is not described
as paper-exact, released-code exact, or a numerical reproduction because no
verifiable author implementation was found and the paper leaves several runtime
choices unspecified.

## Implemented flow

```text
current noisy mini-batch
-> one model forward producing logits and penultimate embeddings
-> directed batch-local kNN and Eq. (1) adjacency
-> A^T A and symmetric degree normalization
-> fixed-step Eq. (3) label dilution from noisy one-hot labels
-> stable-index Eq. (5) cross-epoch history
-> Eq. (6) agreement selection
-> summed selected noisy-label CE
```

Diluted vectors are detached selection evidence, not probability targets. Empty
selected batches commit the successfully observed label history but skip
backward and the optimizer step.

## Fidelity policy

The paper explicitly supplies ResNet-18, 200 epochs, batch size 256, SGD with
learning rate 0.05, momentum 0.9 and weight decay 0.0005, a ten-fold learning
rate decrease after epoch 100, alpha 0.99, beta 0.9, and recommends k=8.

The smoke profile uses explicit engineering choices: k=15 for its batch size 16
to avoid an undefined zero-degree toy graph, gamma=1, inner-product neighbor
ranking, no feature L2 normalization, three fixed dilution rounds,
current-value first history initialization, summed selected loss, skipped empty
updates, and disabled augmentation. These are not asserted to be paper defaults.
Batch size affects the batch-local graph and therefore the algorithm itself.
Formal use of smaller k retains the strict degree-zero failure contract and
therefore requires a short data/model sanity run before long training.

## Unified commands

```powershell
python -m lnl_toolbox.cli.main doctor
python -m lnl_toolbox.cli.main list experiments --profile smoke
python -m lnl_toolbox.cli.main papers show lend
python -m lnl_toolbox.cli.main papers config lend --profile smoke --path-only
python -m lnl_toolbox.cli.main validate --recipe cifar10-lend-smoke
python -m lnl_toolbox.cli.main run --recipe cifar10-lend-smoke --dry-run
python -m lnl_toolbox.cli.main run --recipe cifar10-lend-smoke
python -m lnl_toolbox.cli.main resume <run-directory>
```

`--epochs N` overrides only `lend.training.epochs`. Resume validates the method,
graph, dilution, history and selection policies; model/optimizer/scheduler;
stable sample mapping; and noise manifest. A completed resume with the same
epoch target is a no-op. Increasing the epoch target is supported.

## Outputs and limits

Inspect `resolved_config.yaml`, `environment.json`, `noise_manifest.npz`,
`metrics.jsonl`, `last.pt`, `best.pt`, and `final_metrics.json`. Synthetic-noise
oracle selection metrics are diagnostics only and never control training or
checkpoint selection.

The bundled configuration is a tiny workflow smoke. Formal CIFAR-10/CIFAR-100
training, multi-seed aggregation, and reproduction of paper tables remain
pending.
