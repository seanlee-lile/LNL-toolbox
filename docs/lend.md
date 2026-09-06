# LEND method contract

LEND is implemented as a stateful single-model Algorithm because its online
batch graph consumes current embeddings and its `[N,C]` diluted-label history is
owned across epochs. It is intentionally not forced into the stateless Selector
contract.

The current contract is:

- `method: lend`, `execution.runner: lend`;
- one feature-aware forward per batch;
- detached directed kNN adjacency and exact `A.T @ A` symmetric normalization;
- detached fixed-step label dilution without row normalization;
- stable-index history with current-value first observation;
- hard agreement selection and summed noisy-label CE;
- empty selections skip parameter updates but commit observed history;
- epoch-seeded loaders and strict checkpoint/resume.

This is a 论文方法对齐（paper-oriented）工程流程（engineering workflow）; it is
not a claim that numerical reproduction has been verified. For the current
recommended `lnl` commands and fidelity boundaries, see the
[LEND reproduction guide](../papers/lend/reproduction.md).
