# 标签噪声学习的代码实现 Baseline

在代码实现层面，需要区分两种 baseline：

1. **实验 baseline**：Standard Cross-Entropy（CE），几乎所有 LNL 论文都应与它比较；
2. **实现 baseline**：普通训练循环、逐样本状态、可信度估计和加权损失构成的通用程序骨架。

不存在一段可以原封不动覆盖所有 LNL 方法的训练代码，但大量方法共享以下数据流：

```text
模型产生预测
    ↓
根据预测、loss 或历史估计样本状态
    ↓
生成 weight / mask / corrected target
    ↓
计算逐样本 loss
    ↓
加权聚合并更新模型
```

## 1. Level 0：Standard CE

最基本的监督训练代码为：

```python
for epoch in range(num_epochs):
    model.train()

    for batch in train_loader:
        images = batch["image"].to(device)
        noisy_targets = batch["target"].to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, noisy_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Standard CE 直接拟合观测标签分布：

$$
P_\theta(\widetilde{Y}\mid X),
$$

而不是显式恢复真实标签后验：

$$
P(Y\mid X).
$$

虽然 CE 容易在训练后期记忆错误标签，但它是代码实现的零号基线。许多 LNL 方法都可以理解为在 CE 训练循环上修改某一部分：

- robust loss：替换交叉熵；
- sample selection：增加逐样本 mask；
- reweighting：增加逐样本 weight；
- label correction：替换训练 target；
- transition correction：在干净预测和观测标签之间加入转移过程；
- multi-network：增加模型与优化器；
- semi-supervised LNL：将数据划分为有监督与无监督分支。

## 2. Level 1：Sample-aware baseline

Sample-aware baseline 保留逐样本信息，再根据论文方法决定每个样本如何参与训练：

```python
for epoch in range(num_epochs):
    algorithm.on_epoch_start(
        model=model,
        sample_state=sample_state,
    )

    for batch in train_loader:
        images = batch["image"].to(device)
        noisy_targets = batch["target"].to(device)
        indices = batch["index"]

        logits = model(images)

        per_sample_loss = F.cross_entropy(
            logits,
            noisy_targets,
            reduction="none",
        )

        inference = algorithm.infer_sample_state(
            logits=logits,
            noisy_targets=noisy_targets,
            per_sample_loss=per_sample_loss,
            indices=indices,
            sample_state=sample_state,
        )

        final_targets = inference.targets
        sample_weights = inference.weights

        per_sample_loss = algorithm.compute_loss(
            logits=logits,
            targets=final_targets,
            inference=inference,
        )

        loss = reduce_weighted_loss(
            per_sample_loss,
            sample_weights,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        algorithm.update_history(
            indices=indices,
            logits=logits.detach(),
            losses=per_sample_loss.detach(),
            inference=inference,
        )
```

它对应以下程序结构：

```text
Dataset(index)
→ Model
→ Per-sample Loss
→ Selector / Reweighter / Corrector
→ Weighted Reduction
→ Optimizer
→ Sample History
```

## 3. 最重要的数据契约

每个训练样本必须返回稳定的全局索引：

```python
batch = {
    "image": image,
    "target": noisy_target,
    "index": global_index,
}
```

合成噪声实验还可以提供只用于评估的真值字段：

```python
batch = {
    "image": image,
    "target": noisy_target,
    "index": global_index,

    # 仅供 evaluator 使用，训练算法不得访问
    "clean_target": clean_target,
    "is_clean": is_clean,
}
```

全局索引用于读写逐样本状态：

```python
loss_history[index]
clean_probability[index]
selected_mask[index]
corrected_target[index]
candidate_classes[index]
eta[index]
prediction_history[index]
```

不能用 `batch_idx` 代替 `global_index`。当 DataLoader 使用 shuffle 时，batch 的位置不再对应固定样本。

## 4. 统一的逐样本推断结果

不同论文估计的概率对象不同，因此统一接口不能只允许返回“干净/噪声”二值结果：

```python
from dataclasses import dataclass


@dataclass
class SampleInference:
    # 样本为干净样本的连续概率
    clean_probability: Tensor | None = None

    # 是否选中样本
    selected_mask: Tensor | None = None

    # 样本对训练目标的贡献权重
    sample_weight: Tensor | None = None

    # 修正后的硬标签
    corrected_target: Tensor | None = None

    # 修正后的软标签分布
    target_distribution: Tensor | None = None

    # CCS 等方法使用的候选类别
    candidate_classes: Tensor | None = None

    # 供日志和 evaluator 使用
    metadata: dict | None = None
```

不同方法主要填充不同字段：

| 方法 | 主要输出或状态 |
|---|---|
| CE | 所有样本的 `sample_weight=1` |
| GCE | 通常不需要样本推断，只替换 loss |
| DSS BASE | `selected_mask` |
| MDA | 调整后的选择概率或预测分布 |
| CCS | `candidate_classes` |
| Co-teaching | 两组交叉选择索引 |
| DivideMix | `clean_probability`、软标签 |
| UPM | `eta`、真实标签后验分布 |
| Reweighting | `sample_weight` |
| Label correction | `corrected_target` 或软标签分布 |

## 5. Selector baseline

Sample-selection 方法可以共享一个最小 selector 接口：

```python
class SampleSelector:
    def update(
        self,
        *,
        indices,
        logits,
        noisy_targets,
        per_sample_loss,
        history,
    ):
        ...

    def select(self, indices):
        ...
```

### 5.1 DSS BASE selector

```python
class PredictionAgreementSelector:
    @torch.no_grad()
    def update(
        self,
        *,
        indices,
        logits,
        noisy_targets,
        **kwargs,
    ):
        predictions = logits.argmax(dim=1)

        self.selected_mask[indices] = (
            predictions == noisy_targets
        ).cpu()

    def select(self, indices):
        return self.selected_mask[indices]
```

### 5.2 Small-loss selector

```python
class SmallLossSelector:
    def select(self, per_sample_loss, keep_rate):
        count = max(
            1,
            int(len(per_sample_loss) * keep_rate),
        )

        selected_indices = torch.argsort(
            per_sample_loss.detach()
        )[:count]

        mask = torch.zeros(
            len(per_sample_loss),
            dtype=torch.bool,
            device=per_sample_loss.device,
        )

        mask[selected_indices] = True
        return mask
```

### 5.3 概率式 selector

概率式 selector 不返回硬 mask，而是返回连续的 clean probability：

```python
class ProbabilisticSelector:
    def select(self, losses):
        clean_probability = fit_loss_mixture(losses)
        return clean_probability
```

因此，toolbox 的 selector 输出不能被限制为布尔类型。

## 6. Per-sample loss baseline

LNL 实现中的一个重要原则是：先计算逐样本损失，再决定如何聚合。

```python
per_sample_loss = F.cross_entropy(
    logits,
    targets,
    reduction="none",
)
```

如果过早使用默认均值归约：

```python
loss = F.cross_entropy(logits, targets)
```

就会丢失逐样本信息，无法继续完成：

- 小损失选样；
- loss distribution 拟合；
- 样本重加权；
- loss history 记录；
- clean/noisy detection 指标计算。

统一的损失聚合函数可以写成：

```python
def reduce_weighted_loss(
    per_sample_loss,
    sample_weight=None,
    normalization="batch",
):
    if sample_weight is None:
        return per_sample_loss.mean()

    weighted = per_sample_loss * sample_weight

    if normalization == "batch":
        return weighted.sum() / len(per_sample_loss)

    if normalization == "selected":
        denominator = sample_weight.sum().clamp_min(1.0)
        return weighted.sum() / denominator

    raise ValueError(normalization)
```

两种归一化方式具有不同含义：

- `batch`：除以原始 batch size；选择率降低时，梯度整体也会减小；
- `selected`：除以权重之和；不论选择率如何，选中样本的平均梯度规模相对稳定。

阅读论文和官方代码时，必须核对其实际采用哪一种方式。

## 7. Sample history baseline

许多 LNL 方法依赖训练历史，因此需要逐样本状态存储：

```python
@dataclass
class SampleState:
    latest_loss: Tensor             # [N]
    loss_ema: Tensor                # [N]
    clean_probability: Tensor       # [N]
    selected_mask: Tensor           # [N]
    latest_prediction: Tensor       # [N, K]
    corrected_targets: Tensor       # [N]
```

具体算法可以维护私有状态，而不必把所有字段加入公共核心。

### 7.1 DSS 状态

```python
@dataclass
class DSSState:
    marginal_ema: Tensor            # [K]
    adjusted_predictions: Tensor    # [N, K]
    candidate_mask: Tensor          # [N, K]
    confidence_history: Tensor      # [N, K, T]
```

### 7.2 UPM 状态

```python
@dataclass
class UPMState:
    eta: Tensor                     # [N]
    psi: Tensor                     # [N, K]
    posterior_q: Tensor             # [N, K]
```

### 7.3 DivideMix 状态

```python
@dataclass
class DivideMixState:
    loss_history: Tensor            # [N, T]
    clean_probability: Tensor       # [N]
    pseudo_targets: Tensor          # [N, K]
```

公共框架负责提供状态保存与 checkpoint 能力，算法负责解释私有状态的含义。

## 8. Level 2：Algorithm-owned training

复杂 LNL 方法经常包含完整的多阶段生命周期：

```text
warm-up
→ 全数据统计
→ 样本划分
→ 当前 epoch 训练
→ 历史更新
→ checkpoint
```

因此，复杂算法不能被简化成单个 `loss_fn`：

```python
class LNLAlgorithm:
    def setup(self, context):
        ...

    def on_epoch_start(self, state):
        # 更新 selector、拟合 GMM、计算全数据 loss 等
        ...

    def train_step(self, batch, state):
        ...

    def on_epoch_end(self, state):
        # 更新逐样本历史和统计指标
        ...

    def state_dict(self):
        ...

    def load_state_dict(self, checkpoint):
        ...
```

Runner 只负责生命周期调度：

```python
for epoch in range(num_epochs):
    algorithm.on_epoch_start(state)

    for batch in train_loader:
        result = algorithm.train_step(batch, state)

    algorithm.on_epoch_end(state)
```

Runner 不应理解：

- 什么是 clean sample；
- 算法有几个模型或优化器；
- 当前使用 CE、GMM 还是其他方法；
- 是否存在候选类别；
- 当前处于 warm-up、EM 还是半监督阶段。

这些都属于具体 Algorithm。

## 9. 不同论文修改 baseline 的位置

将 Standard CE 写成最简形式：

```python
logits = model(images)
targets = noisy_targets
weights = torch.ones(batch_size)
loss = CE(logits, targets)
```

不同方法主要修改以下位置。

### 9.1 Robust loss

```python
loss = GCE(logits, noisy_targets)
```

只替换损失函数。

### 9.2 Sample selection

```python
weights = selector(logits, losses, history)
loss = weights * CE(logits, noisy_targets)
```

增加样本选择或权重估计。

### 9.3 Label correction

```python
targets = corrector(
    noisy_targets,
    predictions,
    history,
)

loss = CE(logits, targets)
```

修改训练目标。

### 9.4 Transition correction

```python
clean_probabilities = softmax(logits)
noisy_probabilities = clean_probabilities @ transition_matrix
loss = NLL(noisy_probabilities, noisy_targets)
```

在干净预测和观测标签之间加入噪声转移过程。

### 9.5 Co-teaching

```python
logits_a = model_a(images)
logits_b = model_b(images)

selected_by_a = small_loss(loss_a)
selected_by_b = small_loss(loss_b)

update_a(loss_a[selected_by_b])
update_b(loss_b[selected_by_a])
```

增加第二个模型、第二个优化器和交叉选样。

### 9.6 DivideMix

```python
clean_probability = fit_gmm(all_sample_losses)

labeled, unlabeled = split_dataset(
    clean_probability
)

loss = supervised_loss + unsupervised_loss
```

增加全数据统计、数据划分与半监督训练阶段。

## 10. 三级实现 baseline

### Level 0：Standard supervised baseline

```text
Dataset → Model → Per-sample Loss → Mean → Optimizer
```

适用于 CE、GCE 和普通鲁棒损失。

### Level 1：Sample-aware baseline

```text
Dataset(index)
→ Model
→ Per-sample Loss
→ Selector / Reweighter
→ Weighted Reduction
→ Optimizer
→ Sample History
```

适用于 DSS BASE、small-loss、reweighting 和简单 label correction。

### Level 2：Algorithm-owned training

```text
Runner
→ Algorithm
   ├── model(s)
   ├── optimizer(s)
   ├── sample state
   ├── phase state
   ├── selector / corrector
   └── train_step
```

适用于 Co-teaching、JoCoR、UPM、DivideMix 等复杂方法。

## 11. 统一损失形式

大量 LNL 方法可以抽象为：

$$
\boxed{
L
=
\frac{1}{B}
\sum_{i=1}^{B}
w_i\,
\ell\!\left(
f_\theta(x_i),q_i
\right)
}.
$$

其中：

- $w_i$：样本权重，可以是 $0/1$ mask 或连续 clean probability；
- $q_i$：训练目标，可以是 noisy label、修正标签或软标签分布；
- $\ell$：CE、GCE、CCS loss 或其他损失；
- $f_\theta$：一个或多个模型构成的预测过程。

从代码实现角度阅读一篇 LNL 论文时，可以固定询问：

1. $w_i$ 如何得到？
2. $q_i$ 如何得到？
3. $\ell$ 如何计算和归约？
4. 算法需要保存哪些逐样本历史？
5. 这些状态在 batch、epoch 还是训练阶段边界更新？

DSS 的 BASE 只是该结构中的一个具体 selector，而上述结构才是代码实现层面更通用的 baseline。
