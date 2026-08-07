# APL（NCE+RCE）CIFAR-10 单次复现计划

## 目标与复现边界

复现 Normalized Loss Functions / APL（ICML 2020）中的 CIFAR-10、对称噪声率 `0.2`、NCE+RCE 设置。只运行 seed `1` 一次，论文表 2 的 `89.22 ± 0.27%` 只作为中心值参照，不宣称复现均值或标准差。

采用作者代码兼容评测：完整 50,000 个 noisy training 样本训练，每个 epoch 在 clean test 上评测并选择最高准确率。所有产物必须明确记录 `test_selection_leakage: true`，该协议不得成为 Toolbox 默认值。

## 原料复用与必要增量

现有原料直接复用：

- CIFAR Dataset、global index、Noise Manifest 和 checkpoint；
- NCE、RCE、APL 及统一的 `Loss(logits, targets) -> [B]`；
- SGD、cosine scheduler、监督 runner、metrics 和训练曲线。

只补充以下通用原料：

1. 必要新增 `src/lnl_toolbox/models/cifar_cnn.py`，实现通用 `CifarCnn8`：
   - 6 个 Conv-BN-ReLU，通道为 `64,64,128,128,196,196`；
   - 三次 max-pooling；
   - `4×4×196 -> 256 -> num_classes` 两个全连接层；
   - Conv 使用 Kaiming uniform，Linear 使用 Xavier uniform。
   `tiny_cnn.py` 保持不变；模型配置名为 `cifar_cnn8`，不得使用 APL 专属名称。
2. 扩展唯一 symmetric generator，增加 `sampling: per_class`；每类分别翻转 20%，默认 global sampling 不变。
3. 扩展 `StandardUpdatePolicy`，增加默认关闭的 `max_grad_norm`；APL 配置使用 `5.0`。
4. 在唯一监督 runner 中增加通用、显式授权的 model-selection split；默认 validation 行为不变。

不得新增 `apl_loss.py`、`apl_dataset.py`、`apl_algorithm.py`、`apl_runner.py` 或 APL 专属 checkpoint/evaluator。

## 固定配置

```yaml
seed: 1
data:
  name: cifar10
  root: data/cifar10
  validation_size: 0
  augment: true
  preprocessing: standard
noise:
  name: symmetric
  rate: 0.2
  seed: 1
  sampling: per_class
loss:
  name: apl
  alpha: 1.0
  beta: 1.0
  active: {name: nce}
  passive:
    name: rce
    log_zero: -9.210340371976184
model:
  name: cifar_cnn8
optimizer:
  name: sgd
  lr: 0.01
  momentum: 0.9
  weight_decay: 0.0001
  nesterov: false
parameter_update:
  name: standard
  max_grad_norm: 5.0
scheduler:
  name: cosine
  t_max: 120
loader:
  batch_size: 128
  num_workers: 8
  pin_memory: true
trainer:
  epochs: 120
  device: auto
  progress:
    enabled: true
    curves: true
evaluation:
  selection_split: test
  allow_test_selection: true
  loss: {name: ce}
```

新增两个必要菜谱：

- `configs/experiment/apl_cifar10_noise02_smoke.yaml`
- `configs/experiment/apl_cifar10_noise02_reproduction.yaml`

## 公共接口与兼容要求

- `evaluation.selection_split` 默认为 `validation`；选择 `test` 时必须同时设置 `allow_test_selection: true`。
- 指标使用 `selection_split`、`selection_loss`、`selection_accuracy` 和 `best_selection_accuracy`；默认模式继续保留已有 `validation_*` 输出。
- checkpoint 保存版本化的 selection split、best epoch 和 best selection metric；旧 checkpoint 按 validation selection 安全加载。
- RCE 使用 `log_zero=ln(1e-4)` 对齐作者代码的 one-hot clamp，不修改 RCE 公共公式。
- stdout 只保存逐 epoch JSON 和 final JSON；非交互环境不写批次进度条。
- 通用能力实际实现后，在 `paper-implementation-guideline.md` 的重叠能力主索引登记唯一位置和后续使用论文；计划阶段不得提前写成“已有”。

## 验证与验收

1. 现有 NCE、RCE、APL 数学与 `[B]` 协议测试继续通过。
2. 验证 CifarCnn8 的层数、通道、初始化和 `[B,10]` 输出。
3. 验证 per-class symmetric noise 每类精确翻转 20%、不保留原标签、seed/hash 可复现。
4. 验证 gradient clipping 与 PyTorch 手算一致，关闭时更新行为不变。
5. 验证 test selection 必须显式授权，训练使用完整 50k，clean target 不进入训练 batch。
6. 验证 checkpoint/resume 保持 manifest、selection split、optimizer、scheduler 和 global step。
7. 依次运行 focused tests、完整 unittest、CUDA 2-epoch smoke 和 resume。
8. 正式运行 120 epochs，产物保存到 `artifacts/reproductions/apl-nce-rce-cifar10-noise02-seed1/`。
9. 记录 best epoch、best clean-test accuracy、与 89.22% 的差值、显存、耗时、manifest hash 和实现差异。偏差超过 2 个百分点时标记待诊断，不根据 test 结果事后调参。

## Git 与协作边界

- 实施前先单独审阅并收口当前未提交的 GCE 复现工作，不把 `merge_plan2.md` 纳入提交。
- APL 实施不得修改 Selector、CDR、TransitionEstimator、taxonomy 研究结论或同事模块。
- 数据、checkpoint、日志和运行产物不得提交。
- 修改、commit 和 push 分别遵守独立授权边界。
