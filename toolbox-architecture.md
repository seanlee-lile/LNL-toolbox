# 标签噪声学习 Toolbox：Baseline 与整体架构

> 架构调整（2026-07）：框架核心改为任务无关的运行协议。本文后续的 CIFAR、噪声生成、CE/GCE、Co-teaching 等内容是 LNL 参考插件和实验路线，不是核心层的必选组成。

## 0. 通用核心边界

```text
Application / Experiment Definition
                │
                ▼
┌──────────────────────────────────────────────┐
│ Generic Core                                 │
│ Batch · ExperimentContext · Algorithm        │
│ RunState · StepResult · Evaluator · Storage  │
└──────────────────────────────────────────────┘
                │ capability discovery
                ▼
┌──────────────────────────────────────────────┐
│ PluginCatalog                                │
│ dataset · model · algorithm · evaluator ...  │
└──────────────────────────────────────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
  LNL built-ins      future plugins
  CE/GCE/noise       text/graph/SSL/etc.
```

核心层遵循以下约束：

1. `Batch.payload` 是不透明对象，不强制包含 `image`、`target` 或 `index`；这些字段由具体任务插件定义。
2. `Algorithm.step` 自主管理模型、优化器和私有状态；Runner 不理解梯度、标签、噪声或网络数量。
3. `cycle` 可以代表 epoch、EM 轮次、主动学习轮次、扩散阶段等，不在核心中固定语义。
4. `Evaluator` 只消费统一的 `StepResult`；accuracy、AUROC、clean precision 等均为插件。
5. 配置系统通过 adapter 生成 `ExperimentContext.config`，核心不绑定 YAML 或 Hydra。
6. checkpoint 只组合框架 `RunState` 与实现了 `Stateful` 协议的组件状态。

当前 `noise/`、`losses/` 和 `algorithms/coteaching.py` 保留为 LNL 示例实现，并通过 `plugins/builtin/` 注册。它们用于验证扩展能力，不构成框架最小安装条件。

## 1. 第一版目标

第一版只聚焦**图像多分类**，目标不是收集尽可能多的论文实现，而是建立一个公平、可复现、方便新增算法的实验框架：

1. 同一数据集、网络、数据增强和训练预算下比较不同方法；
2. 支持合成噪声与真实噪声；
3. 保存每个样本的噪声信息，支持样本选择、标签修正和实例依赖噪声算法；
4. 一条命令运行实验，并保存配置、日志、模型和逐样本结果；
5. 新增一个普通单网络算法时，原则上只需增加一个算法类和一个 YAML 配置。

首版建议支持：

- 数据集：CIFAR-10（必做）、CIFAR-100（推荐）；
- 噪声：symmetric、asymmetric/pairflip、instance-dependent；
- 模型：PreActResNet-18 或 ResNet-18，所有方法保持一致；
- 方法：CE、GCE；随后增加 Co-teaching、CORES²；
- 高级方法：DivideMix 放在第二阶段，不作为第一版完成条件。

## 2. 推荐 Baseline

### 2.1 主基线：Standard CE

Standard CE 是必须保留的对照组。它不做样本过滤或标签修正，直接使用噪声标签训练：

\[
L_{CE}=-\frac{1}{B}\sum_{i=1}^{B}\log p_\theta(\tilde y_i\mid x_i)
\]

建议默认配置：

| 项目 | 默认值 |
|---|---|
| dataset | CIFAR-10 |
| backbone | PreActResNet-18 |
| optimizer | SGD, momentum=0.9, weight_decay=5e-4 |
| epochs | 200 |
| batch size | 128 |
| learning rate | 0.1 + cosine decay |
| augmentation | random crop + horizontal flip + normalize |
| seeds | 1, 2, 3 |
| noise type | symmetric / asymmetric / instance-dependent |
| noise rate | 0.2 / 0.4 / 0.6 |
| selection model | 禁止使用噪声测试集选最优模型 |

报告 `mean ± std`，至少包含：

- `test/accuracy_last`：最后若干轮平均测试准确率；
- `test/accuracy_best`：按干净验证集选择的最佳结果；
- `train/clean_accuracy`：仅在合成噪声诊断时计算；
- `noise/realized_rate`：实际翻转比例，而不是只写请求比例；
- 训练时间、参数量和随机种子。

### 2.2 最小鲁棒基线：GCE

GCE 只替换损失函数，能验证 toolbox 的“loss 型算法”接口是否合理：

\[
L_{GCE}=\frac{1-p_\theta(\tilde y\mid x)^q}{q},\quad q\in(0,1]
\]

建议先使用 `q=0.7`，其他设置与 CE 完全相同。CE 与 GCE 构成最小可交付 baseline；两者跑通后再接入复杂方法。

### 2.3 用于验证架构扩展性的代表方法

| 类别 | 方法 | 验证的框架能力 | 首版优先级 |
|---|---|---|---|
| 普通训练 | CE | 单模型、单优化器 | P0 |
| 鲁棒损失 | GCE | 可插拔 loss | P0 |
| 样本选择 | Co-teaching | 双模型、双优化器、交叉选样 | P1 |
| IDN 方法 | CORES² | 逐样本筛选、未知噪声率 | P1 |
| 半监督式 LNL | DivideMix | warm-up、GMM、双网络、多视图 | P2 |
| 数据诊断 | Cleanlab adapter | 离线概率、标签质量排序 | P2 |

不要一开始同时复现十几个算法。只要 CE、GCE、Co-teaching 和 CORES² 共用同一数据及评测管线，已经足以证明架构覆盖了普通训练、鲁棒损失、多网络训练和实例依赖噪声四类需求。

## 3. 最重要的数据协议

每个训练样本必须返回稳定的全局索引：

```python
{
    "image": Tensor,
    "target": int,          # 当前训练使用的 noisy target
    "index": int,           # 数据集全局 index，shuffle 后仍不变
    "clean_target": int,    # 仅合成噪声的评测/诊断使用
    "is_clean": bool,       # 仅评测/诊断使用
}
```

算法训练时不得读取 `clean_target` 和 `is_clean`。它们只交给 evaluator，以免产生不公平的“真值泄漏”。真实噪声数据集可以把这两个字段设为 `None`。

你当前草稿中的：

```python
eta[batch_idx : batch_idx + len(images)]
```

在 `DataLoader(shuffle=True)` 时是错误的。必须改成：

```python
eta[index]
```

否则样本级 `eta`、历史预测、loss EMA 和 clean probability 都会写到错误样本上。

## 4. 噪声清单（Noise Manifest）

噪声不应该在每次 `__getitem__` 时随机生成。首次生成后保存清单，后续所有算法读取同一份清单：

```text
artifacts/noise/cifar10/idn/rate_0.4/seed_1.npz
```

清单至少保存：

```python
{
    "version": "1.0",
    "dataset": "cifar10",
    "dataset_fingerprint": "...",
    "noise_type": "instance_dependent",
    "seed": 1,
    "requested_rate": 0.4,
    "realized_rate": 0.3978,
    "clean_targets": ndarray[N],
    "noisy_targets": ndarray[N],
    "flip_mask": ndarray[N],
    "transition_matrix": ndarray[C, C] | None,
}
```

对于实例依赖噪声，额外保存生成器名称、参数以及每样本转移概率或其可重建信息。加载时校验数据集长度和 fingerprint，避免清单错配。

## 5. 推荐目录结构

```text
LNL-toolbox/
├── configs/
│   ├── config.yaml
│   ├── dataset/            # cifar10.yaml, cifar100.yaml
│   ├── noise/              # clean, symmetric, asymmetric, idn
│   ├── model/              # preact_resnet18.yaml
│   ├── algorithm/          # ce, gce, coteaching, cores2
│   └── experiment/         # 可复现实验组合
├── src/lnl_toolbox/
│   ├── cli/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── make_noise.py
│   ├── data/
│   │   ├── datasets.py
│   │   ├── noisy_dataset.py
│   │   ├── transforms.py
│   │   └── samplers.py
│   ├── noise/
│   │   ├── base.py
│   │   ├── symmetric.py
│   │   ├── asymmetric.py
│   │   ├── instance_dependent.py
│   │   └── manifest.py
│   ├── models/
│   │   ├── factory.py
│   │   └── preact_resnet.py
│   ├── losses/
│   │   ├── cross_entropy.py
│   │   └── gce.py
│   ├── algorithms/
│   │   ├── base.py
│   │   ├── standard.py
│   │   ├── coteaching.py
│   │   ├── cores2.py
│   │   └── dividemix.py
│   ├── engine/
│   │   ├── runner.py
│   │   ├── state.py
│   │   ├── checkpoint.py
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── classification.py
│   │   ├── noise_detection.py
│   │   └── reporter.py
│   ├── registry.py
│   └── utils/              # seed、logging、distributed 等
├── tests/
│   ├── test_noise.py
│   ├── test_dataset_index.py
│   ├── test_losses.py
│   └── test_smoke_train.py
├── scripts/
│   └── benchmark_cifar10.ps1
├── artifacts/              # gitignore：noise、runs、checkpoints
├── pyproject.toml
└── README.md
```

## 6. 核心分层

```text
Hydra config / CLI
        │
        ▼
Experiment Builder ── Registry
        │
        ├── Dataset ── Noise Generator / Manifest
        ├── Model(s)
        ├── Algorithm ── Loss / Selector / Label Corrector
        └── Runner ── Evaluator / Logger / Checkpoint
```

各层职责：

- **Dataset**：只负责读取数据、增强、返回全局 index 和标签元信息；
- **Noise**：生成或加载噪声，绝不包含训练逻辑；
- **Model**：只定义网络，不感知噪声类型；
- **Algorithm**：拥有模型、优化器和样本级状态，实现论文训练逻辑；
- **Runner**：只控制 epoch、设备、AMP、断点恢复、日志和回调；
- **Evaluator**：分类性能与噪声识别性能分开计算；
- **Registry/Config**：通过名称组合组件，避免在 `train.py` 中不断增加 `if/elif`。

## 7. 算法接口

不要把 Runner 写死成“一个模型 + 一个 loss + 一个 optimizer”。建议让算法拥有训练细节：

```python
class LNLAlgorithm(Protocol):
    def setup(self, context) -> None: ...
    def train_step(self, batch, state) -> dict[str, Tensor]: ...
    @torch.no_grad()
    def eval_step(self, batch, state) -> dict[str, Tensor]: ...
    def on_epoch_start(self, state) -> None: ...
    def on_epoch_end(self, state) -> dict[str, float]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
```

这样：

- CE/GCE 的 `train_step` 是单模型训练；
- Co-teaching 可以维护两个模型和两个 optimizer；
- DivideMix 可以在 `on_epoch_start` 计算全数据 loss、拟合 GMM 并重建 sampler；
- 你草稿中的 UPM 可以维护 `eta[N]`、`psi[N,C]`，并用 `batch["index"]` 安全读写。

Runner 不理解具体论文公式，只负责生命周期。

## 8. 配置示例

```yaml
# configs/experiment/cifar10_idn_ce.yaml
defaults:
  - /dataset: cifar10
  - /noise: instance_dependent
  - /model: preact_resnet18
  - /algorithm: ce
  - _self_

seed: 1
trainer:
  epochs: 200
  device: cuda
  amp: true
  deterministic: true

noise:
  rate: 0.4
  seed: ${seed}

output_dir: artifacts/runs/${dataset.name}/${noise.name}/${algorithm.name}/seed_${seed}
```

命令行形式：

```bash
python -m lnl_toolbox.cli.make_noise +experiment=cifar10_idn_ce
python -m lnl_toolbox.cli.train +experiment=cifar10_idn_ce algorithm=gce algorithm.q=0.7
python -m lnl_toolbox.cli.evaluate run_dir=artifacts/runs/...
```

## 9. 评测协议

### 分类指标

- Top-1 accuracy；
- 最佳 epoch、最后 10 个 epoch 的均值；
- macro-F1（类别不平衡或真实数据集时）；
- ECE/NLL（如果 toolbox 声称支持概率质量）。

### 噪声识别指标

仅当算法输出 `clean_score`、`noise_score` 或选样 mask 时计算：

- AUROC / AUPRC；
- clean sample precision、recall；
- 每个 epoch 的 selected ratio；
- label correction accuracy。

### 公平性规则

1. 同组实验共享 noise manifest、backbone、增强、epoch 和随机种子；
2. 训练过程不访问合成噪声的干净标签；
3. 测试集永远不注入噪声；
4. 合成噪声同时报告请求噪声率和实际噪声率；
5. 至少 3 个随机种子；正式论文结果建议 5 个；
6. 保存完整 resolved config、代码版本、环境版本和 checkpoint；
7. 区分“按验证集选出的 best”与“训练结束的 last”，不要只报最高测试准确率。

## 10. 最小测试集

首版至少写以下自动化测试：

1. `noise_rate=0` 时标签完全不变；
2. symmetric noise 不会把标签翻转回原类别；
3. 同一 seed 生成完全相同的 manifest；
4. 不同 seed 通常生成不同 flip mask；
5. shuffle 后 `index -> noisy_target` 映射保持正确；
6. GCE 在 `q -> 0` 时数值上接近 CE；
7. 断点恢复后样本级状态（如 `eta`、loss history）不丢失；
8. 用一个极小假数据集完成 2 个 epoch 的 smoke test。

## 11. 实施顺序

### M0：可运行骨架

- `pyproject.toml`、配置系统、registry；
- CIFAR-10、ResNet、Runner；
- CE 在 clean CIFAR-10 上跑通；
- checkpoint、CSV/JSON 日志和 seed 控制。

### M1：噪声基准

- noise manifest；
- symmetric、asymmetric、instance-dependent；
- CE + GCE；
- 3 seeds benchmark 与汇总表。

### M2：验证复杂算法接口

- Co-teaching；
- CORES²；
- noise detection metrics；
- 与论文官方实现做小规模结果核对。

### M3：高级与真实噪声

- DivideMix；
- CIFAR-10N、Clothing1M 等真实噪声数据；
- Cleanlab adapter；
- 文档、示例 notebook、CI。

## 12. 验收标准

第一版完成的判断标准应是：

```text
同一份 CIFAR-10 noise manifest
    ├── CE（单网络）
    ├── GCE（鲁棒 loss）
    ├── Co-teaching（双网络选样）
    └── CORES²（IDN 样本筛选）

均可通过一条配置命令运行、断点恢复，并自动生成 mean ± std 汇总表。
```

做到这里，toolbox 的核心价值已经成立。之后新增论文方法只是扩展 `algorithms/`，而不需要复制数据加载、日志、评测和噪声生成代码。

## 参考项目与论文

- [cleanlab](https://github.com/cleanlab/cleanlab)：数据质量诊断、标签问题识别以及模型无关 API 的参考；
- [DivideMix](https://github.com/LiJunnan1992/DivideMix)：双网络、多阶段 noisy-label 训练流程的参考；
- [Co-teaching](https://github.com/bhanML/Co-teaching)：双网络交叉选样算法的官方实现；
- [CORES²](https://github.com/UCSC-REAL/cores)：instance-dependent label noise 的样本筛选方法；
- [GCE](https://papers.nips.cc/paper_files/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html)：可插拔鲁棒损失基线；
- [Hydra experiment configuration](https://hydra.cc/docs/patterns/configuring_experiments/)：实验配置组合方式；
- [torchvision datasets](https://docs.pytorch.org/vision/stable/datasets)：统一 Dataset/DataLoader 接口的参考。
