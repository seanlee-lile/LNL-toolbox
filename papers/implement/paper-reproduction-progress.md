# 论文复现进度与参数台账

> 归档位置与维护规则见本目录的 `README.md`。

本文件是论文复现的唯一进度入口。`paper-implementation-guideline.md` 说明算法应如何实现；本文件只记录“是否已经实际运行”、使用了哪些参数、产物在哪里以及结果与原文相差多少。

状态定义：

- `未开始`：只有论文阅读或接口规划；
- `组件完成`：核心公式或组件通过测试，但没有完整论文实验；
- `Smoke 通过`：论文配置的小样本闭环通过；
- `单次复现`：至少完成一次论文规模实验；
- `完整复现`：覆盖论文要求的重复次数、设置和汇总指标。

## 总表

| # | 论文/方法 | 当前组件 | 复现状态 | 下一缺口 |
|---:|---|---|---|---|
| 1 | UPM | 指南 | 未开始 | InstanceNoiseModel、PosteriorRefiner、Pipeline |
| 2 | CAL | 指南 | 未开始 | StatisticEstimator、RiskCorrector |
| 3 | PDL | 指南 | 未开始 | 实例级 `T(x)` 与 Pipeline |
| 4 | JoCoR | Selector 基础 | 未开始 | 双网络 Algorithm |
| 5 | DSS | 指南 | 未开始 | 有状态可靠性估计与 Selector |
| 6 | CDR | ParameterUpdatePolicy | 组件完成 | noisy-validation early stopping 与论文实验 |
| 7 | CNLCU | Selector 基础 | 未开始 | loss history、不确定性与双网络 |
| 8 | MentorNet | WeightProvider 基础 | 未开始 | Mentor/Student Pipeline |
| 9 | Co-teaching | small-loss/交换 helper | 未开始 | 双网络训练与恢复 |
| 10 | Loss Correction | Anchor estimator | 组件完成 | Forward/Backward RiskCorrector |
| 11 | Normalized Loss/APL | NCE、MAE、RCE、APL | 组件完成 | 按 `apl/plan.md` 完成单次复现 |
| 12 | GCE | 标准 GCE | 单次复现 | 后续如需完整复现，再补 5 次重复与其他噪声设置 |
| 13 | VolMinNet | Transition 基础 | 未开始 | 可训练 NoiseModel |
| 14 | Natarajan Risk | Weight/Risk 基础 | 未开始 | 二分类 RiskCorrector |
| 15 | T-Revision | Anchor/Weight 基础 | 未开始 | 三阶段 Pipeline |
| 16 | Dual-T | TransitionEstimator | 组件完成 | 与 RiskCorrector 组成训练闭环 |
| 17 | MC-LDCE | 指南 | 未开始 | StatisticEstimator、global objective |
| 18 | Importance Reweighting | Binary RCN WeightProvider | 组件完成 | posterior/rate estimator 与 runner |
| 19 | CWD | 指南 | 未开始 | class-wise StatisticEstimator |
| 20 | PCSE | 指南 | 未开始 | 特征统计与 post-processing |
| 21 | DLD | 指南 | 未开始 | diffusion label Pipeline |
| 22 | FINE | Selector 基础 | 未开始 | forgetting/negative regularizer |
| 23 | CA2C | 双网络接口待建 | 未开始 | asymmetric Pipeline |
| 24 | DivideMix | Selector 基础 | 未开始 | GMM、MixMatch、双网络 Pipeline |
| 25 | L2RW | WeightProvider 基础 | 未开始 | clean meta-batch 与 MetaUpdater |
| 26 | LEND | 指南 | 未开始 | FeatureGraph 与 indexed label state |

## 复现增量与复用审计表

该表用于约束“菜谱组合原料”的实现方式。`计划中` 只表示已经审阅并保存实施计划，不表示代码已经存在。

| 论文 | 复用原料 | 新增/扩展通用原料 | 论文专属菜谱 | 新增依据 | 后续使用者 | 状态 |
|---|---|---|---|---|---|---|
| GCE | CIFAR Dataset、Noise Manifest、SGD/scheduler、统一 runner、checkpoint、逐样本 Loss 协议 | `GeneralizedCrossEntropyLoss`；CIFAR ResNet-34；`gce2018` per-pixel mean preprocessing；可选 noisy validation；终端进度与 SVG 曲线 | `gce_cifar10_noise02_smoke.yaml`、`gce_cifar10_noise02_reproduction.yaml` | GCE 公式和论文实验模型/预处理此前不在生产闭环；进度能力为所有长实验共用 | 后续 robust loss、transition correction 及其他 CIFAR 长实验 | 单次复现；详见 [GCE 结果](gce/result.md) |
| APL | CIFAR Dataset、Noise Manifest、NCE、RCE、APL、SGD/cosine、统一 runner、checkpoint、metrics | 计划新增通用 `CifarCnn8`；symmetric `per_class` sampling；StandardUpdatePolicy gradient clipping；显式 model-selection split | 计划新增 `apl_cifar10_noise02_smoke.yaml`、`apl_cifar10_noise02_reproduction.yaml` | 对齐论文及作者代码的 8 层 CNN、逐类噪声、梯度裁剪和 test-selection 协议；不得形成 APL 专属公共模块 | 其他 CIFAR loss 论文、使用逐类对称噪声或梯度裁剪的实验 | 计划中；尚未实现，详见 [APL 计划](apl/plan.md) |

## 单篇复现记录

- GCE：[单次复现结果](gce/result.md)
- APL：[单次复现计划](apl/plan.md)（计划中，尚无结果文件）
