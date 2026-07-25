# 代码文件职责说明

本表覆盖当前项目中的源代码、配置、测试和主要文档。`papers/` 下的 PDF、CIFAR 二进制文件和自动生成的 `__pycache__/` 不逐个解释。

## 1. 仓库根目录

| 文件 | 作用 |
|---|---|
| `README.md` | 项目入口，说明当前定位、核心能力和文档导航。 |
| `pyproject.toml` | Python 包信息、依赖、可选训练依赖、命令行入口及 `src` 包发现规则。 |
| `toolbox-architecture.md` | LNL toolbox 的长期设计、实验公平性、Noise Manifest 和阶段路线；其中具体算法是参考插件规划。 |
| `.gitignore` | 忽略缓存、虚拟环境、构建产物、运行输出和临时文件。 |

## 2. 通用核心 `src/lnl_toolbox/core/`

| 文件 | 作用 |
|---|---|
| `core/__init__.py` | 汇总并公开核心类型，调用方可从 `lnl_toolbox.core` 统一导入。 |
| `core/component.py` | 定义最小 `Component` 生命周期和可选 `Stateful` checkpoint 协议。 |
| `core/context.py` | 定义 `ExperimentContext`，保存工作目录、配置、seed 和外部服务。 |
| `core/batch.py` | 定义通用 `Batch`；payload 不透明，不强制图像或标签格式。 |
| `core/algorithm.py` | 定义任务无关 `Algorithm` 生命周期协议。 |
| `core/state.py` | 定义 Runner 管理的 `RunState`，记录 cycle、step、phase、指标和元数据。 |
| `core/result.py` | 定义 `StepResult` 和 `Artifact`，统一算法步骤的返回格式。 |
| `core/evaluator.py` | 定义 evaluator 的 `update/compute/reset` 协议，不预设具体指标。 |
| `core/storage.py` | 定义 `ArtifactSink`、`CheckpointStore`、`Checkpoint` 和 `ArtifactRef` 存储边界。 |

## 3. 插件系统 `src/lnl_toolbox/plugins/`

| 文件 | 作用 |
|---|---|
| `plugins/__init__.py` | 公开 `PluginCatalog` 和 `PluginSpec`。 |
| `plugins/catalog.py` | 实现按 kind/name 注册、构建及 capability 查询的插件目录。 |
| `plugins/builtin/__init__.py` | 公开内置示例插件目录构造函数。 |
| `plugins/builtin/catalog.py` | 分 kind 注册 PyTorch/NumPy loss、噪声生成器、通用 `batch_selector`、Anchor/Dual-T `transition_estimator`、ParameterUpdatePolicy 及旧 Co-teaching helper，并提供各自独立的配置构造入口。 |
| `registry.py` | 早期的单类型轻量 Registry；暂时保留以兼容已有代码，长期可由 PluginCatalog 取代。 |

## 4. Runner 与算法接口

| 文件 | 作用 |
|---|---|
| `engine/__init__.py` | 公开 `run_cycles` 和兼容名称 `run_epochs`。 |
| `engine/runner.py` | 执行 setup、run、cycle、step、evaluator 和 close；不处理模型或梯度。 |
| `algorithms/base.py` | 将旧 `Algorithm/TrainState` 导入映射到新的通用核心，保持兼容。 |
| `algorithms/coteaching.py` | NumPy 版 Co-teaching 保留率日程和小损失交叉选样函数。 |
| `algorithms/update_policy.py` | 定义通用 ParameterUpdateInput/Result/Policy、普通 StandardUpdatePolicy，以及 policy checkpoint 身份协议。 |
| `algorithms/cdr.py` | 实现 CDR 的全局逐标量 criticality、确定性 top-k 和论文 Eq. (5)/(6) 参数更新。 |
| `algorithms/supervised.py` | 单模型监督训练步骤；将 detached 逐样本 loss 交给通用 Selector，再把归约后的 scalar objective 交给 ParameterUpdatePolicy。 |
| `algorithms/__init__.py` | 汇总算法兼容接口、Co-teaching 函数和可选 PyTorch ParameterUpdatePolicy。 |

## 5. 数据层

| 文件/目录 | 作用 |
|---|---|
| `data/contracts.py` | 早期 LNL 分类样本结构，包含 image、target、index 和可选干净标签；属于具体任务协议。 |
| `data/cifar.py` | 读取 CIFAR-10/100 官方 pickle，转换为 `[N,32,32,3]` uint8 图像并验证标签。 |
| `data/torch_cifar.py` | 提供分层划分、图像变换和只读取原始标签的 PyTorch Dataset；返回稳定的 `input/target/index`。 |
| `data/noisy_dataset.py` | 按显式 global-index mapping 包装训练 Dataset 并替换 target；不会向 batch 暴露 clean label。 |
| `data/__init__.py` | 公开 CIFAR 读取函数、干净 Dataset 和 noisy wrapper。 |
| `data/cifar-10-batches-py/` | 用户放入的 CIFAR-10 官方 Python 数据。 |
| `data/cifar-100-python/` | 用户放入的 CIFAR-100 官方 Python 数据。 |

## 6. LNL 示例能力

| 文件 | 作用 |
|---|---|
| `noise/manifest.py` | `NoiseManifest` 的数据结构、标签 fingerprint、NPZ 保存/加载，以及数据集、长度、标签范围和概率的训练前校验。 |
| `noise/generators.py` | symmetric、pairflip 和基于 class score 的示例 IDN 生成器。 |
| `noise/transition.py` | 验证 `T[i,j]=P(noisy=j|clean=i)` 行随机矩阵；提供 `KnownTransition`、版本化 `TransitionArtifact`、NPZ roundtrip 和哈希篡改检测。 |
| `noise/estimators.py` | 定义无 clean-label 的 `PosteriorSnapshot`、`TransitionEstimator` Protocol，并实现 Anchor 与 Dual-T 离线 estimator。 |
| `noise/__init__.py` | 公开 Noise Manifest、生成器、后验快照、estimator 和转移矩阵产物协议。 |
| `losses/numpy_losses.py` | NumPy 版逐样本 CE 与 GCE，用于数学验证，不执行神经网络反向传播。 |
| `losses/torch_losses.py` | PyTorch 版逐样本 CE、标准 GCE、NCE、MAE、RCE、严格 P0 APL，以及 `[B]` 输出合同校验。 |
| `losses/__init__.py` | 公开 NumPy 参考函数；安装 PyTorch 时同时公开可训练 loss。 |
| `losses/loss板块第一轮.md` | Loss 实现简报及 Config、Algorithm、Selector、Evaluator 间的统一调用协议。 |
| `selectors/base.py` | 定义单 batch 的 `SelectionInput`、`SelectionResult`、无状态 `Selector` Protocol 及输入输出校验。 |
| `selectors/basic.py` | 实现选择全部样本的 `AllSelector` 和 schedule-driven、stable-index tie-break 的 `SmallLossSelector`。 |
| `selectors/schedules.py` | 定义无状态 keep-rate schedule，支持固定浮点、显式 constant 和零基 epoch linear 配置。 |
| `selectors/__init__.py` | 公开通用 Selector 合同、基础实现和边界校验。 |
| `estimators/base.py` | 定义 sample-aligned `ReliabilityResult`、泛型 `ReliabilityEstimator[InputT]` 和轻量 `StatisticResult[StatisticT]`；reliability score 固定为越大越可靠，不直接兼容低分优先的 `SmallLossSelector`。 |
| `estimators/__init__.py` | 公开 Reliability/Statistic estimation 合同及边界验证；不接入 plugin、配置或训练生命周期。 |
| `estimators/dividemix_gmm.py` | 实现 DivideMix 中独立的 epoch-level two-Gaussian clean-probability 子模块；CPU float64 拟合并按 stable index 返回高分代表更可靠的证据，不包含完整 DivideMix Pipeline。 |
| `estimators/selection_adapter.py` | 按 stable index 将 dataset-level reliability 查找、抽取并重排为 batch 输入，再固定取负转换为低分优先的 `SelectionInput`；不调用 Selector 或决定 threshold/split。 |
| `treatments/base.py` | 定义内部 `ContributionResult`，统一表达 hard mask、连续非负样本权重和标量统计。 |
| `treatments/reduction.py` | 定义 `ReductionSpec`，按 weight-sum mean、batch mean 或 sum 归约逐样本 loss。 |
| `treatments/selector_adapter.py` | 将现有 hard Selector 适配为 mask 加全一权重，保持旧配置和数值行为。 |
| `treatments/weights.py` | 定义泛型 WeightProvider、通用 WeightResult 和不依赖具体输入字段的 adapter；BinaryRCNWeightInput 与对应 provider 实现二分类 asymmetric-RCN 的论文精确 importance-weight 公式，不负责 posterior 或噪声率估计。 |
| `treatments/__init__.py` | 公开内部 sample-treatment、reducer、Selector adapter 和连续权重合同。 |
| `evaluation/metrics.py` | NumPy 版 accuracy 和选样 precision/recall。 |
| `evaluation/__init__.py` | 公开当前示例指标。 |

## 7. CLI

| 文件 | 作用 |
|---|---|
| `cli/__init__.py` | 共享中文 `PromptSession`、实验模板发现、Loss 选择、clean/generated/external 标签模式和最终确认；不包含训练数学。 |
| `cli/train.py` | 通用训练入口；无参数进入向导，有参数时保持 argparse/YAML 调用。 |
| `cli/clean_train.py` | Clean baseline 入口；交互选择模型、scheduler、恢复或多 seed。 |
| `cli/make_noise.py` | 交互或参数化地从 `.npy` 标签生成 symmetric/pairflip Noise Manifest。 |
| `cli/inspect_data.py` | 交互或参数化地验证 CIFAR-10/100 并输出划分摘要。 |

## 8. 训练编排

| 文件 | 作用 |
|---|---|
| `training/experiment.py` | 唯一监督训练器；统一构造模型、Loss、batch Selector、ParameterUpdatePolicy、optimizer、scheduler、clean/noisy Dataset、评测和产物。 |
| `training/clean_baseline.py` | clean-only 包装和多 seed 汇总；检测到 noise 配置立即拒绝。 |
| `training/noisy_labels.py` | 生成或导入 manifest，规范化为 run-local v2，校验恢复身份并生成无标签泄漏的元数据。 |
| `training/checkpoint.py` | 保存 checkpoint v2 的模型、优化器和 ParameterUpdatePolicy 身份/私有状态，并安全读取旧格式。 |
| `training/snapshots.py` | 在 inference mode 下收集 noisy posterior、target 和稳定 global index，排序后构造 `PosteriorSnapshot`；不执行 warm-up 训练。 |

## 9. 配置

| 文件 | 作用 |
|---|---|
| `configs/README.md` | 说明 YAML 是 LNL 示例配置，核心只接收 mapping，不依赖 YAML/Hydra。 |
| `configs/algorithm/ce.yaml` | CE 示例参数。 |
| `configs/algorithm/gce.yaml` | 标准 GCE 的 `q` 参数；不包含隐式截断阈值。 |
| `configs/algorithm/nce.yaml` | NCE 的数值稳定参数。 |
| `configs/algorithm/apl.yaml` | 严格正权重及 NCE + RCE 嵌套配置。 |
| `configs/algorithm/coteaching.yaml` | Co-teaching 示例参数。 |
| `configs/algorithm/cdr.yaml` | paper-mode CDR 的噪声率、L1 系数和参数范围。 |
| `configs/noise/symmetric.yaml` | symmetric noise 示例参数。 |
| `configs/noise/instance_dependent.yaml` | 示例 IDN 参数。 |
| `configs/experiment/cifar10_symmetric_ce_smoke.yaml` | symmetric 0.4 + CE 的统一 noisy runner smoke 配置。 |
| `configs/experiment/cifar10_symmetric_small_loss_smoke.yaml` | symmetric 0.4 + CE + 固定 0.5 keep-rate SmallLossSelector 的单模型 smoke 配置。 |
| `configs/experiment/cifar10_symmetric_small_loss_linear_smoke.yaml` | symmetric 0.4 + CE + 从 1.0 线性变化到 0.5 的 SmallLossSelector smoke 配置。 |
| `configs/experiment/cifar10_symmetric_cdr_smoke.yaml` | symmetric 0.4 + CE + paper-mode CDR ParameterUpdatePolicy 的 smoke 配置。 |
| 实验 YAML 顶层 `selector` | 省略时为 `all`；当前支持 `all` 和固定 keep-rate `small_loss`。 |
| 实验 YAML 顶层 `noise` | 省略时 clean；`name/rate/seed` 生成噪声，或用 `manifest` 导入外部映射，两种方式互斥。 |

## 10. 测试

| 文件 | 验证内容 |
|---|---|
| `tests/test_core.py` | 通用 Runner 生命周期、状态推进和 close 行为。 |
| `tests/test_plugins.py` | capability 查询、loss/`batch_selector`/旧 Co-teaching/transition/update-policy kind 隔离，以及 Anchor/Dual-T builder 与各独立调用回路。 |
| `tests/test_registry.py` | 旧 Registry 注册和构建。 |
| `tests/test_cifar_reader.py` | 用小型临时 pickle 验证 CIFAR-10/100 解码逻辑。 |
| `tests/test_noise.py` | 噪声生成、manifest roundtrip/身份校验、非法标签与概率拒绝、KnownTransition 和恢复训练 manifest 身份约束。 |
| `tests/test_transition_estimators.py` | Snapshot/collector 合同、Anchor 与 Dual-T 数学和顺序不变性、Artifact roundtrip/篡改检测及 Tensor 转换。 |
| `tests/test_estimators.py` | Reliability/Statistic 基础结果合同、stable-index 对齐、score 方向和 metrics 校验。 |
| `tests/test_dividemix_gmm.py` | DivideMix GMM clean-probability 子组件的拟合、确定性、退化输入、可选依赖与输出合同。 |
| `tests/test_reliability_selection_adapter.py` | dataset-level reliability 按 stable index 抽取、重排并显式转换为低分优先 SelectionInput。 |
| `tests/test_losses.py` | P0 loss 公式、GCE 极低概率梯度、逐样本 shape、极端数值和 APL 论文约束。 |
| `tests/test_coteaching.py` | 双网络交叉选样和保留率日程。 |
| `tests/test_update_policy.py` | 通用 ParameterUpdatePolicy 输入输出、Standard 更新等价性和 checkpoint 身份协议。 |
| `tests/test_cdr.py` | CDR Eq. (3)-(6)、稳定 top-k、失败边界、Selector 组合、plugin、checkpoint 和 CPU/CUDA 一致性。 |
| `tests/test_cli.py` | Prompt 重试/取消、GCE/APL 配置、APL 正权重输入和交互/参数模式兼容。 |
| `tests/test_noisy_ce_baseline.py` | 统一 runner 的 generated noise、clean evaluation、manifest 元数据、Loss 与 Selector 配置接入。 |
| `tests/test_selectors.py` | 通用 Selector 输入输出合同、固定比例、最少选择、stable-index tie-break 和失败边界。 |
| `tests/test_treatments.py` | ContributionResult、ReductionSpec、Selector adapter 和显式 loss 归约合同。 |
| `tests/test_importance_reweighting.py` | Binary asymmetric-RCN 权重公式、通用 WeightProvider adapter、detach 和 batch-mean 梯度。 |

## 11. 调研脚本与文档

| 文件 | 作用 |
|---|---|
| `scripts/download_papers.ps1` | 下载论文并检查 PDF 文件头，生成来源 manifest。 |
| `scripts/extract_papers.py` | 从论文 PDF 抽取文本供摘要整理使用。 |
| `docs/paper-summaries.md` | 26 篇 LNL 论文的中文摘要、代码链接与伪代码。 |
| `docs/usage-guide.md` | 面向使用者的当前功能、命令和下一步测试说明。 |
| `docs/architecture.md` | 通用核心、插件、Runner 与 CIFAR 数据流的架构图。 |
| `docs/file-map.md` | 本文档，解释每个代码文件的职责。 |
| `papers/README.md` | 论文目录、下载状态和缺失 PDF 说明。 |
| `papers/transition-estimator-audit.md` | Anchor 与 Dual-T transition estimator 的论文依据、公式、实现边界、差异和验证状态。 |
