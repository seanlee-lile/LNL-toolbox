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
| `plugins/builtin/catalog.py` | 注册 PyTorch/NumPy loss、噪声生成器、Co-teaching selector、Anchor 和 Dual-T transition estimator，并提供配置 builder。 |
| `registry.py` | 早期的单类型轻量 Registry；暂时保留以兼容已有代码，长期可由 PluginCatalog 取代。 |

## 4. Runner 与算法接口

| 文件 | 作用 |
|---|---|
| `engine/__init__.py` | 公开 `run_cycles` 和兼容名称 `run_epochs`。 |
| `engine/runner.py` | 执行 setup、run、cycle、step、evaluator 和 close；不处理模型或梯度。 |
| `algorithms/base.py` | 将旧 `Algorithm/TrainState` 导入映射到新的通用核心，保持兼容。 |
| `algorithms/coteaching.py` | NumPy 版 Co-teaching 保留率日程和小损失交叉选样函数。 |
| `algorithms/__init__.py` | 汇总算法兼容接口和 Co-teaching 函数。 |

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
| `plugins/builtin/catalog.py` | 将 PyTorch loss 注册为 `loss`、NumPy 参考实现注册为 `numpy_loss`，将 Anchor/Dual-T 注册为 `transition_estimator`；递归构造并校验 APL。 |
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
| `training/experiment.py` | 唯一监督训练器；统一构造模型、Loss、optimizer、scheduler、clean/noisy Dataset、评测和产物。`run_experiment` 为兼容入口。 |
| `training/clean_baseline.py` | clean-only 包装和多 seed 汇总；检测到 noise 配置立即拒绝。 |
| `training/noisy_labels.py` | 生成或导入 manifest，规范化为 run-local v2，校验恢复身份并生成无标签泄漏的元数据。 |
| `training/checkpoint.py` | 保存 checkpoint v2，并安全读取旧 CE-baseline 顶层格式和旧 Loss 嵌套格式。 |
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
| `configs/noise/symmetric.yaml` | symmetric noise 示例参数。 |
| `configs/noise/instance_dependent.yaml` | 示例 IDN 参数。 |
| `configs/experiment/cifar10_symmetric_ce_smoke.yaml` | symmetric 0.4 + CE 的统一 noisy runner smoke 配置。 |
| 实验 YAML 顶层 `noise` | 省略时 clean；`name/rate/seed` 生成噪声，或用 `manifest` 导入外部映射，两种方式互斥。 |

## 10. 测试

| 文件 | 验证内容 |
|---|---|
| `tests/test_core.py` | 通用 Runner 生命周期、状态推进和 close 行为。 |
| `tests/test_plugins.py` | capability 查询、loss 类型隔离、递归 APL 构造，以及 Anchor/Dual-T builder/catalog 约束一致性。 |
| `tests/test_registry.py` | 旧 Registry 注册和构建。 |
| `tests/test_cifar_reader.py` | 用小型临时 pickle 验证 CIFAR-10/100 解码逻辑。 |
| `tests/test_noise.py` | 噪声生成、manifest roundtrip/身份校验、非法标签与概率拒绝、KnownTransition 和恢复训练 manifest 身份约束。 |
| `tests/test_transition_estimators.py` | Snapshot/collector 合同、Anchor 与 Dual-T 数学和顺序不变性、Artifact roundtrip/篡改检测及 Tensor 转换。 |
| `tests/test_losses.py` | P0 loss 公式、GCE 极低概率梯度、逐样本 shape、极端数值和 APL 论文约束。 |
| `tests/test_coteaching.py` | 双网络交叉选样和保留率日程。 |
| `tests/test_cli.py` | Prompt 重试/取消、GCE/APL 配置、APL 正权重输入和交互/参数模式兼容。 |
| `tests/test_noisy_ce_baseline.py` | 统一 runner 的 generated noise、clean evaluation、manifest 元数据和非 CE loss 接入。 |

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
