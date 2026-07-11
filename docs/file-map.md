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
| `plugins/builtin/catalog.py` | 将 CE/GCE、三种噪声生成和 Co-teaching selector 注册成可选示例插件。 |
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
| `data/__init__.py` | 公开 CIFAR 读取函数和数据类型。 |
| `data/cifar-10-batches-py/` | 用户放入的 CIFAR-10 官方 Python 数据。 |
| `data/cifar-100-python/` | 用户放入的 CIFAR-100 官方 Python 数据。 |

## 6. LNL 示例能力

| 文件 | 作用 |
|---|---|
| `noise/manifest.py` | `NoiseManifest` 的数据结构、标签 fingerprint、NPZ 保存与加载。 |
| `noise/generators.py` | symmetric、pairflip 和基于 class score 的示例 IDN 生成器。 |
| `noise/__init__.py` | 公开 Noise Manifest 和生成器。 |
| `losses/numpy_losses.py` | NumPy 版逐样本 CE 与 GCE，用于数学验证，不执行神经网络反向传播。 |
| `losses/__init__.py` | 公开损失函数。 |
| `evaluation/metrics.py` | NumPy 版 accuracy 和选样 precision/recall。 |
| `evaluation/__init__.py` | 公开当前示例指标。 |

## 7. CLI

| 文件 | 作用 |
|---|---|
| `cli/__init__.py` | CLI 包标记。 |
| `cli/make_noise.py` | 从 `.npy` 标签数组生成 symmetric/pairflip Noise Manifest。 |
| `cli/inspect_data.py` | 读取并验证本地 CIFAR-10/100，输出样本数、尺寸、类别范围和类别计数。 |

## 8. 配置

| 文件 | 作用 |
|---|---|
| `configs/README.md` | 说明 YAML 是 LNL 示例配置，核心只接收 mapping，不依赖 YAML/Hydra。 |
| `configs/algorithm/ce.yaml` | CE 示例参数。 |
| `configs/algorithm/gce.yaml` | GCE 示例参数。 |
| `configs/algorithm/coteaching.yaml` | Co-teaching 示例参数。 |
| `configs/noise/symmetric.yaml` | symmetric noise 示例参数。 |
| `configs/noise/instance_dependent.yaml` | 示例 IDN 参数。 |

## 9. 测试

| 文件 | 验证内容 |
|---|---|
| `tests/test_core.py` | 通用 Runner 生命周期、状态推进和 close 行为。 |
| `tests/test_plugins.py` | capability 查询及与 LNL 无关的自定义插件。 |
| `tests/test_registry.py` | 旧 Registry 注册和构建。 |
| `tests/test_cifar_reader.py` | 用小型临时 pickle 验证 CIFAR-10/100 解码逻辑。 |
| `tests/test_noise.py` | 零噪声、翻转约束、seed 可复现、manifest roundtrip、IDN 概率归一化。 |
| `tests/test_losses.py` | GCE 在 `q→0` 时逼近 CE。 |
| `tests/test_coteaching.py` | 双网络交叉选样和保留率日程。 |

## 10. 调研脚本与文档

| 文件 | 作用 |
|---|---|
| `scripts/download_papers.ps1` | 下载论文并检查 PDF 文件头，生成来源 manifest。 |
| `scripts/extract_papers.py` | 从论文 PDF 抽取文本供摘要整理使用。 |
| `docs/paper-summaries.md` | 26 篇 LNL 论文的中文摘要、代码链接与伪代码。 |
| `docs/usage-guide.md` | 面向使用者的当前功能、命令和下一步测试说明。 |
| `docs/architecture.md` | 通用核心、插件、Runner 与 CIFAR 数据流的架构图。 |
| `docs/file-map.md` | 本文档，解释每个代码文件的职责。 |
| `papers/README.md` | 论文目录、下载状态和缺失 PDF 说明。 |

