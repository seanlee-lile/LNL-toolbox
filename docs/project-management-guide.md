# LNL Toolbox 项目管理指南

## 1. 本指南写给谁

本指南面向能够熟练使用 Windows、命令行、Git 和常见办公软件，但没有 AI 或深度学习开发经验的项目成员。你的主要职责不是推导论文公式，而是保证：文件放置正确、配置有记录、测试可执行、实验结果可追溯、提交内容整洁。

项目当前已经具备一条基础闭环：读取 CIFAR 数据、划分训练/验证/测试集、用 TinyCNN 和交叉熵训练、记录指标、保存 checkpoint、恢复训练。论文中的 LNL 算法尚未批量接入。

## 2. 先理解五个基本概念

1. **数据集**：用于训练和测试的图片及标签。真实数据放在仓库根目录的 `data/`，不会上传 Git。
2. **模型**：根据图片输出类别预测的程序。当前模型是 TinyCNN。
3. **算法**：规定模型如何训练，例如计算损失、更新参数。当前完整训练算法是普通监督分类。
4. **配置**：用 YAML 文件记录数据路径、训练轮数、batch size 等参数。不要把实验参数散落在聊天记录里。
5. **checkpoint**：训练过程的存档，可用于继续训练。文件通常很大，不上传 Git。

## 3. Git 应提交和不应提交的内容

应提交：源码、测试、配置模板、正式使用文档、架构文档、论文来源清单。

不应提交：CIFAR 数据、论文 PDF、模型权重、checkpoint、运行日志、临时文件、个人问题记录、IDE 设置和一次性论文处理脚本。

特别注意：`.gitignore` 使用 `/data/`，前面的 `/` 表示只忽略仓库根目录的数据集。`src/lnl_toolbox/data/` 是程序源码，必须提交。

## 4. 推荐的日常工作流程

### 4.1 开始工作

```powershell
git pull
conda activate pytorch
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

如果测试未通过，不要直接开始新增算法。先记录失败的测试名称和错误信息。

### 4.2 修改配置或代码后

先运行全部测试，再运行两轮小规模 smoke experiment：

```powershell
python -m unittest discover -s tests -v
python -m lnl_toolbox.cli.train --config configs/experiment/cifar10_smoke.yaml
```

### 4.3 提交前

```powershell
git status
git diff
git add .
git status
git commit -m "简短说明本次修改"
git push
```

检查 `git status` 时，如果出现 `.pt`、`.pth`、数据集文件、论文 PDF 或 `artifacts/`，先停止提交并检查 `.gitignore`。

## 5. 根目录文件

| 文件 | 功能 | 项目成员如何处理 |
|---|---|---|
| `.gitignore` | 规定哪些本地文件不进入 Git。 | 新增大型输出目录时更新；不要随意取消数据、权重和日志的忽略规则。 |
| `README.md` | 项目首页和文档导航。 | 对外功能发生变化时更新，保持简短。 |
| `pyproject.toml` | 定义包名、Python 版本、依赖和命令行入口。 | 增加正式依赖或 CLI 时修改；修改后需重新安装或验证环境。 |
| `toolbox-architecture.md` | 长期架构、LNL 实验公平性和算法接入路线。 | 用于决策，不记录临时问题；重大架构变化才修改。 |

## 6. 正式文档 `docs/`

| 文件 | 功能 | 项目成员如何处理 |
|---|---|---|
| `docs/usage-guide.md` | 告诉使用者如何准备数据、训练、恢复和测试。 | 命令、数据路径或输出文件变化时同步修改。 |
| `docs/file-map.md` | 面向开发者的文件职责速查表。 | 新增、删除或移动文件时更新。 |
| `docs/project-management-guide.md` | 本指南，面向无 AI 经验的项目成员。 | 项目流程或提交规则变化时更新。 |

论文摘要 `docs/paper-summaries.md` 属于本地研究材料，当前被 `.gitignore` 排除，不在 push 范围内。

## 7. 配置文件 `configs/`

配置文件只保存参数，不实现算法。修改前复制一份新配置通常比覆盖已有基准更安全。

| 文件 | 功能 | 关键字段或注意事项 |
|---|---|---|
| `configs/README.md` | 解释配置层与核心框架的关系。 | 配置机制变化时更新。 |
| `configs/experiment/cifar10_smoke.yaml` | 两轮、小样本快速实验。 | 用于确认程序能跑通，不用于报告正式精度。 |
| `configs/experiment/cifar10_clean.yaml` | 30 轮干净 CIFAR-10 基准实验。 | 正式运行前确认数据路径、设备和训练轮数。 |
| `configs/algorithm/ce.yaml` | CE 示例算法配置。 | `name` 和 loss 名称必须与插件注册一致。 |
| `configs/algorithm/gce.yaml` | GCE 示例配置。 | `q` 控制 CE 与 MAE 之间的折中。 |
| `configs/algorithm/coteaching.yaml` | Co-teaching 参数示例。 | 当前只有选样函数，尚非完整端到端训练实现。 |
| `configs/noise/symmetric.yaml` | 对称噪声参数示例。 | 记录噪声率和 seed。 |
| `configs/noise/instance_dependent.yaml` | 实例依赖噪声参数示例。 | 当前是基于 class score 的示例生成器，不等同于某篇论文完整算法。 |

## 8. Python 包入口

| 文件 | 功能 | 是否经常修改 |
|---|---|---|
| `src/lnl_toolbox/__init__.py` | 暴露最常用的核心类型并记录版本。 | 很少；只有稳定公共 API 变化时修改。 |
| `src/lnl_toolbox/runtime.py` | 选择 CPU/GPU并统一设置 Python、NumPy、PyTorch 随机种子。 | 设备策略或可复现规则变化时修改。 |
| `src/lnl_toolbox/registry.py` | 旧版单类型注册器，保留兼容性。 | 原则上不新增功能，未来可能由 PluginCatalog 替代。 |

## 9. 通用核心 `src/lnl_toolbox/core/`

核心层不应直接出现 CIFAR、CE、噪声率或某篇论文名称。

| 文件 | 功能 | 修改风险 |
|---|---|---|
| `core/__init__.py` | 汇总核心公开类型，让外部统一导入。 | 中；删除导出可能破坏其他模块。 |
| `core/component.py` | 定义组件的 `setup/close` 和状态保存协议。 | 高；所有插件可能依赖。 |
| `core/context.py` | 定义实验工作目录、配置、seed 和服务容器。 | 高；影响组件初始化。 |
| `core/batch.py` | 用 `payload + metadata` 包装一次算法输入。 | 高；必须保持任务无关。 |
| `core/algorithm.py` | 定义算法从开始、循环、step 到结束的生命周期。 | 很高；新增论文算法会依赖它。 |
| `core/state.py` | 记录 cycle、step、phase、停止标记、指标和元数据。 | 高；影响 checkpoint 恢复。 |
| `core/result.py` | 定义每一步返回的输出、指标、artifact 和元数据。 | 高；影响 Runner 与 evaluator。 |
| `core/evaluator.py` | 定义评测器的更新、汇总和清空接口。 | 高；不要写死 accuracy。 |
| `core/storage.py` | 定义 artifact 与 checkpoint 的通用存储接口。 | 中；当前具体训练另有 PyTorch checkpoint 实现。 |

## 10. 数据源码 `src/lnl_toolbox/data/`

这里是必须 push 的源码，与根目录 `/data/` 中不上传的数据文件不同。

| 文件 | 功能 | 使用说明 |
|---|---|---|
| `data/__init__.py` | 对外导出 CIFAR 数据类型和读取函数。 | 新增稳定数据接口时更新。 |
| `data/contracts.py` | 早期 LNL 分类样本协议，包含 image、target、index 等字段。 | 属于兼容代码；不要让通用 core 反向依赖它。 |
| `data/cifar.py` | 解码 CIFAR-10/100 官方 pickle，检查图片和标签形状。 | 数据读取异常时首先检查。 |
| `data/torch_cifar.py` | 分层划分数据、图像增强、归一化并生成 PyTorch Dataset。 | 修改增强方式会影响实验公平性，必须记录。 |

## 11. 模型与损失

| 文件 | 功能 | 使用说明 |
|---|---|---|
| `models/__init__.py` | 导出当前模型。 | 新模型稳定后再加入。 |
| `models/tiny_cnn.py` | 小型三段 CNN，用于验证端到端训练。 | 是基础设施测试模型，不是论文主干网络。 |
| `losses/__init__.py` | 导出 NumPy 损失函数。 | 公共 API 变化时修改。 |
| `losses/numpy_losses.py` | NumPy CE/GCE，用于数学和单元测试。 | 不负责反向传播。 |
| `losses/torch_losses.py` | PyTorch CE 包装，用于真实训练。 | 新增训练 loss 时保持逐样本/聚合语义清楚。 |

## 12. 算法 `src/lnl_toolbox/algorithms/`

| 文件 | 功能 | 当前成熟度 |
|---|---|---|
| `algorithms/__init__.py` | 导出算法接口及 Co-teaching 工具函数。 | 兼容层。 |
| `algorithms/base.py` | 将旧 `TrainState` 映射到新 `RunState`。 | 兼容层，尽量不扩展。 |
| `algorithms/supervised.py` | 普通监督分类的完整训练 step：前向、loss、反向和更新。 | 已用于干净 CIFAR 闭环。 |
| `algorithms/coteaching.py` | 计算保留率并让双网络交叉选择小损失样本。 | 仅选样核心，不是完整 Co-teaching 训练器。 |

## 13. 插件系统 `src/lnl_toolbox/plugins/`

| 文件 | 功能 | 使用说明 |
|---|---|---|
| `plugins/__init__.py` | 导出插件目录类型。 | 很少修改。 |
| `plugins/catalog.py` | 按类别和名称注册、创建、查询插件。 | 接入新算法时通常使用，不应写入算法公式。 |
| `plugins/builtin/__init__.py` | 导出内置插件目录构造函数。 | 很少修改。 |
| `plugins/builtin/catalog.py` | 注册 CE/GCE、噪声生成器和 Co-teaching selector 示例。 | 新插件稳定后可登记；登记不代表完整论文复现。 |

## 14. 噪声模块 `src/lnl_toolbox/noise/`

| 文件 | 功能 | 使用说明 |
|---|---|---|
| `noise/__init__.py` | 导出 NoiseManifest 和三种生成器。 | 公共 API。 |
| `noise/manifest.py` | 保存原标签、噪声标签、转移概率、fingerprint 和元数据。 | 改格式时必须考虑旧文件兼容。 |
| `noise/generators.py` | 生成 symmetric、pairflip 和示例 IDN 标签。 | 生成规则变化会影响所有算法比较，必须固定 seed 并记录版本。 |

## 15. Runner、训练与评测

| 文件 | 功能 | 使用说明 |
|---|---|---|
| `engine/__init__.py` | 导出 `run_cycles/run_epochs`。 | 公共入口。 |
| `engine/runner.py` | 通用生命周期 Runner，不理解模型、标签或梯度。 | 修改前运行 `test_core.py`。 |
| `training/__init__.py` | 导出具体训练和 checkpoint 函数。 | 很少修改。 |
| `training/checkpoint.py` | 用 `torch.save/load` 保存算法状态、RunState、epoch 和配置。 | 恢复训练问题首先检查；权重文件不上传 Git。 |
| `training/experiment.py` | 组装 CIFAR、DataLoader、TinyCNN、optimizer、训练、验证、测试和日志。 | 当前端到端主流程；接算法时应逐步拆成插件组装，不要无限堆 `if/else`。 |
| `evaluation/__init__.py` | 导出 NumPy 示例指标。 | 公共入口。 |
| `evaluation/metrics.py` | NumPy accuracy 和选样 precision/recall。 | 用于算法诊断。 |
| `evaluation/classification.py` | 在 PyTorch DataLoader 上计算平均 loss 和 accuracy。 | 当前验证/测试使用。 |

## 16. 命令行 `src/lnl_toolbox/cli/`

| 文件 | 功能 | 常用命令 |
|---|---|---|
| `cli/__init__.py` | 提供共享交互提示、模板发现和训练配置覆盖，不执行训练数学。 | 由其他 CLI 导入。 |
| `cli/inspect_data.py` | 交互或参数化地检查 CIFAR 文件、尺寸和类别分布。 | `lnl-inspect-data` |
| `cli/make_noise.py` | 交互或参数化地从 `.npy` 标签生成噪声清单。 | `lnl-make-noise` |
| `cli/train.py` | 无参数进入向导；有参数时读取 YAML、覆盖 epochs 并调用通用训练。 | `lnl-train` |
| `cli/clean_train.py` | Clean baseline 向导及单次、恢复、多 seed 调度。 | `lnl-clean-train` |

## 17. 单元测试 `tests/`

测试文件不是多余开发痕迹，必须 push。它们帮助非 AI 开发人员判断修改是否破坏项目。

| 文件 | 验证内容 |
|---|---|
| `tests/test_core.py` | 通用生命周期、step 推进、指标回传和关闭行为。 |
| `tests/test_plugins.py` | 插件注册、能力查询及非 LNL 插件兼容性。 |
| `tests/test_registry.py` | 旧 Registry 的注册和构建。 |
| `tests/test_cifar_reader.py` | 用临时假 CIFAR pickle 检查 CIFAR-10/100 解码。 |
| `tests/test_noise.py` | 噪声率、翻转约束、随机种子、manifest 保存加载和 IDN 概率。 |
| `tests/test_losses.py` | P0 loss 公式、极端数值、梯度和参数校验。 |
| `tests/test_coteaching.py` | 双网络交叉选样和保留率。 |
| `tests/test_torch_training.py` | PyTorch 数据集、训练 step、设备解析和 checkpoint 恢复。 |
| `tests/test_cli.py` | 交互重试/取消、配置构造、dispatch 和 argparse 兼容。 |

## 18. 论文来源目录 `papers/`

| 文件 | 功能 | 提交规则 |
|---|---|---|
| `papers/README.md` | 说明论文分类、下载状态和缺失文件。 | 提交。 |
| `papers/manifest.json` | 记录论文文件名、来源 URL、下载状态和大小。 | 提交，便于其他成员自行获取论文。 |
| `papers/**/*.pdf` | 论文原文。 | 不提交，避免仓库过大及再分发风险。 |

## 19. 角色分工建议

- **项目管理员**：维护 README、使用说明、配置命名、Git 提交范围和实验清单。
- **算法开发者**：阅读论文，实现算法插件和专属测试，不改通用 core，除非有明确设计评审。
- **实验执行者**：只改配置，保存运行命令、Git commit、seed 和最终指标。
- **代码审查者**：检查算法是否读取了不该使用的干净标签、是否公平共享数据划分和训练预算。

## 20. 何时算一次修改完成

一次修改至少满足：

1. `python -m unittest discover -s tests -v` 全部通过；
2. 涉及训练主流程时，smoke experiment 能运行结束；
3. 新增文件已在 `docs/file-map.md` 和本指南登记；
4. 新命令已写入 `docs/usage-guide.md`；
5. `git status` 不包含数据、PDF、权重、日志和个人笔记；
6. commit message 能说明“改了什么”，而不是只写 `update`。

