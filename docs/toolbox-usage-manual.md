# LNL Toolbox 使用指南

本文面向第一次使用 LNL Toolbox 的用户，以当前 Web 界面的实际操作顺序为主。你不需要先了解内部代码；从数据集开始，工具箱会引导你检查数据、选择可用方法、预演训练并查看结果。

> Smoke 用于快速确认工程流程，不能代表论文最终性能。

## 1. 安装并启动 Web

在项目根目录打开 PowerShell，创建 Python 3.10 或更高版本的环境并安装项目：

```powershell
conda create -n lnl-toolbox python=3.11 -y
conda activate lnl-toolbox
python -m pip install -e ".[train]"
lnl web --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765/`。如果当前 shell 找不到 `lnl`，源码 checkout 可使用：

```powershell
$env:PYTHONPATH = "src"
python -m lnl_toolbox.cli.main web --host 127.0.0.1 --port 8765
```

当前左侧导航依次为：

1. **快速开始**：从数据集出发完成第一次实验。
2. **新手教程**：按 `doctor → list → validate → dry-run → run → resume` 学习完整流程。
3. **新建 YAML**：从论文配置或监督组合生成项目 YAML。
4. **本地数据集**：登记、实际检查和训练验证本机数据。
5. **参数 Sweep**：组合多个参数取值和随机种子。
6. **运行管理**：恢复训练、生成报告和比较曲线。
7. **论文方法**：查看论文、公开配置、限制和运行入口。
8. **自由输入**：执行一条由后端安全解析的 `lnl` 命令。

## 2. 第一次使用：Quick Start

第一次使用建议进入 **快速开始**。真实流程是：

```text
打开 Web
→ 快速开始
→ 输入路径或点击“浏览”
→ 自动识别并继续
→ adapter 识别、登记和实际加载
→ 选择标签噪声
→ 查看方法状态
→ 选择方法
→ 补充方法输入
→ 预演
→ 确认并开始训练
→ 运行管理
```

### 2.1 数据集

在“1. 数据集”输入目录或单文件路径，也可以点击 **浏览**，然后点击 **自动识别并继续**。系统会探测文件布局、选择 adapter、登记并实际加载数据。成功后显示数据集名称、train/test 样本数、类别数、标签情况和噪声来源。

若有多个候选 adapter，由用户确认；无法自动识别时，转到 **本地数据集** 手动登记。已经登记的数据可从下拉框选择。点击 **更换数据集** 可以开始另一个数据集。

### 2.2 标签噪声

在“2. 标签噪声”中：

- 原生噪声数据集直接使用 observed labels，不再叠加人工噪声。
- 干净数据可以保持干净，或选择工具箱支持的人工噪声类型、噪声率和 seed。
- 标签噪声状态未知时，页面要求确认；继续只表示使用 observed labels，不会把它们声明为干净标签。

### 2.3 方法和运行

“3. 方法”按“可直接运行”“补充输入后可运行”“当前实现不适用”“兼容性元数据不完整”分组。选择方法后，“4. 配置来源与运行”显示配置来源、计划状态和缺少的输入。

方法噪声率先验等 recipe 输入可以在这里补充；数据集事实或外部 artifact 不能伪造，应转到 **本地数据集** 或 **新建 YAML**。计划 ready 后：

- **预演**：显示数据、runner、epoch 和输出计划，不训练。
- **确认并开始训练**：再次确认后启动训练。

## 3. 本地数据集登记与检查

需要精确选择 adapter 或路径时，进入 **本地数据集**。三个阶段不可混为一谈：

```text
1. 登记路径 → 2. 实际检查 → 3. 一轮验证
```

“已登记”不等于“检查通过”，“检查通过”也不等于“训练已验证”。

### 3.1 登记

在“操作”选择 **1 · 登记本机路径**，填写：

- **新别名**和**适配器**；
- 目录型数据的**数据根目录**；
- UCI 等单文件数据的**单文件路径**；
- CIFAR-N adapter 需要的**人工标签文件**。

登记只保存路径，不复制或删除原始数据。

### 3.2 Inspect 和一轮验证

选择已登记数据，将“操作”切换为 **2 · 实际加载 train/test**。inspect 会真实加载数据并检查样本、类别、split、标签和 fingerprint。

inspect 通过后选择 **3 · 训练 1 epoch 验证**，验证从加载到参数更新的工程链路。它不代表论文结果。

### 3.3 登记第二个数据集

无需删除第一个数据集。再次选择 **1 · 登记本机路径**，填写新别名、adapter 和路径即可。“已登记数据”下拉框用于切换。删除登记只移除 catalog 记录，不删除原文件。

## 4. 补充 UNKNOWN 数据事实

选择 **使用已登记数据训练** 后，“数据集优先流程”才会显示系统自动识别的信息：

- 数据集、adapter、模态和类别数；
- train/validation/test 数量；
- observed train labels 和 stable indices；
- 噪声状态、噪声来源和 dataset fingerprint。

系统无法确定时，页面才允许用户确认：

- **标签是否含噪声**；
- **噪声来源**；
- **干净训练标签**是否存在；
- **真实噪声率**的状态、数值和来源。

不知道的信息可以保持 Unknown，不要为了启用按钮而猜测。

**真实噪声率**属于数据集事实；**当前方法噪声率先验**属于所选 recipe。两者是独立字段。预训练 checkpoint/role 也属于方法输入，应由 recipe 或项目 YAML 提供。

## 5. 理解方法兼容性

工具箱用自动识别的数据事实、用户确认的 UNKNOWN 事实和具体 recipe 要求计算兼容性：

- **COMPATIBLE / 可用**：可以直接 **检查**、**预演**或**运行**。
- **NEEDS_INPUT / 需补充条件**：方法原则上适用，但还缺方法先验、manifest、checkpoint、预训练角色或配置字段。
- **INCOMPATIBLE / 不兼容**：模态、类别、标签合同或数据来源等硬要求不满足。

在 recipe 下拉框中选择具体配置后，页面只显示该配置的“当前选择”卡片。卡片会说明可用、需补充条件或不兼容的真实原因；不再提供旧的分类列表或“显示不兼容配置”checkbox。`NEEDS_INPUT` 并非永久不可运行：补齐对应 prerequisite 后，卡片会恢复可运行状态；`INCOMPATIBLE` 才表示硬合同不满足。

“需要确认的数据集信息”只填写 DatasetProfile 无法自动确认的事实。checkpoint、外部 artifact、class prior 等方法专属输入不是数据事实，应按“当前选择”卡片提示在 YAML/方法配置中提供。

## 6. 选择论文方法和具体配置

进入 **论文方法**：

1. 选择论文方法。
2. 选择具体“工具箱配置”。
3. 查看论文问题、机制、训练流程、已知限制，以及配置字段和实现位置。
4. 选择 **运行正式配置**、**编辑为项目 YAML** 或 **使用此配置做 Sweep**。

“运行正式配置”使用当前下拉框选中的 recipe，仍会经过数据和前置条件检查。配置存在不代表论文数值已经复现，应同时查看 fidelity、reproduction status 和 availability。

## 7. Smoke 与 Reproduction

### Smoke

- 数据或预算较小、epoch 较少；
- 验证数据加载、runner、训练、checkpoint 和 resume；
- 成功只表示工程链路可用，不代表论文最终准确率。

### Reproduction

- 使用较完整的数据、模型和预算，耗时通常更长；
- 可能要求外部数据、manifest、预训练模型或 artifact；
- 名称中有 reproduction 也不自动等于 paper-exact，以 catalog 状态和已知限制为准。

建议先跑 smoke，再 validate 和 dry-run reproduction，确认设备、预算和依赖后才开始正式训练。

## 8. 新手教程六步

进入 **新手教程**，选择 Smoke 实验、本地数据集（可选）和独立输出目录：

| 步骤 | 实际命令 | 完成条件 |
|---|---|---|
| 1. 检查环境 | `lnl doctor` | 退出码为 0，关键项目显示 OK |
| 2. 选择 Smoke 实验 | `lnl list experiments --profile smoke --format json` | 列表成功加载并包含所选 recipe |
| 3. 验证配置与数据 | `lnl validate --recipe <recipe> --check-data` | 验证成功并显示配置路径和 runner |
| 4. 预演训练计划 | `lnl run --recipe <recipe> --dry-run --check-data` | 显示计划且未产生训练状态 |
| 5. 运行 Smoke | `lnl run --recipe <recipe> --output-dir <dir> --check-data` | 生成 resolved config、指标和 checkpoint |
| 6. 检查并恢复 | 检查目录；需要时 `lnl resume <dir> --checkpoint last` | 未完成时恢复；已完成时显示无需恢复 |

**退出码 0 表示命令成功完成，不是“退出程序”。** 当前步骤只有 passed 或 not-needed 后，**下一步**才解锁。成功命令会更新进度；完成的 run 不会重复执行无意义的 resume。

“快速命令（跳过逐步教程）”可生成 validate、dry-run、短训练或 Sweep，但不会替代逐步教程状态。

## 9. 新建或编辑项目 YAML

进入 **新建 YAML**，选择：

- **论文正式配置**：从论文公开 recipe 开始；或
- **通用监督组合**：选择基础配方、loss、样本选择器和更新策略。

可以指定“目标已登记数据集”，让 Web 先检查 recipe 兼容性。推荐流程：

```text
选择论文或基础配方
→ 查看兼容性
→ 显示/编辑 YAML
→ 设置项目内的新保存路径
→ 保存
→ 验证此 YAML
→ 运行此 YAML，或保存并转到 Sweep
```

内置 recipe 不允许直接覆盖。Web 会建议 `configs/experiment/<recipe>-custom.yaml` 一类的新路径。修改论文训练预算、模型、数据、噪声或方法参数可能降低 reproduction fidelity。

## 10. Validate、Dry-run 和 Run

三者含义不同：

```powershell
# 检查配置、数据、方法 preflight 和 compatibility，不训练
lnl validate --recipe <recipe> --check-data

# 显示真实执行计划，不训练
lnl run --recipe <recipe> --dry-run --check-data

# 正式启动
lnl run --recipe <recipe> --output-dir artifacts/runs/<name> --check-data
```

Web 中：

- Quick Start 提供 **预演**、**确认并开始训练**；
- 数据兼容性区域提供 **检查**、**预演**、**运行**；
- YAML 编辑区提供 **验证此 YAML**、**运行此 YAML**；
- 页面下方 PREVIEW 显示当前模块生成的真实命令，执行前应核对 recipe、数据和输出目录。

## 11. 参数 Sweep

进入 **参数 Sweep**：

1. 选择内置 recipe 或项目 YAML；
2. 选择允许 Sweep 的字段；
3. 为字段填写多个候选值；
4. 可选填写多个 seed；
5. 预览笛卡尔积和输出目录；
6. 确认后执行。

论文锁定字段不会作为普通可调参数。Sweep 中断后应从已有目录恢复，不要删除目录重跑。

## 12. 运行管理、结果和 Resume

训练开始后进入 **运行管理**：

- **resume · 恢复训练**：选择目录和 `last`/`best`，先点 **检查恢复内容**；只有显示“可以恢复”后才生成命令。
- **report · 生成报告**：读取一个运行目录。
- **compare · 汇总比较**：扫描运行/Sweep 目录，筛选 run 并绘制逐 epoch 指标曲线。

常见产物：

| 文件 | 含义 |
|---|---|
| `final_metrics.json` | 完成训练后的最终指标摘要 |
| `metrics.jsonl` | 逐 epoch/阶段训练日志，不是图片曲线 |
| `best.pt` | 按验证规则保存的最佳 checkpoint |
| `last.pt` | 最近训练边界的 checkpoint，通常用于恢复 |
| `resolved_config.yaml` | 本次实际运行的完整配置 |
| data/noise artifacts | split、fingerprint、noise manifest 等复现信息 |

恢复检查还显示阶段、epoch、checkpoint 进度、最佳记录、目录文件和 resolved config。未完成训练通常从 `last` 恢复：

```powershell
lnl resume artifacts/runs/<name> --checkpoint last
```

已完成时页面会明确显示无需恢复，不要反复 resume。

## 13. MentorNet 前置准备

MentorNet 的正确流程是：

```text
准备 Mentor 特征
→ 训练并冻结 MentorArtifact
→ Student 训练
```

在 **论文方法 → MentorNet** 选择配置后，页面显示：

- `MentorArtifact：READY / NOT READY`
- `步骤 1 · Mentor 特征：READY / NOT READY`
- `步骤 2 · 冻结的 MentorArtifact：READY / NOT READY`
- `步骤 3 · Student 训练：AVAILABLE / BLOCKED`

当前 smoke 配置支持本地 preparation：

1. 点击 **准备 Mentor 数据**；
2. features ready 后点击 **训练 MentorArtifact**；
3. 工具箱加载并验证 artifact，而不只检查 `.pt` 是否存在；
4. `MentorArtifact: READY` 后 Student 才可运行。

对应 CLI 阶段为：

```powershell
lnl mentor prepare --config <teacher-config> --output-dir <mentor-dir>
lnl mentor train --config <teacher-config> --output <mentor-artifact.pt>
lnl run --recipe <student-recipe> --check-data
```

MentorNet reproduction 可能要求外部准备、identity-compatible 的正式 artifact。此时仍显示 readiness，但不提供本地准备按钮；不能用 smoke artifact 冒充正式 artifact。Student 不会在 artifact 缺失时偷偷读取 clean labels 并训练 Mentor。

## 13.1 PCSE、DLD 与 CAL 前置条件

- **PCSE**：当前只接受身份兼容且角色明确的 UPM 主模型、监督 CE 模型或 Co-teaching peer A 模型。可用 `LNL_PCSE_SOURCE_RUN` 提供来源运行目录，但仍必须满足 YAML 中的 identity 检查；不能把任意 checkpoint 当作合法来源。
- **DLD**：formal/paper-oriented 配置需要真实、已验证的 pretrained feature extractor。Toolbox 不会自动联网下载 torchvision 权重；random frozen 模型仅用于 engineering/smoke，不能视为正式 pretrained source。
- **CAL**：需要 external aligned clean/noisy label artifact，而不是 pretrained model。clean vector 仅用于 identity、alignment 和 evaluation，不参与 optimizer 更新。

这些卡片的 `READY`、`NEEDS_INPUT` 或 `INVALID` 状态决定当前配置是否可运行。满足条件后可恢复 Run；不要把“有 recipe”或 smoke 成功理解为已完成论文数值复现。

## 14. 常见问题

### 数据集未登记或只有登记状态

进入 **本地数据集 → 1 · 登记本机路径**。登记后继续 **2 · 实际加载 train/test**；需要证明训练链路时再做 **3 · 训练 1 epoch 验证**。

### 方法“不兼容”或“需补充条件”

查看 reason code 和 required inputs。“不兼容”表示硬合同不满足；“需补充条件”表示应补方法先验、manifest、checkpoint 或 YAML 字段。不要反复点击 Run 绕过检查。

### UNKNOWN 怎么填

只填写从官方说明、元数据、外部 manifest 或本地测量中确定的事实；不知道时保持 Unknown。真实噪声率和方法噪声率先验不能混用。

### MentorArtifact NOT READY

到 **论文方法 → MentorNet** 查看三阶段状态。支持本地 preparation 时依次点击 **准备 Mentor 数据**、**训练 MentorArtifact**；正式 reproduction 若要求外部 artifact，应按页面说明提供匹配产物。

### 配置错误

阅读日志中的具体字段、路径和 preflight 错误，修正项目 YAML 后重新 Validate，再做 Dry-run。错误未解决前不要启动训练。

### `lnl` 找不到

确认环境已激活并安装项目。源码 checkout 可临时使用：

```powershell
$env:PYTHONPATH = "src"
python -m lnl_toolbox.cli.main --help
```

### Smoke 成功是否代表复现论文

不代表。Smoke 只证明短预算工程链路可用；论文复现还取决于正式数据、预算、模型、外部 artifact、fidelity 和实验协议。

### 训练结束先看什么

先看 `final_metrics.json` 和 `resolved_config.yaml`，再用 `metrics.jsonl` 检查过程，并确认 `best.pt`、`last.pt` 和 data/noise artifacts。需要恢复时回到 **运行管理** 先做恢复检查。
