# LNL Toolbox CLI 使用手册

本手册面向使用命令行开展实验的科研与工程用户，按“准备数据 → 检查配置 → 预演 → 执行 → 查看结果”组织。命令示例在仓库根目录执行；请将示例数据路径替换为本机真实路径。

需要图形界面时，运行 `lnl web`，参见 [Web 说明](../web/README.md)。Scratch 使用独立命令入口，不等同于正式 Experiment。

## 目录

- [1. Quick Start](#1-quick-start)
- [2. 安装与 CLI 入口](#2-安装与-cli-入口)
- [3. 核心概念](#3-核心概念)
- [4. 推荐工作流](#4-推荐工作流)
- [5. 核心命令参考](#5-核心命令参考)
- [6. Dataset 工作流](#6-dataset-工作流)
- [7. Paper / Recipe / Config](#7-paper--recipe--config)
- [8. Prerequisites 与失败诊断](#8-prerequisites-与失败诊断)
- [9. MentorNet](#9-mentornetproducer-与-student-分开)
- [10. 参数 Sweep](#10-参数-sweep)
- [11. Compare 与 Report](#11-compare-与-report)
- [12. Scratch CLI](#12-scratch-cli独立生命周期)
- [13. Legacy / 专用入口](#13-legacy--专用入口)
- [14. Practical Tips](#14-practical-tips)
- [15. 常用实验示例](#15-常用实验示例)
- [16. Exit Status 与输出文件](#16-exit-status-与输出文件)
- [17. Advanced / 机器可读输出](#17-advanced--机器可读输出)

## 1. Quick Start

已安装环境后，先检查环境并查找短实验：

```powershell
lnl doctor
lnl list experiments --profile smoke
lnl papers config gce --profile smoke --path-only
```

准备对应数据，再按同一 recipe 依次检查、预演和执行：

```powershell
lnl validate --recipe gce-cifar10-noise02-smoke --check-data
lnl run --recipe gce-cifar10-noise02-smoke --dry-run
lnl run --recipe gce-cifar10-noise02-smoke --output-dir artifacts/runs/gce-smoke
```

中断且尚未完成时使用 `lnl resume artifacts/runs/gce-smoke`；完成后使用 `lnl compare artifacts/runs/gce-smoke` 或 `lnl report artifacts/runs/gce-smoke --output-dir artifacts/reports/gce-smoke` 查看结果。

数据路径必须满足 recipe 的配置要求；缺少数据或前置输入时，先处理诊断，不要跳过检查。普通命令 exit code 0 表示命令成功完成，不表示论文数值复现成功。

Smoke 是工程链路快速检查；正式配置提供更完整的实验预算，但配置存在、预演通过或训练完成都不等于论文数值已复现。

## 2. 安装与 CLI 入口

需要 Python 3.10 或更高版本。建议使用隔离环境：

```powershell
conda create -n lnl-toolbox python=3.11
conda activate lnl-toolbox
python -m pip install -e ".[train]"
lnl --help
```

训练依赖包括 PyTorch 等；GPU 支持取决于本机安装的 PyTorch 与驱动。不要把系统中另一个 Python 的测试结果当作当前环境结果。

源码开发时，也可使用等价模块入口：

```powershell
$env:PYTHONPATH = "src"
python -m lnl_toolbox.cli.main --help
```

找不到 `lnl` 时先确认已激活安装项目的环境，而不是重复安装到其他解释器。

## 3. 核心概念

| 名称 | 含义 |
| --- | --- |
| Paper | 论文方法及其实现、限制和复现说明 |
| Recipe | catalog 中有 ID 的预设实验 |
| Config | 实际加载的 YAML 实验配置，可来自 recipe 或自己的文件 |
| Dataset | adapter 加载的数据；登记 alias 是本机路径与属性的引用 |
| Experiment | 使用配置开展的一次正式实验执行 |
| Run directory | 某次运行的配置、日志、指标和 checkpoint 所在目录 |
| Checkpoint | 可供相应 runner 恢复的模型与训练状态文件 |
| Artifact | 方法所需或产生的外部输入，例如 MentorArtifact、标签文件 |
| Result | 已完成运行的最终指标，供 compare/report 汇总 |

“已实现”“工程流程可运行”“论文方法对齐（paper-oriented）”“条件性正式流程（conditional）”与“数值复现已验证”是不同结论。以当前 catalog 和对应 reproduction 文档为准。

## 4. 推荐工作流

```text
doctor → 选择 recipe/config → 准备并检查数据
       → 补齐方法前置输入 → validate --check-data
       → run --dry-run → run → compare/report
                                  ↓
                           未完成运行可 resume
```

每次正式执行都会重新检查当前外部状态。上次检查通过，不保证后来被移动或替换的数据、artifact、环境变量仍然有效。

## 5. 核心命令参考

### 5.1 doctor：环境诊断

**用途**：检查 Python、依赖、CUDA 与项目路径等环境条件；不会安装依赖或训练模型。

**用法**：

```powershell
lnl doctor
lnl doctor --config configs/experiment/cifar10_clean_smoke.yaml --check-data
```

**输入/选项**：`--config PATH` 增加配置相关检查，`--project-root PATH` 指定项目根目录，`--check-data` 增加数据检查。

**输出/退出**：逐项显示诊断结果；成功为 0，检查失败为 1。环境诊断不代替具体实验 preflight。

### 5.2 validate：验证配置与可选的数据检查

**用法**：

```powershell
lnl validate --recipe gce-cifar10-noise02-smoke
lnl validate --recipe gce-cifar10-noise02-smoke --check-data
```

**输入**：位置参数 SOURCE、`--recipe ID`、`--config PATH` 三选一。SOURCE 是 recipe ID 或 YAML 路径。

**选项**：`--data ALIAS` 使用已登记数据，`--project-root PATH` 指定项目根目录。

**行为**：

- 默认验证配置和方法约束，不进行完整的数据/compatibility/prerequisite preflight；部分方法配置检查仍可能读取 artifact，因此不能一概称为“纯静态检查”。
- `--check-data` 可能实际加载 train/test，检查 dataset contract、compatibility 以及适用的方法 prerequisite/consumer。
- 不训练模型，不等同于 `data verify`，也不生成训练验证证据。

**输出/退出**：显示配置、runner 和检查结果；正常成功为 0，捕获的配置或运行条件错误通常为 2，并输出诊断。缺失输入应按提示修正后重新检查。

### 5.3 run：预演或正式训练

**用法**：

```powershell
lnl run --recipe gce-cifar10-noise02-smoke --dry-run
lnl run --recipe gce-cifar10-noise02-smoke --output-dir artifacts/runs/gce-smoke
```

**输入**：SOURCE / `--recipe` / `--config` 三选一。

| 选项 | 作用 |
| --- | --- |
| `--data ALIAS` | 使用已登记数据；不保证该数据适合当前方法 |
| `--output-dir PATH` | 指定输出位置；以运行最终报告的实际目录为准 |
| `--dry-run` | 默认执行完整 preflight 并查看解析后的训练计划，不正式训练 |
| `--check-data` | 保留的显式检查选项；默认已经检查 |
| `--no-check-data` | 只允许与 dry-run 一起使用，跳过完整数据及相关 preflight |
| `--epochs N` | 支持的训练预算覆盖；多阶段方法可能有限制 |
| `--set PATH=VALUE` | 重复设置已有配置字段 |
| `--resume PATH` | 使用指定 checkpoint 恢复，需 runner 支持 |

**执行准入**：正式 run 在进入 runner 前重新验证配置、当前数据、compatibility 和适用的 prerequisite，包括 source 的 schema、身份及 consumer 校验。正式训练不能用 `--no-check-data` 绕过准入。

**输出/退出**：成功打印实际运行目录，正常退出 0；受控错误通常退出 2。未指定输出目录时使用对应 runner 的默认行为，不应假设所有方法采用同一个目录规则。

**注意**：简化 dry-run 仍保留配置级检查；它不是正式执行许可。预算覆盖的具体效果先看 dry-run；方法特例参见 reproduction 文档。

### 5.4 resume：继续已有运行

**用法**：

```powershell
lnl resume artifacts/runs/gce-smoke
lnl resume artifacts/runs/gce-smoke --checkpoint best
```

**输入**：RUN_DIR，目录中必须有原运行的 `resolved_config.yaml`。

**选项**：默认选择 last checkpoint；`--checkpoint best` 选择 best。具体恢复能力和文件要求由 runner 决定。

**行为**：

- 已完成运行严格 no-op，不重跑训练。
- 未完成运行在真正恢复前重新检查当前数据、compatibility 和 prerequisite。
- 不自动重跑 producer，不创建新实验身份，不重置 checkpoint。
- `lnl resume RUN_DIR` 不接受扩 epoch 参数，只按原 resolved config 恢复。

**输出/退出**：显示恢复结果或已完成提示；正常为 0，受控错误通常为 2。

`lnl run --config ... --resume CHECKPOINT` 是“显式配置 + checkpoint”，不等同于 `lnl resume RUN_DIR`。若要调整预算，复制/修改 YAML，先 dry-run，再按对应 runner 的恢复合同执行前一种方式；不能保证所有方法都允许这样扩展。

## 6. Dataset 工作流

### 6.1 登记与发现

```powershell
lnl data register my-cifar10 --adapter cifar10 --root "F:\datasets\cifar10"
lnl data register my-heart --adapter uci_binary --path "D:\datasets\statlog-heart\heart.dat"
lnl data list
lnl data status my-cifar10
lnl data path my-cifar10
lnl data show my-cifar10
```

目录 adapter 使用 `--root`，单文件 adapter 使用 `--path`，不要同时携带无关 source 参数。`--labels` 仅用于确实需要该输入的 adapter，例如 CIFAR-N，不应用于其他 adapter。

登记写入用户级 catalog，不复制数据；登记成功不代表已加载、训练验证通过、兼容或方法 READY。Windows 路径含空格或中文时用引号包围。

### 6.2 inspect 与 declare

```powershell
lnl data inspect my-cifar10
lnl data declare my-cifar10 --noise-origin unknown
```

`inspect` 实际加载数据，检查 train/test 与 profile，并可更新登记状态；不训练模型。

`declare` 写入系统不能自动确定的数据事实。可声明噪声来源、噪声率及其来源、clean labels 等；完整选项见 `lnl data declare --help`。Unknown 可以保持 Unknown，不要为了通过检查伪造属性。

数据集真实噪声率与方法 noise prior 不是同一个概念。方法专用 checkpoint、artifact 和参数应放在方法配置中，而非 DatasetProfile。

### 6.3 verify 与训练

```powershell
lnl data verify my-cifar10 --output-dir artifacts/data-verification/my-cifar10
lnl methods compatible --dataset my-cifar10
lnl run --recipe gce-cifar10-noise02-smoke --data my-cifar10 --dry-run
```

默认 verify 使用短训练检查加载和训练链路，写入运行产物及训练验证证据，因此不是只读查询。自选 `--recipe` 时还受该方法预算覆盖与前置要求限制，不应假设所有多阶段流程都只执行一个总 epoch。

compatibility 常见输出为 `COMPATIBLE`（兼容）、`COMPATIBLE_WITH_REQUIREMENTS`（需补充条件）、`INCOMPATIBLE`（不兼容）。它与 prerequisite 的 READY/NEEDS_INPUT/INVALID 是不同层次；请同时读取 reason，而不是只看状态名称。

### 6.4 删除登记

```powershell
lnl data remove my-cifar10
```

删除 catalog 登记，不删除原始数据。登记/declare/inspect/verify 都可能更新本机状态；需要审计时先记录原有登记信息。

## 7. Paper / Recipe / Config

```powershell
lnl papers list
lnl papers show pcse
lnl papers config gce --profile smoke --path-only
lnl list experiments --profile reproduction --include-conditional
lnl list components
```

Paper ID 标识论文方法，recipe ID 标识具体实验。先查询可用配置，不要把 Paper ID、recipe ID、文件路径混用。

使用自己的 YAML：

```powershell
Copy-Item configs/experiment/cifar10_clean_smoke.yaml my-experiment.yaml
lnl validate --config my-experiment.yaml --check-data
lnl run --config my-experiment.yaml --dry-run
lnl run --config my-experiment.yaml --output-dir artifacts/runs/my-experiment
```

复制已有 YAML 只是一个起点；也可自行维护配置。修改数据、模型或预算后重新验证。图形编辑入口见 [Web 说明](../web/README.md)。

Formal/reproduction profile 可能是 conditional；recipe 存在不表示 prerequisite 已准备好，更不表示数值复现已验证。

## 8. Prerequisites 与失败诊断

正式执行检查外部输入的真实状态，而不仅是路径存在：

```text
CONFIGURED → PATH_EXISTS → SCHEMA_VALID
           → IDENTITY_COMPATIBLE → CONSUMER_VALIDATED
```

这些是校验层级，不代表每个失败都会显示全部层级。`PATH_EXISTS` 不等于 READY。

| 方法 | 需要提供什么 | 供给位置与边界 |
| --- | --- | --- |
| PCSE | 可严格恢复的预训练分类器 source | 按 YAML source 配置；支持 UPM、CE/supervised best、Co-teaching peer A。run-directory source 可通过配置指定的 LNL_PCSE_SOURCE_RUN 提供 |
| DLD | 真实预训练特征提取器 | 按 extractor/source YAML；正式配置要求真实权重，不自动联网下载。UPM run-directory source 使用对应配置的 LNL_DLD_SOURCE_RUN |
| CAL | 对齐的 external clean/noisy label artifact | YAML 的 noise.path、clean_key、noisy_key；clean vector 只用于身份、对齐和评估，不参与 optimizer 更新 |
| MentorNet | 冻结 MentorArtifact | 使用 preparation 配置/专门生产流程，详见下一节 |

PCSE 不接受任意 checkpoint：role、schema、数据身份、manifest 与映射必须符合 consumer 要求。DLD 的 random/repository frozen model 仅适用于工程/smoke 路径，不是正式 pretrained source。环境变量只在对应 YAML source 配置使用它时生效。

CLI 失败诊断会尽可能保留：

- prerequisite type/role、status、level、reason；
- configured source、config field、source env；
- expected、preparation 与 action；
- 多个 blocker，而非只显示第一个。

状态一般为 READY / NEEDS_INPUT / INVALID。配置级早期错误可能只有未评价的 context，不要把它误读成已完成 readiness 判定。缺失字段以实际诊断为准，不应猜测或伪造。

处理顺序：读 reason → 找到 config field/source env → 提供真实文件或运行目录 → 重新 checked validate/dry-run。INVALID 也可能是可修复的 source 问题；不要修改数据事实来掩盖方法输入缺失。

例如，下面是诊断字段的示意，不代表每个方法都返回相同内容：

```text
错误: 前置输入未就绪
Prerequisite: <实际 requirement 名称>
status: NEEDS_INPUT
reason: <缺失原因>
config field: <实际配置字段>
action: <提供输入的具体方式>
```

详细边界见 [PCSE](../papers/pcse/reproduction.md)、[DLD](../papers/dld/reproduction.md) 和 [论文模块覆盖说明](../papers/lnl-26-paper-module-coverage.md)。

## 9. MentorNet：producer 与 Student 分开

```text
Mentor features → Frozen MentorArtifact → Student
```

先查询需求：

```powershell
lnl mentor status --recipe mentornet-dd-cifar100-symmetric04-smoke
```

下例有意展示 Mentor producer 与 Student 可属于不同数据域；实际使用必须以 status 给出的 artifact identity 要求为准。

```powershell
lnl mentor prepare --config configs/experiment/mentornet_dd_teacher_cifar10_symmetric04.yaml --output-dir data/mentornet/cifar10-symmetric04-seed20260729
lnl mentor train --config configs/experiment/mentornet_dd_teacher_cifar10_symmetric04.yaml --output data/mentornet/cifar10-symmetric04-seed20260729/mentor_artifact.pt
lnl run --recipe mentornet-dd-cifar100-symmetric04-smoke --output-dir artifacts/runs/mentor-student
```

上述 producer 与 Student 可属于不同数据域，不能随意替换成同一个 alias。先看 status 指出的 artifact 身份与路径要求；路径存在仍需验证。

smoke 可使用 toolbox preparation；正式 reproduction 可能要求外部 identity-compatible artifact，不能承诺所有正式 MentorNet artifact 都能本地自动生成。Student 不会在失败时自动训练 producer，也不能用 clean labels 静默构造不符合合同的 Mentor 训练输入。

`mentor status` 是 readiness 查询：READY 为 0，未就绪通常为 1；这不同于正式 run 的错误退出。MentorNet preparation 的具体要求先用 `lnl papers show mentornet` 与 `lnl mentor status` 查询。

## 10. 参数 Sweep

先检查整批计划：

```powershell
lnl sweep cifar10-clean-smoke --data my-cifar10 --seeds 1 2 3 --matrix 'optimizer.lr=[0.001,0.003]' --output-dir artifacts/sweeps/ce-lr --dry-run
```

去掉 dry-run 执行：

```powershell
lnl sweep cifar10-clean-smoke --data my-cifar10 --seeds 1 2 3 --matrix 'optimizer.lr=[0.001,0.003]' --output-dir artifacts/sweeps/ce-lr
lnl sweep status artifacts/sweeps/ce-lr
```

`--matrix` 为已有字段指定取值数组，可重复使用；`--set` 设置固定覆盖值。`--seeds` 指定重复实验种子。

dry-run 保留整批计划检查。actual Sweep 在每个成员真正执行时做 admission；无效成员进入现有 failure isolation，其他合法成员可继续。有成员失败时最终返回非零，不要把“继续执行”理解为全部成功。

Sweep 是多个实验运行的编排，不是集群 scheduler。

复用同一计划与输出目录时会检查已有 Sweep 状态，跳过已完成成员，并按支持情况恢复中断成员。不要用单次运行的 `lnl resume` 代替 Sweep 计划管理。

## 11. Compare 与 Report

```powershell
lnl compare artifacts/runs
lnl compare artifacts/runs --group-by method,noise.rate --format json
lnl report artifacts/runs --output-dir artifacts/reports/first
```

输入可为运行目录或包含运行的根目录。compare 汇总结果并可按配置字段分组；report 生成 report.md、summary.csv、summary.json。

建议显式指定独立报告目录，避免把报告写入输入目录。损坏、失败或不可比较的记录可能被排除并产生提示；检查公平比较条件，不要只看均值。`--strict` 可使存在被排除记录时返回非零。

## 12. Scratch CLI：独立生命周期

Scratch 面向组件/公式组合，不使用 `lnl scratch`：

```powershell
python -m lnl_toolbox.scratch.cli list-blocks
python -m lnl_toolbox.scratch.cli validate src/lnl_toolbox/scratch/recipes/examples/ce_synthetic.yaml
python -m lnl_toolbox.scratch.cli run src/lnl_toolbox/scratch/recipes/examples/ce_synthetic.yaml --output artifacts/scratch/ce-demo --max-epochs 1 --max-batches 2
```

validate 检查 Scratch recipe/graph；run 使用独立 executor，不等同于正式 Experiment preflight。运行预算上限适合开发检查，不证明论文复现。

常见输出有 recipe.yaml、resolved_recipe.yaml、progress.json、stdout.log、metrics.jsonl。不要假设它与正式 runner 共享 checkpoint/resume 合同。参见 [Scratch 说明](../src/lnl_toolbox/scratch/README.md)。

## 13. Legacy / 专用入口

新实验优先使用统一 `lnl`；以下现存入口不应被误认为同一套参数：

| 入口 | 定位 |
| --- | --- |
| lnl-train | 正式实验兼容入口，继承统一执行 admission |
| lnl-clean-train | clean-only 专用/兼容入口，适用于其既有 clean baseline 流程 |
| lnl-inspect-data | 专用数据检查工具 |
| lnl-make-noise | 噪声 manifest 生产工具，不是训练命令 |

模块入口 `python -m lnl_toolbox.cli.clean_train` 仍可用于专用 clean 流程。具体参数先看各自 help；不要直接套用统一 run 的所有参数。参见 [clean 专用指南](clean-training-guide.md)。

## 14. Practical Tips

- 路径先用真实小数据验证；不要将示例路径原样当作已准备的数据。
- 用明确的输出目录区分实验和 Sweep；以程序报告的实际 run directory 为准。
- 多阶段方法预算先看 dry-run，不要假设一个 epochs 参数覆盖所有阶段。
- 保留 resolved config、seed、日志与 source provenance，避免移动后无法恢复。
- 不要把 best.pt 当作所有场景的继续训练入口；last.pt 通常更适合继续，仍需遵守 runner 合同。
- UNKNOWN 不等于 false；数据事实、方法 prior、方法 artifact 分开管理。
- 权重、数据与生成运行不应提交到代码仓库。
- 训练成功与数值复现是不同验收目标。

## 15. 常用实验示例

对已登记 CIFAR 数据使用正式统一链路：

```powershell
lnl validate --recipe gce-cifar10-noise02-smoke --data my-cifar10 --check-data
lnl run --recipe gce-cifar10-noise02-smoke --data my-cifar10 --dry-run
lnl run --recipe gce-cifar10-noise02-smoke --data my-cifar10 --output-dir artifacts/runs/gce-local
lnl compare artifacts/runs/gce-local
lnl report artifacts/runs/gce-local --output-dir artifacts/reports/gce-local
```

查找配置、登记数据、Mentor preparation 和 Sweep 的可复制示例分别见第 6、7、9、10 节。恢复未完成运行：

```powershell
lnl resume artifacts/runs/gce-local
```

对自定义 YAML 的未完成运行，先检查再使用明确 checkpoint：

```powershell
lnl run --config my-experiment.yaml --dry-run
lnl run --config my-experiment.yaml --resume artifacts/runs/my-experiment/last.pt --output-dir artifacts/runs/my-experiment-resumed
```

后一个示例要求已有匹配 checkpoint 且 runner 支持，不是从零创建模型的方式。如果目标是继续一个已有、未完成的 Toolbox Run，优先使用 `lnl resume RUN_DIR`。`lnl run --resume CHECKPOINT` 表示“显式 config + checkpoint”的一次执行，不等同于恢复原 Run directory。各方法的专属 artifact 和恢复限制见相应 [论文目录](../papers)。

## 16. Exit Status 与输出文件

| 场景 | 常见退出码 |
| --- | --- |
| help 或命令成功完成 | 0 |
| validate/run/resume 捕获的输入、配置或执行条件错误 | 2 |
| 参数解析错误 | 2 |
| doctor 检查失败、mentor status 未就绪 | 1 |
| actual Sweep 存在失败成员 | 1 |
| compare/report 的 strict 排除检查失败 | 1 |
| Scratch 执行错误 | 1 |

不同命令的非零含义不同；同时读取 stderr/诊断。不要假设所有未处理异常都遵循同一错误文本或退出码。

stdout 主要用于正常结果，stderr 主要用于错误诊断；readiness 查询等专用命令也可能在正常输出中报告未就绪。Result/report 是指标及汇总，不是 checkpoint，不能用于恢复训练。

正式运行常见文件：

| 文件 | 含义 |
| --- | --- |
| final_metrics.json | 最终指标 |
| metrics.jsonl | 逐阶段/epoch 指标日志，不是图片 |
| best.pt | 最佳验证 checkpoint（若 runner 提供） |
| last.pt | 最近 checkpoint（若 runner 提供） |
| resolved_config.yaml | 实际执行配置 |

具体方法可能产生额外 artifact，见对应 reproduction 文档。

## 17. Advanced / 机器可读输出

```powershell
lnl list experiments --profile smoke --format json
lnl list components --format tsv
lnl data list --format json
lnl data status my-cifar10 --format json
lnl data inspect my-cifar10 --format json
lnl methods compatible --dataset my-cifar10 --format json
lnl sweep status artifacts/sweeps/ce-lr --format json
```

只使用各命令 help 明确支持的格式。没有统一的 run/validate/resume JSON 错误协议；不要把人类可读 diagnostics 当作稳定机器接口。`report` 的 JSON 是生成文件，不等于它支持 `--format json`。

进一步阅读：[CLI 简明教程](LNL-Toolbox-简明操作教程.md)、[配置说明](../configs/README.md)、[数据流指南](data-flow-guide.md)。方法级技术细节留在对应 reproduction 文档，本手册不替代论文或数值复现报告。
