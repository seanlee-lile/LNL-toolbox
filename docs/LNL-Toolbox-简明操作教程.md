# LNL Toolbox CLI 简明操作教程

## 1. 这份教程适合谁

本文是 CLI 用户和已经熟悉基本流程用户的命令速查，主线是：

```text
doctor → list → validate → dry-run → run → resume
```

如果你第一次使用 Toolbox，或希望通过 Web 完成 Quick Start、数据集登记、compatibility、论文方法、YAML、Sweep、运行管理和 MentorNet preparation，请先阅读 [Web-first 完整用户手册](toolbox-usage-manual.md)。

`smoke` 是短预算工程链路检查；`reproduction` 是更完整的正式配置。两者都不自动代表已经复现论文数值，具体 fidelity 和状态以 catalog 为准。

## 2. 安装与 CLI 入口

项目要求 Python 3.10 或更高版本。在仓库根目录安装训练依赖：

```powershell
conda activate lnl-toolbox
python -m pip install -e ".[train]"
lnl --help
```

如果 console script 暂时不可用，可使用完全等价的模块入口：

```powershell
$env:PYTHONPATH = "src"
python -m lnl_toolbox.cli.main --help
```

下文中的 `lnl` 均可替换为 `python -m lnl_toolbox.cli.main`。

## 3. 最短实验命令链

```powershell
lnl doctor
lnl list experiments --profile smoke
lnl validate --recipe <recipe> --check-data
lnl run --recipe <recipe> --dry-run --check-data
lnl run --recipe <recipe> --output-dir <dir> --check-data
```

命令显示 `exit code 0` 表示成功完成，不是“退出程序”。正式运行前建议始终执行 Validate 和 Dry-run。

## 4. 查看可用实验和组件

```powershell
lnl list experiments
lnl list experiments --profile smoke
lnl list experiments --profile reproduction
lnl list experiments --include-conditional
lnl list components
lnl list components --kind loss
```

实验列表中的 `METHOD` 是用户选择的方法，`RUNNER` 是实际执行生命周期的 runner。底层 component 存在不表示对应论文已经具备完整 workflow。

## 5. 查询论文配置

```powershell
lnl papers list
lnl papers show <paper-id>
lnl papers config <paper-id> --profile smoke --path-only
lnl papers config <paper-id> --profile reproduction --path-only
lnl papers config <paper-id> --profile smoke --resolved
```

注意区分：

- `implementation_status`：实现到了组件、workflow 还是用户可运行方法。
- `configuration_fidelity`：当前配置与论文协议的接近程度。
- `reproduction_status`：是否完成所需数值实验。
- `availability`：是否还依赖外部数据或 artifact。

## 6. Validate、Dry-run 和 Run

### Validate

```powershell
lnl validate --recipe <recipe> --check-data
```

Validate 不训练模型。使用 `--check-data` 时，它可能实际加载数据，并执行 dataset、compatibility 和 method-specific preflight，因此不只是读取 YAML 的“静态检查”。

### Dry-run

```powershell
lnl run --recipe <recipe> --dry-run --check-data
```

Dry-run 显示解析后的训练计划，用于核对 runner、dataset、method requirements、训练预算、设备和输出位置，不正式训练。

### Run

```powershell
lnl run --recipe <recipe> --output-dir artifacts/runs/<name> --check-data
```

使用全新或明确的输出目录，避免混入旧运行状态。`--epochs` 的具体作用可能因 runner 而异；先在 Dry-run 中检查最终预算。方法特例应查看对应论文 reproduction 文档。

## 7. 使用自己的 YAML

你可以通过 Web 创建 YAML，也可以在 CLI 中复制已有配置，或使用 compose：

```powershell
# 方式一：复制接近目标的配置
Copy-Item configs/experiment/<base>.yaml configs/experiment/my-experiment.yaml

# 方式二：由内置 recipe 创建新配置
lnl compose create --base <recipe> --output configs/experiment/my-experiment.yaml
```

修改后依次执行：

```powershell
lnl validate --config configs/experiment/my-experiment.yaml --check-data
lnl run --config configs/experiment/my-experiment.yaml --dry-run --check-data
lnl run --config configs/experiment/my-experiment.yaml `
  --output-dir artifacts/runs/my-experiment --check-data
```

不要直接覆盖内置 recipe。修改论文预算、模型、数据或算法参数后，也不要继续把结果描述成未改变的 paper-exact 配置。

## 8. 数据集与 Compatibility 的 CLI 入口

常用数据命令：

```powershell
lnl data list
lnl data register <alias> --adapter <adapter> --root <path>
lnl data inspect <alias>
lnl data verify <alias>
lnl data status <alias>
```

单文件数据使用 `--path`；需要额外标签文件的 adapter 使用 `--labels`。登记只保存位置，`inspect` 才会实际加载数据，`verify` 会运行一轮训练验证。

查询已登记数据集可用的方法：

```powershell
lnl methods compatible --dataset <alias>
```

Compatibility 三种结果：

- `COMPATIBLE`：可以直接使用。
- `NEEDS_INPUT`：方法适用，但还缺先验、manifest、checkpoint 或配置字段；补齐后可再次 Validate/Dry-run。
- `INCOMPATIBLE`：数据或配置不满足硬要求。

无法确认的数据事实可以保持 Unknown，不要为通过检查伪造属性。数据集真实噪声率不等于某个方法的噪声率先验。

完整的数据登记和声明流程见 [Web-first 完整用户手册](toolbox-usage-manual.md)。

## 9. MentorNet 前置准备

MentorNet Student 的正式顺序是：

```text
Mentor 特征 → 冻结的 MentorArtifact → Student 训练
```

先检查 Student recipe 的 readiness：

```powershell
lnl mentor status --recipe <mentornet-student-recipe>
```

支持 Toolbox preparation 的 smoke workflow 可执行：

```powershell
lnl mentor prepare --config <teacher-config> --output-dir <mentor-dir>
lnl mentor train --config <teacher-config> --output <mentor-artifact.pt>
lnl mentor status --recipe <mentornet-student-recipe>
lnl run --recipe <mentornet-student-recipe> --check-data
```

Student 不会在 artifact 缺失时偷偷训练 Mentor。正式 reproduction 可能要求外部准备且 identity-compatible 的 artifact，并不保证都能由本地 smoke producer 自动生成；以 `mentor status`、`papers show` 和对应配置说明为准。

## 10. Resume

先确认运行目录包含 `resolved_config.yaml` 和所选 checkpoint：

```powershell
lnl resume artifacts/runs/<name> --checkpoint last
lnl resume artifacts/runs/<name> --checkpoint best
```

`last.pt` 通常表示最近完成的训练边界；`best.pt` 表示按该方法验证规则保存的最佳 checkpoint。具体 runner 对 best/last 的恢复能力可能不同，已完成运行也可能明确返回无需恢复。复杂方法先查看对应 reproduction 文档或在 Web“运行管理”中检查恢复内容。

不要在这份速查中假设所有 runner 都能用同一种 checkpoint 或统一 epoch 覆盖继续训练。

## 11. Sweep、报告和比较

```powershell
lnl sweep --recipe <recipe> --seeds 1 2 3 `
  --output-dir artifacts/sweeps/<name> --dry-run
lnl report artifacts/runs/<name>
lnl compare artifacts/sweeps/<name>
```

参数 Sweep 的字段矩阵和完整运行管理也可在 Web 中完成。

## 12. 常见输出文件

| 文件 | 含义 |
|---|---|
| `final_metrics.json` | 最终指标摘要 |
| `metrics.jsonl` | 逐阶段或逐 epoch 日志，不是图片 |
| `best.pt` | 最佳 checkpoint |
| `last.pt` | 最近 checkpoint |
| `resolved_config.yaml` | 本次实际执行的配置 |

具体方法可能生成 transition、posterior、feature 或其他额外 artifact。请查看 [论文文档索引](../papers/README.md) 以及仓库中对应方法已有的 `reproduction.md`，不要要求所有 runner 产生相同文件集合。

## 13. 常见错误

### 找不到 `lnl`

确认激活了安装项目的环境，并执行 `python -m pip install -e ".[train]"`。也可以临时使用模块入口。

### 数据集未登记或路径错误

先执行 `lnl data list`、`lnl data inspect <alias>`，再用 `validate --check-data`。不要通过关闭检查把配置指向不相关数据。

### Compatibility 不是 COMPATIBLE

阅读 reason code。`NEEDS_INPUT` 需要补输入；`INCOMPATIBLE` 表示存在硬合同冲突。Unknown 可以保持 Unknown。

### Conditional recipe 不能运行

先准备错误信息中要求的数据、manifest、checkpoint 或 artifact。MentorNet 使用 `lnl mentor status/prepare/train`；其他方法查看对应 reproduction 文档。

### `--epochs` 被拒绝或效果不符合预期

不同 runner 的预算字段不同。先 Dry-run；多阶段方法应修改 YAML 中明确的阶段字段，并查看对应方法文档。

### Smoke 成功是否代表论文复现

不代表。Smoke 只证明短预算工程链路可用。

## 14. 进一步阅读

- Web Quick Start、数据登记、YAML、Sweep、运行管理：[LNL Toolbox 使用指南](toolbox-usage-manual.md)
- 论文实现和复现文档索引：[papers/README.md](../papers/README.md)
- 已有方法文档示例：[Dual-T](../papers/dual-t/reproduction.md)、[DLD](../papers/dld/reproduction.md)、[DivideMix](../papers/dividemix/reproduction.md)、[UPM](../papers/upm/reproduction.md)、[PCSE](../papers/pcse/reproduction.md)

执行任何长训练前，先以当前 `lnl --help`、具体 recipe 的 Validate/Dry-run 输出和对应 reproduction 文档为准。
