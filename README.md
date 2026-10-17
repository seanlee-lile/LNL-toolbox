# LNL Toolbox

LNL Toolbox 用于噪声标签学习（Learning with Noisy Labels）实验，提供论文配置查询、数据检查、训练、恢复与结果比较。可通过统一 CLI 或本地 Web 界面使用。

## 核心特点

- 查询论文方法、实验 recipe 与实现限制。
- 使用内置 recipe 或自定义 YAML，检查配置后执行实验。
- 登记本机数据，并区分数据事实、方法输入与兼容性。
- 支持运行恢复、参数 Sweep 和结果报告。

## Quick Start

需要 Python 3.10 或更高版本。克隆仓库后，在独立环境中安装：

```powershell
git clone https://github.com/seanlee-lile/LNL-toolbox.git
Set-Location LNL-toolbox
conda create -n lnl-toolbox python=3.11
conda activate lnl-toolbox
python -m pip install -e ".[train]"
```

检查环境并发现实验：

```powershell
lnl doctor
lnl list experiments --profile smoke
lnl papers config gce --profile smoke --path-only
```

先按所选配置准备真实数据；路径登记和加载检查见 [CLI 使用手册：Dataset 工作流](docs/toolbox-usage-manual.md#6-dataset-工作流)。不要将示例路径当作已准备的数据，也不要假设训练会自动下载数据。

然后检查、预演并运行同一 recipe：

```powershell
lnl validate --recipe gce-cifar10-noise02-smoke --check-data
lnl run --recipe gce-cifar10-noise02-smoke --dry-run
lnl run --recipe gce-cifar10-noise02-smoke --output-dir artifacts/runs/gce-smoke
```

checked validate 和默认 dry-run 不训练模型，但可能实际加载数据并读取外部 artifact。正式 run 会重新检查当前执行条件；失败时先按诊断补齐输入。

## CLI / Web 入口

命令行：

```powershell
lnl --help
lnl papers list
lnl papers show pcse
```

SOURCE 可以是 recipe ID 或 YAML 路径，也可显式使用 `--recipe` 或 `--config`，三选一。完整用法见 [CLI 使用手册](docs/toolbox-usage-manual.md)，日常速查见 [CLI 简明操作教程](docs/LNL-Toolbox-简明操作教程.md)。

本地 Web：

```powershell
lnl web --host 127.0.0.1 --port 8765
```

默认打开浏览器，主页为 `http://127.0.0.1:8765/`。Web 提供实验配置、论文查询和运行管理入口；Recipe 编辑器位于 `/recipe`。操作说明见 [Web README](web/README.md)。

## 论文与实验概览

当前 Paper Catalog 收录 26 篇方法，包括 GCE、Co-teaching、DivideMix、JoCoR、PCSE、DLD 等。具体 profile、可用性和限制以当前查询结果为准：

```powershell
lnl papers list
lnl list experiments --profile reproduction --include-conditional
```

论文方法不等于可任意组合的单个组件。方法说明与复现记录见 [论文目录](papers/README.md)；模块划分背景见 [26 篇论文模块覆盖说明](papers/lnl-26-paper-module-coverage.md)，其中历史状态不能替代当前 catalog 与方法文档。

## 实验流程

```text
doctor → discover → dataset → validate --check-data → dry-run → run
                                                               ↓
                                                   resume / compare / report
```

`discover` 和 `dataset` 表示配置发现与数据准备阶段，不是额外的顶层命令。继续已有未完成运行使用 `lnl resume RUN_DIR`；已完成运行的 resume 是 no-op。恢复、Sweep、比较、报告及产物说明统一见 [CLI 使用手册](docs/toolbox-usage-manual.md)。

## 复现边界

- **Smoke**：短预算工程检查，不代表论文性能。
- **Reproduction / formal profile**：较完整的正式实验配置，可能仍有实现差异或外部输入要求。
- **Conditional**：需先提供并验证方法所需资源；文件存在不等于 READY。
- **数值复现验证**：需要实际实验结果与论文目标对照，不能从配置存在、dry-run 或 smoke 成功推断。

PCSE、DLD、CAL、MentorNet 等存在专门的 source/artifact 要求；缺失或无效时可能无法运行。准备方式见 [Prerequisites 与失败诊断](docs/toolbox-usage-manual.md#8-prerequisites-与失败诊断) 和对应方法文档，不要用任意 checkpoint 替代指定输入。

## Documentation

| 文档 | 用途 |
| --- | --- |
| [CLI 使用手册](docs/toolbox-usage-manual.md) | 完整命令、数据工作流、前置输入、恢复与输出 |
| [CLI 简明操作教程](docs/LNL-Toolbox-简明操作教程.md) | 常用命令速查 |
| [Web README](web/README.md) | Web 启动和操作说明 |
| [论文目录](papers/README.md) | 方法说明与复现文档入口 |
| [配置说明](configs/README.md) | YAML 与 conditional recipe 合同 |
| [开发指南](docs/development-guide.md) | 开发与测试入口 |
| [数据流指南](docs/data-flow-guide.md) | 数据和实验流程的技术说明 |

组件组合使用 `lnl compose --help` 查找入口；不在首页展开完整教程。现有 legacy/专用命令仍保留，定位见 [CLI 使用手册](docs/toolbox-usage-manual.md#13-legacy--专用入口)。

## Development

在已安装开发环境的仓库根目录运行测试：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

提交前还应运行受影响模块的测试；Web 测试入口见 [Web README](web/README.md)。本地数据与生成运行通常放在 `data/` 和 `artifacts/`，不要将数据、权重或训练产物提交到 Git。
