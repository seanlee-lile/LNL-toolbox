# LNL Toolbox 简明操作教程

> 只讲怎么操作，以及每一步为什么要做

本教程对应 `integration` 分支、提交
`fcd3de376e0ab3847178600d3ea14c36f02bced1` 的实际命令行行为。
推荐按下面的顺序操作：

```text
doctor → list → validate → dry-run → run → resume
```

`smoke` 配置用于快速检查链路，论文配置用于表达更接近论文的实验协议；两者都不等于已经复现论文数值。

## 1. 开始之前：安装与环境检查

以下示例以 Windows PowerShell 为主，并假设当前目录是仓库根目录。

### 1.1 激活环境并安装

```powershell
conda activate pytorch
python -m pip install -e ".[train]"
```

`-e` 是 editable install：源码修改后通常不需要重新安装，同时会根据
`pyproject.toml` 生成正式的 `lnl` 命令。

如果你使用的环境名称不是 `pytorch`，请替换为自己的 Conda 环境名。本项目要求
Python 3.10 或更高版本。

### 1.2 找不到 `lnl` 时怎么办

先确认安装命令是在当前环境中执行的：

```powershell
Get-Command lnl
```

如果当前环境尚未生成 `lnl` 命令，可使用完全等价的模块入口：

```powershell
python -m lnl_toolbox.cli.main --help
```

本文后续优先写 `lnl`。需要使用模块入口时，把每条命令开头的 `lnl` 换成：

```text
python -m lnl_toolbox.cli.main
```

### 1.3 检查环境

```powershell
lnl doctor
```

这一步检查 Python、NumPy、PyYAML、PyTorch、torchvision、CUDA、项目根目录、配置目录和输出位置。它帮助尽早发现环境问题，但不会证明某个训练一定成功。

若还要检查某份配置引用的数据路径是否存在：

```powershell
lnl doctor --config configs/experiment/cifar10_symmetric_ce_smoke.yaml --check-data
```

`--check-data` 检查配置中的路径是否存在；它不代替真正的数据读取和训练 smoke。

## 2. 浏览可运行实验

### 2.1 列出 smoke 配置

```powershell
lnl list experiments --profile smoke
```

输出中常见字段如下：

| 字段 | 含义 |
|---|---|
| `RECIPE` | 内置配置的稳定名称，供 `--recipe` 使用 |
| `PROFILE` | `smoke`、`experiment` 或 `reproduction` |
| `DATASET` / `NOISE` | 数据集和噪声设置 |
| `METHOD` | 用户选择的方法 |
| `RUNNER` | 实际负责生命周期的执行器 |
| `IMPLEMENTATION` | 当前实现层级，如 component、workflow、user_ready |
| `FIDELITY` | 配置与论文设置的接近程度 |
| `REPRODUCTION` | 长训练或数值复现是否已完成 |
| `AVAILABILITY` | 当前是否可直接运行，或是否有外部条件 |
| `EPOCHS` | 能明确表示时显示训练轮数；多阶段方法可能显示 `-` |

适合第一次操作的 noisy-label 入门配置是：

```text
cifar10-symmetric-ce-smoke
```

它是两轮、少量样本的 CIFAR-10 symmetric-noise CE 链路检查，不是正式实验。

### 2.2 列出底层组件

```powershell
lnl list components
```

这会显示 loss、selector、transition estimator、parameter update policy 等可复用组件。组件可以被构造或测试，不代表对应论文的完整训练方法已经实现。

可以按 kind 筛选，例如：

```powershell
lnl list components --kind loss
```

### 2.3 为什么有些 recipe 默认不显示

需要预训练 teacher、外部 artifact 或额外数据的配置属于 conditional recipe，默认列表会隐藏它们：

```powershell
lnl list experiments --profile smoke --include-conditional
```

例如 MentorNet student workflow 需要事先准备 `MentorArtifact`，不是开箱即跑。运行前先验证：

```powershell
lnl validate --recipe mentornet-dd-cifar100-symmetric04-smoke
```

如果 artifact 缺失，`validate` 会报告所需路径。不要把 conditional 当成普通 runnable recipe。

## 3. 先检查，再运行

### 3.1 验证内置 recipe

```powershell
lnl validate --recipe cifar10-symmetric-ce-smoke
```

验证成功时会打印配置路径、runner 和项目根目录。`validate` 可以提前发现大量配置问题，包括：

- 未知 method 或 runner；
- 缺少必要的配置 mapping；
- 不支持的模型、optimizer、scheduler 或 plugin 名；
- 多阶段方法自身的阶段配置错误；
- conditional workflow 缺少外部 artifact；
- 配合 `--check-data` 时，配置引用的数据路径不存在。

它是静态预检，不会创建模型并完整跑过数据，因此不能替代 smoke 或正式训练。

### 3.2 使用 dry-run 看执行计划

```powershell
lnl run --recipe cifar10-symmetric-ce-smoke --dry-run
```

当前 dry-run 会验证配置并打印：

- 配置文件和项目根目录；
- runner；
- 数据集和数据路径；
- 标签/噪声来源；
- 模型；
- 训练轮数或阶段轮数；
- device；
- best checkpoint 的选择 split、target 语义和主指标；
- 输出根目录。

dry-run 不进入 `run_experiment`，因而不会启动训练，也不会生成 checkpoint 或训练输出目录。

### 3.3 正式启动一个 smoke

建议显式指定一个全新输出目录：

```powershell
lnl run `
  --recipe cifar10-symmetric-ce-smoke `
  --output-dir artifacts/runs/tutorial-ce-smoke
```

这样做便于定位 checkpoint，也避免和其他运行混在一起。如果目录中已有旧结果，请换一个新目录，不要直接覆盖。

需要同时检查数据路径时：

```powershell
lnl run --recipe cifar10-symmetric-ce-smoke --check-data --dry-run
```

## 4. 查找论文方法

### 4.1 列出 paper catalog

```powershell
lnl papers list
```

paper catalog 中四个状态不要混为一谈：

| 状态 | 回答的问题 |
|---|---|
| `implementation_status` | 代码实现到了组件、workflow，还是用户可运行的方法阶段 |
| `configuration_fidelity` | 某份配置与论文实验设置有多接近 |
| `reproduction_status` | 是否真的完成了所需长训练和数值复现 |
| `availability` | 当前可直接运行，还是依赖外部 artifact/数据 |

因此，“有 paper config”不等于“已经复现论文数值”，`user_ready` 也不自动表示论文全部实验均已完成。

### 4.2 查看 T-Revision

```powershell
lnl papers show t-revision
```

这会说明论文机制、Toolbox 生命周期、配置映射、实现路径、限制和推荐命令。

查看 smoke 配置的路径：

```powershell
lnl papers config t-revision --profile smoke --path-only
```

查看解析后的配置内容：

```powershell
lnl papers config t-revision --profile smoke --resolved
```

T-Revision 当前提供的是 Reweight-R workflow；smoke 通过或配置可运行，不代表已经完成正式数值和多 seed 复现。

### 4.3 查看 JoCoR

```powershell
lnl papers show jocor
lnl papers config jocor --profile smoke --resolved
lnl validate --recipe jocor-cifar10-symmetric05-smoke
lnl run --recipe jocor-cifar10-symmetric05-smoke --dry-run
```

JoCoR 是多模型生命周期，runner 为 `multi_model`。它不是把几个普通 Selector 任意拼起来得到的别名。

如果同一论文有多个配置，可用 `--variant` 精确选择；可用 profile 只有 `smoke` 和 `reproduction`：

```powershell
lnl papers config <paper-id> --profile reproduction --variant <variant> --path-only
```

## 5. 自定义实验

### 5.1 复制 YAML，而不是修改内置 recipe

先复制一份已经接近目标的配置：

```powershell
Copy-Item `
  configs/experiment/cifar10_symmetric_ce_smoke.yaml `
  C:/temp/my-lnl-experiment.yaml
```

修改副本后依次执行：

```powershell
lnl validate --config C:/temp/my-lnl-experiment.yaml
lnl run --config C:/temp/my-lnl-experiment.yaml --dry-run
lnl run --config C:/temp/my-lnl-experiment.yaml --output-dir C:/temp/my-lnl-run
```

这样可以保留内置 recipe 作为可比较基准，也避免把个人路径和实验参数写回仓库。

### 5.2 `--epochs` 不是对所有方法都相同

普通单阶段方法通常覆盖 `trainer.epochs`：

```powershell
lnl run --recipe cifar10-symmetric-ce-smoke --epochs 5 --dry-run
```

预览会显示训练轮数变为 5。

T-Revision 的统一参数只覆盖 revision 阶段：

```powershell
lnl run --recipe cifar10-t-revision-smoke --epochs 5 --dry-run
```

当前预览为：

```text
2/2/5 (stage1/classifier/revision)
```

也就是覆盖 `t_revision.revision.epochs`，不会修改 stage 1 和 classifier initialization。

Dual-T 和 PCSE 的 `--epochs` 含义不唯一，CLI 会拒绝：

```powershell
lnl run --recipe cifar10-dual-t-smoke --epochs 5 --dry-run
```

对此类多阶段方法，请复制 YAML，明确修改具体阶段，例如 Dual-T 的
`posterior_stage.epochs` 或 `final_stage.epochs`，再运行 `validate` 和 dry-run。

非 resumable 的 binary runner 也不支持统一 `--epochs`。不要笼统认为所有方法都能使用同一个 epoch override。

## 6. 恢复训练

### 6.1 从默认 checkpoint 恢复

```powershell
lnl resume artifacts/runs/tutorial-ce-smoke
```

`resume` 会从运行目录读取：

```text
resolved_config.yaml
last.pt
```

`last.pt` 通常表示最后一个已完成训练状态，适合继续尚未达到目标 epoch 的运行。

### 6.2 选择 best.pt

```powershell
lnl resume artifacts/runs/tutorial-ce-smoke --checkpoint best
```

这个参数只是在运行目录中选择 `best.pt` 交给对应 runner。它不保证每一种方法都允许从 best checkpoint 继续训练。

例如 T-Revision 明确只允许用自己的 `last.pt` 恢复阶段状态；把 `best.pt` 当作 resumable run state 会失败。复杂方法的 best checkpoint 可能只是评估 artifact。

### 6.3 增加总 epoch 后继续

`lnl resume` 本身没有 `--epochs` 参数。需要扩展训练目标时，应使用配置副本和显式 checkpoint：

```powershell
Copy-Item `
  artifacts/runs/tutorial-ce-smoke/resolved_config.yaml `
  C:/temp/tutorial-ce-extended.yaml

lnl run `
  --config C:/temp/tutorial-ce-extended.yaml `
  --resume artifacts/runs/tutorial-ce-smoke/last.pt `
  --epochs 5 `
  --output-dir artifacts/runs/tutorial-ce-smoke
```

先加 `--dry-run` 检查 epoch 语义，再移除它真正恢复。多阶段方法仍应修改 YAML 中具体阶段的 epoch 字段，而不是强行使用统一覆盖。

恢复时，runner 会校验自己拥有的配置、method identity、阶段状态及必要 artifact。改变数据、噪声 manifest、模型、阶段身份或 artifact provenance 等关键内容可能被拒绝，这是为了避免错误续跑。

达到配置目标后的 resume 是否严格 no-op 由具体 runner 决定。许多方法会直接返回已有完成结果，但这是方法级语义，不是所有 runner 的统一承诺。

## 7. 看懂输出结果

### 7.1 常见输出

不同 runner 不保证生成完全相同的文件。常见文件如下：

| 文件 | 它回答的问题 |
|---|---|
| `resolved_config.yaml` | 这次实际使用了什么配置 |
| `environment.json` | Python、PyTorch、设备等运行环境是什么 |
| `noise_manifest.npz` | 哪些 stable sample index 被映射成了什么 noisy label |
| `noise_summary.json` | 噪声率、manifest hash 等摘要是什么 |
| `metrics.jsonl` | 每个 epoch 或阶段产生了哪些指标 |
| `last.pt` | 最后完成到哪里，恢复训练需要什么状态 |
| `best.pt` | 按该方法选择规则保存的最佳模型或状态 |
| `final_metrics.json` | 最终 epoch、global step、best 指标和 test 指标是什么 |
| `training_curves.svg` | 配置启用曲线时，指标如何随 epoch 变化 |

clean run 不需要 noisy-label manifest；部分 runner 也可能使用自己的环境或摘要结构。因此判断运行是否完整时，应结合具体 method 文档，而不是机械要求所有文件都存在。

### 7.2 多阶段方法的额外 artifact

以 T-Revision 为例，它还会生成：

| 文件 | 用途 |
|---|---|
| `stage1_best.pt` | 第一阶段 noisy classifier 的最佳状态 |
| `posterior_snapshot.npz` | 按 stable index 对齐的 posterior 快照 |
| `transition_initial.npz` | 初始转移矩阵及 provenance |
| `stage2a_best.pt` | classifier initialization 阶段的最佳状态 |
| `transition_revised.npz` | revision 完成后的转移矩阵 |

Dual-T、PCSE、Importance Reweighting、PDL 等方法会有各自的 snapshot 或 transition/statistic artifact。普通 supervised run 不会生成这些文件。

### 7.3 如何确认 test 没有参与选模

先看 dry-run 的：

```text
最佳模型依据: <split> / targets=<target source> / primary=<metric>
```

再检查 `resolved_config.yaml` 中的 `evaluation.selection_split`，以及 runner 写入
`final_metrics.json` 的 selection 字段。推荐使用 validation 选择 best checkpoint，只在训练完成后用 clean test 做最终评估。

如果配置明确使用 test 选模，那就是评估协议本身包含 test selection；不能把对应结果描述成无 test leakage 的普通验证选择。

## 8. 常见问题

### `lnl` 命令找不到

确认激活了正确环境并执行：

```powershell
python -m pip install -e ".[train]"
```

临时可改用：

```powershell
python -m lnl_toolbox.cli.main --help
```

### 数据目录不存在

运行：

```powershell
lnl validate --recipe <recipe> --check-data
```

然后检查 recipe 的 resolved config 中 `data.root` 或 `data.path`。不要为了绕过检查而把配置指向不相关数据。

### 出现 unknown method

先运行：

```powershell
lnl list experiments
lnl list components
```

方法名必须由当前 runner registry 支持。底层 component 的名字不一定能直接作为完整 method 运行。

### 想要的 recipe 没显示

检查 profile、dataset 过滤条件和 conditional 列表：

```powershell
lnl list experiments --include-conditional
```

也可以通过 `lnl papers show <paper-id>` 查找论文绑定的真实 recipe。

### conditional recipe 无法 validate

阅读错误中给出的 artifact 或数据路径，先完成前置步骤。MentorNet 等方法不能在缺少 teacher artifact 时直接训练 student。

### `--epochs` 被拒绝

这通常表示方法有多个阶段，统一覆盖存在歧义。复制 YAML 并修改明确的阶段字段；T-Revision 是例外，它把统一覆盖定义为 revision epoch target。

### smoke 成功是否代表复现论文

不代表。smoke 只证明一条小规模链路能够运行。论文复现还可能要求正式数据规模、指定超参数、多 seed、mean/std 和论文级结果对照。

### 如何判断 test 是否参与选模

查看 dry-run 的“最佳模型依据”、resolved config 的 `evaluation.selection_split` 和最终指标中的 selection/leakage 字段。不要仅凭文件名 `best.pt` 推断选择协议。

## 9. 最短命令清单

```powershell
# 1. 检查环境
lnl doctor

# 2. 找一个小实验
lnl list experiments --profile smoke

# 3. 浏览组件
lnl list components

# 4. 验证配置
lnl validate --recipe cifar10-symmetric-ce-smoke

# 5. 只看计划，不训练
lnl run --recipe cifar10-symmetric-ce-smoke --dry-run

# 6. 运行到明确的新目录
lnl run --recipe cifar10-symmetric-ce-smoke `
  --output-dir artifacts/runs/tutorial-ce-smoke

# 7. 恢复 last.pt
lnl resume artifacts/runs/tutorial-ce-smoke

# 8. 查找论文方法
lnl papers list
lnl papers show t-revision
lnl papers config t-revision --profile smoke --path-only
```

## 10. CWD 五折复现协议与 FINE 报告边界

CWD 正式 recipe 会自动执行五个隔离 fold，并只在每个 fold 训练结束后读取其
held-out test：

```powershell
lnl validate --recipe cwd-cifar10-reproduction
lnl run --recipe cwd-cifar10-reproduction --dry-run
lnl run --recipe cwd-cifar10-reproduction `
  --output-dir artifacts/runs/cwd-five-fold
lnl resume artifacts/runs/cwd-five-fold
```

每个 fold 位于 `fold-0/` 至 `fold-4/`。根目录 `last.pt` 记录已完成 folds；恢复时
已完成 fold 严格跳过，被中断 fold 从自身 `last.pt` 继续。只有五折全部完成后才生成
`aggregate_metrics.json`，其中 `test_accuracy_mean/std` 仅来自五次最终 test。当前代码
状态是 **protocol-ready**；在实际跑完 200 epochs × 5 folds 前不得写成
protocol-completed。

FINE 当前保持 fixed-budget、test-final-only。历史记录中存在一次训练及 last-10
ledger 数值，但当前 runner 没有保存逐 epoch test 指标或最后十个模型，因此无法从
现有 artifact 重建 last-k。后续若要生成可审计 last-k，需另行批准保存最后十个模型，
并在训练全部结束后统一测试；不得恢复逐 epoch test 或让 test 参与选模。

Linux/macOS 下命令名称和参数相同；将 PowerShell 的反引号续行改为反斜杠，或写成单行即可。
