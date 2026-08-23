# LNL Toolbox

LNL Toolbox 是一个面向 Learning with Noisy Labels（噪声标签学习）实验的可复现工具箱。使用者可以通过统一的 `lnl` 命令检查环境、浏览实验、组合组件、预检配置、运行训练、恢复 checkpoint，以及查看知名论文在 toolbox 中的具体实现。

## 1. 安装

要求 Python 3.10 或更高版本。先克隆仓库，再在独立环境中安装训练依赖：

```powershell
git clone https://github.com/seanlee-lile/LNL-toolbox.git
Set-Location LNL-toolbox
conda create -n lnl-toolbox python=3.11 -y
conda activate lnl-toolbox
python -m pip install -e ".[train]"
```

安装完成后确认统一入口可用：

```powershell
lnl --help
```

启动本地数据管理网页（默认自动打开浏览器）：

```powershell
lnl web
```

主控制台为 `http://127.0.0.1:8765/`；Recipe/YAML 编辑子页面为
`http://127.0.0.1:8765/recipe`。使用 `lnl web --no-open` 可只启动服务。

原有的 `lnl-train`、`lnl-clean-train`、`lnl-inspect-data` 和 `lnl-make-noise` 命令仍然保留。

### 准备 CIFAR 数据

内置 CIFAR recipe 读取官方 Python pickle，不会静默下载数据。请从 CIFAR 官方页面下载
`cifar-10-python.tar.gz` 或 `cifar-100-python.tar.gz`，解压后整理为：

```text
data/cifar10/data_batch_1 ... data_batch_5, test_batch, batches.meta
data/cifar100/train, test, meta
```

然后执行一次完整但不训练的预检：

```powershell
lnl run cifar10-symmetric-ce-smoke --dry-run
```

真实 UCI workflow 的数据准备命令见对应 reproduction 文档；数据和训练产物都不应提交到 Git。

### 登记本机数据集

数据路径可以保存在机器本地目录中，不必写入共享 YAML：

```powershell
lnl data register lab-cifar10 --adapter cifar10 --root F:/datasets/cifar10
lnl data list
lnl data status lab-cifar10
lnl data path lab-cifar10
lnl data inspect lab-cifar10
lnl data verify lab-cifar10 --recipe cifar10-clean-smoke
lnl run cifar10-symmetric-ce-smoke --data lab-cifar10
```

`list/status/path` 显示所有受支持数据集的 readiness 和本机位置；`inspect` 必须由正式
adapter 成功加载 train/test 后才标记 `ready`。`verify` 会先执行同一检查，再实际完成
1 epoch 并保存独立的 `training_verified` 证据。可用 `register/show/remove` 管理登记。
目录默认保存在 `%LOCALAPPDATA%/lnl-toolbox/datasets.json`，不会提交到 Git；
也可用 `LNL_DATA_CATALOG` 指向另一台机器自己的目录文件。训练过程不会自动下载数据。

## 2. Quick Start

新用户推荐按下面的顺序操作：

```powershell
lnl doctor
lnl list experiments --profile smoke
lnl run cifar10-symmetric-ce-smoke --dry-run
lnl run cifar10-symmetric-ce-smoke
```

`lnl run <source>` 中的 `<source>` 可以是内置 recipe 名称，也可以是 YAML
配置文件路径：

```powershell
lnl run cifar10-symmetric-ce-smoke
lnl run configs/experiment/cifar10_symmetric_ce_smoke.yaml
```

原有的显式写法继续兼容，但不再是主要推荐形式：

```powershell
lnl run --recipe cifar10-symmetric-ce-smoke
lnl run --config configs/experiment/cifar10_symmetric_ce_smoke.yaml
```

### 第一步：检查环境

```powershell
lnl doctor
```

意义：确认 Python、PyTorch、CUDA、配置目录和输出位置是否可用。`FAIL` 项会给出修复方向；CUDA 不可用时仍可使用 CPU。

### 第二步：浏览可运行实验

```powershell
lnl list experiments --profile smoke
```

意义：查看适合快速验证的实验。默认输出为带编号和中文标签的分块列表，其中：

- `recipe`：每个编号条目的名称，可以直接用于后续命令；
- `规模`：`smoke` 用于快速验证，`reproduction` 用于论文规模实验；
- `方法`：实验实现的方法或主要组件；
- `执行器`：toolbox 实际采用的训练生命周期；
- `训练轮数`：配置计划运行的 epoch 数。

如果需要把列表交给 PowerShell 或其他脚本处理，可以输出稳定的 TSV：

```powershell
lnl list experiments --profile smoke --format tsv
```
- `RECIPE`：可以直接运行的配置名称；
- `PROFILE`：`smoke` 用于快速验证，`reproduction` 用于论文规模实验；
- `METHOD`：实验实现的方法或主要组件；
- `RUNNER`：toolbox 实际采用的训练生命周期；
- `IMPLEMENTATION`：组件、workflow 或可直接调用的方法实现状态；
- `FIDELITY`：smoke、工程配置或论文导向配置的忠实度；
- `REPRODUCTION`：数值复现是否实际执行，不能由配置名称推断；
- `AVAILABILITY`：`runnable` 可直接运行，`conditional` 仍需外部 artifact；
- `EPOCHS`：配置的训练轮数。

第一次建议选择 `cifar10-symmetric-ce-smoke`。

### 第三步：预检实际运行内容

```powershell
lnl run cifar10-symmetric-ce-smoke --dry-run
```

意义：解析并验证配置、runner、数据路径、外部 artifact 和必要依赖，然后显示
运行计划。`--dry-run` 不启动训练，也不创建 checkpoint。数据尚未准备、只想检查
配置结构时，可以显式使用 `--no-check-data`；此时不能据此判断正式训练已经就绪。

请重点确认：

- `标签来源` 是否为预期噪声；
- `最佳模型依据` 是否使用 validation，而不是 test；
- `执行器` 是否与方法一致；
- `输出根目录` 是否正确。

`lnl validate` 保留给配置开发、CI 和高级排错，不是普通用户每次运行前的必经步骤。

### 第四步：开始训练

```powershell
lnl run cifar10-symmetric-ce-smoke
```

意义：按 recipe 启动完整实验，并保存 resolved config、指标、噪声 manifest 和 checkpoint。

## 3. 查看知名论文 Config

列出当前已经具有可运行配置的论文：

```powershell
lnl papers list
```

意义：只列出仓库中已有可运行配置的论文。默认按论文分块展示标题、出处、profile、实现保真度、执行器和建议命令；脚本处理时可追加 `--format tsv`。

查看某篇论文在 toolbox 中的详细实现：

```powershell
lnl papers show dual-t
lnl papers show jocor
lnl papers show fine
```

意义：了解论文核心机制、toolbox 生命周期、论文概念与 YAML 字段的对应关系、标签使用边界、已知实现差异和推荐命令。

查看论文的原始配置：

```powershell
lnl papers config dual-t --profile smoke --variant cifar10-sym20
```

查看补齐绝对路径后的配置：

```powershell
lnl papers config dual-t --profile smoke --variant cifar10-sym20 --resolved
```

只获取配置文件路径：

```powershell
lnl papers config dual-t --profile smoke --variant cifar10-sym20 --path-only
```

`smoke` 表示缩小规模的通路验证；`reproduction` 表示论文规模配置，但不自动代表已经复现论文报告的数值。请同时查看 `implementation_status`、`configuration_fidelity`、`reproduction_status`、`availability` 和论文详情中的限制说明。

## 4. 使用自定义 YAML

复制一份现有配置后，可以直接把 YAML 路径作为 source：

```powershell
lnl run configs/experiment/my_experiment.yaml --dry-run
lnl run configs/experiment/my_experiment.yaml
```

临时修改参数时重复使用 `--set PATH=VALUE`，无需复制 YAML：

```powershell
lnl run cifar10-symmetric-ce-smoke `
  --set trainer.epochs=20 `
  --set optimizer.lr=0.01 `
  --set seed=42
```

`--set` 只能覆盖已经存在的 dotted path，路径拼错会在训练前失败。临时实验使用
`--set`；需要长期保存、评审或复用的配置应创建 YAML 或使用 `lnl compose create`。

每份可运行配置都应明确声明：

```yaml
execution:
  runner: supervised
```

专用 runner 包括 Co-teaching、CNLCU、Dual-T、T-Revision、UPM、DLD、DivideMix、LEND、PCSE、Importance Reweighting 等完整方法生命周期。请以 `lnl list experiments` 的 `RUNNER` 字段为准；未知方法、未知 runner 或专用配置被送入错误 runner 时，toolbox 会在训练前失败，不会静默改跑普通监督实验。

MentorNet 等依赖外部训练 artifact 的 recipe 默认不会出现在直接可运行列表中；使用 `lnl list experiments --include-conditional` 查看，并先运行 `lnl validate` 获取缺失 artifact 的明确提示。

PCSE 的真实 CIFAR profile 同样属于 conditional workflow：它要求一个严格匹配的 UPM
`main_best` checkpoint 和 noise manifest。先阅读
`papers/pcse/reproduction.md`，准备 source artifact 并设置
`LNL_PCSE_SOURCE_RUN`；缺失或 identity 不匹配时，`validate` 和 dry-run 会在训练前失败。

兼容的显式写法和训练预算快捷参数仍然可用：

```powershell
lnl run --config configs/experiment/my_experiment.yaml --epochs 5
lnl run --config configs/experiment/my_experiment.yaml --output-dir artifacts/my-run
```

## 5. 恢复训练

```powershell
lnl resume artifacts/runs/20260802-120000
```

默认读取 `last.pt`。若要从最佳 checkpoint 恢复：

```powershell
lnl resume artifacts/runs/20260802-120000 --checkpoint best
```

意义：自动读取运行目录中的 `resolved_config.yaml` 和 checkpoint。恢复时会检查配置、噪声映射和组件身份，避免把 checkpoint 接到不兼容实验。

已完成的 run 执行 `lnl resume` 是严格 no-op；中断的 run 才会从所选 checkpoint
继续。

## 6. Sweep、比较与报告

顺序运行多个独立 seed：

```powershell
lnl sweep cifar10-symmetric-ce-smoke --seeds 1 2 3 4 5
```

每个 seed 都是通过 `ExperimentService` 执行的普通 run，拥有独立的配置、artifact、
checkpoint 和结果。再次执行同一 sweep 时，已完成任务跳过，失败任务重试，中断且有
checkpoint 的任务继续。

matrix sweep 使用 YAML 描述研究维度：

```yaml
version: 1
base:
  recipe: cifar10-symmetric-ce-smoke
matrix:
  noise.rate: [0.2, 0.4]
  optimizer.lr: [0.1, 0.01]
seeds: [1, 2, 3]
```

先预览展开计划，再执行和检查状态：

```powershell
lnl sweep experiment.yaml --dry-run
lnl sweep experiment.yaml
lnl sweep status artifacts/sweeps/<sweep-id>
```

比较完成的 run：

```powershell
lnl compare artifacts/sweeps/<sweep-id>
lnl compare artifacts/sweeps/<sweep-id> `
  --group-by method `
  --group-by noise.rate
```

每组输出 metric、`n`、mean、std、median、min 和 max，并在可比组内检查数据、
模型、增强、选择划分与 Noise Manifest。生成复用同一比较结果的报告：

```powershell
lnl report artifacts/sweeps/<sweep-id>
```

输出目录包含 `report.md`、`summary.csv` 和 `summary.json`。

## 7. 如何理解运行产物

典型运行目录包含：

| 文件 | 意义 |
| --- | --- |
| `resolved_config.yaml` | 本次真正使用的完整配置 |
| `environment.json` | Python、PyTorch、CUDA 和设备信息 |
| `noise_manifest.npz` | 不可变的噪声标签映射 |
| `noise_summary.json` | 噪声类型、请求噪声率和实际噪声率 |
| `metrics.jsonl` | 每个 epoch 的训练与验证记录 |
| `final_metrics.json` | 最终测试指标和模型选择信息 |
| `last.pt` | 最新训练状态，用于继续训练 |
| `best.pt` | 按验证指标选出的最佳 checkpoint |

在 `final_metrics.json` 中重点检查：

- `selection_split`：最佳模型使用哪个数据划分选择；
- `test_selection_leakage`：应为 `false`；
- `test_accuracy`：最终测试集指标；
- `noise.effective_train_subset_actual_rate`：实际训练子集中的噪声比例。

## 8. 浏览底层组件

```powershell
lnl list components
lnl list components --kind loss
lnl list components --kind batch_selector
```

意义：按类型查看可组合的 loss、selector、pipeline 和 parameter-update 组件。默认展示每个组件的能力与论文关联；脚本处理时可追加 `--format tsv`。组件不一定等于完整论文方法；完整论文入口应优先从 `lnl papers list` 或 `lnl list experiments` 获取。

## 9. 组合组件并生成配置

先查看指定 runner 的组合边界：

```powershell
lnl compose list --runner supervised
lnl compose list --runner dual_t
```

`supervised` 会列出可以替换的 loss、selector、parameter update 和 pipeline 规则；Dual-T、FINE、Co-teaching 等专用 runner 只展示完整 recipe，不允许被错误拆装。

从内置 smoke recipe 生成一份新配置：

```powershell
lnl compose create `
  --base cifar10-symmetric-ce-smoke `
  --loss gce `
  --selector small_loss `
  --keep-rate 0.6 `
  --output configs/experiment/my_gce_small_loss.yaml
```

意义：保留 base recipe 的数据、模型和训练规模，把损失改为 GCE，并让每个 batch 只使用损失较小的 60% 样本。生成前会检查组件名称、runner、构造参数和跨槽位冲突；已有输出文件不会被覆盖，原 recipe 也不会被修改。

检查并预览生成结果：

```powershell
lnl compose check --config configs/experiment/my_gce_small_loss.yaml
lnl run --config configs/experiment/my_gce_small_loss.yaml --dry-run
```

自定义组合只代表工程实验，不自动对应论文方法或论文复现结果。

## 10. 开发与验证

运行全部测试：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

当前测试覆盖配置路由、噪声 manifest、checkpoint、resume、专用论文生命周期和 clean-label 泄漏防护。

进一步了解仓库内部结构：

- [开发指南](docs/development-guide.md)
- [逐文件职责](docs/file-map.md)
- [数据流指南](docs/data-flow-guide.md)
- [项目管理指南](docs/project-management-guide.md)
- [长期架构与实验路线](toolbox-architecture.md)
- [论文实现进度](papers/implement/paper-reproduction-progress.md)

本地数据放在 `data/`，训练产物放在 `artifacts/`；二者均不应提交到 Git。
