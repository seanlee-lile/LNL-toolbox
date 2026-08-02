# LNL Toolbox

LNL Toolbox 是一个面向 Learning with Noisy Labels（噪声标签学习）实验的可复现工具箱。使用者可以通过统一的 `lnl` 命令检查环境、浏览实验、预检配置、运行训练、恢复 checkpoint，以及查看知名论文在 toolbox 中的具体实现。

## 1. 安装

要求 Python 3.10 或更高版本。推荐在已有 PyTorch 环境中以 editable 模式安装：

```powershell
conda activate pytorch
python -m pip install -e ".[train]"
```

安装完成后确认统一入口可用：

```powershell
lnl --help
```

原有的 `lnl-train`、`lnl-clean-train`、`lnl-inspect-data` 和 `lnl-make-noise` 命令仍然保留。

## 2. 第一次运行：按这五步操作

### 第一步：检查环境

```powershell
lnl doctor
```

意义：确认 Python、PyTorch、CUDA、配置目录和输出位置是否可用。`FAIL` 项会给出修复方向；CUDA 不可用时仍可使用 CPU。

### 第二步：浏览可运行实验

```powershell
lnl list experiments --profile smoke
```

意义：查看适合快速验证的实验。输出中的重要字段：

- `RECIPE`：可以直接运行的配置名称；
- `PROFILE`：`smoke` 用于快速验证，`reproduction` 用于论文规模实验；
- `METHOD`：实验实现的方法或主要组件；
- `RUNNER`：toolbox 实际采用的训练生命周期；
- `EPOCHS`：配置的训练轮数。

第一次建议选择 `cifar10-symmetric-ce-smoke`。

### 第三步：检查配置

```powershell
lnl validate --recipe cifar10-symmetric-ce-smoke
```

意义：训练前检查 YAML、执行器、模型、loss、优化器和路径。检查不会开始训练，也不会下载数据。

若还要确认本地数据目录存在：

```powershell
lnl validate --recipe cifar10-symmetric-ce-smoke --check-data
```

### 第四步：预览实际运行内容

```powershell
lnl run --recipe cifar10-symmetric-ce-smoke --dry-run
```

意义：显示最终数据路径、噪声来源、模型、训练轮数、设备、最佳模型选择依据和输出目录。`--dry-run` 不创建训练产物。

请重点确认：

- `标签来源` 是否为预期噪声；
- `最佳模型依据` 是否使用 validation，而不是 test；
- `执行器` 是否与方法一致；
- `输出根目录` 是否正确。

### 第五步：开始训练

```powershell
lnl run --recipe cifar10-symmetric-ce-smoke
```

意义：按 recipe 启动完整实验，并保存 resolved config、指标、噪声 manifest 和 checkpoint。

## 3. 查看知名论文 Config

列出当前已经具有可运行配置的论文：

```powershell
lnl papers list
```

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

`smoke` 表示缩小规模的通路验证；`reproduction` 表示论文规模配置，但不自动代表已经复现论文报告的数值。请同时查看 `fidelity` 和论文详情中的限制说明。

## 4. 使用自定义 YAML

复制一份现有配置后，可以通过文件路径预检和运行：

```powershell
lnl validate --config configs/experiment/my_experiment.yaml
lnl run --config configs/experiment/my_experiment.yaml --dry-run
lnl run --config configs/experiment/my_experiment.yaml
```

每份可运行配置都应明确声明：

```yaml
execution:
  runner: supervised
```

专用 runner 包括 `coteaching`、`dual_t`、`multi_model`、`cwd`、`fine`、`instance_transition`、`importance_reweighting` 和 `pcse`。未知方法、未知 runner 或专用配置被送入错误 runner 时，toolbox 会在训练前失败，不会静默改跑普通监督实验。

临时覆盖训练轮数或输出位置：

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

## 6. 如何理解运行产物

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

## 7. 浏览底层组件

```powershell
lnl list components
lnl list components --kind loss
lnl list components --kind batch_selector
```

意义：查看可组合的 loss、selector、pipeline 和 parameter-update 组件。组件不一定等于完整论文方法；完整论文入口应优先从 `lnl papers list` 或 `lnl list experiments` 获取。

## 8. 开发与验证

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
