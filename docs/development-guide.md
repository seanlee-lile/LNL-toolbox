# LNL Toolbox 开发指南

## 1. 开发环境

推荐为本项目创建独立 Conda 环境，避免与 Cleanlab、Ultralytics 等其他项目的依赖混在一起。

```powershell
conda create -n lnl-toolbox python=3.11 -y
conda activate lnl-toolbox
```

当前已验证的环境为：

- Python 3.11.15
- PyTorch 2.5.1
- torchvision 0.20.1
- CUDA Runtime 12.1
- NumPy 2.0.1
- PyYAML 6.0.3
- Pillow 11.1.0

## 2. 安装依赖

### NVIDIA GPU（CUDA 12.1）

先安装对应 CUDA 版本的 PyTorch：

```powershell
pip install torch==2.5.1 torchvision==0.20.1 `
  --index-url https://download.pytorch.org/whl/cu121
```

再安装其余固定版本依赖：

```powershell
pip install -r requirements.txt
```

最后以 editable 模式安装项目：

```powershell
pip install -e . --no-deps
```

### 仅使用 CPU

```powershell
pip install torch==2.5.1 torchvision==0.20.1 `
  --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
```

`requirements.txt` 固定版本用于复现实验；`pyproject.toml` 中的版本范围用于描述工具包兼容性。
两条安装路径都包含需要 RandAugment 的公开方法依赖；修改依赖时必须同步维护二者。

## 3. 检查环境

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

预期 GPU 环境输出 PyTorch 版本、`True` 和显卡名称。如果 CUDA 不可用，训练配置中的 `device: auto` 会回退到 CPU。

## 4. 准备数据

数据文件不提交到 Git。请使用以下目录：

```text
data/
├── cifar10/
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── test_batch
│   └── batches.meta
└── cifar100/
    ├── train
    ├── test
    └── meta
```

`data/` 已加入 `.gitignore`。

下载 CIFAR 官方 Python 版本后，只复制解压目录中的数据文件，不要提交压缩包或数据集。
统一 CLI 不会在训练时自动下载数据；普通用户使用 `lnl run <source> --dry-run`
执行与正式训练一致的数据 preflight。`lnl validate --check-data` 主要用于配置开发、
CI 和高级排错。

## 5. 运行测试

项目当前使用 Python 标准库 `unittest`，不要求安装 pytest：

```powershell
python -m unittest discover -s tests -v
```

提交代码前应保证完整测试全部通过。

## 6. 运行训练

人工启动实验时，可以无参数进入交互向导：

```powershell
lnl-train
```

向导先选择现有 YAML，再覆盖常用参数并展示最终配置；确认默认为否。只要命令中出现任意参数，就完全使用 argparse 非交互模式，不会在批处理过程中等待输入。

快速 GPU/CPU smoke test：

```powershell
lnl-train --config configs/experiment/cifar10_smoke.yaml
```

等价的模块运行方式：

```powershell
python -m lnl_toolbox.cli.train `
  --config configs/experiment/cifar10_smoke.yaml
```

完整 CIFAR-10 配置：

```powershell
lnl-train --config configs/experiment/cifar10_clean.yaml
```

每次运行会在 `artifacts/runs/` 下生成：

- `resolved_config.yaml`：最终采用的配置；
- `environment.json`：Python、PyTorch、CUDA、GPU 和 seed；
- `metrics.jsonl`：逐 epoch 指标；
- `last.pt`：模型、优化器、RunState 和算法状态；
- `final_metrics.json`：最终测试指标。

## 7. 恢复训练

`--epochs` 表示恢复后的总 epoch 目标，而不是额外训练轮数。例如，从已经完成 2 个 epoch 的 checkpoint 继续到第 3 个 epoch：

```powershell
lnl-train `
  --config configs/experiment/cifar10_smoke.yaml `
  --resume artifacts/runs/cifar10_smoke/last.pt `
  --epochs 3
```

恢复运行会继续使用 checkpoint 所在的运行目录，并在原有 `metrics.jsonl` 后追加指标。

## 8. 无需激活 Conda 的运行方式

自动化工具可以在激活环境后直接调用解释器：

```powershell
python -m unittest discover -s tests -v
```

不要把本机 Python、数据或临时运行目录的绝对路径写进项目配置、源码或用户文档。

## 9. 开发约定

- 通用协议放在 `src/lnl_toolbox/core/`，不要在核心层写死 PyTorch、CIFAR 或标签噪声假设。
- 数据适配器放在 `data/` Python 模块中，原始数据保存在仓库根目录的 `data/`。
- 模型、loss、算法和 evaluator 保持可替换，论文方法优先实现为插件。
- Dataset 必须返回稳定的全局 `index`，方便后续噪声清单和样本选择算法对应样本。
- 新功能应补充 `unittest`，并保证已有测试不回归。
- 实验结果、checkpoint 和数据文件不得提交到 Git。

## 10. 依赖维护

只有源码直接使用、或运行训练必需的包才应加入 `requirements.txt`。更新关键依赖后，需要重新运行：

```powershell
python -m unittest discover -s tests -v
lnl-train --config configs/experiment/cifar10_smoke.yaml
```

不要直接使用完整 Conda 环境的 `pip freeze` 覆盖 requirements；那会把其他项目的 Cleanlab、OpenCV、Pandas、Ultralytics 等无关依赖一起带入。

## 11. 统一实验服务与质量门禁

用户入口统一调用 `ExperimentService`；runner 只负责训练生命周期，服务层负责共享
preflight、标准运行元数据和 `final_metrics.json` Result Contract。新增 runner 时应在
`training/runners.py` 注册自身的预算路径与 `RunPlan`，CLI 不得按论文名称分支。

常用命令：

```powershell
lnl run cifar10-symmetric-ce-smoke --set trainer.epochs=2 --dry-run
lnl run configs/experiment/cifar10_symmetric_ce_smoke.yaml
lnl sweep cifar10-symmetric-ce-smoke --seeds 1 2 3
lnl sweep sweep-spec.yaml --dry-run
lnl sweep status artifacts/sweeps/example
lnl compare artifacts/sweeps/example
lnl report artifacts/sweeps/example
```

所有活动 YAML 使用 `schema_version: 1`。完整 recipe 写 `kind: experiment`，参数片段写
`kind: fragment`。不要在公开 recipe 中写本机 CIFAR/MNIST root；先执行
`lnl data register`，只有在同一 adapter 登记多份数据时才通过 `--data <alias>` 选择。
配置读取会拒绝未知顶层字段和旧式 `noise.type`、顶层 `epochs/batch_size/learning_rate`。
原始迁移前配置位于 `archive/configs-legacy-2026-08-21/`，仅供恢复和比对。

`--dry-run` 默认验证数据与外部 artifact，只跳过真正训练和 checkpoint 写入；仅当
数据尚未准备时才使用 `--no-check-data`。Sweep matrix 必须调用
`core.config_overrides` 的 dotted-path override，不得自行修改嵌套配置。每个任务由
seed、resolved override 和 config hash 共同标识。

比较层把 `group_by` 视为允许变化的研究维度，把 `require_equal` 视为同组公平比较
必须一致的条件。Noise Manifest 只在同一 seed 和可比条件下跨方法核对；不同 seed
不要求共享 manifest。`lnl report` 必须直接消费 `lnl compare` 使用的同一比较结果，
不得另写 aggregation 或 fairness 逻辑。

CI 在 Python 3.10 和 3.12 上执行 Ruff、完整 unittest、CLI 测试和 coverage；发布 job
分别从 wheel 与 sdist 安装，并验证 `lnl --help`、公开 recipe discovery 和 VolMinNet
smoke 配置预检。
# 注册第三方数据源（2026-08-18）

新增数据集时实现 `DatasetAdapter`：`validate(DataSpec)` 负责路径和布局检查，`load(DataSpec, split, seed=...)` 返回 `RawDatasetSplit`。随后在 `create_dataset_registry()` 的数据源注册函数中登记名称和别名。适配器不得包含论文方法、训练阶段或 optimizer 逻辑。

必须保证：global index 在 shuffle/subset/view 后稳定；observed 与 clean label 分开保存；真实噪声训练 split 不向 batch 暴露 clean label；缺失数据只输出本地准备说明，不下载、不回退。需要持久化预处理或 split 状态的适配器可提供 `identity_artifacts()`，由统一服务写入并验证 run-local JSON。

runner 只允许调用 `prepare_experiment_data()`，不得直接导入 CIFAR reader、`TorchCifarDataset` 或自行构造 `DataLoader`。新增/修改 runner 后运行 `tests/test_data_service.py` 的静态门禁。

## 机器本地数据登记与可用性证据（2026-08-20）

`data/local_catalog.py` 只保存本机路径和验证证据，不保存数据本身。新增适配器登记项时，
必须维持 `registered -> layout_validated -> training_verified` 的严格语义；文件签名改变后，
既有训练证据必须转为 `verification_stale`。不得因为注册成功或一次 `validate()` 成功就宣称
数据可训练。

CLI、Web、doctor、dry-run、run 和 sweep 不得直接读取 `DatasetRegistry`、
`LocalDatasetCatalog` 或自行判断文件布局；它们必须调用 `DataService`。新增数据管理能力时，
readiness 检查必须真实加载 train/test，不能用 `Path.exists()` 代替 adapter 校验。Web
focused tests 必须与 `tests/` 一起进入 CI。

每个新增文件格式至少需要：官方结构的临时 fixture、通过 `ExperimentService` 的实际
1 epoch、`data_manifest.json`、epoch 指标，以及训练 batch 不泄漏 clean target 的既有门禁。
本机真实数据若可用，还应再通过 `lnl data verify`。外部数据格式依据应来自数据集发布方
或论文课题组官方仓库；未经上述实验，不在文档中标记为训练可用。
