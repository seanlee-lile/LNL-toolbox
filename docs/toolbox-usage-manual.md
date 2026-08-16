# LNL Toolbox 实际操作手册

本文档基于统一 CLI 的发布合同整理。示例默认使用 Windows PowerShell、名为
`lnl-toolbox` 的 Conda 环境和用户自己的仓库根目录。

## 1. 启动环境

```powershell
git clone https://github.com/seanlee-lile/LNL-toolbox.git
Set-Location LNL-toolbox
conda create -n lnl-toolbox python=3.11 -y
conda activate lnl-toolbox
python -m pip install -e ".[train]"
```

安装后优先使用 `lnl`；若尚未生成 console script，可在源码 checkout 中临时设置
`$env:PYTHONPATH = "src"` 并使用 `python -m lnl_toolbox.cli.main`。

内置 CIFAR recipe 不自动下载数据。将官方 CIFAR-10 Python pickle 解压到
`data/cifar10/`，或将 CIFAR-100 解压到 `data/cifar100/`，再用
`lnl validate --recipe <recipe> --check-data` 核验。

## 2. 先检查环境和数据

```powershell
python -m lnl_toolbox.cli.main doctor --project-root . --check-data
```

看到 Python、PyTorch、CUDA、配置目录和输出目录均为 `[OK]` 后再运行实验。若只想检查 Python/PyTorch，不检查数据，去掉 `--check-data`。

## 3. 找到可运行的实验

```powershell
python -m lnl_toolbox.cli.main list experiments
python -m lnl_toolbox.cli.main list components
```

实验名来自输出列表，例如：

```text
cifar10-clean
apl-cifar10-noise02-smoke
mc-ldce-cifar10-reproduction
```

也可以直接使用 YAML，不依赖 recipe 名称。

## 4. 运行前校验

使用内置 recipe：

```powershell
python -m lnl_toolbox.cli.main validate --recipe cifar10-clean
```

使用自己的配置：

```powershell
python -m lnl_toolbox.cli.main validate `
  --config configs/experiment/cifar10_clean_smoke.yaml
```

如果需要同时确认数据路径：

```powershell
python -m lnl_toolbox.cli.main validate `
  --config configs/experiment/cifar10_clean_smoke.yaml --check-data
```

## 5. 先做 dry-run

```powershell
python -m lnl_toolbox.cli.main run `
  --config configs/experiment/cifar10_clean_smoke.yaml `
  --output-dir artifacts/test-runs/manual-clean-smoke `
  --dry-run
```

dry-run 会显示实际执行器、数据集、数据路径、模型、epoch、设备、验证集和输出目录，但不会训练。

## 6. 运行一个短实验

```powershell
python -m lnl_toolbox.cli.main run `
  --config configs/experiment/cifar10_clean_smoke.yaml `
  --output-dir artifacts/test-runs/manual-clean-smoke
```

本次实测完成 2 epochs，并生成：

```text
last.pt
best.pt
metrics.jsonl
resolved_config.yaml
training_curves.svg
```

终端会逐 epoch 输出 JSON。重点查看 `train_loss`、`validation_accuracy`、`test_accuracy` 和 `global_step`。

## 7. 从 checkpoint 恢复

推荐让统一 CLI 自动寻找运行目录中的 checkpoint：

```powershell
python -m lnl_toolbox.cli.main resume `
  artifacts/test-runs/manual-clean-smoke --checkpoint last
```

选择最佳模型进行最终评估：

```powershell
python -m lnl_toolbox.cli.main resume `
  artifacts/test-runs/manual-clean-smoke --checkpoint best
```

本次实测 `last` 恢复成功，并保持原有 epoch、step 和指标记录。

## 8. 运行论文配置

论文复现统一使用对应 YAML，不要把 smoke 配置当正式实验：

```powershell
python -m lnl_toolbox.cli.main run `
  --config configs/experiment/pdl_cifar10_reproduction.yaml `
  --output-dir artifacts/reproductions/pdl-cifar10-formal
```

其他论文同理，只替换配置文件和输出目录，例如：

```powershell
python -m lnl_toolbox.cli.main run `
  --config configs/experiment/mc_ldce_cifar10_reproduction.yaml `
  --output-dir artifacts/reproductions/mc-ldce-cifar10-formal

python -m lnl_toolbox.cli.main run `
  --config configs/experiment/l2rw_cifar10_reproduction.yaml `
  --output-dir artifacts/reproductions/l2rw-cifar10-formal
```

正式实验前必须先 dry-run；不要用 `--epochs` 覆盖论文配置，除非只是短跑诊断。

## 9. 训练中检查结果

另开 PowerShell：

```powershell
Get-Content artifacts/reproductions/<run>/metrics.jsonl -Wait
```

或者查看最新 checkpoint 是否存在：

```powershell
Get-Item artifacts/reproductions/<run>/last.pt
Get-Item artifacts/reproductions/<run>/best.pt
```

训练是否完成以 checkpoint 中的 `completed_epoch`、`global_step` 和 `metrics.jsonl` 为准，不以终端窗口是否仍打开为准。

## 10. 常见问题

- `No module named lnl_toolbox`：确认当前目录是仓库根目录，并设置 `$env:PYTHONPATH = "src"`。
- CUDA 不可用：先运行 `doctor`；必要时在 YAML 中显式设置 `trainer.device: cpu` 做 smoke。
- 找不到数据：检查 YAML 的 `data.root`，并用 `doctor --check-data` 或 `validate --check-data`。
- 恢复失败：必须使用同一 YAML、同一输出目录和对应的 `last.pt`；不要用不同配置 resume。
- 训练结果异常：先检查 `resolved_config.yaml`、noise manifest、数据 split、模型名和验证集来源，再判断是否为算法问题。
- 论文算法不得读取 clean training labels；clean labels 只能通过显式、审计过的 trusted-supervision 入口使用。

## 11. 如何理解论文状态

不要用“配置存在”推断论文数值已经复现。以 `lnl papers show <paper-id>` 输出的
`implementation_status`、`configuration_fidelity`、`reproduction_status` 和
`availability` 为准。`conditional` 表示还需外部数据或 artifact；`not_run` 表示尚未完成
对应数值实验，即使方法 workflow 已经可用。
