# 干净标签训练全流程

正式 clean baseline 使用独立入口 `lnl_toolbox.cli.clean_train`。它支持 CIFAR-10/100、TinyCNN、CIFAR ResNet-18、PreActResNet-18、SGD/AdamW、cosine/multistep 学习率、最佳模型保存、断点恢复和多随机种子汇总。

## 单次正式训练

```powershell
$env:PYTHONPATH = "src"
python -m lnl_toolbox.cli.clean_train --config configs/experiment/cifar10_clean_baseline.yaml
```

不激活 Conda 时可直接使用：

```powershell
$env:PYTHONPATH = "src"
& "F:\Miniconda\envs\pytorch\python.exe" -m lnl_toolbox.cli.clean_train --config configs/experiment/cifar10_clean_baseline.yaml
```

## Smoke 与恢复

```powershell
python -m lnl_toolbox.cli.clean_train --config configs/experiment/cifar10_clean_smoke.yaml --output-dir artifacts/runs/clean-smoke
python -m lnl_toolbox.cli.clean_train --config configs/experiment/cifar10_clean_smoke.yaml --resume artifacts/runs/clean-smoke/last.pt --epochs 3
```

`--epochs` 表示总 epoch 目标。恢复时会继续 checkpoint 中的 model、optimizer、scheduler、epoch 和 global step。

## 多 seed

```powershell
python -m lnl_toolbox.cli.clean_train --config configs/experiment/cifar10_clean_baseline.yaml --seeds 1 2 3 --output-dir artifacts/runs/cifar10-clean-3seeds
```

多 seed 运行按顺序执行，输出 `summary.json` 与 `summary.csv`，包含测试准确率均值和样本标准差。

## 每次运行的产物

- `resolved_config.yaml`：实际使用配置；
- `environment.json`：Python、PyTorch、CUDA、GPU 和 seed；
- `metrics.jsonl`：每轮训练、验证和学习率；
- `last.pt`：最近一轮完整状态；
- `best.pt`：validation accuracy 最优的完整状态；
- `final_metrics.json`：加载 `best.pt` 后得到的 test 指标和显存峰值。

正式配置默认使用 PreActResNet-18、SGD、cosine scheduler 和 200 epochs。TinyCNN smoke 配置仅用于验证程序链路。
