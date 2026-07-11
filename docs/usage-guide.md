# LNL Toolbox 使用说明

## 当前实现

当前版本已跑通：CIFAR pickle 文件读取、固定种子分层划分、PyTorch DataLoader、TinyCNN、交叉熵训练、验证/测试、JSONL 日志、checkpoint 保存与恢复。第一阶段只训练干净标签；噪声算法将在这个闭环之上作为插件接入。

## 数据目录

仓库采用以下目录，`data/` 已被 Git 忽略：

```text
data/
├── cifar10/   # data_batch_1 ... test_batch, batches.meta
└── cifar100/  # train, test, meta
```

## 运行

在仓库根目录执行。正常使用 Conda 时：

```powershell
conda activate pytorch
$env:PYTHONPATH = "src"
python -m lnl_toolbox.cli.train --config configs/experiment/cifar10_smoke.yaml
```

不激活环境也可以直接指定解释器：

```powershell
$env:PYTHONPATH = "src"
& "F:\Miniconda\envs\pytorch\python.exe" -m lnl_toolbox.cli.train --config configs/experiment/cifar10_smoke.yaml
```

完整配置位于 `configs/experiment/cifar10_clean.yaml`。恢复训练时，`--epochs` 是总 epoch 目标：

```powershell
python -m lnl_toolbox.cli.train --config configs/experiment/cifar10_smoke.yaml --resume artifacts/runs/cifar10_smoke/last.pt --epochs 3
```

每次新运行生成 `resolved_config.yaml`、`environment.json`、`metrics.jsonl`、`last.pt` 和 `final_metrics.json`。恢复运行会继续使用 checkpoint 所在目录并追加日志。

## 测试

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

目前不需要 pytest 或 TensorBoard。
