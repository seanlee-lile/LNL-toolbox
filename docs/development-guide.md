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

## 5. 运行测试

项目当前使用 Python 标准库 `unittest`，不要求安装 pytest：

```powershell
python -m unittest discover -s tests -v
```

当前基线为 19 项测试全部通过。提交代码前至少应运行一次完整测试。

## 6. 运行训练

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

自动化工具可以直接调用环境中的解释器：

```powershell
& "F:\Miniconda\envs\pytorch\python.exe" -m unittest discover -s tests -v
```

本机绝对路径只用于本地自动调试，不应写进项目配置或源代码。其他开发者应使用自己的环境路径或正常执行 `conda activate lnl-toolbox`。

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
