# LNL Toolbox

LNL Toolbox 是一个面向 **Learning with Noisy Labels（LNL）** 研究的实验工具箱。

项目提供：

- 多篇 LNL 论文方法与实验配置
- 统一的数据、噪声和训练接口
- WebUI 实验入口
- **Scratch**：通过积木组合数据、模型、Loss、样本选择、状态更新等操作，搭建和修改算法
- CLI，用于运行、检查和管理实验

## 安装

需要 Python 3.10+。
```powershell
git clone https://github.com/seanlee-lile/LNL-toolbox.git
cd LNL-toolbox

conda create -n lnl-toolbox python=3.11 -y
conda activate lnl-toolbox

python -m pip install -e ".[train]"
```

确认安装：
```powershell
lnl --help
```

## WebUI

启动：
```powershell
lnl web
```

浏览器会自动打开 WebUI。

主要入口：

- **主页**：`http://127.0.0.1:8765/`
- **Scratch**：`http://127.0.0.1:8765/scratch`

如果不希望自动打开浏览器：
```powershell
lnl web --no-open
```

## Scratch

Scratch 用于直接搭建和修改实验算法。

一个典型训练流程可以表示为：
```text
数据
 ↓
模型
 ↓
Forward
 ↓
Loss / 样本选择 / 状态更新
 ↓
Backward
 ↓
Optimizer Step
 ↓
评估
```

可以：

- 从标准单模型骨架开始
- 从双模型骨架开始
- 打开已有论文模板进行修改
- 从空白 Recipe 开始搭建

Scratch 中的积木最终组成可执行 Recipe，并使用同一套运行环境执行。

## 运行已有实验

查看可用实验：
```powershell
lnl list experiments
```

运行一个实验：
```powershell
lnl run <recipe>
```

例如：
```powershell
lnl run cifar10-symmetric-ce-smoke
```

运行前只做检查：
```powershell
lnl run cifar10-symmetric-ce-smoke --dry-run
```

查看论文方法：
```powershell
lnl papers list
```

## 数据集

本机数据集可以登记到 LNL Toolbox：
```powershell
lnl data register my-cifar10 --adapter cifar10 --root F:/datasets/cifar10
lnl data list
lnl data inspect my-cifar10
```

训练数据不会自动提交到 Git。

CIFAR 等数据集需要提前准备到本机；工具箱不会在训练过程中静默下载数据。

## 项目结构
```text
src/lnl_toolbox/
├── scratch/        # Scratch Block / Recipe / WebUI
├── algorithms/     # 方法实现
├── data/           # 数据接口
├── models/         # 模型
├── losses/         # Loss
├── noise/          # 噪声
├── evaluation/     # 评估
└── cli/            # CLI
```

实验配置主要位于：
```text
configs/
```

Scratch 论文 Recipe 位于：
```text
src/lnl_toolbox/scratch/recipes/papers/
```

## 开发

运行测试：
```powershell
pytest
```

代码检查：
```powershell
ruff check .
```

---

LNL Toolbox 的目标不是把每篇论文封装成互相独立的脚本，而是让论文实验中的数据处理、训练操作和算法步骤能够被清楚地查看、运行和重新组合。
