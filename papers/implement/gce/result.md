# GCE：NeurIPS 2018 单次复现结果

## 复现声明

- 原文：`papers/03_robust_loss/12_gce_neurips2018.pdf`
- 当前范围：CIFAR-10、closed-set uniform/symmetric noise `0.2`、标准 `Lq`、单次运行。
- 不在范围：Truncated GCE、其他噪声率、class-dependent/open-set、CIFAR-100、Fashion-MNIST、5 次重复。
- 因此本实验最多达到“单次复现”，不能复现论文标准差。

## 论文参数与 Toolbox 映射

| 含义 | 论文设置 | Toolbox 配置/实现 |
|---|---|---|
| 数据集 | CIFAR-10 | `data.name=cifar10` |
| 划分 | 10% train retained as validation | `data.validation_size=5000` |
| validation 标签 | 与训练集一样被污染 | `noise.validation_targets=noisy` |
| test 标签 | clean | 固定 clean test Dataset |
| 预处理 | per-pixel mean subtraction | `data.preprocessing=gce2018` |
| 增强 | pad 4、32×32 crop、horizontal flip | `data.augment=true` |
| 模型 | ResNet-34 | `model.name=resnet34`、`base_width=64`、层数 `[3,4,6,3]` |
| loss | `(1-p_y^q)/q` | `loss.name=gce` |
| q | 0.7 | `loss.q=0.7` |
| 噪声 | uniform，rate 0.2 | `noise.name=symmetric`、`rate=0.2` |
| 运行次数 | 原文 5 次；本任务 1 次 | `seed=1`、`noise.seed=1` |
| batch size | 128 | `loader.batch_size=128` |
| optimizer | SGD | `optimizer.name=sgd` |
| learning rate | 0.01 | `optimizer.lr=0.01` |
| momentum | 0.9 | `optimizer.momentum=0.9` |
| weight decay | `1e-4` | `optimizer.weight_decay=0.0001` |
| nesterov | 原文未声明 | `optimizer.nesterov=false` |
| epochs | 120 | `trainer.epochs=120` |
| LR schedule | epoch 40/80 除以 10 | multistep `[40,80]`、`gamma=0.1` |
| 模型选择 | validation accuracy 最大 epoch | `best.pt` |
| 论文目标 | `89.83 ± 0.20%` | 单次 test accuracy 与 89.83% 比较 |
| 进度输出 | 原文未规定 | 终端进度、`metrics.jsonl`、`training_curves.svg` |

完整配置：

- Smoke：`configs/experiment/gce_cifar10_noise02_smoke.yaml`
- 正式：`configs/experiment/gce_cifar10_noise02_reproduction.yaml`

## 已知实现差异

1. 原文只描述 uniform sampling；Toolbox 的 symmetric generator 固定选择精确数量的样本，并保证新标签不同于原标签，转移矩阵为对称噪声矩阵。
2. 数据划分和随机数实现使用当前 Toolbox 的 deterministic NumPy/PyTorch 协议；原文没有公布五次运行的 seed。
3. 本任务只运行一次，不能与论文的均值和标准差做统计等价性判断。
4. 软件版本、CUDA、GPU 和 PyTorch 实现与 2018 年环境不同，保存在运行目录的 `environment.json`。

## 执行记录

| 阶段 | 状态 | 产物 | 结果 |
|---|---|---|---|
| 数学与组件测试 | 通过 | focused + 151 项完整 unittest | GCE 公式、ResNet-34、预处理、noisy validation、进度产物全部通过 |
| CUDA smoke（2 epochs） | 通过 | `artifacts/test-runs/gce-noise02-smoke/` | loss 有限；test 10.94%；峰值显存 1258.09 MB；仅验证闭环，不用于论文比较 |
| 正式实验（120 epochs） | 通过 | `artifacts/reproductions/gce-cifar10-noise02-seed1/` | best epoch 51；clean test 88.41%；比论文中心值低 1.42 个百分点 |

## 实际结果

- best epoch：51
- best noisy-validation accuracy：74.10%
- clean test accuracy：88.41%
- 与论文 89.83% 的差值：-1.42 个百分点
- peak CUDA memory：1259.72 MB
- manifest mapping hash：`0743813f6451637b16d51b2a418baacdc1ca08e6f1911b9bf5d05c194aa3345a`
- 训练规模：120 epochs、42,240 optimizer steps
- 学习率核验：epoch 1–40 为 0.01，41–80 为 0.001，81–120 为 0.0001
- 运行耗时：约 1 小时 45 分钟
- stderr：空
- 是否达到预先声明的单次复现标准：是

结论：核心公式、模型、优化流程、噪声率、noisy validation 模型选择和 clean test 评测均形成可恢复闭环。单次结果接近论文报告值，但由于没有执行论文规定的 5 次重复，不能宣称“完整复现”，也不能对 `±0.20` 的标准差做统计比较。
