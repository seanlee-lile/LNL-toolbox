# 干净标签训练架构

```mermaid
flowchart LR
    I["无参数中文向导"] --> C["内存 resolved config"]
    Y["YAML + argparse"] --> C
    C --> E["clean_train 入口"]
    D["CIFAR-10 / CIFAR-100"] --> S["固定 seed 分层划分"]
    S --> L["Dataset / DataLoader"]
    E --> L
    E --> M["TinyCNN / ResNet-18 / PreActResNet-18"]
    E --> O["SGD / AdamW"]
    E --> R["Cosine / MultiStep scheduler"]
    L --> T["Clean training loop"]
    M --> T
    O --> T
    R --> T
    T --> V["Validation"]
    V --> LAST["last.pt"]
    V --> BEST["best.pt"]
    LAST -->|"恢复 model / optimizer / scheduler / RunState"| T
    BEST --> TEST["Clean test"]
    T --> LOG["metrics.jsonl"]
    TEST --> FINAL["final_metrics.json"]
    FINAL --> MULTI["多 seed mean / std"]
```

训练阶段只访问 `input`、干净 `target` 和稳定 `index`。每轮结束在 validation 上选择最佳 checkpoint；完整训练结束后重新加载 `best.pt` 再计算 test，避免使用测试集选择模型。

正式实验使用 `configs/experiment/cifar10_clean_baseline.yaml`，默认 PreActResNet-18、SGD、cosine scheduler 和 200 epochs。`cifar10_clean_smoke.yaml` 使用 TinyCNN 和小数据子集，仅用于全链路调试。
## 2.1 复现底座扩展

四篇论文底座共享以下生命周期边界：

- `ParameterRecord` 负责一次抽样及其来源，不负责自动调参；
- `StandardNoisyERMPipeline` 负责 warm-up、artifact、regularizer 和组件状态传递；
- `SupervisedClassificationAlgorithm` 只组合 base loss、risk correction、selection、weight 和 regularizer；
- `checkpoint.py` 保存 resolved configuration、参数记录、算法私有状态、组件状态和 RNG；
- `progress.py` 只产出标准 epoch 指标，`curve_comparison.py` 负责实验后曲线对齐。

Binary Risk、CWD 和 FINE 的数学实现分别位于 algorithm/estimator/selector 模块，不新增论文专属 `experiment.py` 分支。正式论文实验仍须另行提供论文参数配置和曲线来源。
