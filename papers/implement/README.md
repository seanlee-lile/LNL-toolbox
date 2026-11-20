# 论文实现与复现目录

本目录集中保存“论文如何映射到 Toolbox”以及“论文实验是否已经实际复现”的维护资料。原始论文 PDF、下载清单和来源信息继续保存在上一级 `papers/`，不得复制到这里。

## 目录职责

- `paper-baseline-taxonomy.md`：按训练流程修改位置对论文分类。
- `paper-implementation-guideline.md`：规定每篇论文应复用或扩展的唯一模块、接口和调用顺序。
- `paper-reproduction-progress.md`：历史复现状态、实验记录和复用审计。
- `*-audit.md`、`common-loss-functions.md`：跨论文公式或组件审计。
- `<method>/plan.md`：单篇论文的实施计划。
- `<method>/result.md`：实际执行参数、产物、结果和与原文的差异。

逐方法复现说明见 [DivideMix](../dividemix/reproduction.md)、[DLD](../dld/reproduction.md)、
[Dual-T](../dual-t/reproduction.md)、[Importance Reweighting](../importance-reweighting/reproduction.md)、
[LEND](../lend/reproduction.md)、[PCSE](../pcse/reproduction.md)、[UPM](../upm/reproduction.md)和
[VolMinNet](../volminnet/reproduction.md)。

## 维护规则

1. 复现应像菜谱一样组合已有组件；数据、模型、Loss、Noise、Selector、更新策略、评测和 checkpoint 不得按论文复制。
2. 缺少共享能力时，只在 guideline 指定的唯一位置扩展一次，并在复用审计表登记未来使用者。
3. 只有论文特有的数学、状态或多阶段流程可以使用论文名模块；单纯参数差异只写入 YAML 和 `plan.md`。
4. 全局进度文件只保留索引和审计信息；详细结果放在对应方法子目录，避免重复维护。
5. 尚未执行的内容必须标记为“计划中”，不得写成已经实现或已经通过。
