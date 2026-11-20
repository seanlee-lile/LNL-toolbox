# 论文文件说明

- [论文实现与复现目录](implement/README.md)：说明 taxonomy、guideline、审计、复现计划和结果的维护边界。
- [论文阅读与实现映射](implement/paper-implementation-guideline.md)：统一的论文阅读证据、模块映射和实现前验收清单。
- [26 篇论文中的 Loss 内容与关系](implement/loss-content-audit.md)：按原文关键词、公式与算法框核查 loss 的实际角色和依赖关系。
- [论文复现进度与参数台账](implement/paper-reproduction-progress.md)：复现总表、复用审计和单篇记录索引。

以下是历史获取清单中的分类路径，不代表当前 checkout 包含这些目录或原论文 PDF：

- `01_instance_dependent/`：3 篇实例依赖噪声论文；
- `02_sample_selection/`：样本选择与 curriculum；
- `03_robust_loss/`：loss correction、normalized loss、GCE；
- `04_statistics/`：转移矩阵、质心、逐类统计；
- `05_hybrid/`：扩散、遗忘/抑噪、协同训练、DivideMix、meta reweighting、feature embedding。

本仓库不随附原论文 PDF。[历史获取清单](manifest.json)记录下载时的 URL、文件大小和状态，可用于查找论文来源；它不是 CLI 或 Web 使用的 Paper catalog。运行时目录见 [`src/lnl_toolbox/paper_catalog.json`](../src/lnl_toolbox/paper_catalog.json)。

论文来源清单保留在 `papers/`；从论文到 Toolbox 的实现资料统一放在
`papers/implement/`。单篇复现的计划和结果分别保存在
`papers/implement/<method>/plan.md` 与 `result.md`，不在全局台账重复正文。

历史人工补录记录：

- `02_sample_selection/06_robust_early_learning_hinderin.pdf`：自动下载时 OpenReview 返回浏览器验证页，后由人工补入；`manifest.json` 仍保留当时的失败记录。论文官方页：<https://openreview.net/forum?id=Eql5b1_hTE4>；代码：<https://github.com/xiaoboxia/CDR>。
