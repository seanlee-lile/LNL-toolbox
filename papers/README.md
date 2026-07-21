# 论文文件说明

- [论文阅读与实现标准](reading-standard.md)：统一的阅读成熟度、笔记模板和实现前验收清单。
- [26 篇论文中的 Loss 内容与关系](loss-content-audit.md)：按原文关键词、公式与算法框核查 loss 的实际角色和依赖关系。

- `01_instance_dependent/`：3 篇实例依赖噪声论文；
- `02_sample_selection/`：样本选择与 curriculum；
- `03_robust_loss/`：loss correction、normalized loss、GCE；
- `04_statistics/`：转移矩阵、质心、逐类统计；
- `05_hybrid/`：扩散、遗忘/抑噪、协同训练、DivideMix、meta reweighting、feature embedding。

`manifest.json` 记录自动下载时的 URL、文件大小与状态。当前 26/26 篇原始 PDF 已在本地保存并可解析。

人工补入的条目：

- `02_sample_selection/06_robust_early_learning_hinderin.pdf`：自动下载时 OpenReview 返回浏览器验证页，后由人工补入；`manifest.json` 仍保留当时的失败记录。论文官方页：<https://openreview.net/forum?id=Eql5b1_hTE4>；代码：<https://github.com/xiaoboxia/CDR>。

重新下载时可运行 `scripts/download_papers.ps1`；脚本会跳过已经验证过的 PDF。
