# 论文文件说明

- `01_instance_dependent/`：3 篇实例依赖噪声论文；
- `02_sample_selection/`：样本选择与 curriculum；
- `03_robust_loss/`：loss correction、normalized loss、GCE；
- `04_statistics/`：转移矩阵、质心、逐类统计；
- `05_hybrid/`：扩散、遗忘/抑噪、协同训练、DivideMix、meta reweighting、feature embedding。

`manifest.json` 记录下载 URL、文件大小与状态。当前 25/26 篇原始 PDF 已下载并通过 `%PDF` 文件头及解析检查。

未自动保存的条目：

- `06_elr_iclr2021.pdf`：OpenReview 返回浏览器验证页，公开镜像也拒绝自动下载。论文官方页：<https://openreview.net/forum?id=Eql5b1_hTE4>；代码：<https://github.com/xiaoboxia/CDR>。

重新下载时可运行 `scripts/download_papers.ps1`；脚本会跳过已经验证过的 PDF。

