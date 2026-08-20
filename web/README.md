# LNL Toolbox Web Console

这是一个独立的本地网页入口，位于 web/，不修改现有 CLI、训练 runner 或数据模块。

## 启动

在仓库根目录运行：

    python web/command_console.py

然后打开 http://127.0.0.1:8765。

如果环境中已经安装了 lnl，网页优先调用 lnl ...；否则回退到：

    python -m lnl_toolbox.cli.main ...

网页中的每个按钮都对应一个固定的参数列表。后端使用 shell=False，不接受浏览器传入的任意命令字符串。训练 1 epoch 是专门的短训练示例，输出目录为 artifacts/web-smoke。

## 快速验证

    python -m unittest discover -s web -p "test_*.py" -v

建议首次使用顺序：查看帮助 → 检查环境 → 查看 Smoke 配方 → 验证 Clean 配置 → 预演一次训练 → 训练 1 epoch。
