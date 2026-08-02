# Jittor 迁移版

本目录用于保存 PyTorch 组合版通过轻量验证后的 Jittor 迁移实现。

迁移重点：

- MSHNet backbone
- PConv 的 asymmetric padding 与四路卷积分支
- SLS/SDM loss 中的尺度项和位置项
- 数据读取、训练入口、评估指标与可视化流程
