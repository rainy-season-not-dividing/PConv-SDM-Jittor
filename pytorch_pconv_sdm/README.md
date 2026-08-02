# PyTorch 组合版

本目录用于整理完整的 PyTorch 复现基线：`MSHNet + PConv + SDM`。

计划步骤：

1. 从 `code/mshnet/` 迁移并整理 MSHNet 分割 backbone。
2. 从 `code/pconv_sdloss/` 迁移 PConv 与 SLS/SDM loss。
3. 补齐四组消融配置。
4. 添加 shape、loss forward 和 smoke train 轻量检查。
5. 通过 PyTorch 云端轻量验证后，再开始 Jittor 迁移。
