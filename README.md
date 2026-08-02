# PConv-SDM-Jittor：红外弱小目标检测复现

本仓库用于复现论文 **Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection** 中面向 mask-based segmentation 的核心方法，并将 PyTorch 组合版迁移到 Jittor。

当前仓库处于第一阶段：先整理官方 PyTorch 参考代码，再完成 `MSHNet + PConv + SDM` 的 PyTorch 可运行基线。Jittor 迁移将在 PyTorch 版本通过轻量验证后开始。

## 项目目标

本项目不走 YOLO 检测框架路线，而是围绕红外弱小目标检测中的分割式实验设置，复现以下核心内容：

- MSHNet 分割 backbone
- Pinwheel-shaped Convolution，简称 PConv
- Scale-based Dynamic Loss，简称 SDM Loss
- IRSTD-1K 上的四组消融实验

计划完成的四组实验为：

| 实验组 | Backbone | 卷积模块 | Loss |
| --- | --- | --- | --- |
| 1 | MSHNet | 普通卷积 | SLS |
| 2 | MSHNet | PConv | SLS |
| 3 | MSHNet | 普通卷积 | SDM |
| 4 | MSHNet | PConv | SDM |

## 代码结构

本仓库参考 Jittor 复现仓库常见组织方式，采用“官方 PyTorch 参考代码 + 自整理 PyTorch 组合版 + Jittor 迁移版”的三层结构：

```text
PConv-SDM-Jittor/
  code/
    pconv_sdloss/        # PConv + SD Loss 官方 PyTorch 参考代码
    mshnet/              # MSHNet 官方 PyTorch 参考代码

  pytorch_pconv_sdm/     # 自整理的完整 PyTorch 组合版
    configs/
    dataset/
    models/
    losses/
    tools/
    train.py
    test.py

  jittor_pconv_sdm/      # Jittor 迁移版
    configs/
    dataset/
    models/
    losses/
    tools/
    train.py
    test.py

  scripts/               # 环境检查、数据准备、训练评估辅助脚本
```

## 参考代码来源

`code/` 目录只作为上游参考代码保存，后续不直接在其中开发。正式实现会整理到 `pytorch_pconv_sdm/` 和 `jittor_pconv_sdm/`。

| 参考内容 | 来源仓库 | 当前导入版本 |
| --- | --- | --- |
| PConv 与 SD Loss | https://github.com/JN-Yang/PConv-SDloss-Data | `a801f043c83f73aa9af9ab2f689e59ebef928fc4` |
| MSHNet backbone | https://github.com/Lliu666/MSHNet | `7c5194b8caeb3329ba8a67c75a6928d4dbeb3582` |

需要注意的是，PConv 官方仓库主要提供 PConv 与 SD loss 的参考实现，并不等价于完整的 `MSHNet + PConv + SDM` 分割复现工程。因此本项目会先构建一个干净的 PyTorch 组合版，作为 Jittor 迁移前的对齐基准。

## 复现路线

1. 引入官方 PyTorch 参考代码。
2. 整理 `MSHNet + PConv + SDM` 的 PyTorch 组合版。
3. 在本地完成 `py_compile`、shape check、loss forward 等轻量检查。
4. 在云端 GPU 上完成 PyTorch 轻量 smoke test。
5. 开始 Jittor 迁移，并对齐模型结构、PConv padding、loss 计算和训练入口。
6. 在云端 GPU 上完成 Jittor 轻量 smoke test。
7. 运行 IRSTD-1K 四组正式消融，并整理 IoU、Pd、Fa 与可视化结果。

## PyTorch 轻量检查

当前 PyTorch 组合版已提供以下检查入口：

```bash
python pytorch_pconv_sdm/tools/check_env.py
python -m py_compile pytorch_pconv_sdm/train.py pytorch_pconv_sdm/test.py
python pytorch_pconv_sdm/tools/sanity_check.py
python pytorch_pconv_sdm/tools/smoke_train.py --max-iters 2 --config mshnet_pconv_sdm
```

训练入口示例：

```bash
python pytorch_pconv_sdm/train.py \
  --dataset-dir /path/to/IRSTD-1k \
  --config mshnet_pconv_sdm \
  --batch-size 4 \
  --epochs 400
```

测试入口示例：

```bash
python pytorch_pconv_sdm/test.py \
  --dataset-dir /path/to/IRSTD-1k \
  --weight-path /path/to/best_weight.pkl \
  --config mshnet_pconv_sdm
```

## 环境说明

推荐基础环境：

```text
Python 3.10
PyTorch 2.1.2
CUDA 11.8
Jittor stable release
```

当前 `requirements.txt` 只记录通用 Python 依赖。PyTorch 与 Jittor 的具体安装命令会根据本地或云端环境单独说明。

## 数据与输出

数据集、预训练权重、训练日志、checkpoint、可视化输出等大文件不会提交到仓库。它们应放在本地或云端工作目录中，并通过 `.gitignore` 排除。

建议目录命名：

```text
data/
weights/
checkpoints/
runs/
logs/
outputs/
```

## 当前状态

- [x] 建立正式仓库目录
- [x] 引入 PConv / SD Loss 参考代码
- [x] 引入 MSHNet 参考代码
- [x] 完成 PyTorch 组合版
- [ ] 完成 PyTorch 轻量验证
- [ ] 完成 Jittor 迁移
- [ ] 完成 Jittor 轻量验证
- [ ] 完成四组正式消融实验
