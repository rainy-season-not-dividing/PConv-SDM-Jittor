# PConv-SDM-Jittor 复现流程说明

本文记录本仓库从论文理解、PyTorch 组合版整理、Jittor 迁移到云端实验验证的完整流程。它不是完整训练日志，而是用于说明复现思路、关键代码链路和实验闭环的轻量文档。

## 1. 复现目标

本项目围绕红外弱小目标检测中的 mask-based segmentation 设置，复现并组合以下核心组件：

- MSHNet：作为多尺度分割 backbone。
- PConv：使用四路非对称 padding 和卷积分支增强局部结构建模。
- SLS / SDM loss：围绕红外弱小目标的尺度敏感问题设计损失。
- IRSTD-1K：作为主要实验数据集。

本项目不走 YOLO 检测框路线。PConv 官方仓库提供了 PConv 和 SD Loss 的参考实现，但并不等价于一个完整的 `MSHNet + PConv + SDM` 分割复现工程。因此本项目先整理 PyTorch 组合版，再迁移到 Jittor。

## 2. 代码组织

仓库采用三层结构：

```text
code/
  pconv_sdloss/        # PConv + SD Loss 官方 PyTorch 参考代码
  mshnet/              # MSHNet 官方 PyTorch 参考代码

pytorch_pconv_sdm/     # 自整理 PyTorch 组合版
jittor_pconv_sdm/      # Jittor 迁移版
```

其中 `code/` 目录主要用于保留上游参考代码；正式训练、测试和消融入口放在 `pytorch_pconv_sdm/` 和 `jittor_pconv_sdm/` 中。

## 3. PyTorch 组合版

PyTorch 组合版的作用是建立 Jittor 迁移前的对齐参照。主要整理内容包括：

- `dataset/irstd_dataset.py`：读取 IRSTD-1K 图像、mask 和 train/test split。
- `models/mshnet.py`：整理 MSHNet 多尺度输出结构。
- `models/pconv.py`：实现 PConv 模块，并通过 `use_pconv` 开关控制是否替换普通卷积。
- `losses/sls_sdm_loss.py`：实现 SLS 和 SDM 两类 loss。
- `train.py`：训练入口，记录 `metrics.log` 并保存 `best_weight.pkl` / `checkpoint.pkl`。
- `test.py`：测试入口，加载权重后计算 IoU、Pd、Fa。

PyTorch 50 epoch baseline：

```text
config: mshnet_pconv_sdm
best epoch: 45
IoU=0.597179
Pd=0.911565
Fa=0.00001693
```

## 4. Jittor 迁移

Jittor 迁移的重点不是简单改语法，而是尽量保持 PyTorch 组合版的模块边界、输入输出和训练逻辑一致。

主要迁移点：

- `jittor_pconv_sdm/dataset/irstd_dataset.py`：保持 IRSTD-1K 数据读取和预处理逻辑一致。
- `jittor_pconv_sdm/models/mshnet.py`：迁移 MSHNet 主体和多尺度 head。
- `jittor_pconv_sdm/models/pconv.py`：迁移 PConv 的四路 padding + convolution 分支。
- `jittor_pconv_sdm/losses/sls_sdm_loss.py`：迁移 SLS / SDM loss 计算。
- `jittor_pconv_sdm/tools/adagrad.py`：补齐 Jittor 训练中使用的优化器逻辑。
- `jittor_pconv_sdm/tools/metrics.py`：统一 IoU、Pd、Fa 的计算方式。

迁移后先完成了轻量检查：

```text
py_compile: passed
sanity_check.py: passed
smoke_train.py --config mshnet_pconv_sdm --max-iters 2 --device cuda: passed
four-config ablation smoke: passed
```

随后在官方 IRSTD-1K 上完成 2 epoch debug：

```text
train.exit=0
test.exit=0
epoch=1 loss=0.871432
IoU=0.141926
Pd=0.527211
Fa=0.00052153
```

## 5. 云端实验阶段

云端实验主要分为以下阶段：

1. PyTorch 轻量 smoke test。
2. PyTorch IRSTD-1K 50 epoch baseline。
3. Jittor 轻量检查和真实数据 2 epoch debug。
4. Jittor `mshnet_pconv_sdm` 50 epoch 对齐实验。
5. Jittor 三组 50 epoch 补充消融：`mshnet_sls`、`mshnet_pconv_sls`、`mshnet_sdm`。
6. Jittor `mshnet_pconv_sdm` 200 epoch 主配置长训练。

云端 Jittor 运行需要：

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

该设置用于解决云端 conda 环境 `libstdc++.so.6` 缺少 `GLIBCXX_3.4.30` 的问题。

## 6. 消融与长训练结果

Jittor 50 epoch 四组消融：

| 配置 | PConv | Loss | best epoch | IoU | Pd | Fa |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `mshnet_sls` | 否 | SLS | 38 | 0.603644 | 0.840136 | 0.00001526 |
| `mshnet_pconv_sls` | 是 | SLS | 38 | 0.644378 | 0.901361 | 0.00001488 |
| `mshnet_sdm` | 否 | SDM | 49 | 0.573503 | 0.867347 | 0.00001738 |
| `mshnet_pconv_sdm` | 是 | SDM | 48 | 0.621521 | 0.911565 | 0.00003105 |

观察：

- PConv 对 IoU 的提升比较明显，`mshnet_sls` 到 `mshnet_pconv_sls` 从 0.603644 提升到 0.644378。
- SDM 单独加入时本次设置下 IoU 较低，说明动态尺度损失并不是孤立生效，需要和合适的特征表达模块配合。
- 50 epoch 的完整主配置 `mshnet_pconv_sdm` 的 Pd 最高，可作为 Jittor 迁移对齐参照。

由于 50 epoch 主配置的 best epoch 出现在第 48 轮，接近训练末尾，因此追加 200 epoch 长训练检查是否仍有提升。

200 epoch 主配置结果：

```text
best epoch: 147
IoU=0.652015
Pd=0.928571
Fa=0.00002573
```

与 50 epoch 主配置相比：

```text
50 epoch:  epoch 48,  IoU=0.621521, Pd=0.911565, Fa=0.00003105
200 epoch: epoch 147, IoU=0.652015, Pd=0.928571, Fa=0.00002573
```

因此，200 epoch 结果作为当前主结果，50 epoch 结果保留为迁移对齐参照。

## 7. 可视化与结果保留

`jittor_pconv_sdm/tools/visualize_predictions.py` 用于加载 Jittor 权重并生成预测可视化：

- `original.png`
- `gt.png`
- `pred.png`
- `overlay.png`
- `quad.png`

GitHub 中只保留 3 张精选 `quad.png`，用于展示预测效果和支撑 PPT 汇报。完整 10 张可视化和完整日志保存在本地归档。

## 8. 仓库与归档分工

GitHub 仓库保留：

```text
代码
配置
README / reproduction notes
精简结果表
metrics CSV
训练曲线图
精选预测可视化
```

本地归档保留：

```text
完整 logs/
完整 runs/
best_weight.pkl
checkpoint.pkl
完整云端工作区压缩包
```

这样可以让 GitHub 仓库保持轻量、可读、可复查，同时确保训练证据和权重不会丢失。
