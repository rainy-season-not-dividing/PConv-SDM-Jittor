# Jittor 50 epoch 消融摘要

复用主配置：

```text
mshnet_pconv_sdm
来源：阶段 08 Jittor 50 epoch 主配置对齐实验
best epoch: 48
IoU=0.621521
Pd=0.911565
Fa=0.00003105
run: runs/jittor_pconv_sdm_50epoch/mshnet_pconv_sdm-20260802-162949
```

## mshnet_sls

```text
run_dir: runs/jittor_pconv_sdm_50epoch_ablation/mshnet_sls-20260802-172702
train.exit: 0
test.exit: 0
best epoch: 38
loss=0.799974
IoU=0.603644
Pd=0.840136
Fa=0.00001526
```

## mshnet_pconv_sls

```text
run_dir: runs/jittor_pconv_sdm_50epoch_ablation/mshnet_pconv_sls-20260802-174728
train.exit: 0
test.exit: 0
best epoch: 38
loss=0.730999
IoU=0.644378
Pd=0.901361
Fa=0.00001488
```

## mshnet_sdm

```text
run_dir: runs/jittor_pconv_sdm_50epoch_ablation/mshnet_sdm-20260802-181718
train.exit: 0
test.exit: 0
best epoch: 49
loss=0.849914
IoU=0.573503
Pd=0.867347
Fa=0.00001738
```

## 简要结论

PConv 对 IoU 的提升最明显：`mshnet_sls` 为 0.603644，`mshnet_pconv_sls` 提升到 0.644378。SDM 单独加入时 IoU 较低，说明动态尺度损失需要和合适的特征表达模块配合使用。50 epoch 下完整主配置 `mshnet_pconv_sdm` 的 Pd 最高，可作为迁移对齐结果；后续 200 epoch 长训练进一步提升了主配置表现。
