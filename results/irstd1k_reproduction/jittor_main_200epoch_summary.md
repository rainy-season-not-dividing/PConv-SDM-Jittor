# Jittor 200 epoch 主配置摘要

```text
config: mshnet_pconv_sdm
run_dir: runs/jittor_pconv_sdm_200epoch/mshnet_pconv_sdm-20260802-201441
train.exit: 0
test.exit: 0
```

最优结果：

```text
best epoch: 147
loss=0.780332
IoU=0.652015
Pd=0.928571
Fa=0.00002573
```

测试结果：

```text
IoU: 0.652015
Pd: 0.928571
Fa: 0.00002573
```

与 50 epoch 主配置对比：

```text
50 epoch:  epoch 48,  IoU=0.621521, Pd=0.911565, Fa=0.00003105
200 epoch: epoch 147, IoU=0.652015, Pd=0.928571, Fa=0.00002573
```

结论：200 epoch 主配置在 IoU 和 Pd 上继续提升，Fa 也更低，因此作为当前主结果；50 epoch 主配置保留为 Jittor 迁移对齐参照。
