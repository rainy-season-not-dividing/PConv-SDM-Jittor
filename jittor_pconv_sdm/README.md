# Jittor 迁移版

本目录保存 `MSHNet + PConv + SDM` 的 Jittor 迁移实现，接口尽量与 `pytorch_pconv_sdm/` 保持一致。

## 当前内容

- `models/mshnet.py`：MSHNet backbone，支持 `use_pconv` 开关。
- `models/pconv.py`：PConv 的四路非对称 padding + 卷积分支。
- `losses/sls_sdm_loss.py`：SLS/SDM loss。
- `dataset/irstd_dataset.py`：IRSTD-1K 风格数据读取。
- `configs/ablation.json`：四组消融配置。
- `train.py` / `test.py`：基础训练与测试入口。
- `tools/`：Jittor 环境、shape、loss、smoke train 轻量检查。

## 轻量检查

```bash
python -m py_compile jittor_pconv_sdm/train.py jittor_pconv_sdm/test.py
python jittor_pconv_sdm/tools/check_env.py
python jittor_pconv_sdm/tools/sanity_check.py
python jittor_pconv_sdm/tools/smoke_train.py --max-iters 2 --config mshnet_pconv_sdm
```

## 训练入口示例

```bash
python -m jittor_pconv_sdm.train \
  --dataset-dir /path/to/IRSTD-1K \
  --config mshnet_pconv_sdm \
  --batch-size 4 \
  --epochs 50 \
  --base-size 256 \
  --crop-size 256 \
  --save-dir runs/jittor_pconv_sdm_50epoch
```

## 测试入口示例

```bash
python -m jittor_pconv_sdm.test \
  --dataset-dir /path/to/IRSTD-1K \
  --weight-path /path/to/best_weight.pkl \
  --config mshnet_pconv_sdm \
  --base-size 256
```

## 当前状态

代码迁移初版已完成，并已在 AutoDL 云端完成 Jittor 轻量验证：

```text
Jittor: 1.3.11.0
py_compile: passed
sanity_check.py: passed
smoke_train.py --config mshnet_pconv_sdm --max-iters 2 --device cuda: passed
four-config ablation smoke: passed
```

官方 IRSTD-1K 真实数据 2 epoch debug 也已通过：

```text
log dir: /root/autodl-tmp/PConv-SDM-Jittor/logs/2026-08-02_07_jittor_irstd1k_debug
run dir: /root/autodl-tmp/PConv-SDM-Jittor/runs/jittor_pconv_sdm_debug/mshnet_pconv_sdm-20260802-154228
train.exit: 0
test.exit: 0
epoch=1 loss=0.871432 IoU=0.141926 Pd=0.527211 Fa=0.00052153
```

云端 conda 环境中的 `libstdc++.so.6` 缺少 `GLIBCXX_3.4.30`，运行 Jittor 命令时需要临时加入：

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

下一步先整理迁移提交范围，再准备 Jittor 50 epoch 对齐实验；正式长训练、commit 或 push 前仍需先征求确认。
