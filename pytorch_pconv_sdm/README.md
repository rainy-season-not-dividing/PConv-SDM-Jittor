# PyTorch 组合版

本目录用于整理完整的 PyTorch 复现基线：`MSHNet + PConv + SDM`。

## 当前内容

- `models/mshnet.py`：MSHNet backbone，支持 `use_pconv` 开关。
- `models/pconv.py`：对齐官方 `APConv.py` 的 PConv 模块。
- `losses/sls_sdm_loss.py`：SLS/SDM loss。
- `dataset/irstd_dataset.py`：IRSTD-1K 风格数据读取。
- `configs/ablation.json`：四组消融配置。
- `train.py` / `test.py`：基础训练与测试入口。
- `tools/`：环境、shape、loss、smoke train 轻量检查。

## 本地轻量检查

```bash
python pytorch_pconv_sdm/tools/check_env.py
python -m py_compile pytorch_pconv_sdm/train.py pytorch_pconv_sdm/test.py
python pytorch_pconv_sdm/tools/sanity_check.py
python pytorch_pconv_sdm/tools/smoke_train.py --max-iters 2 --config mshnet_pconv_sdm
```

## 下一步

PyTorch 组合版在本地通过轻量检查后，需要在云端 GPU 环境中完成同一组 smoke test。通过后再开始 Jittor 迁移。
