import argparse
import json
import os.path as osp
import sys
from pathlib import Path

import jittor as jt
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jittor_pconv_sdm.dataset import IRSTDDataset
from jittor_pconv_sdm.models import MSHNet
from jittor_pconv_sdm.tools.metrics import MeanIoU, PDFA


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Jittor PConv-SDM baseline.")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--weight-path", type=str, required=True)
    parser.add_argument("--config", type=str, default="mshnet_sls")
    parser.add_argument("--config-file", type=str, default=osp.join("jittor_pconv_sdm", "configs", "ablation.json"))
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def load_ablation(config_file, config_name):
    with open(config_file, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if config_name not in configs:
        raise KeyError(f"Unknown config '{config_name}'. Available configs: {', '.join(configs)}")
    return configs[config_name]


def to_var(x):
    return x if isinstance(x, jt.Var) else jt.array(x)


def main():
    args = parse_args()
    jt.flags.use_cuda = 1 if args.device == "cuda" else 0
    ablation = load_ablation(args.config_file, args.config)
    # 加载测试数据集
    dataset = IRSTDDataset(
        args.dataset_dir,
        mode="test",
        base_size=args.base_size,
        crop_size=args.base_size,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    model = MSHNet(input_channels=3, use_pconv=ablation["use_pconv"])   # 模型
    state = jt.load(args.weight_path)   # 权重参数
    model.load_state_dict(state.get("state_dict", state.get("net", state))) # 加载权重
    model.eval()    # 评估模式

    # 评估指标
    miou = MeanIoU()
    pdfa = PDFA(image_size=args.base_size)
    for images, masks in tqdm(dataset, desc="Test"):
        images = to_var(images).float32()
        masks = to_var(masks).float32()
        _, pred = model(images, warm_flag=False)
        miou.update(pred, masks)
        pdfa.update(pred, masks)

    pd, fa = pdfa.get()
    print(f"IoU: {miou.get():.6f}")
    print(f"Pd: {pd:.6f}")
    print(f"Fa: {fa:.8f}")


if __name__ == "__main__":
    main()

