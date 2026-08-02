import argparse
import json
import os.path as osp
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pytorch_pconv_sdm.dataset import IRSTDDataset
from pytorch_pconv_sdm.models import MSHNet
from pytorch_pconv_sdm.tools.metrics import MeanIoU, PDFA


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the PyTorch PConv-SDM baseline.")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--weight-path", type=str, required=True)
    parser.add_argument("--config", type=str, default="mshnet_sls")
    parser.add_argument("--config-file", type=str, default=osp.join("pytorch_pconv_sdm", "configs", "ablation.json"))
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_ablation(config_file, config_name):
    with open(config_file, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if config_name not in configs:
        raise KeyError(f"Unknown config '{config_name}'. Available configs: {', '.join(configs)}")
    return configs[config_name]


@torch.no_grad()
def main():
    args = parse_args()
    ablation = load_ablation(args.config_file, args.config)
    device = torch.device(args.device)

    dataset = IRSTDDataset(args.dataset_dir, mode="test", base_size=args.base_size, crop_size=args.base_size)
    loader = DataLoader(dataset, 1, shuffle=False, drop_last=False)
    model = MSHNet(input_channels=3, use_pconv=ablation["use_pconv"]).to(device)
    state = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(state.get("state_dict", state.get("net", state)))
    model.eval()

    miou = MeanIoU()
    pdfa = PDFA(image_size=args.base_size)
    for images, masks in tqdm(loader, desc="Test"):
        images = images.to(device)
        masks = masks.to(device)
        _, pred = model(images, warm_flag=False)
        miou.update(pred, masks)
        pdfa.update(pred, masks)

    pd, fa = pdfa.get()
    print(f"IoU: {miou.get():.6f}")
    print(f"Pd: {pd:.6f}")
    print(f"Fa: {fa:.8f}")


if __name__ == "__main__":
    main()
