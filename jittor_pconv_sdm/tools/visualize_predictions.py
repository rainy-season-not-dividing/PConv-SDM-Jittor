import argparse
import json
import os.path as osp
import re
import sys
from pathlib import Path

import jittor as jt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jittor_pconv_sdm.models import MSHNet


DEFAULT_WEIGHT = osp.join(
    "runs",
    "jittor_pconv_sdm_50epoch",
    "mshnet_pconv_sdm-20260802-162949",
    "best_weight.pkl",
)
DEFAULT_OUTPUT = osp.join(
    "logs",
    "2026-08-02_09_jittor_irstd1k_50epoch_ablation",
    "visualizations",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Save PPT-ready Jittor prediction visualizations.")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--weight-path", type=str, default=DEFAULT_WEIGHT)
    parser.add_argument("--config", type=str, default="mshnet_pconv_sdm")
    parser.add_argument("--config-file", type=str, default=osp.join("jittor_pconv_sdm", "configs", "ablation.json"))
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--names", nargs="*", default=None, help="Optional test image ids without file extensions.")
    return parser.parse_args()


def load_ablation(config_file, config_name):
    with open(config_file, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if config_name not in configs:
        raise KeyError(f"Unknown config '{config_name}'. Available configs: {', '.join(configs)}")
    return configs[config_name]


def read_test_names(dataset_dir):
    list_path = osp.join(dataset_dir, "test.txt")
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def pick_evenly(names, count):
    if count <= 0:
        raise ValueError("--num-samples must be positive.")
    if count >= len(names):
        return names

    raw_indices = np.linspace(0, len(names) - 1, count)
    indices = []
    seen = set()
    for value in raw_indices:
        idx = int(round(float(value)))
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    for idx in range(len(names)):
        if len(indices) >= count:
            break
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    return [names[idx] for idx in indices[:count]]


def safe_name(name):
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("._")
    return cleaned or "sample"


def normalize_image(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    array = (array - mean) / std
    return array.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)


def load_resized_pair(dataset_dir, name, base_size):
    image_path = osp.join(dataset_dir, "images", f"{name}.png")
    mask_path = osp.join(dataset_dir, "masks", f"{name}.png")
    image = Image.open(image_path).convert("RGB").resize((base_size, base_size), Image.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((base_size, base_size), Image.NEAREST)
    return image, mask


def predict_mask(model, image, threshold):
    inputs = jt.array(normalize_image(image)).float32()
    _, logits = model(inputs, warm_flag=False)
    prob = jt.sigmoid(logits).numpy().squeeze()
    return (prob >= threshold).astype(np.uint8)


def save_binary_mask(mask, path):
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def make_overlay(image, pred_mask, alpha):
    base = np.asarray(image, dtype=np.float32)
    red = np.zeros_like(base)
    red[:, :, 0] = 255.0
    mask = pred_mask.astype(bool)
    base[mask] = base[mask] * (1.0 - alpha) + red[mask] * alpha
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def labeled_panel(image, label, label_height=24):
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image.convert("RGB"), (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), label, fill=(0, 0, 0))
    return canvas


def make_quad(original, gt_mask, pred_mask, overlay):
    gap = 8
    panels = [
        labeled_panel(original, "Original"),
        labeled_panel(gt_mask.convert("RGB"), "GT mask"),
        labeled_panel(Image.fromarray(pred_mask * 255, mode="L").convert("RGB"), "Pred mask"),
        labeled_panel(overlay, "Original + Pred overlay"),
    ]
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    return canvas


def write_index(output_dir, records, args):
    index_path = osp.join(output_dir, "index.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("# Prediction visualizations\n\n")
        f.write(f"- config: `{args.config}`\n")
        f.write(f"- weight: `{args.weight_path}`\n")
        f.write(f"- dataset: `{args.dataset_dir}`\n")
        f.write(f"- base_size: `{args.base_size}`\n")
        f.write(f"- threshold: `{args.threshold}`\n")
        f.write(f"- samples: `{len(records)}`\n\n")
        for record in records:
            f.write(f"- `{record['name']}`: `{record['dir']}`\n")


def main():
    args = parse_args()
    jt.flags.use_cuda = 1 if args.device == "cuda" else 0

    ablation = load_ablation(args.config_file, args.config)
    all_names = read_test_names(args.dataset_dir)
    selected_names = args.names if args.names else pick_evenly(all_names, args.num_samples)

    model = MSHNet(input_channels=3, use_pconv=ablation["use_pconv"])
    state = jt.load(args.weight_path)
    model.load_state_dict(state.get("state_dict", state.get("net", state)))
    model.eval()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    records = []
    for idx, name in enumerate(selected_names, start=1):
        original, gt = load_resized_pair(args.dataset_dir, name, args.base_size)
        pred = predict_mask(model, original, args.threshold)
        overlay = make_overlay(original, pred, args.overlay_alpha)

        sample_dir_name = f"{idx:02d}_{safe_name(name)}"
        sample_dir = osp.join(args.output_dir, sample_dir_name)
        Path(sample_dir).mkdir(parents=True, exist_ok=True)

        original.save(osp.join(sample_dir, "original.png"))
        gt.save(osp.join(sample_dir, "gt.png"))
        save_binary_mask(pred, osp.join(sample_dir, "pred.png"))
        overlay.save(osp.join(sample_dir, "overlay.png"))
        make_quad(original, gt, pred, overlay).save(osp.join(sample_dir, "quad.png"))

        records.append({"name": name, "dir": sample_dir_name})
        print(f"[{idx:02d}/{len(selected_names):02d}] saved {name} -> {sample_dir}")

    write_index(args.output_dir, records, args)
    print(f"Saved {len(records)} visualization samples to {args.output_dir}")


if __name__ == "__main__":
    main()
