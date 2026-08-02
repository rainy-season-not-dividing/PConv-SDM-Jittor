import argparse
import json
import os
import os.path as osp
import sys
import time
from pathlib import Path

import jittor as jt
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jittor_pconv_sdm.dataset import IRSTDDataset
from jittor_pconv_sdm.losses import SLSIoULoss, compute_deep_supervision_loss
from jittor_pconv_sdm.models import MSHNet
from jittor_pconv_sdm.tools.adagrad import Adagrad
from jittor_pconv_sdm.tools.metrics import AverageMeter, MeanIoU, PDFA


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Jittor PConv-SDM baseline.")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="mshnet_sls")
    parser.add_argument("--config-file", type=str, default=osp.join("jittor_pconv_sdm", "configs", "ablation.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--warm-epoch", type=int, default=5)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default=osp.join("runs", "jittor_pconv_sdm"))
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def load_ablation(config_file, config_name):
    with open(config_file, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if config_name not in configs:
        raise KeyError(f"Unknown config '{config_name}'. Available configs: {', '.join(configs)}")
    return configs[config_name]


def build_loss(loss_type):
    if loss_type == "sls":
        return SLSIoULoss(with_distance=True, dynamic=False)
    if loss_type == "sdm":
        return SLSIoULoss(with_distance=True, dynamic=True)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def to_var(x):
    return x if isinstance(x, jt.Var) else jt.array(x)


def train_one_epoch(model, loader, optimizer, criterion, warm_epoch, epoch):
    model.train()
    meter = AverageMeter()
    progress = tqdm(loader, desc=f"Epoch {epoch}")
    warm_flag = epoch > warm_epoch

    for images, masks in progress:
        images = to_var(images).float32()
        masks = to_var(masks).float32()
        outputs, pred = model(images, warm_flag=warm_flag)
        loss = compute_deep_supervision_loss(outputs, pred, masks, criterion, warm_epoch, epoch)

        optimizer.step(loss)
        batch_size = images.shape[0]
        meter.update(float(loss.item()), batch_size)
        progress.set_postfix(loss=f"{meter.avg:.4f}")

    return meter.avg


def evaluate(model, loader, image_size):
    model.eval()
    miou = MeanIoU()
    pdfa = PDFA(image_size=image_size)
    for images, masks in tqdm(loader, desc="Eval"):
        images = to_var(images).float32()
        masks = to_var(masks).float32()
        _, pred = model(images, warm_flag=False)
        miou.update(pred, masks)
        pdfa.update(pred, masks)
    pd, fa = pdfa.get()
    return {"IoU": miou.get(), "Pd": pd, "Fa": fa}


def main():
    args = parse_args()
    jt.flags.use_cuda = 1 if args.device == "cuda" else 0
    ablation = load_ablation(args.config_file, args.config)

    train_set = IRSTDDataset(
        args.dataset_dir,
        mode="train",
        base_size=args.base_size,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_set = IRSTDDataset(
        args.dataset_dir,
        mode="val",
        base_size=args.base_size,
        crop_size=args.crop_size,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    model = MSHNet(input_channels=3, use_pconv=ablation["use_pconv"])
    criterion = build_loss(ablation["loss_type"])
    optimizer = Adagrad(model.parameters(), lr=args.lr)

    run_dir = osp.join(args.save_dir, f"{args.config}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    best_iou = 0.0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_set, optimizer, criterion, args.warm_epoch, epoch)
        metrics = evaluate(model, val_set, args.base_size)
        line = (
            f"epoch={epoch} loss={train_loss:.6f} "
            f"IoU={metrics['IoU']:.6f} Pd={metrics['Pd']:.6f} Fa={metrics['Fa']:.8f}"
        )
        print(line)
        with open(osp.join(run_dir, "metrics.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if metrics["IoU"] > best_iou:
            best_iou = metrics["IoU"]
            jt.save(model.state_dict(), osp.join(run_dir, "best_weight.pkl"))

        jt.save(
            {"net": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "iou": best_iou},
            osp.join(run_dir, "checkpoint.pkl"),
        )


if __name__ == "__main__":
    main()
