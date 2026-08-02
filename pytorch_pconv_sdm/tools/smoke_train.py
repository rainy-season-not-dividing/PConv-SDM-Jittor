import argparse
import sys
from pathlib import Path

import torch
from torch.optim import Adagrad

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pytorch_pconv_sdm.losses import SLSIoULoss, compute_deep_supervision_loss
from pytorch_pconv_sdm.models import MSHNet


def parse_args():
    parser = argparse.ArgumentParser(description="Run a tiny random-data training smoke test.")
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--config", choices=["mshnet_sls", "mshnet_pconv_sls", "mshnet_sdm", "mshnet_pconv_sdm"], default="mshnet_pconv_sdm")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    use_pconv = "pconv" in args.config
    use_sdm = args.config.endswith("sdm")
    device = torch.device(args.device)

    torch.manual_seed(0)
    model = MSHNet(input_channels=3, use_pconv=use_pconv).to(device)
    criterion = SLSIoULoss(with_distance=True, dynamic=use_sdm)
    optimizer = Adagrad(model.parameters(), lr=0.01)

    for step in range(args.max_iters):
        image = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        target = torch.zeros(args.batch_size, 1, args.image_size, args.image_size, device=device)
        start = args.image_size // 3
        target[:, :, start : start + 4, start : start + 4] = 1.0

        masks, pred = model(image, warm_flag=True)
        loss = compute_deep_supervision_loss(masks, pred, target, criterion, warm_epoch=0, epoch=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"iter={step} loss={loss.item():.6f}")

    print("Smoke train passed.")


if __name__ == "__main__":
    main()
