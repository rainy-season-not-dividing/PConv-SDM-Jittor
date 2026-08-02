import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pytorch_pconv_sdm.losses import SLSIoULoss, compute_deep_supervision_loss
from pytorch_pconv_sdm.models import MSHNet, PConv


def check_shapes():
    x = torch.randn(2, 16, 64, 64)
    pconv = PConv(16, 32, kernel_size=3, stride=1)
    pconv.eval()
    y = pconv(x)
    assert y.shape == (2, 32, 64, 64), f"Unexpected PConv shape: {tuple(y.shape)}"

    image = torch.randn(2, 3, 64, 64)
    for use_pconv in (False, True):
        model = MSHNet(input_channels=3, use_pconv=use_pconv)
        model.eval()
        masks, pred = model(image, warm_flag=True)
        assert pred.shape == (2, 1, 64, 64), f"Unexpected output shape: {tuple(pred.shape)}"
        assert [tuple(mask.shape) for mask in masks] == [
            (2, 1, 64, 64),
            (2, 1, 32, 32),
            (2, 1, 16, 16),
            (2, 1, 8, 8),
        ]


def check_loss_backward(use_pconv, loss_type):
    model = MSHNet(input_channels=3, use_pconv=use_pconv)
    model.train()
    image = torch.randn(2, 3, 64, 64)
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 20:24, 30:34] = 1.0

    criterion = SLSIoULoss(with_distance=True, dynamic=(loss_type == "sdm"))
    masks, pred = model(image, warm_flag=True)
    loss = compute_deep_supervision_loss(masks, pred, target, criterion, warm_epoch=0, epoch=1)
    assert torch.isfinite(loss), f"{loss_type} loss is not finite"
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0)
    assert torch.isfinite(grad_norm), f"{loss_type} grad norm is not finite"
    print(f"use_pconv={use_pconv} loss_type={loss_type} loss={loss.item():.6f}")


def main():
    torch.manual_seed(0)
    check_shapes()
    check_loss_backward(use_pconv=False, loss_type="sls")
    check_loss_backward(use_pconv=True, loss_type="sls")
    check_loss_backward(use_pconv=False, loss_type="sdm")
    check_loss_backward(use_pconv=True, loss_type="sdm")
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
