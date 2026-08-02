import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jittor as jt

from jittor_pconv_sdm.losses import SLSIoULoss, compute_deep_supervision_loss
from jittor_pconv_sdm.models import MSHNet, PConv
from jittor_pconv_sdm.tools.adagrad import Adagrad


def check_shapes():
    x = jt.randn((2, 16, 64, 64))
    pconv = PConv(16, 32, kernel_size=3, stride=1)
    pconv.eval()
    y = pconv(x)
    assert tuple(y.shape) == (2, 32, 64, 64), f"Unexpected PConv shape: {tuple(y.shape)}"

    image = jt.randn((2, 3, 64, 64))
    for use_pconv in (False, True):
        model = MSHNet(input_channels=3, use_pconv=use_pconv)
        model.eval()
        masks, pred = model(image, warm_flag=True)
        assert tuple(pred.shape) == (2, 1, 64, 64), f"Unexpected output shape: {tuple(pred.shape)}"
        assert [tuple(mask.shape) for mask in masks] == [
            (2, 1, 64, 64),
            (2, 1, 32, 32),
            (2, 1, 16, 16),
            (2, 1, 8, 8),
        ]


def check_loss_backward(use_pconv, loss_type):
    model = MSHNet(input_channels=3, use_pconv=use_pconv)
    model.train()
    image = jt.randn((2, 3, 64, 64))
    target = jt.zeros((2, 1, 64, 64))
    target[:, :, 20:24, 30:34] = 1.0

    criterion = SLSIoULoss(with_distance=True, dynamic=(loss_type == "sdm"))
    optimizer = Adagrad(model.parameters(), lr=0.01)
    masks, pred = model(image, warm_flag=True)
    loss = compute_deep_supervision_loss(masks, pred, target, criterion, warm_epoch=0, epoch=1)
    assert bool(jt.isfinite(loss).all().item()), f"{loss_type} loss is not finite"
    optimizer.step(loss)
    print(f"use_pconv={use_pconv} loss_type={loss_type} loss={float(loss.item()):.6f}")


def main():
    jt.set_global_seed(0)
    check_shapes()
    check_loss_backward(use_pconv=False, loss_type="sls")
    check_loss_backward(use_pconv=True, loss_type="sls")
    check_loss_backward(use_pconv=False, loss_type="sdm")
    check_loss_backward(use_pconv=True, loss_type="sdm")
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
