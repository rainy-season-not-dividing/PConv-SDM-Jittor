import torch
import torch.nn as nn


def autopad(kernel_size, padding=None, dilation=1):
    """Return padding that keeps the standard convolution output shape."""
    if dilation > 1:
        if isinstance(kernel_size, int):
            kernel_size = dilation * (kernel_size - 1) + 1
        else:
            kernel_size = [dilation * (x - 1) + 1 for x in kernel_size]
    if padding is None:
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = [x // 2 for x in kernel_size]
    return padding


class ConvBNAct(nn.Module):
    """Convolution followed by batch normalization and activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        activation=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if activation is True else activation if isinstance(activation, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class PConv(nn.Module):
    """Pinwheel-shaped convolution with asymmetric zero padding.

    This follows the official PyTorch reference in
    `code/pconv_sdloss/model/APConv.py`, keeping the spatial size unchanged
    when `stride=1`.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("PConv requires out_channels to be divisible by 4.")

        k = kernel_size
        pads = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = nn.ModuleList([nn.ZeroPad2d(padding=p) for p in pads])
        self.cw = ConvBNAct(in_channels, out_channels // 4, (1, k), stride, padding=0)
        self.ch = ConvBNAct(in_channels, out_channels // 4, (k, 1), stride, padding=0)
        self.fuse = ConvBNAct(out_channels, out_channels, 2, 1, padding=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.fuse(torch.cat([yw0, yw1, yh0, yh1], dim=1))
