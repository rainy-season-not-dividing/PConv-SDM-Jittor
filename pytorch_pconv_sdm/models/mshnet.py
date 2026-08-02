import torch
import torch.nn as nn

from .pconv import PConv


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        hidden_channels = max(in_channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class ResBlock(nn.Module):
    """MSHNet residual block, with optional PConv in the first spatial layer."""

    def __init__(self, in_channels, out_channels, stride=1, use_pconv=False, pconv_kernel=3):
        super().__init__()
        if stride != 1 and use_pconv:
            raise ValueError("Current PConv integration is intended for stride=1 blocks.")

        if use_pconv:
            self.conv1 = PConv(in_channels, out_channels, kernel_size=pconv_kernel, stride=stride)
            self.bn1 = nn.Identity()
            self.relu1 = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out) * out
        out = self.sa(out) * out
        return self.relu(out + residual)


class MSHNet(nn.Module):
    def __init__(self, input_channels=3, use_pconv=False, pconv_kernel=3):
        super().__init__()
        channels = [16, 32, 64, 128, 256]
        blocks = [2, 2, 2, 2]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, channels[0], 1, 1)

        self.encoder_0 = self._make_layer(channels[0], channels[0], use_pconv, pconv_kernel)
        self.encoder_1 = self._make_layer(channels[0], channels[1], use_pconv, pconv_kernel, blocks[0])
        self.encoder_2 = self._make_layer(channels[1], channels[2], use_pconv, pconv_kernel, blocks[1])
        self.encoder_3 = self._make_layer(channels[2], channels[3], use_pconv, pconv_kernel, blocks[2])

        self.middle_layer = self._make_layer(channels[3], channels[4], use_pconv, pconv_kernel, blocks[3])

        self.decoder_3 = self._make_layer(channels[3] + channels[4], channels[3], use_pconv, pconv_kernel, blocks[2])
        self.decoder_2 = self._make_layer(channels[2] + channels[3], channels[2], use_pconv, pconv_kernel, blocks[1])
        self.decoder_1 = self._make_layer(channels[1] + channels[2], channels[1], use_pconv, pconv_kernel, blocks[0])
        self.decoder_0 = self._make_layer(channels[0] + channels[1], channels[0], use_pconv, pconv_kernel)

        self.output_0 = nn.Conv2d(channels[0], 1, 1)
        self.output_1 = nn.Conv2d(channels[1], 1, 1)
        self.output_2 = nn.Conv2d(channels[2], 1, 1)
        self.output_3 = nn.Conv2d(channels[3], 1, 1)
        self.final = nn.Conv2d(4, 1, 3, 1, 1)

    def _make_layer(self, in_channels, out_channels, use_pconv, pconv_kernel, block_num=1):
        layers = [ResBlock(in_channels, out_channels, use_pconv=use_pconv, pconv_kernel=pconv_kernel)]
        for _ in range(block_num - 1):
            layers.append(ResBlock(out_channels, out_channels, use_pconv=use_pconv, pconv_kernel=pconv_kernel))
        return nn.Sequential(*layers)

    def forward(self, x, warm_flag=False):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], dim=1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], dim=1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], dim=1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], dim=1))

        if warm_flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)
            output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
            return [mask0, mask1, mask2, mask3], output

        return [], self.output_0(x_d0)
