from jittor import nn
import jittor as jt

from .pconv import Identity, PConv


class ChannelAttention(nn.Module):
    """
    通道注意力模块（Channel Attention Module）。

    该模块用于自适应地调整每个通道的重要性。它同时利用全局平均池化和全局最大池化
    来聚合空间信息，然后通过一个共享的两层 MLP（由两个 1x1 卷积实现）生成通道权重，
    最后将两个分支的结果相加并通过 Sigmoid 激活，得到最终的通道注意力图。

    参考论文：CBAM: Convolutional Block Attention Module.

    Args:
        in_channels (int): 输入特征图的通道数。
        ratio (int): 通道缩减比例，用于控制隐藏层的大小。隐藏层通道数为
                     max(in_channels // ratio, 1)，默认值为 16。
    """
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        hidden_channels = max(in_channels // ratio, 1)
        # 自适应全局平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 自适应全局最大池化层
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        """
        前向传播，计算通道注意力权重。

        Args:
            x (torch.Tensor): 输入特征图，形状为 (N, C, H, W)，其中 N 为批量大小，
                              C 为通道数，H 和 W 为空间高度和宽度。

        Returns:
            torch.Tensor: 通道注意力权重，形状为 (N, C, 1, 1)，每个元素取值范围为 [0, 1]。
        """
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    空间注意力模块（Spatial Attention Module）。

    该模块用于关注特征图中不同空间位置的重要性。它首先在通道维度上对输入分别进行
    平均池化和最大池化，得到两个 2D 特征图（形状 N x 1 x H x W），然后将两者拼接
    为一个 N x 2 x H x W 的特征图，再通过一个卷积核（大小为 7 或 3）的卷积层生成
    单通道的空间注意力图，最后经过 Sigmoid 激活。

    Args:
        kernel_size (int): 卷积核的大小，仅允许为 3 或 7，默认值为 7。
                          较大的核能捕获更广的感受野，但计算量稍大。
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        """
        前向传播，计算空间注意力权重。

        Args:
            x (torch.Tensor): 输入特征图，形状为 (N, C, H, W)，其中 N 为批量大小，
                              C 为通道数，H 和 W 为空间维度。

        Returns:
            torch.Tensor: 空间注意力权重，形状为 (N, 1, H, W)，每个元素取值范围为 [0, 1]。
        """
        avg_out = x.mean([1], keepdims=True)
        max_out = x.max([1], keepdims=True)
        return self.sigmoid(self.conv(jt.concat([avg_out, max_out], dim=1)))


class ResBlock(nn.Module):
    """
    MSHNet 残差块，可在在第一个空间卷积层中选用 PConv（部分卷积）。

    Args:
        in_channels (int): 输入特征图的通道数。
        out_channels (int): 输出特征图的通道数（即该残差块输出的通道数）。
        stride (int): 第一个卷积层的步长，默认为 1。若步长不为 1，则不能与 PConv 同时使用。
        use_pconv (bool): 是否在第一层使用 PConv（Partial Convolution），默认为 False。
        pconv_kernel (int): 若使用 PConv，指定其卷积核大小，默认为 3。
    """
    def __init__(self, in_channels, out_channels, stride=1, use_pconv=False, pconv_kernel=3):
        super().__init__()
        if stride != 1 and use_pconv:
            raise ValueError("Current PConv integration is intended for stride=1 blocks.")

        # 第一层卷积可选PConv作为第一个空间卷积层
        if use_pconv:
            self.conv1 = PConv(in_channels, out_channels, kernel_size=pconv_kernel, stride=stride)
            self.bn1 = Identity()
            self.relu1 = Identity()
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu1 = nn.ReLU()

        # 第二层卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 恒等映射：匹配residual和out的张量形状，后续才能直接相加
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None
        # 通道注意力
        self.ca = ChannelAttention(out_channels)
        # 空间注意力
        self.sa = SpatialAttention()
        self.relu = nn.ReLU()

    def execute(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out) * out
        out = self.sa(out) * out
        return self.relu(out + residual)


class MSHNet(nn.Module):
    """
    MSHNet 主网络：一个 U-Net 风格的编码器-解码器结构，带有跳跃连接和多尺度输出。

    网络包含：
        - 初始卷积层
        - 四个编码器阶段（每个阶段包含若干 ResBlock，中间穿插最大池化下采样）
        - 中间层（更深的特征）
        - 四个解码器阶段（通过上采样恢复分辨率，并与编码器对应层拼接）
        - 多尺度输出（四个中间尺度和一个最终融合输出）

    Args:
        input_channels (int): 输入图像的通道数，默认为 3。
        use_pconv (bool): 是否在 ResBlock 中使用 PConv，默认为 False。
        pconv_kernel (int): PConv 的卷积核大小，默认为 3。
    """
    def __init__(self, input_channels=3, use_pconv=False, pconv_kernel=3):
        super().__init__()
        channels = [16, 32, 64, 128, 256]
        blocks = [2, 2, 2, 2]
        # 最大池化
        self.pool = nn.MaxPool2d(2, 2)
        # 上采样，恢复分辨率，双线性插值确保平滑，边角严格对齐，保留边界信息
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        # 适配：将原始图像的通道调整到编码层第一个编码器的输入通道数
        self.conv_init = nn.Conv2d(input_channels, channels[0], 1, 1)
        # 编码层
        self.encoder_0 = self._make_layer(channels[0], channels[0], use_pconv, pconv_kernel)
        self.encoder_1 = self._make_layer(channels[0], channels[1], use_pconv, pconv_kernel, blocks[0])
        self.encoder_2 = self._make_layer(channels[1], channels[2], use_pconv, pconv_kernel, blocks[1])
        self.encoder_3 = self._make_layer(channels[2], channels[3], use_pconv, pconv_kernel, blocks[2])
        # 中间层
        self.middle_layer = self._make_layer(channels[3], channels[4], use_pconv, pconv_kernel, blocks[3])
        # 解码层
        self.decoder_3 = self._make_layer(channels[3] + channels[4], channels[3], use_pconv, pconv_kernel, blocks[2])
        self.decoder_2 = self._make_layer(channels[2] + channels[3], channels[2], use_pconv, pconv_kernel, blocks[1])
        self.decoder_1 = self._make_layer(channels[1] + channels[2], channels[1], use_pconv, pconv_kernel, blocks[0])
        self.decoder_0 = self._make_layer(channels[0] + channels[1], channels[0], use_pconv, pconv_kernel)
        # 输出层
        self.output_0 = nn.Conv2d(channels[0], 1, 1)
        self.output_1 = nn.Conv2d(channels[1], 1, 1)
        self.output_2 = nn.Conv2d(channels[2], 1, 1)
        self.output_3 = nn.Conv2d(channels[3], 1, 1)
        # 融合各层预测输出
        self.final = nn.Conv2d(4, 1, 3, 1, 1)

    def _make_layer(self, in_channels, out_channels, use_pconv, pconv_kernel, block_num=1):
        """
        构建一个由多个 ResBlock 组成的`nn.Sequential`容器

        Args:
            in_channels (int): 输入特征图的通道数。
            out_channels (int): 输出特征图的通道数。
            use_pconv (bool): 是否在 ResBlock 中使用部分卷积（PConv）。
            pconv_kernel (int): 若使用 PConv，指定卷积核大小。
            block_num (int): 该层包含的 ResBlock 总数，默认为 1。

        Returns:
            nn.Sequential: 包含 `block_num` 个 ResBlock 的顺序容器。
        """
        layers = [ResBlock(in_channels, out_channels, use_pconv=use_pconv, pconv_kernel=pconv_kernel)]
        for _ in range(block_num - 1):
            layers.append(ResBlock(out_channels, out_channels, use_pconv=use_pconv, pconv_kernel=pconv_kernel))
        return nn.Sequential(*layers)

    def execute(self, x, warm_flag=False):
        """
        前向传播。

        Args:
            x (jt.Var): 输入图像，形状 (N, input_channels, H, W)。
            warm_flag (bool): 若为 True，则返回所有尺度的中间预测以及最终融合输出；
                               若为 False，仅返回最终的单尺度输出（即 decoder_0 的输出）。

        Returns:
            tuple: (中间预测列表或空列表, 最终输出)
                   - 若 warm_flag=True，返回 ([mask0, mask1, mask2, mask3], output)
                   - 若 warm_flag=False，返回 ([], 最终输出)
        """
        # 编码层，下采样
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))
        # 解码层，上采样
        x_d3 = self.decoder_3(jt.concat([x_e3, self.up(x_m)], dim=1))
        x_d2 = self.decoder_2(jt.concat([x_e2, self.up(x_d3)], dim=1))
        x_d1 = self.decoder_1(jt.concat([x_e1, self.up(x_d2)], dim=1))
        x_d0 = self.decoder_0(jt.concat([x_e0, self.up(x_d1)], dim=1))
        # 预热训练，关注浅层低分辨率的损失
        if warm_flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)
            output = self.final(jt.concat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
            return [mask0, mask1, mask2, mask3], output
        # 推理阶段，只关注最后解码的高分辨率预测图
        return [], self.output_0(x_d0)
