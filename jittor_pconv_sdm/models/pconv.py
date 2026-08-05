from jittor import nn
import jittor as jt


def autopad(kernel_size, padding=None, dilation=1):
    """
    自动计算填充大小，以确保标准卷积操作后输出特征图的尺寸不变（Same Padding）。

    该函数会根据卷积核大小和空洞率（dilation）计算出合适的填充值。
    若 dilation > 1，会先计算实际等效卷积核大小（感受野），再计算填充。

    Args:
        kernel_size (int 或 tuple): 卷积核的尺寸。
        padding (int 或 tuple, 可选): 手动指定的填充大小。若为 None，则自动计算。
        dilation (int): 空洞卷积的空洞率，默认为 1（即标准卷积），给卷积核元素之间插入空格的参数，插入空格数为（dialation-1)。

    Returns:
        int 或 list: 计算出的填充大小，与 kernel_size 类型保持一致。
    """

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


class Identity(nn.Module):
    def execute(self, x):
        return x


class ConvBNAct(nn.Module):
    """
    标准卷积块：由卷积层（Conv2d）+ 批归一化层（BatchNorm2d）+ 激活函数（Activation）组成。

    该模块将三个常用操作打包，简化代码复用。激活函数支持灵活配置。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int 或 tuple): 卷积核大小，默认为 1。
        stride (int 或 tuple): 步长，默认为 1。
        padding (int 或 tuple, 可选): 填充大小。若为 None，则通过 autopad 自动计算。
        groups (int): 分组卷积的组数，默认为 1。
        dilation (int): 空洞率，默认为 1。
        activation (bool 或 nn.Module): 激活函数配置。
            - 若为 True，使用默认的 SiLU（Swish）激活函数。
            - 若为 False，使用 Identity（即无激活函数）。
            - 若传入 nn.Module 实例，则直接使用该激活函数。
    """

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
        # 默认标准卷积
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
        # 批量归一化
        self.bn = nn.BatchNorm2d(out_channels)
        # 激活函数
        self.act = nn.SiLU() if activation is True else activation if isinstance(activation, nn.Module) else Identity() # 默认采用silu（）激活，也可以用传参的激活函数，或者无激活函数

    def execute(self, x):
        return self.act(self.bn(self.conv(x)))


class PConv(nn.Module):
    """
    风车形非对称卷积（Pinwheel-shaped Partial Convolution），一种轻量级替代标准 3x3 卷积的方案。

    该模块通过 4 个方向的不对称零填充，结合水平（1xk）和垂直（kx1）非对称卷积核，
    提取多方向特征，最后通过 2x2 融合卷积恢复空间尺寸。
    参数量约为标准 3x3 卷积的 1/3，同时保持输入输出尺寸一致。

    注意：要求 out_channels 必须能被 4 整除，因为内部会将输出通道均分为 4 个分支。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数（必须能被 4 整除）。
        kernel_size (int): 卷积核大小（如 3 或 5），用于水平和垂直非对称卷积，默认为 3。
        stride (int): 步长（当前仅支持 1，以保证尺寸对齐），默认为 1。

    Raises:
        ValueError: 当 out_channels 不能被 4 整除时抛出。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("PConv requires out_channels to be divisible by 4.")

        k = kernel_size
        pads = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)] # 左右上下
        self.pad = [nn.ZeroPad2d(padding=p) for p in pads]
        # 水平卷积
        self.cw = ConvBNAct(in_channels, out_channels // 4, (1, k), stride, padding=0)
        # 垂直卷积
        self.ch = ConvBNAct(in_channels, out_channels // 4, (k, 1), stride, padding=0)
        # 2*2标准卷积
        self.fuse = ConvBNAct(out_channels, out_channels, 2, 1, padding=0)

    def execute(self, x):
        yw0 = self.cw(self.pad[0](x))   # shape(B,C_out//4,H_in+1,H_out+1)
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.fuse(jt.concat([yw0, yw1, yh0, yh1], dim=1))    # shape(B,C_out,H_in,H_out)

