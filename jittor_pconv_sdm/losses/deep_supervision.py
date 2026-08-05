from jittor import nn


def compute_deep_supervision_loss(outputs, pred, target, criterion, warm_epoch, epoch):
    """
    计算深度监督损失（Deep Supervision Loss）。

    深度监督策略：模型除了输出主预测图（pred）外，还会输出多个中间层的辅助预测图（outputs）。
    本函数对每个预测图分别计算与真实标签（经过适当下采样）的损失，并将所有损失求和后取平均，
    以鼓励模型中间层也学习有意义的特征表示，加速收敛并提高分割精度。

    Args:
        outputs (list[jt.Var]): 模型中间层输出的辅助预测图列表，顺序从浅层到深层（空间分辨率递减）
        pred (jt.Var): 模型最终的主预测图（通常分辨率与输入相同）
        target (jt.Var): 真实标签（与输入图像分辨率相同）
        criterion (Loss): 损失函数（如交叉熵、Dice损失等），需支持 (pred, target, warm_epoch, epoch) 调用形式
        warm_epoch (int): 预热阶段轮数，用于损失函数内部可能的动态加权策略
        epoch (int): 当前训练轮次

    Returns:
        jt.Var: 标量张量，表示平均深度监督损失
    """
    loss = criterion(pred, target, warm_epoch, epoch)
    down = nn.MaxPool2d(2, 2)
    current_target = target
    for index, mask in enumerate(outputs):
        if index > 0:
            current_target = down(current_target)
        loss = loss + criterion(mask, current_target, warm_epoch, epoch)
    return loss / (len(outputs) + 1)

