import math

from jittor import nn
import jittor as jt


class SoftIoULoss(nn.Module):
    """
    Soft IoU Loss
    将网络输出经过 Sigmoid 转为概率，再计算 Soft IoU。
    最终损失 = 1 - 平均 IoU。
    """
    def execute(self, pred_log, target):
        # 将 logits 转换为 [0,1] 概率
        pred = jt.sigmoid(pred_log)

        # 平滑项，避免分母为0
        smooth = 1.0

        # 预测与真实标签的交集
        intersection = pred * target

        # 分别统计交集、预测区域、真实区域面积
        intersection_sum = intersection.sum([1, 2, 3])
        pred_sum = pred.sum([1, 2, 3])
        target_sum = target.sum([1, 2, 3])

        # 计算soft IOU
        iou = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)

        # 返回平均 IoU 损失
        return 1 - iou.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss
    常用于医学分割和红外目标检测，
    更关注预测区域与真实区域的重叠程度。
    """
    def execute(self, pred_log, target):
        # 将 logits 转换为 [0,1] 概率
        pred = jt.sigmoid(pred_log)

        # 平滑项，避免分母为0
        smooth = 1.0

        # 预测与真实标签的交集
        intersection = pred * target

        # 分别统计交集、预测区域、真实区域面积
        intersection_sum = intersection.sum([1, 2, 3])
        pred_sum = pred.sum([1, 2, 3])
        target_sum = target.sum([1, 2, 3])

        # Dice
        dice = (2 * intersection_sum + smooth) / (pred_sum + target_sum + intersection_sum + smooth)
        return 1 - dice.mean()


class SLSIoULoss(nn.Module):
    """
    SLS loss and SDM loss for mask-based infrared small target segmentation.

    该损失结合了：
        - 基础 IoU 损失（经 α 加权，考虑目标尺度差异）
        - 位置损失（可选）：基于质心的角度和长度差异
        - 动态缩放调制（可选）：根据目标大小和图像尺寸自适应调整 base_loss 和 location_loss 的权重

    参数:
        with_distance (bool): 是否引入位置损失（location_loss），默认为 True。
        dynamic (bool): 是否启用动态缩放调制（根据目标尺寸动态调整权重），默认为 False，为True时就是SDM Loss.
        delta (float): 动态调制中的超参数，控制权重上限，默认为 0.5。
        reference_size (int): 参考图像尺寸（通常为 512），用于计算尺度因子 r_oc，默认为 512。
    """

    def __init__(self, with_distance=True, dynamic=False, delta=0.5, reference_size=512):
        super().__init__()
        self.with_distance = with_distance
        self.dynamic = dynamic
        self.delta = delta  # 动态调制中 β 的上限
        self.reference_size = reference_size

    def execute(self, pred_log, target, warm_epoch=5, epoch=0):
        """
        参数:
            pred_log (jt.Var): 网络输出的 logits，形状 [B, 1, H, W]。
            target (jt.Var): 二值掩码标签，形状 [B, 1, H, W]。
            warm_epoch (int): 预热轮数，在预热期内仅使用基础 IoU 损失，默认为 5。
            epoch (int): 当前训练轮数，用于判断是否处于预热期，默认为 0。

        返回:
            jt.Var: 标量损失值。
        """
        # 将 logits 转换为概率（0~1）
        pred = jt.sigmoid(pred_log)
        h, w = pred.shape[2], pred.shape[3]
        smooth = 0.0

        # 计算soft IOU 的基础统计量
        intersection = pred * target
        intersection_sum = intersection.sum([1, 2, 3])
        pred_sum = pred.sum([1, 2, 3])
        target_sum = target.sum([1, 2, 3])

        # 计算 α 权重（尺度敏感因子）
        # dis 表示预测与目标数量差异的一半的平方，用于惩罚数量不匹配
        dis = jt.pow((pred_sum - target_sum) / 2, 2)
        # α 的分子：较小值 + dis，分母：较大值 + dis，使得 α 在 [0,1] 之间，
        # 当预测和目标数量接近时 α 接近 1，差距大时 α 接近 0（但分子分母都有 smooth 避免除零）
        alpha = (jt.minimum(pred_sum, target_sum) + dis + smooth) / (
            jt.maximum(pred_sum, target_sum) + dis + smooth
        )
        # 计算IOU
        iou = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)

        if epoch <= warm_epoch:
            return 1 - iou.mean()

        siou_loss = alpha * iou
        base_loss = 1 - siou_loss.mean()
        if not self.with_distance:
            return base_loss

        location_loss = location_loss_fn(pred, target)
        if not self.dynamic:
            return base_loss + location_loss

        # r_oc：参考图像面积与当前特征图面积之比，反映尺度变化
        r_oc = (self.reference_size * self.reference_size) / (w * h)

        # beta 的计算：target_sum * delta * r_oc / 81 （81 为经验常数，对应小目标尺寸阈值）
        beta = (target_sum * self.delta * r_oc) / 81

        # 将 beta 限制在 [0, delta] 范围内
        beta = jt.where(beta > self.delta, jt.ones_like(beta) * self.delta, beta)
        beta = beta.mean()

        # 最终损失 = (1+β) * base_loss + (1-β) * location_loss
        # 当目标较大（β 较大）时，更侧重 base_loss；目标较小时，更侧重 location_loss

        return (1 + beta) * base_loss + (1 - beta) * location_loss


def location_loss_fn(pred, target):
    """
    Location Loss:
    计算预测热力图与目标热力图之间的位置损失（Location Loss），
    该损失基于预测和目标热力图中“质心”的极坐标（角度与长度）差异。
    
    损失由两部分组成：
    - 长度损失：预测质心到原点的距离与目标质心到原点距离的比值（取小/取大），
      值越接近1表示长度越一致，损失项为 (1 - 比值)。
    - 角度损失：预测质心与目标质心相对于原点的角度差（使用反正切），
      然后转换为弧度归一化到 [0, 1] 区间内。

    最终损失为 batch 内各样本的长度损失与角度损失之和的平均。

    参数:
        pred (jt.Var): 预测的热力图，形状为 [batch_size, 1, H, W]，值域通常在 [0,1]。
        target (jt.Var): 目标热力图，形状与 pred 相同。

    返回:
        jt.Var: 标量损失值。
    """
    # 初始化总损失
    loss = jt.array(0.0)

    # pred.shape(B,1,H,W)
    batch_size = pred.shape[0]
    h, w = pred.shape[2], pred.shape[3]

    # 归一化x、y坐标网络 (1,H,W)
    x_index = jt.arange(0, w).float32().reshape(1, 1, w).repeat(1, h, 1) / w
    y_index = jt.arange(0, h).float32().reshape(1, h, 1).repeat(1, 1, w) / h

    # 平滑项
    smooth = 1e-8

    for i in range(batch_size):
        # 计算预测热力图 i 的质心坐标（加权平均）
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()

        # 计算目标热力图 i 的质心坐标（加权平均）
        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        # 角度损失：将角度差（通过 atan 计算）归一化到 [0,1]
        # 4/pi^2 保证当角度差为 pi/2 时损失为1，因为 atan 值域 [-pi/2, pi/2]
        angle_loss = (4 / (math.pi**2)) * jt.sqr(
            jt.atan(pred_centery / (pred_centerx + smooth)) # 预测质心角度
            - jt.atan(target_centery / (target_centerx + smooth))   # 目标质心角度
        )
        # 计算预测质心到原点的长度（距离）
        pred_length = jt.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
        # 计算目标质心到原点的长度
        target_length = jt.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)
        # 长度损失：用 min/max 比值衡量长度接近程度，比值越接近1损失越小
        length_loss = jt.minimum(pred_length, target_length) / (jt.maximum(pred_length, target_length) + smooth)
        # 累加当前样本损失： (1 - 长度损失) + 角度损失，然后除以 batch_size 取平均
        loss = loss + (1 - length_loss + angle_loss) / batch_size

    return loss
