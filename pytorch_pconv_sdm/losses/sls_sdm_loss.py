import torch
import torch.nn as nn


class SoftIoULoss(nn.Module):
    def forward(self, pred_log, target):
        pred = torch.sigmoid(pred_log)
        smooth = 1.0
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        iou = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)
        return 1 - iou.mean()


class DiceLoss(nn.Module):
    def forward(self, pred_log, target):
        pred = torch.sigmoid(pred_log)
        smooth = 1.0
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        dice = (2 * intersection_sum + smooth) / (pred_sum + target_sum + intersection_sum + smooth)
        return 1 - dice.mean()


class SLSIoULoss(nn.Module):
    """SLS loss and SDM loss for mask-based infrared small target segmentation.

    `dynamic=False, with_distance=True` reproduces the MSHNet SLS-style loss.
    `dynamic=True, with_distance=True` enables the SDM weighting from the PConv
    reference loss.
    """

    def __init__(self, with_distance=True, dynamic=False, delta=0.5, reference_size=512):
        super().__init__()
        self.with_distance = with_distance
        self.dynamic = dynamic
        self.delta = delta
        self.reference_size = reference_size

    def forward(self, pred_log, target, warm_epoch=5, epoch=0):
        pred = torch.sigmoid(pred_log)
        h, w = pred.shape[2], pred.shape[3]
        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        dis = torch.pow((pred_sum - target_sum) / 2, 2)
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (
            torch.max(pred_sum, target_sum) + dis + smooth
        )
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

        r_oc = (self.reference_size * self.reference_size) / (w * h)
        beta = (target_sum * self.delta * r_oc) / 81
        beta = torch.where(beta > self.delta, torch.full_like(beta, self.delta), beta)
        beta = beta.mean()
        return (1 + beta) * base_loss + (1 - beta) * location_loss


def location_loss_fn(pred, target):
    loss = pred.new_tensor(0.0)
    batch_size = pred.shape[0]
    h, w = pred.shape[2], pred.shape[3]
    x_index = torch.arange(0, w, 1, device=pred.device, dtype=pred.dtype).view(1, 1, w).repeat(1, h, 1) / w
    y_index = torch.arange(0, h, 1, device=pred.device, dtype=pred.dtype).view(1, h, 1).repeat(1, 1, w) / h
    smooth = 1e-8

    for i in range(batch_size):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()
        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        angle_loss = (4 / (torch.pi**2)) * torch.square(
            torch.arctan(pred_centery / (pred_centerx + smooth))
            - torch.arctan(target_centery / (target_centerx + smooth))
        )
        pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
        target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)
        length_loss = torch.min(pred_length, target_length) / (torch.max(pred_length, target_length) + smooth)
        loss = loss + (1 - length_loss + angle_loss) / batch_size

    return loss
