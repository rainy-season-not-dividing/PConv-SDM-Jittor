from jittor import nn


def compute_deep_supervision_loss(outputs, pred, target, criterion, warm_epoch, epoch):
    loss = criterion(pred, target, warm_epoch, epoch)
    down = nn.MaxPool2d(2, 2)
    current_target = target
    for index, mask in enumerate(outputs):
        if index > 0:
            current_target = down(current_target)
        loss = loss + criterion(mask, current_target, warm_epoch, epoch)
    return loss / (len(outputs) + 1)

