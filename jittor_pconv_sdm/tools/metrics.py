import numpy as np
import jittor as jt


class AverageMeter:
    """
    平均数值记录器 

    用于计算和存储在一系列迭代过程中的数值平均值（如损失值）。
    常用于训练/验证过程中，统计每个 epoch 的平均 loss。
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class MeanIoU:
    """
    平均交并比（Mean IoU）计算器

    用于计算二分类分割任务中的平均交并比 (IoU)。
    注意：输入的是 logits（未经过 Sigmoid 的原始输出），内部会自行处理。
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.intersection = 0.0
        self.union = 0.0

    def update(self, logits, target, threshold=0.5):
        pred = (jt.sigmoid(logits).numpy() > threshold).astype(np.float32)
        label = (target.numpy() > 0.5).astype(np.float32)
        # 交集像素数量和
        inter = (pred * label).sum()
        # 并集像素数量和
        union = ((pred + label) > 0).astype(np.float32).sum()
        self.intersection += float(inter)
        self.union += float(union)

    def get(self):
        # 返回交并比（平滑项防止分母为0）
        return self.intersection / (self.union + np.spacing(1))


class PDFA:
    """
    PDFA 评估指标（检测概率与虚警面积）

    用于评估小目标检测/分割任务的指标：PD (检测概率) 和 FA (虚警面积)。
    核心逻辑：
    1. 将预测图和标签图分别提取连通区域（视为一个个独立的"目标"）。
    2. 通过质心之间的欧氏距离（小于 match_distance）将预测区域与标签区域进行匹配。
    3. 匹配上的预测视为"正确检测" (detected_targets)。
    4. 未匹配上的预测区域视为"虚警" (false_alarm)，累加其像素面积。
    5. 最终计算 PD = 正确检测数 / 真实目标总数，FA = 虚警像素面积 / (图像总面积 * 图像数量)。
    """
    def __init__(self, image_size, threshold=0.5, match_distance=3):
        self.image_size = image_size
        self.threshold = threshold
        self.match_distance = match_distance
        self.reset()

    def reset(self):
        self.false_alarm_area = 0.0
        self.detected_targets = 0.0
        self.total_targets = 0.0
        self.image_count = 0

    def update(self, logits, target):
        pred = (jt.sigmoid(logits).numpy() > self.threshold).astype(np.uint8)
        label = (target.numpy() > 0.5).astype(np.uint8)

        for pred_i, label_i in zip(pred, label):
            # 恢复二维图像矩阵，计算连通区域
            pred_mask = pred_i.reshape(self.image_size, self.image_size)
            label_mask = label_i.reshape(self.image_size, self.image_size)
            # 提取连通区域
            pred_regions = connected_regions(pred_mask)
            label_regions = connected_regions(label_mask)
            self.total_targets += len(label_regions)
            self.image_count += 1
            # 质心匹配
            matched_pred_indices = set()
            for label_region in label_regions:
                label_centroid = np.array(label_region["centroid"])
                for pred_idx, pred_region in enumerate(pred_regions):
                    if pred_idx in matched_pred_indices:
                        continue
                    pred_centroid = np.array(pred_region["centroid"])
                    # 计算欧氏距离，判断是否属于同一目标区域
                    if np.linalg.norm(pred_centroid - label_centroid) < self.match_distance:
                        matched_pred_indices.add(pred_idx)
                        self.detected_targets += 1
                        break
            # 累加虚警面积
            for pred_idx, pred_region in enumerate(pred_regions):
                if pred_idx not in matched_pred_indices:
                    self.false_alarm_area += pred_region["area"]

    def get(self):
        """
        计算并返回最终的评估指标：PD (检测概率) 和 FA (虚警率/面积占比)。
        
        返回:
            pd (float): 检测概率 = 正确检测数 / 总真实目标数（加上微小值防除零）
            fa (float): 归一化虚警面积 = 虚警像素总面积 / (单图面积 * 图像总数量)
        """
        image_area = self.image_size * self.image_size * max(self.image_count, 1)
        fa = self.false_alarm_area / image_area
        pd = self.detected_targets / (self.total_targets + np.spacing(1))
        return pd, fa


def connected_regions(mask):
    """
    查找连通区域的辅助函数

    在二值掩码中查找 8-连通的连通区域，并返回每个区域的面积和质心坐标。
    算法：基于栈的深度优先搜索 (DFS)，遍历所有像素，遇到未访问的前景像素即开始搜索相邻区域。

    Args:
        mask (np.ndarray): 二维二值数组 (H, W)，True/False 或 0/1

    Returns:
        list[dict]: 每个字典包含 "area" (像素个数) 和 "centroid" (质心 y, x)
    """
    mask = mask.astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    regions = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            ys = []
            xs = []

            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                # 检查该像素的8邻域
                for ny in range(max(cy - 1, 0), min(cy + 2, height)):
                    for nx in range(max(cx - 1, 0), min(cx + 2, width)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            regions.append({"area": len(ys), "centroid": (float(np.mean(ys)), float(np.mean(xs)))})

    return regions

