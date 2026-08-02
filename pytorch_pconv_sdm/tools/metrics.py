import numpy as np
import torch


class AverageMeter:
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
    def __init__(self):
        self.reset()

    def reset(self):
        self.intersection = 0.0
        self.union = 0.0

    def update(self, logits, target, threshold=0.5):
        pred = (torch.sigmoid(logits) > threshold).float()
        target = (target > 0.5).float()
        inter = (pred * target).sum().item()
        union = ((pred + target) > 0).float().sum().item()
        self.intersection += inter
        self.union += union

    def get(self):
        return self.intersection / (self.union + np.spacing(1))


class PDFA:
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
        pred = (torch.sigmoid(logits) > self.threshold).detach().cpu().numpy().astype(np.uint8)
        label = (target > 0.5).detach().cpu().numpy().astype(np.uint8)

        for pred_i, label_i in zip(pred, label):
            pred_mask = pred_i.reshape(self.image_size, self.image_size)
            label_mask = label_i.reshape(self.image_size, self.image_size)

            pred_regions = connected_regions(pred_mask)
            label_regions = connected_regions(label_mask)
            self.total_targets += len(label_regions)
            self.image_count += 1

            matched_pred_indices = set()
            for label_region in label_regions:
                label_centroid = np.array(label_region["centroid"])
                for pred_idx, pred_region in enumerate(pred_regions):
                    if pred_idx in matched_pred_indices:
                        continue
                    pred_centroid = np.array(pred_region["centroid"])
                    if np.linalg.norm(pred_centroid - label_centroid) < self.match_distance:
                        matched_pred_indices.add(pred_idx)
                        self.detected_targets += 1
                        break

            for pred_idx, pred_region in enumerate(pred_regions):
                if pred_idx not in matched_pred_indices:
                    self.false_alarm_area += pred_region["area"]

    def get(self):
        image_area = self.image_size * self.image_size * max(self.image_count, 1)
        fa = self.false_alarm_area / image_area
        pd = self.detected_targets / (self.total_targets + np.spacing(1))
        return pd, fa


def connected_regions(mask):
    """Return area and centroid for 8-connected components in a binary mask."""
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
                for ny in range(max(cy - 1, 0), min(cy + 2, height)):
                    for nx in range(max(cx - 1, 0), min(cx + 2, width)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            regions.append(
                {
                    "area": len(ys),
                    "centroid": (float(np.mean(ys)), float(np.mean(xs))),
                }
            )

    return regions
