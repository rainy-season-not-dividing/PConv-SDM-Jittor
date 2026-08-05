import os.path as osp
import random

import numpy as np
from jittor.dataset import Dataset
from PIL import Image, ImageFilter, ImageOps


class IRSTDDataset(Dataset):
    """
    IRSTD-1K 数据集读取类（Jittor）。

    数据集目录结构：
        dataset_dir/
        ├── trainval.txt
        ├── test.txt
        ├── images/
        │     ├── xxx.png
        └── masks/
              ├── xxx.png

    功能：
        1. 根据 train/test 读取图片名称
        2. 加载图像与标签
        3. 训练阶段执行数据增强
        4. 测试阶段执行固定预处理
        5. 转换为网络可输入的数据格式
    """

    def __init__(self, dataset_dir, mode="train", base_size=256, crop_size=256, **dataset_attrs):
        """
        初始化数据集。

        参数：
            dataset_dir : 数据集根目录
            mode        : train / val / test
            base_size   : 缩放尺寸
            crop_size   : 随机裁剪尺寸
            dataset_attrs : Jittor Dataset额外参数
        """
        super().__init__()
        if mode not in ("train", "val", "test"):
            raise ValueError(f"Unsupported dataset mode: {mode}")

        txtfile = "trainval.txt" if mode == "train" else "test.txt"
        self.list_path = osp.join(dataset_dir, txtfile)
        self.images_dir = osp.join(dataset_dir, "images")
        self.masks_dir = osp.join(dataset_dir, "masks")
        self.mode = mode
        self.base_size = base_size
        self.crop_size = crop_size

        with open(self.list_path, "r", encoding="utf-8") as f:
            self.names = [line.strip() for line in f if line.strip()]

        self.set_attrs(total_len=len(self.names), **dataset_attrs)

    def __getitem__(self, index):
        """
        根据索引读取一张图片及对应标签。

        返回：
            image : (3,H,W)
            mask  : (1,H,W)
        """
        name = self.names[index]
        image = Image.open(osp.join(self.images_dir, f"{name}.png")).convert("RGB")
        mask = Image.open(osp.join(self.masks_dir, f"{name}.png")).convert("L")

        if self.mode == "train":
            image, mask = self._sync_transform(image, mask)
        else:
            image, mask = self._val_transform(image, mask)

        return self._image_to_array(image), self._mask_to_array(mask)

    def _sync_transform(self, image, mask):
        """
        训练阶段同步数据增强。

        注意：
            Image与Mask必须做完全相同的几何变换，
            否则标签位置会错误。
        """
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # 几何变换，增加数据多样性， 提高模型泛化能力
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 随机缩放，随机尺度增强
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        width, height = image.size
        if height > width:
            out_h = long_size
            out_w = int(width * long_size / height + 0.5)   
            short_size = out_w
        else:
            out_w = long_size
            out_h = int(height * long_size / width + 0.5)
            short_size = out_h
        # 双线性插值
        image = image.resize((out_w, out_h), Image.BILINEAR)
        # 最近邻插值
        mask = mask.resize((out_w, out_h), Image.NEAREST)

        # 裁剪填充
        if short_size < self.crop_size:
            pad_h = self.crop_size - out_h if out_h < self.crop_size else 0
            pad_w = self.crop_size - out_w if out_w < self.crop_size else 0
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0) # 左上右下
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=0)

        # 随机裁剪
        width, height = image.size
        x1 = random.randint(0, width - self.crop_size)
        y1 = random.randint(0, height - self.crop_size)
        image = image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        # 对图像进行随机轻微强度的高斯模糊数据增强，提高模型对图像质量变化的适应能力
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return image, mask

    def _val_transform(self, image, mask):
        """
        验证/测试阶段预处理。

        不做随机增强，仅Resize。
        """
        image = image.resize((self.base_size, self.base_size), Image.BILINEAR)  # 双线性插值会让图像更平滑。 反向映射到原图，找到原图中的邻居，根据距离计算权重，并得到插值
        mask = mask.resize((self.base_size, self.base_size), Image.NEAREST) # 最近邻插 保持0/255不变
        return image, mask

    @staticmethod
    def _image_to_array(image):
        """
        图像转NumPy并归一化，使三个通道的数据分布更一致，加快网络收敛。

        流程：
            RGB
             ↓
            ndarray(H,W,C)
             ↓
            /255
             ↓
            ImageNet均值方差归一化
             ↓
            CHW
        """
        array = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        array = (array - mean) / std
        return array.transpose(2, 0, 1).astype(np.float32)

    @staticmethod
    def _mask_to_array(mask):
        """
        标签转NumPy。

        流程：
            PIL(L)
              ↓
            ndarray(H,W)
              ↓
            /255
              ↓
            增加Channel维
              ↓
            (1,H,W)
        """
        array = np.asarray(mask, dtype=np.float32) / 255.0
        return array[None, :, :].astype(np.float32)

