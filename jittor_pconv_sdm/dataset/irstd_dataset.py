import os.path as osp
import random

import numpy as np
from jittor.dataset import Dataset
from PIL import Image, ImageFilter, ImageOps


class IRSTDDataset(Dataset):
    """IRSTD-1K style dataset for Jittor.

    Expected layout:

    ```text
    dataset_dir/
      trainval.txt
      test.txt
      images/*.png
      masks/*.png
    ```
    """

    def __init__(self, dataset_dir, mode="train", base_size=256, crop_size=256, **dataset_attrs):
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
        name = self.names[index]
        image = Image.open(osp.join(self.images_dir, f"{name}.png")).convert("RGB")
        mask = Image.open(osp.join(self.masks_dir, f"{name}.png")).convert("L")

        if self.mode == "train":
            image, mask = self._sync_transform(image, mask)
        else:
            image, mask = self._val_transform(image, mask)

        return self._image_to_array(image), self._mask_to_array(mask)

    def _sync_transform(self, image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

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

        image = image.resize((out_w, out_h), Image.BILINEAR)
        mask = mask.resize((out_w, out_h), Image.NEAREST)

        if short_size < self.crop_size:
            pad_h = self.crop_size - out_h if out_h < self.crop_size else 0
            pad_w = self.crop_size - out_w if out_w < self.crop_size else 0
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=0)

        width, height = image.size
        x1 = random.randint(0, width - self.crop_size)
        y1 = random.randint(0, height - self.crop_size)
        image = image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return image, mask

    def _val_transform(self, image, mask):
        image = image.resize((self.base_size, self.base_size), Image.BILINEAR)
        mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
        return image, mask

    @staticmethod
    def _image_to_array(image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        array = (array - mean) / std
        return array.transpose(2, 0, 1).astype(np.float32)

    @staticmethod
    def _mask_to_array(mask):
        array = np.asarray(mask, dtype=np.float32) / 255.0
        return array[None, :, :].astype(np.float32)

