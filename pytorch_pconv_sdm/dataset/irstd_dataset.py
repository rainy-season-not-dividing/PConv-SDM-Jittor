import os.path as osp
import random

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps


class IRSTDDataset(data.Dataset):
    """IRSTD-1K style dataset.

    Expected directory layout:

    ```text
    dataset_dir/
      trainval.txt
      test.txt
      images/*.png
      masks/*.png
    ```
    """

    def __init__(self, dataset_dir, mode="train", base_size=256, crop_size=256):
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

        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.mask_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = Image.open(osp.join(self.images_dir, f"{name}.png")).convert("RGB")
        mask = Image.open(osp.join(self.masks_dir, f"{name}.png")).convert("L")

        if self.mode == "train":
            image, mask = self._sync_transform(image, mask)
        else:
            image, mask = self._val_transform(image, mask)

        return self.image_transform(image), self.mask_transform(mask)

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
