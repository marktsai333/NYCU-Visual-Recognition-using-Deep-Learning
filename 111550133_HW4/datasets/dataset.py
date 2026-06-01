"""Dataset classes for HW4 image restoration (Rain / Snow)."""

import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RestorationDataset(Dataset):
    """
    Loads (degraded, clean) pairs from the HW4 dataset layout:

        <data_dir>/train/degraded/rain-1.png  ... rain-1600.png
                                  snow-1.png  ... snow-1600.png
        <data_dir>/train/clean/rain_clean-1.png  ...
                               snow_clean-1.png  ...

    Automatically splits into train / val via `val_ratio`.
    """

    def __init__(
        self,
        data_dir,
        split='train',
        patch_size=128,
        augment=True,
        val_ratio=0.1,
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

        deg_dir = os.path.join(data_dir, 'train', 'degraded')
        clean_dir = os.path.join(data_dir, 'train', 'clean')

        pairs = []
        for dtype in ('rain', 'snow'):
            for i in range(1, 1601):
                deg = os.path.join(deg_dir, f'{dtype}-{i}.png')
                cln = os.path.join(clean_dir, f'{dtype}_clean-{i}.png')
                if os.path.isfile(deg) and os.path.isfile(cln):
                    pairs.append((deg, cln))

        n_val = max(1, int(len(pairs) * val_ratio))
        if split == 'train':
            self.pairs = pairs[n_val:]
        else:
            self.pairs = pairs[:n_val]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        deg_path, cln_path = self.pairs[idx]
        deg = Image.open(deg_path).convert('RGB')
        cln = Image.open(cln_path).convert('RGB')

        deg, cln = self._random_crop(deg, cln)

        if self.augment:
            deg, cln = self._augment(deg, cln)

        return self.to_tensor(deg), self.to_tensor(cln)

    def _random_crop(self, deg, cln):
        w, h = deg.size
        ps = self.patch_size
        if w < ps or h < ps:
            # Pad if image is smaller than patch size
            pad_w = max(0, ps - w)
            pad_h = max(0, ps - h)
            deg = transforms.functional.pad(deg, (0, 0, pad_w, pad_h), padding_mode='reflect')
            cln = transforms.functional.pad(cln, (0, 0, pad_w, pad_h), padding_mode='reflect')
            w, h = deg.size
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        return deg.crop((x, y, x + ps, y + ps)), cln.crop((x, y, x + ps, y + ps))

    @staticmethod
    def _augment(deg, cln):
        if random.random() > 0.5:
            deg = deg.transpose(Image.FLIP_LEFT_RIGHT)
            cln = cln.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            deg = deg.transpose(Image.FLIP_TOP_BOTTOM)
            cln = cln.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            deg = deg.rotate(angle)
            cln = cln.rotate(angle)
        return deg, cln


class TestDataset(Dataset):
    """
    Loads test images from:
        <data_dir>/test/degraded/0.png ... 99.png
    """

    def __init__(self, data_dir):
        self.to_tensor = transforms.ToTensor()
        test_dir = os.path.join(data_dir, 'test', 'degraded')
        self.files = sorted(
            [f for f in os.listdir(test_dir) if f.endswith('.png')],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        self.paths = [os.path.join(test_dir, f) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.to_tensor(img), self.files[idx]
