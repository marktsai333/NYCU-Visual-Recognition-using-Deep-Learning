"""Cell instance segmentation dataset for HW3."""

import os

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

CELL_CLASSES = ["class1", "class2", "class3", "class4"]


class CellDataset(Dataset):
    """Load cell microscopy TIF images with per-class instance masks."""

    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_dirs = sorted(
            d
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        img_dir = os.path.join(self.root_dir, self.image_dirs[idx])
        image = _load_tif_as_rgb(os.path.join(img_dir, "image.tif"))
        h, w = image.shape[:2]

        boxes, labels, masks = _load_instance_masks(img_dir, h, w)

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        n = len(boxes)
        if n == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, h, w), dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
            target = {
                "boxes": boxes_t,
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.from_numpy(np.stack(masks)).to(torch.uint8),
                "image_id": torch.tensor([idx]),
                "area": area,
                "iscrowd": torch.zeros((n,), dtype=torch.int64),
            }

        if self.transforms is not None:
            image_t, target = self.transforms(image_t, target)

        return image_t, target


def _load_tif_as_rgb(path):
    """Read a TIFF file and return a uint8 HWC RGB numpy array."""
    img = tifffile.imread(path)

    # Collapse extra leading dims (e.g. time-series TIFF)
    while img.ndim > 3:
        img = img[0]

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[0] in (1, 3, 4):
        # CHW → HWC
        img = img.transpose(1, 2, 0)

    if img.shape[-1] == 4:
        img = img[..., :3]
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / (hi - lo + 1e-8) * 255).clip(0, 255).astype(np.uint8)

    return np.ascontiguousarray(img)


def _load_instance_masks(img_dir, h, w):
    """Return boxes [x1,y1,x2,y2], 1-indexed labels, and binary masks."""
    boxes, labels, masks = [], [], []
    for cls_idx, cls_name in enumerate(CELL_CLASSES):
        path = os.path.join(img_dir, f"{cls_name}.tif")
        if not os.path.exists(path):
            continue

        cls_mask = tifffile.imread(path)
        # Squeeze to 2D in case of extra dims
        while cls_mask.ndim > 2:
            cls_mask = cls_mask[0]

        for inst_id in np.unique(cls_mask):
            if inst_id == 0:
                continue
            binary = (cls_mask == inst_id).astype(np.uint8)
            ys, xs = np.where(binary)
            if ys.size == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_idx + 1)  # 1-indexed; 0 = background
            masks.append(binary)

    return boxes, labels, masks
