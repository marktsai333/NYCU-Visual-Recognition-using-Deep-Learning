"""Data augmentation transforms compatible with Mask R-CNN targets."""

import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target
        _, _, w = image.shape
        image = TF.hflip(image)
        if len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
        if len(target["masks"]) > 0:
            target["masks"] = target["masks"].flip(-1)
        return image, target


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target
        _, h, _ = image.shape
        image = TF.vflip(image)
        if len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
            target["boxes"] = boxes
        if len(target["masks"]) > 0:
            target["masks"] = target["masks"].flip(-2)
        return image, target


class RandomRotation90:
    """Rotate by 0 / 90 / 180 / 270 degrees with equal probability."""

    def __call__(self, image, target):
        k = random.randint(0, 3)
        if k == 0:
            return image, target

        _, h, w = image.shape
        image = torch.rot90(image, k, dims=[-2, -1])

        if len(target["masks"]) > 0:
            target["masks"] = torch.rot90(target["masks"], k, dims=[-2, -1])

        if len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            if k == 1:  # 90° CCW → new size (w, h)
                new_boxes = torch.stack([y1, w - x2, y2, w - x1], dim=1)
            elif k == 2:  # 180° → same size (h, w)
                new_boxes = torch.stack([w - x2, h - y2, w - x1, h - y1], dim=1)
            else:  # 270° CCW → new size (w, h)
                new_boxes = torch.stack([h - y2, x1, h - y1, x2], dim=1)

            new_h, new_w = image.shape[-2], image.shape[-1]
            new_boxes[:, 0].clamp_(0, new_w)
            new_boxes[:, 1].clamp_(0, new_h)
            new_boxes[:, 2].clamp_(0, new_w)
            new_boxes[:, 3].clamp_(0, new_h)
            target["boxes"] = new_boxes
            target["area"] = (new_boxes[:, 2] - new_boxes[:, 0]) * (
                new_boxes[:, 3] - new_boxes[:, 1]
            )

        return image, target


class RandomScale:
    """Scale jitter: resize image and targets by a random factor."""

    def __init__(self, scale_range=(0.7, 1.1)):
        self.scale_range = scale_range

    def __call__(self, image, target):
        scale = random.uniform(*self.scale_range)
        _, h, w = image.shape
        new_h = max(1, round(h * scale))
        new_w = max(1, round(w * scale))
        image = TF.resize(image, [new_h, new_w], antialias=True)

        if len(target["masks"]) > 0:
            m = target["masks"].float().unsqueeze(1)
            m = F.interpolate(m, size=(new_h, new_w), mode="nearest")
            target["masks"] = m.squeeze(1).to(torch.uint8)

        if len(target["boxes"]) > 0:
            sx, sy = new_w / w, new_h / h
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            boxes[:, [0, 2]].clamp_(0, new_w)
            boxes[:, [1, 3]].clamp_(0, new_h)
            target["boxes"] = boxes
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return image, target


class RandomGaussianBlur:
    """Random Gaussian blur (image only, targets unchanged)."""

    def __init__(self, prob=0.3, kernel_size=5, sigma=(0.1, 2.0)):
        self.prob = prob
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image, target):
        if random.random() < self.prob:
            sigma = random.uniform(*self.sigma)
            image = TF.gaussian_blur(image, self.kernel_size, sigma)
        return image, target


class GaussianNoise:
    """Additive Gaussian noise for robustness to imaging artifacts."""

    def __init__(self, prob=0.3, std=0.04):
        self.prob = prob
        self.std = std

    def __call__(self, image, target):
        if random.random() < self.prob:
            noise = torch.randn_like(image) * self.std
            image = (image + noise).clamp(0.0, 1.0)
        return image, target


class ColorJitter:
    """Apply random brightness, contrast, and saturation jitter."""

    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, image, target):
        if self.brightness > 0 and random.random() < 0.5:
            f = random.uniform(max(0.0, 1 - self.brightness), 1 + self.brightness)
            image = TF.adjust_brightness(image, f)
        if self.contrast > 0 and random.random() < 0.5:
            f = random.uniform(max(0.0, 1 - self.contrast), 1 + self.contrast)
            image = TF.adjust_contrast(image, f)
        if self.saturation > 0 and random.random() < 0.5:
            f = random.uniform(max(0.0, 1 - self.saturation), 1 + self.saturation)
            image = TF.adjust_saturation(image, f)
        return image, target
