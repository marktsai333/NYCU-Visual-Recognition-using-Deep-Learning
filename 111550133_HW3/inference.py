"""Generate instance segmentation predictions for CodaBench submission.

Includes optional Test-Time Augmentation (TTA) with flip + scale ensemble.

Usage:
    python inference.py --checkpoint checkpoints/best.pth --tta
    python inference.py --checkpoint checkpoints/best.pth --tta --scale-tta
"""

import argparse
import json
import os

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils

from model import build_model


def load_test_image(path: str) -> np.ndarray:
    """Read a test TIFF and return uint8 HWC RGB array."""
    img = tifffile.imread(path)
    while img.ndim > 3:
        img = img[0]
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = img.transpose(1, 2, 0)
    if img.shape[-1] == 4:
        img = img[..., :3]
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / (hi - lo + 1e-8) * 255).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def encode_rle(binary_mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def predict_single(model, img_t, device):
    """Run model on one image tensor, return raw prediction dict."""
    with torch.no_grad():
        return model([img_t.to(device)])[0]


def _resize_img(img_t: torch.Tensor, scale: float) -> torch.Tensor:
    """Resize a CHW float tensor by scale factor."""
    _, h, w = img_t.shape
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    return F.interpolate(img_t.unsqueeze(0), size=(new_h, new_w),
                         mode="bilinear", align_corners=False).squeeze(0)


def merge_predictions(preds_list, orig_h, orig_w):
    """Merge predictions from TTA views back to original coordinate space.

    Each entry: (pred_dict, flip_h, flip_w, scale)
    """
    all_scores, all_labels, all_boxes, all_masks = [], [], [], []

    for pred, flip_h, flip_w, scale in preds_list:
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        boxes = pred["boxes"].cpu().numpy().copy()
        masks = pred["masks"].cpu().numpy()  # (N, 1, H_scaled, W_scaled)

        scaled_h = round(orig_h * scale)
        scaled_w = round(orig_w * scale)

        # Un-flip in scaled coordinate space
        if flip_h:
            boxes[:, [1, 3]] = scaled_h - boxes[:, [3, 1]]
            masks = masks[:, :, ::-1, :]
        if flip_w:
            boxes[:, [0, 2]] = scaled_w - boxes[:, [2, 0]]
            masks = masks[:, :, :, ::-1]

        # Un-scale boxes back to original coordinates
        if scale != 1.0:
            boxes[:, [0, 2]] /= scale
            boxes[:, [1, 3]] /= scale
            # Resize masks from (scaled_h, scaled_w) to (orig_h, orig_w)
            m_t = torch.from_numpy(masks.astype(np.float32))
            m_t = F.interpolate(m_t, size=(orig_h, orig_w), mode="nearest")
            masks = m_t.numpy()

        all_scores.append(scores)
        all_labels.append(labels)
        all_boxes.append(boxes)
        all_masks.append((masks > 0.5).astype(np.uint8))

    return (
        np.concatenate(all_scores),
        np.concatenate(all_labels),
        np.concatenate(all_boxes),
        np.concatenate(all_masks),
    )


def _mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """Compute IoU between two binary masks (shape: H×W)."""
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / float(union + 1e-8)


def nms_predictions(scores, labels, boxes, masks, iou_thresh=0.5, use_mask_iou=True):
    """Per-class NMS on merged TTA predictions using mask or box IoU."""
    keep_scores, keep_labels, keep_boxes, keep_masks = [], [], [], []

    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        cls_scores = scores[idx]
        cls_boxes = boxes[idx]
        cls_masks = masks[idx]  # (N, 1, H, W)

        order = np.argsort(-cls_scores)
        cls_scores = cls_scores[order]
        cls_boxes = cls_boxes[order]
        cls_masks = cls_masks[order]

        kept = []
        suppressed = np.zeros(len(cls_scores), dtype=bool)
        for i in range(len(cls_scores)):
            if suppressed[i]:
                continue
            kept.append(i)
            for j in range(i + 1, len(cls_scores)):
                if suppressed[j]:
                    continue
                if use_mask_iou:
                    iou = _mask_iou(cls_masks[i, 0], cls_masks[j, 0])
                else:
                    iou = _box_iou(cls_boxes[i], cls_boxes[j])
                if iou > iou_thresh:
                    suppressed[j] = True

        keep_scores.append(cls_scores[kept])
        keep_labels.append(np.full(len(kept), cls, dtype=np.int64))
        keep_boxes.append(cls_boxes[kept])
        keep_masks.append(cls_masks[kept])

    if not keep_scores:
        return scores, labels, boxes, masks

    return (
        np.concatenate(keep_scores),
        np.concatenate(keep_labels),
        np.concatenate(keep_boxes),
        np.concatenate(keep_masks),
    )


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def parse_args():
    p = argparse.ArgumentParser(description="Run inference and generate submission")
    p.add_argument("--test-dir", default="data/test")
    p.add_argument("--test-json", default="data/test_image_name_to_ids.json")
    p.add_argument("--checkpoint", default="checkpoints/best.pth")
    p.add_argument("--backbone", default="resnet101", choices=["resnet50", "resnet101"])
    p.add_argument("--output", default="test-results.json")
    p.add_argument("--score-thresh", type=float, default=0.3,
                   help="Minimum score threshold (lower = more recall, default 0.3)")
    p.add_argument("--tta", action="store_true",
                   help="Enable Test-Time Augmentation (H+V flip ensemble)")
    p.add_argument("--scale-tta", action="store_true",
                   help="Add 0.8x and 1.2x scale augmentation to TTA (requires --tta)")
    p.add_argument("--box-nms", action="store_true",
                   help="Use box IoU for TTA NMS instead of mask IoU (faster)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  TTA: {args.tta}  |  scale-TTA: {args.scale_tta}")
    print(f"score-thresh: {args.score_thresh}  |  NMS: {'box' if args.box_nms else 'mask'} IoU")

    model = build_model(backbone=args.backbone, trainable_layers=0)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    with open(args.test_json) as f:
        test_info = json.load(f)
    name_to_id = {item["file_name"]: item["id"] for item in test_info}

    results = []
    test_files = sorted(f for f in os.listdir(args.test_dir) if f.endswith(".tif"))

    for filename in test_files:
        if filename not in name_to_id:
            print(f"  [skip] {filename} not in test JSON")
            continue

        image_id = name_to_id[filename]
        img = load_test_image(os.path.join(args.test_dir, filename))
        orig_h, orig_w = img.shape[:2]

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if args.tta:
            scales = [1.0, 0.8] if args.scale_tta else [1.0]
            flip_combos = [(False, False), (False, True), (True, False), (True, True)]

            preds_list = []
            for scale in scales:
                base_t = _resize_img(img_t, scale) if scale != 1.0 else img_t
                for fh, fw in flip_combos:
                    view_t = base_t
                    if fh:
                        view_t = view_t.flip(-2)
                    if fw:
                        view_t = view_t.flip(-1)
                    pred = predict_single(model, view_t, device)
                    preds_list.append((pred, fh, fw, scale))
                    torch.cuda.empty_cache()

            scores, labels, boxes, masks = merge_predictions(preds_list, orig_h, orig_w)
            scores, labels, boxes, masks = nms_predictions(
                scores, labels, boxes, masks,
                use_mask_iou=not args.box_nms,
            )
        else:
            pred = predict_single(model, img_t, device)
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()
            boxes = pred["boxes"].cpu().numpy()
            masks = pred["masks"].cpu().numpy()

        count = 0
        for i, score in enumerate(scores):
            if score < args.score_thresh:
                continue
            binary = masks[i, 0] if masks.dtype == np.uint8 else (masks[i, 0] > 0.5).astype(np.uint8)
            b = boxes[i]
            results.append({
                "image_id": image_id,
                "bbox": [float(b[0]), float(b[1]),
                         float(b[2] - b[0]), float(b[3] - b[1])],
                "score": float(score),
                "category_id": int(labels[i]),
                "segmentation": encode_rle(binary),
            })
            count += 1

        print(f"{filename} (id={image_id}): {count} predictions")

    with open(args.output, "w") as f:
        json.dump(results, f)

    print(f"\nSaved {len(results)} predictions → {args.output}")
    print("投影片上寫 'test-reults.json'（typo），若上傳失敗請改名再試。")


if __name__ == "__main__":
    main()
