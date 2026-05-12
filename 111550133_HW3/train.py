"""Training script for HW3 cell instance segmentation.

Usage:
    python train.py --data-dir data/train --epochs 50 --batch-size 2
"""

import argparse
import csv
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import transforms as T
from dataset import CellDataset
from model import build_model, count_parameters


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transforms():
    return T.Compose([
        T.RandomHorizontalFlip(prob=0.5),
        T.RandomVerticalFlip(prob=0.5),
        T.RandomRotation90(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.RandomGaussianBlur(prob=0.3),
        T.GaussianNoise(prob=0.3, std=0.04),
    ])


def get_val_transforms():
    return T.Compose([])


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, loader, device, scaler, grad_clip=5.0):
    model.train()
    total = 0.0
    n = len(loader)
    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        try:
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            print(f"\r  [OOM skip] batch {i+1}", flush=True)
            continue

        total += loss.item()
        avg = total / (i + 1)
        done = int(30 * (i + 1) / n)
        bar = "#" * done + "-" * (30 - done)
        print(f"\r  [{bar}] {i+1}/{n}  loss={avg:.4f}", end="", flush=True)

    print()
    return total / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Validation (COCO AP50)
# ---------------------------------------------------------------------------

def evaluate_ap50(model, loader, device):
    """Compute segmentation AP@0.5 using pycocotools."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import pycocotools.mask as mask_utils

    model.eval()

    gt_dict = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": f"class{i}"} for i in range(1, 5)],
    }
    results = []
    ann_id = 0

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for images, targets in loader:
            images_dev = [img.to(device) for img in images]
            outputs = model(images_dev)

            for img, target, output in zip(images, targets, outputs):
                img_id = int(target["image_id"].item())
                h, w = img.shape[-2], img.shape[-1]
                gt_dict["images"].append({"id": img_id, "height": h, "width": w})

                # Ground-truth annotations
                for j in range(len(target["boxes"])):
                    b = target["boxes"][j].cpu().numpy()
                    m = target["masks"][j].cpu().numpy().astype(np.uint8)
                    rle = mask_utils.encode(np.asfortranarray(m))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    ann_id += 1
                    gt_dict["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(target["labels"][j].item()),
                        "segmentation": rle,
                        "area": float((b[2] - b[0]) * (b[3] - b[1])),
                        "bbox": [float(b[0]), float(b[1]),
                                 float(b[2] - b[0]), float(b[3] - b[1])],
                        "iscrowd": 0,
                    })

                # Predictions
                scores = output["scores"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                boxes = output["boxes"].cpu().numpy()
                masks = output["masks"].cpu().numpy()

                for j in range(len(scores)):
                    binary = (masks[j, 0] > 0.5).astype(np.uint8)
                    rle = mask_utils.encode(np.asfortranarray(binary))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    b = boxes[j]
                    results.append({
                        "image_id": img_id,
                        "category_id": int(pred_labels[j]),
                        "segmentation": rle,
                        "score": float(scores[j]),
                        "bbox": [float(b[0]), float(b[1]),
                                 float(b[2] - b[0]), float(b[3] - b[1])],
                    })

    if not gt_dict["annotations"] or not results:
        return 0.0

    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.params.iouThrs = np.array([0.5])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Mask R-CNN for cell segmentation")
    p.add_argument("--data-dir", default="data/train",
                   help="Path to training folder (contains per-image sub-dirs)")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="Fraction of training images used for validation")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-every", type=int, default=5,
                   help="Run COCO AP50 validation every N epochs")
    p.add_argument("--trainable-backbone-layers", type=int, default=4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--backbone", default="resnet101",
                   choices=["resnet50", "resnet101"],
                   help="Backbone architecture (resnet101 recommended for higher AP)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from (e.g. checkpoints/best.pth)")
    p.add_argument("--start-epoch", type=int, default=1,
                   help="Epoch to start from when resuming")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset split ──────────────────────────────────────────────────────
    all_dirs = sorted(
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    )
    n_val = max(1, int(len(all_dirs) * args.val_ratio))
    n_train = len(all_dirs) - n_val
    print(f"Train: {n_train}  Val: {n_val}")

    # Two separate Dataset instances so each can have different transforms
    train_ds = CellDataset(args.data_dir, transforms=get_train_transforms())
    val_ds = CellDataset(args.data_dir, transforms=get_val_transforms())

    train_loader = DataLoader(
        Subset(train_ds, list(range(n_train))),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda" and args.num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(val_ds, list(range(n_train, len(all_dirs)))),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(
        backbone=args.backbone,
        trainable_layers=args.trainable_backbone_layers,
    )
    model.to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state)
        print(f"Resumed from {args.resume} (start epoch {args.start_epoch})")
    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params / 1e6:.1f}M")

    # ── Optimizer & scheduler ──────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - args.warmup_epochs, 1),
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs]
    )

    scaler = torch.amp.GradScaler("cuda")

    # ── CSV log ────────────────────────────────────────────────────────────
    log_path = os.path.join(args.output_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "lr", "val_ap50"])

    # ── Training loop ──────────────────────────────────────────────────────
    best_ap50 = 0.0

    for epoch in range(args.start_epoch, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
        scheduler.step()
        elapsed = time.time() - t0

        current_lr = scheduler.get_last_lr()[0]
        ap50 = None

        if epoch % args.val_every == 0:
            torch.cuda.empty_cache()
            ap50 = evaluate_ap50(model, val_loader, device)
            if ap50 > best_ap50:
                best_ap50 = ap50
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, "best.pth"),
                )

        # Log to CSV
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{loss:.4f}", f"{current_lr:.2e}",
                                    f"{ap50:.4f}" if ap50 is not None else ""])

        # Console output
        line = (f"Epoch {epoch:3d}/{args.epochs} | "
                f"loss={loss:.4f} | lr={current_lr:.2e} | {elapsed:.1f}s")
        if ap50 is not None:
            line += f" | val AP50={ap50:.4f}"
            if ap50 >= best_ap50:
                line += " [best saved]"
        print(line)

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, f"epoch_{epoch:03d}.pth"),
            )

    torch.save(model.state_dict(), os.path.join(args.output_dir, "last.pth"))
    print(f"\nTraining complete. Best val AP50: {best_ap50:.4f}")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
