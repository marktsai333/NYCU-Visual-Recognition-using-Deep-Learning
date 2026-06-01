"""Training script for PromptIR image restoration (HW4).

Usage:
    python train.py --data_dir /path/to/dataset

Optional flags:
    --epochs 200 --batch_size 8 --patch_size 128 --lr 2e-4
    --dim 48 --loss combined   # 'charbonnier' or 'combined' (adds FFT loss)
    --resume checkpoints/last.pth
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import RestorationDataset
from models.promptir import PromptIR
from utils.losses import CharbonnierLoss, CombinedLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_psnr(pred, target):
    """PSNR on 0-255 float tensors."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20.0 * torch.log10(torch.tensor(255.0) / torch.sqrt(mse))


def save_checkpoint(state, path):
    torch.save(state, path)
    print(f'  Checkpoint saved -> {path}')


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for degraded, clean in tqdm(loader, desc='  train', leave=False):
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            restored = model(degraded)
            loss = criterion(restored, clean)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    torch.cuda.empty_cache()
    total_psnr = 0.0
    count = 0
    for degraded, clean in tqdm(loader, desc='  val  ', leave=False):
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with autocast():
            restored = model(degraded).clamp(0.0, 1.0)

        for i in range(restored.shape[0]):
            psnr = compute_psnr(restored[i] * 255.0, clean[i] * 255.0)
            total_psnr += psnr.item()
            count += 1

    return total_psnr / max(count, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_model(args):
    return PromptIR(
        inp_channels=3,
        out_channels=3,
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_refinement_blocks=args.num_refinement_blocks,
        heads=args.heads,
        ffn_expansion_factor=2.66,
        bias=False,
        layer_norm_type='WithBias',
        prompt_len=5,
        prompt_size=32,
    )


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- Datasets & loaders ---
    train_ds = RestorationDataset(
        args.data_dir, split='train',
        patch_size=args.patch_size, augment=True, val_ratio=args.val_ratio,
    )
    val_ds = RestorationDataset(
        args.data_dir, split='val',
        patch_size=args.patch_size, augment=False, val_ratio=args.val_ratio,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f'Train: {len(train_ds)} pairs | Val: {len(val_ds)} pairs')

    # --- Model ---
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Parameters: {n_params:.2f} M')

    # --- Loss ---
    if args.loss == 'combined':
        criterion = CombinedLoss(fft_weight=0.1)
    else:
        criterion = CharbonnierLoss()

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    scaler = GradScaler()

    # --- Resume ---
    start_epoch = 1
    best_psnr = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if args.finetune:
            # Only load weights; reset optimizer/scheduler with new lr and epochs
            print(f'Finetune mode: loaded weights only (lr={args.lr}, epochs={args.epochs})')
        else:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_psnr = ckpt.get('best_psnr', 0.0)
            print(f'Resumed from epoch {ckpt["epoch"]} (best PSNR {best_psnr:.2f})')

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_psnr = validate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        print(
            f'Epoch {epoch:3d}/{args.epochs} | '
            f'loss {train_loss:.4f} | val PSNR {val_psnr:.2f} dB | '
            f'lr {lr:.2e} | {elapsed:.0f}s'
        )

        # Save last checkpoint every epoch (small overhead, allows resume)
        save_checkpoint(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_psnr': best_psnr,
            },
            os.path.join(args.save_dir, 'last.pth'),
        )

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'psnr': best_psnr},
                os.path.join(args.save_dir, 'best.pth'),
            )
            print(f'  -> New best: {best_psnr:.2f} dB')

        # Periodic checkpoint
        if epoch % args.save_interval == 0:
            save_checkpoint(
                {'epoch': epoch, 'model_state_dict': model.state_dict()},
                os.path.join(args.save_dir, f'epoch_{epoch:04d}.pth'),
            )

    print(f'Training finished. Best val PSNR: {best_psnr:.2f} dB')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PromptIR for image restoration')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing train/ and test/ folders')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--finetune', action='store_true',
                        help='Load weights only (reset optimizer/scheduler); use with --resume for fine-tuning')

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of data used for validation')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=20,
                        help='Save a named checkpoint every N epochs')

    # Loss
    parser.add_argument('--loss', type=str, default='charbonnier',
                        choices=['charbonnier', 'combined'],
                        help='Loss function: charbonnier or combined (+ FFT)')

    # Model
    parser.add_argument('--dim', type=int, default=48)
    parser.add_argument('--num_blocks', type=int, nargs=4, default=[4, 6, 6, 8],
                        metavar=('L1', 'L2', 'L3', 'LAT'))
    parser.add_argument('--num_refinement_blocks', type=int, default=4)
    parser.add_argument('--heads', type=int, nargs=4, default=[1, 2, 4, 8],
                        metavar=('H1', 'H2', 'H3', 'H4'))

    args = parser.parse_args()
    main(args)
