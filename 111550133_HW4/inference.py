"""Inference script — generates pred.npz for CodaBench submission.

Usage:
    python inference.py --data_dir /path/to/dataset --checkpoint checkpoints/best.pth

Options:
    --tta          Enable 8× test-time augmentation (flips + rotations) for ~+0.1 dB
    --output       Output file path (default: pred.npz)
    --dim / --num_blocks / --heads  must match the training configuration
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from datasets.dataset import TestDataset
from models.promptir import PromptIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pad_to_multiple(x, multiple=8):
    """Reflect-pad x so height and width are multiples of `multiple`."""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, h, w


@torch.no_grad()
def restore_single(model, img_tensor, device, use_tta=False):
    """Restore one image tensor (C, H, W) in [0, 1]."""
    x = img_tensor.unsqueeze(0).to(device)
    x, orig_h, orig_w = pad_to_multiple(x, multiple=8)

    if not use_tta:
        out = model(x).clamp(0.0, 1.0)
        return out[0, :, :orig_h, :orig_w]

    # 8-fold TTA: all combinations of (flip_h, flip_v, rot90)
    outputs = []
    for flip_h in (False, True):
        for flip_v in (False, True):
            for do_rot in (False, True):
                t = x.clone()
                if flip_h:
                    t = torch.flip(t, dims=[-1])
                if flip_v:
                    t = torch.flip(t, dims=[-2])
                if do_rot:
                    t = torch.rot90(t, k=1, dims=[-2, -1])

                o = model(t).clamp(0.0, 1.0)

                if do_rot:
                    o = torch.rot90(o, k=-1, dims=[-2, -1])
                if flip_v:
                    o = torch.flip(o, dims=[-2])
                if flip_h:
                    o = torch.flip(o, dims=[-1])
                outputs.append(o)

    out = torch.stack(outputs).mean(0)
    return out[0, :, :orig_h, :orig_w]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_model(args, device):
    model = PromptIR(
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
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')
    return model


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  TTA: {args.tta}')

    model = build_model(args, device)

    dataset = TestDataset(args.data_dir)
    print(f'Test images: {len(dataset)}')

    results = {}
    for img_tensor, filename in tqdm(dataset, desc='Restoring'):
        restored = restore_single(model, img_tensor, device, use_tta=args.tta)
        # Convert to uint8 NumPy array with shape (3, H, W)
        arr = (restored.cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        results[filename] = arr

    np.savez(args.output, **results)
    print(f'Saved {len(results)} images to {args.output}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate pred.npz for CodaBench submission')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing test/ folder')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (best.pth)')
    parser.add_argument('--output', type=str, default='pred.npz',
                        help='Output file name (must be pred.npz inside the submission zip)')
    parser.add_argument('--tta', action='store_true',
                        help='Enable 8× test-time augmentation')

    # Model architecture (must match training)
    parser.add_argument('--dim', type=int, default=48)
    parser.add_argument('--num_blocks', type=int, nargs=4, default=[4, 6, 6, 8],
                        metavar=('L1', 'L2', 'L3', 'LAT'))
    parser.add_argument('--num_refinement_blocks', type=int, default=4)
    parser.add_argument('--heads', type=int, nargs=4, default=[1, 2, 4, 8],
                        metavar=('H1', 'H2', 'H3', 'H4'))

    args = parser.parse_args()
    main(args)
