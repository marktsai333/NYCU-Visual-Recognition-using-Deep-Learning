"""Generate side-by-side visual comparison figures for the report.

Usage:
    python visualize.py \
        --data_dir E:/mtsai/HW4/hw4_realse_dataset \
        --checkpoint E:/mtsai/HW4/checkpoints3/best.pth \
        --dim 64 --num_refinement_blocks 6

Outputs:
    visual_comparison_rain.png
    visual_comparison_snow.png
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.promptir import PromptIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path):
    return transforms.ToTensor()(Image.open(path).convert('RGB'))


@torch.no_grad()
def restore(model, img_tensor, device):
    x = img_tensor.unsqueeze(0).to(device)
    _, _, h, w = x.shape
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    out = model(x).clamp(0.0, 1.0)
    return out[0, :, :h, :w].cpu()


def psnr(pred, gt):
    mse = np.mean((pred.astype(float) - gt.astype(float)) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0


def to_np(t):
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def make_figure(rows, save_path, title):
    """rows: list of (deg_np, rest_np, clean_np, psnr_val, label)"""
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 4.5 * len(rows)))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    if len(rows) == 1:
        axes = [axes]
    col_titles = ['Degraded Input', 'Restored (Ours)', 'Clean Ground Truth']
    for col, ct in enumerate(col_titles):
        axes[0][col].set_title(ct, fontsize=12, fontweight='bold', pad=8)
    for row_idx, (deg, rest, cln, psnr_val, label) in enumerate(rows):
        axes[row_idx][0].imshow(deg)
        axes[row_idx][0].set_ylabel(label, fontsize=10, rotation=90, labelpad=6)
        axes[row_idx][1].imshow(rest)
        axes[row_idx][1].set_xlabel(f'PSNR = {psnr_val:.2f} dB', fontsize=10)
        axes[row_idx][2].imshow(cln)
        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {save_path}')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PromptIR(
        inp_channels=3, out_channels=3,
        dim=args.dim, num_blocks=args.num_blocks,
        num_refinement_blocks=args.num_refinement_blocks,
        heads=args.heads, ffn_expansion_factor=2.66,
        bias=False, layer_norm_type='WithBias',
        prompt_len=5, prompt_size=32,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')

    deg_dir = os.path.join(args.data_dir, 'train', 'degraded')
    cln_dir = os.path.join(args.data_dir, 'train', 'clean')
    base = os.path.dirname(os.path.abspath(__file__))

    # Pick 3 rain samples and 3 snow samples (from first 160 = val split)
    rain_ids = [1, 40, 100]
    snow_ids = [1, 40, 100]

    for dtype, ids, clean_prefix in [
        ('rain', rain_ids, 'rain_clean'),
        ('snow', snow_ids, 'snow_clean'),
    ]:
        rows = []
        for i in ids:
            deg_path = os.path.join(deg_dir, f'{dtype}-{i}.png')
            cln_path = os.path.join(cln_dir, f'{clean_prefix}-{i}.png')
            if not (os.path.isfile(deg_path) and os.path.isfile(cln_path)):
                print(f'  [skip] {deg_path}')
                continue
            deg_t = load_image(deg_path)
            cln_t = load_image(cln_path)
            rest_t = restore(model, deg_t, device)
            deg_np = to_np(deg_t)
            cln_np = to_np(cln_t)
            rest_np = to_np(rest_t)
            p = psnr(rest_np, cln_np)
            rows.append((deg_np, rest_np, cln_np, p, f'{dtype}-{i}'))

        if rows:
            save_path = os.path.join(base, f'visual_comparison_{dtype}.png')
            make_figure(rows, save_path, f'Visual Comparison — {dtype.capitalize()} Removal')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dim', type=int, default=48)
    parser.add_argument('--num_blocks', type=int, nargs=4, default=[4, 6, 6, 8])
    parser.add_argument('--num_refinement_blocks', type=int, default=4)
    parser.add_argument('--heads', type=int, nargs=4, default=[1, 2, 4, 8])
    args = parser.parse_args()
    main(args)
