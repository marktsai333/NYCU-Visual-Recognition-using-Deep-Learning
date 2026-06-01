"""Parse training logs and generate training curve figures for the report.

Usage:
    python plot_results.py
Outputs:
    training_curves.png  — loss + val PSNR for all runs
"""

import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(filepath):
    """Return (epochs, losses, psnrs) parsed from a train*.log file."""
    epochs, losses, psnrs = [], [], []
    if not os.path.isfile(filepath):
        print(f'  [warn] log not found: {filepath}')
        return epochs, losses, psnrs
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(
                r'Epoch\s+(\d+)/\d+\s*\|\s*loss\s*([\d.]+)\s*\|\s*val PSNR\s*([\d.]+)',
                line,
            )
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                psnrs.append(float(m.group(3)))
    return epochs, losses, psnrs


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    # Run 1: dim=48, 200 epochs
    r1_e, r1_l, r1_p = parse_log(os.path.join(base, 'checkpoints_run1', 'train_run1.log'))

    # Run 3: dim=64 — two log segments (crashed + resumed)
    r3a_e, r3a_l, r3a_p = parse_log(os.path.join(base, 'train3.log'))
    r3b_e, r3b_l, r3b_p = parse_log(os.path.join(base, 'train3b.log'))
    r3_e = r3a_e + r3b_e
    r3_l = r3a_l + r3b_l
    r3_p = r3a_p + r3b_p

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('PromptIR Training Curves', fontsize=14, fontweight='bold')

    # --- Loss ---
    ax = axes[0]
    if r1_e:
        ax.plot(r1_e, r1_l, color='steelblue', linewidth=1.5,
                label=f'dim=48  (30.1M params, public {30.48} dB)')
    if r3_e:
        ax.plot(r3_e, r3_l, color='tomato', linewidth=1.5,
                label=f'dim=64  (52.5M params, public {30.84} dB)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Charbonnier + FFT Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Val PSNR ---
    ax = axes[1]
    if r1_e:
        ax.plot(r1_e, r1_p, color='steelblue', linewidth=1.5,
                label=f'dim=48  (best val {max(r1_p):.2f} dB)')
    if r3_e:
        ax.plot(r3_e, r3_p, color='tomato', linewidth=1.5,
                label=f'dim=64  (best val {max(r3_p):.2f} dB)')
    ax.axhline(y=26, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Weak baseline')
    ax.axhline(y=30, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Strong baseline')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Val PSNR (dB)', fontsize=12)
    ax.set_title('Validation PSNR', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(base, 'training_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
