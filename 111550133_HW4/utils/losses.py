"""Loss functions for image restoration."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier loss: sqrt((x - y)^2 + eps^2).

    Smoother than L1 at zero, more robust than L2 to outliers.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class FFTLoss(nn.Module):
    """Frequency-domain L1 loss on the amplitude spectrum.

    Penalises high-frequency errors that pixel-space losses may underweight.
    """

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        return F.l1_loss(pred_amp, target_amp)


class CombinedLoss(nn.Module):
    """Charbonnier + weighted FFT loss."""

    def __init__(self, fft_weight=0.1, eps=1e-3):
        super().__init__()
        self.charb = CharbonnierLoss(eps)
        self.fft = FFTLoss()
        self.fft_weight = fft_weight

    def forward(self, pred, target):
        return self.charb(pred, target) + self.fft_weight * self.fft(pred, target)
