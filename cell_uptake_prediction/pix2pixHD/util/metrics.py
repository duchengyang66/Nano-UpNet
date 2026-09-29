"""
Image quality metrics: PSNR, SSIM.

Both are computed in [0, 1] float space on tensors of shape (B, C, H, W).
For our task, the red channel of fluorescence image contains nearly all
the signal; we still compute metrics on all 3 channels (consistent with
the generator's 3-channel output).
"""

import math
import torch
import torch.nn.functional as F


def psnr(pred, target, max_val=1.0):
    """
    Mean PSNR over a batch.
    pred, target: tensors in [0, 1], shape (B, C, H, W) or (C, H, W).
    max_val: maximum possible value (1.0 for normalized [0,1] tensors).
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    mse = F.mse_loss(pred, target, reduction='none')
    # mean over C, H, W per image
    mse_per_img = mse.flatten(1).mean(dim=1)
    psnr_per_img = 10.0 * torch.log10((max_val ** 2) / (mse_per_img + 1e-12))
    val = psnr_per_img.mean().item()

    if squeeze:
        return psnr_per_img.item()
    return val


def _gaussian_window(window_size, sigma):
    """Generate a 1D Gaussian kernel (for SSIM)."""
    g = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
                      for x in range(window_size)])
    return g / g.sum()


def _create_window(window_size, channel):
    """Create a 2D Gaussian window for SSIM."""
    _1d = _gaussian_window(window_size, 1.5).unsqueeze(1)
    _2d = _1d @ _1d.t()  # outer product -> (window_size, window_size)
    window = _2d.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)
    return window.expand(channel, 1, window_size, window_size).contiguous()


def ssim(pred, target, window_size=11, max_val=1.0):
    """
    Mean SSIM over a batch.
    pred, target: tensors in [0, 1], shape (B, C, H, W) or (C, H, W).
    Returns a value in [-1, 1]; higher is better.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, C, H, W = pred.shape
    window = _create_window(window_size, C).to(pred.device).to(pred.dtype)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu1_mu2

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    val = ssim_map.mean().item()
    if squeeze:
        return val
    return val

