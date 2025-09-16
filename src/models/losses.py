import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _gaussian_kernel1d(kernel_size: int = 5, sigma: float = 1.0, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _gaussian_blur(img: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Separable Gaussian blur for 4D tensor [B, C, H, W].
    """
    b, c, h, w = img.shape
    dtype = img.dtype
    device = img.device
    k1d = _gaussian_kernel1d(kernel_size, sigma, device=device, dtype=dtype)
    kernel_x = k1d.view(1, 1, 1, kernel_size)
    kernel_y = k1d.view(1, 1, kernel_size, 1)
    # Apply per-channel depthwise convolution
    img = F.conv2d(img, kernel_x.expand(c, 1, 1, kernel_size), padding=(0, kernel_size // 2), groups=c)
    img = F.conv2d(img, kernel_y.expand(c, 1, kernel_size, 1), padding=(kernel_size // 2, 0), groups=c)
    return img


def _downsample(img: torch.Tensor) -> torch.Tensor:
    return F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=False)


def _upsample(img: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(img, size=size, mode='bilinear', align_corners=False)


def _downsample_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Downsample a boolean or float mask to half resolution using average pooling, then threshold to boolean.
    Input shape: [B, 1, H, W] or [B, C, H, W]. Returns boolean mask [B, 1, h, w].
    """
    if mask.dtype != torch.float32 and mask.dtype != torch.float64:
        mask_f = mask.float()
    else:
        mask_f = mask
    # If mask has multiple channels, reduce to single-channel validity via min/all using product
    if mask_f.shape[1] > 1:
        mask_f = (mask_f > 0.5).all(dim=1, keepdim=True).float()
    pooled = F.avg_pool2d(mask_f, kernel_size=2, stride=2, ceil_mode=False)
    return (pooled > 0.5)


class LaplacianPyramidLoss(nn.Module):
    """
    Multi-scale L1 loss computed on a Laplacian pyramid.

    - Builds a Gaussian pyramid, then forms Laplacian bands L_i = G_i - upsample(G_{i+1}).
    - Computes L1 between corresponding Laplacian bands (and optionally the coarsest Gaussian) across levels.
    - If a validity mask is provided, it is downsampled per level and used to compute masked L1.

    Args:
        levels: Number of pyramid levels (>=1). If levels=1, this reduces to standard L1 on the image.
        kernel_size: Gaussian kernel size.
        sigma: Gaussian sigma.
        include_lowpass: Whether to include the coarsest Gaussian level in the loss.
        level_weights: Optional tensor/list of per-level weights (length = levels for bands; +1 if include_lowpass).
    """
    def __init__(self, levels: int = 3, kernel_size: int = 5, sigma: float = 1.0,
                 include_lowpass: bool = True, level_weights: Optional[torch.Tensor] = None):
        super().__init__()
        assert levels >= 1
        self.levels = levels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.include_lowpass = include_lowpass
        self.register_buffer('level_weights', None, persistent=False)
        if level_weights is not None:
            lw = torch.as_tensor(level_weights, dtype=torch.float32)
            self.register_buffer('level_weights', lw, persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert pred.shape == target.shape, "pred and target must have same shape [B, C, H, W]"
        b, c, h, w = pred.shape
        device = pred.device

        # Build Gaussian pyramids
        Gp = [pred]
        Gt = [target]
        for _ in range(self.levels - 1):
            Gp.append(_downsample(_gaussian_blur(Gp[-1], self.kernel_size, self.sigma)))
            Gt.append(_downsample(_gaussian_blur(Gt[-1], self.kernel_size, self.sigma)))

        # Build Laplacian bands
        Lp = []
        Lt = []
        for i in range(self.levels - 1):
            up = _upsample(Gp[i + 1], size=(Gp[i].shape[-2], Gp[i].shape[-1]))
            ut = _upsample(Gt[i + 1], size=(Gt[i].shape[-2], Gt[i].shape[-1]))
            Lp.append(Gp[i] - up)
            Lt.append(Gt[i] - ut)
        # Coarsest level (Gaussian)
        if self.include_lowpass:
            Lp.append(Gp[-1])
            Lt.append(Gt[-1])

        # Prepare per-level masks
        masks = []
        if mask is not None:
            # ensure shape [B, 1, H, W]
            if mask.dim() == 4 and mask.shape[1] != 1:
                mask_lvl = (mask > 0.5).all(dim=1, keepdim=True)
            elif mask.dim() == 4:
                mask_lvl = mask > 0.5
            else:
                # assume [B, H, W]
                mask_lvl = mask.unsqueeze(1) > 0.5
            masks.append(mask_lvl)
            for i in range(self.levels - 1):
                mask_lvl = _downsample_mask(mask_lvl)
                masks.append(mask_lvl)
            if self.include_lowpass and len(masks) < len(Lp):
                # already has levels elements due to loop above
                pass
        else:
            masks = [None] * len(Lp)

        # Determine level weights
        if self.level_weights is not None:
            lw = self.level_weights.to(device=device)
            if lw.numel() != len(Lp):
                raise ValueError(f"level_weights length {lw.numel()} must match number of levels {len(Lp)}")
        else:
            lw = torch.ones(len(Lp), device=device) / len(Lp)

        # Accumulate masked L1 per level
        total = pred.new_tensor(0.0)
        for i, (lp, lt) in enumerate(zip(Lp, Lt)):
            m = masks[i]
            diff = torch.abs(lp - lt)
            if m is not None:
                # broadcast mask to channels
                if m.shape[1] == 1 and diff.shape[1] > 1:
                    m = m.expand(-1, diff.shape[1], -1, -1)
                valid = m
                if valid.any():
                    loss_i = (diff[valid]).mean()
                else:
                    loss_i = diff.new_tensor(0.0)
            else:
                loss_i = diff.mean()
            total = total + lw[i] * loss_i
        return total
