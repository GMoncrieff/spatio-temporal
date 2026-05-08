"""Sanity tests for the conditional diffusion U-Net wrapper."""
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.diffusion_unet import ConditionalDiffusionUNet


def test_forward_shape():
    model = ConditionalDiffusionUNet(
        cond_channels=50,
        sample_size=64,
        base_channels=64,            # smaller for fast CI
        channel_mults=(1, 2, 2, 4),
    )
    B, H, W = 2, 64, 64
    noisy = torch.randn(B, 1, H, W)
    cond = torch.randn(B, 50, H, W)
    t = torch.randint(0, 1000, (B,))
    out = model(noisy, cond, t)
    assert out.shape == (B, 1, H, W)


def test_param_count_in_range():
    model = ConditionalDiffusionUNet(
        cond_channels=50,
        sample_size=64,
        base_channels=128,
        channel_mults=(1, 2, 2, 4),
    )
    n_params = sum(p.numel() for p in model.parameters())
    # Spec target: ~50–80M parameters with base 128 + mults [1,2,2,4]
    assert 30_000_000 < n_params < 150_000_000, f"unexpected param count: {n_params}"


def test_cond_channel_mismatch_raises():
    model = ConditionalDiffusionUNet(cond_channels=10, sample_size=32, base_channels=32)
    noisy = torch.randn(1, 1, 32, 32)
    cond_wrong = torch.randn(1, 9, 32, 32)
    with pytest.raises(ValueError):
        model(noisy, cond_wrong, torch.tensor([0]))
