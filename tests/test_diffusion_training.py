"""Sanity tests for DiffusionLightningModule training and sampling."""
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.diffusion_lightning import DiffusionLightningModule


# Small architecture for fast tests
B, T, C_DYN, C_STAT, H, W = 2, 3, 2, 3, 32, 32
COND_CHANNELS = T * C_DYN + C_STAT + 1  # +1 for hm_t_normalized; no locenc


def _make_module(seed: int = 0):
    torch.manual_seed(seed)
    return DiffusionLightningModule(
        cond_channels=COND_CHANNELS,
        sample_size=H,
        base_channels=32,
        channel_mults=(1, 2, 2, 4),
        attention_head_dim=16,
        num_train_timesteps=200,
        num_inference_steps=4,
        location_encoder_kwargs=None,
    )


def _make_batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    input_dynamic = torch.randn(B, T, C_DYN, H, W, generator=g)
    input_static = torch.randn(B, C_STAT, H, W, generator=g)
    target_dhm = torch.randn(B, 1, H, W, generator=g) * 0.3
    valid_mask = torch.ones(B, H, W, dtype=torch.bool)
    hm_t_normalized = input_dynamic[:, 2, 0:1]  # [B, 1, H, W]
    lonlat = torch.zeros(B, H, W, 2)
    return dict(
        input_dynamic=input_dynamic,
        input_static=input_static,
        lonlat=lonlat,
        target_dhm=target_dhm,
        valid_mask=valid_mask,
        hm_t_normalized=hm_t_normalized,
    )


def test_training_step_loss_is_finite():
    module = _make_module()
    batch = _make_batch()
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"


def test_loss_decreases_over_n_steps():
    module = _make_module(seed=42)
    batch = _make_batch(seed=42)
    opt = module.configure_optimizers()
    if isinstance(opt, (list, tuple)):
        opt = opt[0]

    losses = []
    for step in range(60):
        opt.zero_grad()
        loss = module.training_step(batch, batch_idx=step)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    assert final < initial * 0.85, (
        f"loss did not decrease enough on a fixed synthetic batch: "
        f"initial mean(0:5)={initial:.4f}, final mean(-5:)={final:.4f}"
    )


def test_sample_shape():
    module = _make_module()
    module.train(False)
    cond = torch.randn(B, COND_CHANNELS, H, W)
    samples = module.sample(cond, n_samples=3, num_inference_steps=4)
    assert samples.shape == (3, B, 1, H, W)
