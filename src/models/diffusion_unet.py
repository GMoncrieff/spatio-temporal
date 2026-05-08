"""Conditional 2D diffusion U-Net wrapping diffusers.UNet2DModel.

The wrapper concatenates the conditioning tensor with the noisy target along
the channel axis before each step, so conditioning is supplied to every
denoising layer (no cross-attention needed for spatial covariates).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from diffusers import UNet2DModel


class ConditionalDiffusionUNet(nn.Module):
    """Wraps `UNet2DModel` for conditional Δhm prediction.

    Args:
        cond_channels: number of conditioning channels concatenated to the noisy
            target. Output is always 1 channel (Δhm v- or ε-prediction).
        sample_size: spatial size of input/output (square chips).
        base_channels: channel width at the highest resolution.
        channel_mults: per-stage multipliers; depth = len(channel_mults).
        attention_at_low_two: when True, the two lowest-resolution stages use
            self-attention down/up blocks; the higher stages are plain conv blocks.
        layers_per_block: residual layers per resolution stage.
        attention_head_dim: head dim for self-attention in attn blocks.
    """

    def __init__(
        self,
        cond_channels: int,
        sample_size: int = 64,
        base_channels: int = 128,
        channel_mults: Sequence[int] = (1, 2, 2, 4),
        attention_at_low_two: bool = True,
        layers_per_block: int = 2,
        attention_head_dim: int = 64,
    ):
        super().__init__()
        if len(channel_mults) != 4:
            raise ValueError(
                f"channel_mults must have exactly 4 entries, got {len(channel_mults)}"
            )
        block_out_channels = tuple(base_channels * m for m in channel_mults)

        if attention_at_low_two:
            down_block_types = (
                "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D",
            )
            up_block_types = (
                "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",
            )
        else:
            down_block_types = ("DownBlock2D",) * 4
            up_block_types = ("UpBlock2D",) * 4

        self.cond_channels = cond_channels
        self.sample_size = sample_size

        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=cond_channels + 1,
            out_channels=1,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=layers_per_block,
            attention_head_dim=attention_head_dim,
        )

    def forward(
        self,
        noisy_target: torch.Tensor,
        conditioning: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_target.shape[1] != 1:
            raise ValueError(f"noisy_target must have 1 channel, got {noisy_target.shape[1]}")
        if conditioning.shape[1] != self.cond_channels:
            raise ValueError(
                f"conditioning has {conditioning.shape[1]} channels, "
                f"expected {self.cond_channels}"
            )
        x = torch.cat([noisy_target, conditioning], dim=1)
        return self.unet(x, timesteps).sample
