"""
3D UNet for spatio-temporal prediction.
Drop-in replacement for ConvLSTM in SpatioTemporalPredictor.

Input: [B, T, C, H, W] (batch, time, channels, height, width)
Output: [B, hidden_dim, H, W] (batch, hidden_dim, height, width)

Includes optional temporal processing (LSTM) on skip connections between
encoder and decoder layers, following ThemedaUNet design.
"""

import torch
import torch.nn as nn
from typing import List
from enum import Enum


class TemporalProcessorType(Enum):
    """Enum for temporal processor types."""
    NONE = "none"
    LSTM = "lstm"
    GRU = "gru"


def Conv3D(*args, **kwargs):
    """Wrapper for 3D convolution."""
    return nn.Conv3d(*args, **kwargs)


def ConvTranspose3D(*args, **kwargs):
    """Wrapper for 3D transposed convolution."""
    return nn.ConvTranspose3d(*args, **kwargs)


def time_distributed_combine(x):
    """
    Combine batch and time dimensions for processing.
    Input: [B, T, C, H, W] or [B, C, T, H, W]
    Returns: (combined, time_distributed, batch_size, timesteps)
    """
    batch_size = x.shape[0]
    timesteps = 0
    time_distributed = (len(x.shape) == 5)
    if time_distributed:
        timesteps = x.shape[1]
        new_shape = (batch_size * timesteps,) + x.shape[2:]
        x = x.contiguous().view(new_shape)
    return x, time_distributed, batch_size, timesteps


def spatial_combine(x):
    """
    Combine spatial dimensions for temporal processing.
    Input: [B, T, C, H, W]
    Output: (combined, batch_size, height, width, timesteps, features)
    """
    batch_size = x.shape[0]
    height = x.shape[-2]
    width = x.shape[-1]
    timesteps = x.shape[1]
    features = x.shape[2]
    new_shape = (batch_size * height * width, timesteps, features)
    x = x.permute(0, 3, 4, 1, 2).contiguous().view(new_shape)
    return x, batch_size, height, width, timesteps, features


@torch.jit.script
def autocrop_3d(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """
    Center-crops the encoder_layer to the size of the decoder_layer for 3D tensors.
    Handles spatial dimensions (H, W) and temporal dimension (T).
    """
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]  # (T, H, W)
        es = decoder_layer.shape[2:]  # (T, H, W)
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        assert ds[2] >= es[2]
        encoder_layer = encoder_layer[
            :,
            :,
            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2),
            ((ds[2] - es[2]) // 2):((ds[2] + es[2]) // 2),
        ]
    return encoder_layer, decoder_layer


class ResBlock3D(nn.Module):
    """3D Residual block with optional downsampling (spatial only, not temporal)."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
        padding_mode: str = "reflect",
        kernel_size: int = 3,
    ):
        super().__init__()
        self.padding_mode = padding_mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        padding = (kernel_size - 1) // 2

        if downsample:
            # Downsample only spatial dimensions (H, W), not temporal (T)
            # stride=(1, 2, 2) means no downsampling in T, 2x in H and W
            self.conv1 = Conv3D(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=(1, 2, 2), padding=padding,
                padding_mode=padding_mode
            )
            self.shortcut = Conv3D(
                in_channels, out_channels,
                kernel_size=1, stride=(1, 2, 2), padding_mode=padding_mode
            )
        else:
            self.conv1 = Conv3D(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=1, padding=padding,
                padding_mode=padding_mode
            )
            self.shortcut = nn.Identity()

        self.conv2 = Conv3D(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=1, padding=padding,
            padding_mode=padding_mode
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(x + shortcut)
        return x


class DownBlock3D(nn.Module):
    """3D downsampling block with residual connections."""
    def __init__(
        self,
        in_channels: int,
        downsample: bool = True,
        growth_factor: float = 2.0,
        kernel_size: int = 3,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.padding_mode = padding_mode

        if downsample:
            self.out_channels = int(growth_factor * self.out_channels)

        self.block1 = ResBlock3D(
            in_channels=in_channels,
            out_channels=self.out_channels,
            downsample=downsample,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
        )
        self.block2 = ResBlock3D(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            downsample=False,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)
        x2 = self.block2(x1)
        return x2


class UpBlock3D(nn.Module):
    """3D upsampling block with skip connections (spatial only, not temporal)."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: str = "reflect",
        resblock_kernel_size: int = 3,
        upsample_kernel_size: int = 2,
    ):
        super().__init__()
        self.padding_mode = padding_mode
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Upsample only spatial dimensions (H, W), not temporal (T)
        # stride=(1, 2, 2) means no upsampling in T, 2x in H and W
        self.upsample = ConvTranspose3D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(1, upsample_kernel_size, upsample_kernel_size),
            stride=(1, 2, 2),
        )

        self.block1 = ResBlock3D(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            downsample=False,
            kernel_size=resblock_kernel_size,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor, shortcut: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x, shortcut = autocrop_3d(x, shortcut)
        x = x + shortcut
        x = self.block1(x)
        return x


class UNet3D(nn.Module):
    """
    3D UNet for spatio-temporal prediction with optional temporal processing.
    
    Input: [B, C_in, T, H, W] where C_in = num_dynamic_channels + num_static_channels
    Output: [B, hidden_dim, H, W]
    
    The temporal dimension is processed as part of the 3D convolutions,
    but the output is collapsed to 2D spatial with hidden_dim channels.
    
    Optional temporal processors (LSTM/GRU) operate on skip connections
    between encoder and decoder layers, following ThemedaUNet design.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 16,
        padding_mode: str = "reflect",
        growth_factor: float = 2.0,
        kernel_size: int = 3,
        layers: int = 3,
        temporal_processor_type: str = "none",
        temporal_layers: int = 1,
        temporal_bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.padding_mode = padding_mode
        self.growth_factor = growth_factor
        self.kernel_size = kernel_size
        self.layers = layers

        # Parse temporal processor type
        if isinstance(temporal_processor_type, str):
            temporal_processor_type = TemporalProcessorType[temporal_processor_type.upper()]
        self.temporal_processor_type = temporal_processor_type

        current_num_features = in_channels

        # Encoder (downsampling path)
        self.downblock_layers = nn.ModuleList()
        for layer_idx in range(layers):
            downblock = DownBlock3D(
                in_channels=current_num_features,
                downsample=True,
                growth_factor=growth_factor,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
            )
            self.downblock_layers.append(downblock)
            current_num_features = downblock.out_channels

        # Temporal processors for skip connections
        self.temporal_processors = nn.ModuleList()
        if self.temporal_processor_type != TemporalProcessorType.NONE:
            for downblock in self.downblock_layers:
                rnn_kwargs = dict(
                    batch_first=True,
                    bidirectional=False,
                    input_size=downblock.in_channels,
                    hidden_size=downblock.in_channels,
                    num_layers=temporal_layers,
                    bias=temporal_bias,
                )
                if self.temporal_processor_type == TemporalProcessorType.LSTM:
                    temporal_processor = nn.LSTM(**rnn_kwargs)
                elif self.temporal_processor_type == TemporalProcessorType.GRU:
                    temporal_processor = nn.GRU(**rnn_kwargs)
                else:
                    raise ValueError(f"Unknown temporal processor type: {self.temporal_processor_type}")
                self.temporal_processors.append(temporal_processor)

        # Decoder (upsampling path)
        self.upblock_layers = nn.ModuleList()
        for downblock in reversed(self.downblock_layers):
            upblock = UpBlock3D(
                in_channels=downblock.out_channels,
                out_channels=downblock.in_channels,
                padding_mode=padding_mode,
                resblock_kernel_size=kernel_size,
            )
            self.upblock_layers.append(upblock)
            current_num_features = upblock.out_channels

        # Final prediction layer: project to hidden_dim
        # Input: [B, current_num_features, T, H, W]
        # Output: [B, hidden_dim, H, W]
        self.prediction_layer = Conv3D(
            in_channels=current_num_features,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, C_in, T, H, W]
        
        Returns:
            [B, hidden_dim, H, W]
        """
        # Encoder: [B, C, T, H, W]
        encoded_list = []
        for downblock in self.downblock_layers:
            encoded_list.append(x)
            x = downblock(x)

        # Apply temporal processing to skip connections if enabled
        if self.temporal_processor_type != TemporalProcessorType.NONE:
            temporal_encoded_list = []
            for encoded, temporal_processor in zip(encoded_list, self.temporal_processors):
                encoded_shape = encoded.shape
                # Reshape from [B, C, T, H, W] to [B, T, C, H, W]
                encoded = encoded.permute(0, 2, 1, 3, 4).contiguous()
                # Combine spatial dimensions for temporal processing
                encoded, batch_size, height, width, timesteps, features = spatial_combine(encoded)
                # encoded is now [B*H*W, T, C]
                
                # Apply temporal processor (LSTM/GRU)
                encoded, _ = temporal_processor(encoded)
                # encoded is now [B*H*W, T, C]
                
                # Reshape back to [B, T, C, H, W]
                encoded = encoded.contiguous().view((batch_size, height, width, timesteps, features))
                encoded = encoded.permute(0, 4, 3, 1, 2).contiguous()
                # encoded is now [B, C, T, H, W]
                
                assert encoded.shape == encoded_shape, \
                    f"Shape mismatch after temporal processing: {encoded.shape} vs {encoded_shape}"
                
                temporal_encoded_list.append(encoded)
            
            encoded_list = temporal_encoded_list

        # Decoder
        for encoded, upblock in zip(reversed(encoded_list), self.upblock_layers):
            x = upblock(x, encoded)

        # Prediction layer: [B, C, T, H, W] → [B, hidden_dim, T, H, W]
        x = self.prediction_layer(x)

        # Collapse temporal dimension by taking the last timestep
        # [B, hidden_dim, T, H, W] → [B, hidden_dim, H, W]
        x = x[:, :, -1, :, :]

        return x
