import torch
import torch.nn as nn
from .swin_unet import SwinTransformerSys
from ..locationencoder import LocationEncoder


class SwinUNetPredictor(nn.Module):
    """
    Swin-UNet wrapper for spatio-temporal prediction.
    Flattens temporal dimension into channels and uses Swin-UNet for prediction.
    
    Args:
        hidden_dim: Maps to embed_dim in Swin-UNet (default 96)
        num_static_channels: Number of static input channels
        num_dynamic_channels: Number of dynamic channels per timestep
        num_layers: Number of encoder layers (used to create depths)
        kernel_size: Maps to window_size
        use_location_encoder: Whether to use location encoder
        locenc_backbone: Location encoder backbone type
        locenc_hparams: Location encoder hyperparameters
        locenc_out_channels: Output channels for location encoder
    """
    def __init__(
        self,
        hidden_dim: int = 96,
        num_static_channels: int = 1,
        num_dynamic_channels: int = 1,
        num_layers: int = 4,
        kernel_size: int = 4,
        use_location_encoder: bool = True,
        locenc_backbone=("sphericalharmonics", "siren"),
        locenc_hparams=None,
        locenc_out_channels: int = 8,
    ):
        super().__init__()
        self.num_static_channels = int(num_static_channels)
        self.num_dynamic_channels = int(num_dynamic_channels)
        self.use_location_encoder = bool(use_location_encoder)
        self.locenc_out_channels = int(locenc_out_channels)
        
        # Location encoder setup (Option A: append to static channels)
        if self.use_location_encoder and self.locenc_out_channels > 0:
            if locenc_hparams is None:
                locenc_hparams = dict(
                    legendre_polys=10,
                    dim_hidden=64,
                    num_layers=2,
                    optimizer=dict(lr=1e-4, wd=1e-3),
                    num_classes=self.locenc_out_channels
                )
            else:
                locenc_hparams = dict(locenc_hparams)
                locenc_hparams['num_classes'] = self.locenc_out_channels
            self.location_encoder = LocationEncoder(locenc_backbone[0], locenc_backbone[1], locenc_hparams)
        else:
            self.location_encoder = None
        
        # Calculate total input channels after flattening time
        # Dynamic channels will be flattened: T * C_dyn
        # We'll assume T=3 for now (fixed in dataloader)
        self.num_timesteps = 3
        total_dynamic_channels = self.num_timesteps * self.num_dynamic_channels
        
        # Static channels + location encoder channels (if enabled)
        total_static_channels = self.num_static_channels
        if self.use_location_encoder and self.locenc_out_channels > 0:
            total_static_channels += self.locenc_out_channels
        
        # Total input channels to Swin-UNet
        total_in_channels = total_dynamic_channels + total_static_channels
        
        # Map parameters to Swin-UNet configuration
        embed_dim = hidden_dim
        # Create depths based on num_layers (replicate for encoder)
        depths = [2] * num_layers
        depths_decoder = [1] + [2] * (num_layers - 1)
        
        # Scale num_heads with embed_dim (standard ratios)
        # For embed_dim=96: [3, 6, 12, 24]
        # For embed_dim=64: [2, 4, 8, 16]
        base_heads = max(1, embed_dim // 32)
        num_heads = [base_heads * (2 ** i) for i in range(num_layers)]
        
        # Window size: use kernel_size, but ensure it divides patch resolution
        # With img_size=128, patch_size=4 → patches_resolution=32
        # Valid window sizes: 1, 2, 4, 8, 16, 32
        window_size = kernel_size
        if window_size not in [1, 2, 4, 8, 16]:
            # Default to 4 if invalid
            window_size = 4
        
        print(f"SwinUNetPredictor config:")
        print(f"  Input channels: {total_in_channels} ({total_dynamic_channels} dynamic + {total_static_channels} static)")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Depths: {depths}")
        print(f"  Depths decoder: {depths_decoder}")
        print(f"  Num heads: {num_heads}")
        print(f"  Window size: {window_size}")
        
        # Create Swin-UNet model
        self.swin_unet = SwinTransformerSys(
            img_size=128,
            patch_size=4,
            in_chans=total_in_channels,
            num_classes=1,  # Single output channel (HM prediction)
            embed_dim=embed_dim,
            depths=depths,
            depths_decoder=depths_decoder,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            final_upsample="expand_first"
        )
    
    def forward(self, input_dynamic, input_static, lonlat=None):
        """
        Args:
            input_dynamic: [B, T, C_dyn, H, W]
            input_static: [B, C_static, H, W]
            lonlat: [B, H, W, 2] (optional, for location encoder)
        
        Returns:
            pred: [B, 1, H, W]
        """
        B, T, C_dyn, H, W = input_dynamic.shape
        orig_H, orig_W = H, W
        
        # Compute location encoder features if enabled
        if self.use_location_encoder and (self.location_encoder is not None) and (lonlat is not None):
            # Vectorized: process entire batch at once
            ll_flat = lonlat.reshape(B * H * W, 2)  # [B*H*W, 2]
            feats = self.location_encoder(ll_flat)  # [B*H*W, C_loc]
            loc_feats = feats.view(B, H, W, self.locenc_out_channels).permute(0, 3, 1, 2).contiguous()
            # Append to static channels
            input_static = torch.cat([input_static, loc_feats], dim=1)  # [B, C_static + C_loc, H, W]
        
        # Flatten temporal dimension: [B, T, C_dyn, H, W] → [B, T*C_dyn, H, W]
        input_dynamic_flat = input_dynamic.reshape(B, T * C_dyn, H, W)
        
        # Concatenate dynamic (flattened) and static channels
        x = torch.cat([input_dynamic_flat, input_static], dim=1)  # [B, T*C_dyn + C_static, H, W]
        
        # Pad to expected size (128×128) if needed for Swin-UNet
        expected_size = 128
        pad_h = max(0, expected_size - H)
        pad_w = max(0, expected_size - W)
        
        if pad_h > 0 or pad_w > 0:
            # Pad right and bottom with replicate padding (no size limits)
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        
        # Pass through Swin-UNet
        pred = self.swin_unet(x)  # [B, 1, H_padded, W_padded]
        
        # Crop back to original size if padded
        if pad_h > 0 or pad_w > 0:
            pred = pred[:, :, :orig_H, :orig_W]
        
        return pred
