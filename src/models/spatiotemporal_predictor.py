import torch
import torch.nn as nn
from ..locationencoder import LocationEncoder
from .convlstm import ConvLSTM

class SpatioTemporalPredictor(nn.Module):
    """
    Multi-horizon spatio-temporal predictor using ConvLSTM with independent prediction heads.
    
    Architecture:
    - input: [B, T, C_d, H, W] dynamic + [B, C_s, H, W] static
    - ConvLSTM processes temporal sequence → shared representation
    - 12 independent prediction heads:
        * 4 central heads (one per horizon): Optimized for accuracy + spatial patterns
        * 4 lower quantile heads (2.5%): Optimized for lower bound estimation
        * 4 upper quantile heads (97.5%): Optimized for upper bound estimation
    - output: [B, 12, H, W] (predicted HM at 4 horizons × 3 predictions)
      
      Output channel ordering:
        - Channels [0, 3, 6, 9]: Lower 2.5% quantile (q=0.025)
        - Channels [1, 4, 7, 10]: Central prediction (optimized for MSE+SSIM+Lap+Hist)
        - Channels [2, 5, 8, 11]: Upper 97.5% quantile (q=0.975)
      Horizons: 5yr, 10yr, 15yr, 20yr
    
    Key Design:
    - Central heads are INDEPENDENT from quantile heads (no shared gradients)
    - Central prediction optimized for multiple objectives, not statistical median
    - Quantile heads only receive pinball loss gradients
    - Quantile heads are smaller (hidden_dim/2) for efficiency
    """
    def __init__(self, hidden_dim=16, kernel_size=3, num_layers=1, num_static_channels=1, num_dynamic_channels=1,
                 use_location_encoder: bool = True,
                 locenc_backbone=("sphericalharmonics", "siren"),
                 locenc_hparams=None,
                 locenc_out_channels: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_static_channels = int(num_static_channels)
        self.num_dynamic_channels = int(num_dynamic_channels)
        self.use_location_encoder = bool(use_location_encoder)
        self.locenc_out_channels = int(locenc_out_channels)
        if self.use_location_encoder and self.locenc_out_channels > 0:
            # Default hparams if not provided
            if locenc_hparams is None:
                locenc_hparams = dict(legendre_polys=10, dim_hidden=64, num_layers=2,
                                      optimizer=dict(lr=1e-4, wd=1e-3), num_classes=self.locenc_out_channels)
            else:
                locenc_hparams = dict(locenc_hparams)
                locenc_hparams['num_classes'] = self.locenc_out_channels
            self.location_encoder = LocationEncoder(locenc_backbone[0], locenc_backbone[1], locenc_hparams)
        else:
            self.location_encoder = None
        self.convlstm = ConvLSTM(
            input_dim=self.num_dynamic_channels + self.num_static_channels + (self.locenc_out_channels if (self.use_location_encoder and self.locenc_out_channels > 0) else 0),  # C_d dynamic + C_s static + C_loc
            hidden_dim=hidden_dim,
            kernel_size=(kernel_size, kernel_size),
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        # Multi-horizon prediction with independent heads for central and quantile predictions
        # Central heads: Optimized for accuracy + spatial patterns (MSE, SSIM, Laplacian, Histogram)
        # Quantile heads: Optimized purely for uncertainty estimation (Pinball loss only)
        self.num_horizons = 4
        
        # Central prediction heads (one per horizon)
        # These produce the "best estimate" optimized for multiple objectives
        self.central_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
            )
            for _ in range(self.num_horizons)
        ])
        
        # Lower quantile heads (2.5%, one per horizon)
        # Smaller networks since quantile estimation is simpler than full prediction
        self.lower_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, 1, kernel_size=1, bias=True),
            )
            for _ in range(self.num_horizons)
        ])
        
        # Upper quantile heads (97.5%, one per horizon)
        self.upper_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, 1, kernel_size=1, bias=True),
            )
            for _ in range(self.num_horizons)
        ])

    def forward(self, input_dynamic, input_static, lonlat=None):
        # input_dynamic: [B, T, C_d, H, W]
        # input_static: [B, C_s, H, W]
        # lonlat: [B, H, W, 2]
        B, T, C, H, W = input_dynamic.shape
        # Optionally compute learnable location features
        if self.use_location_encoder and (self.location_encoder is not None) and (lonlat is not None):
            # Vectorized: process entire batch at once
            ll_flat = lonlat.reshape(B * H * W, 2)  # [B*H*W, 2]
            feats = self.location_encoder(ll_flat)  # [B*H*W, C_loc]
            loc_feats = feats.view(B, H, W, self.locenc_out_channels).permute(0, 3, 1, 2).contiguous()  # [B, C_loc, H, W]
            input_static = torch.cat([input_static, loc_feats], dim=1)
        # Repeat all static channels for each timestep and concat
        static_rep = input_static.unsqueeze(1).repeat(1, T, 1, 1, 1)  # [B, T, C_s, H, W]
        x = torch.cat([input_dynamic, static_rep], dim=2)  # [B, T, C_d+C_s, H, W]
        # ConvLSTM expects [B, T, C, H, W]
        output, _ = self.convlstm(x)
        # output[0]: [B, T, hidden_dim, H, W] (last layer)
        last_hidden = output[0][:, -1]  # [B, hidden_dim, H, W] (last timestep)
        
        # Generate independent predictions for each horizon
        # Each horizon has 3 separate heads: lower, central, upper
        preds = []
        for h_idx in range(self.num_horizons):
            pred_lower = self.lower_heads[h_idx](last_hidden)    # [B, 1, H, W]
            pred_central = self.central_heads[h_idx](last_hidden) # [B, 1, H, W]
            pred_upper = self.upper_heads[h_idx](last_hidden)    # [B, 1, H, W]
            
            # Append in order: lower, central, upper for this horizon
            preds.extend([pred_lower, pred_central, pred_upper])
        
        # Stack predictions: [B, 12, H, W] (4 horizons × 3 predictions)
        # Channel ordering: [lower_5yr, central_5yr, upper_5yr, lower_10yr, central_10yr, upper_10yr, ...]
        pred = torch.cat(preds, dim=1)
        return pred
