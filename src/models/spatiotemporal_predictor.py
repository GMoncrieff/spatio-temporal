import torch
import torch.nn as nn
from .convlstm import ConvLSTM
from ..locationencoder import LocationEncoder

class SpatioTemporalPredictor(nn.Module):
    """
    Model for next-step prediction with ConvLSTM and static variables.
    - input_dynamic: [B, T, C_d, H, W] (C_d dynamic channels per timestep; channel 0 is HM)
    - input_static: [B, C_s, H, W] (C_s static channels, e.g., elevation, slope, climate)
    - lonlat: [B, H, W, 2] (per-pixel longitude, latitude in degrees), optional
    - output: [B, 1, H, W] (predicted next human footprint)
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
        # Deeper prediction head for increased capacity
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
        )

    def forward(self, input_dynamic, input_static, lonlat=None):
        # input_dynamic: [B, T, C_d, H, W]
        # input_static: [B, C_s, H, W]
        # lonlat: [B, H, W, 2]
        B, T, C, H, W = input_dynamic.shape
        # Optionally compute learnable location features
        if self.use_location_encoder and (self.location_encoder is not None) and (lonlat is not None):
            # Build per-batch encoded features
            loc_feats_list = []
            for b in range(B):
                ll = lonlat[b]  # [H, W, 2]
                ll_flat = ll.reshape(-1, 2)  # [H*W, 2]
                feats = self.location_encoder(ll_flat)  # [H*W, C_loc]
                feats = feats.view(H, W, -1).permute(2, 0, 1).contiguous()  # [C_loc, H, W]
                loc_feats_list.append(feats)
            loc_feats = torch.stack(loc_feats_list, dim=0)  # [B, C_loc, H, W]
            input_static = torch.cat([input_static, loc_feats], dim=1)
        # Repeat all static channels for each timestep and concat
        static_rep = input_static.unsqueeze(1).repeat(1, T, 1, 1, 1)  # [B, T, C_s, H, W]
        x = torch.cat([input_dynamic, static_rep], dim=2)  # [B, T, C_d+C_s, H, W]
        # ConvLSTM expects [B, T, C, H, W]
        output, _ = self.convlstm(x)
        # output[0]: [B, T, hidden_dim, H, W] (last layer)
        last_hidden = output[0][:, -1]  # [B, hidden_dim, H, W] (last timestep)
        pred = self.head(last_hidden)   # [B, 1, H, W]
        return pred
