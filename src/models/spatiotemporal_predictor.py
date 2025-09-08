import torch
import torch.nn as nn
from .convlstm import ConvLSTM

class SpatioTemporalPredictor(nn.Module):
    """
    Model for next-step prediction with ConvLSTM and static variables.
    - input_dynamic: [B, T, C_d, H, W] (C_d dynamic channels per timestep; channel 0 is HM)
    - input_static: [B, C_s, H, W] (C_s static channels, e.g., elevation, slope, climate)
    - output: [B, 1, H, W] (predicted next human footprint)
    """
    def __init__(self, hidden_dim=16, kernel_size=3, num_layers=1, num_static_channels=1, num_dynamic_channels=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_static_channels = int(num_static_channels)
        self.num_dynamic_channels = int(num_dynamic_channels)
        self.convlstm = ConvLSTM(
            input_dim=self.num_dynamic_channels + self.num_static_channels,  # C_d dynamic + C_s static
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

    def forward(self, input_dynamic, input_static):
        # input_dynamic: [B, T, C_d, H, W]
        # input_static: [B, C_s, H, W]
        B, T, C, H, W = input_dynamic.shape
        # Repeat all static channels for each timestep and concat
        static_rep = input_static.unsqueeze(1).repeat(1, T, 1, 1, 1)  # [B, T, C_s, H, W]
        x = torch.cat([input_dynamic, static_rep], dim=2)  # [B, T, C_d+C_s, H, W]
        # ConvLSTM expects [B, T, C, H, W]
        output, _ = self.convlstm(x)
        # output[0]: [B, T, hidden_dim, H, W] (last layer)
        last_hidden = output[0][:, -1]  # [B, hidden_dim, H, W] (last timestep)
        pred = self.head(last_hidden)   # [B, 1, H, W]
        return pred
