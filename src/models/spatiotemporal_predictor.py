import torch
import torch.nn as nn
from .convlstm import ConvLSTM

class SpatioTemporalPredictor(nn.Module):
    """
    Model for next-step prediction with ConvLSTM and static variables.
    - input_dynamic: [B, T, 1, H, W] (human footprint)
    - input_static: [B, 1, H, W] (static, e.g. elevation)
    - output: [B, 1, H, W] (predicted next human footprint)
    """
    def __init__(self, hidden_dim=16, kernel_size=3, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.convlstm = ConvLSTM(
            input_dim=2,  # 1 dynamic + 1 static channel
            hidden_dim=hidden_dim,
            kernel_size=(kernel_size, kernel_size),
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        self.head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, input_dynamic, input_static):
        # input_dynamic: [B, T, 1, H, W]
        # input_static: [B, 1, H, W]
        B, T, C, H, W = input_dynamic.shape
        # Repeat static for each timestep and concat
        static_rep = input_static.unsqueeze(1).repeat(1, T, 1, 1, 1)  # [B, T, 1, H, W]
        x = torch.cat([input_dynamic, static_rep], dim=2)  # [B, T, 2, H, W]
        # Merge channel for ConvLSTM: [B, T, 2, H, W] -> [B, T, 2, H, W]
        x = x.view(B, T, 2, H, W)
        # ConvLSTM expects [B, T, C, H, W]
        output, _ = self.convlstm(x)
        # output[0]: [B, T, hidden_dim, H, W] (last layer)
        last_hidden = output[0][:, -1]  # [B, hidden_dim, H, W] (last timestep)
        pred = self.head(last_hidden)   # [B, 1, H, W]
        return pred
