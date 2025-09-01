import torch
import torch.nn as nn
from .convlstm import ConvLSTM

class SpatioTemporalPredictor(nn.Module):
    """
    Next-step predictor with ConvLSTM supporting multi-channel dynamic inputs and multi-channel static covariates.
    - input_dynamic: [B, T, C_dyn, H, W]  (e.g., HM + components)
    - input_static:  [B, C_static, H, W]  (e.g., elevation, slope, climate)
    - output:        [B, 1, H, W]        (predicted next HM)
    """
    def __init__(self, hidden_dim: int = 16, kernel_size: int = 3, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        # Lazy init modules that depend on input channel count
        self.convlstm = None
        self.head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def _lazy_init(self, c_in: int):
        if self.convlstm is None:
            self.convlstm = ConvLSTM(
                input_dim=c_in,
                hidden_dim=self.hidden_dim,
                kernel_size=(self.kernel_size, self.kernel_size),
                num_layers=self.num_layers,
                batch_first=True,
                bias=True,
                return_all_layers=False,
            )

    def forward(self, input_dynamic: torch.Tensor, input_static: torch.Tensor) -> torch.Tensor:
        # input_dynamic: [B, T, C_dyn, H, W]
        # input_static:  [B, C_static, H, W]
        B, T, C_dyn, H, W = input_dynamic.shape
        # Repeat static over timesteps and concat along channel dim
        static_rep = input_static.unsqueeze(1).repeat(1, T, 1, 1, 1)  # [B, T, C_static, H, W]
        x = torch.cat([input_dynamic, static_rep], dim=2)            # [B, T, C_dyn+C_static, H, W]
        c_in = x.size(2)
        # Lazy init ConvLSTM with correct input_dim
        self._lazy_init(c_in)
        # Ensure lazily created module lives on the same device as inputs
        if next(self.convlstm.parameters()).device != x.device:
            self.convlstm = self.convlstm.to(x.device)
        output, _ = self.convlstm(x)
        last_hidden = output[0][:, -1]  # [B, hidden_dim, H, W]
        pred = self.head(last_hidden)   # [B, 1, H, W]
        return pred
