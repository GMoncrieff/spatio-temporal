import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.models.convlstm import ConvLSTM

def test_convlstm_forward():
    # Dummy input: batch=2, timesteps=3, channels=1, height=128, width=128
    x = torch.randn(2, 3, 1, 128, 128)
    model = ConvLSTM(
        input_dim=1,
        hidden_dim=[8],
        kernel_size=(3, 3),
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False
    )
    output, last_state = model(x)
    # Output is a list with one tensor: [batch, timesteps, hidden_dim, H, W]
    assert isinstance(output, list)
    assert output[0].shape == (2, 3, 8, 128, 128)
    assert len(last_state) == 1
    h, c = last_state[0]
    assert h.shape == (2, 8, 128, 128)
    assert c.shape == (2, 8, 128, 128)
