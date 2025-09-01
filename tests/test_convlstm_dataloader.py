import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.models.convlstm import ConvLSTM
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from torchgeo_dataloader import get_dataloader

def test_convlstm_accepts_dataloader_batch():
    loader = get_dataloader(batch_size=2, chip_size=128, timesteps=3, chips_per_epoch=2)
    batch = next(iter(loader))
    model = ConvLSTM(
        input_dim=1,
        hidden_dim=[8],
        kernel_size=(3, 3),
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False
    )
    # Only use dynamic input for ConvLSTM
    # Select HM channel (channel 0) -> [B, T, 1, H, W]
    x = batch['input_dynamic'][:, :, 0:1, ...]
    output, _ = model(x)
    assert output[0].shape == (2, 3, 8, 128, 128)
