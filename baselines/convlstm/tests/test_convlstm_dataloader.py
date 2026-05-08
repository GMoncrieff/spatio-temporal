import torch
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from baselines.convlstm.models.convlstm import ConvLSTM
from torchgeo_dataloader import get_dataloader

def test_convlstm_accepts_dataloader_batch():
    loader = get_dataloader(batch_size=2, chip_size=128, timesteps=3, chips_per_epoch=2)
    batch = next(iter(loader))
    C_dyn = loader.dataset.C_dyn
    model = ConvLSTM(
        input_dim=C_dyn,
        hidden_dim=[8],
        kernel_size=(3, 3),
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False
    )
    # Only use dynamic input for ConvLSTM
    x = batch['input_dynamic']  # [B, T, C, H, W]
    output, _ = model(x)
    assert output[0].shape == (2, 3, 8, 128, 128)
