import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.models.lightning_module import SpatioTemporalLightningModule
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from torchgeo_dataloader import get_dataloader

def test_lightning_module_forward():
    loader = get_dataloader(batch_size=2, chip_size=128, timesteps=3, chips_per_epoch=2)
    batch = next(iter(loader))
    C_dyn = loader.dataset.C_dyn
    C_static = batch['input_static'].shape[1]
    model = SpatioTemporalLightningModule(hidden_dim=8, lr=1e-3,
                                          num_dynamic_channels=C_dyn,
                                          num_static_channels=C_static)
    # Forward pass
    preds = model(batch['input_dynamic'], batch['input_static'])
    assert preds.shape == (2, 1, 128, 128)
    # Training step (should return scalar loss)
    loss = model.training_step(batch, 0)
    assert loss.dim() == 0
