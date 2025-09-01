import pytest
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from torchgeo_dataloader import get_dataloader

def test_dataloader_batch_shapes():
    """
    Test that the dataloader yields batches of the expected shapes only.
    """
    loader = get_dataloader(batch_size=2, chip_size=128, timesteps=3, chips_per_epoch=2)
    batch = next(iter(loader))
    # Dynamic: [B, T, C_dyn, H, W] where C_dyn = 1 (HM) + 8 components = 9
    assert batch['input_dynamic'].ndim == 5
    assert batch['input_dynamic'].shape[:2] == (2, 3)
    assert batch['input_dynamic'].shape[2] == 9
    assert batch['input_dynamic'].shape[-2:] == (128, 128)
    # Static: [B, C_static, H, W] with 5 static channels (elevation, slope, BIO05/06/12)
    assert batch['input_static'].shape == (2, 5, 128, 128)
    # Target: [B, H, W]
    assert batch['target'].shape == (2, 128, 128)
    # Dynamic validity mask: [B, T, 1, H, W]
    assert 'dynamic_valid_mask' in batch
    assert batch['dynamic_valid_mask'].shape == (2, 3, 1, 128, 128)
    assert batch['dynamic_valid_mask'].dtype == torch.bool
    # Target validity mask: [B, 1, H, W]
    assert 'target_valid_mask' in batch
    assert batch['target_valid_mask'].shape == (2, 1, 128, 128)
    assert batch['target_valid_mask'].dtype == torch.bool
