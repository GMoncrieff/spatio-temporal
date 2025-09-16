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
    C_dyn = loader.dataset.C_dyn
    C_static = loader.dataset.C_static
    assert batch['input_dynamic'].shape == (2, 3, C_dyn, 128, 128)
    assert batch['input_static'].shape == (2, C_static, 128, 128)
    assert batch['target'].shape == (2, 128, 128)
