import pytest
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from torchgeo_dataloader import get_dataloader

def test_torchgeo_dataloader():
    """
    Test that the torchgeo dataloader loads data for a given year and returns expected shapes.
    """
    loader = get_dataloader(year_idx=0)  # 1990
    batch = next(iter(loader))
    # Check keys
    assert 'image' in batch and 'target' in batch and 'year' in batch
    # Check types
    assert isinstance(batch['image'], torch.Tensor)
    assert isinstance(batch['target'], torch.Tensor)
    # Check shape: static (C, H, W), target (1, H, W)
    assert batch['image'].ndim == 4  # (B, C, H, W)
    assert batch['target'].ndim == 4  # (B, 1, H, W)
    # Check batch size
    assert batch['image'].shape[0] == 1
    assert batch['target'].shape[0] == 1
    # Check year
    assert int(batch['year'][0]) == 1990
