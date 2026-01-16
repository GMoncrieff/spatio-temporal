import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from torchgeo_dataloader import get_dataloader


def _has_deps() -> bool:
    return (
        importlib.util.find_spec("xarray") is not None
        and importlib.util.find_spec("xbatcher") is not None
        and importlib.util.find_spec("dask") is not None
        and importlib.util.find_spec("icechunk") is not None
    )


def _default_icechunk_repo_exists() -> bool:
    repo_path = Path(__file__).parent.parent / "scripts" / "notebooks" / "hm_global.icechunk"
    return repo_path.exists()


@pytest.mark.skipif(
    (not _has_deps()) or (not _default_icechunk_repo_exists()),
    reason="Zarr backend requires xarray/xbatcher/dask/icechunk and an existing scripts/notebooks/hm_global.icechunk repo",
)
def test_zarr_dataloader_batch_shapes():
    repo_path = str(Path(__file__).parent.parent / "scripts" / "notebooks" / "hm_global.icechunk")
    loader = get_dataloader(
        backend="zarr",
        zarr_path=repo_path,
        batch_size=2,
        chip_size=128,
        timesteps=3,
        chips_per_epoch=2,
        mode="random",
    )
    batch = next(iter(loader))
    C_dyn = loader.dataset.C_dyn
    C_static = loader.dataset.C_static

    assert batch["input_dynamic"].shape == (2, 3, C_dyn, 128, 128)
    assert batch["input_static"].shape == (2, C_static, 128, 128)
    assert batch["target"].shape == (2, 128, 128)

    # multi-horizon keys should exist
    assert "target_5yr" in batch
    assert "target_10yr" in batch
    assert "target_15yr" in batch
    assert "target_20yr" in batch


@pytest.mark.skipif(
    (not _has_deps()) or (not _default_icechunk_repo_exists()),
    reason="Zarr backend requires xarray/xbatcher/dask/icechunk and an existing scripts/notebooks/hm_global.icechunk repo",
)
def test_zarr_dataloader_compatible_with_convlstm():
    from src.models.convlstm import ConvLSTM

    repo_path = str(Path(__file__).parent.parent / "scripts" / "notebooks" / "hm_global.icechunk")
    loader = get_dataloader(
        backend="zarr",
        zarr_path=repo_path,
        batch_size=2,
        chip_size=64,
        timesteps=3,
        chips_per_epoch=2,
        mode="random",
    )
    batch = next(iter(loader))

    model = ConvLSTM(
        input_dim=loader.dataset.C_dyn,
        hidden_dim=[8],
        kernel_size=(3, 3),
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    )
    x = batch["input_dynamic"]
    output, _ = model(x)
    assert output[0].shape[0] == 2
    assert output[0].shape[1] == 3


@pytest.mark.skipif(
    (not _has_deps()) or (not _default_icechunk_repo_exists()),
    reason="Zarr backend requires xarray/xbatcher/dask/icechunk and an existing scripts/notebooks/hm_global.icechunk repo",
)
def test_icechunk_data_reading():
    """Test that icechunk data reading works and returns expected variables."""
    repo_path = str(Path(__file__).parent.parent / "scripts" / "notebooks" / "hm_global.icechunk")
    loader = get_dataloader(
        backend="zarr",
        zarr_path=repo_path,
        batch_size=1,
        chip_size=64,
        timesteps=3,
        chips_per_epoch=1,
        mode="random",
    )
    batch = next(iter(loader))
    
    # Check that we have the expected dynamic variables
    dataset = loader.dataset
    available_vars = list(dataset.ds.data_vars.keys())
    print(f"Available variables in icechunk dataset: {available_vars}")
    
    expected_dynamic_vars = ["AA"] + ["AG", "BU", "EX", "FR", "HI", "NS", "PO", "TI", "gdp", "population"]
    actual_dynamic_vars = [v for v in expected_dynamic_vars if v in dataset.ds.data_vars]
    
    assert len(actual_dynamic_vars) > 0, "Should have at least AA variable"
    assert "AA" in actual_dynamic_vars, "AA variable must be present"
    
    # Check static variables - they might have different naming convention
    actual_static_vars = [v for v in available_vars if v not in expected_dynamic_vars]
    
    assert len(actual_static_vars) > 0, f"Should have at least some static variables. Available: {available_vars}"
    
    # Verify data shapes
    assert batch["input_dynamic"].shape[0] == 1  # batch size
    assert batch["input_dynamic"].shape[1] == 3  # timesteps
    assert batch["input_dynamic"].shape[2] == len(actual_dynamic_vars)  # dynamic channels
    assert batch["input_dynamic"].shape[3] == 64  # height
    assert batch["input_dynamic"].shape[4] == 64  # width
    
    assert batch["input_static"].shape[0] == 1  # batch size
    assert batch["input_static"].shape[1] == len(actual_static_vars)  # static channels
    assert batch["input_static"].shape[2] == 64  # height
    assert batch["input_static"].shape[3] == 64  # width
