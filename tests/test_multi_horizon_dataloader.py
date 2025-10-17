"""
Test multi-horizon dataloader.
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.torchgeo_dataloader import get_dataloader


def test_dataloader_output_keys():
    """Test that dataloader returns all required keys including multi-horizon targets."""
    # Create a small dataloader for testing
    train_loader = get_dataloader(
        batch_size=2,
        chip_size=128,
        timesteps=3,
        mode="random",
        chips_per_epoch=5,
        num_workers=0,
        include_components=True
    )
    
    # Get one batch
    batch = next(iter(train_loader))
    
    # Check required keys
    required_keys = [
        'input_dynamic',
        'input_static',
        'lonlat',
        'target_5yr',
        'target_10yr',
        'target_15yr',
        'target_20yr',
        'timestep'
    ]
    
    for key in required_keys:
        assert key in batch, f"Missing key: {key}"
    
    print(f"✓ Dataloader returns all required keys: {list(batch.keys())}")


def test_dataloader_target_shapes():
    """Test that all targets have correct shapes."""
    batch_size = 2
    chip_size = 128
    
    train_loader = get_dataloader(
        batch_size=batch_size,
        chip_size=chip_size,
        timesteps=3,
        mode="random",
        chips_per_epoch=5,
        num_workers=0,
        include_components=True
    )
    
    batch = next(iter(train_loader))
    
    # Check each target shape
    for horizon in ['target_5yr', 'target_10yr', 'target_15yr', 'target_20yr']:
        target = batch[horizon]
        expected_shape = (batch_size, chip_size, chip_size)
        assert target.shape == expected_shape, \
            f"{horizon} has shape {target.shape}, expected {expected_shape}"
    
    print(f"✓ All targets have correct shape: ({batch_size}, {chip_size}, {chip_size})")


def test_dataloader_input_shapes():
    """Test that inputs have correct shapes."""
    batch_size = 2
    chip_size = 128
    timesteps = 3
    num_dynamic_channels = 11  # 1 HM + 10 covariates
    num_static_channels = 7
    
    train_loader = get_dataloader(
        batch_size=batch_size,
        chip_size=chip_size,
        timesteps=timesteps,
        mode="random",
        chips_per_epoch=5,
        num_workers=0,
        include_components=True
    )
    
    batch = next(iter(train_loader))
    
    # Check input_dynamic shape
    input_dynamic = batch['input_dynamic']
    expected_dyn_shape = (batch_size, timesteps, num_dynamic_channels, chip_size, chip_size)
    assert input_dynamic.shape == expected_dyn_shape, \
        f"input_dynamic has shape {input_dynamic.shape}, expected {expected_dyn_shape}"
    
    # Check input_static shape
    input_static = batch['input_static']
    expected_static_shape = (batch_size, num_static_channels, chip_size, chip_size)
    assert input_static.shape == expected_static_shape, \
        f"input_static has shape {input_static.shape}, expected {expected_static_shape}"
    
    # Check lonlat shape (can be either [B, H, W, 2] or [B, H*W, 2])
    lonlat = batch['lonlat']
    # Accept both formats
    valid_shapes = [
        (batch_size, chip_size, chip_size, 2),
        (batch_size, chip_size * chip_size, 2)
    ]
    assert lonlat.shape in valid_shapes, \
        f"lonlat has shape {lonlat.shape}, expected one of {valid_shapes}"
    
    print(f"✓ Input shapes correct:")
    print(f"  - input_dynamic: {input_dynamic.shape}")
    print(f"  - input_static: {input_static.shape}")
    print(f"  - lonlat: {lonlat.shape}")


def test_dataloader_targets_different():
    """Test that targets for different horizons are actually different."""
    train_loader = get_dataloader(
        batch_size=2,
        chip_size=128,
        timesteps=3,
        mode="random",
        chips_per_epoch=5,
        num_workers=0,
        include_components=True
    )
    
    batch = next(iter(train_loader))
    
    # Get all targets
    targets = {
        '5yr': batch['target_5yr'],
        '10yr': batch['target_10yr'],
        '15yr': batch['target_15yr'],
        '20yr': batch['target_20yr']
    }
    
    # Check that targets are different from each other
    horizons = list(targets.keys())
    for i in range(len(horizons)):
        for j in range(i+1, len(horizons)):
            h1, h2 = horizons[i], horizons[j]
            # Compute difference (ignoring NaNs)
            diff = torch.abs(targets[h1] - targets[h2])
            valid_diff = diff[torch.isfinite(diff)]
            if valid_diff.numel() > 0:
                mean_diff = valid_diff.mean().item()
                # Targets should be different (not identical)
                assert mean_diff > 1e-6, f"Targets for {h1} and {h2} are too similar"
    
    print(f"✓ Targets for different horizons are independent")


def test_dataloader_consistency():
    """Test that dataloader produces consistent batches."""
    train_loader = get_dataloader(
        batch_size=2,
        chip_size=128,
        timesteps=3,
        mode="random",
        chips_per_epoch=10,
        num_workers=0,
        include_components=True
    )
    
    # Get multiple batches
    batches = []
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        batches.append(batch)
    
    # Check that all batches have same structure
    first_keys = set(batches[0].keys())
    for i, batch in enumerate(batches[1:], 1):
        assert set(batch.keys()) == first_keys, \
            f"Batch {i} has different keys than batch 0"
    
    print(f"✓ Dataloader produces consistent batches across iterations")


if __name__ == "__main__":
    print("Running multi-horizon dataloader tests...\n")
    test_dataloader_output_keys()
    test_dataloader_target_shapes()
    test_dataloader_input_shapes()
    test_dataloader_targets_different()
    test_dataloader_consistency()
    print("\n✅ All dataloader tests passed!")
