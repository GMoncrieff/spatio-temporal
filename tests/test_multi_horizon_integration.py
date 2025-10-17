"""
Integration test for multi-horizon training loop.
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.torchgeo_dataloader import get_dataloader
from src.models.lightning_module import SpatioTemporalLightningModule
import pytorch_lightning as pl


def test_training_loop_integration():
    """Test that a complete training loop works with multi-horizon setup."""
    print("Testing multi-horizon training loop integration...")
    
    # Create small dataloaders
    train_loader = get_dataloader(
        batch_size=2,
        chip_size=64,
        timesteps=3,
        mode="random",
        chips_per_epoch=5,
        num_workers=0,
        include_components=True
    )
    
    val_loader = get_dataloader(
        batch_size=2,
        chip_size=64,
        timesteps=3,
        mode="random",
        chips_per_epoch=3,
        num_workers=0,
        include_components=True
    )
    
    # Create model
    model = SpatioTemporalLightningModule(
        hidden_dim=16,
        lr=1e-3,
        num_static_channels=7,
        num_dynamic_channels=11,
        num_layers=1,
        kernel_size=3,
        use_location_encoder=False,  # Disable for faster testing
        ssim_weight=2.0,
        laplacian_weight=1.0,
        histogram_weight=0.67,
        histogram_warmup_epochs=0  # No warmup for testing
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='cpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False
    )
    
    # Run training
    try:
        trainer.fit(model, train_loader, val_loader)
        print("✓ Training loop completed successfully")
        return True
    except Exception as e:
        print(f"✗ Training loop failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass_shapes():
    """Test that forward pass produces correct shapes."""
    print("Testing forward pass shapes...")
    
    # Create model
    model = SpatioTemporalLightningModule(
        hidden_dim=16,
        lr=1e-3,
        num_static_channels=7,
        num_dynamic_channels=11,
        num_layers=1,
        kernel_size=3,
        use_location_encoder=False
    )
    
    # Create dummy batch
    batch_size = 2
    chip_size = 64
    batch = {
        'input_dynamic': torch.randn(batch_size, 3, 11, chip_size, chip_size),
        'input_static': torch.randn(batch_size, 7, chip_size, chip_size),
        'target_5yr': torch.randn(batch_size, chip_size, chip_size),
        'target_10yr': torch.randn(batch_size, chip_size, chip_size),
        'target_15yr': torch.randn(batch_size, chip_size, chip_size),
        'target_20yr': torch.randn(batch_size, chip_size, chip_size),
        'lonlat': torch.randn(batch_size, chip_size * chip_size, 2)
    }
    
    # Run training step
    model.train()
    loss = model.training_step(batch, 0)
    
    assert loss.numel() == 1, f"Loss should be scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    
    print(f"✓ Forward pass produces valid loss: {loss.item():.6f}")
    return True


def test_validation_step():
    """Test that validation step works."""
    print("Testing validation step...")
    
    # Create model
    model = SpatioTemporalLightningModule(
        hidden_dim=16,
        lr=1e-3,
        num_static_channels=7,
        num_dynamic_channels=11,
        num_layers=1,
        kernel_size=3,
        use_location_encoder=False
    )
    
    # Create dummy batch
    batch_size = 2
    chip_size = 64
    batch = {
        'input_dynamic': torch.randn(batch_size, 3, 11, chip_size, chip_size),
        'input_static': torch.randn(batch_size, 7, chip_size, chip_size),
        'target_5yr': torch.randn(batch_size, chip_size, chip_size),
        'target_10yr': torch.randn(batch_size, chip_size, chip_size),
        'target_15yr': torch.randn(batch_size, chip_size, chip_size),
        'target_20yr': torch.randn(batch_size, chip_size, chip_size),
        'lonlat': torch.randn(batch_size, chip_size * chip_size, 2)
    }
    
    # Run validation step
    model.eval()
    with torch.no_grad():
        loss = model.validation_step(batch, 0)
    
    assert loss.numel() == 1, f"Loss should be scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    
    print(f"✓ Validation step produces valid loss: {loss.item():.6f}")
    return True


if __name__ == "__main__":
    print("Running multi-horizon integration tests...\n")
    
    results = []
    results.append(("Forward pass shapes", test_forward_pass_shapes()))
    results.append(("Validation step", test_validation_step()))
    results.append(("Training loop integration", test_training_loop_integration()))
    
    print("\n" + "="*60)
    print("INTEGRATION TEST RESULTS")
    print("="*60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some integration tests failed!")
        sys.exit(1)
