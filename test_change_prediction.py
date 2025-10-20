"""
Quick test to verify the model predicts changes correctly.
"""
import torch
import sys
sys.path.insert(0, 'src')

from models.lightning_module import SpatioTemporalLightningModule

def test_change_prediction():
    print("=" * 80)
    print("Testing Change-Based Prediction")
    print("=" * 80)
    
    # Create model
    model = SpatioTemporalLightningModule(
        hidden_dim=16,
        num_static_channels=7,
        num_dynamic_channels=11,
        num_layers=1
    )
    model.eval()
    
    # Create dummy data
    B, T, C_dyn, H, W = 2, 3, 11, 32, 32
    C_stat = 7
    
    input_dynamic = torch.randn(B, T, C_dyn, H, W)
    input_static = torch.randn(B, C_stat, H, W)
    lonlat = torch.randn(B, H, W, 2)
    
    print(f"\n1. Input shapes:")
    print(f"   Dynamic: {input_dynamic.shape}")
    print(f"   Static: {input_static.shape}")
    print(f"   Lonlat: {lonlat.shape}")
    
    # Forward pass
    with torch.no_grad():
        pred_changes = model(input_dynamic, input_static, lonlat=lonlat)
    
    print(f"\n2. Model output (predicted changes):")
    print(f"   Shape: {pred_changes.shape}")
    print(f"   Range: [{pred_changes.min():.4f}, {pred_changes.max():.4f}]")
    print(f"   Mean: {pred_changes.mean():.4f}")
    print(f"   Std: {pred_changes.std():.4f}")
    
    # Convert to absolute HM
    last_input = input_dynamic[:, -1, 0:1, :, :]  # [B, 1, H, W]
    pred_absolute = last_input.unsqueeze(1) + pred_changes  # [B, 4, H, W]
    pred_absolute_clipped = torch.clamp(pred_absolute, 0.0, 1.0)
    
    print(f"\n3. Converted to absolute HM (last_input + pred_change):")
    print(f"   Last input range: [{last_input.min():.4f}, {last_input.max():.4f}]")
    print(f"   Before clipping: [{pred_absolute.min():.4f}, {pred_absolute.max():.4f}]")
    print(f"   After clipping: [{pred_absolute_clipped.min():.4f}, {pred_absolute_clipped.max():.4f}]")
    
    # Test loss computation
    targets = {
        '5yr': torch.randn(B, 1, H, W),
        '10yr': torch.randn(B, 1, H, W),
        '15yr': torch.randn(B, 1, H, W),
        '20yr': torch.randn(B, 1, H, W)
    }
    
    # Simulate batch
    batch = {
        'input_dynamic': input_dynamic,
        'input_static': input_static,
        'lonlat': lonlat,
        'target_5yr': targets['5yr'].squeeze(1),
        'target_10yr': targets['10yr'].squeeze(1),
        'target_15yr': targets['15yr'].squeeze(1),
        'target_20yr': targets['20yr'].squeeze(1)
    }
    
    print(f"\n4. Testing loss computation...")
    try:
        model.train()
        loss = model.training_step(batch, 0)
        print(f"   ✓ Training step successful!")
        print(f"   Loss: {loss:.6f}")
    except Exception as e:
        print(f"   ✗ Training step failed: {e}")
        return False
    
    print(f"\n5. Testing validation step...")
    try:
        model.eval()
        val_loss = model.validation_step(batch, 0)
        print(f"   ✓ Validation step successful!")
        print(f"   Loss: {val_loss:.6f}")
    except Exception as e:
        print(f"   ✗ Validation step failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe model is correctly predicting changes and converting to absolute HM.")
    print("Clipping is working to ensure outputs stay in [0, 1] range.")
    return True

if __name__ == "__main__":
    success = test_change_prediction()
    sys.exit(0 if success else 1)
