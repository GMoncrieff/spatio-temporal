"""
Test script to verify both ConvLSTM and Swin-UNet models work correctly.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.models.lightning_module import SpatioTemporalLightningModule


def test_model(model_type: str):
    """Test a model with dummy inputs."""
    print(f"\n{'='*60}")
    print(f"Testing {model_type.upper()} model")
    print(f"{'='*60}")
    
    # Create model (use different kernel sizes for different models)
    # ConvLSTM works best with odd kernel sizes (3, 5, 7)
    # Swin-UNet uses kernel_size as window_size (4, 8, 16)
    kernel_size = 3 if model_type == "convlstm" else 4
    
    model = SpatioTemporalLightningModule(
        hidden_dim=64 if model_type == "convlstm" else 96,
        num_static_channels=5,
        num_dynamic_channels=9,
        num_layers=2,
        kernel_size=kernel_size,
        use_location_encoder=True,
        locenc_out_channels=8,
        model_type=model_type,
    )
    
    print(f"Model created successfully!")
    
    # Create dummy inputs matching dataloader output
    B, T, C_dyn, H, W = 2, 3, 9, 128, 128
    C_static = 5
    
    input_dynamic = torch.randn(B, T, C_dyn, H, W)
    input_static = torch.randn(B, C_static, H, W)
    lonlat = torch.randn(B, H, W, 2)
    
    print(f"Input shapes:")
    print(f"  Dynamic: {list(input_dynamic.shape)}")
    print(f"  Static: {list(input_static.shape)}")
    print(f"  Lonlat: {list(lonlat.shape)}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_dynamic, input_static, lonlat=lonlat)
    
    print(f"Output shape: {list(output.shape)}")
    
    # Check output shape
    expected_shape = [B, 1, H, W]
    assert list(output.shape) == expected_shape, f"Expected {expected_shape}, got {list(output.shape)}"
    
    print(f"✓ Output shape is correct!")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters:")
    print(f"  Total: {num_params:,}")
    print(f"  Trainable: {num_trainable:,}")
    
    print(f"✓ {model_type.upper()} test passed!\n")
    
    return model


def test_variable_sizes():
    """Test Swin-UNet with variable input sizes (for inference edge tiles)."""
    print(f"\n{'='*60}")
    print("Testing SWIN-UNET with variable input sizes")
    print(f"{'='*60}")
    
    model = SpatioTemporalLightningModule(
        hidden_dim=96,
        num_static_channels=5,
        num_dynamic_channels=9,
        num_layers=2,
        kernel_size=4,
        use_location_encoder=True,
        locenc_out_channels=8,
        model_type="swin_unet",
    )
    model.eval()
    
    # Test different sizes (common during inference edge tiles)
    test_sizes = [(128, 128), (128, 99), (100, 128), (64, 64), (50, 75)]
    
    for H, W in test_sizes:
        B, T, C_dyn = 1, 3, 9
        C_static = 5
        
        input_dynamic = torch.randn(B, T, C_dyn, H, W)
        input_static = torch.randn(B, C_static, H, W)
        lonlat = torch.randn(B, H, W, 2)
        
        with torch.no_grad():
            output = model(input_dynamic, input_static, lonlat=lonlat)
        
        expected_shape = [B, 1, H, W]
        assert list(output.shape) == expected_shape, f"Size {H}×{W}: Expected {expected_shape}, got {list(output.shape)}"
        print(f"  ✓ Size {H}×{W}: Output shape correct")
    
    print(f"✓ Variable size test passed!\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Model Architecture Test Suite")
    print("="*60)
    
    # Test both models
    try:
        convlstm_model = test_model("convlstm")
        swin_model = test_model("swin_unet")
        
        # Test variable sizes for Swin-UNet (important for inference)
        test_variable_sizes()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nBoth models are working correctly and can be switched via --model_type flag.")
        print("Swin-UNet handles variable input sizes (with padding) for inference edge tiles.")
        print("\nUsage examples:")
        print("  python scripts/train_lightning.py --model_type convlstm")
        print("  python scripts/train_lightning.py --model_type swin_unet --hidden_dim 96 --kernel_size 4")
        print()
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
