"""
Test multi-horizon SpatioTemporalPredictor model.
"""
import torch
import pytest
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor


def test_model_output_shape():
    """Test that model outputs correct shape for 4 horizons."""
    batch_size = 2
    timesteps = 3
    height, width = 64, 64
    num_dynamic_channels = 11  # 1 HM + 10 covariates
    num_static_channels = 7
    
    # Create model
    model = SpatioTemporalPredictor(
        hidden_dim=16,
        kernel_size=3,
        num_layers=1,
        num_static_channels=num_static_channels,
        num_dynamic_channels=num_dynamic_channels,
        use_location_encoder=False  # Disable for simpler test
    )
    
    # Create dummy inputs
    input_dynamic = torch.randn(batch_size, timesteps, num_dynamic_channels, height, width)
    input_static = torch.randn(batch_size, num_static_channels, height, width)
    
    # Forward pass
    output = model(input_dynamic, input_static, lonlat=None)
    
    # Check output shape
    assert output.shape == (batch_size, 4, height, width), \
        f"Expected shape ({batch_size}, 4, {height}, {width}), got {output.shape}"
    
    print(f"✓ Model output shape correct: {output.shape}")


def test_model_with_location_encoder():
    """Test model with location encoder enabled."""
    batch_size = 2
    timesteps = 3
    height, width = 32, 32
    num_dynamic_channels = 11
    num_static_channels = 7
    
    # Create model with location encoder
    model = SpatioTemporalPredictor(
        hidden_dim=16,
        kernel_size=3,
        num_layers=1,
        num_static_channels=num_static_channels,
        num_dynamic_channels=num_dynamic_channels,
        use_location_encoder=True,
        locenc_out_channels=8
    )
    
    # Create dummy inputs
    input_dynamic = torch.randn(batch_size, timesteps, num_dynamic_channels, height, width)
    input_static = torch.randn(batch_size, num_static_channels, height, width)
    lonlat = torch.randn(batch_size, height, width, 2)  # Random lon/lat
    
    # Forward pass
    output = model(input_dynamic, input_static, lonlat=lonlat)
    
    # Check output shape
    assert output.shape == (batch_size, 4, height, width), \
        f"Expected shape ({batch_size}, 4, {height}, {width}), got {output.shape}"
    
    print(f"✓ Model with location encoder output shape correct: {output.shape}")


def test_model_gradients():
    """Test that gradients flow through all 4 prediction heads."""
    batch_size = 2
    timesteps = 3
    height, width = 32, 32
    num_dynamic_channels = 11
    num_static_channels = 7
    
    # Create model
    model = SpatioTemporalPredictor(
        hidden_dim=16,
        kernel_size=3,
        num_layers=1,
        num_static_channels=num_static_channels,
        num_dynamic_channels=num_dynamic_channels,
        use_location_encoder=False
    )
    
    # Create dummy inputs
    input_dynamic = torch.randn(batch_size, timesteps, num_dynamic_channels, height, width)
    input_static = torch.randn(batch_size, num_static_channels, height, width)
    
    # Forward pass
    output = model(input_dynamic, input_static, lonlat=None)
    
    # Create dummy targets for each horizon
    targets = torch.randn(batch_size, 4, height, width)
    
    # Compute loss for each horizon
    loss = torch.nn.functional.mse_loss(output, targets)
    
    # Backward pass
    loss.backward()
    
    # Check that all heads have gradients
    for i, head in enumerate(model.heads):
        for name, param in head.named_parameters():
            assert param.grad is not None, f"Head {i} parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Head {i} parameter {name} has NaN gradients"
    
    print(f"✓ Gradients flow through all 4 prediction heads")


def test_model_independent_heads():
    """Test that each prediction head produces different outputs."""
    batch_size = 2
    timesteps = 3
    height, width = 32, 32
    num_dynamic_channels = 11
    num_static_channels = 7
    
    # Create model
    model = SpatioTemporalPredictor(
        hidden_dim=16,
        kernel_size=3,
        num_layers=1,
        num_static_channels=num_static_channels,
        num_dynamic_channels=num_dynamic_channels,
        use_location_encoder=False
    )
    
    # Create dummy inputs
    input_dynamic = torch.randn(batch_size, timesteps, num_dynamic_channels, height, width)
    input_static = torch.randn(batch_size, num_static_channels, height, width)
    
    # Forward pass
    output = model(input_dynamic, input_static, lonlat=None)
    
    # Check that predictions for different horizons are different
    # (they should be since heads are initialized randomly)
    for i in range(4):
        for j in range(i+1, 4):
            diff = torch.abs(output[:, i] - output[:, j]).mean()
            assert diff > 1e-6, f"Predictions for horizon {i} and {j} are too similar"
    
    print(f"✓ Each prediction head produces independent outputs")


if __name__ == "__main__":
    print("Running multi-horizon model tests...\n")
    test_model_output_shape()
    test_model_with_location_encoder()
    test_model_gradients()
    test_model_independent_heads()
    print("\n✅ All model tests passed!")
