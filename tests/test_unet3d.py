"""
Test UNet3D as a drop-in replacement for ConvLSTM.
"""
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.models.unet3d import UNet3D, TemporalProcessorType
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor


def test_unet3d_forward():
    """Test UNet3D forward pass with correct input/output shapes."""
    batch_size = 2
    timesteps = 3
    in_channels = 18  # 11 dynamic + 7 static
    height, width = 64, 64
    hidden_dim = 16
    
    # Create model
    model = UNet3D(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        kernel_size=3,
        layers=2,
    )
    
    # Create dummy input: [B, C, T, H, W]
    x = torch.randn(batch_size, in_channels, timesteps, height, width)
    
    # Forward pass
    output = model(x)
    
    # Check output shape: [B, hidden_dim, H, W]
    assert output.shape == (batch_size, hidden_dim, height, width), \
        f"Expected shape ({batch_size}, {hidden_dim}, {height}, {width}), got {output.shape}"
    
    print(f"✓ UNet3D output shape correct: {output.shape}")


def test_unet3d_vs_convlstm_spatiotemporal():
    """Test that UNet3D produces same output shape as ConvLSTM in SpatioTemporalPredictor."""
    batch_size = 2
    timesteps = 3
    num_dynamic_channels = 11
    num_static_channels = 7
    height, width = 64, 64
    hidden_dim = 16
    
    # Create inputs
    input_dynamic = torch.randn(batch_size, timesteps, num_dynamic_channels, height, width)
    input_static = torch.randn(batch_size, num_static_channels, height, width)
    
    # Test with ConvLSTM
    model_convlstm = SpatioTemporalPredictor(
        hidden_dim=hidden_dim,
        num_static_channels=num_static_channels,
        num_dynamic_channels=num_dynamic_channels,
        num_layers=1,
        kernel_size=3,
        use_location_encoder=False,
        temporal_processor="convlstm",
    )
    output_convlstm = model_convlstm(input_dynamic, input_static, lonlat=None)
    
    # Test with UNet3D
    model_unet3d = SpatioTemporalPredictor(
        hidden_dim=hidden_dim,
        num_static_channels=num_static_channels,
        num_dynamic_channels=num_dynamic_channels,
        num_layers=1,
        kernel_size=3,
        use_location_encoder=False,
        temporal_processor="unet3d",
    )
    output_unet3d = model_unet3d(input_dynamic, input_static, lonlat=None)
    
    # Both should produce same output shape: [B, 12, H, W] (4 horizons × 3 predictions)
    assert output_convlstm.shape == output_unet3d.shape, \
        f"Shape mismatch: ConvLSTM {output_convlstm.shape} vs UNet3D {output_unet3d.shape}"
    assert output_convlstm.shape == (batch_size, 12, height, width), \
        f"Expected shape ({batch_size}, 12, {height}, {width}), got {output_convlstm.shape}"
    
    print(f"✓ ConvLSTM output shape: {output_convlstm.shape}")
    print(f"✓ UNet3D output shape: {output_unet3d.shape}")
    print(f"✓ Both models produce same output shape")


def test_unet3d_with_temporal_processor():
    """Test UNet3D with LSTM temporal processor on skip connections."""
    batch_size = 2
    timesteps = 3
    in_channels = 18
    height, width = 64, 64
    hidden_dim = 16
    
    # Create model with LSTM temporal processor
    model = UNet3D(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        kernel_size=3,
        layers=2,
        temporal_processor_type="lstm",
        temporal_layers=1,
    )
    
    # Create dummy input: [B, C, T, H, W]
    x = torch.randn(batch_size, in_channels, timesteps, height, width)
    
    # Forward pass
    output = model(x)
    
    # Check output shape: [B, hidden_dim, H, W]
    assert output.shape == (batch_size, hidden_dim, height, width), \
        f"Expected shape ({batch_size}, {hidden_dim}, {height}, {width}), got {output.shape}"
    
    print(f"✓ UNet3D with LSTM temporal processor output shape correct: {output.shape}")


def test_unet3d_temporal_processor_in_predictor():
    """Test UNet3D with temporal processor in SpatioTemporalPredictor."""
    batch_size = 2
    timesteps = 3
    num_dynamic_channels = 11
    num_static_channels = 7
    height, width = 64, 64
    hidden_dim = 16
    
    # Create inputs
    input_dynamic = torch.randn(batch_size, timesteps, num_dynamic_channels, height, width)
    input_static = torch.randn(batch_size, num_static_channels, height, width)
    
    # Test with UNet3D + LSTM temporal processor
    model_unet3d_lstm = SpatioTemporalPredictor(
        hidden_dim=hidden_dim,
        num_static_channels=num_static_channels,
        num_dynamic_channels=num_dynamic_channels,
        num_layers=1,
        kernel_size=3,
        use_location_encoder=False,
        temporal_processor="unet3d",
        unet3d_temporal_processor="lstm",
        unet3d_temporal_layers=1,
    )
    output_lstm = model_unet3d_lstm(input_dynamic, input_static, lonlat=None)
    
    # Test with UNet3D without temporal processor
    model_unet3d_none = SpatioTemporalPredictor(
        hidden_dim=hidden_dim,
        num_static_channels=num_static_channels,
        num_dynamic_channels=num_dynamic_channels,
        num_layers=1,
        kernel_size=3,
        use_location_encoder=False,
        temporal_processor="unet3d",
        unet3d_temporal_processor="none",
    )
    output_none = model_unet3d_none(input_dynamic, input_static, lonlat=None)
    
    # Both should produce same output shape: [B, 12, H, W]
    assert output_lstm.shape == output_none.shape, \
        f"Shape mismatch: LSTM {output_lstm.shape} vs None {output_none.shape}"
    assert output_lstm.shape == (batch_size, 12, height, width), \
        f"Expected shape ({batch_size}, 12, {height}, {width}), got {output_lstm.shape}"
    
    print(f"✓ UNet3D with LSTM temporal processor output shape: {output_lstm.shape}")
    print(f"✓ UNet3D without temporal processor output shape: {output_none.shape}")
    print(f"✓ Both produce identical output structure")


if __name__ == "__main__":
    test_unet3d_forward()
    test_unet3d_vs_convlstm_spatiotemporal()
    test_unet3d_with_temporal_processor()
    test_unet3d_temporal_processor_in_predictor()
    print("\n✓ All tests passed!")
