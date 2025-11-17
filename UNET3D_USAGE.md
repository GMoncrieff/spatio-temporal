# 3D UNet as ConvLSTM Replacement

## Overview

The `UNet3D` class is now available as a drop-in replacement for `ConvLSTM` in the `SpatioTemporalPredictor`. Both models accept the same input format and produce identical output shapes, making it easy to experiment with different temporal processors.

**New Feature:** UNet3D now supports optional LSTM/GRU temporal processing on skip connections between encoder and decoder layers, following the ThemedaUNet design pattern.

## Architecture Comparison

| Aspect | ConvLSTM | UNet3D |
|--------|----------|--------|
| **Type** | Recurrent (processes time sequentially) | Feedforward (processes all time steps together) |
| **Input** | [B, T, C, H, W] | [B, C, T, H, W] (permuted internally) |
| **Output** | [B, hidden_dim, H, W] | [B, hidden_dim, H, W] |
| **Temporal Handling** | Sequential LSTM cells | 3D convolutions with skip connections |
| **Spatial Downsampling** | 2x per layer | 2x per layer (spatial only, T preserved) |
| **Memory Usage** | Lower (sequential) | Higher (all timesteps in memory) |
| **Inference Speed** | Slower (sequential) | Faster (parallel) |

## Usage

### In Python Code

```python
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor

# Using ConvLSTM (default)
model_convlstm = SpatioTemporalPredictor(
    hidden_dim=16,
    num_dynamic_channels=11,
    num_static_channels=7,
    num_layers=1,
    kernel_size=3,
    temporal_processor="convlstm"  # default
)

# Using UNet3D without temporal processing
model_unet3d = SpatioTemporalPredictor(
    hidden_dim=16,
    num_dynamic_channels=11,
    num_static_channels=7,
    num_layers=1,
    kernel_size=3,
    temporal_processor="unet3d",
    unet3d_temporal_processor="none"
)

# Using UNet3D with LSTM temporal processing on skip connections
model_unet3d_lstm = SpatioTemporalPredictor(
    hidden_dim=16,
    num_dynamic_channels=11,
    num_static_channels=7,
    num_layers=1,
    kernel_size=3,
    temporal_processor="unet3d",
    unet3d_temporal_processor="lstm",
    unet3d_temporal_layers=1
)

# Using UNet3D with GRU temporal processing
model_unet3d_gru = SpatioTemporalPredictor(
    hidden_dim=16,
    num_dynamic_channels=11,
    num_static_channels=7,
    num_layers=1,
    kernel_size=3,
    temporal_processor="unet3d",
    unet3d_temporal_processor="gru",
    unet3d_temporal_layers=2
)

# Both produce identical output shapes
input_dynamic = torch.randn(2, 3, 11, 64, 64)  # [B, T, C_d, H, W]
input_static = torch.randn(2, 7, 64, 64)       # [B, C_s, H, W]

output_convlstm = model_convlstm(input_dynamic, input_static)  # [2, 12, 64, 64]
output_unet3d = model_unet3d(input_dynamic, input_static)      # [2, 12, 64, 64]
```

### In Lightning Training

**UNet3D is now the default model.** Simply use the existing parameters:

```bash
# Default: UNet3D with LSTM on skip connections
python scripts/train_lightning.py \
    --hidden_dim 64 \
    --num_layers 2 \
    --kernel_size 3

# Customize architecture (all params map to UNet3D)
python scripts/train_lightning.py \
    --hidden_dim 32 \
    --num_layers 3 \
    --kernel_size 5

# With other options
python scripts/train_lightning.py \
    --hidden_dim 64 \
    --num_layers 2 \
    --kernel_size 3 \
    --use_location_encoder true \
    --locenc_out_channels 8 \
    --max_epochs 100
```

**Parameter Mapping:**
- `--hidden_dim`: UNet3D hidden dimension (encoder/decoder channels)
- `--num_layers`: UNet3D number of encoder/decoder layers
- `--kernel_size`: UNet3D convolution kernel size
- LSTM on skip connections: Fixed at 1 layer (cannot be changed from CLI)

### In Lightning Module

```python
from src.models.lightning_module import SpatioTemporalLightningModule

# Default: UNet3D with LSTM on skip connections
# (temporal processor is fixed, cannot be changed)
model = SpatioTemporalLightningModule(
    hidden_dim=64,
    num_dynamic_channels=11,
    num_static_channels=7,
    num_layers=2,
    kernel_size=3,
    use_location_encoder=True,
    locenc_out_channels=8,
)

# Customize architecture
model = SpatioTemporalLightningModule(
    hidden_dim=32,
    num_dynamic_channels=11,
    num_static_channels=7,
    num_layers=3,
    kernel_size=5,
)
```

**Note:** UNet3D is hardcoded as the default temporal processor with LSTM on skip connections (1 layer). To use ConvLSTM instead, modify `lightning_module.py` line 62-64.

## Key Design Decisions

### Temporal Dimension Handling

The UNet3D **preserves the temporal dimension throughout** the encoder-decoder path:

- **Encoder:** Downsamples spatial dimensions (H, W) by 2x per layer, keeps T constant
- **Decoder:** Upsamples spatial dimensions (H, W) by 2x per layer, keeps T constant
- **Temporal Processing:** Optional LSTM/GRU on skip connections processes temporal dynamics at each spatial location
- **Output:** Takes the last timestep: `output[:, :, -1, :, :]` → [B, hidden_dim, H, W]

This design allows the model to:
1. Process all timesteps together (unlike sequential LSTM)
2. Maintain temporal context through skip connections
3. Add temporal dynamics via LSTM/GRU on skip connections (optional)
4. Avoid temporal aliasing from aggressive downsampling

### Temporal Processing on Skip Connections

When enabled, LSTM/GRU processors operate on skip connections following ThemedaUNet design:

- Each encoder layer has its own temporal processor instance
- Temporal processors reshape skip connections: [B, C, T, H, W] → [B*H*W, T, C]
- Process temporal dynamics at each spatial pixel independently
- Reshape back: [B*H*W, T, C] → [B, C, T, H, W]
- Processed skip connections feed into decoder layers

This allows the model to:
1. Learn temporal patterns at each spatial location
2. Preserve spatial structure while adding temporal context
3. Decouple temporal and spatial processing

### Architecture Components

**ResBlock3D:** Residual block with optional spatial downsampling
- Stride=(1, 2, 2) for downsampling (no temporal downsampling)
- Reflect padding for better edge handling
- Two conv layers with ReLU activation

**DownBlock3D:** Encoder block
- Two ResBlocks: one with downsampling, one without
- Progressively increases channels

**UpBlock3D:** Decoder block
- Transposed convolution with spatial upsampling only
- One ResBlock after upsampling
- Skip connection from corresponding encoder layer

**Temporal Processors:** LSTM or GRU cells
- One per encoder layer
- Input size = encoder layer input channels
- Hidden size = encoder layer input channels
- Batch-first processing on [B*H*W, T, C] tensors

## Output Format

Both models produce identical output: [B, 12, H, W]

```
Channel layout (4 horizons × 3 predictions):
- Channels [0, 3, 6, 9]:   Lower 2.5% quantile
- Channels [1, 4, 7, 10]:  Central prediction (MSE+SSIM+Lap+Hist optimized)
- Channels [2, 5, 8, 11]:  Upper 97.5% quantile

Horizons: 5yr, 10yr, 15yr, 20yr
```

## Testing

Run the test suite to verify both models work correctly:

```bash
cd /Users/glen.moncrieff/python/spatio_temporal
KMP_DUPLICATE_LIB_OK=TRUE python tests/test_unet3d.py
```

Expected output:
```
✓ UNet3D output shape correct: torch.Size([2, 16, 64, 64])
✓ ConvLSTM output shape: torch.Size([2, 12, 64, 64])
✓ UNet3D output shape: torch.Size([2, 12, 64, 64])
✓ Both models produce same output shape

✓ All tests passed!
```

## Performance Considerations

### Memory Usage
- **ConvLSTM:** Lower (processes one timestep at a time)
- **UNet3D:** Higher (keeps all timesteps in memory)

For typical training with T=3 timesteps, this is usually not a concern.

### Inference Speed
- **ConvLSTM:** Slower (sequential processing, T iterations)
- **UNet3D:** Faster (parallel processing, single forward pass)

### Gradient Flow
Both models support the same gradient flow design:
- Central loss (MSE+SSIM+Lap+Hist) → backbone + central_heads
- Pinball loss → quantile_heads only (NOT backbone)

## Troubleshooting

### Error: "Unknown temporal_processor: ..."
Make sure you use lowercase: `temporal_processor="unet3d"` or `temporal_processor="convlstm"`

### Shape mismatch errors
Verify input shapes:
- `input_dynamic`: [B, T, C_d, H, W]
- `input_static`: [B, C_s, H, W]
- Model output: [B, 12, H, W]

### OMP error on macOS
If you see "OMP: Error #15: Initializing libomp.dylib", run with:
```bash
KMP_DUPLICATE_LIB_OK=TRUE python your_script.py
```
