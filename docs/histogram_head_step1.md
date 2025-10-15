# Histogram Prediction Head - Step 1 Implementation

## Overview

This implements **Step 1** of the histogram prediction head: adding a second head that predicts the tile-level distribution of change magnitudes.

## Architecture

### Model Changes (`src/models/spatiotemporal_predictor.py`)

1. **New Parameters**:
   - `use_histogram_head`: Enable/disable histogram prediction
   - `histogram_bins`: Custom bin edges (default: `[-1, -0.05, -0.005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1]`)

2. **Histogram Head Components**:
   - **Tile Embedding**: Mean pooling over spatial dimensions `[B, hidden_dim, H, W] → [B, hidden_dim]`
   - **MLP**: 3-layer network with dropout
     ```
     hidden_dim → hidden_dim*2 → hidden_dim → num_bins
     ```
   - **Output**: Softmax over bins giving probability distribution `p_hat`

3. **Forward Pass**:
   - Without histogram head: Returns `pred` (pixel predictions)
   - With histogram head: Returns `(pred, hist_probs)` tuple

### Loss Function (`src/models/histogram_loss.py`)

Combines two loss terms:

1. **Cross-Entropy**: `CE(p_obs, p_hat)`
   - Standard classification loss
   - Treats histogram as a categorical distribution

2. **Wasserstein-2 Distance**: `W2(p_obs, p_hat)`
   - Metric on probability distributions
   - Uses squared distances between bin midpoints
   - Sensitive to how far apart distributions are
   - For 1D ordered bins: `W2² = Σ(CDF_obs - CDF_hat)² * bin_width`

**Total Loss**: `L_hist = CE + λ_w2 * W2`

### Observed Histogram Computation

Function: `compute_observed_histogram(changes, bin_edges, mask=None)`

- Takes continuous pixel-level changes `[B, H, W]`
- Bins them into histogram using provided edges
- Applies optional validity mask
- Returns normalized proportions `p_obs [B, num_bins]`

## Training Integration (`src/models/lightning_module.py`)

### New Hyperparameters

- `use_histogram_head`: Enable histogram head (default: `False`)
- `histogram_bins`: Custom bin edges (default: `None` uses model default)
- `histogram_weight`: Weight for histogram loss in total loss (default: `0.5`)
- `histogram_lambda_w2`: Weight for W2 term within histogram loss (default: `0.1`)

### Training/Validation Steps

1. **Forward pass** returns `(preds, hist_probs)` when enabled
2. **Compute observed histogram** from ground truth changes:
   ```python
   delta_true = target - last_input  # [B, 1, H, W]
   p_obs = compute_observed_histogram(delta_true, bins, mask)
   ```
3. **Compute histogram loss**:
   ```python
   hist_loss, ce_loss, w2_loss = histogram_loss_fn(p_obs, hist_probs)
   ```
4. **Add to total loss**:
   ```python
   total_loss = pixel_loss + ssim_loss + lap_loss + histogram_weight * hist_loss
   ```

### Logged Metrics

- `train_hist_ce`: Cross-entropy component
- `train_hist_w2`: Wasserstein-2 component  
- `train_hist_loss`: Total histogram loss
- `val_hist_ce`, `val_hist_w2`, `val_hist_loss`: Validation equivalents

## Default Bin Configuration

**Bin Edges**: `[-1, -0.05, -ε, +ε, 0.02, 0.05, 0.1, 0.2, 0.5, 1]` where `ε = 0.005`

**Bin Midpoints** (for Wasserstein distance):
```
[-0.525, -0.0275, -0.0025, 0.0125, 0.035, 0.075, 0.15, 0.35, 0.75]
```

**Interpretation**:
- Bins capture different magnitudes of change
- Small ε separates near-zero changes from true zeros
- Logarithmic-like spacing for larger changes

## Usage

### Enable During Training

```bash
python scripts/train_lightning.py \
  --use_histogram_head true \
  --histogram_weight 0.5 \
  --histogram_lambda_w2 0.1 \
  --hidden_dim 64 \
  --num_layers 4
```

### In Code

```python
from src.models.lightning_module import SpatioTemporalLightningModule

model = SpatioTemporalLightningModule(
    hidden_dim=64,
    num_layers=4,
    use_histogram_head=True,
    histogram_weight=0.5,
    histogram_lambda_w2=0.1,
)
```

## Testing

Run tests:
```bash
python tests/test_histogram_head.py
```

Tests verify:
- ✓ Histogram head produces correct output shapes
- ✓ Probabilities sum to 1
- ✓ Loss computation works correctly
- ✓ Observed histogram computation handles masks

## Next Steps

**Step 2**: Tie pixel and histogram heads together by comparing predicted histogram to the distribution implied by pixel predictions (via differentiable binning).

**Step 3**: Add consistency regularization to ensure local (pixel) and global (tile) estimates are self-consistent.

## Files Modified/Created

### Created:
- `src/models/histogram_loss.py` - Loss functions for histogram head
- `tests/test_histogram_head.py` - Unit tests
- `docs/histogram_head_step1.md` - This documentation

### Modified:
- `src/models/spatiotemporal_predictor.py` - Added histogram head architecture
- `src/models/lightning_module.py` - Integrated histogram loss into training

## Parameters Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_histogram_head` | `False` | Enable histogram prediction |
| `histogram_bins` | `None` | Custom bin edges (uses default if None) |
| `histogram_weight` | `0.5` | Weight for histogram loss |
| `histogram_lambda_w2` | `0.1` | Weight for Wasserstein-2 within histogram loss |

## Model Size Impact

With `hidden_dim=64`:
- Histogram head parameters: ~12K (64→128→64→9)
- Negligible compared to ConvLSTM backbone (~1.1M parameters)
