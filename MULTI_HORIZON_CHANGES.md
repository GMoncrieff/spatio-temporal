# Multi-Horizon Forecasting Implementation

## Overview
Converted the model from single 20-year forecast to multi-horizon forecasting (5yr, 10yr, 15yr, 20yr).

## Changes Made

### 1. Model Architecture (`src/models/spatiotemporal_predictor.py`)
**Before:** Single prediction head → `[B, 1, H, W]`
**After:** 4 parallel prediction heads → `[B, 4, H, W]`

```python
# Old
self.head = nn.Sequential(...)

# New
self.heads = nn.ModuleList([
    nn.Sequential(...) for _ in range(4)
])
```

Each horizon (5yr, 10yr, 15yr, 20yr) has its own independent prediction head.

### 2. Data Loader (`scripts/torchgeo_dataloader.py`)
**Before:** Single target `target` for year 2020
**After:** 4 targets for years 2005, 2010, 2015, 2020

```python
# Batch now contains:
{
    'input_dynamic': [B, T, C_dyn, H, W],  # T=3 (1990, 1995, 2000)
    'input_static': [B, C_static, H, W],
    'target_5yr': [B, H, W],   # 2005
    'target_10yr': [B, H, W],  # 2010
    'target_15yr': [B, H, W],  # 2015
    'target_20yr': [B, H, W],  # 2020
    'lonlat': [B, H*W, 2]
}
```

### 3. Lightning Module (`src/models/lightning_module.py`)

#### New Helper Function
Added `_compute_horizon_losses()` to compute all loss components for a single horizon:
- MSE (on deltas)
- MAE (on absolute values)
- SSIM
- Laplacian Pyramid
- Histogram (W2 with rarity weighting)

#### Training Step
```python
# For each horizon:
for h_idx, h_name in enumerate(['5yr', '10yr', '15yr', '20yr']):
    pred_h = preds_all[:, h_idx:h_idx+1, :, :]
    target_h = targets[h_name]
    losses_h = self._compute_horizon_losses(pred_h, target_h, ...)
    horizon_losses.append(losses_h)

# Average across horizons
avg_total = mean([h['total'] for h in horizon_losses])
```

#### Loss Weighting
Remains the same per horizon:
- MSE: 1.0 (50% contribution)
- SSIM: 2.0 (20% contribution)
- Laplacian: 1.0 (10% contribution)
- Histogram: 0.67 (20% contribution)

Final loss is the **average** across all 4 horizons.

### 4. Training Script (`scripts/train_lightning.py`)
- Updated to use `fixed_target_years` instead of `fixed_target_year`
- Visualization uses 20yr predictions/targets
- Full validation metrics computed on 20yr horizon

### 5. Histogram Loss (`src/models/histogram_loss.py`)
- **Removed CE term**, now uses only W2 distance
- Added **per-horizon rarity weighting** computed from training data
- Bin weights calculated separately for each horizon (5yr, 10yr, 15yr, 20yr) from 10 batches
- Removed `/num_bins` normalization for stronger gradients
- Each horizon uses its own set of bin weights during loss calculation

**Why per-horizon weights?**
- 5yr forecasts have less change → more concentrated around no-change bin
- 20yr forecasts have more change → more spread across bins
- Separate weights ensure fair comparison within each horizon's characteristic distribution

## Tests Created

### 1. Model Test (`tests/test_multi_horizon_model.py`)
- ✅ Output shape: `[B, 4, H, W]`
- ✅ Gradients flow through all 4 heads
- ✅ Each head produces independent outputs
- ✅ Works with location encoder

### 2. Dataloader Test (`tests/test_multi_horizon_dataloader.py`)
- ✅ Returns all 4 target keys
- ✅ Correct shapes for all targets
- ✅ Targets for different horizons are different
- ✅ Consistent across batches

### 3. Integration Test (`tests/test_multi_horizon_integration.py`)
- ✅ Forward pass produces valid loss
- ✅ Validation step works
- ✅ Full training loop completes

## Default Configuration

```python
# Input
fixed_input_years = (1990, 1995, 2000)  # 3 timesteps

# Targets
fixed_target_years = (2005, 2010, 2015, 2020)  # 4 horizons

# Loss weights
ssim_weight = 2.0
laplacian_weight = 1.0
histogram_weight = 0.67
histogram_warmup_epochs = 20
```

## Running Training

```bash
python scripts/train_lightning.py \
    --train_chips 1000 \
    --val_chips 200 \
    --batch_size 8 \
    --hidden_dim 32 \
    --num_layers 2 \
    --max_epochs 100
```

## Key Benefits

1. **Multi-scale temporal learning**: Model learns patterns at 5, 10, 15, and 20 year scales
2. **Better generalization**: Training on multiple horizons provides more diverse supervision
3. **Flexible inference**: Can evaluate performance at different forecast horizons
4. **Equal weighting**: All horizons contribute equally to the loss

## What's Completed ✅

1. **Visualization**: ✅ Multi-horizon plots created
   - 6x4 grid showing all horizons
   - Row 0: Inputs (HM 1990, 1995, 2000, Elevation)
   - Rows 1-4: Each horizon (Target, Pred, Error, Delta)
   - Row 5: Change histograms for all 4 horizons
   - Separate visualization for each forecast horizon

2. **Inference**: ✅ Multi-horizon predictions
   - Model outputs all 4 horizons simultaneously
   - Separate accumulator arrays for each horizon
   - Writes 4 separate GeoTIFF files:
     - `prediction_2005_blended.tif` (5yr)
     - `prediction_2010_blended.tif` (10yr)
     - `prediction_2015_blended.tif` (15yr)
     - `prediction_2020_blended.tif` (20yr)
   - Tile-based prediction with distance-weighted blending

## What's Still TODO

1. **Metrics**: Log per-horizon metrics to W&B (currently only averaged)

2. **Analysis**: Compare performance across different forecast horizons

3. **Hexbin plots**: Update scatter plots to show per-horizon comparisons

## Backward Compatibility

The code maintains backward compatibility by:
- Checking for `target_20yr` first, falling back to `target` if not found
- Using last horizon (20yr) for visualization when only one is needed
- Keeping the same loss weighting scheme
