# Quantile Prediction Implementation

## Overview

Upgraded model to predict uncertainty quantiles (2.5%, 50%, 97.5%) for each of 4 time horizons, enabling uncertainty quantification.

## Completed Steps ✅

### 1. Created Pinball Loss (`src/models/pinball_loss.py`)
- Quantile regression loss function
- Asymmetric penalty: penalizes underestimates for lower quantile, overestimates for upper quantile
- Tested with `python src/models/pinball_loss.py`

### 2. Updated Model Architecture (`src/models/spatiotemporal_predictor.py`)
- **Output changed**: `[B, 4, H, W]` → `[B, 12, H, W]`
- Each prediction head now outputs 3 quantiles instead of 1
- Channel ordering: `[lower_5yr, median_5yr, upper_5yr, lower_10yr, median_10yr, upper_10yr, ...]`

### 3. Updated Lightning Module (`src/models/lightning_module.py`)
- Added pinball loss instances for lower (q=0.025) and upper (q=0.975) quantiles
- Updated `_compute_horizon_losses()` to accept 3 predictions per horizon
- **Loss components**:
  - MSE, SSIM, Laplacian, Histogram: computed on **median only**
  - Pinball losses: computed on **lower and upper** quantiles
  - Total loss = MSE + pinball_lower + pinball_upper + SSIM + Laplacian + Histogram
- Added **coverage metric**: % of targets within [lower, upper] interval (target: 95%)
- Updated training_step and validation_step to extract and process all 3 quantiles
- **New metrics logged**:
  - Per-horizon: `train/val_pinball_lower_{horizon}`, `train/val_pinball_upper_{horizon}`
  - Coverage: `val_coverage_{horizon}`, `val_coverage_total`
  - Averaged: `train/val_pinball_lower_total`, `train/val_pinball_upper_total`

### 4. Updated Validation Visualizations (`scripts/train_lightning.py`)
- **Figure layout changed**: 6 rows × 5 columns → 6 rows × 7 columns
- **New columns added**:
  - Column 5: Δ Lower 2.5% (change predicted by lower quantile)
  - Column 6: Δ Upper 97.5% (change predicted by upper quantile)
- Columns now show:
  1. Target
  2. Pred Median
  3. Error (median vs target)
  4. Δ Obs (observed change)
  5. Δ Pred Median
  6. **Δ Lower 2.5%** (NEW)
  7. **Δ Upper 97.5%** (NEW)
- Updated batch data storage to include all quantile predictions

## Remaining Steps 🚧

### 5. Update Large-Area Predictions (INCOMPLETE)

**Location**: `scripts/train_lightning.py` around lines 1300-1470

**Required changes**:

#### A. Update tile processing loop (around line 1329)
```python
# Current: Extract 4 channels
pred_h = batch_preds[tile_idx, h_idx, :hi, :wj]

# Needed: Extract 3 quantiles per horizon
pred_lower = batch_preds[tile_idx, 3*h_idx, :hi, :wj]
pred_median = batch_preds[tile_idx, 3*h_idx+1, :hi, :wj]
pred_upper = batch_preds[tile_idx, 3*h_idx+2, :hi, :wj]
```

#### B. Update accumulators (around line 1128)
```python
# Current: 4 accumulators (one per horizon)
accum_horizons = {h: np.zeros(...) for h in ['5yr', '10yr', '15yr', '20yr']}

# Needed: 12 accumulators (3 quantiles × 4 horizons)
accum_horizons = {
    '5yr_lower': np.zeros(...), '5yr': np.zeros(...), '5yr_upper': np.zeros(...),
    '10yr_lower': np.zeros(...), '10yr': np.zeros(...), '10yr_upper': np.zeros(...),
    # ...
}
```

#### C. Update output writing (around line 1469)
```python
# Current: Write 4 files
for h_name, h_year in zip(horizon_names, horizon_years):
    out_path = out_dir / f"prediction_{h_year}_blended.tif"
    with rasterio.open(out_path, 'w', **out_profile) as dst:
        dst.write(out_horizons[h_name], 1)

# Needed: Write 12 files (3 per horizon)
quantiles = ['lower', 'median', 'upper']
for h_name, h_year in zip(horizon_names, horizon_years):
    for q_name in quantiles:
        key = f"{h_name}_{q_name}" if q_name != 'median' else h_name
        out_path = out_dir / f"prediction_{h_year}_{q_name}_blended.tif"
        with rasterio.open(out_path, 'w', **out_profile) as dst:
            dst.write(out_horizons[key], 1)
```

**Output files** (12 total):
- `prediction_2005_lower_blended.tif`
- `prediction_2005_median_blended.tif`
- `prediction_2005_upper_blended.tif`
- `prediction_2010_lower_blended.tif`
- `prediction_2010_median_blended.tif`
- `prediction_2010_upper_blended.tif`
- (... and so on for 2015, 2020)

### 6. Test Changes

#### A. Quick Training Test
```bash
python scripts/train_lightning.py \
  --train_chips 100 \
  --val_chips 40 \
  --batch_size 4 \
  --max_epochs 2 \
  --hidden_dim 16 \
  --num_layers 1
```

**Expected**:
- Training runs without errors
- New pinball loss metrics appear in logs
- Coverage metrics computed (should be around 0-100% initially, will calibrate with training)
- Validation plots show 7 columns with quantile changes

#### B. Check Model Output Shape
```python
import torch
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor

model = SpatioTemporalPredictor(hidden_dim=16, num_layers=1, 
                                num_dynamic_channels=11, num_static_channels=7)
x_dyn = torch.randn(2, 3, 11, 128, 128)  # [B=2, T=3, C=11, H=128, W=128]
x_stat = torch.randn(2, 7, 128, 128)     # [B=2, C=7, H=128, W=128]
lonlat = torch.randn(2, 128, 128, 2)     # [B=2, H=128, W=128, 2]

out = model(x_dyn, x_stat, lonlat=lonlat)
print(f"Output shape: {out.shape}")  # Should be: torch.Size([2, 12, 128, 128])
```

#### C. Verify Coverage Calibration
After training for ~20-50 epochs, check W&B:
- `val_coverage_total` should approach ~95% (well-calibrated model)
- If too low (<90%): predictions too confident, intervals too narrow
- If too high (>98%): predictions too uncertain, intervals too wide

## Architecture Summary

**Model Output**: `[B, 12, H, W]`
- Channels 0, 3, 6, 9: Lower 2.5% quantile (one per horizon)
- Channels 1, 4, 7, 10: **Central prediction** (one per horizon) - NOT statistical median!
- Channels 2, 5, 8, 11: Upper 97.5% quantile (one per horizon)

**Key Design: Independent Prediction Heads**
- **3 separate heads per horizon** (12 total heads)
- Central heads: Full-size (hidden_dim → hidden_dim → 1)
- Quantile heads: Smaller (hidden_dim → hidden_dim/2 → 1)
- **No shared gradients** between central and quantile predictions

**Loss Function** (Independent Gradients):
```
Central prediction receives:
  MSE + SSIM_weight × SSIM + Laplacian_weight × Laplacian + Histogram_weight × Histogram

Lower quantile receives:
  Pinball_lower(q=0.025)

Upper quantile receives:
  Pinball_upper(q=0.975)

Total = Central_losses + Pinball_lower + Pinball_upper
```

**Why "Central" not "Median"?**
- Central prediction is optimized for **multiple objectives** (accuracy + spatial patterns)
- NOT constrained to be the statistical median
- Quantiles are independent and only optimized for bracketing uncertainty

**Metrics**:
- Standard: MSE, MAE, SSIM, Laplacian, Histogram (on **central** only)
- Quantile: Pinball losses (on lower/upper only)
- Calibration: Coverage percentage (% of targets in [lower, upper])

**Visualization**: 6 rows × 7 columns
- Shows target, **central** prediction, error, observed/predicted changes for central, lower, and upper

**Predictions**: 12 GeoTIFF files per region
- 3 files per horizon (lower, **central**, upper)
- Enables uncertainty mapping and risk assessment

## Benefits

1. **Uncertainty Quantification**: Know confidence in predictions
2. **Risk Assessment**: Map areas with high prediction uncertainty
3. **Decision Support**: Different actions for different confidence levels
4. **Model Calibration**: Coverage metric ensures predictions are well-calibrated
5. **Robust Planning**: Account for best/worst case scenarios

## Next Actions

1. Complete Step 5 (update large-area prediction code)
2. Run Step 6 (testing)
3. Train full model and verify coverage calibration
4. Use quantile predictions for risk mapping and decision support

## Status
- **Completed**: Steps 1-4 (core architecture, losses, metrics, visualization)
- **Remaining**: Step 5 (large-area predictions), Step 6 (testing)
- **Est. completion time**: ~30 minutes
