# Quantile Prediction Implementation

## Overview

Upgraded model to predict uncertainty quantiles (2.5%, 50%, 97.5%) for each of 4 time horizons, enabling uncertainty quantification.

## Implementation Status: ✅ COMPLETE

All steps have been implemented and tested.

### 1. Created Pinball Loss (`src/models/pinball_loss.py`) ✅
- Quantile regression loss function
- Asymmetric penalty: penalizes underestimates for lower quantile, overestimates for upper quantile
- **Fixed**: NaN handling - filters valid pixels BEFORE computing loss
- Tested with `python src/models/pinball_loss.py`

### 2. Updated Model Architecture (`src/models/spatiotemporal_predictor.py`) ✅
- **Output changed**: `[B, 4, H, W]` → `[B, 12, H, W]`
- **Independent heads**: 12 separate prediction heads (3 per horizon)
- Channel ordering: `[lower_5yr, central_5yr, upper_5yr, lower_10yr, central_10yr, upper_10yr, ...]`
- **Central heads**: Full-size (hidden_dim → hidden_dim → 1)
- **Quantile heads**: Smaller (hidden_dim → hidden_dim/2 → 1) for efficiency

### 3. Updated Lightning Module (`src/models/lightning_module.py`) ✅
- Added pinball loss instances for lower (q=0.025) and upper (q=0.975) quantiles
- Updated `_compute_horizon_losses()` to accept 3 predictions per horizon
- **Loss components** (INDEPENDENT gradients):
  - MSE, SSIM, Laplacian, Histogram: computed on **central only**
  - Pinball losses: computed on **lower and upper** quantiles only
  - Total loss = MSE + pinball_lower + pinball_upper + 2.0×SSIM + 1.0×Laplacian + 0.67×Histogram
- Added **coverage metric**: % of targets within [lower, upper] interval (target: ~95%)
- Enhanced console output during validation showing pinball losses and coverage
- **New metrics logged**:
  - Per-horizon: `train/val_pinball_lower_{horizon}`, `train/val_pinball_upper_{horizon}`
  - Coverage: `val_coverage_{horizon}`, `val_coverage_total`
  - Averaged: `train/val_pinball_lower_total`, `train/val_pinball_upper_total`

### 4. Updated Validation Visualizations (`scripts/train_lightning.py`) ✅
- **Figure layout changed**: 6 rows × 5 columns → 6 rows × 7 columns
- **New columns added**:
  - Column 5: Δ Lower 2.5% (change predicted by lower quantile)
  - Column 6: Δ Upper 97.5% (change predicted by upper quantile)
- Columns now show:
  1. Target
  2. Pred Central
  3. Error (central vs target)
  4. Δ Obs (observed change)
  5. Δ Pred Central
  6. **Δ Lower 2.5%** (NEW)
  7. **Δ Upper 97.5%** (NEW)
- Updated batch data storage to include all quantile predictions

### 5. Updated Large-Area Predictions ✅

**Location**: `scripts/train_lightning.py` lines 1154-1432

**Changes implemented**:

#### A. Updated prediction years
- **Input years**: 2010, 2015, 2020 (most recent available data)
- **Target years**: 2025, 2030, 2035, 2040 (future predictions)
- Uses 2020 as spatial reference raster

#### B. Tile processing extracts 3 quantiles per horizon
```python
# Extract 3 quantiles for each horizon
pred_lower = batch_preds[tile_idx, 3*h_idx, :hi, :wj]
pred_central = batch_preds[tile_idx, 3*h_idx+1, :hi, :wj]
pred_upper = batch_preds[tile_idx, 3*h_idx+2, :hi, :wj]
```

#### C. 12 accumulators created (3 quantiles × 4 horizons)
```python
quantile_names = ['lower', 'central', 'upper']
for h in horizon_names:
    for q in quantile_names:
        key = f"{h}_{q}"
        accum_horizons[key] = np.zeros((Hwin, Wwin), dtype=np.float64)
```

#### D. Writes 12 GeoTIFF output files
```python
for h_name, h_year in zip(horizon_names, horizon_years):
    for q_name in quantile_names:
        out_path = out_dir / f"prediction_{h_year}_{q_name}_blended.tif"
        with rasterio.open(out_path, 'w', **out_profile) as dst:
            dst.write(out_horizons[f"{h_name}_{q_name}"], 1)
```

**Output files** (12 total per region):
- `prediction_2025_lower_blended.tif` (5yr, 2.5% quantile)
- `prediction_2025_central_blended.tif` (5yr, central estimate)
- `prediction_2025_upper_blended.tif` (5yr, 97.5% quantile)
- `prediction_2030_lower_blended.tif` (10yr, 2.5% quantile)
- `prediction_2030_central_blended.tif` (10yr, central estimate)
- `prediction_2030_upper_blended.tif` (10yr, 97.5% quantile)
- (... and so on for 2035, 2040)

### 6. Testing ✅

All tests have passed successfully.

#### A. Architecture Test
```bash
python test_independent_heads.py
```
**Results**:
- ✅ 12 independent heads verified (4 central, 4 lower, 4 upper)
- ✅ Output shape: `[B, 12, H, W]`
- ✅ Quantile heads smaller than central heads
- ✅ Gradients flow independently

#### B. Model Output Verification
```python
import torch
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor

model = SpatioTemporalPredictor(hidden_dim=64, num_layers=2, 
                                num_dynamic_channels=11, num_static_channels=7)
x_dyn = torch.randn(2, 3, 11, 128, 128)  # [B=2, T=3, C=11, H=128, W=128]
x_stat = torch.randn(2, 7, 128, 128)     # [B=2, C=7, H=128, W=128]
lonlat = torch.randn(2, 128, 128, 2)     # [B=2, H=128, W=128, 2]

out = model(x_dyn, x_stat, lonlat=lonlat)
print(f"Output shape: {out.shape}")  # Output: torch.Size([2, 12, 128, 128]) ✅
```

#### C. Coverage Calibration (During Training)
Monitor these metrics in W&B:
- `val_coverage_total` should approach ~95% with training (well-calibrated model)
- If too low (<85%): predictions too confident, intervals too narrow
- If too high (>98%): predictions too uncertain, intervals too wide
- **Current observations**: Coverage typically starts at 80-90% and calibrates toward 95%

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

## Usage Examples

### Training
```bash
python scripts/train_lightning.py \
  --max_epochs 100 \
  --train_chips 500 \
  --val_chips 100 \
  --batch_size 8 \
  --hidden_dim 64 \
  --num_layers 2
```

### Prediction (Future Years: 2025-2040)
```bash
python scripts/train_lightning.py \
  --checkpoint "model-artifact:v0" \
  --max_epochs 0 \
  --predict_after_training true \
  --predict_region config/region_to_predict.geojson
```

### Uncertainty Mapping
```python
import rasterio
# Load predictions
with rasterio.open('prediction_2025_lower_blended.tif') as src:
    lower = src.read(1)
with rasterio.open('prediction_2025_upper_blended.tif') as src:
    upper = src.read(1)

# Compute 95% confidence interval width
uncertainty = upper - lower
# High values = high uncertainty, low values = high confidence
```

## Implementation Status

✅ **COMPLETE** - All components implemented and tested
- Core architecture with independent heads
- Pinball loss with NaN handling
- Training and validation pipeline
- Large-area prediction (future years 2025-2040)
- Comprehensive testing and verification

## See Also

- [`docs/INDEPENDENT_HEADS_IMPLEMENTATION.md`](INDEPENDENT_HEADS_IMPLEMENTATION.md): Architecture details
- [`docs/LOSS_WEIGHTING_EXPLAINED.md`](LOSS_WEIGHTING_EXPLAINED.md): Loss composition
- [`docs/PREDICTION_YEARS_UPDATE.md`](PREDICTION_YEARS_UPDATE.md): Future prediction details
- [`docs/PINBALL_LOSS_FIX.md`](PINBALL_LOSS_FIX.md): NaN handling fix
