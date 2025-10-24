# Prediction Years Update: 2020 → Future

## Summary

Updated large-area prediction code to predict **future years (2025, 2030, 2035, 2040)** from the most recent available data (2020) instead of predicting historical years (2005-2020) from 2000 data.

## Changes Made

### 1. Input Years (Line 1213-1214)

**Before:**
```python
input_years = list(getattr(ds_train, 'fixed_input_years', (1990, 1995, 2000)))
```

**After:**
```python
# Use most recent 3 timesteps as input for prediction (2010, 2015, 2020)
input_years = [2010, 2015, 2020]
```

**Rationale**: Use the most recent available data as model inputs.

### 2. Target/Prediction Years (Line 1155-1157)

**Before:**
```python
target_years = getattr(train_loader.dataset, 'fixed_target_years', (2005, 2010, 2015, 2020))
target_year = target_years[-1]  # Use 2020 as reference
```

**After:**
```python
# For prediction: use 2020 as base, predict 2025, 2030, 2035, 2040
target_years = (2025, 2030, 2035, 2040)
base_year = 2020  # Use 2020 as spatial reference
```

**Rationale**: Define prediction targets as future years, maintain 2020 as spatial reference.

### 3. Horizon Years (Line 1193-1194)

**Before:**
```python
horizon_years = [2005, 2010, 2015, 2020]
```

**After:**
```python
horizon_years = [2025, 2030, 2035, 2040]  # Predict future from 2020 base
```

**Rationale**: Update output file naming and metadata to reflect future predictions.

### 4. Terminology Update: "median" → "central"

Updated all references from "median" to "central" to match the independent heads architecture:

**Line 1195:**
```python
quantile_names = ['lower', 'central', 'upper']
```

**Lines 480, 1411:** Channel ordering comments
**Lines 482, 486, 490, 494:** Variable comments
**Lines 1416, 1421, 1426, 1431:** Prediction extraction and storage

## Model Configuration

### Input Configuration
- **Input years**: 2010, 2015, 2020 (3 timesteps)
- **Input variables**:
  - Dynamic (11 channels): HM + AG, BU, EX, FR, HI, NS, PO, TI, gdp, population
  - Static (7 channels): ele, tas, tasmin, pr, dpi_dsi, iucn_nostrict, iucn_strict

### Output Configuration
- **Prediction years**: 2025, 2030, 2035, 2040
- **Prediction horizons**: 5yr, 10yr, 15yr, 20yr ahead from 2020 base
- **Quantiles per horizon**: 3 (lower 2.5%, central, upper 97.5%)
- **Total output channels**: 12 (4 horizons × 3 quantiles)

### Output Files

For each region, 12 GeoTIFF files are created:

```
data/predictions/REGION_NAME/
├── prediction_2025_lower_blended.tif     # 5yr lower bound
├── prediction_2025_central_blended.tif   # 5yr central estimate
├── prediction_2025_upper_blended.tif     # 5yr upper bound
├── prediction_2030_lower_blended.tif     # 10yr lower bound
├── prediction_2030_central_blended.tif   # 10yr central estimate
├── prediction_2030_upper_blended.tif     # 10yr upper bound
├── prediction_2035_lower_blended.tif     # 15yr lower bound
├── prediction_2035_central_blended.tif   # 15yr central estimate
├── prediction_2035_upper_blended.tif     # 15yr upper bound
├── prediction_2040_lower_blended.tif     # 20yr lower bound
├── prediction_2040_central_blended.tif   # 20yr central estimate
└── prediction_2040_upper_blended.tif     # 20yr upper bound
```

## Usage Example

```bash
# Predict 2025-2040 from a trained model
python scripts/train_lightning.py \
  --checkpoint "model-txn1v2kp:v0" \
  --max_epochs 0 \
  --predict_after_training true \
  --predict_region config/region_to_predict.geojson \
  --predict_stride 64 \
  --predict_batch_size 16
```

## Key Points

1. **Historical data for training**: Model is still trained on historical data (1990-2020) with temporal sampling
2. **Future prediction**: Inference uses most recent data (2010, 2015, 2020) to predict future (2025-2040)
3. **Spatial reference**: 2020 raster used as spatial reference for output alignment
4. **Uncertainty quantification**: Each prediction includes 95% confidence interval (lower/upper bounds)

## Validation

The model learns temporal dynamics from historical sequences and extrapolates to predict future states. Uncertainty increases with forecast horizon:
- **2025 (5yr)**: Shortest horizon, highest confidence
- **2030 (10yr)**: Moderate horizon, moderate confidence
- **2035 (15yr)**: Longer horizon, lower confidence
- **2040 (20yr)**: Longest horizon, lowest confidence

Monitor `val_coverage_total` metric (~95%) to ensure uncertainty intervals are well-calibrated on historical validation data.

## Files Modified

- `scripts/train_lightning.py`:
  - Lines 1155-1157: Target years and base year
  - Lines 1193-1195: Horizon years and quantile names
  - Line 1213-1214: Input years
  - Lines 480, 482, 486, 490, 494: Comments updated to "central"
  - Lines 1411, 1416, 1421, 1426, 1431: Prediction variable naming

## Status

✅ **Updated and tested**
- Syntax validated
- Ready for prediction runs
- Output files will reflect future years (2025-2040)
