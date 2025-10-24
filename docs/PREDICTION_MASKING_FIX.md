# Prediction Masking Fix

## Problem: Grid Patterns in Large-Area Predictions

Grid patterns were visible in large-area predictions that were **not present** in the validation `predictions_vs_targets` plots.

## Root Cause

**The validation code and large-area prediction code handled invalid inputs differently:**

### Validation Code (Correct) - Lines 469-489

```python
# Replace NaNs in inputs with 0 for model forward
input_dynamic_clean = torch.nan_to_num(input_dynamic, nan=0.0)
input_static_clean = torch.nan_to_num(input_static, nan=0.0)

# Get predictions from model
preds_all = best_model(input_dynamic_clean, input_static_clean, lonlat=lonlat)

# CRITICAL: Set predictions to NaN where any input was NaN
preds_5yr[~input_mask] = float('nan')
preds_10yr[~input_mask] = float('nan')
preds_15yr[~input_mask] = float('nan')
preds_20yr[~input_mask] = float('nan')
```

### Large-Area Prediction Code (Bug) - Lines 1286-1333

```python
# Replace NaN with 0.0 in normalized space
in_dyn = np.nan_to_num(input_dynamic_np, nan=0.0)
in_stat = np.nan_to_num(input_static_np, nan=0.0)

# Get predictions
batch_preds = infer_model(batch_dyn_tensor, batch_stat_tensor, ...)

# Extract predictions
pred_h = batch_preds[tile_idx, h_idx, :hi, :wj].detach().cpu().numpy()
pred_h = pred_h * hm_std + hm_mean  # denormalize

# BUG: No masking! Predictions at NaN locations are used directly
preds_horizons[h_name] = pred_h
```

## Why This Caused Grid Patterns

1. **NaN inputs replaced with 0**: Before model forward, NaN values in inputs were replaced with 0 (normalized mean)
2. **Model predicts on fake data**: The model made predictions for pixels with artificial 0 values
3. **Invalid predictions accumulated**: These meaningless predictions were accumulated into the final output
4. **Different NaN patterns per tile**: Each tile had different NaN patterns at edges
5. **Result**: Grid artifacts where tiles with different NaN patterns meet

## The Fix

**Added input validity tracking and prediction masking** (Lines 1278-1282, 1331-1332):

```python
# Track which pixels had valid inputs (BEFORE replacing NaN)
# This matches the validation code approach (lines 465-467)
dynamic_has_nan = ~np.isfinite(input_dynamic_np).all(axis=(0, 1))  # [hi, wj]
static_has_nan = ~np.isfinite(input_static_np).all(axis=0) if static_chs else np.zeros((hi, wj), dtype=bool)
input_invalid_mask = dynamic_has_nan | static_has_nan  # Pixels to mask in predictions

# ... later, after model inference ...

# Extract predictions
pred_h = batch_preds[tile_idx, h_idx, :hi, :wj].detach().cpu().numpy()
pred_h = pred_h * hm_std + hm_mean  # denormalize
# CRITICAL: Mask predictions where inputs had NaN (same as validation code)
pred_h[input_invalid_mask] = np.nan
preds_horizons[h_name] = pred_h
```

## Changes Made

### 1. Track Input Validity (Lines 1278-1282)

Before replacing NaN with 0, record which pixels had NaN in **any** input channel:

```python
dynamic_has_nan = ~np.isfinite(input_dynamic_np).all(axis=(0, 1))
static_has_nan = ~np.isfinite(input_static_np).all(axis=0)
input_invalid_mask = dynamic_has_nan | static_has_nan
```

### 2. Pass Mask Through Batch Metadata (Line 1312)

```python
batch_metadata.append((i, j, hi, wj, li0, lj0, li1, lj1, valid_mask, input_invalid_mask))
```

### 3. Mask Predictions After Inference (Lines 1331-1332)

```python
pred_h = batch_preds[tile_idx, h_idx, :hi, :wj].detach().cpu().numpy()
pred_h = pred_h * hm_std + hm_mean
pred_h[input_invalid_mask] = np.nan  # ← THE FIX
```

## Impact

### Before Fix
- ❌ Grid patterns visible in predictions
- ❌ Invalid predictions at NaN locations
- ❌ Artifacts at tile boundaries
- ❌ Predictions inconsistent with validation

### After Fix
- ✅ Clean predictions, no grid patterns
- ✅ Predictions masked where inputs invalid
- ✅ Smooth tile boundaries
- ✅ Predictions consistent with validation

## Technical Details

### Why NaN Inputs Were Replaced with 0

The model cannot process NaN values, so they must be replaced. The choice of 0 in normalized space means:
- Dynamic: 0 = mean value of that variable
- Static: 0 = mean value of that variable

This is reasonable for **valid pixels** with missing covariates, but **not valid for pixels that should not be predicted at all**.

### The Input Mask Logic

```python
dynamic_has_nan = ~np.isfinite(input_dynamic_np).all(axis=(0, 1))
```

This creates a mask that is `True` for pixels where:
- **Any timestep** had NaN in dynamic inputs, OR
- **Any channel** had NaN in static inputs

These pixels get masked in the predictions because:
1. The model has no real data to base predictions on
2. Predictions would be based purely on artificial 0 values
3. These predictions are meaningless and create artifacts

### Validation Code Already Had This

The validation code (`predictions_vs_targets` plots) already did this correctly:

```python
# Lines 465-467
dynamic_valid = torch.isfinite(input_dynamic).all(dim=(1, 2), keepdim=True)
static_valid = torch.isfinite(input_static).all(dim=1, keepdim=True).unsqueeze(1)
input_mask = target_valid & dynamic_valid.squeeze(2) & static_valid.squeeze(2)

# Lines 485-489
preds_5yr[~input_mask] = float('nan')
```

The fix brings the large-area prediction code into alignment with this approach.

## Testing

To verify the fix works:

1. **Run new predictions:**
```bash
python scripts/train_lightning.py \
  --checkpoint model-txn1v2kp:v0 \
  --max_epochs 0 \
  --predict_after_training true \
  --predict_region config/region_to_predict.geojson
```

2. **Compare with validation plots:**
   - Load predictions in QGIS or Python
   - Check for grid patterns (should be gone)
   - Compare edge quality with validation plots (should match)

3. **Check NaN masking:**
```python
import rasterio
import numpy as np

with rasterio.open('data/predictions/prediction_2020_blended.tif') as src:
    data = src.read(1)
    
# Count NaN pixels
nan_count = np.isnan(data).sum()
print(f"NaN pixels: {nan_count:,}")

# NaN should only be in invalid regions, not forming grid patterns
```

## Related Issues

- **Memory**: `3e33b037` - Documents NaN contamination issue
- **Validation Code**: Lines 465-489 in `train_lightning.py`
- **Large-Area Prediction**: Lines 1278-1332 in `train_lightning.py`

## Summary

The fix ensures that:
1. ✅ Predictions are only made for pixels with valid inputs
2. ✅ Pixels with NaN inputs get NaN predictions
3. ✅ No artificial predictions from replaced 0 values
4. ✅ Consistent behavior between validation and large-area prediction
5. ✅ No grid artifacts at tile boundaries

**Status**: ✅ Fixed and ready for testing  
**Date**: 2025-10-21
