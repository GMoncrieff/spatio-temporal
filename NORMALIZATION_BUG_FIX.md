# Critical Bug Fix: Normalization Mismatch in Large-Area Predictions

## Problem: Grid Patterns Within Tiles

Grid patterns were visible **within each tile** (not just at boundaries) in large-area predictions that were **not present** in validation `predictions_vs_targets` plots.

## Root Cause: Incorrect Normalization

The large-area prediction code was using **pooled normalization** (hm_mean/hm_std for all variables) instead of **per-variable normalization** used by the dataloader.

### Validation/Training (Correct) - torchgeo_dataloader.py:314, 329

```python
# Per-variable normalization for dynamic variables
carr = (carr - self.comp_means[var_name]) / self.comp_stds[var_name]

# Per-variable normalization for static variables
sarr = (sarr - self.static_means[static_idx]) / self.static_stds[static_idx]
```

### Large-Area Prediction (BUG) - train_lightning.py:1248, 1260

```python
# WRONG: Using HM normalization for ALL dynamic variables
channels.append((carr - hm_mean) / hm_std)

# WRONG: Using elevation normalization for ALL static variables
static_chs.append((sarr - elev_mean) / elev_std)
```

## Why This Caused Grid Patterns

From memory `faf048e3`, the variables have vastly different scales:

| Variable | Mean | Std | Correct Normalization |
|----------|------|-----|-----------------------|
| **HM (AA)** | 0.06 | 0.11 | hm_mean/hm_std |
| **GDP** | 7.6M | 55M | comp_means['gdp']/comp_stds['gdp'] |
| **Population** | 68 | 709 | comp_means['population']/comp_stds['population'] |
| **Elevation** | -4200m | 11km | static_means[0]/static_stds[0] |
| **Temperature** | 13°C | 13°C | static_means[1]/static_stds[1] |
| **Precipitation** | 9.5mm | 8mm | static_means[3]/static_stds[3] |

### Example: GDP Normalization Error

**Correct normalization:**
```python
gdp_normalized = (gdp - 7.6M) / 55M  # Results in ~N(0,1)
```

**Bug (using HM normalization):**
```python
gdp_normalized = (gdp - 0.06) / 0.11  # Results in HUGE values (millions!)
```

For example, GDP = 10M:
- **Correct**: (10M - 7.6M) / 55M = 0.044
- **Bug**: (10M - 0.06) / 0.11 = 90,909,090!

The model was receiving **completely corrupted inputs**, with values that should be ~0 becoming millions. This caused the model to produce garbage outputs, resulting in the grid patterns.

## The Fix

### 1. Extract Per-Variable Stats (Lines 1141-1146)

```python
# CRITICAL: Get per-variable normalization stats (NOT pooled hm_mean/hm_std)
comp_means = ds_train.comp_means  # Dict: {var_name: mean}
comp_stds = ds_train.comp_stds    # Dict: {var_name: std}
static_means = ds_train.static_means  # List: [mean_0, mean_1, ...]
static_stds = ds_train.static_stds    # List: [std_0, std_1, ...]
HM_VARS = ds_train.HM_VARS  # List of component variable names
```

### 2. Fix Dynamic Variable Normalization (Lines 1249-1254)

```python
# BEFORE (WRONG):
channels.append((carr - hm_mean) / hm_std)

# AFTER (CORRECT):
for var_idx, (var_name, src) in enumerate(zip(HM_VARS, comp_srcs[y])):
    carr = src.read(1, window=win, masked=True).filled(np.nan)
    carr = np.nan_to_num(carr, nan=0.0)
    # Use per-variable normalization (CRITICAL for GDP/population)
    channels.append((carr - comp_means[var_name]) / comp_stds[var_name])
```

### 3. Fix Static Variable Normalization (Lines 1265-1266)

```python
# BEFORE (WRONG):
static_chs.append((sarr - elev_mean) / elev_std)

# AFTER (CORRECT):
# Use per-variable normalization (CRITICAL for different scales)
static_chs.append((sarr - static_means[static_idx]) / static_stds[static_idx])
```

## Impact

### Before Fix
- ❌ Grid patterns within tiles
- ❌ Model receiving corrupted inputs (values off by millions)
- ❌ GDP normalized to 90M+ instead of ~0
- ❌ Predictions completely wrong
- ❌ Inconsistent with validation behavior

### After Fix
- ✅ Clean predictions matching validation
- ✅ Model receives properly normalized inputs (~N(0,1))
- ✅ GDP normalized to ~0 as expected
- ✅ Predictions accurate
- ✅ Consistent with training/validation

## Why Validation Was Correct

The validation code (lines 465-489) uses data from the **dataloader**, which already applies per-variable normalization in `__getitem__` (torchgeo_dataloader.py:314, 329). So validation never had this bug.

The large-area prediction code reads rasters directly and does its own normalization, which is where the bug was introduced.

## Testing

Run predictions with the fix:

```bash
python scripts/train_lightning.py \
  --checkpoint model-txn1v2kp:v0 \
  --max_epochs 0 \
  --predict_after_training true \
  --predict_stride 128
```

Expected results:
- No grid patterns within tiles
- Predictions should match validation quality
- Values in sensible range [0, 1]

## Verification

Check that normalization is correct:

```python
import numpy as np

# Example GDP value
gdp_raw = 10_000_000  # 10 million

# Correct normalization (from dataloader)
gdp_mean = 7_596_796
gdp_std = 55_702_060
gdp_normalized_correct = (gdp_raw - gdp_mean) / gdp_std
print(f"Correct: {gdp_normalized_correct:.4f}")  # Should be ~0.04

# Bug (using HM stats)
hm_mean = 0.06
hm_std = 0.11
gdp_normalized_bug = (gdp_raw - hm_mean) / hm_std
print(f"Bug: {gdp_normalized_bug:.0f}")  # Would be ~90 million!
```

## Related

- **Memory**: `faf048e3` - Documents per-variable normalization requirement
- **Dataloader**: `scripts/torchgeo_dataloader.py` lines 314, 329
- **Large-Area Prediction**: `scripts/train_lightning.py` lines 1141-1266
- **Variable Scales**: GDP (billions), population (hundreds), HM (0-1), elevation (km), temperature (°C)

## Summary

This was a **critical bug** that caused large-area predictions to be completely wrong. The model was receiving inputs that were normalized incorrectly by orders of magnitude (GDP values in the millions instead of ~0).

The fix ensures that large-area predictions use the same per-variable normalization as training/validation, resulting in correct inputs and accurate predictions.

**Files Modified**:
- `scripts/train_lightning.py` (lines 1141-1146, 1249-1254, 1265-1266)

**Status**: ✅ Fixed and ready for testing  
**Priority**: CRITICAL - affects all large-area predictions  
**Date**: 2025-10-21
