# Masking Strategy: Training vs Prediction

## Why Target HM has More Missing Data in Visualizations

The **target rasters appear to have more missing data** than input rasters in the training visualizations. This is **intentional and correct** for showing what pixels the model actually trains on.

## Different Masking for Different Purposes

### 1. **Training/Validation** (Strict Masking)
**Purpose:** Only learn from pixels where we have complete information

**Requirements for a pixel to be VALID:**
- ✅ Target HM is not NaN
- ✅ ALL input dynamic channels are not NaN (HM + AG, BU, EX, FR, HI, NS, PO, TI, gdp, population)
- ✅ ALL input timesteps are not NaN (1990, 1995, 2000)
- ✅ ALL static channels are not NaN (elevation, temperature, precipitation, protected areas, etc.)

**Why strict?**
- Ensures model only learns from high-quality training examples
- Prevents gradient updates from pixels with missing covariates
- Maintains data integrity during training

**Code location:**
- `src/models/lightning_module.py` (training_step, validation_step)
- `scripts/train_lightning.py` (validation visualization loop, line 365-367)

### 2. **Inference/Prediction** (Relaxed Masking)
**Purpose:** Make predictions wherever we have core data

**Requirements for a pixel to be VALID:**
- ✅ Primary HM channel at all input timesteps is not NaN (1990, 1995, 2000)
- ✅ First static channel (elevation) is not NaN

**Component channels CAN be NaN:**
- Will be imputed using two-stage strategy (see below)

**Imputation Strategy (Two-Stage):**

**Stage 1: Domain-Specific Imputation (BEFORE normalization)**
For variables where missing means "absence", replace NaN with 0 in original space:

**Dynamic variables:**
- All HM components (AG, BU, EX, FR, HI, NS, PO, TI, gdp, population) → 0
- Interpretation: No pressure/activity where data is missing
- Note: AA (primary HM) keeps NaN - it's the target

**Static variables:**
- elevation → 0 (sea level)
- dpi_dsi → 0 (no development pressure)
- iucn_strict → 0 (not strictly protected)
- iucn_nostrict → 0 (not protected)
- Temperature/precipitation → keep NaN (unknown climate)

**Stage 2: Mean Imputation (AFTER normalization, during forward pass)**
Remaining NaN values use `torch.nan_to_num(nan=0.0)` on **z-score normalized** data:
```python
# After normalization: x_norm = (x - mean) / std
# Setting x_norm = 0.0 means x = mean
```

**Why this two-stage approach:**
1. **Domain knowledge first**: Variables like "population" make more sense as 0 (unpopulated) than mean
2. **Climate variables**: Unknown temperature/precipitation better as mean than 0°C/0mm
3. **Flexibility**: Easy to adjust which variables use which strategy

**Code locations:**
- `scripts/torchgeo_dataloader.py` (line 288-293, 298-308)
- `scripts/train_lightning.py` (line 1031-1032, 1039-1047)

**Why relaxed masking?**
- Maximize spatial coverage in predictions
- More pixels get predictions in production
- Missing covariates don't prevent prediction

**Code location:**
- `scripts/train_lightning.py` (large-area prediction, line 1038-1044, 1056-1059)

## Summary

| Context | Masking | Reason |
|---------|---------|--------|
| **Training** | Strict (all channels required) | Only learn from complete data |
| **Validation** | Strict (all channels required) | Fair evaluation on same quality |
| **Visualization** | Strict (shows training mask) | Show what model actually sees |
| **Prediction** | Relaxed (only HM + elevation) | Maximize coverage in production |

## Visual Comparison

```
Input HM Raster (1990):     Target HM Raster (2020):
[Shows only HM channel]     [Shows where ALL inputs valid]
  ██████████                  ██░░░░████
  ██████████                  ██░░░░████
  ██████████                  ██████████
  
More visible because        More missing because requires
only 1 channel checked      ALL channels valid
```

The extra missing pixels in the target represent areas where:
- Component variables (AG, BU, gdp, etc.) have NaN
- Additional static variables have NaN

This is the **correct training behavior** but is **relaxed for prediction** to maximize coverage.
