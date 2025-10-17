# Phase 2 Complete: Dynamic Year Labels in Visualization

## ✅ What Was Implemented

### Visualization Updates (`scripts/train_lightning.py`)

#### 1. **Extract Year Metadata from Batch** (lines 593-599)
```python
# Extract year metadata from this sample (NEW: supports temporal sampling)
input_years = first_batch.get('input_years', [1990, 1995, 2000])
target_years = first_batch.get('target_years', [2005, 2010, 2015, 2020])
# Handle batch dimension if present
if isinstance(input_years, list) and len(input_years) > 0 and isinstance(input_years[0], list):
    input_years = input_years[b]  # Extract for this sample
    target_years = target_years[b]
```

#### 2. **Input HM Titles** (line 614)
**Before**: `'HM 1990'`, `'HM 1995'`, `'HM 2000'` (hardcoded)

**After**: `f'HM {input_years[t]}'` (dynamic)
- Validation: "HM 1990", "HM 1995", "HM 2000"
- Training (end_year=2005): "HM 1995", "HM 2000", "HM 2005"
- Training (end_year=2010): "HM 2000", "HM 2005", "HM 2010"
- etc.

#### 3. **Target/Prediction Titles** (lines 661, 668)
**Before**: `'Target 2005'`, `'Pred 2010'` (hardcoded)

**After**: `f'Target {h_year} (+{(h_idx+1)*5}yr)'`
- Shows both absolute year AND relative horizon
- Example: "Target 2010 (+5yr)", "Pred 2020 (+15yr)"

**Why show both?**
- **Absolute year**: What year is being predicted
- **Relative horizon**: How far ahead from last input

#### 4. **Error Titles** (line 676)
**After**: `f'Error {h_year}'`
- Dynamic year from target_years

#### 5. **Delta Titles** (lines 684, 691)
**After**: `f'Δ Obs {h_year}'`, `f'Δ Pred {h_year}'`
- Dynamic year from target_years

#### 6. **Histogram Titles** (line 718)
**After**: `f'Δ Histogram {h_year}'`
- Dynamic year from target_years

## 📊 Example Output

### Validation Sample (Fixed Years):
```
Row 0: HM 1990 | HM 1995 | HM 2000 | Elevation | [empty]
Row 1: Target 2005 (+5yr) | Pred 2005 (+5yr) | Error 2005 | Δ Obs 2005 | Δ Pred 2005
Row 2: Target 2010 (+10yr) | Pred 2010 (+10yr) | Error 2010 | Δ Obs 2010 | Δ Pred 2010
Row 3: Target 2015 (+15yr) | Pred 2015 (+15yr) | Error 2015 | Δ Obs 2015 | Δ Pred 2015
Row 4: Target 2020 (+20yr) | Pred 2020 (+20yr) | Error 2020 | Δ Obs 2020 | Δ Pred 2020
Row 5: Δ Histogram 2005 | Δ Histogram 2010 | Δ Histogram 2015 | Δ Histogram 2020 | [empty]
```

### Training Sample (end_year=2010):
```
Row 0: HM 2000 | HM 2005 | HM 2010 | Elevation | [empty]
Row 1: Target 2015 (+5yr) | Pred 2015 (+5yr) | Error 2015 | Δ Obs 2015 | Δ Pred 2015
Row 2: Target 2020 (+10yr) | Pred 2020 (+10yr) | Error 2020 | Δ Obs 2020 | Δ Pred 2020
Row 3: Target 2025 (+15yr) | Pred 2025 (+15yr) | Error 2025 | Δ Obs 2025 | Δ Pred 2025  ← NaN
Row 4: Target 2030 (+20yr) | Pred 2030 (+20yr) | Error 2030 | Δ Obs 2030 | Δ Pred 2030  ← NaN
Row 5: Δ Histogram 2015 | Δ Histogram 2020 | Δ Histogram 2025 | Δ Histogram 2030 | [empty]
```

Note: Rows with missing data (2025, 2030) will show as blank/NaN in plots

## 🎯 Complete Implementation Summary

### Phase 1 + Phase 2 Together:

**✅ Dataloader** (`torchgeo_dataloader.py`)
- Temporal sampling with 4 configurations
- Year metadata in batches
- Missing year handling (NaN)

**✅ Training** (`train_lightning.py`)
- Training: temporal sampling enabled
- Validation: fixed years (1990-2000 → 2005-2020)
- Year-specific dataloader configs

**✅ Visualization** (`train_lightning.py`)
- Dynamic titles from batch metadata
- Shows both absolute year + relative horizon
- Supports variable years per sample

**✅ Loss/Metrics** (already working)
- NaN handling via masking
- No changes needed

## 🧪 Testing Checklist

### Before Running Training:

- [ ] Code compiles without syntax errors
- [ ] All imports present

### During First Epoch:

- [ ] Training starts without errors
- [ ] Validation runs with fixed years
- [ ] Year metadata present in batches
- [ ] Loss values are reasonable (not all NaN)

### Check Visualization (W&B):

- [ ] "Predictions_vs_Targets" logged
- [ ] Input titles show correct years (vary per sample in training)
- [ ] Target/Pred titles show year + offset
- [ ] Missing horizons (2025+) handled gracefully

### Check Metrics:

- [ ] `train_mae_5yr`, `train_mae_10yr`, etc. logged
- [ ] `val_mae_5yr`, `val_mae_10yr`, etc. logged
- [ ] Values are reasonable (not all 0 or NaN)

### Training Behavior:

- [ ] Training loss decreases over epochs
- [ ] Validation metrics are consistent (fixed years)
- [ ] No memory issues
- [ ] GPU utilization good

## 🐛 Potential Issues & Solutions

### Issue 1: KeyError on 'input_years' or 'target_years'
**Cause**: Batch doesn't have year metadata

**Solution**: Check dataloader - ensure it's returning metadata
```python
# In batch
print(batch.keys())  # Should include 'input_years', 'target_years'
```

### Issue 2: IndexError on target_years[h_idx]
**Cause**: target_years is wrong shape

**Solution**: Check batch extraction logic (lines 597-599)

### Issue 3: Plots show "None" in titles
**Cause**: Years are None

**Solution**: Check default values in line 594-595

### Issue 4: All targets are NaN for some configs
**Expected**: This is normal for end_year=2015 (only 2020 is valid)

**Not a bug**: Loss will ignore NaN targets automatically

## 📈 Expected Training Output

```bash
Epoch 0:
  train_mae_5yr: 0.045
  train_mae_10yr: 0.052
  train_mae_15yr: 0.058  # May be NaN for some batches
  train_mae_20yr: 0.065  # May be NaN for some batches
  
  val_mae_5yr: 0.048     # Always valid (fixed years)
  val_mae_10yr: 0.055    # Always valid
  val_mae_15yr: 0.061    # Always valid
  val_mae_20yr: 0.069    # Always valid
```

## 🚀 Running the Training

```bash
# Standard training with temporal sampling
python scripts/train_lightning.py \
    --max_epochs 100 \
    --batch_size 8 \
    --train_chips 1000 \
    --val_chips 200

# The dataloader will automatically:
# - Use temporal sampling for training (4 configs)
# - Use fixed years for validation (1990-2000 → 2005-2020)
# - Log visualizations with dynamic years
```

## 🎉 What You Get

1. **4x Training Data Diversity**
   - Same spatial tiles, 4 different temporal contexts
   
2. **Consistent Validation**
   - Always 1990-2000 → 2005-2020
   - Fair comparison across epochs
   
3. **Clear Visualizations**
   - Actual years shown in plots
   - Both absolute and relative time
   
4. **Robust to Missing Data**
   - NaN targets handled automatically
   - Loss masks invalid pixels/horizons

## 📝 Next Steps

1. **Run training** and monitor first few epochs
2. **Check W&B** for visualization quality
3. **Verify metrics** are reasonable
4. **Compare** to previous runs (should improve due to more data)

## 🔧 Troubleshooting

If issues arise:
1. Check `PHASE1_COMPLETE.md` for dataloader details
2. Verify year metadata in batches
3. Test with small dataset first (`--train_chips 10`)
4. Check W&B logs for actual year values

Good luck! 🚀
