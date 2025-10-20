# Major Change: Model Now Predicts Changes (Deltas) Directly

## 🎯 Summary

The model architecture has been fundamentally changed from predicting **absolute HM values** to predicting **HM changes (deltas)** directly. This removes autocorrelation and makes all losses focus on dynamics.

## 📋 Changes Made

### 1. **Loss Computation** (`src/models/lightning_module.py`)

#### Before:
- Model predicted: Absolute HM values
- MSE: On changes (computed from predictions)
- SSIM: On absolute HM values
- Laplacian: On absolute HM values  
- Histogram: On changes (computed from predictions)

#### After:
- Model predicts: **Changes (deltas) directly**
- MSE: On changes ✅
- SSIM: **On changes** (NEW - data_range=2.0)
- Laplacian: **On changes** (NEW)
- Histogram: On changes ✅
- MAE: On absolute HM (monitoring only, reconstructed from last_input + pred_change)

#### Key Code Changes:
```python
# Function signature changed
def _compute_horizon_losses(self, pred_change, target_h, last_input, mask_h, horizon_name=""):
    # pred_change is now the model output (not pred_absolute)
    
    # Reconstruct absolute for MAE monitoring
    pred_absolute = last_input + pred_change
    pred_absolute = torch.clamp(pred_absolute, 0.0, 1.0)  # Clip to [0, 1]
    mae = self.mae_fn(pred_absolute[mask_h], target_h[mask_h])
    
    # SSIM on changes (NEW)
    ssim_val = ssim(pred_change_sanitized, delta_true_sanitized, data_range=2.0)
    
    # Laplacian on changes (NEW)
    lap_loss = self.lap_loss(pred_change_sanitized, delta_true_sanitized, mask=mask_h)
```

### 2. **Weighted/Unweighted Loss Logging**

Added both unweighted and weighted losses for transparency:

**Training Metrics:**
- `train_mse` - Unweighted MSE
- `train_mae` - Unweighted MAE (monitoring, shown in progress bar)
- `train_ssim` - Unweighted SSIM (shown in progress bar)
- `train_ssim_weighted` - SSIM × 2.0
- `train_lap` - Unweighted Laplacian
- `train_lap_weighted` - Laplacian × 1.0
- `train_hist` - Unweighted Histogram
- `train_hist_weighted` - Histogram × 0.67
- `train_total_loss` - Weighted sum (shown in progress bar)

**Console Output** (every 50 batches):
```
[Train] Batch 0:
  MSE: 0.001234
  MAE: 0.005678 (monitoring only)
  SSIM: 0.123456 (×2.0) = 0.246912
  Laplacian: 0.012345 (×1.0) = 0.012345
  Histogram: 0.023456 (×0.67) = 0.015716
  Total Loss: 0.276207
```

### 3. **Visualization Updates** (`scripts/train_lightning.py`)

#### Validation Visualization (line ~385):
```python
# Model outputs changes
pred_changes_all = best_model(input_dynamic_clean, input_static_clean, lonlat=lonlat)

# Get last input
last_input = input_dynamic_clean[:, -1, 0:1, :, :]

# Convert to absolute HM and clip
preds_5yr = torch.clamp(last_input + pred_changes_all[:, 0:1, :, :], 0.0, 1.0)
preds_10yr = torch.clamp(last_input + pred_changes_all[:, 1:2, :, :], 0.0, 1.0)
# ... etc
```

**Result**: Plots still show absolute HM values (as requested)

#### Large-Area Prediction (line ~1163):
```python
# Model outputs changes
batch_pred_changes = infer_model(batch_dyn_tensor, batch_stat_tensor, lonlat=batch_lonlat_tensor)

# Get last input for each tile
last_input_tile = batch_last_input[tile_idx, :hi, :wj].detach().cpu().numpy()
last_input_tile = last_input_tile * hm_std + hm_mean

# Convert change to absolute HM and clip
pred_change = batch_pred_changes[tile_idx, h_idx, :hi, :wj].detach().cpu().numpy()
pred_change = pred_change * hm_std + hm_mean
pred_h = np.clip(last_input_tile + pred_change, 0.0, 1.0)
```

**Result**: Predictions still output absolute HM maps (as requested)

### 4. **Clipping to [0, 1]**

All absolute HM values are now clipped:
- ✅ Loss computation: `torch.clamp(pred_absolute, 0.0, 1.0)`
- ✅ Validation visualization: `torch.clamp(last_input + pred_changes, 0.0, 1.0)`
- ✅ Large-area prediction: `np.clip(last_input_tile + pred_change, 0.0, 1.0)`

## 🔄 What Changed vs What Stayed the Same

### Changed:
1. **Model output**: Now predicts changes (Δ) instead of absolute values
2. **SSIM loss**: Now computed on changes (data_range=2.0)
3. **Laplacian loss**: Now computed on changes
4. **Loss logging**: Added weighted/unweighted versions
5. **Console output**: Shows detailed loss breakdown

### Stayed the Same:
1. **Plots**: Still show absolute HM values (changes converted back)
2. **Prediction outputs**: Still output absolute HM rasters
3. **Model architecture**: Same ConvLSTM structure
4. **Loss weights**: Same (MSE:1.0, SSIM:2.0, Lap:1.0, Hist:0.67)
5. **Data format**: Still normalized with same mean/std

## 📊 Expected Benefits

### 1. **No Autocorrelation**
- Absolute HM values are highly autocorrelated (high HM stays high)
- Changes remove this autocorrelation
- Model focuses on actual dynamics

### 2. **Consistent Loss Focus**
- All losses now on changes (except MAE monitoring)
- MSE, SSIM, Laplacian, Histogram all optimize dynamics
- No conflict between losses

### 3. **Better Spatial Patterns**
- SSIM on changes ensures realistic change patterns
- Laplacian on changes ensures sharp change boundaries
- Not constrained by absolute value autocorrelation

### 4. **Bounded Predictions**
- Clipping ensures final outputs are valid [0, 1]
- Even if predicted changes are extreme

## ⚠️ Potential Issues to Watch

### 1. **SSIM on Changes**
- Changes can be negative (-1 to +1 range)
- Using data_range=2.0 for SSIM
- May behave differently than SSIM on absolute values

### 2. **Clipping Effects**
- If many predictions hit 0 or 1 boundaries, gradients affected
- Monitor how often clipping occurs

### 3. **Normalization**
- Changes still normalized with hm_std
- May need different normalization for changes

## 🧪 Testing Checklist

### Before Training:
- [ ] Check model loads without errors
- [ ] Verify data shapes match expected
- [ ] Confirm loss computation runs

### During Training (First Epoch):
- [ ] Losses are finite (not NaN/Inf)
- [ ] MSE values reasonable (~0.001-0.1)
- [ ] SSIM values reasonable (~0.0-1.0)
- [ ] Console output shows weighted/unweighted
- [ ] Progress bar shows metrics

### After First Validation:
- [ ] Validation runs without errors
- [ ] Plots show reasonable HM values [0, 1]
- [ ] Changes look realistic in delta plots
- [ ] No systematic bias (e.g., all predictions too high/low)

### After Training:
- [ ] Model converges (loss decreases)
- [ ] Validation metrics stable
- [ ] Predictions visually reasonable
- [ ] Large-area prediction works

## 🚀 How to Run

```bash
# Standard training
python scripts/train_lightning.py \
    --max_epochs 100 \
    --batch_size 8 \
    --train_chips 1000 \
    --val_chips 200

# Monitor console for detailed loss breakdown every 50 batches
# Check W&B for weighted/unweighted metrics
```

## 📝 Files Modified

1. **`src/models/lightning_module.py`**
   - Modified `_compute_horizon_losses()` to accept pred_change
   - Changed SSIM to operate on changes (data_range=2.0)
   - Changed Laplacian to operate on changes
   - Added clipping for absolute HM reconstruction
   - Added weighted/unweighted loss logging
   - Added console output every 50 batches

2. **`scripts/train_lightning.py`**
   - Updated validation visualization to convert changes to absolute
   - Updated large-area prediction to convert changes to absolute
   - Added clipping in both places

## 💡 Key Insights

**The hybrid approach is no longer needed!**

Before: Predict absolute, compute some losses on absolute, some on changes
- Pros: Direct supervision, bounded outputs
- Cons: Mixed loss signals, autocorrelation

After: Predict changes, compute all losses on changes, convert to absolute for output
- Pros: All losses aligned, no autocorrelation, better dynamics
- Cons: Need clipping step, slightly more complex
- **Result**: Should perform better!

## 🔍 Monitoring

Watch these metrics in W&B:
- `train_total_loss` - Should decrease steadily
- `train_mae` - Absolute error (interpretable)
- `train_ssim` & `train_ssim_weighted` - Check both
- `train_lap` & `train_lap_weighted` - Check both  
- `train_hist` & `train_hist_weighted` - Should activate after warmup

Console output provides immediate feedback on loss components!
