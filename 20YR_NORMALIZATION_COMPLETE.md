# 20-Year Horizon Normalization - COMPLETE ✅

## Implementation
Using **20-year horizon statistics** for normalizing all horizons (5yr, 10yr, 15yr, 20yr).

## Why This Approach?
- **Covers maximum expected changes**: 20-year statistics span the largest changes
- **Consistent normalization**: All horizons normalized with the same scale
- **Simpler than per-horizon**: Single mean/std pair instead of 4
- **Still learnable**: 20yr std ≈ 0.06-0.08 gives normalized targets in ±3-5 std range

## Files Modified

### torchgeo_dataloader.py ✅
**Lines 129-161**: Delta statistics computation
```python
offset_20yr = 4  # 20-year horizon (4 timesteps)
for t1 in range(len(self._hm_files) - offset_20yr):
    t2 = t1 + offset_20yr
    # Sample 20-year changes...
    
self.delta_mean = np.mean(delta_samples)
self.delta_std = np.std(delta_samples) + 1e-8
```

**Line 401**: Apply uniform normalization
```python
target_delta = (delta_raw - self.delta_mean) / self.delta_std
```

### train_lightning.py ✅
Already uses `delta_mean` and `delta_std` throughout:
- Line 202-203: Get from dataloader
- Line 220-221: Pass to model
- Lines 410-413, 440-443: Denormalize predictions/targets
- Line 1061, 1259: Large-area prediction

### lightning_module.py ✅
Already uses `delta_mean` and `delta_std`:
- Line 35-36, 61-62: Store as instance variables
- Lines 125-126, 153-154: Denormalize for MAE and histogram loss
- Lines 240, 342: Pass to _compute_horizon_losses

## Expected Results

### Before (year-to-year normalization):
```
delta_std = 0.01266 (consecutive years)
20-year change of 0.36: (0.36 - 0.002) / 0.01266 = 28.3 std ❌ (unlearnable)
```

### After (20-year normalization):
```
delta_std ≈ 0.070 (20-year horizon)
5-year change of 0.08:   (0.08 - 0.002) / 0.070 ≈ 1.1 std ✅
10-year change of 0.15:  (0.15 - 0.002) / 0.070 ≈ 2.1 std ✅
15-year change of 0.25:  (0.25 - 0.002) / 0.070 ≈ 3.5 std ✅
20-year change of 0.36:  (0.36 - 0.002) / 0.070 ≈ 5.1 std ✅
```

All horizons now in learnable range (±1-5 std deviations)!

## Next Steps
1. Delete old checkpoints (used wrong normalization)
2. Run training
3. Check console output for: `Delta (20yr): mean=X, std=Y`
4. Verify hexbin plots show full prediction range

Ready to train! 🚀
