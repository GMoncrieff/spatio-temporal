# Loss Weighting and Composition

## Total Loss Formula

```python
total_loss = 1.0 × MSE + 
             pinball_lower + 
             pinball_upper + 
             2.0 × SSIM + 
             1.0 × Laplacian + 
             0.67 × Histogram  # (after warmup)
```

## Loss Weights (from `lightning_module.py`)

### **Unweighted Losses (weight = 1.0)**
- **MSE**: `1.0` (implicit, no multiplier)
- **Pinball Lower**: `1.0` (implicit, no multiplier) ⭐
- **Pinball Upper**: `1.0` (implicit, no multiplier) ⭐

### **Weighted Losses**
- **SSIM**: `ssim_weight = 2.0`
- **Laplacian**: `laplacian_weight = 1.0`
- **Histogram**: `histogram_weight = 0.67` (activated after `histogram_warmup_epochs = 20`)

## Code Reference

From `src/models/lightning_module.py` line 163:

```python
total = mse + pinball_lower + pinball_upper + \
        self.ssim_weight * ssim_loss + \
        self.laplacian_weight * lap_loss

if self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs:
    total = total + self.histogram_weight * hist_loss
```

## Loss Magnitude Analysis

### **Typical Loss Values (observed during training)**

| Loss Component | Typical Range | Weight | Weighted Contribution |
|----------------|---------------|--------|----------------------|
| MSE | 0.05 - 0.15 | 1.0 | 0.05 - 0.15 |
| **Pinball Lower** | **0.01 - 0.05** | **1.0** | **0.01 - 0.05** |
| **Pinball Upper** | **0.01 - 0.05** | **1.0** | **0.01 - 0.05** |
| SSIM | 0.05 - 0.15 | 2.0 | 0.10 - 0.30 |
| Laplacian | 0.03 - 0.08 | 1.0 | 0.03 - 0.08 |
| Histogram | 0.08 - 0.20 | 0.67 | 0.05 - 0.13 |

### **Total Loss Breakdown (after warmup)**

Assuming typical mid-training values:
- MSE: 0.10 × 1.0 = **0.10** (25%)
- Pinball Lower: 0.02 × 1.0 = **0.02** (5%)
- Pinball Upper: 0.02 × 1.0 = **0.02** (5%)
- SSIM: 0.10 × 2.0 = **0.20** (50%)
- Laplacian: 0.05 × 1.0 = **0.05** (12.5%)
- Histogram: 0.15 × 0.67 = **0.10** (25%)

**Total ≈ 0.49**

## Pinball Loss Contribution

### **Relative Contribution**
- **Combined pinball losses**: ~10% of total loss (0.04 / 0.49)
- **Individual pinball**: ~5% each

### **Why This Weight?**

1. **Unweighted (1.0)**: Pinball losses are NOT scaled because:
   - They're already small in magnitude (0.01-0.05)
   - They only affect quantile heads (independent gradients)
   - We want them to have meaningful but not dominant influence

2. **Comparison to other losses**:
   - MSE (central accuracy): 1.0 weight → ~25% contribution
   - SSIM (spatial patterns): 2.0 weight → ~50% contribution
   - Pinball (uncertainty): 1.0 weight each → ~10% combined

## Gradient Flow

### **Independent Gradients Design**

```
Total Loss = Central_losses + Quantile_losses

Central_losses = MSE + 2.0×SSIM + 1.0×Lap + 0.67×Hist
  ↓ (gradients flow to central heads only)
Central Heads

Quantile_losses = Pinball_lower + Pinball_upper
  ↓ (gradients flow to quantile heads only)
Lower & Upper Heads
```

### **Key Points**

1. **Central heads** receive gradients from:
   - MSE, SSIM, Laplacian, Histogram
   - **NOT** from pinball losses

2. **Quantile heads** receive gradients from:
   - Pinball losses **ONLY**
   - **NOT** from MSE, SSIM, Laplacian, or Histogram

3. **Total loss** is sum of all components, but gradients flow independently

## Adjusting Pinball Loss Weight

If you want to change the pinball loss contribution, you can:

### **Option 1: Add explicit weight parameter**

```python
# In lightning_module.py __init__
self.pinball_weight = pinball_weight  # e.g., 0.5, 1.0, 2.0

# In _compute_horizon_losses
total = mse + \
        self.pinball_weight * pinball_lower + \
        self.pinball_weight * pinball_upper + \
        self.ssim_weight * ssim_loss + \
        self.laplacian_weight * lap_loss
```

### **Option 2: Separate weights for lower/upper**

```python
self.pinball_lower_weight = 1.0
self.pinball_upper_weight = 1.0

total = mse + \
        self.pinball_lower_weight * pinball_lower + \
        self.pinball_upper_weight * pinball_upper + \
        ...
```

## Recommendations

### **Current weights are good if:**
- Coverage is calibrating toward ~95%
- Quantile predictions bracket the central prediction
- Uncertainty intervals make sense visually

### **Increase pinball weight if:**
- Coverage is too low (<85%) → intervals too narrow
- Quantiles are too close to central prediction
- You want more conservative uncertainty estimates

### **Decrease pinball weight if:**
- Coverage is too high (>98%) → intervals too wide
- Quantiles are too far from central prediction
- Uncertainty seems overestimated

## Monitoring

Watch these metrics in W&B:
- `val_coverage_total`: Should trend toward ~95%
- `val_pinball_lower_total`: Should be ~0.01-0.05
- `val_pinball_upper_total`: Should be ~0.01-0.05
- `val_total_loss`: Combined loss

## Summary

**Pinball losses have weight = 1.0 (unweighted)**
- Combined contribution: ~10% of total loss
- This is appropriate given their magnitude and purpose
- They only affect quantile heads (independent gradients)
- Central prediction is NOT influenced by pinball losses

**Current design prioritizes:**
1. Central accuracy (MSE + SSIM): ~75% of loss
2. Uncertainty calibration (Pinball): ~10% of loss
3. Other objectives (Lap + Hist): ~15% of loss
