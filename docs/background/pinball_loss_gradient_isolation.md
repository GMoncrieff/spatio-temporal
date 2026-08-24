# Pinball Loss Gradient Isolation

## Overview
Modified the training process to ensure pinball loss gradients **only affect quantile head parameters**, not the shared ConvLSTM backbone or central prediction heads.

## Architecture
```
Input → ConvLSTM (backbone) → last_hidden → {
    ├── central_heads[0-3]  (4 heads for central predictions)
    ├── lower_heads[0-3]    (4 heads for 2.5% quantile)
    └── upper_heads[0-3]    (4 heads for 97.5% quantile)
}
```

## Gradient Flow Design

### Before Changes
All losses backpropagated through entire model:
```
Central loss (MSE+SSIM+Lap+Hist) → ConvLSTM + all heads
Pinball loss                      → ConvLSTM + all heads
```
**Problem**: Pinball loss affected backbone and central heads, mixing uncertainty estimation with accuracy optimization.

### After Changes
Separated gradient flow using manual optimization:
```
Central loss (MSE+SSIM+Lap+Hist) → ConvLSTM + central_heads ONLY
Pinball loss                      → lower_heads + upper_heads ONLY
```

**Benefits**:
1. **Backbone focus**: ConvLSTM learns features optimized for accuracy and spatial quality only
2. **Head specialization**: 
   - Central heads optimize for MSE+SSIM+Laplacian+Histogram
   - Quantile heads optimize purely for calibration (pinball loss)
3. **No gradient interference**: Quantile estimation doesn't corrupt feature learning

## Implementation Details

### Manual Optimization Process
Modified `SpatioTemporalLightningModule.training_step()`:

```python
# Step 1: Set manual optimization flag
self.automatic_optimization = False

# Step 2: In training_step, perform manual backward passes
opt = self.optimizers()

# Compute central loss (MSE + SSIM + Laplacian + Histogram)
central_loss = avg_mse + ssim_weight * avg_ssim + laplacian_weight * avg_lap + ...

# Compute pinball loss (lower + upper quantiles)
pinball_loss = avg_pinball_lower + avg_pinball_upper

# Step 3: Backprop central loss through all parameters
opt.zero_grad()
self.manual_backward(central_loss)

# Step 4: Save gradients for non-quantile parameters
quantile_params = set(lower_heads.parameters() + upper_heads.parameters())
saved_grads = {}
for name, param in self.named_parameters():
    if param not in quantile_params and param.grad is not None:
        saved_grads[name] = param.grad.clone()

# Step 5: Backprop pinball loss (accumulates gradients)
self.manual_backward(pinball_loss)

# Step 6: Restore saved gradients for non-quantile parameters
# This removes pinball loss gradients from backbone and central heads
for name, param in self.named_parameters():
    if name in saved_grads:
        param.grad = saved_grads[name]

# Step 7: Optimizer step with isolated gradients
opt.step()
```

### Key Parameters Affected

**Parameters that receive ONLY central loss gradients:**
- `model.convlstm.*` (ConvLSTM backbone)
- `model.central_heads.*` (central prediction heads)
- `model.location_encoder.*` (if enabled)

**Parameters that receive ONLY pinball loss gradients:**
- `model.lower_heads.*` (2.5% quantile heads)
- `model.upper_heads.*` (97.5% quantile heads)

**Parameters that receive NO gradients from either loss:**
- None (all parameters participate in at least one loss)

## Testing Recommendations

To verify gradient isolation is working:

```python
# Add to training_step for debugging (first batch only)
if batch_idx == 0 and self.current_epoch == 0:
    # Check ConvLSTM gradients are from central loss only
    for name, param in self.model.convlstm.named_parameters():
        if param.grad is not None:
            print(f"ConvLSTM {name}: grad norm = {param.grad.norm().item():.6f}")
    
    # Check quantile heads have gradients
    for name, param in self.model.lower_heads.named_parameters():
        if param.grad is not None:
            print(f"Lower head {name}: grad norm = {param.grad.norm().item():.6f}")
```

## Performance Implications

**Computational Cost:**
- Minimal overhead: one additional backward pass + gradient storage/restoration
- Extra memory: temporary storage of ~half the model's gradients
- Estimated overhead: <5% increase in training time

**Model Performance:**
- Expected: Better quantile calibration (coverage closer to 95%)
- Expected: No degradation in central prediction quality
- Expected: Faster convergence for uncertainty bounds

## Compatibility

- ✅ PyTorch Lightning automatic logging
- ✅ Model checkpointing
- ✅ Learning rate schedulers
- ✅ Gradient clipping (if enabled in Trainer)
- ✅ Mixed precision training (with `self.manual_backward()`)
- ⚠️  Multiple optimizers: currently uses single optimizer; would need modification for separate optimizers

## References

- PyTorch Lightning Manual Optimization: https://lightning.ai/docs/pytorch/stable/common/optimization.html
- Pinball Loss: Koenker & Bassett (1978) "Regression Quantiles"
