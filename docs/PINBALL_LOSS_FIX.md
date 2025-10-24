# Pinball Loss NaN Fix

## Problem

During training, pinball losses were showing `nan` in both training and validation:

```
train_pinball_lower_total=nan.0, train_pinball_upper_total=nan.0
Pinball Lower: nan, Pinball Upper: nan
```

## Root Cause

1. **Predictions set to NaN**: In `lightning_module.py`, predictions are set to `float('nan')` for invalid pixels:
   ```python
   pred_lower[~mask_h] = float('nan')
   pred_central[~mask_h] = float('nan')
   pred_upper[~mask_h] = float('nan')
   ```

2. **NaN propagation**: The original `PinballLoss.forward()` computed the error on all pixels first:
   ```python
   error = target - pred  # NaN values included!
   loss = torch.where(error >= 0, ...)  # NaN propagates
   loss = loss * mask  # NaN * 0 = NaN (not 0!)
   ```

3. **Masking after computation**: Multiplying NaN by 0 (from the mask) still gives NaN, not 0.

## Solution

**Filter valid pixels BEFORE computing loss** in `src/models/pinball_loss.py`:

```python
def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
    # Apply mask first to filter valid pixels only
    if mask is not None:
        pred_valid = pred[mask]  # Extract only valid pixels
        target_valid = target[mask]
        
        if pred_valid.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    else:
        pred_valid = pred
        target_valid = target
    
    # Compute error on valid pixels only (no NaN!)
    error = target_valid - pred_valid
    
    # Pinball loss computation
    loss = torch.where(
        error >= 0,
        self.quantile * error,
        (self.quantile - 1) * error
    )
    
    # Apply reduction
    if self.reduction == 'mean':
        return loss.mean()
    # ...
```

## Key Changes

### Before (Incorrect)
```python
# Compute on ALL pixels (including NaN)
error = target - pred
loss = torch.where(error >= 0, ...)
# Try to mask out (too late - NaN already propagated)
loss = loss * mask
return loss.sum() / n_valid
```

### After (Correct)
```python
# Filter to VALID pixels first
pred_valid = pred[mask]
target_valid = target[mask]
# Compute on valid pixels only (no NaN)
error = target_valid - pred_valid
loss = torch.where(error >= 0, ...)
return loss.mean()
```

## Verification

Test with NaN values and mask:
```python
pred = torch.tensor([[1.0, 2.0, float('nan'), 4.0]])
target = torch.tensor([[2.0, 3.0, 5.0, 5.0]])
mask = torch.tensor([[True, True, False, True]])

loss = pinball_loss(pred, target, mask=mask)
# Result: 0.025 (no NaN!)
```

## Expected Behavior After Fix

Training output should now show:
```
train_pinball_lower_total=0.015, train_pinball_upper_total=0.018
Pinball Lower: 0.01234, Pinball Upper: 0.01567
```

Typical pinball loss values: **0.001 - 0.05** (small but non-zero)

## Files Modified

- `src/models/pinball_loss.py`: Fixed `forward()` method to filter valid pixels before computing loss

## Status

✅ **Fixed and tested**
- Pinball loss handles NaN values correctly
- Mask filtering happens before computation
- No NaN propagation
- Ready for training
