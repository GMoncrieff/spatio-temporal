# Histogram Visualization in Prediction Plots

## Overview

When the histogram head is enabled, the prediction visualization includes an additional row showing histogram comparisons for each tile.

## Visualization Layout

### Without Histogram Head (2 rows × 5 columns)
```
Row 1: Input HM t1 | Input HM t2 | Input HM t3 | Elevation | (empty)
Row 2: Target HM   | Pred HM     | Error       | Obs Delta | Pred Delta
```

### With Histogram Head (3 rows × 5 columns)
```
Row 1: Input HM t1      | Input HM t2      | Input HM t3      | Elevation    | (empty)
Row 2: Target HM        | Pred HM          | Error            | Obs Delta    | Pred Delta
Row 3: Observed Hist    | Hist Head Pred   | Pixel Binned     | Comparison   | Bin Edges
```

## Third Row Details (Histogram Comparison)

### Panel 1: Observed Change Histogram (Blue)
- **Source**: Ground truth changes (target - last input)
- **Computation**: Bins continuous pixel changes into histogram
- **Purpose**: Shows the true distribution of changes in the tile

### Panel 2: Histogram Head Prediction (Red)
- **Source**: Direct output from histogram prediction head
- **Computation**: Forward pass through tile embedding → MLP → softmax
- **Purpose**: Shows what the histogram head predicts the distribution should be

### Panel 3: Pixel Predictions Binned (Green)
- **Source**: Pixel-level predictions from regression head
- **Computation**: Bins predicted pixel changes into histogram
- **Purpose**: Shows the distribution implied by pixel predictions

### Panel 4: Overlay Comparison
- **Shows**: All three histograms side-by-side with different colors
- **Purpose**: Easy visual comparison of consistency between:
  - Ground truth (blue)
  - Histogram head (red)
  - Pixel predictions (green)

### Panel 5: Bin Edges
- **Shows**: Text display of bin edge values
- **Default**: `[-1.0, -0.05, -0.005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]`

## Interpretation

### Good Consistency
When the model is well-trained, you should see:
- **Red bars close to blue bars**: Histogram head accurately predicts tile distribution
- **Green bars close to blue bars**: Pixel predictions aggregate to correct distribution
- **Red and green bars similar**: Pixel and histogram heads are self-consistent

### Poor Consistency
Warning signs:
- **Red far from blue**: Histogram head is not learning tile distributions
- **Green far from blue**: Pixel predictions don't match ground truth distribution
- **Red and green diverge**: Heads are making inconsistent predictions

## Example Use Cases

### 1. Debugging Histogram Head
If red bars are uniform or random:
- Histogram head may not be training properly
- Try increasing `--histogram_weight`
- Check that histogram loss is decreasing

### 2. Checking Pixel-Histogram Consistency
If green and red bars diverge significantly:
- Pixel and histogram heads are not aligned
- This is expected in Step 1 (no consistency loss yet)
- Will be addressed in Step 2 with consistency regularization

### 3. Identifying Bias
If all predictions (red/green) systematically over/under-predict certain bins:
- Model may have systematic bias
- Check data distribution and normalization
- Consider adjusting bin edges

## Technical Details

### Histogram Computation
All histograms use the same bin edges defined in the model:
```python
histogram_bins = [-1.0, -0.05, -0.005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
```

This creates 9 bins representing different change magnitudes.

### Masking
All histograms respect the validity mask:
- Only valid pixels (finite target, finite inputs) are included
- Invalid pixels are excluded from histogram computation
- Ensures fair comparison across all three distributions

### Normalization
All histograms are normalized to sum to 1 (proportions, not counts).

## Logging

These visualizations are logged to W&B under the key `"Predictions_vs_Targets"` at the end of training.

Each sample in the first validation batch gets its own visualization with histogram comparison.

## Code Location

Implementation: `scripts/train_lightning.py` lines ~397-467

The visualization is automatically enabled when `--use_histogram_head true` is set.
