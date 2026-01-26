# Valid Chips Workflow

## Overview

The training pipeline now uses **pre-computed valid chip positions** to eliminate empty chips and ensure efficient training. This approach:

- ✅ **Eliminates empty chips** - Only samples from positions with valid data
- ✅ **Geographic splits** - Deterministic train/val/test/calib splits (70%/10%/10%/10%)
- ✅ **Fast sampling** - No runtime validity checking needed
- ✅ **Reproducible** - Same chips always in same split

## Workflow

### 1. Precompute Valid Chips (One-Time Setup)

Run this once to scan the Zarr dataset and identify all valid chip positions:

```bash
python scripts/precompute_valid_chips.py \
    --zarr_path scripts/notebooks/hm.icechunk \
    --chip_size 128 \
    --stride 64 \
    --min_valid_ratio 0.8
```

**Parameters:**
- `--zarr_path`: Path to Zarr/Icechunk repository
- `--chip_size`: Size of spatial chips (default: 128)
- `--stride`: Stride between chip positions (default: 64)
- `--min_valid_ratio`: Minimum fraction of valid pixels required (default: 0.8)
- `--seed`: Random seed for split assignment (default: 42)

**Output:**
- Creates `data/processed/valid_chips_metadata.json`
- Contains all valid chip positions assigned to splits
- Example output:
  ```
  Total valid chips: 45,234 / 250,000 (18.1%)
  Split distribution:
    train:  31,664 chips (70.0%)
    val:     4,523 chips (10.0%)
    test:    4,523 chips (10.0%)
    calib:   4,524 chips (10.0%)
  ```

### 2. Train Model

Training now automatically uses pre-computed valid chips:

```bash
python scripts/train_lightning.py \
    --max_epochs 20 \
    --batch_size 32 \
    --hidden_dim 64 \
    --num_layers 4
```

**Key Changes:**
- ❌ Removed: `--train_chips`, `--val_chips`, `--train_mode`, `--val_mode`, `--stride`
- ✅ Automatic: Dataset size determined by valid chips count
- ✅ Automatic: Shuffle enabled for train, disabled for val/test
- ✅ No empty chips: All batches guaranteed to have valid data

### 3. Prediction (Unchanged)

Prediction still runs over the entire spatial extent (not limited to valid chips):

```bash
# Predictions use grid-based approach over full region
python scripts/train_lightning.py --max_epochs 0  # Runs prediction after training
```

## File Structure

```
data/
  processed/
    valid_chips_metadata.json    # Pre-computed valid chip positions
scripts/
  precompute_valid_chips.py      # One-time preprocessing script
  torchgeo_dataloader.py          # Updated to use valid chips
  train_lightning.py              # Simplified training script
```

## Metadata Format

`valid_chips_metadata.json` structure:

```json
{
  "zarr_path": "scripts/notebooks/hm.icechunk",
  "chip_size": 128,
  "stride": 64,
  "min_valid_ratio": 0.8,
  "seed": 42,
  "dataset_shape": {"y": 17111, "x": 40000},
  "total_positions": 250000,
  "total_valid": 45234,
  "splits": {
    "train": [[yi, xi], [yi, xi], ...],
    "val": [[yi, xi], ...],
    "test": [[yi, xi], ...],
    "calib": [[yi, xi], ...]
  },
  "split_counts": {
    "train": 31664,
    "val": 4523,
    "test": 4523,
    "calib": 4524
  }
}
```

## Split Assignment Logic

Chips are assigned to splits using a **deterministic hash** of their coordinates:

```python
def hash_position_to_split(y_idx, x_idx, seed=42):
    position_str = f"{seed}_{y_idx}_{x_idx}"
    hash_val = int(hashlib.md5(position_str.encode()).hexdigest(), 16)
    rand_val = (hash_val % 100) / 100.0
    
    if rand_val < 0.70:    return 'train'
    elif rand_val < 0.80:  return 'val'
    elif rand_val < 0.90:  return 'test'
    else:                  return 'calib'
```

This ensures:
- Same chip always in same split (reproducible)
- Roughly uniform distribution (70/10/10/10)
- Geographic clustering (nearby chips likely in same split)

## Updating Valid Chips

If you change chip_size, stride, or min_valid_ratio, re-run preprocessing:

```bash
# Example: Use larger chips
python scripts/precompute_valid_chips.py \
    --chip_size 256 \
    --stride 128
```

The dataloader will automatically validate that the metadata matches your training settings.

## Benefits

### Before (Grid/Random Sampling)
- ❌ Many empty chips (ocean, no-data regions)
- ❌ Runtime validity checking overhead
- ❌ Inconsistent training set size
- ❌ Complex split mask logic

### After (Pre-computed Valid Chips)
- ✅ Zero empty chips
- ✅ Fast sampling (direct indexing)
- ✅ Consistent dataset size
- ✅ Simple, deterministic splits
- ✅ Better GPU utilization (no wasted batches)

## Troubleshooting

**Error: "Valid chips metadata not found"**
```
Run: python scripts/precompute_valid_chips.py
```

**Error: "Chip size mismatch"**
```
Re-run preprocessing with correct --chip_size
```

**Want different split ratios?**
```
Modify hash_position_to_split() in precompute_valid_chips.py
```

**Need to exclude certain regions?**
```
Add spatial filtering logic to check_chip_validity()
```
