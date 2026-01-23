# Preprocessing Cache System

This directory contains scripts to precompute expensive operations that are repeated on every training run. By caching these computations, you can significantly speed up training iterations.

## What Gets Cached

1. **Normalization Statistics** (`normalization_stats.pkl`)
   - Per-variable mean/std for HM target (AA)
   - Per-variable mean/std for dynamic covariates (AG, BU, EX, FR, HI, NS, PO, TI, gdp, population)
   - Per-variable mean/std for static variables (ele, tas, tasmin, pr, dpi_dsi, iucn_nostrict, iucn_strict)

2. **Valid Chip Indices** (`valid_chips_{chip_size}_{stride}_{min_valid_ratio}.pkl`)
   - All chip positions (y_starts, x_starts)
   - Geographic split assignments (train/val/test/calib)
   - Pre-filtered valid chips per split (based on min_valid_ratio)

3. **Histogram Weights** (`histogram_weights.pkl`)
   - Bin weights for each prediction horizon (5yr, 10yr, 15yr, 20yr)
   - Used for balanced histogram loss

## Quick Start

### Run All Preprocessing

```bash
# Run all preprocessing steps (recommended)
python scripts/preprocess/run_all_preprocessing.py

# Force recomputation even if cache exists
python scripts/preprocess/run_all_preprocessing.py --force

# Use specific platform for histogram weights
python scripts/preprocess/run_all_preprocessing.py --platform m1_mac
```

### Run Individual Steps

```bash
# 1. Normalization statistics
python scripts/preprocess/precompute_dataset_stats.py

# 2. Valid chips
python scripts/preprocess/precompute_valid_chips.py \
    --chip_size 128 \
    --stride 128 \
    --min_valid_ratio 0.5

# 3. Histogram weights (requires cached stats and chips)
python scripts/preprocess/precompute_histogram_weights.py \
    --platform m1_mac \
    --num_batches 10
```

## How It Works

### Automatic Cache Loading

The dataloader (`torchgeo_dataloader.py`) automatically checks for cached data:

1. **On first run** (no cache):
   - Computes all statistics from scratch
   - Prints helpful tips to run preprocessing
   - Training starts but initialization is slow (~3-5 minutes)

2. **With cache**:
   - Loads precomputed data instantly
   - Initialization is fast (~10-20 seconds)
   - Training starts immediately

### Cache Location

All cached files are stored in: `data/cache/`

```
data/cache/
├── normalization_stats.pkl              # ~1-2 KB
├── valid_chips_128_128_0.5.pkl         # ~100-500 KB
└── histogram_weights.pkl                # ~1-2 KB
```

## When to Recompute Cache

You should recompute the cache when:

- **Dataset changes**: New Zarr data uploaded
- **Hyperparameters change**:
  - `chip_size` or `stride` → Recompute valid chips
  - `min_valid_ratio` → Recompute valid chips
  - `random_seed` → Recompute valid chips (changes geographic splits)
- **Variables change**: Adding/removing covariates → Recompute normalization stats

## Performance Impact

### Without Cache (First Run)
```
Opening Zarr Dataset (train split)...
Computing chip positions and geographic splits...
  train split: 29030 chips (of 41496 total)
Pre-filtering chips with min_valid_ratio=0.5...
  Sampled validity rate: 27.8%
  Final: 24840 valid chips for train split
Computing normalization statistics...
  [Sampling 256 random chips across all variables]
  AA: mean=0.0933, std=0.1588
=== Dataset ready: 20 chips per epoch ===
Time: ~3-5 minutes
```

### With Cache (Subsequent Runs)
```
Opening Zarr Dataset (train split)...
Loading valid chips from cache...
  train split: 24840 valid chips (from cache)
Loading normalization statistics from cache...
  AA: mean=0.0933, std=0.1588
=== Dataset ready: 20 chips per epoch ===
Time: ~10-20 seconds
```

**Speedup: ~10-15x faster initialization**

## Cache Invalidation

The cache does NOT automatically invalidate. If you change parameters, you must:

1. Delete old cache files manually:
   ```bash
   rm data/cache/*.pkl
   ```

2. Or use `--force` flag:
   ```bash
   python scripts/preprocess/run_all_preprocessing.py --force
   ```

## Advanced Usage

### Custom Parameters

```bash
# Use different chip size/stride
python scripts/preprocess/run_all_preprocessing.py \
    --chip_size 256 \
    --stride 128 \
    --min_valid_ratio 0.3

# Use different Zarr path
python scripts/preprocess/run_all_preprocessing.py \
    --zarr_path s3://my-bucket/hm-data
```

### Debugging

```python
# Load and inspect cached data
import pickle

# Check normalization stats
with open("data/cache/normalization_stats.pkl", "rb") as f:
    stats = pickle.load(f)
    print(f"HM mean: {stats['hm_mean']}")
    print(f"Static vars: {stats['static_var_names']}")

# Check valid chips
with open("data/cache/valid_chips_128_128_0.5.pkl", "rb") as f:
    chips = pickle.load(f)
    print(f"Train chips: {len(chips['valid_chips']['train'])}")
    print(f"Val chips: {len(chips['valid_chips']['val'])}")
```

## Integration with Training

The preprocessing system is fully integrated with `train_lightning.py`. No code changes needed:

```bash
# First run: automatically computes and suggests caching
python scripts/train_lightning.py --max_epochs 20

# After running preprocessing: uses cache automatically
python scripts/preprocess/run_all_preprocessing.py
python scripts/train_lightning.py --max_epochs 20  # Much faster!
```

## Troubleshooting

### Cache not loading

**Problem**: Dataloader still computes stats even though cache exists

**Solution**: Check that cache file parameters match:
- Chip size must match
- Stride must match  
- Min valid ratio must match

### Out of memory during preprocessing

**Problem**: Preprocessing crashes with OOM error

**Solution**: Reduce `--stat_samples` or `--check_ratio`:
```bash
python scripts/preprocess/precompute_dataset_stats.py --stat_samples 50
python scripts/preprocess/precompute_valid_chips.py --check_ratio 0.1
```

### Stale cache

**Problem**: Dataset updated but using old statistics

**Solution**: Force recomputation:
```bash
python scripts/preprocess/run_all_preprocessing.py --force
```
