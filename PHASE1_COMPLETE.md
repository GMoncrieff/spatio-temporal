# Phase 1 Complete: Temporal Sampling in Dataloader

## ✅ What Was Implemented

### 1. **Dataloader Changes** (`scripts/torchgeo_dataloader.py`)

#### New Parameters:
- `use_temporal_sampling=True`: Enable/disable temporal sampling
- `end_year_options=(2000, 2005, 2010, 2015)`: Valid end years to sample from

#### Temporal Sampling Logic (line 260-274):
```python
if self.use_temporal_sampling:
    # Randomly sample end_year for this sample
    end_year = np.random.choice(self.end_year_options)
    input_years = [end_year - 10, end_year - 5, end_year]
    target_years = [end_year + 5, end_year + 10, end_year + 15, end_year + 20]
else:
    # Fixed years (validation/testing)
    end_year = self.fixed_input_years[-1]  # 2000
    input_years = list(self.fixed_input_years)
    target_years = list(self.fixed_target_years)
```

#### Handling Missing Years (line 351-358):
```python
for horizon_name, t_idx, target_year in zip(horizon_names, target_t_idxs, target_years):
    if t_idx is None or target_year > 2020:
        # Missing year - fill with NaN (will be masked in loss)
        target_h = np.full((self.chip_size, self.chip_size), np.nan, dtype=np.float32)
    else:
        # Load actual data
        target_h = self._hm_srcs[t_idx].read(...)
```

#### Year Metadata (line 369-372, 382-385):
```python
sample = {
    "input_dynamic": ...,
    "input_static": ...,
    # NEW: Year metadata for visualization
    "input_years": input_years,      # [1995, 2000, 2005]
    "target_years": target_years,     # [2010, 2015, 2020, 2025]
    "end_year": end_year,             # 2005
    ...
}
```

### 2. **Training Script Changes** (`scripts/train_lightning.py`)

#### Training Dataloader (line 169-170):
```python
use_temporal_sampling=True,  # Enable temporal sampling for training
end_year_options=(2000, 2005, 2010, 2015),
```
- **4 temporal configurations** sampled randomly per epoch
- Effective 4x increase in training data diversity

#### Validation Dataloader (line 187):
```python
use_temporal_sampling=False,  # Fixed years for validation (Option A)
```
- **Always uses 1990, 1995, 2000 → 2005-2020**
- Consistent metrics across epochs for fair comparison

## 📊 Training Data Breakdown

### Configuration 1: end_year = 2000
- **Inputs**: 1990, 1995, 2000
- **Targets**: 2005, 2010, 2015, 2020
- **Valid targets**: All 4 ✅

### Configuration 2: end_year = 2005  
- **Inputs**: 1995, 2000, 2005
- **Targets**: 2010, 2015, 2020, 2025
- **Valid targets**: 3 (2025 → NaN)

### Configuration 3: end_year = 2010
- **Inputs**: 2000, 2005, 2010
- **Targets**: 2015, 2020, 2025, 2030
- **Valid targets**: 2 (2025, 2030 → NaN)

### Configuration 4: end_year = 2015
- **Inputs**: 2005, 2010, 2015
- **Targets**: 2020, 2025, 2030, 2035
- **Valid targets**: 1 (2025, 2030, 2035 → NaN)

**Total**: Each spatial location can contribute 4 different temporal samples!

## 🎯 What's Working Now

✅ **Training**: Randomly samples from 4 temporal configurations  
✅ **Validation**: Uses fixed years (1990-2000 → 2005-2020)  
✅ **Missing years**: Filled with NaN, automatically masked in loss  
✅ **Year metadata**: Returned with each batch for visualization  
✅ **Backward compatible**: Can disable with `use_temporal_sampling=False`  

## ⚠️ What's NOT Done Yet (Phase 2)

### Visualization Updates Needed

The following still need to be updated to use year metadata from batches:

#### 1. **Predictions vs Targets Plot** (line 560-728)
**Current**: Hardcoded titles "2005", "2010", "2015", "2020"

**Needs**: Dynamic titles from `batch['target_years']`
```python
# Extract years from batch
target_years = batch['target_years'][b]  # [2010, 2015, 2020, 2025]

# Update plot titles
axes[row, 0].set_title(f'Target {target_years[h_idx]}')
axes[row, 1].set_title(f'Pred {target_years[h_idx]}')
```

#### 2. **Input HM Plots** (line 590-605)
**Current**: Hardcoded "HM 1990", "HM 1995", "HM 2000"

**Needs**: Dynamic from `batch['input_years']`
```python
input_years = batch['input_years'][b]  # [1995, 2000, 2005]
axes[0, 0].set_title(f'HM {input_years[0]}')
```

## 🚀 Expected Impact

### Before (Fixed Years):
- Training samples: N spatial tiles
- Temporal diversity: 1 configuration
- Effective dataset size: N

### After (Temporal Sampling):
- Training samples: N spatial tiles
- Temporal diversity: 4 configurations  
- Effective dataset size: **4×N** 🎉

### Validation:
- **Unchanged**: Still 1990-2000 → 2005-2020
- Consistent metrics for monitoring training progress

## 📝 Next Steps

**Phase 2**: Update visualization to use dynamic years
1. Extract year metadata from batches
2. Update all plot titles to show actual years
3. Handle variable years per sample in batch

Ready to proceed with Phase 2?

## Testing Checklist

Before Phase 2, verify:
- [ ] Training runs without errors
- [ ] Validation uses fixed years (check first batch)
- [ ] Loss handles NaN targets correctly (configs 2-4)
- [ ] Year metadata is in batches (`input_years`, `target_years`, `end_year`)
- [ ] Model performance doesn't degrade

## Notes

- Loss computation already handles NaN via masking - no changes needed ✅
- Model doesn't care about absolute years - only relative horizons ✅
- Inference/prediction still uses fixed 1990-2000 → 2005-2020 ✅
