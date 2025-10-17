# Temporal Sampling Strategy - Implementation Plan

## Current Approach (Fixed)
- **Inputs**: Always 1990, 1995, 2000
- **Targets**: Always 2005, 2010, 2015, 2020
- **Training samples**: ~N samples (one per spatial tile)

## New Approach (Dynamic)
- **Randomly select end_year** from [2000, 2005, 2010, 2015] per training sample
- **Compute inputs**: [end_year - 10, end_year - 5, end_year]
- **Compute targets**: [end_year + 5, end_year + 10, end_year + 15, end_year + 20]
- **Training samples**: ~4×N samples (4 temporal configurations × spatial tiles)

## Example Configurations

| End Year | Input Years | Target Years | Valid Targets | Missing Targets |
|----------|-------------|--------------|---------------|-----------------|
| 2000 | 1990, 1995, 2000 | 2005, 2010, 2015, 2020 | All 4 | None |
| 2005 | 1995, 2000, 2005 | 2010, 2015, 2020, 2025 | 3 | 2025 |
| 2010 | 2000, 2005, 2010 | 2015, 2020, 2025, 2030 | 2 | 2025, 2030 |
| 2015 | 2005, 2010, 2015 | 2020, 2025, 2030, 2035 | 1 | 2025, 2030, 2035 |

**Note**: Missing targets (beyond 2020) will be filled with NaN and masked during loss computation.

## Benefits
1. **4x more training samples** - Better data utilization
2. **Temporal robustness** - Model learns from different temporal contexts
3. **Recent data** - Can train on more recent observations (2010, 2015)
4. **Better generalization** - Model sees variety of temporal patterns

## Files Requiring Changes

### 1. `scripts/torchgeo_dataloader.py` (Major changes)
**Current**:
- Fixed `fixed_input_years = (1990, 1995, 2000)`
- Fixed `fixed_target_years = (2005, 2010, 2015, 2020)`

**New**:
- Remove fixed years
- Add `end_year_options = [2000, 2005, 2010, 2015]`
- In `__getitem__`:
  - Randomly sample `end_year` from options
  - Compute `input_years = [end_year - 10, end_year - 5, end_year]`
  - Compute `target_years = [end_year + 5, end_year + 10, end_year + 15, end_year + 20]`
  - Load data for computed years
  - For missing years (> 2020), return NaN arrays

**Changes needed**:
- Line ~70-80: Add `end_year_options` parameter
- Line ~250-340 (`__getitem__`): Add temporal sampling logic
- Handle missing years gracefully (return NaN)

### 2. `scripts/train_lightning.py` (Moderate changes)

#### Visualization (lines 560-728)
**Current**: Hardcoded titles "2005", "2010", "2015", "2020"

**New**: 
- Extract actual years from batch metadata
- Update titles to show actual years: "Target 2010", "Pred 2010", etc.
- Handle variable target years per sample

**Changes needed**:
- Store actual years in batch (add to dataloader output)
- Update plot titles dynamically
- Handle missing horizons (some samples may have NaN for later horizons)

#### Full Validation Metrics (lines 432-539)
**Current**: Fixed horizon processing

**New**:
- Still accumulate per nominal horizon (5yr, 10yr, 15yr, 20yr)
- But now represents different absolute years per sample
- Loss calculation already handles NaN, so should work

**No major changes needed** - loss already masks NaN targets

#### Hexbin/Histogram (lines 730-869)
**Current**: Fixed to 20yr horizon

**New**: Already updated to per-horizon, should work as-is

**No changes needed** - already per-horizon

#### Large-area Prediction (lines 857-1188)
**Keep as-is**: Use fixed 1990, 1995, 2000 → 2005-2020 for production predictions

**No changes needed** - inference uses fixed years for consistency

### 3. `src/models/lightning_module.py` (No changes needed)
- Loss computation already handles NaN targets via masking
- Model doesn't care about absolute years, only relative horizons
- Should work as-is ✅

## Implementation Strategy

### Phase 1: Update Dataloader (torchgeo_dataloader.py)
1. Add `end_year_options` parameter to `__init__`
2. Modify `__getitem__` to:
   - Sample random `end_year`
   - Compute `input_years` and `target_years`
   - Load data for those years
   - Handle missing years (fill with NaN)
3. Return year metadata with batch (for plotting)

### Phase 2: Update Visualization (train_lightning.py)
1. Extract year information from batch
2. Update plot titles to show actual years
3. Test with small dataset

### Phase 3: Testing
1. Verify all 4 temporal configurations work
2. Check loss computation handles NaN correctly
3. Verify metrics are computed correctly
4. Test visualization shows correct years

### Phase 4: Validation Strategy Decision
**Question**: Should validation also randomize temporal sampling?

**Option A (Recommended)**: Fixed validation
- Validation always uses end_year = 2000 (same as current)
- Consistent evaluation across epochs
- Comparable metrics over time

**Option B**: Random validation
- Validation also samples random end_years
- More representative of training distribution
- But metrics vary more between epochs

**Recommendation**: Use Option A (fixed validation) for stability

## Data Loading Details

### Handling Missing Years

```python
# In __getitem__
for target_year in target_years:
    if target_year > 2020:
        # Fill with NaN - no data available
        target_arr = np.full((H, W), np.nan, dtype=np.float32)
    else:
        # Load actual data
        year_idx = year_to_idx[target_year]
        target_arr = self._hm_srcs[year_idx].read(...)
```

### Year Metadata

Return actual years with batch:
```python
return {
    'input_dynamic': ...,
    'input_static': ...,
    'target_5yr': ...,
    'target_10yr': ...,
    'target_15yr': ...,
    'target_20yr': ...,
    'input_years': input_years,  # NEW: [1995, 2000, 2005]
    'target_years': target_years,  # NEW: [2010, 2015, 2020, 2025]
    'end_year': end_year,  # NEW: 2005
    ...
}
```

## Potential Issues & Solutions

### Issue 1: Component variables missing for early/late years
**Solution**: Already handled - NaN imputation strategy in place

### Issue 2: Batch mixing different temporal configs
**Solution**: This is fine - model processes each sample independently

### Issue 3: Metrics interpretation
**Solution**: Metrics still represent "5yr ahead", "10yr ahead", etc., just from different base years

### Issue 4: Visualization showing different years per sample
**Solution**: Show year ranges in plot titles: "2010/2015/2020" or show individual sample's years

## Migration Path

1. **Backward compatible**: Add flag `use_temporal_sampling=False` (default)
2. **Test thoroughly**: Verify with small dataset first
3. **Gradual rollout**: Enable for training, keep validation fixed
4. **Monitor metrics**: Ensure no unexpected behavior

## Expected Impact

- **Training time**: Similar (same number of forward passes per epoch)
- **Effective data**: 4x increase in temporal diversity
- **Memory**: No change (batch size unchanged)
- **Model performance**: Should improve due to more diverse training signal
- **Validation metrics**: May change slightly (different data distribution)

## Questions to Resolve

1. **Validation sampling**: Fixed (end_year=2000) or random?
   - **Recommendation**: Fixed for metric consistency
   
2. **Inference**: Keep 1990-2000 → 2005-2020?
   - **Recommendation**: Yes, for production consistency
   
3. **Weight by valid targets**: Should samples with more valid targets (e.g., end_year=2000) have different weight?
   - **Recommendation**: No, loss already averages over valid horizons
   
4. **Plot labels**: Show actual years or relative horizons?
   - **Recommendation**: Both - "5yr: 2010" or "2010 (+5yr)"

## Next Steps

Ready to proceed? I can implement this in phases:
1. First update dataloader with temporal sampling
2. Then update visualization to handle variable years
3. Finally test and validate

Would you like me to start with Phase 1 (dataloader changes)?
