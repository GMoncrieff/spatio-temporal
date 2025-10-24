# Documentation Index

This directory contains comprehensive documentation for the Spatio-Temporal Human Modification Forecasting project.

## Core Architecture & Implementation

### [QUANTILE_PREDICTION_IMPLEMENTATION.md](QUANTILE_PREDICTION_IMPLEMENTATION.md)
Complete guide to the quantile regression implementation.
- **Topics**: Independent prediction heads, pinball loss, coverage calibration
- **Status**: ✅ Complete
- **Key features**: 12 output channels (3 quantiles × 4 horizons), uncertainty quantification

### [INDEPENDENT_HEADS_IMPLEMENTATION.md](INDEPENDENT_HEADS_IMPLEMENTATION.md)
Details on the independent prediction heads architecture.
- **Topics**: Gradient separation, central vs quantile heads, architectural decisions
- **Status**: ✅ Complete
- **Key design**: Central heads (full-size), quantile heads (smaller, independent)

### [LOSS_WEIGHTING_EXPLAINED.md](LOSS_WEIGHTING_EXPLAINED.md)
Loss function composition and gradient flow.
- **Topics**: Loss weights, contributions, optimization objectives
- **Current weights**: MSE (1.0), SSIM (2.0), Laplacian (1.0), Histogram (0.67), Pinball (1.0 each)
- **Key insight**: Independent gradients - central and quantile heads don't interfere

## Data & Preprocessing

### [NORMALIZATION_BUG_FIX.md](NORMALIZATION_BUG_FIX.md)
Per-variable normalization for mixed-scale inputs.
- **Problem**: GDP (billions) and HM components (0-1) had vastly different scales
- **Solution**: Separate mean/std for each variable → ~N(0,1)
- **Variables**: 11 dynamic (HM + covariates), 7 static (climate, terrain, protected areas)

### [MASKING_STRATEGY.md](MASKING_STRATEGY.md)
NaN handling and masking approach.
- **Topics**: Valid pixel masks, edge artifacts, NaN propagation
- **Current approach**: Sanitize NaN to 0.0 before model forward, apply mask to loss

## Predictions

### [PREDICTION_YEARS_UPDATE.md](PREDICTION_YEARS_UPDATE.md)
Updated prediction to forecast future years.
- **Input years**: 2010, 2015, 2020
- **Target years**: 2025, 2030, 2035, 2040 (future)
- **Output**: 12 GeoTIFF files per region (3 quantiles × 4 horizons)

### [PREDICTION_MASKING_FIX.md](PREDICTION_MASKING_FIX.md)
Fixes for prediction masking and edge handling.
- **Topics**: NaN handling during inference, edge artifact mitigation

### [BATCHED_PREDICTION.md](BATCHED_PREDICTION.md)
Efficient batched prediction for large areas.
- **Topics**: GPU batch processing, overlap blending, memory management
- **Default batch size**: 16 tiles processed in parallel

## Training & Optimization

### [TEMPORAL_SAMPLING_PLAN.md](TEMPORAL_SAMPLING_PLAN.md)
Temporal sampling strategy for training.
- **Topics**: Random temporal sampling, data augmentation, validation consistency
- **Training**: Random end years (2000, 2005, 2010, 2015) + rolling window
- **Validation**: Fixed years (1990, 1995, 2000 → 2005-2020) for consistency

### [WANDB_SWEEP_GUIDE.md](WANDB_SWEEP_GUIDE.md)
Hyperparameter sweeps with Weights & Biases.
- **Topics**: Sweep configuration, hyperparameter tuning, experiment tracking
- **Usage**: `wandb sweep config/sweep_config.yaml`

## Checkpoints & Deployment

### [CHECKPOINT_USAGE.md](CHECKPOINT_USAGE.md)
Loading and using trained model checkpoints.
- **Topics**: W&B artifacts, local checkpoints, prediction from checkpoint
- **Examples**: Load from W&B, load from file, inference-only mode

## Bug Fixes & Updates

### [PINBALL_LOSS_FIX.md](PINBALL_LOSS_FIX.md)
Fixed NaN propagation in pinball loss.
- **Problem**: NaN values in predictions caused loss to be NaN
- **Solution**: Filter valid pixels BEFORE computing loss
- **Status**: ✅ Fixed and tested

## Historical Documentation

### [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md)
Initial single-horizon ConvLSTM implementation.
- **Milestone**: Basic spatio-temporal forecasting with ConvLSTM
- **Status**: Superseded by multi-horizon implementation

### [PHASE2_COMPLETE.md](PHASE2_COMPLETE.md)
Multi-horizon forecasting implementation.
- **Milestone**: Extended to 4 horizons (5yr, 10yr, 15yr, 20yr)
- **Status**: Extended with quantile predictions

### [MULTI_HORIZON_CHANGES.md](MULTI_HORIZON_CHANGES.md)
Detailed changes for multi-horizon support.
- **Topics**: Model output expansion, loss computation, validation changes

## Quick Start Guides

### For New Users
1. Start with [`README.md`](../README.md) in root directory
2. Review [QUANTILE_PREDICTION_IMPLEMENTATION.md](QUANTILE_PREDICTION_IMPLEMENTATION.md) for architecture
3. Check [LOSS_WEIGHTING_EXPLAINED.md](LOSS_WEIGHTING_EXPLAINED.md) for training details
4. Use [WANDB_SWEEP_GUIDE.md](WANDB_SWEEP_GUIDE.md) for hyperparameter tuning

### For Training
1. Review [TEMPORAL_SAMPLING_PLAN.md](TEMPORAL_SAMPLING_PLAN.md) for data strategy
2. Check [NORMALIZATION_BUG_FIX.md](NORMALIZATION_BUG_FIX.md) for preprocessing
3. Use [WANDB_SWEEP_GUIDE.md](WANDB_SWEEP_GUIDE.md) for experiments

### For Prediction
1. Review [CHECKPOINT_USAGE.md](CHECKPOINT_USAGE.md) for loading models
2. Check [PREDICTION_YEARS_UPDATE.md](PREDICTION_YEARS_UPDATE.md) for future forecasting
3. Use [BATCHED_PREDICTION.md](BATCHED_PREDICTION.md) for large areas

### For Debugging
1. [PINBALL_LOSS_FIX.md](PINBALL_LOSS_FIX.md) - NaN issues
2. [MASKING_STRATEGY.md](MASKING_STRATEGY.md) - Data quality issues
3. [PREDICTION_MASKING_FIX.md](PREDICTION_MASKING_FIX.md) - Prediction artifacts

## Key Concepts

### Quantile Regression
The model predicts three values per horizon:
- **Lower (2.5%)**: Conservative lower bound
- **Central**: Best estimate (NOT median, optimized for accuracy + patterns)
- **Upper (97.5%)**: Conservative upper bound

Together, these form a **95% confidence interval**.

### Independent Heads
- **12 separate neural networks** (3 per horizon)
- **Central heads**: Optimized for MSE + SSIM + Laplacian + Histogram
- **Quantile heads**: Optimized ONLY for pinball loss
- **No gradient interference** between objectives

### Coverage Calibration
The `val_coverage_total` metric tracks what percentage of validation targets fall within [lower, upper] intervals. Target: **~95%**
- **< 85%**: Model too confident (intervals too narrow)
- **> 98%**: Model too uncertain (intervals too wide)

### Multi-Horizon Forecasting
Four forecast horizons:
- **5yr** (2025): Most confident
- **10yr** (2030): Moderate confidence
- **15yr** (2035): Lower confidence  
- **20yr** (2040): Least confident

Uncertainty naturally increases with forecast horizon.

## File Organization

```
docs/
├── README.md (this file)                    # Index and navigation
├── QUANTILE_PREDICTION_IMPLEMENTATION.md    # Core architecture
├── INDEPENDENT_HEADS_IMPLEMENTATION.md      # Design details
├── LOSS_WEIGHTING_EXPLAINED.md              # Loss functions
├── NORMALIZATION_BUG_FIX.md                 # Data preprocessing
├── MASKING_STRATEGY.md                      # NaN handling
├── PREDICTION_YEARS_UPDATE.md               # Future forecasting
├── PREDICTION_MASKING_FIX.md                # Prediction fixes
├── BATCHED_PREDICTION.md                    # Large-area prediction
├── TEMPORAL_SAMPLING_PLAN.md                # Training strategy
├── WANDB_SWEEP_GUIDE.md                     # Hyperparameter tuning
├── CHECKPOINT_USAGE.md                      # Model deployment
├── PINBALL_LOSS_FIX.md                      # Bug fix
├── PHASE1_COMPLETE.md                       # Historical
├── PHASE2_COMPLETE.md                       # Historical
└── MULTI_HORIZON_CHANGES.md                 # Historical
```

## External Resources

- **W&B Project**: https://wandb.ai/glennwithtwons/spatio-temporal-convlstm
- **HM Dataset Paper**: https://www.nature.com/articles/s41597-025-04892-2
- **Main Repository**: See root [`README.md`](../README.md)

## Contributing

When adding new documentation:
1. Create markdown file in `docs/` directory
2. Add entry to this index
3. Update root [`README.md`](../README.md) if necessary
4. Use clear section headers and code examples
5. Include status (✅ Complete, 🚧 In Progress, ⚠️ Deprecated)

---

**Last Updated**: 2025-10-24  
**Documentation Status**: ✅ Current and complete
