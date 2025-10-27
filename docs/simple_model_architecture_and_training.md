# Model Architecture and Training Pipeline (Simplified)

**Document Version:** 1.0  
**Last Updated:** 2025-10-27  
**Model:** SpatioTemporalPredictor with ConvLSTM

This is a condensed version of the full technical documentation. For detailed implementation examples and code, see `model_architecture_and_training.md`.

---

## Preamble and Overview

### The Problem

Predicting future landscape modification requires understanding both **spatial** relationships (how neighboring areas influence each other) and **temporal** dynamics (how past trends inform future trajectories). Traditional approaches often treat pixels independently or ignore time-series patterns, missing crucial context.

### Our Approach

We combine three key insights:

1. **Temporal Dependencies**: Historical sequences (4 timesteps at 5-year intervals) capture momentum and trends
2. **Spatial Context**: Convolutional operations preserve spatial relationships between neighboring pixels
3. **Uncertainty Quantification**: Prediction intervals (2.5th-97.5th percentile) alongside central predictions provide range of plausible futures

### Model Output

For each of **4 future horizons** (5, 10, 15, 20 years ahead), the model produces:
- **Central prediction**: Best estimate optimized for accuracy
- **Lower bound** (2.5th percentile): Optimistic scenario
- **Upper bound** (97.5th percentile): Pessimistic scenario

Total: 12 prediction channels per sample

### Architecture Summary

**Components:**
- **Input Processing**: 11 dynamic channels + 7 static channels + 8 location encoding channels
- **Temporal Encoder**: ConvLSTM (1 layer, 16 hidden channels)
- **Prediction Heads**: 12 independent convolutional networks (4 horizons × 3 quantiles)

**Training:**
- Multi-objective loss: MSE (50%) + SSIM (20%) + Laplacian (10%) + Histogram (20%)
- Quantile predictions use pinball loss
- Per-variable normalization handles heterogeneous scales
- Validation on geographically separate regions

**Data:**
- Spatial resolution: 1 km
- Temporal resolution: 5-year intervals (1990-2020)
- Coverage: Near-global extent
- Target: Human Modification (HM) index [0, 1]

---

## Input Data

### Three Types of Input

**1. Dynamic Variables (11 channels)**

Time-varying data capturing landscape evolution:

| Code | Variable | Description |
|------|----------|-------------|
| AA | Human Modification | All threats combined (target variable) |
| AG | Agricultural | Agricultural land extent |
| BU | Built-up | Residential, commercial areas |
| EX | Extraction | Energy production and mining |
| FR | Biological Resource Use | Forest harvest, logging |
| HI | Human Accessibility | Access and intrusion |
| NS | Natural Systems | Systems modification |
| PO | Pollution | Pollution sources |
| TI | Transportation | Transport corridors |
| gdp | GDP | Economic activity (millions) |
| population | Population | Total population count (thousands) |

Available years: 1990, 1995, 2000, 2005, 2010, 2015, 2020

**2. Static Variables (7 channels)**

Time-invariant geographic characteristics:

| Code | Variable | Scale |
|------|----------|-------|
| ele | Elevation | meters |
| tas | Temperature | °C |
| tasmin | Min Temperature | °C |
| pr | Precipitation | mm/year |
| dpi_dsi | Distance to Protected Areas | varies |
| iucn_nostrict | Protected Areas (III-VI) | [0, 1] |
| iucn_strict | Protected Areas (Ia-II) | [0, 1] |

**3. Geographic Location (8 channels)**

- Input: Latitude/longitude coordinates
- Encoded via spherical harmonics (10 terms) + SIREN network (2 layers, 64 units)
- Captures unmeasured location-specific factors (cultural, political, economic contexts)

### Data Organization

**Files:**
- Dynamic: `HM_{YEAR}_{VARIABLE}_1000.tiff` (e.g., `HM_2020_AA_1000.tiff`)
- Static: `hm_static_{VARIABLE}_1000.tiff` (e.g., `hm_static_ele_1000.tiff`)

**Input Sequences:**
- Model uses 4 consecutive timesteps to predict future horizons
- Example: [1990, 1995, 2000, 2005] → predict [2010, 2015, 2020, 2025]

---

## Data Transformations

### Why Normalize?

Variables have vastly different scales:
- HM components: [0, 1]
- GDP: millions
- Elevation: thousands of meters
- Temperature: -50 to +50°C

Without normalization, large-magnitude variables dominate gradients during training.

### Per-Variable Normalization

**Dynamic variables:** Each of 11 channels normalized independently
- Compute mean and standard deviation for each variable across all timesteps
- Apply: `normalized = (value - mean) / std`
- Result: All variables centered at 0 with unit variance

**Static variables:** Each of 7 channels normalized independently
- Same process as dynamic variables
- Separate statistics for each static layer

**Location encoding:** Learned network handles coordinate transformation

### NaN Handling Strategy

**Problem:** Missing data (oceans, no-data regions) can contaminate predictions through convolutions

**Workflow:**
1. **Preserve NaNs** during data loading
2. **Compute validity mask** for each sample
3. **Replace NaNs with 0.0** before model forward pass (required for PyTorch operations)
4. **Apply mask to loss** computation (exclude invalid pixels from gradients)
5. **Set predictions to NaN** for invalid regions in outputs

**Note:** Pixels adjacent to NaN regions may still show edge artifacts due to convolution operations spreading information from sanitized (NaN→0) regions.

### Data Augmentation

**Spatial sampling:** Random 256×256 crops from global extent during training
- Ensures diversity in training batches
- Prevents overfitting to specific geographic regions

**No other augmentation:** Rotations/flips not used to preserve geographic meaning

### Data Leakage Prevention

**Critical:** Only use data **up to the last input year** when preparing sequences
- Never peek at future timesteps
- Example: If predicting 2010-2025 from inputs [1990-2005], don't load 2010+ data for covariates

---

## Model Structure

### Architecture Flow

**Input → LocationEncoder → ConvLSTM → Prediction Heads → Output**

### 1. Location Encoder

**Purpose:** Encode geographic position as learnable features

**Components:**
- **Spherical Harmonics**: 10 Legendre polynomial terms capture large-scale spatial patterns
- **SIREN Network**: 2 hidden layers (64 units each) with sine activations learn fine-grained location features

**Input:** Lat/lon coordinates [B, H, W, 2]  
**Output:** Location features [B, 8, H, W]

**Why?** Captures unmeasured location-specific factors (e.g., cultural contexts, policies) without explicitly encoding every detail

### 2. Input Fusion

Concatenate three input types along channel dimension:
- Dynamic: [B, T=4, C=11, H, W]
- Static: [B, C=7, H, W] → expanded to [B, T=4, C=7, H, W] (repeated)
- Location: [B, C=8, H, W] → expanded to [B, T=4, C=8, H, W]

**Combined:** [B, T=4, C=26, H, W] (11 dynamic + 7 static + 8 location)

### 3. ConvLSTM Temporal Encoder

**Purpose:** Process temporal sequences while preserving spatial structure

**Architecture:**
- 1 ConvLSTM layer
- Hidden dimension: 16 channels
- Kernel size: 3×3
- Processes all 4 timesteps sequentially

**Output:** Final hidden state [B, 16, H, W] containing temporal-spatial encoding

**Why ConvLSTM?** Unlike standard LSTM (loses spatial structure) or 3D convolutions (limited temporal memory), ConvLSTM maintains both spatial relationships and temporal dependencies.

### 4. Prediction Heads (12 Independent Networks)

**Structure:**

Each head is a small convolutional network:
- **Central predictions** (4 heads): Conv(16→16) → ReLU → Conv(16→1)
- **Quantile predictions** (8 heads): Conv(16→8) → ReLU → Conv(8→1)

**Independence:** Each horizon-quantile pair has its own network with separate weights
- No parameter sharing between horizons
- Allows learning horizon-specific patterns

**Output Shape:** [B, 12, H, W]
- Channels 0-2: 5yr (lower, central, upper)
- Channels 3-5: 10yr (lower, central, upper)
- Channels 6-8: 15yr (lower, central, upper)
- Channels 9-11: 20yr (lower, central, upper)

### Design Rationale

**Why small model?**
- Global 1km data = millions of pixels → small model prevents overfitting
- Rich input features (26 channels) provide sufficient information
- Computational efficiency for inference on large regions

**Why independent heads?**
- Different horizons have different uncertainty characteristics
- Separate optimization prevents interference between objectives
- Explicit quantile modeling ensures calibrated prediction intervals

---

## Loss Functions

### Multi-Objective Optimization

**Challenge:** Single loss function (e.g., MSE) optimizes only pixel-wise accuracy, resulting in blurry, spatially incoherent predictions.

**Solution:** Combine 5 complementary loss functions, each addressing different aspects of prediction quality.

### 1. Mean Squared Error (MSE)

**Purpose:** Pixel-wise accuracy

**Targets:** Central predictions (4 horizons)

**Formula:** Average of squared differences between prediction and target

**Weight:** 1.0 (baseline)

**Contribution:** ~50% of total loss

### 2. Structural Similarity Index (SSIM)

**Purpose:** Preserve spatial patterns and perceptual similarity

**How it works:** Compares local patterns (luminance, contrast, structure) in sliding windows

**Targets:** Central predictions

**Weight:** 2.0

**Contribution:** ~20% of total loss

**Why?** Prevents over-smoothing; ensures predictions maintain realistic spatial structure

### 3. Laplacian Pyramid Loss

**Purpose:** Multi-scale detail preservation

**How it works:**
- Build image pyramids (4 levels: original, 1/2, 1/4, 1/8 resolution)
- Compute differences between pyramid levels (detail at each scale)
- Compare prediction vs. target detail at all scales

**Targets:** Central predictions

**Weight:** 1.0

**Contribution:** ~10% of total loss

**Why?** Ensures both fine details (local features) and coarse patterns (regional trends) are preserved

### 4. Histogram Loss

**Purpose:** Match distribution of predicted changes to real change distributions

**How it works:**
- Bin change values (Δ = prediction - last_input) into 8 bins
- Compute predicted distribution (softmax over bins)
- Compare to target distribution using cross-entropy + Wasserstein distance

**Targets:** Central predictions

**Weight:** 0.67

**Contribution:** ~20% of total loss

**Warmup:** Inactive for first 20 epochs (allows model to learn basic patterns before enforcing distributional constraints)

**Why?** Prevents unrealistic change patterns (e.g., all pixels changing by same amount)

### 5. Pinball Loss (Quantile Regression)

**Purpose:** Train quantile predictions to produce calibrated uncertainty intervals

**How it works:** Asymmetric penalty favoring under-prediction for lower quantile, over-prediction for upper quantile

**Targets:** Lower and upper bound predictions (8 total)

**Weight:** 1.0 per quantile

**Why?** Ensures prediction intervals have correct coverage (95% of observations should fall within bounds)

### Loss Calibration

**Target contributions:**
- MSE: 50%
- SSIM: 20%
- Laplacian: 10%
- Histogram: 20%

Weights are tuned based on typical loss magnitudes to achieve these proportions.

**Total loss formula:**
```
total_loss = 1.0×MSE + 2.0×SSIM + 1.0×Laplacian + 0.67×Histogram + Pinball_lower + Pinball_upper
```

### Gradient Assignment

- **Central predictions:** Receive gradients from MSE + SSIM + Laplacian + Histogram
- **Quantile predictions:** Receive gradients only from Pinball loss
- No gradient interference between prediction types

---

## Training

### Framework and Configuration

**Framework:** PyTorch Lightning
- Handles training loop, validation, logging
- Checkpoint management and recovery
- Multi-GPU support

**Optimizer:** Adam
- Learning rate: 1e-3
- Beta1: 0.9, Beta2: 0.999
- No weight decay (large dataset prevents overfitting)

**Key Hyperparameters:**
- Batch size: 8
- Patch size: 256×256 pixels
- Max epochs: 100
- Early stopping patience: 10 epochs

### Data Loading

**Training:**
- Random 256×256 spatial crops from global extent
- Shuffle: True
- ~524k pixels per batch (8 patches × 65k pixels/patch)

**Validation:**
- Fixed spatial tiles (reproducible evaluation)
- Geographically separate from training regions
- Shuffle: False

### Training Loop

**Each Epoch:**
1. **Training phase:** Process all training batches with gradient updates
2. **Validation phase:** Evaluate on validation set (no gradient updates)
3. **Checkpointing:** Save if validation loss improved
4. **Early stopping check:** Stop if no improvement for 10 epochs

**Histogram Warmup:**
- Epochs 0-19: Histogram loss = 0 (not active)
- Epoch 20+: Histogram loss activates
- Allows model to learn basic patterns before enforcing distributional constraints

### Model Selection

**Primary metric:** `val_total_loss`

Best checkpoint selected based on lowest combined validation loss, ensuring model generalizes to unseen regions while balancing all objectives.

### Experiment Tracking

**Weights & Biases Integration:**

Logged metrics (per epoch):
- Training: `train_loss`, `train_mae_total`, `train_ssim_loss_total`, `train_lap_loss_total`, `train_hist_loss_total`, `train_total_loss`
- Validation: Same metrics with `val_` prefix
- Per-horizon: MAE and coverage for each forecast horizon
- System: GPU utilization, memory usage, training speed

### Common Issues

| Issue | Solution |
|-------|----------|
| GPU OOM | Reduce batch_size or patch_size |
| Loss not decreasing | Check data loading, verify gradient flow |
| Validation loss increasing | Overfitting - stop earlier or add regularization |
| Blurry predictions | Increase SSIM weight |
| Poor coverage (≠95%) | Check pinball loss implementation |

### Computational Requirements

**Single training run:**
- Time: 12-48 hours (GPU dependent)
- GPU memory: 8-16 GB
- Recommended: NVIDIA RTX 3090, A100, or equivalent

---

## Accuracy Assessment

### Evaluation Philosophy

"Works" means different things for different aspects:
- **Point accuracy:** Individual pixels close to reality (MAE, MSE)
- **Spatial coherence:** Realistic patterns, not noise (SSIM)
- **Detail preservation:** Boundaries and features maintained (Laplacian)
- **Distribution realism:** Changes match real distributions (Histogram)
- **Uncertainty calibration:** 95% of observations within intervals (Coverage)
- **Generalization:** Performance on unseen geographic regions

### Key Metrics

**1. Mean Absolute Error (MAE)**
- Average pixel-wise error
- Units: HM scale [0, 1]
- Excellent: <0.05, Good: 0.05-0.10, Fair: 0.10-0.15, Poor: >0.15

**2. Structural Similarity (SSIM)**
- Pattern preservation
- Range: [0, 1] where 1 = perfect match
- Excellent: >0.9, Good: 0.8-0.9, Fair: 0.6-0.8, Poor: <0.6

**3. Coverage (Quantile Calibration)**
- Percentage of observations within [lower, upper] bounds
- Target: 95% (for 2.5th-97.5th percentile interval)
- Acceptable: 93-97%

**4. Other Metrics**
- Laplacian loss: Multi-scale detail (lower is better)
- Histogram loss: Distribution match (lower is better)
- Pinball loss: Quantile quality (lower is better)

### Per-Horizon Analysis

**Expected degradation:** Accuracy decreases with forecast horizon

| Horizon | Expected MAE | Expected SSIM |
|---------|--------------|---------------|
| 5yr     | 0.05-0.08    | 0.85-0.90     |
| 10yr    | 0.07-0.11    | 0.80-0.85     |
| 15yr    | 0.09-0.14    | 0.75-0.82     |
| 20yr    | 0.11-0.17    | 0.70-0.80     |

**Why degradation?** Longer extrapolation = more uncertainty, cumulative errors, unpredictable events

**Red flags:**
- Flat performance across horizons → Model not learning temporal dynamics
- Sudden drop at one horizon → Issue with specific target data
- Better long-term than short-term → Data leakage

### Validation Strategy

**Spatial split:**
- Training and validation regions geographically separate
- Tests if model learned general dynamics vs. memorizing locations

**Overfitting indicators:**
- Training MAE 10-20% better than validation: Acceptable
- Training MAE >30% better than validation: Problem

### Reporting

**Standard table:**

| Metric | 5yr | 10yr | 15yr | 20yr | Average |
|--------|-----|------|------|------|---------|
| MAE | 0.063 | 0.091 | 0.118 | 0.152 | 0.106 |
| SSIM | 0.874 | 0.832 | 0.791 | 0.743 | 0.810 |
| Coverage | 94.8% | 95.2% | 95.7% | 96.1% | 95.5% |

**Key takeaways:**
1. Overall performance level
2. Temporal degradation pattern
3. Uncertainty calibration
4. Generalization ability
5. Qualitative spatial patterns

---

## Prediction

### Workflow Overview

Once trained, the model generates predictions for any region:

1. **Define region:** GeoJSON polygon specifying area of interest
2. **Load data:** Historical sequences + static variables + coordinates
3. **Tile-based processing:** Divide into manageable chunks (GPU memory constraints)
4. **Generate predictions:** Forward pass through model
5. **Stitch tiles:** Combine into seamless regional map
6. **Save outputs:** GeoTIFF format with 12 bands

### Tile-Based Processing

**Why?** Large regions (1000×1000 km = 1M pixels) don't fit in GPU memory

**Strategy:**
- Divide region into tiles (e.g., 512×512 pixels)
- Process each tile independently
- Use overlap (64-128 pixels) to prevent edge artifacts
- Blend overlapping regions with distance-weighted averaging

**Processing time estimates:**

| Region Size | Tiles | GPU Time | CPU Time |
|-------------|-------|----------|----------|
| 100×100 km | ~100 | ~2 min | ~10 min |
| 500×500 km | ~1000 | ~20 min | ~2 hours |
| 1000×1000 km | ~4000 | ~80 min | ~8 hours |

### Output Format

**GeoTIFF Structure (12 bands):**
- Band 1: Lower bound, 5yr (2.5th percentile)
- Band 2: Central prediction, 5yr
- Band 3: Upper bound, 5yr (97.5th percentile)
- Band 4-6: 10yr (lower, central, upper)
- Band 7-9: 15yr (lower, central, upper)
- Band 10-12: 20yr (lower, central, upper)

**Properties:**
- Data type: Float32
- No data value: NaN
- Georeference: Matches input data (e.g., WGS84)
- Compression: LZW

### Post-Processing

**Change maps:** Compute difference from last input year to predictions

**Risk maps:** Binary maps of areas predicted to exceed threshold (e.g., HM > 0.7)

**Uncertainty maps:** Interval width (upper - lower) as measure of prediction uncertainty

### Visualization

**Map types:**
- **Prediction maps:** Show predicted HM values for each horizon
- **Error maps:** Prediction - actual (for validation)
- **Uncertainty maps:** Interval width visualization
- **Change distributions:** Histograms comparing predicted vs. actual changes

**Color schemes:**
- HM values: White (low) to red (high)
- Change: Blue (decrease) to red (increase)
- Uncertainty: Viridis colormap

### GIS Integration

**QGIS/ArcGIS:** Load GeoTIFF directly, each band as separate layer

**Python (rasterio/GeoPandas):** Read bands, overlay with vector boundaries, perform spatial analysis

**Use cases:**
- Conservation planning
- Policy evaluation
- Risk assessment
- Monitoring target setting

---

## Summary

This pipeline combines spatio-temporal modeling (ConvLSTM), multi-objective optimization (5 loss functions), and uncertainty quantification (quantile regression) to produce probabilistic forecasts of landscape change. Key innovations include per-variable normalization for heterogeneous data, independent prediction heads for multi-horizon forecasting, and careful handling of missing data to prevent contamination.

For full implementation details, code examples, and formulas, see the complete technical documentation: `model_architecture_and_training.md`
