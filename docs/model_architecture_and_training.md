# Model Architecture and Training Pipeline

**Document Version:** 1.0  
**Last Updated:** 2025-10-27  
**Model:** SpatioTemporalPredictor with ConvLSTM

---

## Preamble and Overview

### Intuitive Explanation

Predicting how human modification of landscapes will evolve over time is fundamentally a spatio-temporal problem. We need to understand not just *where* change happens, but *how* patterns of change unfold across both space and time. Traditional approaches often treat pixels independently or ignore temporal dynamics, but landscapes are interconnected systems where change in one location influences neighboring areas, and past trends inform future trajectories.

Our approach combines three key insights:

1. **Temporal Dependencies Matter**: How a landscape changes from 1990→2000→2010 contains crucial information for predicting 2020. We use historical sequences (4 timesteps) to learn temporal patterns and momentum.

2. **Spatial Context is Essential**: A pixel's future depends on its neighbors. We preserve spatial relationships through convolutional operations rather than treating each location independently.

3. **Uncertainty Quantification is Critical**: Predicting the future is inherently uncertain. Rather than providing a single "best guess," we estimate prediction intervals (2.5th and 97.5th percentiles) alongside central predictions, giving users a range of plausible futures.

The model predicts human modification (HM) values at **four future horizons** (5, 10, 15, and 20 years ahead) based on historical observations. For each horizon, it provides three predictions:
- A **central prediction** optimized for accuracy and spatial coherence
- A **lower bound** (2.5th percentile) representing optimistic scenarios
- An **upper bound** (97.5th percentile) representing pessimistic scenarios

This multi-horizon, multi-quantile approach allows users to understand both the expected trajectory and the range of uncertainty at different time scales.

### Technical Overview

**Model Architecture:** SpatioTemporalPredictor with ConvLSTM backbone

**Core Components:**
1. **Input Processing**
   - Dynamic variables: 11 time-varying channels (HM + 10 covariates)
   - Static variables: 7 time-invariant channels (topography, climate, protected areas)
   - Location encoding: 8 channels from spherical harmonics + SIREN network

2. **Temporal Processing**
   - ConvLSTM with 1 layer, hidden dimension 16
   - Processes sequences of 4 historical timesteps
   - Preserves spatial structure (H×W) throughout temporal encoding

3. **Prediction Heads**
   - 12 independent prediction heads (4 horizons × 3 quantiles)
   - Central heads: Conv2d(16→16) → ReLU → Conv2d(16→1)
   - Quantile heads: Conv2d(16→8) → ReLU → Conv2d(8→1)

4. **Training Objective**
   - Central predictions: Multi-objective optimization (MSE + SSIM + Laplacian + Histogram)
   - Quantile predictions: Pinball loss for uncertainty estimation
   - Loss weights calibrated for balanced contribution (MSE 50%, SSIM 20%, Laplacian 10%, Histogram 20%)

**Data Characteristics:**
- **Spatial Resolution:** 1 km (resampled from original data)
- **Temporal Resolution:** 5-year intervals (1990, 1995, 2000, 2005, 2010, 2015, 2020)
- **Coverage:** Near-global extent
- **Target Variable:** Human Modification (HM) index [0, 1] representing anthropogenic landscape alteration

**Training Strategy:**
- Per-variable normalization to handle heterogeneous scales (GDP in millions, elevation in meters, HM in [0,1])
- Histogram loss warmup (inactive for first 20 epochs) to allow model to learn basic patterns first
- NaN masking to handle missing data and prevent contamination from no-data regions
- Validation on separate spatial regions to assess generalization

**Output Format:**
- Shape: [Batch, 12, Height, Width]
- Channel ordering: [lower₅yr, central₅yr, upper₅yr, lower₁₀yr, central₁₀yr, upper₁₀yr, ...]
- Values: HM predictions in [0, 1] range

---

## Input Data

### Intuitive Explanation

The model learns from three types of information, each serving a distinct purpose:

1. **Dynamic (Time-Varying) Variables**: These are the "movie" of landscape change—variables that evolve over time and capture how human activities, economic development, and land use have changed historically. The primary target is the Human Modification (HM) index itself, but we also include complementary variables like agricultural extent, built-up areas, GDP, and population. By observing how these variables co-evolved in the past, the model learns patterns like "when GDP increases, urban areas tend to expand" or "agricultural intensification often precedes built-up growth."

2. **Static (Time-Invariant) Variables**: These are the "stage" on which change happens—fundamental geographic characteristics that don't change over our timescale but strongly influence *where* and *how* change occurs. Mountains constrain urban expansion differently than plains; protected areas slow development; climate determines agricultural suitability. These variables provide context that helps the model understand spatial constraints and opportunities.

3. **Geographic Location**: Two places with identical characteristics may still differ because of their geographic context—a forest in the Amazon faces different pressures than one in Siberia. We encode latitude/longitude using spherical harmonics and a neural network, allowing the model to learn location-specific patterns without explicitly encoding every geographic detail.

Together, these inputs give the model a rich understanding of each location: its history (dynamic), its fundamental characteristics (static), and its geographic context (location). The model learns which combinations of these factors predict future change.

### Technical Details

#### Data Structure and Organization

**File Organization:**
```
data/raw/hm_global/
├── HM_1990_AA_1000.tiff         # Target variable (HM) for 1990
├── HM_1990_AG_1000.tiff         # Agricultural extent for 1990
├── HM_1990_BU_1000.tiff         # Built-up areas for 1990
├── ...                          # Other dynamic variables
├── HM_2020_AA_1000.tiff         # Most recent year
├── hm_static_ele_1000.tiff      # Elevation (static)
├── hm_static_tas_1000.tiff      # Temperature (static)
└── ...                          # Other static variables
```

**Naming Convention:**
- Dynamic: `HM_{YEAR}_{VARIABLE}_1000.tiff`
- Static: `hm_static_{VARIABLE}_1000.tiff`

#### Dynamic Variables (11 Channels)

Time-varying data available at 5-year intervals: **1990, 1995, 2000, 2005, 2010, 2015, 2020**

| Variable | Code | Description | Original Scale |
|----------|------|-------------|----------------|
| **Human Modification** | AA | All threats combined (composite index of anthropogenic modification) | [0, 1] |
| Agricultural | AG | Agricultural land extent and intensity | [0, 1] |
| Built-up | BU | Residential, commercial and recreation areas | [0, 1] |
| Extraction | EX | Energy production and mining | [0, 1] |
| Biological Resource Use | FR | Forest harvest, logging, and other biological resource use | [0, 1] |
| Human Accessibility | HI | Human accessibility and intrusion (HA and HI are interchangeable) | [0, 1] |
| Natural Systems Modification | NS | Natural systems modification | [0, 1] |
| Pollution | PO | Pollution sources and impacts | [0, 1] |
| Transportation | TI | Transportation and service corridors | [0, 1] |
| **GDP** | gdp | Gross Domestic Product per grid cell | ~[0, millions] |
| **Population Count** | population | Total population per grid cell | ~[0, thousands] |

**Key Notes on Dynamic Variables:**
- All HM component variables (AA, AG, BU, etc.) are pre-scaled to [0, 1] in the source data
- GDP and population have vastly different scales and are normalized separately (see Data Transformations)
- Model uses **4 consecutive timesteps** as input to predict future horizons
- For example: [1990, 1995, 2000, 2005] → predict [2010, 2015, 2020, 2025]

#### Static Variables (7 Channels)

Time-invariant geographic and climatic variables:

| Variable | Code | Description | Original Scale |
|----------|------|-------------|----------------|
| **Elevation** | ele | Height above sea level | meters |
| **Temperature** | tas | Mean annual temperature | °C |
| **Minimum Temperature** | tasmin | Mean minimum temperature | °C |
| **Precipitation** | pr | Mean annual precipitation | mm/year |
| **Distance to Protected** | dpi_dsi | Distance to protected areas | varies |
| **IUCN (Not Strict)** | iucn_nostrict | Protected areas (categories III-VI) | [0, 1] |
| **IUCN (Strict)** | iucn_strict | Protected areas (categories Ia-II) | [0, 1] |

**Excluded Static Variables:**
- `ele_asp_cosin`, `ele_asp_sin`: Aspect (circular) - removed to reduce redundancy
- `ele_slope`: Slope - can be derived from elevation, removed for simplicity
- `ele_tpi`: Topographic Position Index - removed to reduce dimensionality

#### Geographic Location (8 Channels)

**Input:** Latitude and Longitude coordinates for each pixel `[B, H, W, 2]`

**Processing:**
1. **Spherical Harmonics**: Encode geographic position using Legendre polynomials (10 terms)
   - Respects Earth's spherical geometry
   - Captures large-scale spatial patterns (continents, climate zones)

2. **SIREN Network**: Sinusoidal representation network learns location-specific features
   - 2 hidden layers with 64 units each
   - Produces 8-dimensional location encoding per pixel

**Output:** `[B, 8, H, W]` learned geographic features

**Purpose:** 
- Captures unmeasured location-specific factors (cultural, political, economic contexts)
- Allows model to learn region-specific dynamics (e.g., tropical vs. temperate change patterns)
- Provides smooth spatial interpolation of learned patterns

#### Data Specifications

**Spatial Characteristics:**
- **Resolution:** 1 km (1000m as indicated by `_1000` suffix)
- **Coordinate System:** (To be specified - likely WGS84 or similar)
- **Extent:** Near-global coverage (not just sub-Saharan Africa)
- **Format:** GeoTIFF raster files

**Temporal Characteristics:**
- **Time Points:** 7 timesteps (1990, 1995, 2000, 2005, 2010, 2015, 2020)
- **Interval:** 5 years
- **Input Sequence Length:** 4 timesteps
- **Prediction Horizons:** +5yr, +10yr, +15yr, +20yr from last input timestep

**Missing Data Handling:**
- NaN values present in original data (ocean, no-data regions)
- Preserved as NaN through preprocessing
- Masked during training (see Data Transformations)

**Data Loading:**
- Implementation: `scripts/torchgeo_dataloader.py`
- On-the-fly loading and preprocessing
- Spatial sampling for training batches
- Random crop augmentation during training

---

## Data Transformations

### Intuitive Explanation

Raw data comes in wildly different scales: elevation is measured in thousands of meters, temperature in tens of degrees Celsius, GDP in millions of dollars, while HM values are already in the [0,1] range. If we fed these directly to a neural network, the model would be overwhelmed by the large-magnitude variables (GDP, elevation) and largely ignore the smaller ones (HM, temperature).

This is like trying to hear a whisper next to a jackhammer—the quiet signal gets drowned out. Normalization solves this by transforming all variables to comparable scales, typically with mean≈0 and standard deviation≈1. This gives every variable an equal "voice" during training.

However, we face a critical challenge: **missing data (NaNs)**. Oceans have no elevation, some regions lack GDP data, and data coverage varies geographically. We need a strategy that:
1. Preserves NaN information (don't fill with fake values before knowing where valid data is)
2. Prevents NaN contamination during convolutions (neighboring valid pixels shouldn't be corrupted)
3. Normalizes each variable independently (GDP and elevation shouldn't share statistics)

Our approach: **per-variable normalization computed only on valid (non-NaN) pixels**, followed by careful masking during model training. Each variable is standardized using its own mean and standard deviation, computed from the non-NaN pixels across the entire dataset. During training, we track which pixels have valid inputs and only compute losses on those locations.

### Technical Details

#### Normalization Strategy

**Per-Variable Z-Score Normalization:**

For each variable independently:
```
normalized_value = (raw_value - mean_of_variable) / std_of_variable
```

**Why per-variable?**
- Different variables have fundamentally different distributions
- Example scales in raw data:
  - HM components: [0, 1]
  - Elevation: mean≈-4200m, std≈11,000m
  - Temperature: mean≈13°C, std≈13°C
  - GDP: mean≈7.6M, std≈55M
  - Population: mean≈hundreds, std varies widely

**Implementation Location:** `scripts/torchgeo_dataloader.py`

#### Dynamic Variable Normalization

**Process:**
1. **Initialization Phase** (dataset `__init__`):
   ```python
   for var_name in HM_VARS:
       # Load ALL years for this variable
       # Compute mean/std across valid (non-NaN) pixels only
       self.comp_means[var_name] = valid_pixels.mean()
       self.comp_stds[var_name] = valid_pixels.std()
   ```

2. **Loading Phase** (per sample):
   ```python
   for var_name in HM_VARS:
       carr = load_raster(file_path)
       # Apply per-variable normalization
       carr = (carr - self.comp_means[var_name]) / self.comp_stds[var_name]
   ```

**Variables Normalized:**
- AA (HM target)
- AG, BU, EX, FR, HI, NS, PO, TI (HM components)
- gdp, population (economic/demographic)

**Result:** All 11 dynamic channels transformed to approximately N(0,1) distribution

#### Static Variable Normalization

**Process:**
1. **Initialization Phase**:
   ```python
   for static_idx, static_file in enumerate(static_files):
       sarr = load_raster(static_file)
       # Compute mean/std for THIS variable only
       self.static_means[static_idx] = valid_pixels.mean()
       self.static_stds[static_idx] = valid_pixels.std()
   ```

2. **Loading Phase**:
   ```python
   for static_idx, static_file in enumerate(static_files):
       sarr = load_raster(static_file)
       sarr = (sarr - self.static_means[static_idx]) / self.static_stds[static_idx]
   ```

**Variables Normalized:**
- ele, tas, tasmin, pr, dpi_dsi, iucn_nostrict, iucn_strict

**Result:** All 7 static channels transformed to approximately N(0,1) distribution

#### NaN Handling Strategy

**Critical Principle:** Preserve NaN through normalization, sanitize only at model input

**Workflow:**

1. **Data Loading** (torchgeo_dataloader.py):
   - Load raw GeoTIFF data (includes NaNs)
   - Apply per-variable normalization
   - **NaNs remain as NaN** (operations on NaN produce NaN)
   - Output: `[B, T, C, H, W]` with NaNs preserved

2. **Before Model Forward** (lightning_module.py, training_step/validation_step):
   ```python
   # Compute validity mask from RAW inputs (before sanitization)
   dynamic_valid = torch.isfinite(input_dynamic).all(dim=(1,2), keepdim=True)
   static_valid = torch.isfinite(input_static).all(dim=1, keepdim=True)
   
   # Replace NaN with 0 for model forward (mean imputation in normalized space)
   input_dynamic = torch.nan_to_num(input_dynamic, nan=0.0)
   input_static = torch.nan_to_num(input_static, nan=0.0)
   
   # Forward pass with sanitized inputs
   predictions = model(input_dynamic, input_static, lonlat)
   
   # Set predictions to NaN where inputs were invalid
   predictions[~mask] = float('nan')
   ```

3. **Loss Computation**:
   ```python
   # Extract only valid pixels for loss computation
   valid_predictions = predictions[mask]
   valid_targets = targets[mask]
   
   # Compute loss only on valid pixels
   loss = loss_fn(valid_predictions, valid_targets)
   ```

**Why sanitize to 0.0?**
- In normalized space (mean=0, std=1), zero represents the mean value
- This is equivalent to mean imputation
- Prevents NaN propagation through convolutions
- But we still track and exclude these pixels from loss computation

**Known Issue:** Edge contamination
- Convolutional operations can spread information from sanitized (fake 0) pixels to neighboring valid pixels
- Causes artifacts near coastlines or no-data boundaries
- Mitigation: Masking ensures we don't compute loss on these contaminated regions
- Future improvement: Implement partial convolutions or attention mechanisms that explicitly handle missing data

#### Location Encoding Preprocessing

**Longitude/Latitude Extraction:**
```python
# For each pixel in the spatial tile
for i in range(height):
    for j in range(width):
        lon, lat = geotransform_to_lonlat(i, j, geotransform)
        lonlat[i, j] = [lon, lat]
```

**No normalization needed:**
- LocationEncoder (spherical harmonics + SIREN) expects raw geographic coordinates
- Network learns appropriate scaling through its sinusoidal activations
- Output is learned features in arbitrary units

#### Spatial Sampling and Augmentation

**Training Patches:**
- Random spatial crops of size `[patch_size, patch_size]` (e.g., 256×256 pixels)
- Ensures diversity in training samples
- Allows training on large spatial extents without loading entire scenes

**Augmentation:**
- Random horizontal flips (50% probability)
- Random vertical flips (50% probability)
- No rotation (preserves geographic orientation)
- No color jittering (values have physical meaning)

**Validation/Testing:**
- Fixed spatial tiles (no randomness)
- No augmentation
- Ensures reproducible evaluation

#### Data Prevention Strategies

**No Data Leakage:**
- For predicting year Y, only use data from years < Y
- Input sequence: [Y-15, Y-10, Y-5, Y] → Predict: [Y+5, Y+10, Y+15, Y+20]
- Example: [1990, 1995, 2000, 2005] → [2010, 2015, 2020, 2025]
- **Critical:** Never include target years in covariates

**Validation Split:**
- Spatial split (not random pixel split)
- Train regions and validation regions are geographically separate
- Tests model's ability to generalize to unseen locations
- Prevents spatial autocorrelation from inflating performance metrics

**Implementation Details:**

```python
# Pseudocode from torchgeo_dataloader.py
class HMDataset:
    def __init__(self):
        # Compute normalization stats once
        self.comp_means, self.comp_stds = compute_dynamic_stats()
        self.static_means, self.static_stds = compute_static_stats()
    
    def __getitem__(self, idx):
        # Load temporal sequence
        input_sequence = []
        for year in input_years:
            dynamic_frame = load_and_normalize_dynamic(year)
            input_sequence.append(dynamic_frame)
        
        # Load static (same for all timesteps)
        static_data = load_and_normalize_static()
        
        # Load targets (multiple horizons)
        targets = {}
        for horizon in [5, 10, 15, 20]:
            target_year = input_years[-1] + horizon
            targets[f'{horizon}yr'] = load_target(target_year)
        
        # Extract location
        lonlat = extract_lonlat_from_geotransform()
        
        return {
            'input_dynamic': torch.stack(input_sequence),  # [T, C, H, W]
            'input_static': static_data,                   # [C, H, W]
            'lonlat': lonlat,                             # [H, W, 2]
            **targets                                      # {5yr, 10yr, 15yr, 20yr}
        }
```

---

## Model Structure

### Intuitive Explanation

The model architecture reflects our understanding of how landscape change works: it's a process that unfolds through time, across space, with location-specific characteristics, and inherent uncertainty.

**The Three-Stage Architecture:**

1. **Input Fusion Stage**: We start by combining all our information sources—the temporal history (dynamic variables), the geographic constraints (static variables), and the location context (learned from coordinates). Think of this as assembling all the pieces of the puzzle before trying to see the picture.

2. **Temporal Encoding Stage (ConvLSTM)**: This is where the model learns temporal patterns. The ConvLSTM processes the 4-timestep sequence, learning things like "urban areas tend to grow outward" or "deforestation accelerates after road construction." Crucially, it's *convolutional*, meaning it preserves spatial structure—it understands that neighboring pixels influence each other. It's also *recurrent* (LSTM), meaning it maintains a "memory" of past states as it processes the sequence, allowing it to capture momentum and trends.

3. **Multi-Horizon, Multi-Quantile Prediction Stage**: From the learned temporal-spatial representation, we generate 12 separate predictions through independent neural network "heads":
   - **4 horizons** (5, 10, 15, 20 years): Near-term predictions can be more confident; long-term predictions face more uncertainty
   - **3 quantiles per horizon**: Rather than just predicting "HM will be 0.5", we predict a range: "it will likely be between 0.3 (lower bound) and 0.7 (upper bound), with a best estimate of 0.5 (central)"

**Why Independent Heads?**

The central prediction is optimized for spatial coherence and accuracy (using losses that care about patterns and textures). The quantile predictions are optimized purely for capturing uncertainty bounds (using pinball loss). These are fundamentally different objectives, so we give them separate neural networks. This prevents the model from being confused about whether it should prioritize making accurate central predictions or reliable uncertainty bounds.

**Why ConvLSTM Instead of Simpler Alternatives?**

- **vs. Regular LSTM**: Spatial patterns matter. Regular LSTMs flatten images to vectors, destroying spatial relationships. ConvLSTM preserves the 2D structure.
- **vs. CNN alone**: Temporal dynamics matter. CNNs can see multiple timesteps as channels, but can't learn temporal sequences like "acceleration" or "deceleration" of change.
- **vs. 3D CNN**: ConvLSTM explicitly models temporal dependencies through recurrence, not just local temporal patterns. It has a "memory" of the full sequence.
- **vs. Transformer**: For 1km global data, spatial attention would be computationally prohibitive. ConvLSTM's local convolutional operations are efficient and physically meaningful (nearby pixels interact).

### Technical Details

#### Overall Architecture

**Model Class:** `SpatioTemporalPredictor` (defined in `src/models/spatiotemporal_predictor.py`)

**Data Flow:**
```
Input Dynamic [B, T=4, C=11, H, W]  ─┐
Input Static  [B, C=7, H, W]        ─┤
Lon/Lat       [B, H, W, 2]          ─┘
                                      │
                                      ├─→ LocationEncoder
                                      │   [B, H, W, 2] → [B, 8, H, W]
                                      │
                                      ├─→ Concatenate All Inputs
                                      │   [B, T=4, C=26, H, W]
                                      │   (11 dynamic + 7 static + 8 location)
                                      │
                                      ├─→ ConvLSTM
                                      │   [B, T=4, 26, H, W] → [B, 16, H, W]
                                      │   (last hidden state)
                                      │
                                      ├─→ 12 Independent Prediction Heads
                                      │   
                                      ├──→ Central Heads (×4)
                                      │    Conv2d(16→16) → ReLU → Conv2d(16→1)
                                      │    Output: [B, 1, H, W] per horizon
                                      │
                                      └──→ Quantile Heads (×8)
                                           Conv2d(16→8) → ReLU → Conv2d(8→1)
                                           Output: [B, 1, H, W] per quantile
                                      
Final Output: [B, 12, H, W]
Channel ordering: [lower₅, central₅, upper₅, lower₁₀, central₁₀, upper₁₀, ...]
```

#### Component 1: LocationEncoder

**Purpose:** Learn location-specific features from raw latitude/longitude

**Architecture:**
```python
LocationEncoder(
    backbone="sphericalharmonics",  # Position encoding
    model="siren",                   # Learned transformation
    hparams={
        'legendre_polys': 10,       # Spherical harmonic order
        'dim_hidden': 64,            # Hidden layer size
        'num_layers': 2,             # Network depth
        'num_classes': 8             # Output channels
    }
)
```

**Processing Steps:**

1. **Spherical Harmonics Encoding**:
   - Maps (lat, lon) to Legendre polynomial basis
   - 10 polynomial terms capture different spatial scales
   - Respects spherical geometry of Earth
   - Output: High-dimensional representation

2. **SIREN Network** (Sinusoidal Representation Network):
   - 2 hidden layers with 64 units each
   - Sinusoidal activations: `sin(w·x + b)`
   - Learns smooth, continuous spatial functions
   - Maps spherical harmonics to 8 learned features

**Input:** `[B, H, W, 2]` - longitude and latitude per pixel

**Output:** `[B, 8, H, W]` - learned location features

**Why This Matters:**
- Captures unmeasured geographic context (culture, policy, economic systems)
- Allows model to learn "African savannas behave differently than Asian steppes"
- Provides inductive bias: nearby locations should have similar features (smooth interpolation)

**Implementation:** Uses external library from `src.locationencoder`

#### Component 2: Input Concatenation

**Operation:**
```python
# Repeat static channels for each timestep
static_rep = input_static.unsqueeze(1).repeat(1, T, 1, 1, 1)  # [B, T, 7, H, W]

# Concatenate: dynamic + static + location
x = torch.cat([input_dynamic, static_rep, loc_feats_rep], dim=2)  # [B, T, 26, H, W]
```

**Result:** Unified tensor with 26 channels per timestep:
- 11 dynamic (time-varying)
- 7 static (time-invariant, but repeated for each timestep)
- 8 location (learned features)

#### Component 3: ConvLSTM

**Architecture:**
```python
ConvLSTM(
    input_dim=26,           # Combined input channels
    hidden_dim=16,          # Hidden state channels
    kernel_size=(3, 3),     # Spatial receptive field
    num_layers=1,           # Recurrent layers
    batch_first=True,       # [B, T, C, H, W] format
    bias=True,
    return_all_layers=False # Only return final layer
)
```

**ConvLSTM Cell Mathematics:**

At each timestep t, the cell computes:

```
Combined_t = Concat(X_t, H_{t-1})           # [B, 26+16, H, W]

# All gates use same convolution but different output slices
Gates = Conv2D(Combined_t)                   # [B, 4*16, H, W]

# Split into 4 gates
i_t = σ(Gates[:, 0:16, :, :])               # Input gate
f_t = σ(Gates[:, 16:32, :, :])              # Forget gate  
o_t = σ(Gates[:, 32:48, :, :])              # Output gate
g_t = tanh(Gates[:, 48:64, :, :])           # Cell gate

# Update cell state
C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t            # Element-wise operations

# Compute hidden state
H_t = o_t ⊙ tanh(C_t)                       # [B, 16, H, W]
```

**Key Properties:**

- **σ (sigmoid)**: Gates values ∈ [0,1], controlling information flow
- **tanh**: Bounded activation ∈ [-1,1] for cell state
- **⊙**: Element-wise (Hadamard) product
- **Conv2D**: Preserves spatial structure, enables spatial context
- **Recurrence**: H_t and C_t carry information forward through time

**Processing:**
1. **Timestep 1** (e.g., 1990): Initialize H₀=0, C₀=0, process first input
2. **Timestep 2** (e.g., 1995): Use H₁, C₁ from previous step
3. **Timestep 3** (e.g., 2000): Use H₂, C₂ from previous step
4. **Timestep 4** (e.g., 2005): Use H₃, C₃ from previous step

**Output:**
- Full sequence: `[B, T=4, 16, H, W]` hidden states for all timesteps
- **We use only the last timestep**: `[B, 16, H, W]` (H₄)
- This final hidden state encodes the full temporal sequence

**Why This Works:**
- Early timesteps influence later ones through recurrent connections
- The final H₄ contains "summary" of temporal evolution
- LSTM gates allow learning what to remember/forget from each timestep

**Implementation:** `src/models/convlstm.py`

#### Component 4: Prediction Heads

**Design Philosophy:** Independent heads for independent objectives

**4 Central Prediction Heads** (one per horizon: 5yr, 10yr, 15yr, 20yr)

```python
central_heads[i] = nn.Sequential(
    nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True),
    nn.ReLU(inplace=True),
    nn.Conv2d(16, 1, kernel_size=1, bias=True)
)
```

**Architecture:**
- First conv: Spatial integration (3×3 kernel), maintains 16 channels
- ReLU: Non-linearity
- Second conv: Channel reduction (1×1 kernel), produces single output channel
- **No final activation**: Output is unbounded (can predict any HM value)

**Input:** `[B, 16, H, W]` from ConvLSTM
**Output:** `[B, 1, H, W]` per horizon

**Optimization:** Trained with MSE + SSIM + Laplacian + Histogram losses

**4 Lower Quantile Heads** (2.5th percentile, one per horizon)

```python
lower_heads[i] = nn.Sequential(
    nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=True),
    nn.ReLU(inplace=True),
    nn.Conv2d(8, 1, kernel_size=1, bias=True)
)
```

**Architecture:**
- First conv: Reduces to 8 channels (lighter than central head)
- ReLU: Non-linearity
- Second conv: Single output channel

**Optimization:** Trained with Pinball loss (quantile=0.025)

**4 Upper Quantile Heads** (97.5th percentile, one per horizon)

```python
upper_heads[i] = nn.Sequential(
    nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=True),
    nn.ReLU(inplace=True),
    nn.Conv2d(8, 1, kernel_size=1, bias=True)
)
```

**Architecture:** Identical to lower heads

**Optimization:** Trained with Pinball loss (quantile=0.975)

**Why Quantile Heads Are Smaller:**
- Uncertainty estimation is conceptually simpler than full prediction
- Reduces parameter count (efficiency)
- Prevents overfitting to noise in tails of distribution

**Forward Pass Example:**

```python
# From ConvLSTM
last_hidden = convlstm_output[:, -1]  # [B, 16, H, W]

# Generate all predictions
predictions = []
for horizon_idx in range(4):
    pred_lower = lower_heads[horizon_idx](last_hidden)    # [B, 1, H, W]
    pred_central = central_heads[horizon_idx](last_hidden) # [B, 1, H, W]
    pred_upper = upper_heads[horizon_idx](last_hidden)    # [B, 1, H, W]
    
    predictions.extend([pred_lower, pred_central, pred_upper])

# Concatenate: [B, 12, H, W]
output = torch.cat(predictions, dim=1)
```

**Channel Ordering in Output:**
- Channels 0, 1, 2: Lower, Central, Upper for 5-year horizon
- Channels 3, 4, 5: Lower, Central, Upper for 10-year horizon
- Channels 6, 7, 8: Lower, Central, Upper for 15-year horizon
- Channels 9, 10, 11: Lower, Central, Upper for 20-year horizon

#### Model Parameters

**Total Parameters:** ~XXX K (to be computed)

**Breakdown by Component:**
- LocationEncoder: ~XX K parameters
- ConvLSTM: 
  - Gates convolution: (26+16) × 16×4 × 3×3 = ~24K parameters
- Central heads: 4 × [(16×16×3×3) + (16×1×1×1)] ≈ 9K parameters each
- Quantile heads: 8 × [(16×8×3×3) + (8×1×1×1)] ≈ 4.5K parameters each

**Memory Footprint:**
- Single forward pass (256×256 patch): ~XXX MB
- Batch size 8: ~XXX MB

**Computational Efficiency:**
- ConvLSTM is the bottleneck (recurrent processing)
- Prediction heads are lightweight (single forward pass)
- Location encoder computed once per spatial tile

#### Design Decisions and Rationale

**Why hidden_dim=16?**
- Balance between model capacity and efficiency
- Sufficient for capturing temporal patterns in HM data
- Prevents overfitting given available training data
- Allows larger batch sizes for stable training

**Why num_layers=1?**
- Temporal sequences are short (4 timesteps)
- Deeper networks didn't improve validation performance
- Single layer is more interpretable
- Reduces computational cost

**Why kernel_size=3?**
- 3×3 receptive field captures immediate neighbors
- Sufficient for local spatial context
- Larger kernels (5×5, 7×7) didn't improve results
- Computational efficiency

**Why independent heads instead of shared decoder?**
- Different objectives (accuracy vs. uncertainty)
- Prevents gradient conflicts
- Allows specialization
- Improves both central and quantile predictions

**Why ConvLSTM instead of attention?**
- Spatial convolution is inductive bias for geographic data
- More parameter-efficient than full attention
- Locality assumption is valid for landscape processes
- Computational feasibility for 1km global data

**Implementation Files:**
- Model: `src/models/spatiotemporal_predictor.py`
- ConvLSTM: `src/models/convlstm.py`
- LocationEncoder: `src/locationencoder/`
- Lightning wrapper: `src/models/lightning_module.py`

---

## Loss Functions

### Intuitive Explanation

Training a model is about defining what "good" means. For landscape prediction, "good" is multifaceted—we don't just want pixel-wise accuracy; we want predictions that look realistic, preserve spatial patterns, capture the right distribution of changes, and provide reliable uncertainty estimates.

Think of it like evaluating a painting: you wouldn't just count how many pixels match the reference photo. You'd also ask: Does it capture the overall composition? Are textures realistic? Are colors distributed naturally? Our loss functions work the same way, each evaluating a different aspect of prediction quality.

**The Four Perspectives on Central Predictions:**

1. **MSE (Mean Squared Error)**: "Are you close to the right answer?" This is the baseline—how far off are individual pixels? Contributes **50%** of total loss.

2. **SSIM (Structural Similarity)**: "Does it look right structurally?" Human perception is sensitive to patterns and structures. A prediction that's off by 0.1 everywhere but preserves spatial patterns is often more useful than one that's randomly accurate on some pixels. Contributes **20%**.

3. **Laplacian Pyramid Loss**: "Are fine details preserved?" Landscapes have structure at multiple scales: broad regions (forests, cities) and fine details (roads, small clearings). This loss ensures we don't just get the broad strokes right but also maintain sharp boundaries and small features. Contributes **10%**.

4. **Histogram Loss**: "Is the distribution of changes realistic?" Real landscape change follows patterns: most pixels change a little, some change moderately, few change dramatically. This loss ensures our predictions match this distribution, preventing unrealistic scenarios like "every pixel changes by exactly 0.2." Contributes **20%**.

**For Uncertainty (Quantile Predictions):**

5. **Pinball Loss**: "Are your prediction intervals calibrated?" For the 2.5th percentile, exactly 2.5% of observations should fall below it. Pinball loss specifically optimizes for this, ensuring our uncertainty estimates are neither too conservative (intervals too wide) nor too optimistic (intervals too narrow).

**Why Multiple Losses?**

A single loss would create blind spots. MSE alone produces blurry predictions (averaging is safe). SSIM alone might sacrifice pixel accuracy for nice patterns. By combining losses with carefully tuned weights, we get predictions that are accurate, structured, detailed, and distributional realistic.

**The Warmup Strategy:**

The histogram loss is computationally complex and requires understanding overall change distributions. If we enable it from the start, the model gets confused trying to simultaneously learn basic patterns and match distributions. We use a **20-epoch warmup**: for the first 20 epochs, histogram loss is turned off, letting the model learn fundamental relationships. After epoch 20, it activates, refining predictions to match realistic change distributions.

### Technical Details

#### Loss Components for Central Predictions

**Total Loss Formula:**

```
# Epochs 0-19 (warmup)
total_loss = 1.0 × MSE + 2.0 × SSIM + 1.0 × Laplacian

# Epochs 20+ (after warmup)
total_loss = 1.0 × MSE + 2.0 × SSIM + 1.0 × Laplacian + 0.67 × Histogram
```

**Target Loss Contributions (after warmup):**
- MSE: 50% (weight 1.0 × typical value ~0.1 = 0.1)
- SSIM: 20% (weight 2.0 × typical value ~0.1 = 0.2)
- Laplacian: 10% (weight 1.0 × typical value ~0.05 = 0.05)
- Histogram: 20% (weight 0.67 × typical value ~0.15 = 0.1)

Weights are calibrated based on empirical loss magnitudes during training.

#### 1. Mean Squared Error (MSE)

**Purpose:** Pixel-wise accuracy

**Formula:**
```
MSE = (1/N) Σ (prediction - target)²
```

Where N = number of valid (non-NaN) pixels

**Computation:**
```python
# Compute on deltas (change from last input)
delta_pred = prediction - last_input
delta_true = target - last_input

# Mask to valid pixels only
valid_pred = delta_pred[mask]
valid_true = delta_true[mask]

# MSE on valid pixels
mse = torch.mean((valid_pred - valid_true) ** 2)
```

**Properties:**
- **Sensitive to outliers**: Large errors penalized quadratically
- **Scale-dependent**: Works in [0,1] normalized space
- **Differentiable**: Smooth gradients for optimization
- **Unbiased**: No systematic over/under-prediction

**Why on deltas?**
- Change (delta) is what we're modeling, not absolute values
- Removes baseline bias (a pixel at 0.3 staying at 0.3 vs. changing to 0.4)
- More interpretable: predicting change magnitude

**Weight:** 1.0 (baseline)

**Implementation:** PyTorch `nn.MSELoss(reduction='mean')`

#### 2. Structural Similarity Index (SSIM)

**Purpose:** Perceptual quality and spatial structure preservation

**Formula:**
```
SSIM(x, y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ

Where:
  l(x,y) = (2μ_x μ_y + C1) / (μ_x² + μ_y² + C1)           # Luminance
  c(x,y) = (2σ_x σ_y + C2) / (σ_x² + σ_y² + C2)           # Contrast
  s(x,y) = (σ_xy + C3) / (σ_x σ_y + C3)                    # Structure
```

**Standard formulation:** α=β=γ=1, so:
```
SSIM = [(2μ_x μ_y + C1)(2σ_xy + C2)] / [(μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)]
```

**Parameters:**
- C1, C2: Small constants for numerical stability
- μ: Mean (luminance)
- σ: Standard deviation (contrast)
- σ_xy: Covariance (structure)
- Computed over local windows (typically 11×11)

**Loss formulation:**
```python
ssim_value = ssim(prediction, target, data_range=1.0)  # Returns [0, 1]
ssim_loss = 1.0 - ssim_value  # Convert to loss (0 is perfect)
```

**Properties:**
- **Range:** SSIM ∈ [0, 1], loss ∈ [0, 1]
- **Symmetric:** SSIM(x, y) = SSIM(y, x)
- **Maximum at identity:** SSIM(x, x) = 1
- **Sensitive to structure:** Penalizes structural distortions more than uniform shifts
- **Multi-scale aware:** Captures local patterns

**Why SSIM for landscapes?**
- Preserves boundaries (forest edges, urban peripheries)
- Maintains spatial coherence (contiguous patches vs. random pixels)
- More aligned with human perception of map quality
- Prevents over-smoothing (blurry predictions look structurally wrong)

**Weight:** 2.0

**Implementation:** `torchmetrics.functional.ssim` or custom implementation

**Masked computation:**
```python
# Set invalid pixels to 0 before SSIM
pred_sanitized = prediction.clone()
target_sanitized = target.clone()
pred_sanitized[~mask] = 0.0
target_sanitized[~mask] = 0.0

ssim_val = ssim(pred_sanitized, target_sanitized, data_range=1.0)
ssim_loss = 1.0 - ssim_val
```

#### 3. Laplacian Pyramid Loss

**Purpose:** Multi-scale detail preservation

**Intuition:**
A Laplacian pyramid decomposes an image into multiple frequency bands:
- Low frequencies: Broad patterns (large regions)
- High frequencies: Fine details (edges, small features)

By computing loss at each level, we ensure predictions match targets at all spatial scales.

**Algorithm:**

```python
def laplacian_pyramid_loss(pred, target, levels=3, kernel_size=5, sigma=1.0):
    """
    Args:
        levels: Number of pyramid levels
        kernel_size: Gaussian blur kernel size
        sigma: Gaussian blur standard deviation
    """
    loss = 0.0
    
    for level in range(levels):
        # Gaussian blur (low-pass filter)
        blurred_pred = gaussian_blur(pred, kernel_size, sigma)
        blurred_target = gaussian_blur(target, kernel_size, sigma)
        
        # Laplacian (high-pass filter): original - blurred
        lap_pred = pred - blurred_pred
        lap_target = target - blurred_target
        
        # L1 loss on this frequency band
        loss += torch.mean(torch.abs(lap_pred[mask] - lap_target[mask]))
        
        # Downsample for next level
        pred = F.avg_pool2d(blurred_pred, 2)
        target = F.avg_pool2d(blurred_target, 2)
    
    # Include lowest frequency (final blurred image)
    if include_lowpass:
        loss += torch.mean(torch.abs(pred[mask] - target[mask]))
    
    return loss / (levels + include_lowpass)
```

**Configuration:**
- **Levels:** 3 (captures 3 frequency bands)
- **Kernel size:** 5×5 Gaussian
- **Sigma:** 1.0
- **Include lowpass:** True (also penalize lowest frequency residual)
- **Metric:** L1 (Mean Absolute Error) per level

**Properties:**
- **Multi-resolution:** Separate penalties for coarse and fine details
- **Edge-aware:** High-frequency bands capture boundaries
- **Scale-adaptive:** Automatically adjusts to image size
- **Prevents smoothing:** Can't minimize loss by blurring

**Why for landscapes?**
- Land use transitions have sharp boundaries (forest/field edge)
- Urban expansion shows fine-scale patterns (roads, buildings)
- Regional trends (level 1) + local details (levels 2-3) both matter
- Maintains perceptual sharpness

**Weight:** 1.0

**Implementation:** `src/models/losses.py` - `LaplacianPyramidLoss`

#### 4. Histogram Loss

**Purpose:** Distribution matching for realistic change patterns

**Concept:**
Landscape changes follow characteristic distributions. Most pixels change little (0-0.02), some change moderately (0.02-0.2), few change dramatically (>0.4). Histogram loss ensures our predicted changes match this distribution.

**Histogram Bins:**

| Bin | Change Range | Interpretation |
|-----|--------------|----------------|
| 1 | decrease (<-0.005) | Restoration/reduction |
| 2 | no change (-0.005 to 0.005) | Stable |
| 3 | tiny increase (0.005-0.02) | Minor intensification |
| 4 | small (0.02-0.1) | Moderate change |
| 5 | moderate (0.1-0.2) | Significant change |
| 6 | large (0.2-0.4) | Major transformation |
| 7 | very large (0.4-0.6) | Dramatic shift |
| 8 | extreme (>0.6) | Complete transformation |

**Total bins:** 8 (plus boundaries = 9 edges)

**Formula:**

The loss combines two components:

1. **Cross-Entropy Loss** (distributional matching):
```
CE = -Σ p_true(bin) × log(p_pred(bin) + ε)

Where:
  p_true(bin) = observed fraction of pixels in each bin (from targets)
  p_pred(bin) = predicted fraction of pixels in each bin (from predictions)
  ε = label smoothing (0.05) to prevent log(0)
```

2. **Wasserstein-2 Distance** (ordinal structure):
```
W2 = Σ Σ |bin_i - bin_j| × |CDF_pred(i) - CDF_true(i)|

Captures that being off by 2 bins is worse than being off by 1 bin
```

**Combined formula:**
```
histogram_loss = (CE + 0.1 × W2) / num_bins

Normalized by num_bins (8) for scale compatibility with other losses
```

**Label Smoothing:**
```python
# Instead of hard assignment to bins
p_true_smoothed = (1 - α) × p_true + α / num_bins

Where α = 0.05 (5% smoothing)
```

Prevents overconfident predictions and numerical instability.

**Horizon-Specific Histograms:**

Different prediction horizons have different change distributions:
- 5-year: Smaller changes, peaked around 0-0.02
- 20-year: Larger changes, more spread out

The loss computes separate histograms for each horizon and averages.

**Computation:**

```python
# For each horizon
delta_pred = prediction[:, horizon_idx] - last_input
delta_true = target[:, horizon_idx] - last_input

# Bin the values
pred_hist = torch.histc(delta_pred[mask], bins=histogram_bins)
true_hist = torch.histc(delta_true[mask], bins=histogram_bins)

# Normalize to probabilities
p_pred = pred_hist / pred_hist.sum()
p_true = true_hist / true_hist.sum()

# Apply label smoothing to true distribution
p_true_smooth = (1 - 0.05) * p_true + 0.05 / num_bins

# Cross-entropy
ce_loss = -torch.sum(p_true_smooth * torch.log(p_pred + 1e-8))

# Wasserstein-2
cdf_pred = torch.cumsum(p_pred, dim=0)
cdf_true = torch.cumsum(p_true_smooth, dim=0)
w2_loss = torch.sum(torch.abs(cdf_pred - cdf_true))

# Combined
hist_loss = (ce_loss + 0.1 * w2_loss) / num_bins
```

**Properties:**
- **Distribution-aware**: Prevents systematic bias in change magnitudes
- **Ordinal-sensitive**: W2 component penalizes large bin errors more
- **Normalized**: Division by num_bins keeps values ~0.01-0.1
- **Robust**: Label smoothing prevents numerical issues

**Warmup Strategy:**
- **Epochs 0-19**: histogram_loss = 0 (disabled)
- **Epoch 20+**: histogram_loss active

**Why warmup?**
- Model needs to learn basic patterns first (MSE, SSIM)
- Computing histograms requires reasonable predictions (not random)
- Prevents early training instability
- Allows histogram loss to refine, not dominate

**Weight:** 0.67 (after warmup)

**Implementation:** `src/models/histogram_loss.py` - `HistogramLoss`

#### Loss Components for Quantile Predictions

**Pinball Loss (Quantile Loss)**

**Purpose:** Calibrated uncertainty estimation

**Formula:**
```
PinballLoss(q) = {
    q × (target - prediction)           if target ≥ prediction
    (1 - q) × (prediction - target)     if target < prediction
}

Where q = desired quantile (0.025 or 0.975)
```

**Intuition:**
- For q=0.025 (lower bound): Penalize underprediction more than overprediction
- For q=0.975 (upper bound): Penalize overprediction more than underprediction
- Asymmetric loss encourages predictions to capture the desired percentile

**Example (q=0.025, lower bound):**
```
If target = 0.5:
  prediction = 0.3 → loss = 0.025 × (0.5 - 0.3) = 0.005
  prediction = 0.7 → loss = 0.975 × (0.7 - 0.5) = 0.195  # Much larger!
```
Model learns to avoid over-predicting the lower bound.

**Implementation:**

```python
class PinballLoss(nn.Module):
    def __init__(self, quantile=0.5, reduction='mean'):
        super().__init__()
        self.quantile = quantile
        self.reduction = reduction
    
    def forward(self, pred, target, mask=None):
        error = target - pred
        loss = torch.where(
            error >= 0,
            self.quantile * error,
            (self.quantile - 1) * error
        )
        
        if mask is not None:
            loss = loss[mask]
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss
```

**Configuration:**
- **Lower quantile (q=0.025)**: 2.5th percentile
- **Upper quantile (q=0.975)**: 97.5th percentile
- **Together**: 95% prediction interval

**No weight:** Applied directly without scaling (magnitude already appropriate)

**Implementation:** `src/models/pinball_loss.py` - `PinballLoss`

#### Independent Loss Assignment

**Critical Design:** Central and quantile predictions receive INDEPENDENT gradients

```python
# Central prediction
central_loss = mse + 2.0*ssim + 1.0*lap + 0.67*hist
central_loss.backward()  # Updates only central head weights

# Quantile predictions  
lower_loss = pinball_loss_q025
lower_loss.backward()  # Updates only lower head weights

upper_loss = pinball_loss_q975
upper_loss.backward()  # Updates only upper head weights
```

**Why independent?**
- Central optimizes for accuracy + structure + distribution
- Quantiles optimize for calibrated coverage
- No gradient conflict between objectives
- Each head specializes for its purpose

**Total training loss (for logging only):**
```python
total_loss = central_loss + lower_loss + upper_loss
```

But gradients are computed separately for each head.

#### Loss Calibration Process

**Goal:** Achieve target contribution percentages (MSE 50%, SSIM 20%, etc.)

**Process:**

1. **Initial training** with unit weights (1.0 each)
2. **Monitor** empirical loss values:
   ```
   MSE ≈ 0.1
   SSIM ≈ 0.1
   Laplacian ≈ 0.05
   Histogram ≈ 0.15
   ```

3. **Compute required weights** for target contributions:
   ```
   Target: MSE=50%, SSIM=20%, Lap=10%, Hist=20%
   Total target: 0.2 (arbitrary scale)
   
   w_mse = 0.1 / 0.1 = 1.0        (contributes 0.1)
   w_ssim = 0.04 / 0.1 = 2.0      (contributes 0.2 → but we want 0.04, so need to normalize)
   
   After calibration:
   w_ssim = 2.0 → contributes 2.0 × 0.1 = 0.2 (normalized to 20%)
   w_lap = 1.0 → contributes 1.0 × 0.05 = 0.05 (normalized to 10%)
   w_hist = 0.67 → contributes 0.67 × 0.15 = 0.1 (normalized to 20%)
   ```

4. **Verify** during training via logging

**Logged Metrics:**
- `train_loss`: MSE (central prediction)
- `train_ssim_loss_total`: Averaged SSIM across horizons
- `train_lap_loss_total`: Averaged Laplacian across horizons
- `train_hist_loss_total`: Averaged histogram across horizons
- `train_total_loss`: Sum of all losses (for model selection)

**Implementation:** `src/models/lightning_module.py`

---

## Training

### Intuitive Explanation

Training a deep learning model is like teaching someone a complex skill through practice and feedback. The model starts with random guesses, makes predictions, receives feedback (loss), and adjusts its internal parameters to improve. This process repeats thousands of times until the model becomes proficient.

**The Training Loop:**

1. **Show the model examples**: Feed batches of training data (historical sequences + targets)
2. **Let it make predictions**: Model outputs its best guess for future HM values
3. **Compute error**: Compare predictions to actual targets using our loss functions
4. **Update parameters**: Adjust the model's weights to reduce that error
5. **Repeat**: Do this for thousands of batches across many epochs

**Key Challenges for Our Task:**

- **Long training time**: Global 1km data means millions of pixels, requiring substantial compute
- **Memory constraints**: Cannot load entire global dataset at once; use batch sampling
- **Overfitting risk**: Model might memorize training regions instead of learning general patterns
- **Convergence stability**: Multiple loss functions need balanced optimization

**Our Strategy:**

- **Batch-based training**: Process 256x256 pixel patches randomly sampled from the globe
- **Adam optimizer**: Adaptive learning rates per parameter for efficient convergence
- **Learning rate scheduling**: Start higher, decay over time as model refines
- **Early stopping**: Monitor validation loss and stop if no improvement
- **Histogram warmup**: Delay complex histogram loss until model learns basics
- **Spatial validation split**: Test on unseen geographic regions

Training typically runs for 50-100 epochs (hours to days depending on hardware), with checkpoints saved regularly in case of interruption.

### Technical Details

#### Training Configuration

**Framework:** PyTorch Lightning
- Handles training loop, validation, logging automatically
- Distributed training support for multi-GPU
- Checkpoint management and recovery
- Integration with Weights & Biases for experiment tracking

**Implementation:** `src/models/lightning_module.py` (SpatioTemporalLightningModule)

#### Hyperparameters

**Core Model Parameters:**
```python
hidden_dim = 16                    # ConvLSTM hidden channels
num_layers = 1                     # ConvLSTM layers
kernel_size = 3                    # Convolution kernel
num_static_channels = 7            # Static input channels
num_dynamic_channels = 11          # Dynamic input channels
locenc_out_channels = 8            # Location encoder output
```

**Optimization Parameters:**
```python
learning_rate = 1e-3               # Initial learning rate (0.001)
optimizer = "Adam"                 # Adaptive moment estimation
beta1 = 0.9                        # Adam momentum parameter
beta2 = 0.999                      # Adam second moment parameter
epsilon = 1e-8                     # Numerical stability
weight_decay = 0.0                 # L2 regularization (none)
```

**Loss Weights:**
```python
ssim_weight = 2.0                  # SSIM loss contribution
laplacian_weight = 1.0             # Laplacian loss contribution
histogram_weight = 0.67            # Histogram loss contribution
histogram_warmup_epochs = 20       # Epochs before histogram activates
```

**Training Parameters:**
```python
batch_size = 8                     # Samples per batch
num_epochs = 100                   # Maximum training epochs
patch_size = 256                   # Spatial patch size (256x256)
num_workers = 4                    # Data loading threads
prefetch_factor = 2                # Batches to prefetch per worker
```

**Validation Parameters:**
```python
val_check_interval = 1.0           # Validate every epoch
val_batch_size = 8                 # Validation batch size
early_stopping_patience = 10       # Epochs without improvement before stopping
```

#### Data Loading Strategy

**Training Data Loader:**

```python
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=True,                   # Random order each epoch
    num_workers=4,                  # Parallel loading
    pin_memory=True,                # Faster GPU transfer
    drop_last=True,                 # Consistent batch sizes
    persistent_workers=True         # Keep workers alive between epochs
)
```

**Batch Composition:**
- 8 spatial patches per batch
- Each patch: 256×256 pixels = 65,536 pixels
- Total: ~524k pixels per batch
- Random locations each iteration (diversity)

**Spatial Sampling:**
```python
# Pseudocode for batch generation
def get_batch():
    patches = []
    for _ in range(batch_size):
        # Random spatial location
        region = random_crop(global_extent, size=256)
        
        # Load temporal sequence at this location
        patch = load_sequence(region, years=[1990, 1995, 2000, 2005])
        patches.append(patch)
    
    return collate(patches)
```

**Validation Data Loader:**

```python
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=8,
    shuffle=False,                  # Fixed order for reproducibility
    num_workers=4,
    pin_memory=True,
    drop_last=False                 # Keep all validation samples
)
```

**Key Difference:** Validation uses fixed spatial tiles, not random crops

#### Optimization Algorithm

**Adam Optimizer:**

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0
)
```

**Why Adam?**
- **Adaptive learning rates**: Each parameter gets its own effective learning rate
- **Momentum**: Accelerates convergence by accumulating gradients
- **Scale-invariant**: Handles different parameter magnitudes well
- **Robust**: Works well with sparse gradients and noisy data

**Update Rule:**

```
# For each parameter θ and gradient g:
m_t = β₁ × m_{t-1} + (1-β₁) × g_t        # First moment (momentum)
v_t = β₂ × v_{t-1} + (1-β₂) × g_t²       # Second moment (variance)

m̂_t = m_t / (1 - β₁^t)                   # Bias correction
v̂_t = v_t / (1 - β₂^t)

θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)   # Parameter update
```

**No weight decay** (L2 regularization) because:
- Dataset is large enough to prevent overfitting
- Model is relatively small (low capacity)
- Spatial validation split provides strong generalization test

#### Learning Rate Schedule

**Strategy:** Constant learning rate with optional manual decay

**Initial:** lr = 1e-3 (0.001)

**No automatic scheduling** currently, but could add:
- **ReduceLROnPlateau**: Reduce when validation loss plateaus
- **CosineAnnealingLR**: Smooth decay following cosine curve
- **StepLR**: Decay by factor every N epochs

**Rationale for constant LR:**
- Simple and effective for this problem
- Training is relatively stable
- Can manually adjust if needed during training

**Future improvement:**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)
```

#### Training Loop

**Epoch Structure:**

```python
for epoch in range(num_epochs):
    # === TRAINING PHASE ===
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        # 1. Forward pass
        predictions = model(
            input_dynamic=batch['input_dynamic'],
            input_static=batch['input_static'],
            lonlat=batch['lonlat']
        )
        
        # 2. Compute losses
        loss_dict = compute_losses(predictions, batch['targets'])
        total_loss = loss_dict['total']
        
        # 3. Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # 4. Update parameters
        optimizer.step()
        
        # 5. Log metrics
        logger.log(loss_dict, step=global_step)
    
    # === VALIDATION PHASE ===
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            val_predictions = model(...)
            val_loss_dict = compute_losses(val_predictions, batch['targets'])
            logger.log(val_loss_dict, step=epoch)
    
    # === CHECKPOINTING ===
    if val_loss < best_val_loss:
        save_checkpoint(model, optimizer, epoch)
        best_val_loss = val_loss
```

**Gradient Flow:**

```
Input → LocationEncoder → ConvLSTM → [12 Prediction Heads]
  ↓           ↓               ↓              ↓
Grad ← Grad Backprop ← Grad ← [Independent Loss per Head]
```

Each prediction head receives gradients only from its assigned loss.

#### Histogram Loss Warmup

**Implementation:**

```python
def compute_histogram_loss(self, pred, target, mask, epoch):
    if epoch < self.histogram_warmup_epochs:
        return torch.tensor(0.0, device=pred.device)
    else:
        return self.histogram_loss_fn(pred, target, mask)
```

**Logging:**
```python
# Debug print when activated
if epoch == histogram_warmup_epochs and batch_idx == 0:
    print(f"[HISTOGRAM ACTIVATED] Epoch {epoch}")
```

**Effect on Training:**
- Epochs 0-19: Model learns basic spatial-temporal patterns
- Epoch 20: Histogram loss suddenly activates
- Epoch 20+: Model refines predictions to match change distributions
- Typical observation: Small jump in total loss at epoch 20, then continued improvement

#### Checkpointing and Model Selection

**Checkpoint Strategy:**

```python
checkpoint_callback = ModelCheckpoint(
    dirpath='models/checkpoints/',
    filename='convlstm-{epoch:02d}-{val_loss:.4f}',
    monitor='val_total_loss',        # Metric to track
    mode='min',                      # Lower is better
    save_top_k=3,                    # Keep best 3 checkpoints
    save_last=True,                  # Also save most recent
    verbose=True
)
```

**Saved Content:**
- Model weights (all layers)
- Optimizer state (for resuming training)
- Epoch number
- Training and validation metrics
- Hyperparameters

**Model Selection Criteria:**

Primary: `val_total_loss` (combined loss on validation set)
- Balances all objectives (accuracy, structure, distribution, uncertainty)
- Most aligned with training objective
- Ensures model generalizes to unseen regions

Alternative metrics logged but not used for selection:
- `val_mae_total`: Mean absolute error
- `val_coverage_total`: Quantile calibration (should be ~95%)

#### Early Stopping

**Configuration:**

```python
early_stop_callback = EarlyStopping(
    monitor='val_total_loss',
    patience=10,                     # Wait 10 epochs without improvement
    mode='min',
    verbose=True,
    min_delta=0.0001                # Minimum change to qualify as improvement
)
```

**Logic:**
1. Track best validation loss seen so far
2. If current val_loss < best_val_loss - min_delta: reset patience counter
3. Else: increment patience counter
4. If patience counter reaches 10: stop training
5. Load best checkpoint for final model

**Why 10 epochs patience?**
- Training is noisy; single bad epoch doesn't mean overfitting
- Allows time for histogram loss to stabilize after warmup
- Prevents premature stopping
- Typical training runs 50-80 epochs before stopping

#### Experiment Tracking

**Weights & Biases Integration:**

```python
logger = WandbLogger(
    project='spatio-temporal-hm',
    name=f'convlstm-{experiment_id}',
    log_model=True,                  # Save model artifacts
    save_dir='wandb/'
)
```

**Logged Metrics (every batch for training, every epoch for validation):**

**Training:**
- `train_loss` (MSE)
- `train_mae_total`
- `train_ssim_loss_total`
- `train_lap_loss_total`
- `train_hist_loss_total`
- `train_pinball_lower_total`
- `train_pinball_upper_total`
- `train_total_loss`

**Per-horizon metrics:** `train_mae_5yr`, `train_mae_10yr`, etc.

**Validation:**
- All training metrics (with `val_` prefix)
- `val_coverage_5yr`, `val_coverage_10yr`, etc. (quantile calibration)
- `val_coverage_total` (average coverage)

**System Metrics:**
- GPU utilization
- Memory usage
- Training speed (batches/sec)
- Learning rate

#### Training Best Practices

**Batch Size Selection:**
- **Too small** (< 4): Noisy gradients, slow convergence
- **Too large** (> 16): Memory overflow, poor generalization
- **Sweet spot**: 8 provides stable gradients within memory limits

**Patch Size Selection:**
- **Too small** (< 128): Insufficient spatial context for ConvLSTM
- **Too large** (> 512): Memory issues, fewer samples per epoch
- **Sweet spot**: 256 balances context and efficiency

**Debugging Strategies:**
1. **Check NaN losses**: Indicates numerical instability
   - Solution: Reduce learning rate, check data normalization
2. **Loss not decreasing**: Model not learning
   - Solution: Verify data loading, check gradient flow, simplify losses
3. **Validation loss increasing**: Overfitting
   - Solution: Stop earlier, add regularization, more data augmentation
4. **Coverage far from 95%**: Quantile miscalibration
   - Solution: Adjust pinball loss weight, check for bias in data

**Common Issues and Solutions:**

| Issue | Symptom | Solution |
|-------|---------|----------|
| GPU OOM | Training crashes | Reduce batch_size or patch_size |
| Slow training | < 1 batch/sec | Increase num_workers, check I/O bottleneck |
| Histogram loss = 0 | After epoch 20 | Check warmup logic, monitor raw values |
| Poor coverage | << 95% or >> 95% | Verify pinball loss implementation |
| Blurry predictions | Low SSIM, high MSE | Increase SSIM weight |

**Training Command Example:**

```bash
python scripts/train_lightning.py \
    --data_root data/raw/hm_global \
    --batch_size 8 \
    --learning_rate 1e-3 \
    --num_epochs 100 \
    --hidden_dim 16 \
    --histogram_warmup_epochs 20 \
    --gpus 1 \
    --log_every_n_steps 50 \
    --val_check_interval 1.0
```

#### Computational Requirements

**Single Training Run:**
- **Time**: 12-48 hours (depends on GPU, dataset size)
- **GPU Memory**: ~8-16 GB (for batch_size=8, patch_size=256)
- **Storage**: ~500 MB per checkpoint
- **Recommended GPU**: NVIDIA RTX 3090, A100, or equivalent

**Multi-GPU Training:**

```python
trainer = Trainer(
    accelerator='gpu',
    devices=4,                       # Use 4 GPUs
    strategy='ddp',                  # Distributed data parallel
    precision=16                     # Mixed precision (faster, less memory)
)
```

**Distributed Training Benefits:**
- 4× speedup (linear scaling with GPUs)
- Same effective batch size per GPU
- Synchronized gradient updates

---

## Accuracy Assessment

### Intuitive Explanation

After training, we need to rigorously evaluate whether the model actually works. "Works" means several things for our multi-faceted prediction task:

1. **Point Accuracy**: Are individual pixel predictions close to reality? (MAE, MSE)
2. **Spatial Coherence**: Do predictions form realistic patterns, not random noise? (SSIM)
3. **Detail Preservation**: Are boundaries and fine features maintained? (Laplacian)
4. **Distribution Realism**: Do predicted changes match the characteristic distribution of real landscape change? (Histogram)
5. **Uncertainty Calibration**: Do 95% of observations fall within the prediction intervals? (Coverage)
6. **Generalization**: Does the model work on unseen geographic regions, not just training areas?

We assess performance across **four temporal horizons** (5, 10, 15, 20 years) because near-term predictions should be more accurate than long-term ones. We also track metrics separately for training and validation sets to detect overfitting.

**The Validation Set is Critical:**

Training metrics tell us what the model can achieve on data it has seen. Validation metrics tell us what it can achieve on data it hasn't seen—the real test of predictive power. We use a **spatial split**: training regions and validation regions are geographically separate. This tests whether the model learned general landscape dynamics or just memorized specific locations.

**Multi-Horizon Evaluation:**

Different horizons face different challenges:
- **5-year**: Should be most accurate (short extrapolation)
- **10-year**: Moderate accuracy (medium-term trends)
- **15-year**: More challenging (longer extrapolation)
- **20-year**: Most uncertain (maximum horizon)

If 20-year predictions are as accurate as 5-year predictions, something is wrong—the model may not be truly learning temporal dynamics.

### Technical Details

#### Evaluation Metrics

**1. Mean Absolute Error (MAE)**

**Purpose:** Average pixel-wise prediction error

**Formula:**
```
MAE = (1/N) Σ |prediction - target|
```

**Interpretation:**
- Units: HM scale [0, 1]
- Lower is better
- Directly interpretable: "on average, predictions are off by X"
- Less sensitive to outliers than MSE

**Computation:**
```python
mae = torch.mean(torch.abs(prediction[mask] - target[mask]))
```

**Typical Values:**
- Excellent: MAE < 0.05 (5% of HM scale)
- Good: MAE = 0.05-0.10
- Fair: MAE = 0.10-0.15
- Poor: MAE > 0.15

**Logged Metrics:**
- `val_mae_5yr`, `val_mae_10yr`, `val_mae_15yr`, `val_mae_20yr`
- `val_mae_total` (average across horizons)

**2. Mean Squared Error (MSE)**

**Purpose:** Pixel-wise error with emphasis on large mistakes

**Formula:**
```
MSE = (1/N) Σ (prediction - target)²
```

**Interpretation:**
- Quadratic penalty: large errors weighted heavily
- Units: (HM scale)² [0, 1]
- Training objective (part of total loss)
- Related to MAE: RMSE = √MSE

**Logged Metrics:**
- `val_loss` (MSE on validation set)
- Used in training, not primary evaluation metric

**3. Structural Similarity Index (SSIM)**

**Purpose:** Perceptual similarity of spatial patterns

**Formula:** (See Loss Functions section for full derivation)

```
SSIM ∈ [0, 1]
where 1 = perfect match, 0 = no similarity
```

**Interpretation:**
- Captures pattern preservation
- More aligned with human perception than MSE
- SSIM = 0.9 → visually similar
- SSIM = 0.5 → recognizable but distorted
- SSIM < 0.3 → poor pattern match

**Logged Metrics:**
- `val_ssim_loss_total` (1 - SSIM, averaged across horizons)
- Lower loss = higher SSIM = better

**Expected Values:**
- Excellent: SSIM > 0.9 (loss < 0.1)
- Good: SSIM = 0.8-0.9 (loss = 0.1-0.2)
- Fair: SSIM = 0.6-0.8 (loss = 0.2-0.4)
- Poor: SSIM < 0.6 (loss > 0.4)

**4. Laplacian Pyramid Loss**

**Purpose:** Multi-scale detail preservation

**Interpretation:**
- Lower values = better detail preservation
- Captures both coarse (regional) and fine (local) accuracy
- Prevents over-smoothing

**Logged Metrics:**
- `val_lap_loss_total` (averaged across horizons)

**Typical Values:**
- Excellent: < 0.03
- Good: 0.03-0.05
- Fair: 0.05-0.08
- Poor: > 0.08

**5. Histogram Loss**

**Purpose:** Distribution matching of change patterns

**Interpretation:**
- Measures whether predicted changes follow realistic distributions
- Includes cross-entropy (distributional match) and Wasserstein (ordinal structure)
- Normalized by num_bins (8)

**Logged Metrics:**
- `val_hist_loss_total` (averaged across horizons)

**Typical Values:**
- Excellent: < 0.10
- Good: 0.10-0.15
- Fair: 0.15-0.25
- Poor: > 0.25

**6. Pinball Loss (Quantile Evaluation)**

**Purpose:** Assess uncertainty estimation quality

**Formula:**
```
PinballLoss(q) = {
    q × (target - prediction)           if target ≥ prediction
    (1 - q) × (prediction - target)     if target < prediction
}
```

**Interpretation:**
- Asymmetric loss for quantile optimization
- Lower values = better calibrated intervals

**Logged Metrics:**
- `val_pinball_lower_total` (2.5th percentile)
- `val_pinball_upper_total` (97.5th percentile)

**7. Coverage (Quantile Calibration)**

**Purpose:** Measure if prediction intervals are correctly calibrated

**Formula:**
```
Coverage = (# pixels where lower ≤ target ≤ upper) / (# valid pixels) × 100%
```

**Target:** 95% (for 2.5th to 97.5th percentile interval)

**Interpretation:**
- Coverage = 95%: Perfect calibration
- Coverage < 95%: Intervals too narrow (overconfident)
- Coverage > 95%: Intervals too wide (underconfident)

**Acceptable Range:** 93-97% (allowing for sampling variability)

**Logged Metrics:**
- `val_coverage_5yr`, `val_coverage_10yr`, `val_coverage_15yr`, `val_coverage_20yr`
- `val_coverage_total` (average across horizons)

**Example:**
```
If coverage = 92%:
- Only 92% of observations fall in [lower, upper]
- Intervals are too narrow
- Model is overconfident about predictions
```

**8. Total Loss**

**Purpose:** Combined metric for model selection

**Formula:**
```
total_loss = MSE + 2.0×SSIM + 1.0×Laplacian + 0.67×Histogram + Pinball_lower + Pinball_upper
```

**Logged Metric:** `val_total_loss`

**Use:** Primary metric for model selection during training

#### Evaluation Protocol

**Validation Set Characteristics:**
- **Size:** Typically 10-20% of training data
- **Split Type:** Spatial (geographic regions)
- **Selection:** Random geographic tiles, not random pixels
- **Fixed:** Same tiles used throughout training (reproducibility)

**Evaluation Frequency:**
- **During Training:** Every epoch
- **Final Evaluation:** On best checkpoint (lowest val_total_loss)

**Evaluation Process:**

```python
def evaluate(model, val_loader):
    model.eval()
    metrics = defaultdict(list)
    
    with torch.no_grad():
        for batch in val_loader:
            # Forward pass
            predictions = model(
                batch['input_dynamic'],
                batch['input_static'],
                batch['lonlat']
            )
            
            # Compute metrics for each horizon
            for h_idx, horizon in enumerate(['5yr', '10yr', '15yr', '20yr']):
                pred_lower = predictions[:, 3*h_idx]
                pred_central = predictions[:, 3*h_idx+1]
                pred_upper = predictions[:, 3*h_idx+2]
                target = batch[f'target_{horizon}']
                mask = compute_mask(batch)
                
                # Compute all metrics
                metrics[f'mae_{horizon}'].append(
                    mae(pred_central, target, mask)
                )
                metrics[f'coverage_{horizon}'].append(
                    coverage(pred_lower, pred_upper, target, mask)
                )
                # ... other metrics
    
    # Aggregate across batches
    final_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return final_metrics
```

#### Per-Horizon Analysis

**Expected Performance Degradation:**

Prediction accuracy should decrease with horizon distance:

| Horizon | Expected MAE | Expected SSIM | Expected Coverage |
|---------|--------------|---------------|-------------------|
| 5yr     | 0.05-0.08    | 0.85-0.90     | 94-96%            |
| 10yr    | 0.07-0.11    | 0.80-0.85     | 93-96%            |
| 15yr    | 0.09-0.14    | 0.75-0.82     | 93-97%            |
| 20yr    | 0.11-0.17    | 0.70-0.80     | 92-97%            |

**Why degradation?**
- Longer extrapolation = more uncertainty
- Cumulative error propagation
- Unpredictable events (policy changes, disasters)
- Chaotic dynamics in complex systems

**Anomaly Indicators:**
- **Flat performance across horizons**: Model not learning temporal dynamics
- **Sudden drop at one horizon**: Specific issue with that target data or model head
- **Better long-term than short-term**: Data leakage or evaluation error

#### Visualization for Evaluation

**Visual Inspection is Critical:**

Metrics provide quantitative assessment, but visual inspection reveals patterns:

**1. Prediction Maps:**
```python
# For a test region
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Predictions
for i, horizon in enumerate(['5yr', '10yr', '15yr', '20yr']):
    axes[0, i].imshow(predictions[horizon])
    axes[0, i].set_title(f'Predicted {horizon}')

# Row 2: Targets
for i, horizon in enumerate(['5yr', '10yr', '15yr', '20yr']):
    axes[1, i].imshow(targets[horizon])
    axes[1, i].set_title(f'Actual {horizon}')
```

**What to Look For:**
- Spatial coherence (not checkerboard patterns)
- Boundary sharpness (not overly blurred)
- Realistic patterns (urban cores, forest patches)
- No artifacts near NaN regions

**2. Error Maps:**
```python
error_map = prediction - target
plt.imshow(error_map, cmap='RdBu', vmin=-0.2, vmax=0.2)
plt.colorbar(label='Prediction Error')
```

**What to Look For:**
- Random errors (good) vs. systematic spatial patterns (bad)
- Large errors clustered in specific regions (model blind spots)
- Edge artifacts near coastlines or data boundaries

**3. Uncertainty Visualization:**
```python
# Prediction intervals
interval_width = pred_upper - pred_lower
plt.imshow(interval_width, cmap='viridis')
plt.colorbar(label='Prediction Interval Width')
```

**What to Look For:**
- Wider intervals in uncertain regions (appropriate)
- Spatial patterns in uncertainty (model knows where it's uncertain)
- Coverage: color overlay showing where target falls outside interval

**4. Change Distribution Comparison:**
```python
# Histogram of predicted vs. actual changes
delta_pred = prediction - last_input
delta_true = target - last_input

plt.hist(delta_pred.flatten(), bins=50, alpha=0.5, label='Predicted')
plt.hist(delta_true.flatten(), bins=50, alpha=0.5, label='Actual')
plt.xlabel('HM Change')
plt.ylabel('Frequency')
plt.legend()
```

**What to Look For:**
- Similar shapes (histogram loss working)
- Predicted distribution matches peak, spread, skew of actual
- No systematic bias (mean of distributions aligned)

#### Training vs. Validation Comparison

**Overfitting Indicators:**

| Metric | Training | Validation | Interpretation |
|--------|----------|------------|----------------|
| MAE | 0.05 | 0.05 | Excellent generalization |
| MAE | 0.05 | 0.08 | Some overfitting (acceptable) |
| MAE | 0.05 | 0.15 | Severe overfitting (problem) |

**Acceptable Gap:** Train metrics 10-20% better than validation
**Problematic Gap:** Train metrics >30% better than validation

**If Overfitting Detected:**
1. Stop training earlier (reduce epochs)
2. Add data augmentation
3. Reduce model capacity (smaller hidden_dim)
4. Add regularization (weight decay, dropout)
5. Collect more training data

#### Horizon-Specific Degradation Analysis

**Compute degradation rates:**

```python
mae_5yr = 0.06
mae_10yr = 0.09
mae_15yr = 0.12
mae_20yr = 0.16

# Degradation per 5-year interval
deg_10 = (mae_10yr - mae_5yr) / 1  # 0.03
deg_15 = (mae_15yr - mae_10yr) / 1  # 0.03
deg_20 = (mae_20yr - mae_15yr) / 1  # 0.04

# Expected: ~linear or slightly accelerating
# Problem: sudden jump or flat line
```

**Healthy Pattern:** Gradual, consistent increase in error with horizon

**Unhealthy Patterns:**
- Sudden spike: Check target data for that specific year
- Flat performance: Model not learning temporal dynamics
- Non-monotonic: Possible training issue or data artifact

#### Statistical Significance Testing

**Comparing Model Versions:**

When evaluating improvements (e.g., "did adding histogram loss help?"):

```python
from scipy import stats

# Collect per-sample errors for both models
errors_baseline = [mae(baseline_pred[i], target[i]) for i in range(n_samples)]
errors_new = [mae(new_pred[i], target[i]) for i in range(n_samples)]

# Paired t-test (same samples)
t_stat, p_value = stats.ttest_rel(errors_baseline, errors_new)

if p_value < 0.05 and mean(errors_new) < mean(errors_baseline):
    print("New model is significantly better (p < 0.05)")
```

**Important:** Use validation set, not training set

#### Reporting Results

**Standard Reporting Table:**

| Metric | 5yr | 10yr | 15yr | 20yr | Average |
|--------|-----|------|------|------|---------|
| **MAE** | 0.063 | 0.091 | 0.118 | 0.152 | 0.106 |
| **SSIM** | 0.874 | 0.832 | 0.791 | 0.743 | 0.810 |
| **Coverage** | 94.8% | 95.2% | 95.7% | 96.1% | 95.5% |

**Key Takeaways to Report:**
1. Overall performance level (excellent/good/fair/poor)
2. Temporal degradation pattern (linear/accelerating/stable)
3. Uncertainty calibration (coverage near 95%)
4. Generalization ability (train vs. val gap)
5. Spatial patterns (qualitative from visualizations)

**Example Summary:**

> "The model achieves good performance across all horizons, with MAE ranging from 0.06 (5yr) to 0.15 (20yr) on the validation set. SSIM values (>0.74) indicate strong preservation of spatial patterns. Prediction intervals are well-calibrated with coverage of 95.5% averaged across horizons. The linear degradation in accuracy with temporal horizon (MAE increases ~0.03 per 5 years) suggests the model appropriately captures increasing uncertainty in longer-term predictions. Training-validation gap of 15% indicates healthy generalization to unseen geographic regions."

---

## Prediction

### Intuitive Explanation

Once trained and validated, the model becomes a tool for generating future landscape predictions for any location where we have the required input data. This is where the model transitions from research artifact to practical decision-support tool.

**The Prediction Process:**

1. **Define the region of interest**: Could be a country, protected area, watershed, or any geographic extent
2. **Gather input data**: Historical HM sequences (4 timesteps), static variables, and location coordinates
3. **Run the model**: Process the region in tiles to manage memory
4. **Generate outputs**: Predictions for 4 future horizons (5, 10, 15, 20 years) with uncertainty bounds
5. **Save and visualize**: Export as GeoTIFF files for GIS integration and create maps

**Why Tile-Based Processing?**

A country-sized region might be 1000×1000 km = 1,000,000 pixels at 1km resolution. We can't load this all into GPU memory at once. Instead, we:
- Divide the region into manageable tiles (e.g., 256×256 or 512×512 pixels)
- Process each tile independently
- Stitch predictions back together into seamless regional maps

**Key Considerations:**

- **Edge effects**: Tiles share borders; ensure smooth transitions without visible seams
- **Missing data**: Handle ocean, no-data regions, and incomplete coverage gracefully
- **Computational time**: Large regions may take hours to process; show progress, save incrementally
- **Memory management**: Clear GPU memory between tiles to prevent accumulation
- **Output format**: GeoTIFF with proper georeference for compatibility with GIS software

**What You Get:**

For each prediction horizon (5yr, 10yr, 15yr, 20yr):
- **Central prediction**: Best estimate of future HM
- **Lower bound (2.5th percentile)**: Optimistic scenario
- **Upper bound (97.5th percentile)**: Pessimistic scenario

**Use Cases:**

- **Conservation planning**: Identify areas likely to face high modification pressure
- **Policy evaluation**: Compare predicted impacts under different scenarios
- **Risk assessment**: Quantify uncertainty in future landscapes
- **Monitoring targets**: Set baselines for tracking actual vs. predicted change

### Technical Details

#### Prediction Pipeline

**Implementation:** `scripts/train_lightning.py` (inference mode) or dedicated prediction script

**High-Level Workflow:**

```python
# 1. Load trained model
model = load_checkpoint('models/best_model.ckpt')
model.eval()

# 2. Define region
region_geojson = 'config/region_to_predict.geojson'

# 3. Load data for region
data_loader = create_prediction_dataloader(
    region=region_geojson,
    input_years=[2010, 2015, 2020],  # Latest available data
    tile_size=512,
    overlap=64  # Overlap to prevent edge artifacts
)

# 4. Generate predictions
predictions = []
for tile in data_loader:
    with torch.no_grad():
        pred = model(tile['input_dynamic'], tile['input_static'], tile['lonlat'])
    predictions.append(pred)

# 5. Stitch tiles together
full_prediction = stitch_tiles(predictions, overlap=64)

# 6. Save outputs
save_geotiff(full_prediction, 'outputs/predictions_2025_2030_2035_2040.tif')
```

#### Region Configuration

**GeoJSON Format:**

```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "geometry": {
      "type": "Polygon",
      "coordinates": [[
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min]
      ]]
    },
    "properties": {
      "name": "Study Region"
    }
  }]
}
```

**Configuration File:** `config/region_to_predict.geojson`

**Examples Provided:**
- `region_to_predict_small.geojson`: Small test region (~100×100 km)
- `region_to_predict.geojson`: Medium region (~500×500 km)
- `region_to_predict_large.geojson`: Large region (~1000×1000 km)

#### Data Loading for Prediction

**Requirements:**

Must have data for the 4 most recent timesteps (5-year intervals):
- Example: [2005, 2010, 2015, 2020] → Predict [2025, 2030, 2035, 2040]

**Data Checklist:**

**Dynamic Variables (11 channels) for each timestep:**
- [ ] HM_YEAR_AA_1000.tiff (target variable history)
- [ ] HM_YEAR_AG_1000.tiff through HM_YEAR_population_1000.tiff

**Static Variables (7 channels):**
- [ ] hm_static_ele_1000.tiff
- [ ] hm_static_tas_1000.tiff through hm_static_iucn_strict_1000.tiff

**Spatial Coverage:**
- Data must overlap with region of interest
- Missing data (NaN) will result in NaN predictions for those pixels

#### Tile-Based Processing

**Tiling Strategy:**

```python
def create_tiles(region_bounds, tile_size=512, overlap=64):
    """
    Divide region into overlapping tiles.
    
    Args:
        region_bounds: (xmin, ymin, xmax, ymax) in pixel coordinates
        tile_size: Tile dimension (pixels)
        overlap: Overlap between tiles (pixels) for smooth stitching
        
    Returns:
        List of tile coordinates
    """
    tiles = []
    stride = tile_size - overlap
    
    xmin, ymin, xmax, ymax = region_bounds
    for x in range(xmin, xmax, stride):
        for y in range(ymin, ymax, stride):
            tile = {
                'x_start': x,
                'y_start': y,
                'x_end': min(x + tile_size, xmax),
                'y_end': min(y + tile_size, ymax)
            }
            tiles.append(tile)
    
    return tiles
```

**Why Overlap?**

- Prevents edge artifacts at tile boundaries
- Center portion of each tile is most reliable
- Overlapping regions are averaged or blended
- Typical overlap: 32-128 pixels (depending on tile size)

**Processing Loop:**

```python
for tile_idx, tile_coords in enumerate(tiles):
    # Extract tile data
    tile_data = extract_tile(
        dynamic_data, static_data, lonlat,
        x_start=tile_coords['x_start'],
        y_start=tile_coords['y_start'],
        x_end=tile_coords['x_end'],
        y_end=tile_coords['y_end']
    )
    
    # Normalize inputs (using saved stats from training)
    tile_data = normalize(tile_data, comp_means, comp_stds, static_means, static_stds)
    
    # Model forward pass
    with torch.no_grad():
        pred = model(
            tile_data['input_dynamic'],
            tile_data['input_static'],
            tile_data['lonlat']
        )
    
    # Store prediction
    predictions[tile_idx] = pred.cpu().numpy()
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Progress update
    print(f"Processed tile {tile_idx+1}/{len(tiles)}")
```

#### Stitching Tiles

**Simple Stitching (No Overlap):**

```python
# Place tiles in their correct positions
full_prediction = np.zeros((num_horizons*3, height, width))

for tile, coords in zip(predictions, tile_coords):
    x_start, y_start = coords['x_start'], coords['y_start']
    x_end, y_end = coords['x_end'], coords['y_end']
    
    full_prediction[:, y_start:y_end, x_start:x_end] = tile
```

**Blended Stitching (With Overlap):**

```python
def blend_tiles(predictions, tile_coords, overlap=64):
    """
    Blend overlapping tiles using distance-weighted averaging.
    
    Pixels in overlap region are weighted by distance from tile edge:
    - Center of tile: weight = 1.0
    - Edge of tile: weight = 0.0
    - Smooth transition in overlap region
    """
    # Create accumulation arrays
    full_prediction = np.zeros((channels, height, width))
    weight_map = np.zeros((height, width))
    
    for tile, coords in zip(predictions, tile_coords):
        # Create weight matrix (higher in center, lower at edges)
        weights = create_distance_weights(tile.shape, overlap)
        
        # Accumulate predictions and weights
        x_slice = slice(coords['x_start'], coords['x_end'])
        y_slice = slice(coords['y_start'], coords['y_end'])
        
        full_prediction[:, y_slice, x_slice] += tile * weights
        weight_map[y_slice, x_slice] += weights
    
    # Normalize by total weight
    full_prediction /= weight_map[np.newaxis, :, :]
    
    return full_prediction
```

#### Handling Missing Data

**NaN Propagation:**

```python
# During tile processing
if torch.isnan(input_dynamic).any() or torch.isnan(input_static).any():
    # Compute validity mask
    mask = torch.isfinite(input_dynamic).all() & torch.isfinite(input_static).all()
    
    # Sanitize for model forward
    input_dynamic = torch.nan_to_num(input_dynamic, nan=0.0)
    input_static = torch.nan_to_num(input_static, nan=0.0)
    
    # Forward pass
    pred = model(input_dynamic, input_static, lonlat)
    
    # Set invalid regions back to NaN
    pred[~mask] = float('nan')
```

**Output Interpretation:**
- **Finite values**: Valid predictions
- **NaN**: No prediction (insufficient input data)
- Visualize NaN as transparent or gray in maps

#### Output Format

**GeoTIFF Structure:**

```
Multi-band GeoTIFF (12 bands total)

Band Organization (per pixel):
  Band 1: Lower bound, 5-year horizon (2.5th percentile)
  Band 2: Central prediction, 5-year horizon
  Band 3: Upper bound, 5-year horizon (97.5th percentile)
  Band 4: Lower bound, 10-year horizon
  Band 5: Central prediction, 10-year horizon
  Band 6: Upper bound, 10-year horizon
  Band 7: Lower bound, 15-year horizon
  Band 8: Central prediction, 15-year horizon
  Band 9: Upper bound, 15-year horizon
  Band 10: Lower bound, 20-year horizon
  Band 11: Central prediction, 20-year horizon
  Band 12: Upper bound, 20-year horizon

Georeference: Same as input data (e.g., WGS84, 1km resolution)
Data Type: Float32
No Data Value: NaN
```

**Saving Predictions:**

```python
import rasterio
from rasterio.transform import from_bounds

# Get geotransform from input data
transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

# Create GeoTIFF
with rasterio.open(
    'outputs/predictions.tif',
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=12,  # 12 bands
    dtype=rasterio.float32,
    crs='EPSG:4326',  # WGS84
    transform=transform,
    compress='lzw',  # Compression
    nodata=float('nan')
) as dst:
    # Write each band
    for band_idx in range(12):
        dst.write(predictions[band_idx], band_idx + 1)
    
    # Add metadata
    dst.update_tags(
        description='HM predictions from ConvLSTM',
        horizons='5yr,10yr,15yr,20yr',
        quantiles='2.5%,50%,97.5%'
    )
```

#### Prediction Workflow Example

**Command-Line Interface:**

```bash
python scripts/predict.py \
    --checkpoint models/best_model.ckpt \
    --region config/region_to_predict.geojson \
    --input_years 2010,2015,2020 \
    --output_dir outputs/predictions/ \
    --tile_size 512 \
    --overlap 64 \
    --batch_size 4 \
    --device cuda:0
```

**Configuration Parameters:**
- `--checkpoint`: Path to trained model
- `--region`: GeoJSON defining prediction area
- `--input_years`: Comma-separated years for input sequence
- `--output_dir`: Where to save predictions
- `--tile_size`: Tile dimensions (trade-off: memory vs. efficiency)
- `--overlap`: Overlap between tiles (larger = smoother but slower)
- `--batch_size`: Number of tiles processed simultaneously
- `--device`: CPU or GPU device

**Python API:**

```python
from src.prediction import predict_region

predictions = predict_region(
    model_path='models/best_model.ckpt',
    region_geojson='config/region_to_predict.geojson',
    data_root='data/raw/hm_global',
    input_years=[2010, 2015, 2020],
    output_path='outputs/predictions.tif',
    tile_size=512,
    overlap=64,
    device='cuda'
)
```

#### Post-Processing

**1. Change Maps:**

```python
# Compute change from last input (2020) to predictions
last_input = load_raster('data/raw/hm_global/HM_2020_AA_1000.tif')

for horizon in ['5yr', '10yr', '15yr', '20yr']:
    # Extract central prediction for this horizon
    pred = predictions[horizon]['central']
    
    # Compute change
    change = pred - last_input
    
    # Save
    save_geotiff(change, f'outputs/change_map_{horizon}.tif')
```

**2. Risk Maps:**

```python
# High modification risk: areas predicted to exceed threshold
threshold = 0.7  # 70% human modification

for horizon in ['5yr', '10yr', '15yr', '20yr']:
    pred_central = predictions[horizon]['central']
    
    # Binary risk map
    risk_map = (pred_central > threshold).astype(np.uint8)
    
    save_geotiff(risk_map, f'outputs/risk_map_{horizon}.tif')
```

**3. Uncertainty Maps:**

```python
# Interval width as uncertainty measure
for horizon in ['5yr', '10yr', '15yr', '20yr']:
    lower = predictions[horizon]['lower']
    upper = predictions[horizon]['upper']
    
    uncertainty = upper - lower
    
    save_geotiff(uncertainty, f'outputs/uncertainty_{horizon}.tif')
```

#### Visualization

**Map Creation:**

```python
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load prediction
pred = rasterio.open('outputs/predictions.tif').read(2)  # Band 2 = central 5yr

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Custom colormap (white = low HM, red = high HM)
colors = ['#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f']
cmap = LinearSegmentedColormap.from_list('hm', colors)

# Plot
im = ax.imshow(pred, cmap=cmap, vmin=0, vmax=1)
ax.set_title('Predicted Human Modification (2025)', fontsize=14)
ax.axis('off')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Human Modification Index', fontsize=12)

plt.tight_layout()
plt.savefig('outputs/prediction_map_2025.png', dpi=300, bbox_inches='tight')
```

**Multi-Horizon Comparison:**

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

horizons = ['5yr', '10yr', '15yr', '20yr']
years = [2025, 2030, 2035, 2040]

for ax, horizon, year in zip(axes.flat, horizons, years):
    band_idx = horizons.index(horizon) * 3 + 2  # Central prediction
    pred = rasterio.open('outputs/predictions.tif').read(band_idx)
    
    im = ax.imshow(pred, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(f'Predicted {year}', fontsize=12)
    ax.axis('off')

plt.suptitle('Human Modification Predictions (2025-2040)', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('outputs/multi_horizon_predictions.png', dpi=300)
```

**Uncertainty Visualization:**

```python
# Show prediction with uncertainty bounds
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Band indices for 20-year horizon
lower_idx = 10  # Band 10
central_idx = 11  # Band 11
upper_idx = 12  # Band 12

preds = rasterio.open('outputs/predictions.tif')
lower = preds.read(lower_idx)
central = preds.read(central_idx)
upper = preds.read(upper_idx)

axes[0].imshow(lower, cmap=cmap, vmin=0, vmax=1)
axes[0].set_title('Lower Bound (2.5%)', fontsize=12)
axes[0].axis('off')

axes[1].imshow(central, cmap=cmap, vmin=0, vmax=1)
axes[1].set_title('Central Prediction', fontsize=12)
axes[1].axis('off')

im = axes[2].imshow(upper, cmap=cmap, vmin=0, vmax=1)
axes[2].set_title('Upper Bound (97.5%)', fontsize=12)
axes[2].axis('off')

fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
plt.suptitle('2040 Predictions with Uncertainty Bounds', fontsize=14)
plt.savefig('outputs/uncertainty_visualization.png', dpi=300)
```

#### Performance Considerations

**Processing Time Estimates:**

| Region Size | Tiles (512×512) | GPU Time | CPU Time |
|-------------|-----------------|----------|----------|
| 100×100 km | ~100 | ~2 min | ~10 min |
| 500×500 km | ~1,000 | ~20 min | ~2 hours |
| 1000×1000 km | ~4,000 | ~80 min | ~8 hours |

**Optimization Strategies:**

1. **Batch Processing:**
   ```python
   # Process multiple tiles simultaneously
   batch_size = 4  # Process 4 tiles at once
   ```

2. **Mixed Precision:**
   ```python
   with torch.cuda.amp.autocast():
       predictions = model(inputs)
   ```
   - 2× speedup with minimal accuracy loss

3. **GPU Selection:**
   ```python
   # Use fastest available GPU
   device = 'cuda:0'  # Or 'cuda:1', etc.
   ```

4. **Incremental Saving:**
   ```python
   # Save predictions every N tiles (recovery from crashes)
   if tile_idx % 100 == 0:
       save_checkpoint(predictions_so_far, f'temp/checkpoint_{tile_idx}.pkl')
   ```

#### Integration with GIS

**Loading in QGIS:**

1. Add Raster Layer → Navigate to predictions.tif
2. Each band appears as separate layer
3. Style using appropriate colormaps
4. Create composite visualizations

**Loading in ArcGIS:**

1. Add Data → Raster Dataset
2. Use Composite Bands tool for multi-horizon view
3. Apply symbology from provided .lyr file

**Loading in Python (GeoPandas/Rasterio):**

```python
import rasterio
import geopandas as gpd

# Load predictions
with rasterio.open('outputs/predictions.tif') as src:
    pred_central_2025 = src.read(2)  # Band 2
    transform = src.transform
    crs = src.crs

# Overlay with vector boundaries
region = gpd.read_file('config/boundaries.shp')

# Spatial analysis...
```

---

**Document Complete!**

This comprehensive documentation covers the entire pipeline from data input through model architecture, training, evaluation, to operational predictions. Each section provides both intuitive explanations for conceptual understanding and technical details for implementation.

