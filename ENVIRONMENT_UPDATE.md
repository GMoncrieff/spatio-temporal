# Environment Configuration Update

## Summary

Updated `environment.yml`, `requirements.txt`, and README setup instructions to reflect the minimal, working package set.

## Changes Made

### 1. `environment.yml`
**Before**: 361 lines with hundreds of pinned dependencies  
**After**: 26 lines with only essential packages

**New configuration**:
```yaml
name: hmforecast
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pytorch-lightning
  - torchmetrics
  - rasterio
  - shapely
  - pyproj
  - scipy
  - matplotlib
  - seaborn
  - pandas
  - numpy
  - scikit-learn
  - hydra-core
  - omegaconf
  - pip
  - pip:
    - torchgeo
    - einops
    - wandb
    - light-the-torch
```

### 2. `requirements.txt`
**Before**: 48 lines with many unused packages (xgboost, mlflow, tensorboard, etc.)  
**After**: 37 lines with only required packages

**Key packages**:
- **Deep Learning**: pytorch-lightning, torchmetrics
- **Geospatial**: torchgeo, rasterio, shapely, pyproj
- **Scientific**: numpy, scipy, pandas, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Config**: hydra-core, omegaconf
- **Utilities**: einops, wandb, light-the-torch

### 3. README.md Setup Section

**Added three installation options**:

#### Option A: Manual conda install (Recommended)
```bash
conda create -n hmforecast python=3.12
conda activate hmforecast
conda install pytorch-lightning torchmetrics rasterio shapely pyproj scipy matplotlib seaborn pandas numpy scikit-learn -c conda-forge
conda install hydra-core omegaconf -c conda-forge
pip install torchgeo einops wandb light-the-torch
ltt install torch torchvision
```

#### Option B: Using environment.yml
```bash
conda env create -f environment.yml
conda activate hmforecast
ltt install torch torchvision
```

#### Option C: Using pip only
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ltt install torch torchvision
```

## Key Features

### 1. PyTorch Installation via light-the-torch
- **Tool**: `light-the-torch` (ltt)
- **Purpose**: Automatically detects hardware and installs appropriate PyTorch
- **Supports**: CPU, CUDA (NVIDIA), MPS (Apple Silicon)
- **Usage**: `ltt install torch torchvision`

### 2. Minimal Dependencies
Only packages actually used in the codebase:
- ✅ pytorch-lightning (training framework)
- ✅ torchmetrics (metrics)
- ✅ torchgeo (geospatial data loading)
- ✅ rasterio (GeoTIFF I/O)
- ✅ shapely, pyproj (geospatial operations)
- ✅ numpy, scipy, pandas (scientific computing)
- ✅ matplotlib, seaborn (visualization)
- ✅ scikit-learn (ML utilities)
- ✅ hydra-core, omegaconf (configuration)
- ✅ einops (tensor operations)
- ✅ wandb (experiment tracking)

### 3. Removed Unused Packages
- ❌ xgboost (not used)
- ❌ mlflow (using wandb instead)
- ❌ tensorboard (using wandb instead)
- ❌ plotly (not used)
- ❌ geopandas (not needed, using rasterio)
- ❌ folium (not used)
- ❌ h5py, netcdf4, xarray (not using zarr/xbatcher)
- ❌ jupyter, ipykernel (development tools, install separately if needed)
- ❌ pytest, black, flake8 (development tools, install separately if needed)

## Installation Workflow

### Recommended: Option A (Manual conda)

1. **Create environment**:
   ```bash
   conda create -n hmforecast python=3.12
   conda activate hmforecast
   ```

2. **Install conda packages** (one command):
   ```bash
   conda install pytorch-lightning torchmetrics rasterio shapely pyproj scipy matplotlib seaborn pandas numpy scikit-learn hydra-core omegaconf -c conda-forge
   ```

3. **Install pip packages**:
   ```bash
   pip install torchgeo einops wandb light-the-torch
   ```

4. **Install PyTorch** (auto-detects hardware):
   ```bash
   ltt install torch torchvision
   ```

### Verification

After installation, verify:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytorch_lightning; print(f'Lightning: {pytorch_lightning.__version__}')"
python -c "import torchgeo; print(f'TorchGeo: {torchgeo.__version__}')"
python -c "import rasterio; print(f'Rasterio: {rasterio.__version__}')"
```

Expected output (versions may vary):
```
PyTorch: 2.x.x
Lightning: 2.x.x
TorchGeo: 0.x.x
Rasterio: 1.x.x
```

## Benefits

1. **Faster installation**: Fewer packages to download and install
2. **Cleaner environment**: Only what's needed
3. **Hardware optimization**: `light-the-torch` ensures correct PyTorch build
4. **Reproducibility**: Minimal dependencies reduce version conflicts
5. **Maintenance**: Easier to update and troubleshoot

## Hardware Support

### Apple Silicon (M1/M2/M3)
```bash
ltt install torch torchvision
# Installs MPS-enabled PyTorch automatically
```

### NVIDIA GPU
```bash
ltt install torch torchvision
# Installs CUDA-enabled PyTorch automatically
```

### CPU Only
```bash
ltt install torch torchvision
# Installs CPU-only PyTorch automatically
```

## Migration Notes

If you have an existing environment:

1. **Export your current packages** (optional backup):
   ```bash
   conda env export > old_environment.yml
   ```

2. **Remove old environment**:
   ```bash
   conda deactivate
   conda env remove -n spatio-temporal-dl  # or your old env name
   ```

3. **Create new environment** using Option A, B, or C above

4. **Test your code**:
   ```bash
   python scripts/train_lightning.py --fast_dev_run
   ```

## Files Modified

- ✅ `environment.yml`: Simplified from 361 to 26 lines
- ✅ `requirements.txt`: Simplified from 48 to 37 lines
- ✅ `README.md`: Updated setup section with 3 installation options

## Status

✅ **Complete** - Environment configuration updated and tested

---

**Date**: 2025-10-24  
**Environment Name**: `hmforecast`  
**Python Version**: 3.12  
**PyTorch Installation**: via `light-the-torch` (hardware auto-detection)
