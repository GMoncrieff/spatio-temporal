# Checkpoint Usage Guide

## ✅ Checkpoint Loading Now Supported!

The training script now supports loading checkpoints from **W&B artifacts** or **local files**.

## Usage

### Option 1: Load from W&B Artifact (Recommended)

```bash
python scripts/train_lightning.py \
  --checkpoint model-txn1v2kp:v0 \
  --max_epochs 0 \
  --predict_after_training true
```

**Short form** (automatically adds entity/project):
- `model-txn1v2kp:v0` → `glennwithtwons/spatio-temporal-convlstm/model-txn1v2kp:v0`

**Full form**:
```bash
--checkpoint glennwithtwons/spatio-temporal-convlstm/model-txn1v2kp:v0
```

### Option 2: Load from Local File

```bash
python scripts/train_lightning.py \
  --checkpoint path/to/model.ckpt \
  --max_epochs 0 \
  --predict_after_training true
```

### Option 3: Train from Scratch (No Checkpoint)

```bash
python scripts/train_lightning.py \
  --max_epochs 50
```

## How It Works

1. **Check local file**: If `--checkpoint` is a valid file path, use it directly
2. **Download from W&B**: If not a local file, try to download as W&B artifact
3. **Create new model**: If no checkpoint provided, create new model with args

## Examples

### Run Predictions with W&B Checkpoint

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_lightning.py \
  --checkpoint model-txn1v2kp:v0 \
  --max_epochs 0 \
  --predict_after_training true \
  --predict_region config/region_to_predict.geojson \
  --predict_stride 64 \
  --disable_wandb
```

### Resume Training from Checkpoint

```bash
python scripts/train_lightning.py \
  --checkpoint model-txn1v2kp:v0 \
  --max_epochs 10
```

### Fine-tune on New Data

```bash
python scripts/train_lightning.py \
  --checkpoint model-txn1v2kp:v0 \
  --max_epochs 20 \
  --train_chips 1000 \
  --val_chips 200
```

## Features

✅ **Automatic W&B download**: Checkpoints downloaded to `artifacts/` directory  
✅ **Local file support**: Use any `.ckpt` file  
✅ **Smart detection**: Automatically determines if path is local or W&B artifact  
✅ **Error handling**: Clear messages if checkpoint not found  
✅ **Configuration display**: Shows loaded model hyperparameters  

## Output

When loading a checkpoint, you'll see:

```
Attempting to download W&B artifact: model-txn1v2kp:v0
✓ Downloaded checkpoint to: artifacts/model-txn1v2kp:v0/model.ckpt

======================================================================
Loading model from checkpoint: artifacts/model-txn1v2kp:v0/model.ckpt
======================================================================
✓ Checkpoint loaded successfully!

Model configuration from checkpoint:
  hidden_dim: 64
  num_layers: 4
  kernel_size: 3
  num_static_channels: 7
  num_dynamic_channels: 11
  use_location_encoder: True
  locenc_out_channels: 8
======================================================================
```

## Troubleshooting

### "Could not load checkpoint"
- Check artifact name is correct
- Verify you're logged into W&B: `wandb login`
- Try full artifact path: `glennwithtwons/spatio-temporal-convlstm/model-txn1v2kp:v0`

### "No checkpoint available for prediction"
- Provide `--checkpoint` argument
- Or train a model first with `--max_epochs > 0`

### "No .ckpt file found in artifact"
- The artifact may not contain a checkpoint file
- Check artifact contents in W&B dashboard

## Implementation Details

The checkpoint loading is handled by `load_checkpoint_path()` function which:

1. Checks if argument is a local file path
2. If not, initializes W&B and downloads artifact
3. Finds `.ckpt` file in downloaded artifact directory
4. Returns path to checkpoint file

The function is called before model creation, so the checkpoint is loaded instead of creating a new model.

## W&B Artifact Format

Expected artifact structure:
```
artifacts/
└── model-txn1v2kp:v0/
    ├── model.ckpt          # The checkpoint file
    └── wandb_manifest.json # W&B metadata
```

## Command Line Reference

```bash
--checkpoint STR    Path to checkpoint or W&B artifact
                    Examples:
                      model-txn1v2kp:v0
                      path/to/model.ckpt
                      glennwithtwons/spatio-temporal-convlstm/model-txn1v2kp:v0
```

---

**Status**: ✅ Fully implemented and tested  
**Date**: 2025-10-21
