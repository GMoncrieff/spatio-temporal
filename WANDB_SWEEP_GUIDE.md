# W&B Sweep Guide

This guide shows you how to run hyperparameter sweeps using Weights & Biases without modifying `train_lightning.py`.

## Quick Start

### 1. Initialize the Sweep

Choose one of the sweep configurations:

**For full hyperparameter search:**
```bash
wandb sweep config/sweep_config.yaml
```

**For quick testing (10 runs, 10 epochs each):**
```bash
wandb sweep config/sweep_config_quick.yaml
```

This will output a sweep ID like: `your-entity/your-project/sweep-id`

### 2. Run Sweep Agents

**Single agent (one machine):**
```bash
wandb agent your-entity/your-project/sweep-id
```

**Multiple parallel agents (on different machines/terminals):**
```bash
# Terminal 1
wandb agent your-entity/your-project/sweep-id

# Terminal 2  
wandb agent your-entity/your-project/sweep-id

# Terminal 3
wandb agent your-entity/your-project/sweep-id
```

The agents will automatically run `train_lightning.py` with different hyperparameter combinations!

### 3. Monitor Progress

Go to your W&B dashboard to see:
- Live training curves for all runs
- Parallel coordinates plot showing parameter correlations
- Hyperparameter importance analysis
- Best run so far

---

## Sweep Configuration Files

### `sweep_config.yaml` (Full Search)

**Parameters being swept:**
- Model: `hidden_dim`, `num_layers`, `kernel_size`
- LocationEncoder: `locenc_out_channels`, `locenc_legendre_polys`
- Loss weights: `ssim_weight`, `laplacian_weight`, `histogram_weight`, `histogram_warmup_epochs`
- Training: `batch_size`, `accumulate_grad_batches`

**Search method:** Bayesian optimization (smart search)
**Stopping:** Early termination of poorly performing runs
**Expected time:** ~2-4 hours per run with default settings

### `sweep_config_quick.yaml` (Quick Test)

**Parameters being swept:**
- Model: `hidden_dim`, `num_layers`
- Loss weights: `ssim_weight`, `histogram_weight`

**Search method:** Random search
**Runs:** 10 trials maximum
**Training:** 10 epochs, 50 train chips, 20 val chips
**Expected time:** ~15-20 minutes per run

---

## Customizing Sweep Configurations

### Search Methods

**Random Search** (`method: random`):
- Good for: Quick exploration, many parameters
- Pros: Simple, parallelizes well
- Cons: May miss optimal combinations

**Grid Search** (`method: grid`):
- Good for: Few parameters, exhaustive search
- Pros: Covers all combinations
- Cons: Exponential growth with parameters

**Bayesian Optimization** (`method: bayes`):
- Good for: Expensive evaluations, complex spaces
- Pros: Smart, finds good solutions faster
- Cons: Sequential (less parallel)

### Parameter Types

**Discrete values:**
```yaml
hidden_dim:
  values: [32, 64, 128]
```

**Continuous range:**
```yaml
ssim_weight:
  min: 0.5
  max: 5.0
  distribution: uniform  # or log_uniform for log scale
```

**Fixed (not swept):**
```yaml
max_epochs:
  value: 50
```

### Early Termination

**Hyperband** (recommended):
```yaml
early_terminate:
  type: hyperband
  min_iter: 10  # Minimum epochs before termination
  eta: 2        # Reduction factor
  s: 2          # Number of brackets
```

**Stop poorly performing runs early to save compute!**

---

## Example: Custom Sweep for Loss Weights Only

Create `config/sweep_loss_weights.yaml`:

```yaml
program: scripts/train_lightning.py
method: bayes
metric:
  name: val_full/total_loss_avg
  goal: minimize

parameters:
  # Sweep only loss weights
  ssim_weight:
    min: 0.5
    max: 5.0
  laplacian_weight:
    min: 0.2
    max: 2.0
  histogram_weight:
    min: 0.2
    max: 1.5
  
  # Fix everything else
  hidden_dim:
    value: 64
  num_layers:
    value: 2
  max_epochs:
    value: 30
  train_chips:
    value: 200
  val_chips:
    value: 40
  predict_after_training:
    value: false
```

Then run:
```bash
wandb sweep config/sweep_loss_weights.yaml
wandb agent your-sweep-id
```

---

## Tips & Best Practices

### 1. Start Small
- Use `sweep_config_quick.yaml` first to test your setup
- Verify W&B logging works correctly
- Check that all metrics are being logged

### 2. Parallelization
- Run multiple agents on different GPUs/machines
- Each agent picks up a new run from the queue
- All results aggregate in the same W&B project

### 3. Resume After Interruption
- If agents crash, just restart with the same sweep ID
- Completed runs are saved, sweep continues from where it left off

### 4. Stop Sweep When Done
```bash
wandb sweep --stop your-sweep-id
```

### 5. Monitor Resource Usage
- Check GPU memory with different `batch_size` and `hidden_dim`
- Use `--accumulate_grad_batches` to reduce memory if needed

### 6. Focus Your Search
- Don't sweep everything at once
- Start with model architecture, then loss weights, then training hyperparams
- Use insights from previous sweeps to narrow the search space

---

## Analyzing Results

### In W&B Dashboard

1. **Parallel Coordinates Plot:** 
   - Shows relationship between hyperparameters and metrics
   - Drag to filter to successful runs
   - Identifies important parameters

2. **Parameter Importance:**
   - Automatically computed correlation with metric
   - Focus future sweeps on important parameters

3. **Compare Runs:**
   - Select best runs and compare side-by-side
   - Look at training curves, validation metrics
   - Download best model checkpoint

### Download Best Run Checkpoint

Once you find the best run:
```python
import wandb
api = wandb.Api()
run = api.run("your-entity/your-project/run-id")
run.file("best_model.ckpt").download()
```

---

## Troubleshooting

**Issue:** "No wandb logging detected"
- **Fix:** Make sure `--disable_wandb` is NOT in sweep config

**Issue:** "Out of memory errors"
- **Fix:** Add constraints to sweep config:
  ```yaml
  batch_size:
    values: [4, 8]  # Don't use 16
  ```

**Issue:** "Sweep not starting"
- **Fix:** Check you're in the right W&B project
- **Fix:** Verify sweep ID is correct

**Issue:** "All runs failing"
- **Fix:** Test single run manually first:
  ```bash
  python scripts/train_lightning.py --max_epochs 2 --train_chips 10
  ```

---

## Advanced: Sweeping with Custom Metrics

To optimize for a different metric (e.g., MAE instead of total loss):

```yaml
metric:
  name: val_full/mae_5yr  # or any logged metric
  goal: minimize
```

Available metrics:
- `val_full/total_loss_avg` (default)
- `val_full/mae_5yr`, `val_full/mae_10yr`, etc.
- `val_full/mse_5yr`, `val_full/ssim_loss_5yr`, etc.
- Any metric logged to W&B during training

---

## Summary Commands

```bash
# 1. Initialize sweep
wandb sweep config/sweep_config.yaml

# 2. Run agent (repeat on multiple machines)
wandb agent your-entity/your-project/sweep-id

# 3. Monitor in browser
# Go to https://wandb.ai/your-entity/your-project

# 4. Stop sweep when satisfied
wandb sweep --stop your-entity/your-project/sweep-id
```

**No code changes needed - it's all command line!** 🎉
