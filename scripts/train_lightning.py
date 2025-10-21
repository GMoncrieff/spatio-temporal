import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import os
import argparse
import json
import random

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from src.models.lightning_module import SpatioTemporalLightningModule
from torchgeo_dataloader import get_dataloader, hm_files, component_files, static_files, years

# Geospatial imports for inference
import rasterio
from rasterio import windows as rio_windows
from rasterio import features as rio_features
from rasterio.transform import rowcol, Affine
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 1 train/val batch for a quick smoke test")
    parser.add_argument("--max_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--train_chips", type=int, default=200, help="Chips per epoch for training")
    parser.add_argument("--val_chips", type=int, default=40, help="Chips per epoch for validation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for train/val")
    parser.add_argument("--train_mode", type=str, default="random", choices=["random", "grid"], help="Sampling mode for training")
    parser.add_argument("--val_mode", type=str, default="grid", choices=["random", "grid"], help="Sampling mode for validation")
    parser.add_argument("--stride", type=int, default=128, help="Stride for grid sampling (pixels)")
    parser.add_argument(
        "--include_components",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?',
        const=True,
        default=True,
        help="Whether to include component covariates (AG, BU, etc.) in dynamic inputs",
    )
    parser.add_argument(
        "--static_channels",
        type=int,
        default=None,
        help="Limit number of static channels (e.g., 1 to use only elevation)",
    )
    # Model complexity
    parser.add_argument("--hidden_dim", type=int, default=64, help="ConvLSTM hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of ConvLSTM layers")
    parser.add_argument("--kernel_size", type=int, default=3, help="Conv kernel size for ConvLSTM")
    # Inference flags
    parser.add_argument(
        "--predict_after_training",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?',
        const=True,
        default=True,
        help="Run large-area prediction and write GeoTIFF after training (default: True)",
    )
    parser.add_argument(
        "--predict_region",
        type=str,
        default=None,
        help="Path to GeoJSON file for prediction region. If not provided, loaded from config/config.yaml",
    )
    parser.add_argument(
        "--predict_stride",
        type=int,
        default=64,
        help="Stride (pixels) between prediction tiles for overlap blending",
    )
    parser.add_argument(
        "--predict_batch_size",
        type=int,
        default=16,
        help="Number of tiles to process in parallel on GPU during prediction (default: 16)",
    )
    parser.add_argument(
        "--use_location_encoder",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?',
        const=True,
        default=True,
        help="Whether to append per-pixel LocationEncoder features to static inputs (default: True)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers (default: 0 for single-threaded)",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients over (default: 1, no accumulation)",
    )
    # Loss weight arguments
    parser.add_argument(
        "--ssim_weight",
        type=float,
        default=2.0,
        help="Weight for SSIM loss (default: 2.0)",
    )
    parser.add_argument(
        "--laplacian_weight",
        type=float,
        default=1.0,
        help="Weight for Laplacian pyramid loss (default: 1.0)",
    )
    parser.add_argument(
        "--histogram_weight",
        type=float,
        default=0.67,
        help="Weight for histogram loss on pixel-level change distributions (default: 0.67)",
    )
    parser.add_argument(
        "--histogram_lambda_w2",
        type=float,
        default=0.1,
        help="Weight for Wasserstein-2 term within histogram loss (default: 0.1)",
    )
    parser.add_argument(
        "--histogram_warmup_epochs",
        type=int,
        default=20,
        help="Number of epochs before histogram loss is applied (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    
    # Set seeds for reproducibility (without strict determinism)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    pl.seed_everything(args.seed, workers=True)
    
    # Split mask file
    split_mask_file = "data/raw/hm_global/split_mask_1000.tif"
    if not os.path.exists(split_mask_file):
        print(f"WARNING: Split mask not found: {split_mask_file}")
        print("Training without train/val/test separation. Run scripts/create_validity_mask.py to create splits.")
        split_mask_file = None
    
    # Data
    train_loader = get_dataloader(
        batch_size=args.batch_size,
        chip_size=128,
        timesteps=3,
        chips_per_epoch=args.train_chips,
        mode=args.train_mode,
        stride=args.stride,
        include_components=args.include_components,
        static_channels=args.static_channels,
        use_temporal_sampling=True,  # Enable temporal sampling for training
        end_year_options=(2000, 2005, 2010, 2015),
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers > 0 else False,
        persistent_workers=True if args.num_workers > 0 else False,
        split_mask_file=split_mask_file,
        split_value=1,  # Train split
    )
    # Validation uses fixed years (1990, 1995, 2000 -> 2005-2020) for consistent metrics
    val_loader = get_dataloader(
        batch_size=args.batch_size,
        chip_size=128,
        timesteps=3,
        chips_per_epoch=args.val_chips,
        mode=args.val_mode,
        stride=args.stride,
        include_components=args.include_components,
        static_channels=args.static_channels,
        use_temporal_sampling=False,  # Fixed years for validation (Option A)
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers > 0 else False,
        persistent_workers=True if args.num_workers > 0 else False,
        split_mask_file=split_mask_file,
        split_value=2,  # Validation split
    )

    # Model
    num_static_channels = getattr(train_loader.dataset, 'C_static', 1)
    num_dynamic_channels = getattr(train_loader.dataset, 'C_dyn', 1)
    model = SpatioTemporalLightningModule(
        hidden_dim=args.hidden_dim,
        lr=1e-3,
        num_static_channels=num_static_channels,
        num_dynamic_channels=num_dynamic_channels,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        use_location_encoder=args.use_location_encoder,
        ssim_weight=args.ssim_weight,
        laplacian_weight=args.laplacian_weight,
        histogram_weight=args.histogram_weight,
        histogram_lambda_w2=args.histogram_lambda_w2,
        histogram_warmup_epochs=args.histogram_warmup_epochs,
    )
    
    # Compute histogram bin weights from training data (per horizon)
    if args.histogram_weight > 0 and hasattr(model, 'histogram_loss_fn'):
        print("\nComputing histogram bin weights for each horizon from 10 training batches...")
        from src.models.histogram_loss import compute_histogram
        
        horizon_names = ['5yr', '10yr', '15yr', '20yr']
        horizon_keys = ['target_5yr', 'target_10yr', 'target_15yr', 'target_20yr']
        all_horizon_counts = {h: [] for h in horizon_names}
        
        device = next(model.parameters()).device
        num_batches_to_sample = min(10, len(train_loader))
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_batches_to_sample:
                break
            
            input_dynamic = batch['input_dynamic'].to(device)
            last_input = input_dynamic[:, -1, 0]  # [B, H, W]
            
            # Compute histograms for each horizon
            for h_name, h_key in zip(horizon_names, horizon_keys):
                target_h = batch[h_key].to(device)
                
                # Compute mask for valid pixels
                target_valid = torch.isfinite(target_h)
                last_input_valid = torch.isfinite(last_input)
                mask = target_valid & last_input_valid
                
                # Compute deltas
                delta_true = target_h - last_input
                
                # Compute histogram
                counts, _ = compute_histogram(delta_true, model.histogram_bins, mask=mask)
                all_horizon_counts[h_name].append(counts.cpu())
        
        # Compute weights for each horizon
        num_bins = len(model.histogram_bins) - 1
        all_bin_weights = []
        
        print("\n" + "="*70)
        print("HISTOGRAM BIN WEIGHTS PER HORIZON (Rarity-Weighted)")
        print("="*70)
        print(f"Bin edges: {model.histogram_bins.tolist()}\n")
        
        for h_idx, h_name in enumerate(horizon_names):
            # Aggregate counts for this horizon
            horizon_counts = torch.cat(all_horizon_counts[h_name], dim=0)
            total_counts = horizon_counts.sum(dim=0)
            
            # Compute inverse frequency weights
            smoothing = 1e-3
            bin_weights = 1.0 / (total_counts + smoothing)
            bin_weights = bin_weights * num_bins / bin_weights.sum()
            all_bin_weights.append(bin_weights)
            
            # Print bin information for this horizon
            print(f"--- {h_name} Horizon ---")
            print(f"Bin | Count  | Proportion | Weight")
            print("-" * 50)
            total_pixels = total_counts.sum().item()
            for i in range(num_bins):
                count = total_counts[i].item()
                proportion = count / total_pixels
                weight = bin_weights[i].item()
                left_edge = model.histogram_bins[i].item()
                right_edge = model.histogram_bins[i+1].item()
                print(f" {i}  | {count:6.0f} | {proportion:9.4f}  | {weight:6.3f}  [{left_edge:+.3f}, {right_edge:+.3f})")
            print()
        
        # Stack all weights and set in model: [num_horizons, num_bins]
        all_bin_weights = torch.stack(all_bin_weights, dim=0).to(device)
        model.histogram_loss_fn.set_bin_weights(all_bin_weights)
        model.histogram_bins_initialized = True
        print("="*70 + "\n")
    
    # Print loss weights at start of training
    print("="*60)
    print("LOSS WEIGHTS")
    print("="*60)
    print(f"MSE weight:        1.0 (fixed)")
    print(f"SSIM weight:       {args.ssim_weight}")
    print(f"Laplacian weight:  {args.laplacian_weight}")
    print(f"Histogram weight:  {args.histogram_weight} (warmup: {args.histogram_warmup_epochs} epochs)")
    print("="*60 + "\n")
    # Set normalization stats for physical-scale MAE logging
    if hasattr(train_loader, 'dataset'):
        ds = train_loader.dataset
        if hasattr(ds, 'hm_mean') and hasattr(ds, 'hm_std'):
            model.hm_mean = ds.hm_mean
            model.hm_std = ds.hm_std

    # Callbacks
    checkpoint_cb = ModelCheckpoint(monitor='val_total_loss', save_top_k=1, mode='min')
    # No early stopping

    # Wandb logger (optional)
    use_wandb = not args.disable_wandb
    wandb_logger = False if not use_wandb else WandbLogger(project='spatio-temporal-convlstm', log_model=True)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb],
        accelerator='auto',
        default_root_dir=os.path.join(os.getcwd(), 'models', 'checkpoints'),
        logger=wandb_logger,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # --- Log predictions from best checkpoint to wandb (rank 0 only) ---
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    # Only the global zero process should log to W&B
    best_ckpt = checkpoint_cb.best_model_path
    should_log_wandb = (
        best_ckpt
        and use_wandb
        and isinstance(trainer.logger, WandbLogger)
        and getattr(trainer, "is_global_zero", True)
    )
    if should_log_wandb:
        import wandb  # import here to avoid touching wandb on non-logging ranks
        experiment = trainer.logger.experiment  # a wandb.Run
        best_model = SpatioTemporalLightningModule.load_from_checkpoint(best_ckpt)
        best_model.eval()
        # Move model and data to same device
        device = next(best_model.parameters()).device
        # Ensure normalization stats are present for logging
        if hasattr(train_loader, 'dataset'):
            ds_train = train_loader.dataset
            if hasattr(ds_train, 'hm_mean') and hasattr(ds_train, 'hm_std'):
                best_model.hm_mean = ds_train.hm_mean
                best_model.hm_std = ds_train.hm_std
        # Inference on entire validation set
        print("\nRunning inference on entire validation set for plotting...")
        all_batches_data = []
        num_batches_processed = 0
        
        for batch_idx, batch in enumerate(val_loader):
            # Use 20yr target for validation metrics (multi-horizon)
            target = batch.get('target_20yr', batch.get('target'))
            # Check for at least one valid (non-NaN) pixel in any sample
            if torch.any(~torch.isnan(target)).item():
                input_dynamic = batch['input_dynamic'].to(device)
                input_static = batch['input_static'].to(device)
                target = target.to(device)
                best_model.eval()
                with torch.no_grad():
                    if input_dynamic.dim() == 4:
                        input_dynamic = input_dynamic.unsqueeze(2)
                
                    # Apply same NaN handling as training/validation
                    # Compute validity mask from RAW inputs
                    target_unsqueezed = target.unsqueeze(1)  # Add channel dimension for consistency
                    target_valid = torch.isfinite(target_unsqueezed)
                    dynamic_valid = torch.isfinite(input_dynamic).all(dim=(1, 2), keepdim=True)
                    static_valid = torch.isfinite(input_static).all(dim=1, keepdim=True).unsqueeze(1)
                    input_mask = target_valid & dynamic_valid.squeeze(2) & static_valid.squeeze(2)
                
                    # Replace NaNs in inputs with 0 for model forward
                    input_dynamic_clean = torch.nan_to_num(input_dynamic, nan=0.0)
                    input_static_clean = torch.nan_to_num(input_static, nan=0.0)
                    # Lon/lat from dataset if present
                    lonlat = batch.get('lonlat', None)
                    if lonlat is not None:
                        lonlat = lonlat.to(device)
                    # Get predictions from model (learnable location encoder)
                    preds_all = best_model(input_dynamic_clean, input_static_clean, lonlat=lonlat)  # [B, 4, H, W]
                    
                    # Store all 4 horizons
                    preds_5yr = preds_all[:, 0:1, :, :].clone()  # [B, 1, H, W]
                    preds_10yr = preds_all[:, 1:2, :, :].clone()
                    preds_15yr = preds_all[:, 2:3, :, :].clone()
                    preds_20yr = preds_all[:, 3:4, :, :].clone()
                    
                    # Set predictions to NaN where any input was NaN
                    preds_5yr[~input_mask] = float('nan')
                    preds_10yr[~input_mask] = float('nan')
                    preds_15yr[~input_mask] = float('nan')
                    preds_20yr[~input_mask] = float('nan')
                    
                    # Squeeze for storage
                    preds_5yr = preds_5yr.squeeze(1)  # [B, H, W]
                    preds_10yr = preds_10yr.squeeze(1)
                    preds_15yr = preds_15yr.squeeze(1)
                    preds_20yr = preds_20yr.squeeze(1)
                
                # Store batch data for later processing (all horizons)
                all_batches_data.append({
                    'input_dynamic': input_dynamic.cpu(),
                    'input_static': input_static.cpu(),
                    'target_5yr': batch.get('target_5yr', target).cpu(),
                    'target_10yr': batch.get('target_10yr', target).cpu(),
                    'target_15yr': batch.get('target_15yr', target).cpu(),
                    'target_20yr': batch.get('target_20yr', target).cpu(),
                    'preds_5yr': preds_5yr.cpu(),
                    'preds_10yr': preds_10yr.cpu(),
                    'preds_15yr': preds_15yr.cpu(),
                    'preds_20yr': preds_20yr.cpu(),
                    'input_mask': input_mask.cpu(),
                    'lonlat': batch.get('lonlat', None)
                })
                num_batches_processed += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Processed {batch_idx + 1} batches...")
        
        if num_batches_processed == 0:
            print("WARNING: No valid (non-NaN) target pixels found in any validation batch for image logging.")
            sys.exit(0)
        
        print(f"✓ Processed {num_batches_processed} validation batches")
        
        # ===== Calculate validation metrics over full validation set =====
        print("\nCalculating validation metrics over full validation set...")
        from src.models.losses import LaplacianPyramidLoss
        from torchmetrics.functional import structural_similarity_index_measure as ssim
        import torch.nn.functional as F
        
        # Initialize loss functions
        lap_loss_fn = LaplacianPyramidLoss(levels=3, kernel_size=5, sigma=1.0, include_lowpass=True)
        
        # Accumulators for metrics per horizon
        horizon_names = ['5yr', '10yr', '15yr', '20yr']
        horizon_keys = ['target_5yr', 'target_10yr', 'target_15yr', 'target_20yr']
        pred_keys = ['preds_5yr', 'preds_10yr', 'preds_15yr', 'preds_20yr']
        
        horizon_metrics = {h: {
            'mse': 0.0, 'mae': 0.0, 'ssim': 0.0, 'lap': 0.0, 'hist': 0.0, 'pixels': 0,
            'mae_no_change': 0.0,  # Baseline: assume no change from last input
            'mae_linear': 0.0       # Baseline: linear extrapolation from recent trend
        } for h in horizon_names}
        
        for batch_data in all_batches_data:
            input_dynamic = batch_data['input_dynamic'].to(device)
            input_mask = batch_data['input_mask'].to(device)
            last_input = input_dynamic[:, -1, 0]  # [B, H, W]
            
            # Process each horizon
            for h_idx, (h_name, target_key, pred_key) in enumerate(zip(horizon_names, horizon_keys, pred_keys)):
                target = batch_data[target_key].to(device)
                preds = batch_data[pred_key].to(device)
                
                # Compute mask for valid pixels
                mask = input_mask.squeeze(1) & torch.isfinite(last_input) & torch.isfinite(target)
                
                if mask.sum() == 0:
                    continue
                
                # Delta predictions and targets
                delta_pred = preds - last_input
                delta_true = target - last_input
                
                valid_delta_pred = delta_pred[mask]
                valid_delta_true = delta_true[mask]
                
                # MSE on deltas
                mse = F.mse_loss(valid_delta_pred, valid_delta_true, reduction='sum')
                horizon_metrics[h_name]['mse'] += mse.item()
                
                # MAE on absolute predictions (Conv-CNN model)
                mae = F.l1_loss(preds[mask], target[mask], reduction='sum')
                horizon_metrics[h_name]['mae'] += mae.item()
                
                # Baseline 1: "No Change" - assume future = most recent past
                pred_no_change = last_input  # Simply use last input as prediction
                mae_no_change = F.l1_loss(pred_no_change[mask], target[mask], reduction='sum')
                horizon_metrics[h_name]['mae_no_change'] += mae_no_change.item()
                
                # Baseline 2: "Linear" - extrapolate from recent trend
                # Calculate trend between last two timesteps
                if input_dynamic.shape[1] >= 2:
                    second_last_input = input_dynamic[:, -2, 0]  # [B, H, W]
                    trend = last_input - second_last_input
                    # Multiply trend by horizon multiplier (1x for 5yr, 2x for 10yr, etc.)
                    horizon_multiplier = h_idx + 1  # 1, 2, 3, 4 for 5yr, 10yr, 15yr, 20yr
                    pred_linear = last_input + (trend * horizon_multiplier)
                    mae_linear = F.l1_loss(pred_linear[mask], target[mask], reduction='sum')
                    horizon_metrics[h_name]['mae_linear'] += mae_linear.item()
                
                # SSIM (requires [B, C, H, W])
                preds_sanitized = preds.unsqueeze(1).clone()
                target_sanitized = target.unsqueeze(1).clone()
                mask_4d = mask.unsqueeze(1)
                preds_sanitized[~mask_4d] = 0.0
                target_sanitized[~mask_4d] = 0.0
                ssim_val = ssim(preds_sanitized, target_sanitized, data_range=1.0)
                ssim_loss = 1.0 - ssim_val
                horizon_metrics[h_name]['ssim'] += ssim_loss.item() * mask.sum().item()
                
                # Laplacian loss
                lap_loss = lap_loss_fn(preds_sanitized, target_sanitized, mask=mask_4d)
                horizon_metrics[h_name]['lap'] += lap_loss.item() * mask.sum().item()
                
                # Histogram loss (if enabled)
                if best_model.histogram_weight > 0 and hasattr(best_model, 'histogram_loss_fn'):
                    hist_loss, _, _ = best_model.histogram_loss_fn(
                        delta_true, delta_pred, mask=mask, horizon_idx=h_idx
                    )
                    horizon_metrics[h_name]['hist'] += hist_loss.item() * mask.sum().item()
                
                horizon_metrics[h_name]['pixels'] += mask.sum().item()
        
        # Compute average metrics per horizon and overall
        print("\n" + "="*70)
        print("FULL VALIDATION SET METRICS (Best Model) - PER HORIZON")
        print("="*70)
        
        horizon_years = [2005, 2010, 2015, 2020]
        horizon_labels = [5, 10, 15, 20]  # Years into future for plotting
        all_total_losses = []
        
        # Store MAEs for plotting
        mae_conv_cnn = []
        mae_no_change_list = []
        mae_linear_list = []
        
        for h_name, h_year in zip(horizon_names, horizon_years):
            metrics = horizon_metrics[h_name]
            if metrics['pixels'] > 0:
                avg_mse = metrics['mse'] / metrics['pixels']
                avg_mae = metrics['mae'] / metrics['pixels']
                avg_mae_no_change = metrics['mae_no_change'] / metrics['pixels']
                avg_mae_linear = metrics['mae_linear'] / metrics['pixels']
                avg_ssim = metrics['ssim'] / metrics['pixels']
                avg_lap = metrics['lap'] / metrics['pixels']
                avg_hist = metrics['hist'] / metrics['pixels'] if best_model.histogram_weight > 0 else 0.0
                
                # Store for plotting
                mae_conv_cnn.append(avg_mae)
                mae_no_change_list.append(avg_mae_no_change)
                mae_linear_list.append(avg_mae_linear)
                
                # Compute total loss
                avg_total = (avg_mse + 
                           best_model.ssim_weight * avg_ssim + 
                           best_model.laplacian_weight * avg_lap)
                if best_model.histogram_weight > 0:
                    avg_total += best_model.histogram_weight * avg_hist
                
                all_total_losses.append(avg_total)
                
                print(f"\n{h_name.upper()} ({h_year}): {metrics['pixels']:,} valid pixels")
                print(f"  MSE:              {avg_mse:.6f}")
                print(f"  MAE (Conv-CNN):   {avg_mae:.6f}")
                print(f"  MAE (No Change):  {avg_mae_no_change:.6f}")
                print(f"  MAE (Linear):     {avg_mae_linear:.6f}")
                print(f"  SSIM loss:        {avg_ssim:.6f}")
                print(f"  Lap loss:         {avg_lap:.6f}")
                if best_model.histogram_weight > 0:
                    print(f"  Hist loss:        {avg_hist:.6f}")
                print(f"  Total loss:       {avg_total:.6f}")
                
                # Log per-horizon metrics to W&B
                experiment.log({
                    f"val_full/mae_{h_name}": avg_mae,
                    f"val_full/mae_no_change_{h_name}": avg_mae_no_change,
                    f"val_full/mae_linear_{h_name}": avg_mae_linear,
                    f"val_full/mse_{h_name}": avg_mse,
                    f"val_full/ssim_loss_{h_name}": avg_ssim,
                    f"val_full/lap_loss_{h_name}": avg_lap,
                    f"val_full/total_loss_{h_name}": avg_total,
                })
                if best_model.histogram_weight > 0:
                    experiment.log({f"val_full/hist_loss_{h_name}": avg_hist})
        
        # Compute and log average across all horizons
        if all_total_losses:
            avg_total_all = sum(all_total_losses) / len(all_total_losses)
            print(f"\nAVERAGE ACROSS ALL HORIZONS:")
            print(f"  Total loss: {avg_total_all:.6f}")
            print("="*70 + "\n")
            
            experiment.log({"val_full/total_loss_avg": avg_total_all})
        else:
            print("WARNING: No valid pixels found for metric calculation")
        
        # Retrieve means/stds for inverse transform
        ds = val_loader.dataset
        if hasattr(ds, 'dataset'):
            ds = ds.dataset  # Unwrap DataLoader if needed
        hm_mean, hm_std = ds.hm_mean, ds.hm_std
        elev_mean, elev_std = ds.elev_mean, ds.elev_std
        
        # Log images from first batch only (for visualization)
        print("\nCreating multi-horizon visualizations from first batch...")
        images = []
        first_batch = all_batches_data[0]
        input_dynamic = first_batch['input_dynamic']
        input_static = first_batch['input_static']
        input_mask = first_batch['input_mask']
        
        # All horizon targets and predictions
        horizon_names = ['5yr', '10yr', '15yr', '20yr']
        targets_all = {
            '5yr': first_batch['target_5yr'],
            '10yr': first_batch['target_10yr'],
            '15yr': first_batch['target_15yr'],
            '20yr': first_batch['target_20yr']
        }
        preds_all = {
            '5yr': first_batch['preds_5yr'],
            '10yr': first_batch['preds_10yr'],
            '15yr': first_batch['preds_15yr'],
            '20yr': first_batch['preds_20yr']
        }
        
        B = input_dynamic.shape[0]
        for b in range(B):
            # Extract year metadata from this sample (NEW: supports temporal sampling)
            input_years = first_batch.get('input_years', [1990, 1995, 2000])
            target_years = first_batch.get('target_years', [2005, 2010, 2015, 2020])
            # Handle batch dimension if present
            if isinstance(input_years, list) and len(input_years) > 0 and isinstance(input_years[0], list):
                input_years = input_years[b]  # Extract for this sample
                target_years = target_years[b]
            
            # Create multi-horizon figure: 6 rows x 5 columns
            # Row 0: Input HM (dynamic years) + Elevation + empty
            # Rows 1-4: Each horizon (Target, Pred, Error, Delta Obs, Delta Pred)
            # Row 5: Change histograms for all horizons
            fig, axes = plt.subplots(6, 5, figsize=(20, 24))
            # Use a single color ramp for all HM images in original 0-1 scale
            hm_vmin, hm_vmax = 0.0, 1.0
            # Input human footprint chips (T=3), unnormalize and label with actual years
            for t in range(3):
                hm_in = input_dynamic[b, t, 0].cpu().numpy() * hm_std + hm_mean
                # Mask input HM by its own validity
                hm_in_plot = np.where(np.isfinite(hm_in), hm_in, np.nan)
                im = axes[0, t].imshow(hm_in_plot, cmap='turbo', vmin=hm_vmin, vmax=hm_vmax)
                axes[0, t].set_title(f'HM {input_years[t]}')
                axes[0, t].axis('off')
                plt.colorbar(im, ax=axes[0, t], fraction=0.046, pad=0.04)
            # Elevation raster backtransformed
            elev_in = input_static[b, 0].cpu().numpy() * elev_std + elev_mean
            im = axes[0, 3].imshow(elev_in, cmap='terrain')
            axes[0, 3].set_title('Elevation (meters)')
            axes[0, 3].axis('off')
            plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
            
            # Hide empty cell in row 0, column 4
            axes[0, 4].axis('off')
            
            # Compute validity mask from RAW inputs
            # For visualization: use same strict mask as training to show what model actually learns from
            input_dynamic_raw = input_dynamic[b].cpu().numpy()
            # Note: This requires ALL dynamic channels valid (HM + components)
            dynamic_valid = np.isfinite(input_dynamic_raw).all(axis=(0, 1))
            input_static_raw = input_static[b].cpu().numpy()
            # Note: This requires ALL static channels valid
            static_valid = np.isfinite(input_static_raw).all(axis=0)
            most_recent_in = (input_dynamic[b, -1, 0].cpu().numpy() * hm_std + hm_mean)
            
            # Histogram bins: 8 bins from decrease to extreme increase
            histogram_bins = np.array([-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0])
            num_bins = len(histogram_bins) - 1
            
            # Plot each horizon in rows 1-4
            for h_idx, h_name in enumerate(horizon_names):
                row = h_idx + 1
                h_year = target_years[h_idx]  # Get actual year for this sample
                
                # Get target and prediction for this horizon
                target_h = targets_all[h_name][b].cpu().numpy() * hm_std + hm_mean
                pred_h = preds_all[h_name][b].cpu().numpy() * hm_std + hm_mean
                
                # Compute mask for this horizon
                target_valid = np.isfinite(target_h)
                valid_mask = target_valid & dynamic_valid & static_valid
                
                # Deltas
                delta_obs = target_h - most_recent_in
                delta_pred = pred_h - most_recent_in
                
                # Column 0: Target (show year + horizon offset)
                target_plot = np.where(valid_mask, target_h, np.nan)
                im = axes[row, 0].imshow(target_plot, cmap='turbo', vmin=hm_vmin, vmax=hm_vmax)
                axes[row, 0].set_title(f'Target {h_year} (+{(h_idx+1)*5}yr)')
                axes[row, 0].axis('off')
                plt.colorbar(im, ax=axes[row, 0], fraction=0.046, pad=0.04)
                
                # Column 1: Prediction (show year + horizon offset)
                pred_plot = np.where(valid_mask, pred_h, np.nan)
                im = axes[row, 1].imshow(pred_plot, cmap='turbo', vmin=hm_vmin, vmax=hm_vmax)
                axes[row, 1].set_title(f'Pred {h_year} (+{(h_idx+1)*5}yr)')
                axes[row, 1].axis('off')
                plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)
                
                # Column 2: Absolute Error
                error = np.abs(pred_h - target_h)
                error_plot = np.where(valid_mask, error, np.nan)
                im = axes[row, 2].imshow(error_plot, cmap='hot', vmin=0.0, vmax=0.5)
                axes[row, 2].set_title(f'Error {h_year}')
                axes[row, 2].axis('off')
                plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)
                
                # Column 3: Delta Observed
                delta_obs_plot = np.where(valid_mask, delta_obs, np.nan)
                im = axes[row, 3].imshow(delta_obs_plot, cmap='seismic', vmin=-0.3, vmax=0.3)
                axes[row, 3].set_title(f'Δ Obs {h_year}')
                axes[row, 3].axis('off')
                plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)
                
                # Column 4: Delta Predicted
                delta_pred_plot = np.where(valid_mask, delta_pred, np.nan)
                im = axes[row, 4].imshow(delta_pred_plot, cmap='seismic', vmin=-0.3, vmax=0.3)
                axes[row, 4].set_title(f'Δ Pred {h_year}')
                axes[row, 4].axis('off')
                plt.colorbar(im, ax=axes[row, 4], fraction=0.046, pad=0.04)
            
            # Row 5: Histograms for all horizons
            for h_idx, h_name in enumerate(horizon_names):
                h_year = target_years[h_idx]  # Get actual year for this sample
                target_h = targets_all[h_name][b].cpu().numpy() * hm_std + hm_mean
                pred_h = preds_all[h_name][b].cpu().numpy() * hm_std + hm_mean
                target_valid = np.isfinite(target_h)
                valid_mask = target_valid & dynamic_valid & static_valid
                
                delta_obs = target_h - most_recent_in
                delta_pred = pred_h - most_recent_in
                
                delta_obs_valid = delta_obs[valid_mask]
                delta_pred_valid = delta_pred[valid_mask]
                
                if len(delta_obs_valid) > 0:
                    counts_obs, _ = np.histogram(delta_obs_valid, bins=histogram_bins)
                    counts_pred, _ = np.histogram(delta_pred_valid, bins=histogram_bins)
                    
                    x = np.arange(num_bins)
                    width = 0.35
                    axes[5, h_idx].bar(x - width/2, counts_obs, width, label='Obs', alpha=0.7, color='blue')
                    axes[5, h_idx].bar(x + width/2, counts_pred, width, label='Pred', alpha=0.7, color='red')
                    axes[5, h_idx].set_yscale('log')
                    axes[5, h_idx].set_title(f'Δ Histogram {h_year}')
                    axes[5, h_idx].legend(fontsize=6)
                    axes[5, h_idx].grid(alpha=0.3)
                else:
                    axes[5, h_idx].text(0.5, 0.5, 'No valid', ha='center', va='center')
                    axes[5, h_idx].axis('off')
            
            # Hide the 5th column in histogram row (only 4 histograms)
            axes[5, 4].axis('off')
            
            plt.tight_layout()
            # Convert to numpy array and log (robust for macOS backend)
            fig.canvas.draw()
            img_rgba = np.array(fig.canvas.buffer_rgba())
            img_rgb = img_rgba[..., :3]
            images.append(wandb.Image(img_rgb, caption=f"Sample {b}"))
            plt.close(fig)

        experiment.log({"Predictions_vs_Targets": images})
        
        # ---- Accumulate diffs from ALL batches for hexbin and histogram (per horizon) ----
        print("\nAccumulating changes from all validation batches (per horizon)...")
        horizon_names = ['5yr', '10yr', '15yr', '20yr']
        horizon_years = [2005, 2010, 2015, 2020]
        
        # Per-horizon accumulators
        diffs_obs_horizons = {h: [] for h in horizon_names}
        diffs_mod_horizons = {h: [] for h in horizon_names}
        
        for batch_data in all_batches_data:
            input_dynamic_batch = batch_data['input_dynamic']
            input_mask_batch = batch_data['input_mask']
            
            B_batch = input_dynamic_batch.shape[0]
            for b in range(B_batch):
                most_recent_in = (input_dynamic_batch[b, -1, 0].numpy() * hm_std + hm_mean)
                
                # Compute validity mask (same for all horizons)
                input_dynamic_raw = input_dynamic_batch[b].numpy()
                dynamic_valid = np.isfinite(input_dynamic_raw).all(axis=(0, 1))
                input_static_raw = batch_data['input_static'][b].numpy()
                static_valid = np.isfinite(input_static_raw).all(axis=0)
                
                # Process each horizon
                for h_name in horizon_names:
                    target_batch = batch_data[f'target_{h_name}']
                    preds_batch = batch_data[f'preds_{h_name}']
                    
                    # Denormalize
                    target_orig = target_batch[b].numpy() * hm_std + hm_mean
                    pred_orig = preds_batch[b].numpy() * hm_std + hm_mean
                    
                    # Compute validity mask for this horizon
                    target_valid = np.isfinite(target_orig)
                    valid_pred_mask = target_valid & dynamic_valid & static_valid
                    
                    # Calculate changes
                    delta = target_orig - most_recent_in
                    pred_delta = pred_orig - most_recent_in
                    
                    # Accumulate valid pixels only
                    diffs_obs_horizons[h_name].append(delta[valid_pred_mask])
                    diffs_mod_horizons[h_name].append(pred_delta[valid_pred_mask])
        
        print(f"✓ Accumulated changes from {len(all_batches_data)} batches for all horizons")

        # ---- Hexbin plots: Observed vs Predicted HM change (per horizon) ----
        import matplotlib.colors as mcolors
        pmin, pmax = -0.01, 0.2
        
        hexbin_images = []
        for h_name, h_year in zip(horizon_names, horizon_years):
            if len(diffs_obs_horizons[h_name]) > 0:
                diff_obs_all = np.concatenate(diffs_obs_horizons[h_name])
                diff_mod_all = np.concatenate(diffs_mod_horizons[h_name])
                
                # Filter to range
                in_range = (
                    (diff_obs_all >= pmin) & (diff_obs_all <= pmax) &
                    (diff_mod_all >= pmin) & (diff_mod_all <= pmax)
                )
                diff_obs_small = diff_obs_all[in_range]
                diff_mod_small = diff_mod_all[in_range]

                fig2 = plt.figure(figsize=(6, 5))
                hb = plt.hexbin(
                    diff_obs_small,
                    diff_mod_small,
                    gridsize=80,
                    cmap="cubehelix",
                    mincnt=1,
                    norm=mcolors.LogNorm(),
                )
                cbar = plt.colorbar(hb)
                cbar.set_label("Count (log scale)")
                # 1:1 line
                plt.plot([pmin, pmax], [pmin, pmax], linestyle="--", color="grey", label="1:1 line")
                # Axes, labels, legend
                plt.axhline(0, color="black", lw=0.5)
                plt.axvline(0, color="black", lw=0.5)
                plt.xlim(pmin, pmax)
                plt.ylim(pmin, pmax)
                plt.xlabel("Observed difference")
                plt.ylabel("Modelled difference")
                plt.title(f"Obs vs Pred HM Change - {h_year}")
                plt.legend(frameon=False, loc="upper left")
                plt.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)
                plt.tight_layout()
                fig2.canvas.draw()
                img_rgba2 = np.array(fig2.canvas.buffer_rgba())
                img_rgb2 = img_rgba2[..., :3]
                hexbin_images.append(wandb.Image(img_rgb2, caption=f"{h_year} ({h_name})"))
                plt.close(fig2)
        
        if hexbin_images:
            experiment.log({"Obs_vs_Pred_HM_Change": hexbin_images})
            
        # ---- Histograms: Observed vs Predicted HM change distribution (per horizon) ----
        bins = [-1, -0.05, 0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
        bin_labels = ['-1 to -0.05', '-0.05 to 0', '0 to 0.005', '0.005 to 0.02', 
                      '0.02 to 0.05', '0.05 to 0.1', '0.1 to 0.2', '0.2 to 0.5', '0.5 to 1']
        
        histogram_images = []
        for h_name, h_year in zip(horizon_names, horizon_years):
            if len(diffs_obs_horizons[h_name]) > 0:
                diff_obs_all = np.concatenate(diffs_obs_horizons[h_name])
                diff_mod_all = np.concatenate(diffs_mod_horizons[h_name])
                
                # Compute histograms on full data (not just filtered range)
                obs_hist, _ = np.histogram(diff_obs_all, bins=bins)
                pred_hist, _ = np.histogram(diff_mod_all, bins=bins)
                
                # Create histogram figure
                fig3, ax = plt.subplots(figsize=(12, 6))
                x = np.arange(len(bin_labels))
                width = 0.35
                
                # Plot bars
                ax.bar(x - width/2, obs_hist, width, label='Observed', alpha=0.8, color='#2ecc71')
                ax.bar(x + width/2, pred_hist, width, label='Predicted', alpha=0.8, color='#3498db')
                
                # Formatting
                ax.set_xlabel('Change Bins', fontsize=12, fontweight='bold')
                ax.set_ylabel('Count (log scale)', fontsize=12, fontweight='bold')
                ax.set_title(f'Observed vs Predicted HM Change Distribution - {h_year}', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(bin_labels, rotation=45, ha='right')
                ax.legend(fontsize=11)
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                fig3.canvas.draw()
                img_rgba3 = np.array(fig3.canvas.buffer_rgba())
                img_rgb3 = img_rgba3[..., :3]
                histogram_images.append(wandb.Image(img_rgb3, caption=f"{h_year} ({h_name})"))
                plt.close(fig3)
        
        if histogram_images:
            experiment.log({"HM_Change_Histogram": histogram_images})
        
        # ---- MAE Comparison Plot: Conv-CNN vs Baselines ----
        print("\nCreating MAE comparison plot (Conv-CNN vs Baselines)...")
        if len(mae_conv_cnn) > 0 and len(mae_no_change_list) > 0 and len(mae_linear_list) > 0:
            fig_mae, ax_mae = plt.subplots(figsize=(10, 6))
            
            # Plot three lines
            ax_mae.plot(horizon_labels, mae_conv_cnn, marker='o', linewidth=2.5, 
                       label='Conv-CNN', color='#3498db', markersize=8)
            ax_mae.plot(horizon_labels, mae_no_change_list, marker='s', linewidth=2.5, 
                       label='No Change', color='#e74c3c', markersize=8, linestyle='--')
            ax_mae.plot(horizon_labels, mae_linear_list, marker='^', linewidth=2.5, 
                       label='Linear', color='#f39c12', markersize=8, linestyle='--')
            
            # Formatting
            ax_mae.set_xlabel('Forecast Horizon (years)', fontsize=12, fontweight='bold')
            ax_mae.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
            ax_mae.set_title('MAE vs Forecast Horizon: Model Comparison', fontsize=14, fontweight='bold')
            ax_mae.set_xticks(horizon_labels)
            ax_mae.legend(fontsize=11, loc='upper left')
            ax_mae.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig_mae.canvas.draw()
            img_rgba_mae = np.array(fig_mae.canvas.buffer_rgba())
            img_rgb_mae = img_rgba_mae[..., :3]
            experiment.log({"MAE_Comparison": wandb.Image(img_rgb_mae, 
                           caption="MAE comparison: Conv-CNN vs Baselines")})
            plt.close(fig_mae)
            print("✓ MAE comparison plot created and logged")

        # Ensure wandb shuts down cleanly
        experiment.finish()
    else:
        if best_ckpt and use_wandb:
            # Only non-zero ranks should skip W&B logging to avoid crashes on multi-GPU
            print("Skipping W&B image logging: not global rank 0 or WandbLogger not active.")

    # Finalize W&B session if used to ensure clean exit
    if use_wandb and getattr(trainer, "is_global_zero", True):
        try:
            import wandb as _wandb
            _wandb.finish()
        except Exception:
            pass

    # -------------------- Large-area prediction to GeoTIFF --------------------
    def _predict_region_and_write(best_ckpt_path: str):
        import time
        start_time = time.time()
        
        print("\n" + "="*70)
        print("LARGE-AREA PREDICTION")
        print("="*70)
        
        # Only rank 0 performs writing
        if not getattr(trainer, "is_global_zero", True):
            return
        # Resolve region path: CLI > config
        region_path = args.predict_region
        if region_path is None:
            cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
            if cfg_path.exists():
                try:
                    with open(cfg_path, 'r') as f:
                        cfg = yaml.safe_load(f)
                    region_path = (
                        cfg.get("inference", {}).get("region_geojson", None)
                    )
                except Exception:
                    region_path = None
        if region_path is None or not os.path.exists(region_path):
            print("⚠ No valid prediction region specified; skipping large-area prediction.")
            return
        
        print(f"Region file: {region_path}")

        # Load region polygon (assume EPSG:4326 if no CRS field)
        with open(region_path, 'r') as f:
            gj = json.load(f)
        # Merge all features into a single geometry collection/polygon list
        geoms = [shape(feat["geometry"]) for feat in gj.get("features", [])]
        if not geoms:
            print("Empty geometry in region GeoJSON; skipping.")
            return
        # Use target HM raster (2020) as spatial reference
        target_years = getattr(train_loader.dataset, 'fixed_target_years', (2005, 2010, 2015, 2020))
        target_year = target_years[-1]  # Use 2020 as reference
        year_to_idx = {y: i for i, y in enumerate(years)}
        target_src_path = hm_files[year_to_idx[target_year]]
        with rasterio.open(target_src_path) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_height, ref_width = ref.height, ref.width
            # Reproject geoms to ref CRS
            # Assume GeoJSON in EPSG:4326 unless a crs member exists (rare in modern GeoJSON)
            transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
            geoms_ref = [shp_transform(lambda x, y: transformer.transform(x, y), g) for g in geoms]
            # Build a unioned geometry
            try:
                from shapely.ops import unary_union
                region_geom = unary_union(geoms_ref)
            except Exception:
                region_geom = geoms_ref[0]
            # Compute bounding rows/cols
            minx, miny, maxx, maxy = region_geom.bounds
            top_left = rowcol(ref_transform, minx, maxy, op=float)
            bottom_right = rowcol(ref_transform, maxx, miny, op=float)
            r0 = int(max(0, np.floor(min(top_left[0], bottom_right[0]))))
            c0 = int(max(0, np.floor(min(top_left[1], bottom_right[1]))))
            r1 = int(min(ref_height, np.ceil(max(top_left[0], bottom_right[0]))))
            c1 = int(min(ref_width, np.ceil(max(top_left[1], bottom_right[1]))))
            if r1 <= r0 or c1 <= c0:
                print("⚠ Region is outside raster extent; skipping.")
                return

            # Prepare accumulators over the bbox window (one per horizon)
            Hwin, Wwin = r1 - r0, c1 - c0
            print(f"Region bounding box: {Hwin} × {Wwin} pixels")
            print(f"  Row range: [{r0}, {r1})")
            print(f"  Col range: [{c0}, {c1})")
            
            # Multi-horizon accumulators
            horizon_names = ['5yr', '10yr', '15yr', '20yr']
            horizon_years = [2005, 2010, 2015, 2020]
            accum_horizons = {h: np.zeros((Hwin, Wwin), dtype=np.float64) for h in horizon_names}
            wsum = np.zeros((Hwin, Wwin), dtype=np.float64)
            nodata_mask_total = np.zeros((Hwin, Wwin), dtype=bool)

            # Stats and config from training dataset
            ds_train = train_loader.dataset
            hm_mean, hm_std = ds_train.hm_mean, ds_train.hm_std
            elev_mean, elev_std = ds_train.elev_mean, ds_train.elev_std
            include_components = bool(getattr(ds_train, 'include_components', True))
            static_list_paths = list(static_files if args.static_channels is None else static_files[:int(args.static_channels)])
            input_years = list(getattr(ds_train, 'fixed_input_years', (1990, 1995, 2000)))
            t_idxs = [year_to_idx[y] for y in input_years]

            # Open all sources
            hm_srcs = [rasterio.open(p) for p in hm_files]
            comp_srcs = {y: [rasterio.open(p) for p in component_files[y]] for y in years} if include_components else {y: [] for y in years}
            stat_srcs = [rasterio.open(p) for p in static_list_paths]

            tile = 128
            stride = int(args.predict_stride)
            from rasterio.windows import Window
            # Precompute a region mask over the bbox for faster per-tile tests
            bbox_transform = ref_transform * Affine.translation(c0, r0)
            bbox_mask = rio_features.geometry_mask([mapping(region_geom)], out_shape=(Hwin, Wwin), transform=bbox_transform, invert=True)

            # Load model for inference
            print(f"\nLoading model from checkpoint: {best_ckpt_path}")
            device = next(model.parameters()).device
            infer_model = SpatioTemporalLightningModule.load_from_checkpoint(best_ckpt_path, map_location=device)
            infer_model.eval()
            print(f"✓ Model loaded on device: {device}")
            
            def lonlat_grid_for_window(i0: int, j0: int, hi: int, wj: int):
                rows = np.arange(i0, i0 + hi)
                cols = np.arange(j0, j0 + wj)
                rr, cc = np.meshgrid(rows, cols, indexing='ij')
                xs, ys = rasterio.transform.xy(ref_transform, rr, cc)
                xs = np.array(xs); ys = np.array(ys)
                # Ensure xs and ys have shape [hi, wj]
                if xs.ndim == 1:
                    xs = xs.reshape(hi, wj)
                    ys = ys.reshape(hi, wj)
                if ref_crs and ref_crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
                    transformer = Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
                    lon, lat = transformer.transform(xs, ys)
                    lon = np.array(lon); lat = np.array(lat)
                    if lon.ndim == 1:
                        lon = lon.reshape(hi, wj)
                        lat = lat.reshape(hi, wj)
                else:
                    lon, lat = xs, ys
                return np.stack([lon, lat], axis=-1).astype(np.float32)  # [hi, wj, 2]

            # Calculate total tiles for progress tracking
            num_tiles_i = len(range(r0, r1, stride))
            num_tiles_j = len(range(c0, c1, stride))
            total_tiles = num_tiles_i * num_tiles_j
            print(f"\nProcessing {total_tiles:,} tiles ({num_tiles_i} × {num_tiles_j})")
            print(f"  Tile size: {tile} × {tile} pixels")
            print(f"  Stride: {stride} pixels")
            print(f"  Target horizons: {horizon_years}")
            print(f"  Input years: {input_years}")
            print()
            
            # Collect all tile coordinates first
            tile_coords = []
            for i in range(r0, r1, stride):
                for j in range(c0, c1, stride):
                    tile_coords.append((i, j))
            
            batch_size = args.predict_batch_size
            print(f"  Batch size: {batch_size} tiles")
            
            tiles_processed = 0
            tiles_skipped = 0
            tiles_with_valid = 0
            last_percent = -1
            tile_start_time = time.time()
            
            import torch
            
            # Process tiles in batches
            for batch_start in range(0, len(tile_coords), batch_size):
                batch_end = min(batch_start + batch_size, len(tile_coords))
                batch_tiles = tile_coords[batch_start:batch_end]
                
                # Prepare batch data
                batch_inputs_dyn = []
                batch_inputs_stat = []
                batch_lonlats = []
                batch_metadata = []  # Store (i, j, hi, wj, li0, lj0, li1, lj1, valid_mask)
                
                for i, j in batch_tiles:
                    hi = min(tile, r1 - i)
                    wj = min(tile, c1 - j)
                    if hi <= 0 or wj <= 0:
                        continue
                    # Local indices in accum arrays
                    li0, lj0 = i - r0, j - c0
                    li1, lj1 = li0 + hi, lj0 + wj
                    submask = bbox_mask[li0:li1, lj0:lj1]
                    if not np.any(submask):
                        tiles_processed += 1
                        tiles_skipped += 1
                        continue
                    win = Window(j, i, wj, hi)
                    # Build inputs
                    dyn_ts = []
                    for t_idx, y in zip(t_idxs, input_years):
                        channels = []
                        arr_hm = hm_srcs[t_idx].read(1, window=win, masked=True).filled(np.nan)
                        # Data is already in [0, 1] range
                        channels.append((arr_hm - hm_mean) / hm_std)
                        if include_components and comp_srcs.get(y, []):
                            for src in comp_srcs[y]:
                                carr = src.read(1, window=win, masked=True).filled(np.nan)
                                # Replace NaN with 0 BEFORE normalization (missing = no pressure/activity)
                                carr = np.nan_to_num(carr, nan=0.0)
                                # Data is already in [0, 1] range
                                # NOTE: Using simplified normalization here (should use per-variable stats)
                                channels.append((carr - hm_mean) / hm_std)
                        dyn_ts.append(np.stack(channels, axis=0))  # [C_dyn, hi, wj]
                    input_dynamic_np = np.stack(dyn_ts, axis=0)  # [T, C_dyn, hi, wj]
                    static_chs = []
                    # Static file order: [ele, tas, tasmin, pr, dpi_dsi, iucn_nostrict, iucn_strict]
                    nan_to_zero_static = {0, 4, 5, 6}  # ele, dpi_dsi, iucn_nostrict, iucn_strict
                    for static_idx, src in enumerate(stat_srcs):
                        sarr = src.read(1, window=win, masked=True).filled(np.nan)
                        # Replace NaN with 0 for specific variables (before normalization)
                        if static_idx in nan_to_zero_static:
                            sarr = np.nan_to_num(sarr, nan=0.0)
                        # NOTE: Using simplified normalization here (should use per-variable stats)
                        static_chs.append((sarr - elev_mean) / elev_std)
                    input_static_np = np.stack(static_chs, axis=0) if static_chs else np.zeros((0, hi, wj), dtype=np.float32)

                    # Valid mask for prediction (less strict than training)
                    # Only require HM channel (index 0) to be valid across all timesteps
                    # Component channels can be NaN (will be replaced with 0.0)
                    hm_valid_all_times = np.isfinite(input_dynamic_np[:, 0, :, :]).all(axis=0)  # [H, W]
                    # Only require first static channel (elevation) to be valid
                    stat_valid = np.isfinite(input_static_np[0]) if static_chs else np.ones((hi, wj), dtype=bool)
                    valid_mask = submask & hm_valid_all_times & stat_valid
                    if not np.any(valid_mask):
                        tiles_processed += 1
                        tiles_skipped += 1
                        continue
                    
                    tiles_with_valid += 1
                    tiles_processed += 1
                    
                    # Add to batch
                    # Replace NaN with 0.0 in normalized space = mean in original space
                    in_dyn = np.nan_to_num(input_dynamic_np, nan=0.0).astype(np.float32)
                    in_stat = np.nan_to_num(input_static_np, nan=0.0).astype(np.float32)
                    lonlat_hw2 = lonlat_grid_for_window(i, j, hi, wj)
                    
                    # Pad to tile size if needed (for edge tiles)
                    if hi < tile or wj < tile:
                        # Pad dynamic: [T, C, hi, wj] -> [T, C, tile, tile]
                        T, C = in_dyn.shape[:2]
                        in_dyn_padded = np.zeros((T, C, tile, tile), dtype=np.float32)
                        in_dyn_padded[:, :, :hi, :wj] = in_dyn
                        in_dyn = in_dyn_padded
                        
                        # Pad static: [C, hi, wj] -> [C, tile, tile]
                        C_stat = in_stat.shape[0]
                        in_stat_padded = np.zeros((C_stat, tile, tile), dtype=np.float32)
                        in_stat_padded[:, :hi, :wj] = in_stat
                        in_stat = in_stat_padded
                        
                        # Pad lonlat: [hi, wj, 2] -> [tile, tile, 2]
                        lonlat_padded = np.zeros((tile, tile, 2), dtype=np.float32)
                        lonlat_padded[:hi, :wj, :] = lonlat_hw2
                        lonlat_hw2 = lonlat_padded
                    
                    batch_inputs_dyn.append(in_dyn)
                    batch_inputs_stat.append(in_stat)
                    batch_lonlats.append(lonlat_hw2)
                    batch_metadata.append((i, j, hi, wj, li0, lj0, li1, lj1, valid_mask))
                
                # Process batch on GPU if we have any valid tiles
                if len(batch_inputs_dyn) > 0:
                    # Stack into batch tensors
                    batch_dyn_tensor = torch.from_numpy(np.stack(batch_inputs_dyn, axis=0)).to(device)  # [B, T, C, H, W]
                    batch_stat_tensor = torch.from_numpy(np.stack(batch_inputs_stat, axis=0)).to(device)  # [B, C, H, W]
                    batch_lonlat_tensor = torch.from_numpy(np.stack(batch_lonlats, axis=0)).to(device)  # [B, H, W, 2]
                    
                    with torch.no_grad():
                        batch_preds = infer_model(batch_dyn_tensor, batch_stat_tensor, lonlat=batch_lonlat_tensor)  # [B, 4, H, W]
                    
                    # Process each tile in the batch
                    for tile_idx, (i, j, hi, wj, li0, lj0, li1, lj1, valid_mask) in enumerate(batch_metadata):
                        # Extract predictions for this tile (crop to actual size if padded)
                        preds_horizons = {}
                        for h_idx, h_name in enumerate(horizon_names):
                            pred_h = batch_preds[tile_idx, h_idx, :hi, :wj].detach().cpu().numpy()  # [hi, wj] normalized (crop padding)
                            pred_h = pred_h * hm_std + hm_mean  # denormalize to [0, 1] scale
                            preds_horizons[h_name] = pred_h
                        
                        # Distance-to-edge weights within tile
                        interior = valid_mask.astype(np.uint8)
                        interior[[0, -1], :] = 0
                        interior[:, [0, -1]] = 0
                        weights = distance_transform_edt(interior)
                        weights = np.where(valid_mask, weights, 0.0)
                        
                        if weights.max() > 0:
                            # Accumulate each horizon
                            for h_name, pred_h in preds_horizons.items():
                                accum_horizons[h_name][li0:li1, lj0:lj1] += pred_h * weights
                            wsum[li0:li1, lj0:lj1] += weights
                        nodata_mask_total[li0:li1, lj0:lj1] |= ~valid_mask
                
                # Progress indicator (after each batch)
                percent = int(100 * tiles_processed / total_tiles)
                if percent != last_percent and percent % 5 == 0:
                    elapsed = time.time() - tile_start_time
                    tiles_per_sec = tiles_processed / elapsed if elapsed > 0 else 0
                    eta_sec = (total_tiles - tiles_processed) / tiles_per_sec if tiles_per_sec > 0 else 0
                    print(f"  Progress: {percent:3d}% ({tiles_processed:,}/{total_tiles:,} tiles) | "
                          f"Speed: {tiles_per_sec:.1f} tiles/s | "
                          f"ETA: {int(eta_sec//60):02d}:{int(eta_sec%60):02d}")
                    last_percent = percent

            # Final blend for all horizons
            print("\n" + "-"*70)
            print("Blending overlapping tiles for all horizons...")
            m = wsum > 0
            out_horizons = {}
            for h_name in horizon_names:
                out_h = np.full((Hwin, Wwin), np.nan, dtype=np.float32)
                out_h[m] = (accum_horizons[h_name][m] / wsum[m]).astype(np.float32)
                out_horizons[h_name] = out_h
            
            # Calculate statistics
            num_valid_pixels = m.sum()
            num_total_pixels = Hwin * Wwin
            valid_percent = 100 * num_valid_pixels / num_total_pixels
            
            print(f"✓ Blending complete")
            print(f"  Valid pixels: {num_valid_pixels:,} / {num_total_pixels:,} ({valid_percent:.1f}%)")

            # Write GeoTIFF for each horizon
            print("\nWriting output GeoTIFFs...")
            out_profile = ref.profile.copy()
            out_profile.update({
                'height': Hwin,
                'width': Wwin,
                'transform': ref_transform * Affine.translation(c0, r0),
                'count': 1,
                'dtype': 'float32',
                'compress': 'deflate'
            })
            out_dir = Path(os.getcwd()) / 'data' / 'predictions'
            out_dir.mkdir(parents=True, exist_ok=True)
            
            out_paths = {}
            for h_name, h_year in zip(horizon_names, horizon_years):
                out_path = out_dir / f"prediction_{h_year}_blended.tif"
                with rasterio.open(out_path, 'w', **out_profile) as dst:
                    dst.write(out_horizons[h_name], 1)
                out_paths[h_name] = out_path
                print(f"  ✓ {h_year}: {out_path}")
            
            # Final summary
            elapsed_total = time.time() - start_time
            print("\n" + "="*70)
            print("PREDICTION SUMMARY")
            print("="*70)
            print(f"Total tiles processed: {tiles_processed:,}")
            print(f"  Tiles with valid data: {tiles_with_valid:,}")
            print(f"  Tiles skipped (no data/outside region): {tiles_skipped:,}")
            print(f"Output dimensions: {Hwin} × {Wwin} pixels")
            print(f"Valid output pixels: {num_valid_pixels:,} ({valid_percent:.1f}%)")
            print(f"\nOutput files:")
            for h_name, h_year in zip(horizon_names, horizon_years):
                print(f"  {h_year}: {out_paths[h_name]}")
            print(f"\nTotal time: {int(elapsed_total//60):02d}:{int(elapsed_total%60):02d}")
            print("="*70 + "\n")

            # Close sources
            for src in hm_srcs:
                src.close()
            for y in comp_srcs:
                for src in comp_srcs[y]:
                    src.close()
            for src in stat_srcs:
                src.close()

    if args.predict_after_training and checkpoint_cb.best_model_path:
        _predict_region_and_write(checkpoint_cb.best_model_path)
