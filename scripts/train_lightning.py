import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from src.models.lightning_module import SpatioTemporalLightningModule
from torchgeo_dataloader import get_dataloader
import os
import argparse
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v = str(v).strip().lower()
    if v in ("yes", "true", "t", "y", "1"):  # truthy
        return True
    if v in ("no", "false", "f", "n", "0"):  # falsy
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 1 train/val batch for a quick smoke test")
    parser.add_argument("--max_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--train_chips", type=int, default=2000, help="Chips per epoch for training")
    parser.add_argument("--val_chips", type=int, default=800, help="Chips per epoch for validation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for train/val")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate for optimizer")
    parser.add_argument("--ssim_weight", type=float, default=0.2, help="Weight for SSIM term in total loss")
    parser.add_argument("--print_model", action="store_true", help="Print model summary and parameter counts before training")
    # Model complexity controls
    parser.add_argument("--hidden_dim", type=int, default=16, help="ConvLSTM hidden channels per layer")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of ConvLSTM layers")
    parser.add_argument("--kernel_size", type=int, default=3, help="Conv kernel size for ConvLSTM cells (odd number)")
    parser.add_argument("--train_mode", type=str, default="random", choices=["random", "grid"], help="Sampling mode for training")
    parser.add_argument("--val_mode", type=str, default="random", choices=["random", "grid"], help="Sampling mode for validation")
    parser.add_argument("--stride", type=int, default=64, help="Stride for grid sampling (pixels)")
    parser.add_argument("--val_every_n_epochs", type=int, default=5, help="Run validation every N epochs (default: 5)")
    # LR finder controls
    parser.add_argument("--lr_find", action="store_true", help="Run LR finder before training and set suggested LR")
    parser.add_argument("--lr_find_only", action="store_true", help="Only run LR finder and exit (do not train)")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum LR for finder sweep")
    parser.add_argument("--lr_max", type=float, default=1.0, help="Maximum LR for finder sweep")
    parser.add_argument("--lr_num", type=int, default=100, help="Number of steps for LR finder sweep")
    # Dataset filtering/statistics
    parser.add_argument("--min_valid_ratio", type=float, default=0.8, help="Minimum valid pixel ratio per chip")
    parser.add_argument("--stat_samples", type=int, default=256, help="Number of random samples to compute normalization stats")
    parser.add_argument(
        "--enforce_input_hm_valid",
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="Require valid dynamic inputs (HM+components) at last timestep when filtering chips",
    )
    args = parser.parse_args()
    # Data
    train_loader = get_dataloader(
        batch_size=args.batch_size,
        chip_size=128,
        timesteps=3,
        chips_per_epoch=args.train_chips,
        mode=args.train_mode,
        stride=args.stride,
        min_valid_ratio=args.min_valid_ratio,
        stat_samples=args.stat_samples,
        enforce_input_hm_valid=args.enforce_input_hm_valid,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    # Validation uses grid sampling; no future horizons in single-step mode
    val_loader = get_dataloader(
        batch_size=args.batch_size,
        chip_size=128,
        timesteps=3,
        chips_per_epoch=args.val_chips,
        mode=args.val_mode,
        stride=args.stride,
        min_valid_ratio=args.min_valid_ratio,
        stat_samples=args.stat_samples,
        enforce_input_hm_valid=args.enforce_input_hm_valid,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    # Model
    model = SpatioTemporalLightningModule(hidden_dim=args.hidden_dim, num_layers=args.num_layers, kernel_size=args.kernel_size, lr=args.lr, ssim_weight=args.ssim_weight)
    # Optional model summary
    if args.print_model:
        # Warm-up: run one forward pass to trigger lazy ConvLSTM initialization
        try:
            batch = next(iter(train_loader))
            model.eval()
            with torch.no_grad():
                input_dynamic = batch['input_dynamic']
                input_static = batch['input_static']
                _ = model(input_dynamic, input_static)
            model.train()
        except Exception as e:
            print(f"[print_model] Warm-up forward failed: {e}. Proceeding with current params.")
        def _count_params(m):
            total = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable, total - trainable
        total, trainable, frozen = _count_params(model)
        print("=== Model Summary ===")
        print(f"LightningModule: {model.__class__.__name__}")
        print(f"Inner model: {getattr(model, 'model', type('N', (), {})()).__class__.__name__}")
        print(f"Learning rate: {getattr(model, 'lr', 'n/a')}")
        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,}")
        print(f"Frozen params: {frozen:,}")
        # Top-level children breakdown
        print("Top-level modules:")
        for name, child in model.named_children():
            ctot, ctrain, cfroz = _count_params(child)
            print(f"  - {name}: total={ctot:,}, trainable={ctrain:,}, frozen={cfroz:,}")
    # Set normalization stats for physical-scale MAE logging
    if hasattr(train_loader, 'dataset'):
        ds = train_loader.dataset
        if hasattr(ds, 'hm_mean') and hasattr(ds, 'hm_std'):
            model.hm_mean = ds.hm_mean
            model.hm_std = ds.hm_std

    # Callbacks
    checkpoint_cb = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
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
        check_val_every_n_epoch=args.val_every_n_epochs,
    )

    # Optional: Learning Rate Finder
    if args.lr_find:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            min_lr=args.lr_min,
            max_lr=args.lr_max,
            num_training=args.lr_num,
            mode='exponential',
        )
        try:
            suggested_lr = float(lr_finder.suggestion())
        except Exception:
            suggested_lr = None
        if suggested_lr is not None:
            print(f"[LR FINDER] Suggested LR: {suggested_lr:.3e}")
            # Set on model for subsequent training
            model.lr = suggested_lr
            if hasattr(model, 'hparams'):
                try:
                    model.hparams.lr = suggested_lr
                except Exception:
                    pass
            # Log to W&B if enabled
            if use_wandb and wandb_logger:
                try:
                    import wandb as _wandb
                    _wandb.log({"lr_finder/suggested_lr": suggested_lr})
                except Exception:
                    pass
        else:
            print("[LR FINDER] No LR suggestion could be computed.")
        if args.lr_find_only:
            sys.exit(0)

    # Train
    trainer.fit(model, train_loader, val_loader)

    # --- Log predictions from best checkpoint to wandb ---
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    import wandb
    # Find best checkpoint
    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt and use_wandb:
        try:
            # Lazy modules may cause unexpected keys; load non-strict for logging purposes
            best_model = SpatioTemporalLightningModule.load_from_checkpoint(best_ckpt, strict=False)
            best_model.eval()
        except Exception as e:
            print(f"WARNING: Failed to load best checkpoint strictly due to: {e}. Skipping image logging.")
            sys.exit(0)
        # Move model and data to same device
        device = next(best_model.parameters()).device
        # Ensure normalization stats are present for logging
        if hasattr(train_loader, 'dataset'):
            ds_train = train_loader.dataset
            if hasattr(ds_train, 'hm_mean') and hasattr(ds_train, 'hm_std'):
                best_model.hm_mean = ds_train.hm_mean
                best_model.hm_std = ds_train.hm_std
        # Inference on a batch of validation data with at least one valid target pixel
        valid_batch_found = False
        for batch in val_loader:
            target = batch['target']
            # Check for at least one valid (non-NaN) pixel in any sample
            if torch.any(~torch.isnan(target)).item():
                input_dynamic = batch['input_dynamic'].to(device)
                input_static = batch['input_static'].to(device)
                target = target.to(device)
                best_model.eval()
                with torch.no_grad():
                    if input_dynamic.dim() == 4:
                        input_dynamic = input_dynamic.unsqueeze(2)
                    preds = best_model(input_dynamic, input_static).squeeze(1)
                valid_batch_found = True
                break
        if not valid_batch_found:
            print("WARNING: No valid (non-NaN) target pixels found in any validation batch for image logging and MAE.")
            sys.exit(0)
        # Log images to wandb
        images = []
        B = input_dynamic.shape[0]
        # Retrieve means/stds for inverse transform
        ds = val_loader.dataset
        if hasattr(ds, 'dataset'):
            ds = ds.dataset  # Unwrap DataLoader if needed
        hm_mean, hm_std = ds.hm_mean, ds.hm_std
        static_mean, static_std = ds.static_mean, ds.static_std
        # Years for labeling come from fixed configuration in dataset
        fixed_input_years = getattr(ds, 'fixed_input_years', (None, None, None))
        fixed_target_year = getattr(ds, 'fixed_target_year', None)
        for b in range(B):
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            # Use a single color ramp for all HM images (data scaled 0–1)
            hm_vmin, hm_vmax = 0.0, 1.0
            # Input human footprint chips (T=3), unnormalize and label with fixed years
            input_years = list(fixed_input_years)
            for t in range(3):
                hm_in = input_dynamic[b, t, 0].cpu().numpy() * hm_std + hm_mean
                im = axes[0, t].imshow(hm_in, cmap='turbo', vmin=hm_vmin, vmax=hm_vmax)
                axes[0, t].set_title(f'HM {input_years[t]}')
                axes[0, t].axis('off')
                plt.colorbar(im, ax=axes[0, t], fraction=0.046, pad=0.04)
            # First static raster (e.g., elevation), unnormalize
            # static_mean/std may be arrays; index 0 for the first channel
            ch0_mean = static_mean[0] if hasattr(static_mean, '__len__') else static_mean
            ch0_std = static_std[0] if hasattr(static_std, '__len__') else static_std
            static0_in = input_static[b, 0].cpu().numpy() * ch0_std + ch0_mean
            im = axes[0, 3].imshow(static0_in, cmap='terrain')
            axes[0, 3].set_title('Static[0] (e.g., Elevation)')
            axes[0, 3].axis('off')
            plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
            # Hide unused subplot in first row
            axes[0, 4].axis('off')
            # Target, unnormalize and label with year
            target_denorm = target[b].cpu().numpy() * hm_std + hm_mean
            target_year = fixed_target_year if fixed_target_year is not None else 'Target'
            im0 = axes[1, 0].imshow(target_denorm, cmap='turbo', vmin=hm_vmin, vmax=hm_vmax)
            axes[1, 0].set_title(f'Target HM {target_year}')
            axes[1, 0].axis('off')
            plt.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)
            # Prediction, unnormalize and label
            pred_denorm = preds[b].cpu().numpy() * hm_std + hm_mean
            im1 = axes[1, 1].imshow(pred_denorm, cmap='turbo', vmin=hm_vmin, vmax=hm_vmax)
            axes[1, 1].set_title(f'Predicted HM {target_year}')
            axes[1, 1].axis('off')
            plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)
            # Error (in original units)
            error = np.abs(pred_denorm - target_denorm)
            im2 = axes[1, 2].imshow(error, cmap='hot', vmin=0.0, vmax=hm_vmax)
            axes[1, 2].set_title('Absolute Error (0-1)')
            axes[1, 2].axis('off')
            plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)
            # Delta image (observed): target - most recent input HM
            most_recent_in = input_dynamic[b, -1, 0].cpu().numpy() * hm_std + hm_mean
            delta = target_denorm - most_recent_in
            vmax_delta = np.nanmax(np.abs(delta))
            im3 = axes[1, 3].imshow(delta, cmap='bwr', vmin=-vmax_delta, vmax=vmax_delta)
            axes[1, 3].set_title(f'Delta HM {target_year}-{input_years[-1]}')
            axes[1, 3].axis('off')
            plt.colorbar(im3, ax=axes[1, 3], fraction=0.046, pad=0.04)
            # Delta image (predicted): prediction - most recent input HM, using the same color scaling as observed delta
            pred_delta = pred_denorm - most_recent_in
            im4 = axes[1, 4].imshow(pred_delta, cmap='bwr', vmin=-vmax_delta, vmax=vmax_delta)
            axes[1, 4].set_title(f'Pred Delta HM {target_year}-{input_years[-1]}')
            axes[1, 4].axis('off')
            plt.colorbar(im4, ax=axes[1, 4], fraction=0.046, pad=0.04)
            plt.tight_layout()
            # Convert to numpy array and log (robust for macOS backend)
            fig.canvas.draw()
            img_rgba = np.array(fig.canvas.buffer_rgba())
            img_rgb = img_rgba[..., :3]
            images.append(wandb.Image(img_rgb, caption=f"Sample {b}"))
            plt.close(fig)
        wandb.log({"Predictions_vs_Targets": images})
        # Ensure wandb shuts down cleanly
        wandb.finish()

    else:
        print("No best checkpoint found for image logging.")

    # Finalize W&B session if used to ensure clean exit
    if use_wandb:
        try:
            import wandb as _wandb
            _wandb.finish()
        except Exception:
            pass
