import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from src.models.lightning_module import SpatioTemporalLightningModule
from torchgeo_dataloader import get_dataloader
import os

if __name__ == "__main__":
    # Data
    train_loader = get_dataloader(batch_size=8, chip_size=128, timesteps=3, chips_per_epoch=200)
    val_loader = get_dataloader(batch_size=8, chip_size=128, timesteps=3, chips_per_epoch=40, mode="grid")

    # Model
    model = SpatioTemporalLightningModule(hidden_dim=16, lr=1e-3)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    # No early stopping

    # Wandb logger
    wandb_logger = WandbLogger(project='spatio-temporal-convlstm', log_model=True)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_cb],
        accelerator='auto',
        default_root_dir=os.path.join(os.getcwd(), 'models', 'checkpoints'),
        logger=wandb_logger
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # --- Log predictions from best checkpoint to wandb ---
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    import wandb
    # Find best checkpoint
    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        best_model = SpatioTemporalLightningModule.load_from_checkpoint(best_ckpt)
        best_model.eval()
        # Move model and data to same device
        device = next(best_model.parameters()).device
        best_model = best_model.to(device)
        batch = next(iter(val_loader))
        input_dynamic = batch['input_dynamic'].to(device)
        input_static = batch['input_static'].to(device)
        target = batch['target'].to(device)
        with torch.no_grad():
            if input_dynamic.dim() == 4:
                input_dynamic = input_dynamic.unsqueeze(2)
            preds = best_model(input_dynamic, input_static).squeeze(1)
        # Log images to wandb
        images = []
        B = input_dynamic.shape[0]
        # Retrieve means/stds for inverse transform
        ds = val_loader.dataset
        if hasattr(ds, 'dataset'):
            ds = ds.dataset  # Unwrap DataLoader if needed
        hm_mean, hm_std = ds.hm_mean, ds.hm_std
        elev_mean, elev_std = ds.elev_mean, ds.elev_std
        for b in range(B):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            # Input human footprint chips (T=3), unnormalize
            for t in range(3):
                hm_in = input_dynamic[b, t, 0].cpu().numpy() * hm_std + hm_mean
                im = axes[0, t].imshow(hm_in, cmap='plasma', vmin=0, vmax=10000)
                axes[0, t].set_title(f'Input HM t-{2-t} (0-10k)')
                axes[0, t].axis('off')
                plt.colorbar(im, ax=axes[0, t], fraction=0.046, pad=0.04)
            # Elevation raster, unnormalize
            elev_in = input_static[b, 0].cpu().numpy() * elev_std + elev_mean
            im = axes[0, 3].imshow(elev_in, cmap='terrain')
            axes[0, 3].set_title('Elevation (meters)')
            axes[0, 3].axis('off')
            plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
            # Target, unnormalize
            target_denorm = target[b].cpu().numpy() * hm_std + hm_mean
            im0 = axes[1, 0].imshow(target_denorm, cmap='magma', vmin=0, vmax=10000)
            axes[1, 0].set_title('Target (0-10k)')
            axes[1, 0].axis('off')
            plt.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)
            # Prediction, unnormalize
            pred_denorm = preds[b].cpu().numpy() * hm_std + hm_mean
            im1 = axes[1, 1].imshow(pred_denorm, cmap='viridis', vmin=0, vmax=10000)
            axes[1, 1].set_title('Prediction (0-10k)')
            axes[1, 1].axis('off')
            plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)
            # Error (in original units)
            error = np.abs(pred_denorm - target_denorm)
            im2 = axes[1, 2].imshow(error, cmap='hot', vmin=0, vmax=10000)
            axes[1, 2].set_title('Absolute Error (0-10k)')
            axes[1, 2].axis('off')
            plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)
            # Hide unused subplot
            axes[1, 3].axis('off')
            plt.tight_layout()
            # Convert to numpy array and log (robust for macOS backend)
            fig.canvas.draw()
            img_rgba = np.array(fig.canvas.buffer_rgba())
            img_rgb = img_rgba[..., :3]
            images.append(wandb.Image(img_rgb, caption=f"Sample {b}"))
            plt.close(fig)
        wandb.log({"Predictions_vs_Targets": images})
    else:
        print("No best checkpoint found for image logging.")
