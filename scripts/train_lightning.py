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
    early_stop_cb = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # Wandb logger
    wandb_logger = WandbLogger(project='spatio-temporal-convlstm', log_model=True)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_cb, early_stop_cb],
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
        for b in range(B):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            im0 = axes[0].imshow(target[b].cpu(), cmap='magma', vmin=0, vmax=1)
            axes[0].set_title('Target')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            im1 = axes[1].imshow(preds[b].cpu(), cmap='viridis', vmin=0, vmax=1)
            axes[1].set_title('Prediction')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            error = (preds[b] - target[b]).abs()
            im2 = axes[2].imshow(error.cpu(), cmap='hot', vmin=0, vmax=1)
            axes[2].set_title('Absolute Error')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            plt.tight_layout()
            # Convert to numpy array and log (robust for macOS backend)
            fig.canvas.draw()
            img_rgba = np.array(fig.canvas.buffer_rgba())
            # Convert RGBA to RGB
            img_rgb = img_rgba[..., :3]
            images.append(wandb.Image(img_rgb, caption=f"Sample {b}"))
            plt.close(fig)
        wandb.log({"Predictions_vs_Targets": images})
    else:
        print("No best checkpoint found for image logging.")
