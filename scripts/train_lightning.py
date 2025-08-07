import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from src.models.lightning_module import SpatioTemporalLightningModule
from torchgeo_dataloader import get_dataloader
import os

if __name__ == "__main__":
    # Data
    train_loader = get_dataloader(batch_size=8, chip_size=128, timesteps=3, chips_per_epoch=200)
    val_loader = get_dataloader(batch_size=8, chip_size=128, timesteps=3, chips_per_epoch=40)

    # Model
    model = SpatioTemporalLightningModule(hidden_dim=16, lr=1e-3)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    early_stop_cb = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=[checkpoint_cb, early_stop_cb],
        accelerator='auto',
        default_root_dir=os.path.join(os.getcwd(), 'models', 'checkpoints')
    )

    # Train
    trainer.fit(model, train_loader, val_loader)
