import torch
import pytorch_lightning as pl
from .spatiotemporal_predictor import SpatioTemporalPredictor

class SpatioTemporalLightningModule(pl.LightningModule):
    def __init__(self, hidden_dim=16, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SpatioTemporalPredictor(hidden_dim=hidden_dim)
        self.loss_fn = torch.nn.MSELoss()
        self.mae_fn = torch.nn.L1Loss()
        self.lr = lr

    def forward(self, input_dynamic, input_static):
        # Ensure input_dynamic is [B, T, 1, H, W]
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        return self.model(input_dynamic, input_static)

    def training_step(self, batch, batch_idx):
        input_dynamic = batch['input_dynamic']
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        preds = self(input_dynamic, batch['input_static'])
        loss = self.loss_fn(preds, batch['target'].unsqueeze(1))
        mae = self.mae_fn(preds, batch['target'].unsqueeze(1))
        self.log('train_loss', loss)
        self.log('train_mae', mae)
        return loss

    def validation_step(self, batch, batch_idx):
        input_dynamic = batch['input_dynamic']
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        preds = self(input_dynamic, batch['input_static'])
        loss = self.loss_fn(preds, batch['target'].unsqueeze(1))
        mae = self.mae_fn(preds, batch['target'].unsqueeze(1))
        self.log('val_loss', loss)
        self.log('val_mae', mae)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
