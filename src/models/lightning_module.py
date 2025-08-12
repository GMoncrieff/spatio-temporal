import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import structural_similarity_index_measure as ssim
from .spatiotemporal_predictor import SpatioTemporalPredictor
import wandb
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class SpatioTemporalLightningModule(pl.LightningModule):
    def __init__(self, hidden_dim: int = 16, lr: float = 1e-3, ssim_weight: float = 0.2):
        super().__init__()
        self.save_hyperparameters()
        self.model = SpatioTemporalPredictor(hidden_dim=hidden_dim)
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.mae_fn = nn.L1Loss(reduction='mean')
        self.lr = lr
        self.ssim_weight = ssim_weight
        # Normalization stats to be set externally
        self.hm_mean = None
        self.hm_std = None

    # Removed AR-specific helpers and panels for single-step setup

    def forward(self, input_dynamic, input_static):
        # Ensure input_dynamic is [B, T, 1, H, W]
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        return self.model(input_dynamic, input_static)

    def training_step(self, batch, batch_idx):
        input_dynamic = batch['input_dynamic']
        input_static = batch['input_static']
        # Replace NaNs in inputs with 0
        input_dynamic = torch.nan_to_num(input_dynamic, nan=0.0)
        input_static = torch.nan_to_num(input_static, nan=0.0)
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        preds = self(input_dynamic, input_static)
        target = batch['target'].unsqueeze(1)
        mask = torch.isfinite(target)
        valid_preds = preds[mask]
        valid_target = target[mask]
        if valid_preds.numel() == 0:
            loss = torch.tensor(0.0, device=preds.device)
            mae = torch.tensor(0.0, device=preds.device)
            ssim_loss = torch.tensor(0.0, device=preds.device)
        else:
            loss = self.loss_fn(valid_preds, valid_target)
            mae = self.mae_fn(valid_preds, valid_target)
            # SSIM expects [B, C, H, W] and values in [0,1]
            preds_masked = preds.clone()
            target_masked = target.clone()
            preds_masked[~mask] = 0.0
            target_masked[~mask] = 0.0
            ssim_val = ssim(preds_masked, target_masked, data_range=1.0)
            ssim_loss = 1.0 - ssim_val
        total_loss = loss + self.ssim_weight * ssim_loss
        self.log('train_loss', loss)
        self.log('train_ssim_loss', ssim_loss)
        self.log('train_total_loss', total_loss)
        return total_loss

    # Removed AR aggregation and epoch-level AR metrics

    def validation_step(self, batch, batch_idx):
        input_dynamic = batch['input_dynamic']
        input_static = batch['input_static']
        # Replace NaNs in inputs with 0
        input_dynamic = torch.nan_to_num(input_dynamic, nan=0.0)
        input_static = torch.nan_to_num(input_static, nan=0.0)
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        preds = self(input_dynamic, input_static)
        target = batch['target'].unsqueeze(1)
        mask = torch.isfinite(target)
        valid_preds = preds[mask]
        valid_target = target[mask]
        # Diagnostics
        num_total = target.numel()
        num_valid = mask.sum().item()
        num_nan = torch.isnan(target).sum().item()
        min_pred = float(preds[mask].min()) if num_valid > 0 else float('nan')
        max_pred = float(preds[mask].max()) if num_valid > 0 else float('nan')
        min_target = float(target[mask].min()) if num_valid > 0 else float('nan')
        max_target = float(target[mask].max()) if num_valid > 0 else float('nan')
        print(f"[VAL] Batch: total={num_total}, valid={num_valid}, nan={num_nan}, pred=[{min_pred:.4f},{max_pred:.4f}], target=[{min_target:.4f},{max_target:.4f}]")
        if valid_preds.numel() == 0:
            print("[VAL] All targets are NaN in this batch!")
            loss = torch.tensor(0.0, device=preds.device)
            mae = torch.tensor(0.0, device=preds.device)
            ssim_loss = torch.tensor(0.0, device=preds.device)
        else:
            loss = self.loss_fn(valid_preds, valid_target)
            # Masked MAE (normalized)
            valid_mask = ~torch.isnan(target)
            mae = F.l1_loss(preds[valid_mask], target[valid_mask])
            self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
            # Physical-scale MAE (original scale, 0-10000)
            if hasattr(self, 'hm_mean') and hasattr(self, 'hm_std'):
                preds_orig = preds[valid_mask] * self.hm_std + self.hm_mean
                target_orig = target[valid_mask] * self.hm_std + self.hm_mean
                mae_orig = F.l1_loss(preds_orig, target_orig)
                self.log('val_mae_original', mae_orig, on_step=False, on_epoch=True, prog_bar=True)
            # SSIM expects [B, C, H, W] and values in [0,1]
            preds_masked = preds.clone()
            target_masked = target.clone()
            preds_masked[~mask] = 0.0
            target_masked[~mask] = 0.0
            ssim_val = ssim(preds_masked, target_masked, data_range=1.0)
            ssim_loss = 1.0 - ssim_val
            print(f"[VAL] Loss: {loss.item():.6f}, MAE: {mae.item():.6f}, SSIM: {ssim_val.item():.6f}")
        total_loss = loss + self.ssim_weight * ssim_loss
        self.log('val_loss', loss)
        self.log('val_ssim_loss', ssim_loss)
        self.log('val_total_loss', total_loss)
        return total_loss
    # Removed AR panel rendering and end-of-fit logging

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
