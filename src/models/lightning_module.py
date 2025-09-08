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
    def __init__(
        self,
        hidden_dim: int = 16,
        lr: float = 1e-3,
        ssim_weight: float = 0.2,
        num_static_channels: int = 1,
        num_dynamic_channels: int = 1,
        num_layers: int = 1,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SpatioTemporalPredictor(
            hidden_dim=hidden_dim,
            num_static_channels=num_static_channels,
            num_dynamic_channels=num_dynamic_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )
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
        # Use last timestep HM channel (0) as baseline to form deltas
        # input_dynamic shape: [B, T, C_d, H, W], HM is channel 0
        last_input = input_dynamic[:, -1, 0:1, :, :]  # [B, 1, H, W]
        # Valid where both target and last_input are finite
        mask = torch.isfinite(target) & torch.isfinite(last_input)
        # Deltas in normalized/native scale
        delta_pred = preds - last_input
        delta_true = target - last_input
        valid_delta_pred = delta_pred[mask]
        valid_delta_true = delta_true[mask]
        if valid_delta_pred.numel() == 0:
            loss = torch.tensor(0.0, device=preds.device)
            mae = torch.tensor(0.0, device=preds.device)
            ssim_loss = torch.tensor(0.0, device=preds.device)
        else:
            # MSE on delta (prediction - last input) vs (target - last input)
            loss = self.loss_fn(valid_delta_pred, valid_delta_true)
            # MAE reported on absolute prediction vs target for reference
            mae = self.mae_fn(preds[mask], target[mask])
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
        last_input = input_dynamic[:, -1, 0:1, :, :]  # [B, 1, H, W]
        mask = torch.isfinite(target) & torch.isfinite(last_input)
        delta_pred = preds - last_input
        delta_true = target - last_input
        valid_delta_pred = delta_pred[mask]
        valid_delta_true = delta_true[mask]
        # Diagnostics
        num_total = target.numel()
        num_valid = mask.sum().item()
        num_nan = torch.isnan(target).sum().item()
        min_pred = float(preds[mask].min()) if num_valid > 0 else float('nan')
        max_pred = float(preds[mask].max()) if num_valid > 0 else float('nan')
        min_target = float(target[mask].min()) if num_valid > 0 else float('nan')
        max_target = float(target[mask].max()) if num_valid > 0 else float('nan')
        print(f"[VAL] Batch: total={num_total}, valid={num_valid}, nan={num_nan}, pred=[{min_pred:.4f},{max_pred:.4f}], target=[{min_target:.4f},{max_target:.4f}]")
        if valid_delta_pred.numel() == 0:
            print("[VAL] All targets are NaN in this batch!")
            loss = torch.tensor(0.0, device=preds.device)
            mae = torch.tensor(0.0, device=preds.device)
            ssim_loss = torch.tensor(0.0, device=preds.device)
        else:
            # Delta MSE loss
            loss = self.loss_fn(valid_delta_pred, valid_delta_true)
            # Masked MAE (normalized) on absolute prediction
            valid_mask = mask
            mae = F.l1_loss(preds[valid_mask], target[valid_mask])
            self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
            # Physical-scale MAE (0-10000): backtransform to original 0-1 scale, then *10000
            if (self.hm_mean is not None) and (self.hm_std is not None):
                preds_bt = preds[valid_mask] * self.hm_std + self.hm_mean
                target_bt = target[valid_mask] * self.hm_std + self.hm_mean
                mae_orig = F.l1_loss(preds_bt * 10000.0, target_bt * 10000.0)
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
