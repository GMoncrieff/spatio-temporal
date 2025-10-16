import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import structural_similarity_index_measure as ssim
from .spatiotemporal_predictor import SpatioTemporalPredictor
from .losses import LaplacianPyramidLoss
from .histogram_loss import HistogramLoss
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
        ssim_weight: float = 0.1,
        laplacian_weight: float = 0.2,
        num_static_channels: int = 1,
        num_dynamic_channels: int = 1,
        num_layers: int = 1,
        kernel_size: int = 3,
        use_location_encoder: bool = True,
        locenc_backbone=("sphericalharmonics", "siren"),
        locenc_hparams=None,
        locenc_out_channels: int = 8,
        histogram_weight: float = 0.0,
        histogram_lambda_w2: float = 0.1,
        histogram_warmup_epochs: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SpatioTemporalPredictor(
            hidden_dim=hidden_dim,
            num_static_channels=num_static_channels,
            num_dynamic_channels=num_dynamic_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            use_location_encoder=use_location_encoder,
            locenc_backbone=locenc_backbone,
            locenc_hparams=locenc_hparams,
            locenc_out_channels=locenc_out_channels,
        )
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.mae_fn = nn.L1Loss(reduction='mean')
        self.lr = lr
        self.ssim_weight = ssim_weight
        self.laplacian_weight = laplacian_weight
        # 3-level Laplacian pyramid by default
        self.lap_loss = LaplacianPyramidLoss(levels=3, kernel_size=5, sigma=1.0, include_lowpass=True)
        # Normalization stats to be set externally
        self.hm_mean = None
        self.hm_std = None
        # Histogram loss for pixel-level change distributions
        self.histogram_weight = histogram_weight
        self.histogram_warmup_epochs = histogram_warmup_epochs
        if self.histogram_weight > 0:
            # Define histogram bins: [-1, -0.05, -ε, +ε, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
            histogram_bins = torch.tensor([-1.0, -0.05, -0.005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
            self.histogram_loss_fn = HistogramLoss(histogram_bins, lambda_w2=histogram_lambda_w2)
            self.register_buffer('histogram_bins', histogram_bins)

    # Removed AR-specific helpers and panels for single-step setup

    def forward(self, input_dynamic, input_static, lonlat=None):
        # Ensure input_dynamic is [B, T, 1, H, W]
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        return self.model(input_dynamic, input_static, lonlat=lonlat)

    def training_step(self, batch, batch_idx):
        input_dynamic = batch['input_dynamic']
        input_static = batch['input_static']
        target = batch['target'].unsqueeze(1)
        lonlat = batch.get('lonlat', None)
        if lonlat is not None:
            lonlat = lonlat.to(input_dynamic.device)
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        
        # Compute validity mask from RAW inputs: require finite target and all input channels
        target_valid = torch.isfinite(target)
        dynamic_valid = torch.isfinite(input_dynamic).all(dim=(1, 2), keepdim=True)
        static_valid = torch.isfinite(input_static).all(dim=1, keepdim=True).unsqueeze(1)
        input_mask = target_valid & dynamic_valid.squeeze(2) & static_valid.squeeze(2)
        
        # Replace NaNs in inputs with 0 for model forward
        input_dynamic = torch.nan_to_num(input_dynamic, nan=0.0)
        input_static = torch.nan_to_num(input_static, nan=0.0)
        
        preds = self(input_dynamic, input_static, lonlat=lonlat)
        
        # Set predictions to NaN where any input was NaN
        preds[~input_mask] = float('nan')
        
        # Use last timestep HM channel (0) as baseline to form deltas
        last_input = input_dynamic[:, -1, 0:1, :, :]
        mask = input_mask & torch.isfinite(last_input)
        
        # Deltas using predictions (which are NaN where inputs were invalid)
        delta_pred = preds - last_input
        delta_true = target - last_input
        valid_delta_pred = delta_pred[mask]
        valid_delta_true = delta_true[mask]
        
        if valid_delta_pred.numel() == 0:
            loss = torch.tensor(0.0, device=preds.device)
            mae = torch.tensor(0.0, device=preds.device)
            ssim_loss = torch.tensor(0.0, device=preds.device)
            lap_loss = torch.tensor(0.0, device=preds.device)
            hist_loss = torch.tensor(0.0, device=preds.device)
        else:
            loss = self.loss_fn(valid_delta_pred, valid_delta_true)
            mae = self.mae_fn(preds[mask], target[mask])
            # SSIM expects [B, C, H, W] and values in [0,1]
            preds_sanitized = preds.clone()
            target_sanitized = target.clone()
            preds_sanitized[~mask] = 0.0
            target_sanitized[~mask] = 0.0
            ssim_val = ssim(preds_sanitized, target_sanitized, data_range=1.0)
            ssim_loss = 1.0 - ssim_val
            # Laplacian Pyramid multi-scale L1 on absolute images with masking
            lap_loss = self.lap_loss(preds_sanitized, target_sanitized, mask=mask)
            
            # Histogram loss on pixel-level change distributions
            hist_loss = torch.tensor(0.0, device=preds.device)
            if self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs:
                # Squeeze deltas to [B, H, W]
                delta_true_2d = delta_true.squeeze(1)
                delta_pred_2d = delta_pred.squeeze(1)
                mask_2d = mask.squeeze(1)
                
                hist_total, hist_ce, hist_w2, p_obs, p_pred = self.histogram_loss_fn(
                    delta_true_2d, delta_pred_2d, mask=mask_2d
                )
                hist_loss = hist_total
                self.log('train_hist_ce', hist_ce)
                self.log('train_hist_w2', hist_w2)
        
        total_loss = loss + self.ssim_weight * ssim_loss + self.laplacian_weight * lap_loss
        if self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs:
            total_loss = total_loss + self.histogram_weight * hist_loss
        
        self.log('train_loss', loss)
        self.log('train_ssim_loss', ssim_loss)
        self.log('train_lap_loss', lap_loss)
        self.log('train_hist_loss', hist_loss)
        self.log('train_total_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        input_dynamic = batch['input_dynamic']
        input_static = batch['input_static']
        target = batch['target'].unsqueeze(1)
        lonlat = batch.get('lonlat', None)
        if lonlat is not None:
            lonlat = lonlat.to(input_dynamic.device)
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        
        # Compute validity mask from RAW inputs
        target_valid = torch.isfinite(target)
        dynamic_valid = torch.isfinite(input_dynamic).all(dim=(1, 2), keepdim=True)
        static_valid = torch.isfinite(input_static).all(dim=1, keepdim=True).unsqueeze(1)
        input_mask = target_valid & dynamic_valid.squeeze(2) & static_valid.squeeze(2)
        
        # Replace NaNs in inputs with 0 for model forward
        input_dynamic = torch.nan_to_num(input_dynamic, nan=0.0)
        input_static = torch.nan_to_num(input_static, nan=0.0)
        
        preds = self(input_dynamic, input_static, lonlat=lonlat)
        
        # Set predictions to NaN where any input was NaN
        preds[~input_mask] = float('nan')
        
        last_input = input_dynamic[:, -1, 0:1, :, :]
        mask = input_mask & torch.isfinite(last_input)
        
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
            lap_loss = torch.tensor(0.0, device=preds.device)
            hist_loss = torch.tensor(0.0, device=preds.device)
        else:
            loss = self.loss_fn(valid_delta_pred, valid_delta_true)
            mae = F.l1_loss(preds[mask], target[mask])
            self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
            # Physical-scale MAE (0-10000): backtransform to 0-1 scale, then scale to 0-10000 for interpretability
            if (self.hm_mean is not None) and (self.hm_std is not None):
                preds_bt = preds[mask] * self.hm_std + self.hm_mean
                target_bt = target[mask] * self.hm_std + self.hm_mean
                mae_orig = F.l1_loss(preds_bt * 10000.0, target_bt * 10000.0)
                self.log('val_mae_original', mae_orig, on_step=False, on_epoch=True, prog_bar=True)
            # SSIM expects [B, C, H, W] and values in [0,1]
            preds_sanitized = preds.clone()
            target_sanitized = target.clone()
            preds_sanitized[~mask] = 0.0
            target_sanitized[~mask] = 0.0
            ssim_val = ssim(preds_sanitized, target_sanitized, data_range=1.0)
            ssim_loss = 1.0 - ssim_val
            # Laplacian Pyramid multi-scale L1
            lap_loss = self.lap_loss(preds_sanitized, target_sanitized, mask=mask)
            
            # Histogram loss on pixel-level change distributions
            hist_loss = torch.tensor(0.0, device=preds.device)
            if self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs:
                # Squeeze deltas to [B, H, W]
                delta_true_2d = delta_true.squeeze(1)
                delta_pred_2d = delta_pred.squeeze(1)
                mask_2d = mask.squeeze(1)
                
                hist_total, hist_ce, hist_w2, p_obs, p_pred = self.histogram_loss_fn(
                    delta_true_2d, delta_pred_2d, mask=mask_2d
                )
                hist_loss = hist_total
                self.log('val_hist_ce', hist_ce, on_step=False, on_epoch=True)
                self.log('val_hist_w2', hist_w2, on_step=False, on_epoch=True)
            
            print(f"[VAL] Loss: {loss.item():.6f}, MAE: {mae.item():.6f}, SSIM: {ssim_val.item():.6f}")
        
        total_loss = loss + self.ssim_weight * ssim_loss + self.laplacian_weight * lap_loss
        if self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs:
            total_loss = total_loss + self.histogram_weight * hist_loss
        
        self.log('val_loss', loss)
        self.log('val_ssim_loss', ssim_loss)
        self.log('val_lap_loss', lap_loss)
        self.log('val_hist_loss', hist_loss, on_step=False, on_epoch=True)
        self.log('val_total_loss', total_loss)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
