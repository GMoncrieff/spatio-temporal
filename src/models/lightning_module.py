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
        ssim_weight: float = 2.0,
        laplacian_weight: float = 1.0,
        num_static_channels: int = 1,
        num_dynamic_channels: int = 1,
        num_layers: int = 1,
        kernel_size: int = 3,
        use_location_encoder: bool = True,
        locenc_backbone=("sphericalharmonics", "siren"),
        locenc_hparams=None,
        locenc_out_channels: int = 8,
        histogram_weight: float = 0.67,
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
        self.histogram_lambda_w2 = histogram_lambda_w2  # Store for reference (not used in new implementation)
        if self.histogram_weight > 0:
            # Define histogram bins: [-1, -0.05, -ε, +ε, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
            histogram_bins = torch.tensor([-1.0, -0.05, -0.005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
            self.histogram_loss_fn = HistogramLoss(histogram_bins)  # Bin weights will be set later
            self.register_buffer('histogram_bins', histogram_bins)
            self.histogram_bins_initialized = False

    # Removed AR-specific helpers and panels for single-step setup
    
    def _compute_horizon_losses(self, pred_h, target_h, last_input, mask_h, horizon_name=""):
        """
        Compute all loss components for a single horizon.
        
        Args:
            pred_h: [B, 1, H, W] predictions for this horizon
            target_h: [B, 1, H, W] targets for this horizon
            last_input: [B, 1, H, W] last input timestep
            mask_h: [B, 1, H, W] validity mask
            horizon_name: str, for logging (e.g., "5yr")
            
        Returns:
            dict with keys: mse, mae, ssim, lap, hist, total
        """
        # Deltas
        delta_pred = pred_h - last_input
        delta_true = target_h - last_input
        valid_delta_pred = delta_pred[mask_h]
        valid_delta_true = delta_true[mask_h]
        
        if valid_delta_pred.numel() == 0:
            return {
                'mse': torch.tensor(0.0, device=pred_h.device),
                'mae': torch.tensor(0.0, device=pred_h.device),
                'ssim': torch.tensor(0.0, device=pred_h.device),
                'lap': torch.tensor(0.0, device=pred_h.device),
                'hist': torch.tensor(0.0, device=pred_h.device),
                'total': torch.tensor(0.0, device=pred_h.device)
            }
        
        # MSE on deltas
        mse = self.loss_fn(valid_delta_pred, valid_delta_true)
        
        # MAE on absolute values
        mae = self.mae_fn(pred_h[mask_h], target_h[mask_h])
        
        # SSIM on absolute images
        pred_sanitized = pred_h.clone()
        target_sanitized = target_h.clone()
        pred_sanitized[~mask_h] = 0.0
        target_sanitized[~mask_h] = 0.0
        ssim_val = ssim(pred_sanitized, target_sanitized, data_range=1.0)
        ssim_loss = 1.0 - ssim_val
        
        # Laplacian loss
        lap_loss = self.lap_loss(pred_sanitized, target_sanitized, mask=mask_h)
        
        # Histogram loss
        hist_loss = torch.tensor(0.0, device=pred_h.device)
        if self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs:
            delta_true_2d = delta_true.squeeze(1)
            delta_pred_2d = delta_pred.squeeze(1)
            mask_2d = mask_h.squeeze(1)
            # Extract horizon index from horizon_name (e.g., "5yr" -> 0)
            horizon_map = {'5yr': 0, '10yr': 1, '15yr': 2, '20yr': 3}
            h_idx = horizon_map.get(horizon_name, 0)
            hist_loss, _, _ = self.histogram_loss_fn(delta_true_2d, delta_pred_2d, mask=mask_2d, horizon_idx=h_idx)
        
        # Total loss for this horizon
        total = mse + self.ssim_weight * ssim_loss + self.laplacian_weight * lap_loss
        if self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs:
            total = total + self.histogram_weight * hist_loss
        
        return {
            'mse': mse,
            'mae': mae,
            'ssim': ssim_loss,
            'lap': lap_loss,
            'hist': hist_loss,
            'total': total
        }

    def forward(self, input_dynamic, input_static, lonlat=None):
        # Ensure input_dynamic is [B, T, 1, H, W]
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        return self.model(input_dynamic, input_static, lonlat=lonlat)

    def training_step(self, batch, batch_idx):
        input_dynamic = batch['input_dynamic']
        input_static = batch['input_static']
        # Multi-horizon targets
        targets = {
            '5yr': batch['target_5yr'].unsqueeze(1),
            '10yr': batch['target_10yr'].unsqueeze(1),
            '15yr': batch['target_15yr'].unsqueeze(1),
            '20yr': batch['target_20yr'].unsqueeze(1)
        }
        lonlat = batch.get('lonlat', None)
        if lonlat is not None:
            lonlat = lonlat.to(input_dynamic.device)
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        
        # Compute validity mask from RAW inputs
        dynamic_valid = torch.isfinite(input_dynamic).all(dim=(1, 2), keepdim=True)
        static_valid = torch.isfinite(input_static).all(dim=1, keepdim=True).unsqueeze(1)
        
        # Replace NaNs in inputs with 0 for model forward
        input_dynamic = torch.nan_to_num(input_dynamic, nan=0.0)
        input_static = torch.nan_to_num(input_static, nan=0.0)
        
        # Forward pass: get predictions for all 4 horizons [B, 4, H, W]
        preds_all = self(input_dynamic, input_static, lonlat=lonlat)
        
        # Use last timestep HM channel (0) as baseline
        last_input = input_dynamic[:, -1, 0:1, :, :]  # [B, 1, H, W]
        
        # Compute losses for each horizon
        horizon_names = ['5yr', '10yr', '15yr', '20yr']
        horizon_losses = []
        
        for h_idx, h_name in enumerate(horizon_names):
            # Extract prediction and target for this horizon
            pred_h = preds_all[:, h_idx:h_idx+1, :, :]  # [B, 1, H, W]
            target_h = targets[h_name]  # [B, 1, H, W]
            
            # Compute mask for this horizon
            target_valid = torch.isfinite(target_h)
            mask_h = target_valid & dynamic_valid.squeeze(2) & static_valid.squeeze(2) & torch.isfinite(last_input)
            
            # Set predictions to NaN where inputs were invalid
            pred_h = pred_h.clone()
            pred_h[~mask_h] = float('nan')
            
            # Compute all losses for this horizon
            losses_h = self._compute_horizon_losses(pred_h, target_h, last_input, mask_h, h_name)
            horizon_losses.append(losses_h)
        
        # Average losses across horizons
        avg_mse = torch.stack([h['mse'] for h in horizon_losses]).mean()
        avg_mae = torch.stack([h['mae'] for h in horizon_losses]).mean()
        avg_ssim = torch.stack([h['ssim'] for h in horizon_losses]).mean()
        avg_lap = torch.stack([h['lap'] for h in horizon_losses]).mean()
        avg_hist = torch.stack([h['hist'] for h in horizon_losses]).mean()
        avg_total = torch.stack([h['total'] for h in horizon_losses]).mean()
        
        # Log averaged metrics
        self.log('train_loss', avg_mse)
        self.log('train_ssim_loss', avg_ssim)
        self.log('train_lap_loss', avg_lap)
        self.log('train_hist_loss', avg_hist)
        self.log('train_total_loss', avg_total)
        
        # Debug print on first batch of warmup epoch
        if batch_idx == 0 and self.current_epoch == self.histogram_warmup_epochs and self.histogram_weight > 0:
            print(f"\n[HISTOGRAM ACTIVATED] Epoch {self.current_epoch}: avg_hist_loss={avg_hist.item():.6f}, weighted={self.histogram_weight * avg_hist.item():.6f}\n")
        
        return avg_total

    def validation_step(self, batch, batch_idx):
        input_dynamic = batch['input_dynamic']
        input_static = batch['input_static']
        # Multi-horizon targets
        targets = {
            '5yr': batch['target_5yr'].unsqueeze(1),
            '10yr': batch['target_10yr'].unsqueeze(1),
            '15yr': batch['target_15yr'].unsqueeze(1),
            '20yr': batch['target_20yr'].unsqueeze(1)
        }
        lonlat = batch.get('lonlat', None)
        if lonlat is not None:
            lonlat = lonlat.to(input_dynamic.device)
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        
        # Compute validity mask from RAW inputs
        dynamic_valid = torch.isfinite(input_dynamic).all(dim=(1, 2), keepdim=True)
        static_valid = torch.isfinite(input_static).all(dim=1, keepdim=True).unsqueeze(1)
        
        # Replace NaNs in inputs with 0 for model forward
        input_dynamic = torch.nan_to_num(input_dynamic, nan=0.0)
        input_static = torch.nan_to_num(input_static, nan=0.0)
        
        # Forward pass: get predictions for all 4 horizons [B, 4, H, W]
        preds_all = self(input_dynamic, input_static, lonlat=lonlat)
        
        # Use last timestep HM channel (0) as baseline
        last_input = input_dynamic[:, -1, 0:1, :, :]  # [B, 1, H, W]
        
        # Compute losses for each horizon
        horizon_names = ['5yr', '10yr', '15yr', '20yr']
        horizon_losses = []
        
        for h_idx, h_name in enumerate(horizon_names):
            # Extract prediction and target for this horizon
            pred_h = preds_all[:, h_idx:h_idx+1, :, :]  # [B, 1, H, W]
            target_h = targets[h_name]  # [B, 1, H, W]
            
            # Compute mask for this horizon
            target_valid = torch.isfinite(target_h)
            mask_h = target_valid & dynamic_valid.squeeze(2) & static_valid.squeeze(2) & torch.isfinite(last_input)
            
            # Set predictions to NaN where inputs were invalid
            pred_h = pred_h.clone()
            pred_h[~mask_h] = float('nan')
            
            # Compute all losses for this horizon
            losses_h = self._compute_horizon_losses(pred_h, target_h, last_input, mask_h, h_name)
            horizon_losses.append(losses_h)
        
        # Average losses across horizons
        avg_mse = torch.stack([h['mse'] for h in horizon_losses]).mean()
        avg_mae = torch.stack([h['mae'] for h in horizon_losses]).mean()
        avg_ssim = torch.stack([h['ssim'] for h in horizon_losses]).mean()
        avg_lap = torch.stack([h['lap'] for h in horizon_losses]).mean()
        avg_hist = torch.stack([h['hist'] for h in horizon_losses]).mean()
        avg_total = torch.stack([h['total'] for h in horizon_losses]).mean()
        
        # Log averaged metrics
        self.log('val_loss', avg_mse, on_step=False, on_epoch=True)
        self.log('val_mae', avg_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ssim_loss', avg_ssim, on_step=False, on_epoch=True)
        self.log('val_lap_loss', avg_lap, on_step=False, on_epoch=True)
        self.log('val_hist_loss', avg_hist, on_step=False, on_epoch=True)
        self.log('val_total_loss', avg_total, on_step=False, on_epoch=True, prog_bar=True)
        
        # Print validation metrics (only for 20yr horizon for brevity)
        if batch_idx == 0:
            losses_20yr = horizon_losses[-1]  # Last horizon is 20yr
            hist_active = self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs
            hist_str = f", Hist: {losses_20yr['hist'].item():.5f}" if hist_active else " [hist off]"
            print(f"[VAL] 20yr - MSE: {losses_20yr['mse'].item():.5f}, MAE: {losses_20yr['mae'].item():.5f}, SSIM: {1.0 - losses_20yr['ssim'].item():.5f}, Lap: {losses_20yr['lap'].item():.5f}{hist_str}")
            print(f"[VAL] Avg across horizons - Total: {avg_total.item():.5f}")
        
        return avg_total
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
