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
    
    def _compute_horizon_losses(self, pred_change, target_h, last_input, mask_h, horizon_name=""):
        """
        Compute all loss components for a single horizon.
        
        Args:
            pred_change: [B, 1, H, W] PREDICTED CHANGES (model now predicts deltas directly)
            target_h: [B, 1, H, W] target absolute HM values
            last_input: [B, 1, H, W] last input timestep (absolute HM)
            mask_h: [B, 1, H, W] validity mask
            horizon_name: str, for logging (e.g., "5yr")
            
        Returns:
            dict with keys: mse, mae, ssim, lap, hist, total (unweighted), and weighted versions
        """
        # True changes
        delta_true = target_h - last_input
        valid_delta_pred = pred_change[mask_h]
        valid_delta_true = delta_true[mask_h]
        
        if valid_delta_pred.numel() == 0:
            zero = torch.tensor(0.0, device=pred_change.device)
            return {
                'mse': zero,
                'mae': zero,
                'ssim': zero,
                'ssim_weighted': zero,
                'lap': zero,
                'lap_weighted': zero,
                'hist': zero,
                'hist_weighted': zero,
                'total': zero
            }
        
        # MSE on changes (deltas)
        mse = self.loss_fn(valid_delta_pred, valid_delta_true)
        
        # MAE on absolute values (monitoring only - not in loss)
        # Reconstruct absolute predictions: pred_absolute = last_input + pred_change
        pred_absolute = last_input + pred_change
        # Clip to [0, 1] range
        pred_absolute = torch.clamp(pred_absolute, 0.0, 1.0)
        mae = self.mae_fn(pred_absolute[mask_h], target_h[mask_h])
        
        # SSIM on CHANGES (NEW: was on absolute, now on changes)
        pred_change_sanitized = pred_change.clone()
        delta_true_sanitized = delta_true.clone()
        pred_change_sanitized[~mask_h] = 0.0
        delta_true_sanitized[~mask_h] = 0.0
        # For changes, use data_range=2.0 (changes can be -1 to +1)
        ssim_val = ssim(pred_change_sanitized, delta_true_sanitized, data_range=2.0)
        ssim_loss = 1.0 - ssim_val
        
        # Laplacian loss on CHANGES (NEW: was on absolute, now on changes)
        lap_loss = self.lap_loss(pred_change_sanitized, delta_true_sanitized, mask=mask_h)
        
        # Histogram loss on changes
        hist_loss = torch.tensor(0.0, device=pred_change.device)
        if self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs:
            delta_true_2d = delta_true.squeeze(1)
            pred_change_2d = pred_change.squeeze(1)
            mask_2d = mask_h.squeeze(1)
            # Extract horizon index from horizon_name (e.g., "5yr" -> 0)
            horizon_map = {'5yr': 0, '10yr': 1, '15yr': 2, '20yr': 3}
            h_idx = horizon_map.get(horizon_name, 0)
            hist_loss, _, _ = self.histogram_loss_fn(delta_true_2d, pred_change_2d, mask=mask_2d, horizon_idx=h_idx)
        
        # Total loss for this horizon (weighted)
        total = mse + self.ssim_weight * ssim_loss + self.laplacian_weight * lap_loss
        if self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs:
            total = total + self.histogram_weight * hist_loss
        
        return {
            'mse': mse,
            'mae': mae,
            'ssim': ssim_loss,  # unweighted
            'ssim_weighted': self.ssim_weight * ssim_loss,  # weighted
            'lap': lap_loss,  # unweighted
            'lap_weighted': self.laplacian_weight * lap_loss,  # weighted
            'hist': hist_loss,  # unweighted
            'hist_weighted': self.histogram_weight * hist_loss if (self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs) else torch.tensor(0.0, device=pred_change.device),  # weighted
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
        # Note: 0 in normalized space = mean in original space (mean imputation)
        input_dynamic = torch.nan_to_num(input_dynamic, nan=0.0)
        input_static = torch.nan_to_num(input_static, nan=0.0)
        
        # Forward pass: get PREDICTED CHANGES for all 4 horizons [B, 4, H, W]
        # MODEL NOW PREDICTS CHANGES DIRECTLY
        pred_changes_all = self(input_dynamic, input_static, lonlat=lonlat)
        
        # Use last timestep HM channel (0) as baseline
        last_input = input_dynamic[:, -1, 0:1, :, :]  # [B, 1, H, W]
        
        # Compute losses for each horizon
        horizon_names = ['5yr', '10yr', '15yr', '20yr']
        horizon_losses = []
        
        for h_idx, h_name in enumerate(horizon_names):
            # Extract PREDICTED CHANGE and target for this horizon
            pred_change = pred_changes_all[:, h_idx:h_idx+1, :, :]  # [B, 1, H, W] - CHANGE
            target_h = targets[h_name]  # [B, 1, H, W] - ABSOLUTE HM
            
            # Compute mask for this horizon
            target_valid = torch.isfinite(target_h)
            mask_h = target_valid & dynamic_valid.squeeze(2) & static_valid.squeeze(2) & torch.isfinite(last_input)
            
            # Set predicted changes to NaN where inputs were invalid
            pred_change = pred_change.clone()
            pred_change[~mask_h] = float('nan')
            
            # Compute all losses for this horizon (pred_change, not pred_absolute)
            losses_h = self._compute_horizon_losses(pred_change, target_h, last_input, mask_h, h_name)
            horizon_losses.append(losses_h)
        
        # Log per-horizon metrics (unweighted and weighted)
        horizon_suffixes = ['5yr', '10yr', '15yr', '20yr']
        for h_name, losses_h in zip(horizon_suffixes, horizon_losses):
            self.log(f'train_mae_{h_name}', losses_h['mae'])
            self.log(f'train_ssim_{h_name}', losses_h['ssim'])  # unweighted
            self.log(f'train_ssim_weighted_{h_name}', losses_h['ssim_weighted'])
            self.log(f'train_lap_{h_name}', losses_h['lap'])  # unweighted
            self.log(f'train_lap_weighted_{h_name}', losses_h['lap_weighted'])
            self.log(f'train_hist_{h_name}', losses_h['hist'])  # unweighted
            self.log(f'train_hist_weighted_{h_name}', losses_h['hist_weighted'])
        
        # Average losses across horizons
        avg_mse = torch.stack([h['mse'] for h in horizon_losses]).mean()
        avg_mae = torch.stack([h['mae'] for h in horizon_losses]).mean()
        avg_ssim = torch.stack([h['ssim'] for h in horizon_losses]).mean()
        avg_ssim_weighted = torch.stack([h['ssim_weighted'] for h in horizon_losses]).mean()
        avg_lap = torch.stack([h['lap'] for h in horizon_losses]).mean()
        avg_lap_weighted = torch.stack([h['lap_weighted'] for h in horizon_losses]).mean()
        avg_hist = torch.stack([h['hist'] for h in horizon_losses]).mean()
        avg_hist_weighted = torch.stack([h['hist_weighted'] for h in horizon_losses]).mean()
        avg_total = torch.stack([h['total'] for h in horizon_losses]).mean()
        
        # Log averaged metrics (both unweighted and weighted)
        self.log('train_loss', avg_mse)
        self.log('train_mae', avg_mae, prog_bar=True)
        self.log('train_ssim', avg_ssim, prog_bar=True)  # unweighted
        self.log('train_ssim_weighted', avg_ssim_weighted)
        self.log('train_lap', avg_lap)  # unweighted
        self.log('train_lap_weighted', avg_lap_weighted)
        self.log('train_hist', avg_hist)  # unweighted
        self.log('train_hist_weighted', avg_hist_weighted)
        self.log('train_total_loss', avg_total, prog_bar=True)
        
        # Print detailed loss breakdown to console (every 50 batches)
        if batch_idx % 50 == 0:
            print(f"\n[Train] Batch {batch_idx}:")
            print(f"  MSE: {avg_mse:.6f}")
            print(f"  MAE: {avg_mae:.6f} (monitoring only)")
            print(f"  SSIM: {avg_ssim:.6f} (×{self.ssim_weight}) = {avg_ssim_weighted:.6f}")
            print(f"  Laplacian: {avg_lap:.6f} (×{self.laplacian_weight}) = {avg_lap_weighted:.6f}")
            print(f"  Histogram: {avg_hist:.6f} (×{self.histogram_weight}) = {avg_hist_weighted:.6f}")
            print(f"  Total Loss: {avg_total:.6f}")
        
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
        # Note: 0 in normalized space = mean in original space (mean imputation)
        input_dynamic = torch.nan_to_num(input_dynamic, nan=0.0)
        input_static = torch.nan_to_num(input_static, nan=0.0)
        
        # Forward pass: get PREDICTED CHANGES for all 4 horizons [B, 4, H, W]
        # MODEL NOW PREDICTS CHANGES DIRECTLY
        pred_changes_all = self(input_dynamic, input_static, lonlat=lonlat)
        
        # Use last timestep HM channel (0) as baseline
        last_input = input_dynamic[:, -1, 0:1, :, :]  # [B, 1, H, W]
        
        # Compute losses for each horizon
        horizon_names = ['5yr', '10yr', '15yr', '20yr']
        horizon_losses = []
        
        for h_idx, h_name in enumerate(horizon_names):
            # Extract PREDICTED CHANGE and target for this horizon
            pred_change = pred_changes_all[:, h_idx:h_idx+1, :, :]  # [B, 1, H, W] - CHANGE
            target_h = targets[h_name]  # [B, 1, H, W] - ABSOLUTE HM
            
            # Compute mask for this horizon
            target_valid = torch.isfinite(target_h)
            mask_h = target_valid & dynamic_valid.squeeze(2) & static_valid.squeeze(2) & torch.isfinite(last_input)
            
            # Set predicted changes to NaN where inputs were invalid
            pred_change = pred_change.clone()
            pred_change[~mask_h] = float('nan')
            
            # Compute all losses for this horizon (pred_change, not pred_absolute)
            losses_h = self._compute_horizon_losses(pred_change, target_h, last_input, mask_h, h_name)
            horizon_losses.append(losses_h)
        
        # Log per-horizon metrics (unweighted and weighted)
        horizon_suffixes = ['5yr', '10yr', '15yr', '20yr']
        for h_name, losses_h in zip(horizon_suffixes, horizon_losses):
            self.log(f'val_mae_{h_name}', losses_h['mae'], on_step=False, on_epoch=True)
            self.log(f'val_ssim_{h_name}', losses_h['ssim'], on_step=False, on_epoch=True)  # unweighted
            self.log(f'val_ssim_weighted_{h_name}', losses_h['ssim_weighted'], on_step=False, on_epoch=True)
            self.log(f'val_lap_{h_name}', losses_h['lap'], on_step=False, on_epoch=True)  # unweighted
            self.log(f'val_lap_weighted_{h_name}', losses_h['lap_weighted'], on_step=False, on_epoch=True)
            self.log(f'val_hist_{h_name}', losses_h['hist'], on_step=False, on_epoch=True)  # unweighted
            self.log(f'val_hist_weighted_{h_name}', losses_h['hist_weighted'], on_step=False, on_epoch=True)
        
        # Average losses across horizons
        avg_mse = torch.stack([h['mse'] for h in horizon_losses]).mean()
        avg_mae = torch.stack([h['mae'] for h in horizon_losses]).mean()
        avg_ssim = torch.stack([h['ssim'] for h in horizon_losses]).mean()
        avg_ssim_weighted = torch.stack([h['ssim_weighted'] for h in horizon_losses]).mean()
        avg_lap = torch.stack([h['lap'] for h in horizon_losses]).mean()
        avg_lap_weighted = torch.stack([h['lap_weighted'] for h in horizon_losses]).mean()
        avg_hist = torch.stack([h['hist'] for h in horizon_losses]).mean()
        avg_hist_weighted = torch.stack([h['hist_weighted'] for h in horizon_losses]).mean()
        avg_total = torch.stack([h['total'] for h in horizon_losses]).mean()
        
        # Log averaged metrics (both unweighted and weighted)
        self.log('val_loss', avg_mse, on_step=False, on_epoch=True)
        self.log('val_mae', avg_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ssim', avg_ssim, on_step=False, on_epoch=True)  # unweighted
        self.log('val_ssim_weighted', avg_ssim_weighted, on_step=False, on_epoch=True)
        self.log('val_lap', avg_lap, on_step=False, on_epoch=True)  # unweighted
        self.log('val_lap_weighted', avg_lap_weighted, on_step=False, on_epoch=True)
        self.log('val_hist', avg_hist, on_step=False, on_epoch=True)  # unweighted
        self.log('val_hist_weighted', avg_hist_weighted, on_step=False, on_epoch=True)
        self.log('val_total_loss', avg_total, on_step=False, on_epoch=True, prog_bar=True)
        
        # Print validation metrics with weighted and unweighted losses
        if batch_idx == 0:
            print(f"\n[Validation] Epoch {self.current_epoch}:")
            print(f"  MSE: {avg_mse:.6f}")
            print(f"  MAE: {avg_mae:.6f} (monitoring only)")
            print(f"  SSIM: {avg_ssim:.6f} (×{self.ssim_weight}) = {avg_ssim_weighted:.6f}")
            print(f"  Laplacian: {avg_lap:.6f} (×{self.laplacian_weight}) = {avg_lap_weighted:.6f}")
            print(f"  Histogram: {avg_hist:.6f} (×{self.histogram_weight}) = {avg_hist_weighted:.6f}")
            print(f"  Total Loss: {avg_total:.6f}")
        
        return avg_total
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
