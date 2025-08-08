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
    def __init__(self, hidden_dim: int = 16, lr: float = 1e-3, ssim_weight: float = 0.2, forecast_horizon: int = 4, num_panel_samples: int = 7):
        super().__init__()
        self.save_hyperparameters()
        self.model = SpatioTemporalPredictor(hidden_dim=hidden_dim)
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.mae_fn = nn.L1Loss(reduction='mean')
        self.lr = lr
        self.ssim_weight = ssim_weight
        self.forecast_horizon = int(forecast_horizon)
        self.num_panel_samples = int(num_panel_samples)
        # Normalization stats to be set externally
        self.hm_mean = None
        self.hm_std = None
        # Buffers for post-training logging (collect multiple samples)
        self._val_samples = []  # list of dicts with input_dynamic, input_static, target1, future_targets
        # Fixed visualization scale for physical HM (must match other panels)
        self.hm_vmin = 0.0
        self.hm_vmax = 10000.0
        # Epoch aggregation for AR metrics
        self._ar_aggr = None

    def _to_colored_image(self, tensor_2d: torch.Tensor) -> np.ndarray:
        """Convert a 2D tensor (H, W) in physical scale to RGB uint8 using turbo colormap.
        Assumes values are in physical units; clips to [hm_vmin, hm_vmax]."""
        arr = tensor_2d.detach().cpu().numpy().astype(np.float32)
        arr = np.clip(arr, self.hm_vmin, self.hm_vmax)
        # Normalize to 0..1
        norm = (arr - self.hm_vmin) / max(self.hm_vmax - self.hm_vmin, 1e-6)
        cmap = cm.get_cmap('turbo')
        rgb = cmap(norm)[..., :3]  # drop alpha
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        return rgb_uint8

    def _is_current_epoch_best(self) -> bool:
        """Check if current epoch produced the best model per ModelCheckpoint."""
        # Find first ModelCheckpoint callback, if any
        for cb in self.trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                monitor = cb.monitor
                if monitor is None:
                    return False
                current = self.trainer.callback_metrics.get(monitor)
                # current can be a tensor; convert to float
                if current is None:
                    return False
                current_val = float(current.detach().cpu().item()) if torch.is_tensor(current) else float(current)
                best_val = cb.best_model_score
                if best_val is None:
                    return False
                best_val = float(best_val.detach().cpu().item()) if torch.is_tensor(best_val) else float(best_val)
                mode = getattr(cb, 'mode', 'min')
                if mode == 'min':
                    return np.isclose(current_val, best_val) or (current_val <= best_val)
                else:
                    return np.isclose(current_val, best_val) or (current_val >= best_val)
        return False

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

    def _init_ar_aggr(self, K: int):
        self._ar_aggr = {
            'mae_sum': [0.0 for _ in range(K)],
            'mae_cnt': [0 for _ in range(K)],
            'mae_orig_sum': [0.0 for _ in range(K)],
            'mae_orig_cnt': [0 for _ in range(K)],
            'ssim_sum': [0.0 for _ in range(K)],
            'ssim_cnt': [0 for _ in range(K)],
            'valid_px_sum': [0 for _ in range(K)],
            'K': K,
        }

    def on_validation_epoch_start(self):
        # Reset aggregators for AR metrics
        self._init_ar_aggr(self.forecast_horizon)

    def on_validation_epoch_end(self):
        # Log aggregated AR metrics once per epoch
        if self._ar_aggr is None:
            return
        K = self._ar_aggr['K']
        for idx in range(K):
            h = idx + 1
            # Normalized MAE
            if self._ar_aggr['mae_cnt'][idx] > 0:
                mae_val = self._ar_aggr['mae_sum'][idx] / max(1, self._ar_aggr['mae_cnt'][idx])
            else:
                mae_val = 0.0
            self.log(f'val_mae_h{h}', mae_val, on_step=False, on_epoch=True)
            # Original-scale MAE
            if self._ar_aggr['mae_orig_cnt'][idx] > 0:
                mae_orig_val = self._ar_aggr['mae_orig_sum'][idx] / max(1, self._ar_aggr['mae_orig_cnt'][idx])
            else:
                mae_orig_val = 0.0
            self.log(f'val_mae_original_h{h}', mae_orig_val, on_step=False, on_epoch=True)
            # SSIM
            if self._ar_aggr['ssim_cnt'][idx] > 0:
                ssim_val = self._ar_aggr['ssim_sum'][idx] / max(1, self._ar_aggr['ssim_cnt'][idx])
            else:
                ssim_val = 0.0
            self.log(f'val_ssim_h{h}', ssim_val, on_step=False, on_epoch=True)
            # Valid pixels
            self.log(f'val_valid_px_h{h}', int(self._ar_aggr['valid_px_sum'][idx]), on_step=False, on_epoch=True)

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
        # --- Autoregressive rollout for validation (multi-horizon) ---
        # The model is trained one-step, but here we roll forward using its own predictions
        if 'future_targets' in batch:
            future_targets = batch['future_targets']  # [B, H_ft, H, W] with NaN padding (typically horizons >=2)
            # Always evaluate up to requested forecast horizon; targets selected per-horizon below
            K = int(self.forecast_horizon)
            # Also get how many are actually valid for this batch (optional, batched tensor)
            actual = batch.get('future_horizons', None) if isinstance(batch, dict) else None
            # Prepare sequence tensor for rolling: [B, T, 1, Hc, Wc]
            seq = input_dynamic  # already [B, T, 1, Hc, Wc]
            # For metrics only; collect first batch tensors for post-fit rendering
            first_sample_preds = []  # keep for potential debugging; panel will be recomputed post-fit
            first_sample_tgts = []
            for h in range(1, K + 1):
                # Predict next step
                pred_next = self(seq, input_static)  # [B, 1, Hc, Wc]
                # Compare to ground truth at horizon h
                if h == 1:
                    tgt_h = target  # immediate next step ground truth
                else:
                    idx_ft = h - 2
                    if future_targets.dim() >= 4 and idx_ft < future_targets.shape[1]:
                        tgt_h = future_targets[:, idx_ft].unsqueeze(1)
                    else:
                        tgt_h = torch.full_like(target, torch.nan)
                # Valid if finite and (optionally) within actual horizons
                mask_h = torch.isfinite(tgt_h)
                valid_px = int(mask_h.sum().item())
                # Initialize aggregators if needed
                if self._ar_aggr is None:
                    self._init_ar_aggr(K)
                idx = h - 1
                # Update valid pixel count
                self._ar_aggr['valid_px_sum'][idx] += valid_px
                if mask_h.any():
                    # MAE sum/count in normalized space
                    abs_err = torch.abs(pred_next - tgt_h)
                    mae_sum = abs_err[mask_h].sum().item()
                    self._ar_aggr['mae_sum'][idx] += float(mae_sum)
                    self._ar_aggr['mae_cnt'][idx] += int(valid_px)
                    # Original scale if stats available
                    if (self.hm_mean is not None) and (self.hm_std is not None):
                        pred_phys = pred_next * self.hm_std + self.hm_mean
                        tgt_phys = tgt_h * self.hm_std + self.hm_mean
                        abs_err_phys = torch.abs(pred_phys - tgt_phys)
                        mae_phys_sum = abs_err_phys[mask_h].sum().item()
                        self._ar_aggr['mae_orig_sum'][idx] += float(mae_phys_sum)
                        self._ar_aggr['mae_orig_cnt'][idx] += int(valid_px)
                    # SSIM
                    preds_masked = pred_next.clone()
                    target_masked = tgt_h.clone()
                    preds_masked[~mask_h] = 0.0
                    target_masked[~mask_h] = 0.0
                    ssim_val_h = ssim(preds_masked, target_masked, data_range=1.0)
                    self._ar_aggr['ssim_sum'][idx] += float(ssim_val_h.item())
                    self._ar_aggr['ssim_cnt'][idx] += 1
                # Update sequence for next rollout step
                seq = torch.cat([seq[:, 1:], pred_next.unsqueeze(1)], dim=1)
                # Collect first-sample tensors for later logging
                first_sample_preds.append(pred_next[0, 0].detach().cpu())
                first_sample_tgts.append(tgt_h[0, 0].detach().cpu())

            # Collect up to num_panel_samples validation samples for post-training panel rendering
            if len(self._val_samples) < self.num_panel_samples:
                B = input_dynamic.shape[0]
                for b in range(B):
                    if len(self._val_samples) >= self.num_panel_samples:
                        break
                    sample_dict = {
                        'input_dynamic': input_dynamic[b].detach().cpu(),  # [T,1,H,W]
                        'input_static': input_static[b].detach().cpu(),    # [1,H,W]
                        'target1': target[b, 0].detach().cpu(),            # [H,W]
                        'future_targets': future_targets[b].detach().cpu() if 'future_targets' in batch else None  # [K,H,W]
                    }
                    self._val_samples.append(sample_dict)

        return total_loss

    def _render_panel(self, preds_phys: list, tgts_phys: list, valid_px_per_h: list) -> np.ndarray:
        """Render a labeled grid panel with rows=horizons 1..K and cols=[Target, Prediction, Difference].
        - Target and Difference are masked where Target is NaN (shown with a distinct 'bad' color).
        - Row labels annotated with valid pixel counts per horizon.
        """
        K = len(preds_phys)
        # Compute symmetric range for diff using only finite diffs
        diffs = []
        max_abs = 1e-6
        for i in range(K):
            p = preds_phys[i].detach().cpu()
            t = tgts_phys[i].detach().cpu()
            d = p - t
            finite = torch.isfinite(d)
            if finite.any():
                max_abs = max(max_abs, float(torch.max(torch.abs(d[finite]))))
            diffs.append(d)

        fig, axes = plt.subplots(K, 3, figsize=(9, 3*K), dpi=150)
        if K == 1:
            axes = np.expand_dims(axes, axis=0)
        # Create colormaps with distinct 'bad' color for masked NaNs
        turbo_bad = cm.get_cmap('turbo').with_extremes(bad='#808080')  # gray for NaNs
        diff_bad = cm.get_cmap('coolwarm').with_extremes(bad='#808080')
        for r in range(K):
            tgt_np = tgts_phys[r].detach().cpu().numpy()
            pred_np = preds_phys[r].detach().cpu().numpy()
            # Mask invalids based on target
            tgt_ma = np.ma.masked_invalid(tgt_np)
            diff_np = diffs[r].detach().cpu().numpy()
            diff_ma = np.ma.masked_array(diff_np, mask=~np.isfinite(tgt_np))
            # Target and Prediction (turbo, fixed physical scale)
            axes[r, 0].imshow(tgt_ma, cmap=turbo_bad, vmin=self.hm_vmin, vmax=self.hm_vmax)
            axes[r, 1].imshow(pred_np, cmap='turbo', vmin=self.hm_vmin, vmax=self.hm_vmax)
            # Difference (coolwarm, symmetric around 0), masked where target invalid
            axes[r, 2].imshow(diff_ma, cmap=diff_bad, vmin=-max_abs, vmax=max_abs)
            # Row label with valid pixel count
            axes[r, 0].set_ylabel(f"H={r+1} (valid_px={int(valid_px_per_h[r])})", fontsize=10)
            # Clean axes
            for c in range(3):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
        # Column titles
        axes[0, 0].set_title('Target', fontsize=12)
        axes[0, 1].set_title('Prediction', fontsize=12)
        axes[0, 2].set_title('Difference (Pred - Target)', fontsize=12)
        plt.tight_layout()
        # Use Agg canvas for backend-agnostic rendering and convert RGBA->RGB
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        rgba = buf.reshape(h, w, 4)
        img = rgba[:, :, :3].copy()
        plt.close(fig)
        return img

    def on_fit_end(self):
        # Build and log a single panel using best checkpoint and multiple val samples
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return
        if len(self._val_samples) == 0:
            return
        # Load best checkpoint weights if available
        ckpt_cb = None
        for cb in self.trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                ckpt_cb = cb
                break
        if ckpt_cb is not None and ckpt_cb.best_model_path:
            state = torch.load(ckpt_cb.best_model_path, map_location=self.device)
            if 'state_dict' in state:
                self.load_state_dict(state['state_dict'])
        # For each collected sample, run rollout and render a panel; then stack vertically
        K = self.forecast_horizon
        panels = []
        for sample in self._val_samples:
            THW = sample['input_dynamic'].unsqueeze(0).to(self.device)  # [1,T,1,H,W]
            S = sample['input_static'].unsqueeze(0).to(self.device)     # [1,1,H,W]
            preds = []
            tgts = []
            valid_px_per_h = []
            seq = THW
            for h in range(1, K + 1):
                # Predict next step
                pred_next = self(seq, S)  # [1,1,H,W]
                preds.append(pred_next[0, 0].detach().cpu())
                # target at h
                if h == 1:
                    tgt_h = sample['target1']
                else:
                    ft = sample['future_targets']
                    if ft is not None and (h - 2) < ft.shape[0]:
                        tgt_h = ft[h - 2]
                    else:
                        tgt_h = torch.full_like(sample['target1'], torch.nan)
                tgts.append(tgt_h)
                # count valid pixels for this horizon (finite target)
                valid_px_per_h.append(int(torch.isfinite(tgt_h).sum().item()))
                # update seq
                seq = torch.cat([seq[:, 1:], pred_next.unsqueeze(1)], dim=1)
            # Unnormalize to physical scale if stats available
            if (self.hm_mean is not None) and (self.hm_std is not None):
                preds_phys = [p * float(self.hm_std) + float(self.hm_mean) for p in preds]
                tgts_phys = [t * float(self.hm_std) + float(self.hm_mean) for t in tgts]
            else:
                preds_phys = preds
                tgts_phys = tgts
            panel = self._render_panel(preds_phys, tgts_phys, valid_px_per_h)
            panels.append(panel)
        # Stack panels vertically into one big panel
        big_panel = np.concatenate(panels, axis=0) if len(panels) > 1 else panels[0]
        self.logger.experiment.log({
            'predictions_vs_targets_ar_panel': wandb.Image(big_panel, caption='Targets vs Predictions ({} samples, H=1..{})'.format(len(self._val_samples), K)),
            'global_step': self.global_step,
        })

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
