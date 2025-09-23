import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import os
import argparse
import json

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from src.models.lightning_module import SpatioTemporalLightningModule
from torchgeo_dataloader import get_dataloader, hm_files, component_files, static_files, years

# Geospatial imports for inference
import rasterio
from rasterio import windows as rio_windows
from rasterio import features as rio_features
from rasterio.transform import rowcol, Affine
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 1 train/val batch for a quick smoke test")
    parser.add_argument("--max_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--train_chips", type=int, default=200, help="Chips per epoch for training")
    parser.add_argument("--val_chips", type=int, default=40, help="Chips per epoch for validation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for train/val")
    parser.add_argument("--train_mode", type=str, default="random", choices=["random", "grid"], help="Sampling mode for training")
    parser.add_argument("--val_mode", type=str, default="grid", choices=["random", "grid"], help="Sampling mode for validation")
    parser.add_argument("--stride", type=int, default=128, help="Stride for grid sampling (pixels)")
    parser.add_argument(
        "--include_components",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?',
        const=True,
        default=True,
        help="Whether to include component covariates (AG, BU, etc.) in dynamic inputs",
    )
    parser.add_argument(
        "--static_channels",
        type=int,
        default=None,
        help="Limit number of static channels (e.g., 1 to use only elevation)",
    )
    # Model complexity
    parser.add_argument("--hidden_dim", type=int, default=64, help="ConvLSTM hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of ConvLSTM layers")
    parser.add_argument("--kernel_size", type=int, default=3, help="Conv kernel size for ConvLSTM")
    # Inference flags
    parser.add_argument(
        "--predict_after_training",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?',
        const=True,
        default=True,
        help="Run large-area prediction and write GeoTIFF after training (default: True)",
    )
    parser.add_argument(
        "--predict_region",
        type=str,
        default=None,
        help="Path to GeoJSON file for prediction region. If not provided, loaded from config/config.yaml",
    )
    parser.add_argument(
        "--predict_stride",
        type=int,
        default=64,
        help="Stride (pixels) between prediction tiles for overlap blending",
    )
    parser.add_argument(
        "--use_location_encoder",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?',
        const=True,
        default=True,
        help="Whether to append per-pixel LocationEncoder features to static inputs (default: True)",
    )
    args = parser.parse_args()
    # Data
    train_loader = get_dataloader(
        batch_size=args.batch_size,
        chip_size=128,
        timesteps=3,
        chips_per_epoch=args.train_chips,
        mode=args.train_mode,
        stride=args.stride,
        include_components=args.include_components,
        static_channels=args.static_channels,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    # Validation uses grid sampling; no future horizons in single-step mode
    val_loader = get_dataloader(
        batch_size=args.batch_size,
        chip_size=128,
        timesteps=3,
        chips_per_epoch=args.val_chips,
        mode=args.val_mode,
        stride=args.stride,
        include_components=args.include_components,
        static_channels=args.static_channels,
        num_workers=9,
        pin_memory=False,
        persistent_workers=False,
    )

    # Model
    num_static_channels = getattr(train_loader.dataset, 'C_static', 1)
    num_dynamic_channels = getattr(train_loader.dataset, 'C_dyn', 1)
    model = SpatioTemporalLightningModule(
        hidden_dim=args.hidden_dim,
        lr=1e-3,
        num_static_channels=num_static_channels,
        num_dynamic_channels=num_dynamic_channels,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        use_location_encoder=args.use_location_encoder,
    )
    # Set normalization stats for physical-scale MAE logging
    if hasattr(train_loader, 'dataset'):
        ds = train_loader.dataset
        if hasattr(ds, 'hm_mean') and hasattr(ds, 'hm_std'):
            model.hm_mean = ds.hm_mean
            model.hm_std = ds.hm_std

    # Callbacks
    checkpoint_cb = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    # No early stopping

    # Wandb logger (optional)
    use_wandb = not args.disable_wandb
    wandb_logger = False if not use_wandb else WandbLogger(project='spatio-temporal-convlstm', log_model=True)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb],
        accelerator='auto',
        default_root_dir=os.path.join(os.getcwd(), 'models', 'checkpoints'),
        logger=wandb_logger,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # --- Log predictions from best checkpoint to wandb (rank 0 only) ---
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    # Only the global zero process should log to W&B
    best_ckpt = checkpoint_cb.best_model_path
    should_log_wandb = (
        best_ckpt
        and use_wandb
        and isinstance(trainer.logger, WandbLogger)
        and getattr(trainer, "is_global_zero", True)
    )
    if should_log_wandb:
        import wandb  # import here to avoid touching wandb on non-logging ranks
        experiment = trainer.logger.experiment  # a wandb.Run
        best_model = SpatioTemporalLightningModule.load_from_checkpoint(best_ckpt)
        best_model.eval()
        # Move model and data to same device
        device = next(best_model.parameters()).device
        # Ensure normalization stats are present for logging
        if hasattr(train_loader, 'dataset'):
            ds_train = train_loader.dataset
            if hasattr(ds_train, 'hm_mean') and hasattr(ds_train, 'hm_std'):
                best_model.hm_mean = ds_train.hm_mean
                best_model.hm_std = ds_train.hm_std
        # Inference on a batch of validation data with at least one valid target pixel
        valid_batch_found = False
        for batch in val_loader:
            target = batch['target']
            # Check for at least one valid (non-NaN) pixel in any sample
            if torch.any(~torch.isnan(target)).item():
                input_dynamic = batch['input_dynamic'].to(device)
                input_static = batch['input_static'].to(device)
                target = target.to(device)
                best_model.eval()
                with torch.no_grad():
                    if input_dynamic.dim() == 4:
                        input_dynamic = input_dynamic.unsqueeze(2)
                
                    # Apply same NaN handling as training/validation
                    # Compute validity mask from RAW inputs
                    target_unsqueezed = target.unsqueeze(1)  # Add channel dimension for consistency
                    target_valid = torch.isfinite(target_unsqueezed)
                    dynamic_valid = torch.isfinite(input_dynamic).all(dim=(1, 2), keepdim=True)
                    static_valid = torch.isfinite(input_static).all(dim=1, keepdim=True).unsqueeze(1)
                    input_mask = target_valid & dynamic_valid.squeeze(2) & static_valid.squeeze(2)
                
                    # Replace NaNs in inputs with 0 for model forward
                    input_dynamic_clean = torch.nan_to_num(input_dynamic, nan=0.0)
                    input_static_clean = torch.nan_to_num(input_static, nan=0.0)
                    # Lon/lat from dataset if present
                    lonlat = batch.get('lonlat', None)
                    if lonlat is not None:
                        lonlat = lonlat.to(device)
                    # Get predictions from model (learnable location encoder)
                    preds = best_model(input_dynamic_clean, input_static_clean, lonlat=lonlat)  # Keep [B, 1, H, W]
                
                    # Set predictions to NaN where any input was NaN
                    preds[~input_mask] = float('nan')
                    
                    # Now squeeze for plotting
                    preds = preds.squeeze(1)  # [B, H, W]
                valid_batch_found = True
                break
        if not valid_batch_found:
            print("WARNING: No valid (non-NaN) target pixels found in any validation batch for image logging and MAE.")
            sys.exit(0)
        # Log images to wandb (backtransformed to original data scale)
        images = []
        # For hexbin plot of observed vs predicted change
        diffs_obs = []
        diffs_mod = []
        B = input_dynamic.shape[0]
        # Retrieve means/stds for inverse transform
        ds = val_loader.dataset
        if hasattr(ds, 'dataset'):
            ds = ds.dataset  # Unwrap DataLoader if needed
        hm_mean, hm_std = ds.hm_mean, ds.hm_std
        elev_mean, elev_std = ds.elev_mean, ds.elev_std
        # Years for labeling come from fixed configuration in dataset
        fixed_input_years = getattr(ds, 'fixed_input_years', (None, None, None))
        fixed_target_year = getattr(ds, 'fixed_target_year', None)
        for b in range(B):
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            # Use a single color ramp for all HM images in original 0-1 scale
            hm_vmin, hm_vmax = 0.0, 1.0
            # Input human footprint chips (T=3), unnormalize and label with fixed years
            input_years = list(fixed_input_years)
            for t in range(3):
                hm_in = input_dynamic[b, t, 0].cpu().numpy() * hm_std + hm_mean
                # Mask input HM by its own validity
                hm_in_plot = np.where(np.isfinite(hm_in), hm_in, np.nan)
                im = axes[0, t].imshow(hm_in_plot, cmap='turbo', vmin=hm_vmin, vmax=hm_vmax)
                axes[0, t].set_title(f'HM {input_years[t]}')
                axes[0, t].axis('off')
                plt.colorbar(im, ax=axes[0, t], fraction=0.046, pad=0.04)
            # Elevation raster backtransformed
            elev_in = input_static[b, 0].cpu().numpy() * elev_std + elev_mean
            im = axes[0, 3].imshow(elev_in, cmap='terrain')
            axes[0, 3].set_title('Elevation (meters)')
            axes[0, 3].axis('off')
            plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
            # Hide unused subplot in first row
            axes[0, 4].axis('off')
            # Target, unnormalize and label with year
            target_orig = target[b].cpu().numpy() * hm_std + hm_mean
            target_year = fixed_target_year if fixed_target_year is not None else 'Target'
            
            # Compute validity mask from RAW inputs to match Lightning module logic
            target_valid = np.isfinite(target_orig)
            # Check all dynamic channels across all timesteps - BEFORE backtransformation
            input_dynamic_raw = input_dynamic[b].cpu().numpy()  # Raw normalized values
            dynamic_valid = np.isfinite(input_dynamic_raw).all(axis=(0, 1))  # [H, W]
            # Check all static channels - BEFORE backtransformation  
            input_static_raw = input_static[b].cpu().numpy()  # Raw normalized values
            static_valid = np.isfinite(input_static_raw).all(axis=0)  # [H, W]
            valid_pred_mask = target_valid & dynamic_valid & static_valid
            
            target_plot = np.where(valid_pred_mask, target_orig, np.nan)
            im0 = axes[1, 0].imshow(target_plot, cmap='turbo', vmin=hm_vmin, vmax=hm_vmax)
            axes[1, 0].set_title(f'Target HM {target_year}')
            axes[1, 0].axis('off')
            plt.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)
            # Prediction, unnormalize and label
            pred_orig = preds[b].cpu().numpy() * hm_std + hm_mean
            pred_plot = np.where(valid_pred_mask, pred_orig, np.nan)
            im1 = axes[1, 1].imshow(pred_plot, cmap='turbo', vmin=hm_vmin, vmax=hm_vmax)
            axes[1, 1].set_title(f'Predicted HM {target_year}')
            axes[1, 1].axis('off')
            plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)
            # Error (in original 0-1 scale)
            error = np.abs(pred_orig - target_orig)
            error_plot = np.where(valid_pred_mask, error, np.nan)
            im2 = axes[1, 2].imshow(error_plot, cmap='hot', vmin=0.0, vmax=hm_vmax)
            axes[1, 2].set_title('Absolute Error (0-1)')
            axes[1, 2].axis('off')
            plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)
            # Delta image (observed): target - most recent input HM
            most_recent_in = (input_dynamic[b, -1, 0].cpu().numpy() * hm_std + hm_mean)
            delta = target_orig - most_recent_in
            delta_plot = np.where(valid_pred_mask, delta, np.nan)
            vmax_delta = np.nanmax(np.abs(delta_plot)) if np.any(np.isfinite(delta_plot)) else 1.0
            im3 = axes[1, 3].imshow(delta_plot, cmap='bwr', vmin=-vmax_delta, vmax=vmax_delta)
            axes[1, 3].set_title(f'Delta HM {target_year}-{input_years[-1]}')
            axes[1, 3].axis('off')
            plt.colorbar(im3, ax=axes[1, 3], fraction=0.046, pad=0.04)
            # Delta image (predicted): prediction - most recent input HM, using the same color scaling as observed delta
            pred_delta = pred_orig - most_recent_in
            pred_delta_plot = np.where(valid_pred_mask, pred_delta, np.nan)
            im4 = axes[1, 4].imshow(pred_delta_plot, cmap='bwr', vmin=-vmax_delta, vmax=vmax_delta)
            axes[1, 4].set_title(f'Pred Delta HM {target_year}-{input_years[-1]}')
            axes[1, 4].axis('off')
            plt.colorbar(im4, ax=axes[1, 4], fraction=0.046, pad=0.04)
            plt.tight_layout()
            # Convert to numpy array and log (robust for macOS backend)
            fig.canvas.draw()
            img_rgba = np.array(fig.canvas.buffer_rgba())
            img_rgb = img_rgba[..., :3]
            images.append(wandb.Image(img_rgb, caption=f"Sample {b}"))
            plt.close(fig)

            # Accumulate diffs for hexbin (mask to valid pixels)
            valid_mask_np = valid_pred_mask
            diffs_obs.append(delta[valid_mask_np])
            diffs_mod.append(pred_delta[valid_mask_np])

        experiment.log({"Predictions_vs_Targets": images})

        # ---- Hexbin plot: Observed vs Predicted HM change ----
        import matplotlib.colors as mcolors
        pmin, pmax = -0.01, 0.2
        if len(diffs_obs) > 0:
            diff_obs_all = np.concatenate(diffs_obs)
            diff_mod_all = np.concatenate(diffs_mod)
            # Filter to range
            in_range = (
                (diff_obs_all >= pmin) & (diff_obs_all <= pmax) &
                (diff_mod_all >= pmin) & (diff_mod_all <= pmax)
            )
            diff_obs_small = diff_obs_all[in_range]
            diff_mod_small = diff_mod_all[in_range]

            fig2 = plt.figure(figsize=(6, 5))
            hb = plt.hexbin(
                diff_obs_small,
                diff_mod_small,
                gridsize=80,
                cmap="cubehelix",
                mincnt=1,
                norm=mcolors.LogNorm(),
            )
            cbar = plt.colorbar(hb)
            cbar.set_label("Count (log scale)")
            # 1:1 line
            plt.plot([pmin, pmax], [pmin, pmax], linestyle="--", color="grey", label="1:1 line")
            # Axes, labels, legend
            plt.axhline(0, color="black", lw=0.5)
            plt.axvline(0, color="black", lw=0.5)
            plt.xlim(pmin, pmax)
            plt.ylim(pmin, pmax)
            plt.xlabel("Observed difference")
            plt.ylabel("Modelled difference")
            plt.legend(frameon=False, loc="upper left")
            plt.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)
            plt.tight_layout()
            fig2.canvas.draw()
            img_rgba2 = np.array(fig2.canvas.buffer_rgba())
            img_rgb2 = img_rgba2[..., :3]
            experiment.log({"Obs_vs_Pred_HM_Change": [wandb.Image(img_rgb2, caption="Obs vs Pred HM change (hexbin)")]})
            plt.close(fig2)

        # Ensure wandb shuts down cleanly
        experiment.finish()
    else:
        if best_ckpt and use_wandb:
            # Only non-zero ranks should skip W&B logging to avoid crashes on multi-GPU
            print("Skipping W&B image logging: not global rank 0 or WandbLogger not active.")

    # Finalize W&B session if used to ensure clean exit
    if use_wandb and getattr(trainer, "is_global_zero", True):
        try:
            import wandb as _wandb
            _wandb.finish()
        except Exception:
            pass

    # -------------------- Large-area prediction to GeoTIFF --------------------
    def _predict_region_and_write(best_ckpt_path: str):
        # Only rank 0 performs writing
        if not getattr(trainer, "is_global_zero", True):
            return
        # Resolve region path: CLI > config
        region_path = args.predict_region
        if region_path is None:
            cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
            if cfg_path.exists():
                try:
                    with open(cfg_path, 'r') as f:
                        cfg = yaml.safe_load(f)
                    region_path = (
                        cfg.get("inference", {}).get("region_geojson", None)
                    )
                except Exception:
                    region_path = None
        if region_path is None or not os.path.exists(region_path):
            print("No valid prediction region specified; skipping large-area prediction.")
            return

        # Load region polygon (assume EPSG:4326 if no CRS field)
        with open(region_path, 'r') as f:
            gj = json.load(f)
        # Merge all features into a single geometry collection/polygon list
        geoms = [shape(feat["geometry"]) for feat in gj.get("features", [])]
        if not geoms:
            print("Empty geometry in region GeoJSON; skipping.")
            return
        # Use target HM raster (2020) as spatial reference
        target_year = getattr(train_loader.dataset, 'fixed_target_year', 2020)
        year_to_idx = {y: i for i, y in enumerate(years)}
        target_src_path = hm_files[year_to_idx[target_year]]
        with rasterio.open(target_src_path) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_height, ref_width = ref.height, ref.width
            # Reproject geoms to ref CRS
            # Assume GeoJSON in EPSG:4326 unless a crs member exists (rare in modern GeoJSON)
            transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
            geoms_ref = [shp_transform(lambda x, y: transformer.transform(x, y), g) for g in geoms]
            # Build a unioned geometry
            try:
                from shapely.ops import unary_union
                region_geom = unary_union(geoms_ref)
            except Exception:
                region_geom = geoms_ref[0]
            # Compute bounding rows/cols
            minx, miny, maxx, maxy = region_geom.bounds
            top_left = rowcol(ref_transform, minx, maxy, op=float)
            bottom_right = rowcol(ref_transform, maxx, miny, op=float)
            r0 = int(max(0, np.floor(min(top_left[0], bottom_right[0]))))
            c0 = int(max(0, np.floor(min(top_left[1], bottom_right[1]))))
            r1 = int(min(ref_height, np.ceil(max(top_left[0], bottom_right[0]))))
            c1 = int(min(ref_width, np.ceil(max(top_left[1], bottom_right[1]))))
            if r1 <= r0 or c1 <= c0:
                print("Region is outside raster extent; skipping.")
                return

            # Prepare accumulators over the bbox window
            Hwin, Wwin = r1 - r0, c1 - c0
            accum = np.zeros((Hwin, Wwin), dtype=np.float64)
            wsum = np.zeros((Hwin, Wwin), dtype=np.float64)
            nodata_mask_total = np.zeros((Hwin, Wwin), dtype=bool)

            # Stats and config from training dataset
            ds_train = train_loader.dataset
            hm_mean, hm_std = ds_train.hm_mean, ds_train.hm_std
            elev_mean, elev_std = ds_train.elev_mean, ds_train.elev_std
            include_components = bool(getattr(ds_train, 'include_components', True))
            static_list_paths = list(static_files if args.static_channels is None else static_files[:int(args.static_channels)])
            input_years = list(getattr(ds_train, 'fixed_input_years', (1990, 1995, 2000)))
            t_idxs = [year_to_idx[y] for y in input_years]

            # Open all sources
            hm_srcs = [rasterio.open(p) for p in hm_files]
            comp_srcs = {y: [rasterio.open(p) for p in component_files[y]] for y in years} if include_components else {y: [] for y in years}
            stat_srcs = [rasterio.open(p) for p in static_list_paths]

            tile = 128
            stride = int(args.predict_stride)
            from rasterio.windows import Window
            # Precompute a region mask over the bbox for faster per-tile tests
            bbox_transform = ref_transform * Affine.translation(c0, r0)
            bbox_mask = rio_features.geometry_mask([mapping(region_geom)], out_shape=(Hwin, Wwin), transform=bbox_transform, invert=True)

            # Load model for inference
            device = next(model.parameters()).device
            infer_model = SpatioTemporalLightningModule.load_from_checkpoint(best_ckpt_path, map_location=device)
            infer_model.eval()
            def lonlat_grid_for_window(i0: int, j0: int, hi: int, wj: int):
                rows = np.arange(i0, i0 + hi)
                cols = np.arange(j0, j0 + wj)
                rr, cc = np.meshgrid(rows, cols, indexing='ij')
                xs, ys = rasterio.transform.xy(ref_transform, rr, cc)
                xs = np.array(xs); ys = np.array(ys)
                if ref_crs and ref_crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
                    transformer = Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
                    lon, lat = transformer.transform(xs, ys)
                else:
                    lon, lat = xs, ys
                return np.stack([lon, lat], axis=-1).astype(np.float32)  # [H, W, 2]

            for i in range(r0, r1, stride):
                for j in range(c0, c1, stride):
                    hi = min(tile, r1 - i)
                    wj = min(tile, c1 - j)
                    if hi <= 0 or wj <= 0:
                        continue
                    # Local indices in accum arrays
                    li0, lj0 = i - r0, j - c0
                    li1, lj1 = li0 + hi, lj0 + wj
                    submask = bbox_mask[li0:li1, lj0:lj1]
                    if not np.any(submask):
                        continue
                    win = Window(j, i, wj, hi)
                    # Build inputs
                    dyn_ts = []
                    for t_idx, y in zip(t_idxs, input_years):
                        channels = []
                        arr_hm = hm_srcs[t_idx].read(1, window=win, masked=True).filled(np.nan)
                        channels.append((arr_hm - hm_mean) / hm_std)
                        if include_components and comp_srcs.get(y, []):
                            for src in comp_srcs[y]:
                                carr = src.read(1, window=win, masked=True).filled(np.nan)
                                channels.append((carr - hm_mean) / hm_std)
                        dyn_ts.append(np.stack(channels, axis=0))  # [C_dyn, hi, wj]
                    input_dynamic_np = np.stack(dyn_ts, axis=0)  # [T, C_dyn, hi, wj]
                    static_chs = []
                    for src in stat_srcs:
                        sarr = src.read(1, window=win, masked=True).filled(np.nan)
                        static_chs.append((sarr - elev_mean) / elev_std)
                    input_static_np = np.stack(static_chs, axis=0) if static_chs else np.zeros((0, hi, wj), dtype=np.float32)

                    # Valid mask consistent with training logic
                    target_valid = np.isfinite(input_dynamic_np[0, 0])  # any single HM layer as spatial support
                    dyn_valid = np.isfinite(input_dynamic_np).all(axis=(0, 1))
                    stat_valid = np.isfinite(input_static_np).all(axis=0) if static_chs else np.ones((hi, wj), dtype=bool)
                    valid_mask = submask & target_valid & dyn_valid & stat_valid
                    if not np.any(valid_mask):
                        continue

                    # Forward pass
                    import torch
                    in_dyn = torch.from_numpy(np.nan_to_num(input_dynamic_np, nan=0.0)).float().unsqueeze(0).to(device)
                    in_stat = torch.from_numpy(np.nan_to_num(input_static_np, nan=0.0)).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        # Build lon/lat grid tensor for the tile if using location encoder
                        lonlat_hw2 = lonlat_grid_for_window(i, j, hi, wj)
                        lonlat_t = torch.from_numpy(lonlat_hw2).to(device).unsqueeze(0)  # [1, H, W, 2]
                        preds = infer_model(in_dyn, in_stat, lonlat=lonlat_t)  # [1, 1, hi, wj]
                    pred_np = preds.squeeze().detach().cpu().numpy()  # normalized
                    pred_np = pred_np * hm_std + hm_mean  # back to 0-1 HM scale

                    # Distance-to-edge weights within tile
                    # Start with interior mask = valid pixels within region
                    interior = valid_mask.astype(np.uint8)
                    # Zero-out 1-pixel border to define edges
                    interior[[0, -1], :] = 0
                    interior[:, [0, -1]] = 0
                    weights = distance_transform_edt(interior)  # 0 at edges, grows inward
                    # Ensure no contribution from invalid pixels
                    weights = np.where(valid_mask, weights, 0.0)
                    if weights.max() > 0:
                        # Accumulate
                        accum[li0:li1, lj0:lj1] += pred_np * weights
                        wsum[li0:li1, lj0:lj1] += weights
                    # Track nodata for areas never covered
                    nodata_mask_total[li0:li1, lj0:lj1] |= ~valid_mask

            # Final blend
            out = np.full((Hwin, Wwin), np.nan, dtype=np.float32)
            m = wsum > 0
            out[m] = (accum[m] / wsum[m]).astype(np.float32)

            # Write GeoTIFF clipped window
            out_profile = ref.profile.copy()
            out_profile.update({
                'height': Hwin,
                'width': Wwin,
                'transform': ref_transform * Affine.translation(c0, r0),
                'count': 1,
                'dtype': 'float32',
                'compress': 'deflate'
            })
            out_dir = Path(os.getcwd()) / 'data' / 'predictions'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"prediction_{target_year}_blended.tif"
            with rasterio.open(out_path, 'w', **out_profile) as dst:
                dst.write(out, 1)
            print(f"Wrote blended prediction GeoTIFF to {out_path}")

            # Close sources
            for src in hm_srcs:
                src.close()
            for y in comp_srcs:
                for src in comp_srcs[y]:
                    src.close()
            for src in stat_srcs:
                src.close()

    if args.predict_after_training and checkpoint_cb.best_model_path:
        _predict_region_and_write(checkpoint_cb.best_model_path)
