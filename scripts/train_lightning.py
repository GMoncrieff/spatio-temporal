import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import os
import argparse
import json
import random

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from src.models.lightning_module import SpatioTemporalLightningModule
from src.models.change_weights import N_CONTEXT_CHANNELS


def _wants_context(args):
    """True when any head is configured to read the past-change context rasters.

    The quantile heads were the first consumer, but the central heads can take the same
    tensor, so the dataset must load it whenever either asks for it.
    """
    return bool(getattr(args, "quantile_context", False) or getattr(args, "central_context", False))


def _csv_floats(spec):
    """'1,1.333,2,4' -> [1.0, 1.333, 2.0, 4.0]; None/'' -> None (today's behaviour)."""
    if spec is None or str(spec).strip() == "":
        return None
    return [float(x) for x in str(spec).split(",")]


def _csv_ints(spec):
    if spec is None or str(spec).strip() == "":
        return None
    return [int(x) for x in str(spec).split(",")]


def _csv_strs(v):
    return tuple(x.strip() for x in str(v).split(",") if x.strip()) if v else ()


def _n_context_channels(args):
    """Head input width. Was the constant N_CONTEXT_CHANNELS; the radii and the
    neighbourhood-HM channels are configurable now, so it is a function of the flags."""
    from src.models.change_weights import context_channel_count
    return context_channel_count(_csv_ints(args.context_radii) or (1, 3, 10, 30, 100),
                                 _csv_strs(args.hm_context_stats),
                                 _csv_ints(args.hm_context_radii) or (3, 30, 100))


def _spline_head_banner(args):
    """The one line that fingerprints the distributional head, or None for the triple head.

    e9 (``--spline_slopes fritsch``) and e10 (``--spline_knots lean9``) ARE the head's
    parameter count, so a run whose flag silently failed to engage would read as "the lever
    does nothing" -- the shape this project has twice mistaken for a finding. Extracted to a
    function so the check that greps this line and the code that prints it can be tested
    against each other, rather than a check being written against text nobody emits.
    """
    if getattr(args, "head_family", "triple") != "spline":
        return None
    from src.models.quantile_spline import knot_preset, n_spline_params
    k = knot_preset(args.spline_knots)
    n = n_spline_params(len(k), args.spline_slopes == "learned")
    return (f"Spline head:       knots {args.spline_knots} (n={len(k)}, bins={len(k) - 1}), "
            f"slopes {args.spline_slopes}, {n} params/horizon")


def _experiment_kwargs(args):
    """Model-phase flags, as constructor kwargs. Every default is today's behaviour."""
    return dict(
        quantile_dhat_context=args.quantile_dhat_context,
        width_parameterisation=args.width_parameterisation,
        convlstm_dilations=_csv_ints(args.convlstm_dilations),
        horizon_loss_weights=_csv_floats(args.horizon_loss_weights),
        loss_on_change=args.loss_on_change,
        pinball_scale_norm=args.pinball_scale_norm,
        lr_schedule=args.lr_schedule,
        lr_warmup_frac=args.lr_warmup_frac,
        lr_min_frac=args.lr_min_frac,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        weight_avg_last=args.weight_avg_last,
        abort_on_nonfinite=args.abort_on_nonfinite,
        head_hidden_layers=args.head_hidden_layers,
        width_head_mode=args.width_head_mode,
        central_target_transform=args.central_target_transform,
        quantile_loss=args.quantile_loss,
        histogram_soft=args.histogram_soft,
        head_family=args.head_family,
        dist_loss=args.dist_loss,
        crps_nodes=args.crps_nodes,
        crps_tail_lam=args.crps_tail_weight,
        crps_tail_u0=args.crps_tail_u0,
        crps_tail_p=args.crps_tail_p,
        mu_mse_weight=args.mu_mse_weight,
        spline_learn_slopes=(args.spline_slopes == 'learned'),
        spline_cumulative_width=args.spline_cumulative_width,
        spline_mean_nodes=args.spline_mean_nodes,
        spline_checkpoint=args.spline_checkpoint,
        chip_weight_correct=args.chip_sampling_correct,
        spline_knots=args.spline_knots,
        crps_tail_lam_lo=args.crps_tail_weight_lo,
        crps_tail_u0_lo=args.crps_tail_u0_lo,
        shape_head_hidden_layers=args.shape_head_hidden_layers,
        shape_head_width=args.shape_head_width,
        context_radii=_csv_ints(args.context_radii),
        hm_context_stats=_csv_strs(args.hm_context_stats),
        hm_context_radii=_csv_ints(args.hm_context_radii),
        isolate_shape_grad=args.isolate_shape_grad,
    )
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
        "--val_stride",
        type=int,
        default=None,
        help="Stride for grid-mode val/test sampling (default: --stride). A larger value "
             "subsamples the held-out geography, which keeps per-epoch validation cheap on "
             "long runs without changing what is being validated.",
    )
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
        default=None,
        help="[Legacy] Run large-area prediction and write GeoTIFF after training. Overrides --run_large_area_prediction when set.",
    )
    parser.add_argument(
        "--run_full_set_evaluation",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?',
        const=True,
        default=True,
        help="Run full test-set evaluation/prediction logging block after training (default: True)",
    )
    parser.add_argument(
        "--run_large_area_prediction",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?',
        const=True,
        default=True,
        help="Run large-area prediction and write GeoTIFFs after training (default: True)",
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
        "--predict_batch_size",
        type=int,
        default=16,
        help="Number of tiles to process in parallel on GPU during prediction (default: 16)",
    )
    parser.add_argument(
        "--predict_final_year",
        type=int,
        default=2040,
        choices=[2020, 2040],
        help="Final prediction year for large-area GeoTIFF output: 2040 (inputs 2010/2015/2020) or 2020 (inputs 1990/1995/2000)",
    )
    # --- Hindcast / ensemble extensions (all additive; defaults reproduce today's behavior) ---
    parser.add_argument(
        "--predict_input_years",
        type=str,
        default=None,
        help="Comma-separated 3 input years (e.g. '1995,2000,2005'). Overrides the legacy "
             "--predict_final_year branch. Targets are base+5/10/15/20.",
    )
    parser.add_argument(
        "--predict_all_windows",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?',
        const=True,
        default=False,
        help="Loop prediction over all 4 hindcast input windows in one process/checkpoint load",
    )
    parser.add_argument(
        "--predict_output_prefix",
        type=str,
        default=None,
        help="Prefix prepended to output GeoTIFF filenames (default None = today's exact names)",
    )
    parser.add_argument(
        "--predict_output_dir",
        type=str,
        default=None,
        help="Directory for prediction GeoTIFFs (default: data/predictions)",
    )
    parser.add_argument(
        "--predict_max_target_year",
        type=int,
        default=None,
        help="Skip writing horizons whose target year exceeds this (e.g. 2020 for hindcasts)",
    )
    parser.add_argument(
        "--predict_restrict_mask",
        type=str,
        default=None,
        help="Raster mask; tiles not overlapping --predict_restrict_values are skipped. "
             "Used to predict only a fold's held-out pixels (exact for those pixels, ~5x cheaper).",
    )
    parser.add_argument(
        "--predict_restrict_values",
        type=str,
        default=None,
        help="Comma-separated values of --predict_restrict_mask to keep",
    )
    parser.add_argument(
        "--fold_mask",
        type=str,
        default=None,
        help="Path to fold_mask_1000.tif; when set with --exclude_fold, replaces split_mask_1000.tif "
             "for train/val chip selection",
    )
    parser.add_argument(
        "--train_all_splits",
        type=lambda x: (str(x).lower() == 'true'), nargs='?', const=True, default=False,
        help="Train on EVERY valid chip in the split mask, ignoring the 70/10/10/10 "
             "train/val/test/calib partition. This is the production setting: the forward "
             "model has no held-out geography to protect, so restricting it to split 1 "
             "throws away 30%% of the world for nothing. Validation still runs on split 2, "
             "which is now in-sample -- that is unavoidable for a production model and is "
             "why the configuration is validated by k-fold instead. Ignored in fold-CV mode "
             "(--exclude_fold), where holding geography out is the whole point.",
    )
    parser.add_argument(
        "--exclude_fold",
        type=int,
        default=None,
        help="Fold id held out from training (its pixels never enter train or val)",
    )
    parser.add_argument(
        "--val_fold",
        type=int,
        default=None,
        help="Fold id used for validation during fold-CV training (default: (exclude_fold %% k) + 1)",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds in --fold_mask (default: 5)",
    )
    parser.add_argument(
        "--norm_stats_json",
        type=str,
        default=None,
        help="JSON sidecar of normalization stats. Loaded if it exists, otherwise written "
             "after the first dataset build (they are not persisted in the .ckpt).",
    )
    parser.add_argument("--wandb_project", type=str, default="spatio-temporal-convlstm")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated W&B tags")
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Lightning devices spec: 'auto', an int count, or a comma-separated device list",
    )
    # --- Quantile-head retraining (the T8/T6 root-cause fix) ---
    parser.add_argument(
        "--split_mask", type=str, default=None,
        help="Override the split mask (e.g. a region-restricted one for development)",
    )
    parser.add_argument(
        "--quantile_context",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?', const=True, default=False,
        help="Feed multi-scale past-change occupancy to the quantile heads. The trunk's "
             "receptive field is ~10px and cannot see whether change occurred 30-100px "
             "away, which is what decides whether change is possible at all.",
    )
    parser.add_argument(
        "--context_pattern", type=str,
        default="data/raw/hm_global/change_context_w{year}_1000.tif",
        help="Full-raster past-change context rasters (band 1 past change, band 2 distance)",
    )
    parser.add_argument(
        "--quantile_class_weighting", type=str, default="none",
        choices=["none", "distance"],
        help="Balance the distance-to-past-change bands in the pinball loss (default: none)",
    )
    parser.add_argument(
        "--central_context",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?', const=True, default=False,
        help="Feed the same past-change context to the central heads. Beyond 100px from "
             "past change, no measured pixel moved by >0.01 in 20 years, and the trunk "
             "cannot see that far.",
    )
    parser.add_argument(
        "--central_residual",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?', const=True, default=False,
        help="Central heads predict change on top of HM_t0 rather than the absolute level, "
             "starting from exact persistence. Measured: with the absolute parameterisation "
             "the model emits change of sd ~0.0075 HM on pixels that did not change.",
    )
    parser.add_argument(
        "--monotone_quantile_width",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?', const=True, default=False,
        help="Quantile heads emit half-widths around the (detached) central forecast that "
             "accumulate across horizons, making spread non-decreasing in lead time and "
             "lower<=central<=upper structural.",
    )
    parser.add_argument(
        "--freeze_trunk",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?', const=True, default=False,
        help="Train the quantile heads only; the trunk and central heads keep the frozen "
             "checkpoint's weights exactly, so the central forecast cannot change.",
    )
    # --- Model-phase experiment flags. All additive; defaults reproduce today's model. ---
    parser.add_argument(
        "--checkpoint_monitor", type=str, default="val_total_loss",
        choices=["val_total_loss", "val_central_loss", "val_loss", "val_crps"],
        help="Metric ModelCheckpoint selects on. val_total_loss (the default) includes "
             "pinball and the histogram term, so a quantile-only change still selects a "
             "different epoch and therefore a different central field; central-only A/Bs "
             "need val_central_loss.",
    )
    parser.add_argument(
        "--horizon_loss_weights", type=str, default=None,
        help="Four comma-separated weights for h=5,10,15,20, renormalised to mean 1. "
             "Training exposure is 4:3:2:1 across horizons (end_year is sampled from "
             "2000/2005/2010/2015 and targets past 2020 are NaN), so h=20 gets a quarter "
             "of h=5's gradient. '1,1.333,2,4' compensates exactly.",
    )
    parser.add_argument(
        "--loss_on_change",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?', const=True, default=False,
        help="Compute SSIM and the Laplacian pyramid on the change field rather than on "
             "absolute HM. Under --central_residual the absolute prediction is HM_t0 plus "
             "a small change, so both terms mostly score the copy.",
    )
    parser.add_argument(
        "--pinball_scale_norm",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?', const=True, default=False,
        help="Divide each pixel's pinball loss by its own detached half-width, making the "
             "quantile objective relative rather than absolute. Not class weighting.",
    )
    parser.add_argument(
        "--quantile_dhat_context",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?', const=True, default=False,
        help="Feed the detached predicted change (and its magnitude) to the quantile "
             "heads. The needed width varies with predicted change and the heads have no "
             "channel carrying it.",
    )
    parser.add_argument(
        "--width_parameterisation", type=str, default="softplus",
        choices=["softplus", "exp"],
        help="How a raw quantile-head output becomes a positive half-width increment. "
             "'exp' makes d(width)/d(raw) proportional to the width itself.",
    )
    parser.add_argument(
        "--convlstm_dilations", type=str, default=None,
        help="Comma-separated per-layer dilation for the ConvLSTM cells, e.g. '1,2,4,8'. "
             "Widens the trunk's ~10 px receptive radius at zero parameter cost. Default "
             "(None) is dilation 1 everywhere, i.e. today's trunk.",
    )
    # --- Round-2 flags: substantial architecture and objective changes ---
    parser.add_argument(
        "--head_hidden_layers", type=int, default=1,
        help="Depth of every prediction head. 1 (default) is Conv3x3 -> ReLU -> Conv1x1; "
             "each extra stage adds a 3x3+ReLU and widens the head's own radius by 1 px.",
    )
    parser.add_argument(
        "--width_head_mode", type=str, default="per_horizon",
        choices=["per_horizon", "joint", "power", "power_plus"],
        help="How the four horizons' half-widths are produced. 'joint' emits all four from "
             "one module so the growth profile in lead time is learned coherently; 'power' "
             "parameterises it as w(h) = w0 * (h/5)**gamma, two per-pixel parameters, which "
             "is exactly the quantity measured to be wrong (the far field's width grows "
             "3.7-8.9x too fast with lead time).",
    )
    parser.add_argument(
        "--central_target_transform", type=str, default="none", choices=["none", "asinh"],
        help="'asinh' gives the central head a variance-stabilised output space: the change "
             "is scale*sinh(raw), linear near zero and reaching large values without large "
             "weights. Zero-init still starts at exact persistence.",
    )
    parser.add_argument(
        "--quantile_loss", type=str, default="pinball", choices=["pinball", "nll"],
        help="'nll' fits (lower, central, upper) as a two-piece normal by log score instead "
             "of fitting two independent percentiles — a different estimand on the same "
             "parameterisation. The Winkler interval score is deliberately not offered: it "
             "is exactly 2/alpha times the pinball sum, so it cannot move the optimum.",
    )
    parser.add_argument(
        "--histogram_soft",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?', const=True, default=False,
        help="Use the differentiable soft-binned histogram. The default hard binning "
             "carries no gradient at all, so the histogram term has never trained anything.",
    )
    # --- The distributional head. 'triple' (the default) is the frozen product exactly. ---
    parser.add_argument(
        "--head_family", type=str, default="triple", choices=["triple", "spline"],
        help="'spline' replaces the (lower, central, upper) triple with a full per-pixel "
             "quantile function trained end to end, so the post-hoc width calibration and "
             "empirical marginal reshaping have nothing left to do. The triple is still "
             "emitted, derived from the spline, so every downstream reader is unchanged.",
    )
    parser.add_argument(
        "--dist_loss", type=str, default="crps", choices=["crps", "nll"],
        help="Objective for the spline head. CRPS is an integral of pinball losses and "
             "inherits their bounded influence per pixel, which is why it survives a "
             "residual with kurtosis ~1e3 where Gaussian NLL inflated the fitted widths by "
             "7x (docs/background/model_phase.md 6.3).",
    )
    parser.add_argument(
        "--crps_nodes", type=int, default=6,
        help="Gauss-Legendre nodes per u-bin. Six keeps the quadrature error below the int16 "
             "storage quantum of 3e-5, so the objective is finer than the product it trains; "
             "three is 1.1e-4 and coarser. Lower it only if memory demands it.",
    )
    parser.add_argument(
        "--crps_tail_weight", type=float, default=0.0,
        help="lambda in w(u) = 1 + lambda * ((u-u0)/(1-u0))_+^p. Each pinball term keeps its "
             "own optimum whatever the weight, so a u-weighting changes where the optimiser "
             "spends effort and never the target -- unlike stratified sampling, which does "
             "move the target and carries an importance correction.",
    )
    parser.add_argument(
        "--predict_qf_levels", type=int, default=64,
        help="Bands in the quantile-function raster written beside the triple by the spline "
             "head. 0 disables it. The grid is normal-spaced with 0.025/0.5/0.975 pinned, so "
             "the qf reproduces the published bounds exactly.",
    )
    parser.add_argument(
        "--spline_checkpoint",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?', const=True, default=True,
        help="Recompute the spline quadratures in the backward pass instead of storing them. "
             "Measured: 22.4 GB of a 24.6 GB card without it at the production batch size. "
             "Mathematically identical; off only for debugging.",
    )
    # ---- stratified chip sampling ------------------------------------------------
    parser.add_argument(
        "--chip_sampling", type=str, default="uniform",
        choices=["uniform", "stratified"],
        help="'stratified' draws training chips with probability rising in the chip's past "
             "change, so rare movers are presented consistently. It samples only from "
             "positions the fold mask already permits, so it adds no leak surface.",
    )
    parser.add_argument(
        "--chip_weights", type=str, default="data/ensemble/chip_weights_128.npz",
        help="Per-chip weight table from scripts/build_chip_weights.py.",
    )
    parser.add_argument(
        "--chip_weight_alpha", type=float, default=4.0,
        help="p(chip) proportional to 1 + alpha * w, with w normalised to mean 1.",
    )
    parser.add_argument(
        "--chip_sampling_correct",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?', const=True, default=True,
        help="Importance-correct the stratified sampler back to the population. Reweighting "
             "samples moves the target distribution, unlike reweighting quantile levels; "
             "False deliberately targets a utility-weighted distribution instead, and the "
             "run is labelled as such.",
    )
    # ---- round 2: knot grid, two-sided tail weight, shape-head capacity, HM context ----
    parser.add_argument(
        "--spline_knots", type=str, default="default14",
        choices=["default14", "body_dense", "deep_lower", "lean9"],
        help="Named knot grid. 'body_dense' adds 0.35/0.45/0.55/0.65: the default grid has "
             "three knots between u=0.10 and u=0.90 while 53%% of pixels move by less than "
             "0.001 over twenty years, and cov50 is the worst-calibrated coverage level. "
             "'deep_lower' adds 0.0001/0.9999, for P(u<0.001) reading 7x nominal at h=20. "
             "'lean9' is a strict subset instead: 8 bins, no 0.001/0.999, asking whether the "
             "tail resolution is information or only capacity.",
    )
    parser.add_argument(
        "--crps_tail_weight_lo", type=float, default=0.0,
        help="Lower-side mirror of --crps_tail_weight. The model is braced for growth that "
             "does not come and blindsided by declines; this puts optimiser effort there.",
    )
    parser.add_argument("--crps_tail_u0_lo", type=float, default=0.05)
    parser.add_argument(
        "--shape_head_hidden_layers", type=int, default=1,
        help="Depth of the shape head only. --head_hidden_layers goes through the shared "
             "factory and would confound this with a change to the central head.",
    )
    parser.add_argument(
        "--shape_head_width", type=int, default=0,
        help="Width of the shape head; 0 means hidden_dim // 2, today's value.",
    )
    parser.add_argument(
        "--context_radii", type=str, default="1,3,10,30,100",
        help="Occupancy radii for the distance-to-past-change band.",
    )
    parser.add_argument(
        "--hm_context_stats", type=str, default="",
        help="Neighbourhood-HM statistics to feed the heads, e.g. 'mean,max'. Empty (the "
             "default) reproduces the round-1 eight-channel context exactly. The model has "
             "never had any information about the LEVEL of development around a pixel — only "
             "where past change happened — and development spreads from development.",
    )
    parser.add_argument("--hm_context_radii", type=str, default="3,30,100")
    parser.add_argument(
        "--hm_context_pattern", type=str,
        default="data/raw/hm_global/hm_context_w{year}_1000.tif",
        help="Built by scripts/prepare_hm_context.py, on the full raster: a 201x201 window "
             "cannot be evaluated inside a 128 px chip.",
    )
    parser.add_argument("--crps_tail_u0", type=float, default=0.95)
    parser.add_argument("--crps_tail_p", type=float, default=2.0)
    parser.add_argument(
        "--mu_mse_weight", type=float, default=1.0,
        help="Weight on MSE(E[Q], y). The published central forecast IS E[Q] -- the mean is "
             "the RMSE-optimal point estimate and this residual is right-skewed, so it is "
             "not the median -- and CRPS presses on it only indirectly. 0 is the pure-CRPS "
             "ablation.",
    )
    parser.add_argument(
        "--spline_slopes", type=str, default="learned", choices=["learned", "fritsch"],
        help="'fritsch' derives every knot slope from the adjacent secants "
             "(monotonicity-preserving, zero parameters) instead of learning them.",
    )
    parser.add_argument(
        "--spline_mean_nodes", type=int, default=8,
        help="Quadrature nodes per bin for E[Q]. Eight, not four: the spline is a rational "
             "function and a steep bin converges slowly.",
    )
    parser.add_argument(
        "--isolate_shape_grad",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?', const=True, default=False,
        help="Keep the distributional loss out of the trunk, as the pinball loss is kept out "
             "today. Off by default: the point of an end-to-end model is that the trunk hears "
             "the objective. On, it is the ablation that measures what that costs.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument(
        "--lr_schedule", type=str, default="none", choices=["none", "cosine"],
        help="Learning-rate schedule. 'cosine' warms up linearly then anneals; stepped by "
             "hand because manual optimization does not drive a Lightning scheduler.",
    )
    parser.add_argument("--lr_warmup_frac", type=float, default=0.05)
    parser.add_argument("--lr_min_frac", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--weight_avg_last", type=int, default=0,
        help="Average the weights of the last N epochs instead of selecting one. On fold 1 "
             "the epoch ModelCheckpoint picked was 140/85/60 across three seeds of the same "
             "configuration while the best 10%% of epochs sat within 3%% of the minimum, so "
             "the argmin is close to arbitrary among the candidates. Implies "
             "--checkpoint_select final. 0 disables it.",
    )
    parser.add_argument(
        "--checkpoint_select", type=str, default="best", choices=["best", "final"],
        help="Which checkpoint prediction uses: the monitored best (default) or the state "
             "at the end of training. 'final' is the coherent choice with an annealed "
             "learning rate or with weight averaging.",
    )
    parser.add_argument(
        "--abort_on_nonfinite",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?', const=True, default=True,
        help="Stop with an error if the weights go non-finite. A diverged run is otherwise "
             "SILENT: ModelCheckpoint never selects a NaN epoch, so the run falls back to its "
             "last healthy checkpoint, finishes, and scores normally. Six of the "
             "distributional round-1 runs died this way and were published anyway. False "
             "restores the old silent behaviour.",
    )
    parser.add_argument(
        "--predict_subsample_blocks", type=int, default=0,
        help="SCREEN MODE: predict only N randomly chosen 128 px blocks of the region instead "
             "of all of it. 0 (the default) predicts everything, today's behaviour. This is "
             "intersected into the same restriction mask the fold hindcast already uses, so "
             "every tile overlapping a kept block is still processed and kept pixels get "
             "EXACTLY the blended value a full run would give -- the screen is exact on the "
             "pixels it keeps, not an approximation of them. Measured on southern Africa: 24 "
             "of 59 blocks reproduces the full-raster ranking at r=0.99 on tail_reach20. On "
             "Africa a 200-block screen is 3.3M px, ~5x the pixels of southern Africa's ENTIRE "
             "scored area, at ~1.3% of the prediction cost.")
    parser.add_argument("--predict_subsample_seed", type=int, default=0)
    parser.add_argument(
        "--spline_cumulative_width",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?', const=True, default=True,
        help="True (default) accumulates non-negative width increments across horizons so the "
             "95 percent width cannot shrink with lead time (T4.2), by construction rather "
             "than by a later pass. False gives each horizon an independent scale, letting "
             "spread shrink -- the ablation that asks whether that constraint is load-bearing "
             "or merely tidy. This is the only monotonicity the model IMPOSES; Q(u) increasing "
             "in u is structural to the spline and is unaffected.")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="Global grad-norm clip applied after the two backward passes, "
                             "0 disables it (today's behaviour)")
    parser.add_argument(
        "--use_location_encoder",
        type=lambda x: (str(x).lower() == 'true'),
        nargs='?',
        const=True,
        default=True,
        help="Whether to append per-pixel LocationEncoder features to static inputs (default: True)",
    )
    parser.add_argument(
        "--locenc_out_channels",
        type=int,
        default=8,
        help="Number of output channels from LocationEncoder (default: 8)",
    )
    parser.add_argument(
        "--locenc_legendre_polys",
        type=int,
        default=10,
        help="Degree of Legendre polynomials for spherical harmonics in LocationEncoder (default: 10)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers (default: 0 for single-threaded)",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients over (default: 1, no accumulation)",
    )
    # Loss weight arguments
    parser.add_argument(
        "--ssim_weight",
        type=float,
        default=2.0,
        help="Weight for SSIM loss (default: 2.0)",
    )
    parser.add_argument(
        "--laplacian_weight",
        type=float,
        default=1.0,
        help="Weight for Laplacian pyramid loss (default: 1.0)",
    )
    parser.add_argument(
        "--histogram_weight",
        type=float,
        default=0.67,
        help="Weight for histogram loss on pixel-level change distributions (default: 0.67)",
    )
    parser.add_argument(
        "--histogram_lambda_w2",
        type=float,
        default=0.1,
        help="Weight for Wasserstein-2 term within histogram loss (default: 0.1)",
    )
    parser.add_argument(
        "--histogram_warmup_epochs",
        type=int,
        default=20,
        help="Number of epochs before histogram loss is applied (default: 20)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file or W&B artifact (e.g., 'model-txn1v2kp:v0' or 'path/to/model.ckpt')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Backward compatibility: legacy flag overrides the new large-area prediction toggle when provided
    run_large_area_prediction = (
        args.predict_after_training
        if args.predict_after_training is not None
        else args.run_large_area_prediction
    )
    
    # Helper function to load checkpoint from W&B artifact or local path
    def load_checkpoint_path(checkpoint_arg):
        """
        Load checkpoint from W&B artifact or local file path.
        
        Args:
            checkpoint_arg: Either a local path or W&B artifact name (e.g., 'model-txn1v2kp:v0')
        
        Returns:
            Path to checkpoint file, or None if not found
        """
        if checkpoint_arg is None:
            return None
        
        # Check if it's a local file first
        if os.path.exists(checkpoint_arg):
            print(f"Using local checkpoint: {checkpoint_arg}")
            return checkpoint_arg
        
        # Try to load as W&B artifact
        try:
            import wandb
            print(f"Attempting to download W&B artifact: {checkpoint_arg}")
            
            # Initialize W&B (will use existing run if in training, or create temp run)
            if wandb.run is None:
                run = wandb.init(project="spatio-temporal-convlstm", job_type="load_checkpoint")
            else:
                run = wandb.run
            
            # Handle different artifact name formats
            if '/' not in checkpoint_arg:
                # Short form: 'model-txn1v2kp:v0' -> 'glennwithtwons/spatio-temporal-convlstm/model-txn1v2kp:v0'
                artifact_name = f"glennwithtwons/spatio-temporal-convlstm/{checkpoint_arg}"
            else:
                artifact_name = checkpoint_arg
            
            artifact = run.use_artifact(artifact_name, type='model')
            artifact_dir = artifact.download()
            
            # Find .ckpt file in artifact directory
            import glob
            ckpt_files = glob.glob(os.path.join(artifact_dir, "*.ckpt"))
            if ckpt_files:
                ckpt_path = ckpt_files[0]
                print(f"✓ Downloaded checkpoint to: {ckpt_path}")
                return ckpt_path
            else:
                print(f"⚠️  No .ckpt file found in artifact")
                return None
                
        except Exception as e:
            print(f"⚠️  Could not load checkpoint '{checkpoint_arg}': {e}")
            return None
    
    # Set seeds for reproducibility (without strict determinism)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    pl.seed_everything(args.seed, workers=True)
    
    # Split mask file
    split_mask_file = args.split_mask or "data/raw/hm_global/split_mask_1000.tif"
    if not os.path.exists(split_mask_file):
        print(f"WARNING: Split mask not found: {split_mask_file}")
        print("Training without train/val/test separation. Run scripts/create_validity_mask.py to create splits.")
        split_mask_file = None

    # Fold-CV mode: the fold mask replaces the 70/10/10/10 split mask. Fold `exclude_fold`
    # is held out entirely (never seen in train or val) so its predictions are genuinely
    # out-of-sample; one other fold serves as the validation set for checkpoint selection.
    train_split_value, train_exclude = 1, None
    val_split_value, test_split_value = 2, 3
    if args.train_all_splits and args.exclude_fold is None:
        # None/None is the dataset's "all data" case (torchgeo_dataloader: "None=all data").
        train_split_value, train_exclude = None, None
        print("=" * 70)
        print("PRODUCTION MODE: training on EVERY chip in the split mask")
        print("  no geography held out; validation on split 2 is in-sample by construction")
        print("=" * 70)
    if args.fold_mask is not None and args.exclude_fold is not None:
        if not os.path.exists(args.fold_mask):
            raise FileNotFoundError(f"--fold_mask not found: {args.fold_mask}")
        split_mask_file = args.fold_mask
        held_out = int(args.exclude_fold)
        val_fold = int(args.val_fold) if args.val_fold is not None else (held_out % args.n_folds) + 1
        if val_fold == held_out:
            raise ValueError("--val_fold must differ from --exclude_fold")
        train_split_value = None
        train_exclude = [held_out, val_fold]
        val_split_value = val_fold
        test_split_value = val_fold
        print("=" * 70)
        print(f"FOLD-CV MODE: holding out fold {held_out} (out-of-sample), "
              f"validating on fold {val_fold}")
        print(f"  Training pool: all folds except {train_exclude}")
        print(f"  Fold mask: {split_mask_file}")
        print("=" * 70)

    # Normalization-stat sidecar (they are plain attributes, absent from the .ckpt)
    cached_norm_stats = None
    if args.norm_stats_json and os.path.exists(args.norm_stats_json):
        with open(args.norm_stats_json, 'r') as f:
            cached_norm_stats = json.load(f)
        print(f"Loaded normalization stats from {args.norm_stats_json}")

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
        use_temporal_sampling=True,  # Enable temporal sampling for training
        end_year_options=(2000, 2005, 2010, 2015),
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers > 0 else False,
        persistent_workers=True if args.num_workers > 0 else False,
        split_mask_file=split_mask_file,
        context_pattern=(args.context_pattern if _wants_context(args) else None),
        hm_context_pattern=(args.hm_context_pattern if _wants_context(args) else None),
        hm_context_stats=_csv_strs(args.hm_context_stats),
        hm_context_radii=_csv_ints(args.hm_context_radii) or (3, 30, 100),
        chip_weights=args.chip_weights,
        chip_sampling=args.chip_sampling,
        chip_weight_alpha=args.chip_weight_alpha,
        split_value=train_split_value,  # Train split (None in fold-CV mode)
        exclude_split_values=train_exclude,
        norm_stats=cached_norm_stats,
    )
    # Persist the sidecar once so later inference entrypoints skip the raster-sampling cost
    if args.norm_stats_json and cached_norm_stats is None:
        cached_norm_stats = train_loader.dataset.norm_stats_dict()
        Path(args.norm_stats_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.norm_stats_json, 'w') as f:
            json.dump(cached_norm_stats, f, indent=2)
        print(f"✓ Wrote normalization stats sidecar: {args.norm_stats_json}")
    val_stride = args.val_stride if args.val_stride is not None else args.stride
    # Validation uses fixed years (1990, 1995, 2000 -> 2005-2020) for consistent metrics
    val_loader = get_dataloader(
        batch_size=args.batch_size,
        chip_size=128,
        timesteps=3,
        chips_per_epoch=args.val_chips,
        mode=args.val_mode,
        stride=val_stride,
        include_components=args.include_components,
        static_channels=args.static_channels,
        use_temporal_sampling=False,  # Fixed years for validation (Option A)
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers > 0 else False,
        persistent_workers=True if args.num_workers > 0 else False,
        split_mask_file=split_mask_file,
        context_pattern=(args.context_pattern if _wants_context(args) else None),
        hm_context_pattern=(args.hm_context_pattern if _wants_context(args) else None),
        hm_context_stats=_csv_strs(args.hm_context_stats),
        hm_context_radii=_csv_ints(args.hm_context_radii) or (3, 30, 100),
        split_value=val_split_value,  # Validation split
        norm_stats=cached_norm_stats,
    )
    # Test uses fixed years (1990, 1995, 2000 -> 2005-2020) for final evaluation
    test_loader = get_dataloader(
        batch_size=args.batch_size,
        chip_size=128,
        timesteps=3,
        chips_per_epoch=args.val_chips,
        mode=args.val_mode,
        stride=val_stride,
        include_components=args.include_components,
        static_channels=args.static_channels,
        use_temporal_sampling=False,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers > 0 else False,
        persistent_workers=True if args.num_workers > 0 else False,
        split_mask_file=split_mask_file,
        context_pattern=(args.context_pattern if _wants_context(args) else None),
        hm_context_pattern=(args.hm_context_pattern if _wants_context(args) else None),
        hm_context_stats=_csv_strs(args.hm_context_stats),
        hm_context_radii=_csv_ints(args.hm_context_radii) or (3, 30, 100),
        split_value=test_split_value,  # Test split
        norm_stats=cached_norm_stats,
    )

    # Model
    num_static_channels = getattr(train_loader.dataset, 'C_static', 1)
    num_dynamic_channels = getattr(train_loader.dataset, 'C_dyn', 1)
    
    # Load from checkpoint if provided
    checkpoint_path = load_checkpoint_path(args.checkpoint)
    if checkpoint_path:
        print("\n" + "="*70)
        print(f"Loading model from checkpoint: {checkpoint_path}")
        print("="*70)
        overrides = dict(
            quantile_context_channels=(_n_context_channels(args) if args.quantile_context else 0),
            quantile_class_weighting=args.quantile_class_weighting,
            freeze_trunk=args.freeze_trunk,
            central_context_channels=(_n_context_channels(args) if args.central_context else 0),
            central_residual=args.central_residual,
            monotone_quantile_width=args.monotone_quantile_width,
            **_experiment_kwargs(args),
        )
        if args.quantile_context or args.central_context:
            # The quantile heads gain input channels, so their first conv no longer matches
            # the checkpoint. Warm-start it: the trained weights are copied into the
            # original channels and the new context channels start at zero, so the model
            # initially reproduces the checkpoint exactly and then learns what the context
            # adds. Random re-initialisation would throw away a trained head for nothing.
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            hp = dict(ckpt.get('hyper_parameters', {}))
            hp.update(overrides)
            model = SpatioTemporalLightningModule(**hp)
            sd = dict(ckpt['state_dict'])
            msd = model.state_dict()
            grown = []
            for k, v in list(sd.items()):
                if k in msd and msd[k].shape != v.shape and v.dim() == 4:
                    new_w = torch.zeros_like(msd[k])
                    new_w[:, :v.shape[1]] = v
                    sd[k] = new_w
                    grown.append(k)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"✓ Checkpoint loaded with {len(grown)} warm-started quantile-head convs")
            if missing:
                print(f"  (randomly initialised: {len(missing)} tensors)")
            print("  Trunk and central heads keep the checkpoint's weights exactly.")
        else:
            model = SpatioTemporalLightningModule.load_from_checkpoint(
                checkpoint_path, strict=not args.freeze_trunk, **overrides)
            print(f"✓ Checkpoint loaded successfully!")
        print(f"\nModel configuration from checkpoint:")
        for key in ['hidden_dim', 'num_layers', 'kernel_size', 'num_static_channels', 
                    'num_dynamic_channels', 'use_location_encoder', 'locenc_out_channels']:
            if key in model.hparams:
                print(f"  {key}: {model.hparams[key]}")
        print("="*70 + "\n")
    else:
        model = SpatioTemporalLightningModule(
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            num_static_channels=num_static_channels,
            num_dynamic_channels=num_dynamic_channels,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            use_location_encoder=args.use_location_encoder,
            locenc_out_channels=args.locenc_out_channels,
            locenc_legendre_polys=args.locenc_legendre_polys,
            ssim_weight=args.ssim_weight,
            laplacian_weight=args.laplacian_weight,
            histogram_weight=args.histogram_weight,
            histogram_lambda_w2=args.histogram_lambda_w2,
            histogram_warmup_epochs=args.histogram_warmup_epochs,
            quantile_context_channels=(_n_context_channels(args) if args.quantile_context else 0),
            quantile_class_weighting=args.quantile_class_weighting,
            freeze_trunk=args.freeze_trunk,
            central_context_channels=(_n_context_channels(args) if args.central_context else 0),
            central_residual=args.central_residual,
            monotone_quantile_width=args.monotone_quantile_width,
            **_experiment_kwargs(args),
        )

    # Compute histogram bin weights from training data (per horizon)
    if args.histogram_weight > 0 and hasattr(model, 'histogram_loss_fn'):
        print("\nComputing histogram bin weights for each horizon from 10 training batches...")
        from src.models.histogram_loss import compute_histogram
        
        horizon_names = ['5yr', '10yr', '15yr', '20yr']
        horizon_keys = ['target_5yr', 'target_10yr', 'target_15yr', 'target_20yr']
        all_horizon_counts = {h: [] for h in horizon_names}
        
        device = next(model.parameters()).device
        num_batches_to_sample = min(10, len(train_loader))
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_batches_to_sample:
                break
            
            input_dynamic = batch['input_dynamic'].to(device)
            last_input = input_dynamic[:, -1, 0]  # [B, H, W]
            
            # Compute histograms for each horizon
            for h_name, h_key in zip(horizon_names, horizon_keys):
                target_h = batch[h_key].to(device)
                
                # Compute mask for valid pixels
                target_valid = torch.isfinite(target_h)
                last_input_valid = torch.isfinite(last_input)
                mask = target_valid & last_input_valid
                
                # Compute deltas
                delta_true = target_h - last_input
                
                # Compute histogram
                counts, _ = compute_histogram(delta_true, model.histogram_bins, mask=mask)
                all_horizon_counts[h_name].append(counts.cpu())
        
        # Compute weights for each horizon
        num_bins = len(model.histogram_bins) - 1
        all_bin_weights = []
        
        print("\n" + "="*70)
        print("HISTOGRAM BIN WEIGHTS PER HORIZON (Rarity-Weighted)")
        print("="*70)
        print(f"Bin edges: {model.histogram_bins.tolist()}\n")
        
        for h_idx, h_name in enumerate(horizon_names):
            # Aggregate counts for this horizon
            horizon_counts = torch.cat(all_horizon_counts[h_name], dim=0)
            total_counts = horizon_counts.sum(dim=0)
            
            # Compute inverse frequency weights
            smoothing = 1e-3
            bin_weights = 1.0 / (total_counts + smoothing)
            bin_weights = bin_weights * num_bins / bin_weights.sum()
            all_bin_weights.append(bin_weights)
            
            # Print bin information for this horizon
            print(f"--- {h_name} Horizon ---")
            print(f"Bin | Count  | Proportion | Weight")
            print("-" * 50)
            total_pixels = total_counts.sum().item()
            for i in range(num_bins):
                count = total_counts[i].item()
                proportion = count / total_pixels
                weight = bin_weights[i].item()
                left_edge = model.histogram_bins[i].item()
                right_edge = model.histogram_bins[i+1].item()
                print(f" {i}  | {count:6.0f} | {proportion:9.4f}  | {weight:6.3f}  [{left_edge:+.3f}, {right_edge:+.3f})")
            print()
        
        # Stack all weights and set in model: [num_horizons, num_bins]
        all_bin_weights = torch.stack(all_bin_weights, dim=0).to(device)
        model.histogram_loss_fn.set_bin_weights(all_bin_weights)
        model.histogram_bins_initialized = True
        print("="*70 + "\n")
    
    # Print loss weights at start of training
    print("="*60)
    print("LOSS WEIGHTS")
    print("="*60)
    print(f"MSE weight:        1.0 (fixed)")
    print(f"SSIM weight:       {args.ssim_weight}")
    print(f"Laplacian weight:  {args.laplacian_weight}")
    print(f"Histogram weight:  {args.histogram_weight} (warmup: {args.histogram_warmup_epochs} epochs)")
    if _wants_context(args):
        # Printed because the round-2 covariate IS a channel count: a run that silently fell
        # back to the eight-channel context would read as "the covariate does nothing".
        print(f"Context channels:  {_n_context_channels(args)} "
              f"(radii {args.context_radii}"
              + (f", hm {args.hm_context_stats} @ {args.hm_context_radii}"
                 if _csv_strs(args.hm_context_stats) else ", no hm context") + ")")
    _spline_banner = _spline_head_banner(args)
    if _spline_banner:
        print(_spline_banner)
    print("="*60 + "\n")
    # Set normalization stats for physical-scale MAE logging
    if hasattr(train_loader, 'dataset'):
        ds = train_loader.dataset
        if hasattr(ds, 'hm_mean') and hasattr(ds, 'hm_std'):
            model.hm_mean = ds.hm_mean
            model.hm_std = ds.hm_std
            # The spline needs them as buffers, not plain attributes: HM's physical range
            # [0, 1] is its support constraint, and it has to be carried in normalized units
            # through a checkpoint round trip.
            model.model.set_norm_stats(ds.hm_mean, ds.hm_std)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(monitor=args.checkpoint_monitor, save_top_k=1, mode='min')
    print(f"Checkpoint selection monitors: {args.checkpoint_monitor}")
    # No early stopping

    # Wandb logger (optional)
    use_wandb = not args.disable_wandb
    wandb_tags = [t.strip() for t in args.wandb_tags.split(',')] if args.wandb_tags else None
    wandb_logger = False if not use_wandb else WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=wandb_tags,
        log_model=True,
    )
    if use_wandb:
        # Record the fold-CV context so hindcast runs are identifiable in the W&B UI
        try:
            wandb_logger.experiment.config.update(
                {
                    "cli_args": vars(args),
                    "exclude_fold": args.exclude_fold,
                    "val_fold": val_split_value if args.exclude_fold is not None else None,
                    "fold_mask": args.fold_mask,
                },
                allow_val_change=True,
            )
        except Exception as e:
            print(f"⚠ Could not log config to W&B: {e}")

    # Devices: 'auto' keeps today's behavior; an explicit spec lets the fold orchestrator
    # pin one process per GPU.
    def _parse_devices(spec):
        if spec is None or spec == "auto":
            return "auto"
        if ',' in spec:
            return [int(x) for x in spec.split(',')]
        return int(spec)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb],
        accelerator='auto',
        devices=_parse_devices(args.devices),
        default_root_dir=os.path.join(os.getcwd(), 'models', 'checkpoints'),
        logger=wandb_logger,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Weight averaging rewrites the in-memory weights in on_train_end, so the epoch-best
    # checkpoint on disk is not the model we mean to publish. Save the end state and point
    # every downstream consumer at it.
    if args.checkpoint_select == 'final' or args.weight_avg_last > 0:
        # These two paths repoint prediction at the end-of-training weights, which bypasses the
        # monitored checkpoint entirely -- so the fallback that (silently) protects a normal run
        # from a diverged one is not there. Check before publishing, not after.
        import torch as _torch
        _bad = [n for n, p in model.named_parameters() if not _torch.isfinite(p).all()]
        if _bad and args.abort_on_nonfinite:
            raise RuntimeError(
                f"refusing to publish end-of-training weights: {len(_bad)} non-finite tensors "
                f"(e.g. {_bad[:3]}). Training diverged, and unlike the monitored checkpoint "
                f"this path has no earlier epoch to fall back to, so prediction would run on "
                f"NaN.")
        _final = os.path.join(os.getcwd(), 'models', 'checkpoints',
                              f'final_fold{args.exclude_fold}_{os.getpid()}.ckpt')
        trainer.save_checkpoint(_final)
        checkpoint_cb.best_model_path = _final
        print(f"Prediction will use the end-of-training checkpoint: {_final}")

    # --- Log validation predictions/metrics from checkpoint to wandb (rank 0 only) ---
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    # Only the global zero process should log to W&B
    # Prefer best checkpoint from current training run; fall back to user-provided checkpoint
    # so this block also works when --max_epochs=0.
    best_ckpt = checkpoint_cb.best_model_path if checkpoint_cb.best_model_path else checkpoint_path
    should_log_wandb = (
        args.run_full_set_evaluation
        and
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
        # Inference on entire test set
        print("\nRunning inference on entire test set for plotting...")
        all_batches_data = []
        num_batches_processed = 0
        
        for batch_idx, batch in enumerate(test_loader):
            # Use 20yr target for validation metrics (multi-horizon)
            target = batch.get('target_20yr', batch.get('target'))
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
                    # Get predictions from model: [B, 12, H, W] (4 horizons × 3 quantiles)
                    # The context tensors were missing here, so under --central_context /
                    # --quantile_context these W&B test metrics were computed with every
                    # context channel silently zeroed. The model refuses that now, which is
                    # how the omission surfaced.
                    _cc = batch.get('change_context')
                    _hc = batch.get('hm_context')
                    preds_all = best_model(
                        input_dynamic_clean, input_static_clean, lonlat=lonlat,
                        change_context=_cc.to(device) if _cc is not None else None,
                        hm_context=_hc.to(device) if _hc is not None else None)
                    
                    # Extract quantile predictions for each horizon
                    # Channel ordering: [lower_5yr, central_5yr, upper_5yr, lower_10yr, central_10yr, upper_10yr, ...]
                    preds_5yr_lower = preds_all[:, 0:1, :, :].clone()  # [B, 1, H, W]
                    preds_5yr = preds_all[:, 1:2, :, :].clone()  # Central
                    preds_5yr_upper = preds_all[:, 2:3, :, :].clone()
                    
                    preds_10yr_lower = preds_all[:, 3:4, :, :].clone()
                    preds_10yr = preds_all[:, 4:5, :, :].clone()  # Central
                    preds_10yr_upper = preds_all[:, 5:6, :, :].clone()
                    
                    preds_15yr_lower = preds_all[:, 6:7, :, :].clone()
                    preds_15yr = preds_all[:, 7:8, :, :].clone()  # Central
                    preds_15yr_upper = preds_all[:, 8:9, :, :].clone()
                    
                    preds_20yr_lower = preds_all[:, 9:10, :, :].clone()
                    preds_20yr = preds_all[:, 10:11, :, :].clone()  # Central
                    preds_20yr_upper = preds_all[:, 11:12, :, :].clone()
                    
                    # Set predictions to NaN where any input was NaN
                    for pred in [preds_5yr_lower, preds_5yr, preds_5yr_upper,
                                 preds_10yr_lower, preds_10yr, preds_10yr_upper,
                                 preds_15yr_lower, preds_15yr, preds_15yr_upper,
                                 preds_20yr_lower, preds_20yr, preds_20yr_upper]:
                        pred[~input_mask] = float('nan')
                    
                    # Squeeze for storage
                    preds_5yr_lower = preds_5yr_lower.squeeze(1)  # [B, H, W]
                    preds_5yr = preds_5yr.squeeze(1)
                    preds_5yr_upper = preds_5yr_upper.squeeze(1)
                    
                    preds_10yr_lower = preds_10yr_lower.squeeze(1)
                    preds_10yr = preds_10yr.squeeze(1)
                    preds_10yr_upper = preds_10yr_upper.squeeze(1)
                    
                    preds_15yr_lower = preds_15yr_lower.squeeze(1)
                    preds_15yr = preds_15yr.squeeze(1)
                    preds_15yr_upper = preds_15yr_upper.squeeze(1)
                    
                    preds_20yr_lower = preds_20yr_lower.squeeze(1)
                    preds_20yr = preds_20yr.squeeze(1)
                    preds_20yr_upper = preds_20yr_upper.squeeze(1)
                
                # Store batch data for later processing (all horizons with quantiles)
                all_batches_data.append({
                    'input_dynamic': input_dynamic.cpu(),
                    'input_static': input_static.cpu(),
                    'target_5yr': batch.get('target_5yr', target).cpu(),
                    'target_10yr': batch.get('target_10yr', target).cpu(),
                    'target_15yr': batch.get('target_15yr', target).cpu(),
                    'target_20yr': batch.get('target_20yr', target).cpu(),
                    'preds_5yr_lower': preds_5yr_lower.cpu(),
                    'preds_5yr': preds_5yr.cpu(),
                    'preds_5yr_upper': preds_5yr_upper.cpu(),
                    'preds_10yr_lower': preds_10yr_lower.cpu(),
                    'preds_10yr': preds_10yr.cpu(),
                    'preds_10yr_upper': preds_10yr_upper.cpu(),
                    'preds_15yr_lower': preds_15yr_lower.cpu(),
                    'preds_15yr': preds_15yr.cpu(),
                    'preds_15yr_upper': preds_15yr_upper.cpu(),
                    'preds_20yr_lower': preds_20yr_lower.cpu(),
                    'preds_20yr': preds_20yr.cpu(),
                    'preds_20yr_upper': preds_20yr_upper.cpu(),
                    'input_mask': input_mask.cpu(),
                    'lonlat': batch.get('lonlat', None)
                })
                num_batches_processed += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Processed {batch_idx + 1} batches...")
        
        if num_batches_processed == 0:
            print("WARNING: No valid (non-NaN) target pixels found in any test batch for image logging.")
            sys.exit(0)
        
        print(f"✓ Processed {num_batches_processed} test batches")
        
        # ===== Calculate metrics over full test set =====
        print("\nCalculating metrics over full test set...")
        from src.models.losses import LaplacianPyramidLoss
        from torchmetrics.functional import structural_similarity_index_measure as ssim
        import torch.nn.functional as F
        
        # Initialize loss functions
        lap_loss_fn = LaplacianPyramidLoss(levels=3, kernel_size=5, sigma=1.0, include_lowpass=True)
        
        # Accumulators for metrics per horizon
        horizon_names = ['5yr', '10yr', '15yr', '20yr']
        horizon_keys = ['target_5yr', 'target_10yr', 'target_15yr', 'target_20yr']
        pred_keys = ['preds_5yr', 'preds_10yr', 'preds_15yr', 'preds_20yr']
        
        horizon_metrics = {h: {
            'mse': 0.0,              # Model MSE on deltas (existing)
            'mae': 0.0,              # Model MAE on absolute predictions
            'mse_abs': 0.0,          # NEW: Model MSE on absolute predictions
            'ssim': 0.0,
            'lap': 0.0,
            'hist': 0.0,
            'pixels': 0,
            'mae_no_change': 0.0,    # Baseline: MAE assuming no change
            'mae_linear': 0.0,       # Baseline: MAE linear extrapolation
            'mse_no_change': 0.0,    # NEW: Baseline MSE assuming no change (absolute)
            'mse_linear': 0.0        # NEW: Baseline MSE linear extrapolation (absolute)
        } for h in horizon_names}
        
        for batch_data in all_batches_data:
            input_dynamic = batch_data['input_dynamic'].to(device)
            input_mask = batch_data['input_mask'].to(device)
            last_input = input_dynamic[:, -1, 0]  # [B, H, W]
            
            # Process each horizon
            for h_idx, (h_name, target_key, pred_key) in enumerate(zip(horizon_names, horizon_keys, pred_keys)):
                target = batch_data[target_key].to(device)
                preds = batch_data[pred_key].to(device)
                
                # Compute mask for valid pixels
                mask = input_mask.squeeze(1) & torch.isfinite(last_input) & torch.isfinite(target)
                
                if mask.sum() == 0:
                    continue
                
                # Delta predictions and targets
                delta_pred = preds - last_input
                delta_true = target - last_input
                
                valid_delta_pred = delta_pred[mask]
                valid_delta_true = delta_true[mask]
                
                # MSE on deltas
                mse = F.mse_loss(valid_delta_pred, valid_delta_true, reduction='sum')
                horizon_metrics[h_name]['mse'] += mse.item()
                
                # MAE on absolute predictions (Conv-CNN model)
                mae = F.l1_loss(preds[mask], target[mask], reduction='sum')
                horizon_metrics[h_name]['mae'] += mae.item()

                # NEW: MSE on absolute predictions (Conv-CNN model)
                mse_abs = F.mse_loss(preds[mask], target[mask], reduction='sum')
                horizon_metrics[h_name]['mse_abs'] += mse_abs.item()
                
                # Baseline 1: "No Change" - assume future = most recent past
                pred_no_change = last_input  # Simply use last input as prediction
                mae_no_change = F.l1_loss(pred_no_change[mask], target[mask], reduction='sum')
                horizon_metrics[h_name]['mae_no_change'] += mae_no_change.item()
                # NEW: Baseline MSE (No Change)
                mse_no_change = F.mse_loss(pred_no_change[mask], target[mask], reduction='sum')
                horizon_metrics[h_name]['mse_no_change'] += mse_no_change.item()
                
                # Baseline 2: "Linear" - extrapolate from recent trend
                # Calculate trend between last two timesteps
                if input_dynamic.shape[1] >= 2:
                    second_last_input = input_dynamic[:, -2, 0]  # [B, H, W]
                    trend = last_input - second_last_input
                    # Multiply trend by horizon multiplier (1x for 5yr, 2x for 10yr, etc.)
                    horizon_multiplier = h_idx + 1  # 1, 2, 3, 4 for 5yr, 10yr, 15yr, 20yr
                    pred_linear = last_input + (trend * horizon_multiplier)
                    mae_linear = F.l1_loss(pred_linear[mask], target[mask], reduction='sum')
                    horizon_metrics[h_name]['mae_linear'] += mae_linear.item()
                    # NEW: Baseline MSE (Linear)
                    mse_linear = F.mse_loss(pred_linear[mask], target[mask], reduction='sum')
                    horizon_metrics[h_name]['mse_linear'] += mse_linear.item()
                
                # SSIM (requires [B, C, H, W])
                preds_sanitized = preds.unsqueeze(1).clone()
                target_sanitized = target.unsqueeze(1).clone()
                mask_4d = mask.unsqueeze(1)
                preds_sanitized[~mask_4d] = 0.0
                target_sanitized[~mask_4d] = 0.0
                ssim_val = ssim(preds_sanitized, target_sanitized, data_range=1.0)
                ssim_loss = 1.0 - ssim_val
                horizon_metrics[h_name]['ssim'] += ssim_loss.item() * mask.sum().item()
                
                # Laplacian loss
                lap_loss = lap_loss_fn(preds_sanitized, target_sanitized, mask=mask_4d)
                horizon_metrics[h_name]['lap'] += lap_loss.item() * mask.sum().item()
                
                # Histogram loss (if enabled)
                if best_model.histogram_weight > 0 and hasattr(best_model, 'histogram_loss_fn'):
                    hist_loss, _, _ = best_model.histogram_loss_fn(
                        delta_true, delta_pred, mask=mask, horizon_idx=h_idx
                    )
                    horizon_metrics[h_name]['hist'] += hist_loss.item() * mask.sum().item()
                
                horizon_metrics[h_name]['pixels'] += mask.sum().item()
        
        # Compute average metrics per horizon and overall
        print("\n" + "="*70)
        print("FULL TEST SET METRICS (Best Model) - PER HORIZON")
        print("="*70)
        
        horizon_years = [2005, 2010, 2015, 2020]
        horizon_labels = [5, 10, 15, 20]  # Years into future for plotting
        all_total_losses = []
        
        # Store MAEs for plotting
        mae_conv_cnn = []
        mae_no_change_list = []
        mae_linear_list = []
        # NEW: Store MSEs for plotting
        mse_conv_cnn = []
        mse_no_change_list = []
        mse_linear_list = []
        
        for h_name, h_year in zip(horizon_names, horizon_years):
            metrics = horizon_metrics[h_name]
            if metrics['pixels'] > 0:
                avg_mse = metrics['mse'] / metrics['pixels']
                avg_mae = metrics['mae'] / metrics['pixels']
                avg_mae_no_change = metrics['mae_no_change'] / metrics['pixels']
                avg_mae_linear = metrics['mae_linear'] / metrics['pixels']
                # NEW: Absolute prediction MSEs
                avg_mse_abs = metrics['mse_abs'] / metrics['pixels']
                avg_mse_no_change = metrics['mse_no_change'] / metrics['pixels']
                avg_mse_linear = metrics['mse_linear'] / metrics['pixels']
                avg_ssim = metrics['ssim'] / metrics['pixels']
                avg_lap = metrics['lap'] / metrics['pixels']
                avg_hist = metrics['hist'] / metrics['pixels'] if best_model.histogram_weight > 0 else 0.0
                
                # Store for plotting
                mae_conv_cnn.append(avg_mae)
                mae_no_change_list.append(avg_mae_no_change)
                mae_linear_list.append(avg_mae_linear)
                # NEW: Store MSEs for plotting
                mse_conv_cnn.append(avg_mse_abs)
                mse_no_change_list.append(avg_mse_no_change)
                mse_linear_list.append(avg_mse_linear)
                
                # Compute total loss
                avg_total = (avg_mse + 
                           best_model.ssim_weight * avg_ssim + 
                           best_model.laplacian_weight * avg_lap)
                if best_model.histogram_weight > 0:
                    avg_total += best_model.histogram_weight * avg_hist
                
                all_total_losses.append(avg_total)
                
                print(f"\n{h_name.upper()} ({h_year}): {metrics['pixels']:,} valid pixels")
                print(f"  MSE:              {avg_mse:.6f}")
                print(f"  MAE (Conv-CNN):   {avg_mae:.6f}")
                print(f"  MAE (No Change):  {avg_mae_no_change:.6f}")
                print(f"  MAE (Linear):     {avg_mae_linear:.6f}")
                print(f"  SSIM loss:        {avg_ssim:.6f}")
                print(f"  Lap loss:         {avg_lap:.6f}")
                if best_model.histogram_weight > 0:
                    print(f"  Hist loss:        {avg_hist:.6f}")
                print(f"  Total loss:       {avg_total:.6f}")
                
                # Log per-horizon metrics to W&B
                experiment.log({
                    f"test_full/mae_{h_name}": avg_mae,
                    f"test_full/mae_no_change_{h_name}": avg_mae_no_change,
                    f"test_full/mae_linear_{h_name}": avg_mae_linear,
                    f"test_full/mse_{h_name}": avg_mse,
                    f"test_full/ssim_loss_{h_name}": avg_ssim,
                    f"test_full/lap_loss_{h_name}": avg_lap,
                    f"test_full/total_loss_{h_name}": avg_total,
                })
                if best_model.histogram_weight > 0:
                    experiment.log({f"test_full/hist_loss_{h_name}": avg_hist})
        
        # Compute and log average across all horizons
        if all_total_losses:
            avg_total_all = sum(all_total_losses) / len(all_total_losses)
            print(f"\nAVERAGE ACROSS ALL HORIZONS:")
            print(f"  Total loss: {avg_total_all:.6f}")
            print("="*70 + "\n")
            
            experiment.log({"test_full/total_loss_avg": avg_total_all})
        else:
            print("WARNING: No valid pixels found for metric calculation")
        
        # Retrieve means/stds for inverse transform
        ds = test_loader.dataset
        if hasattr(ds, 'dataset'):
            ds = ds.dataset  # Unwrap DataLoader if needed
        hm_mean, hm_std = ds.hm_mean, ds.hm_std
        elev_mean, elev_std = ds.elev_mean, ds.elev_std
        
        # Log images from first batch only (for visualization)
        print("\nCreating multi-horizon visualizations from first batch...")
        images = []
        first_batch = all_batches_data[0]
        input_dynamic = first_batch['input_dynamic']
        input_static = first_batch['input_static']
        input_mask = first_batch['input_mask']
        
        # All horizon targets and independent predictions
        horizon_names = ['5yr', '10yr', '15yr', '20yr']
        targets_all = {
            '5yr': first_batch['target_5yr'],
            '10yr': first_batch['target_10yr'],
            '15yr': first_batch['target_15yr'],
            '20yr': first_batch['target_20yr']
        }
        preds_central = {
            '5yr': first_batch['preds_5yr'],
            '10yr': first_batch['preds_10yr'],
            '15yr': first_batch['preds_15yr'],
            '20yr': first_batch['preds_20yr']
        }
        preds_lower = {
            '5yr': first_batch['preds_5yr_lower'],
            '10yr': first_batch['preds_10yr_lower'],
            '15yr': first_batch['preds_15yr_lower'],
            '20yr': first_batch['preds_20yr_lower']
        }
        preds_upper = {
            '5yr': first_batch['preds_5yr_upper'],
            '10yr': first_batch['preds_10yr_upper'],
            '15yr': first_batch['preds_15yr_upper'],
            '20yr': first_batch['preds_20yr_upper']
        }
        
        B = input_dynamic.shape[0]
        for b in range(B):
            # Extract year metadata from this sample (NEW: supports temporal sampling)
            input_years = first_batch.get('input_years', [1990, 1995, 2000])
            target_years = first_batch.get('target_years', [2005, 2010, 2015, 2020])
            # Handle batch dimension if present
            if isinstance(input_years, list) and len(input_years) > 0 and isinstance(input_years[0], list):
                input_years = input_years[b]  # Extract for this sample
                target_years = target_years[b]
            
            # Create multi-horizon figure: 6 rows x 7 columns (added 2 for quantiles)
            # Row 0: Input HM (dynamic years) + Elevation + 3 empty
            # Rows 1-4: Each horizon (Target, Pred, Error, Delta Obs, Delta Pred, Delta Lower, Delta Upper)
            # Row 5: Change histograms for all horizons
            fig, axes = plt.subplots(6, 7, figsize=(28, 24))
            # Use a single color ramp for all HM images in original 0-1 scale
            hm_vmin, hm_vmax = 0.0, 1.0
            # Input human footprint chips (T=3), unnormalize and label with actual years
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
            
            # Hide empty cells in row 0
            axes[0, 4].axis('off')
            axes[0, 5].axis('off')
            axes[0, 6].axis('off')
            
            # Compute validity mask from RAW inputs
            # For visualization: use same strict mask as training to show what model actually learns from
            input_dynamic_raw = input_dynamic[b].cpu().numpy()
            # Note: This requires ALL dynamic channels valid (HM + components)
            dynamic_valid = np.isfinite(input_dynamic_raw).all(axis=(0, 1))
            input_static_raw = input_static[b].cpu().numpy()
            # Note: This requires ALL static channels valid
            static_valid = np.isfinite(input_static_raw).all(axis=0)
            most_recent_in = (input_dynamic[b, -1, 0].cpu().numpy() * hm_std + hm_mean)
            
            # Histogram bins: 8 bins from decrease to extreme increase
            histogram_bins = np.array([-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0])
            num_bins = len(histogram_bins) - 1
            
            # Plot each horizon in rows 1-4
            for h_idx, h_name in enumerate(horizon_names):
                row = h_idx + 1
                h_year = target_years[h_idx]  # Get actual year for this sample
                
                # Get target and independent predictions for this horizon
                target_h = targets_all[h_name][b].cpu().numpy() * hm_std + hm_mean
                pred_h = preds_central[h_name][b].cpu().numpy() * hm_std + hm_mean
                pred_lower = preds_lower[h_name][b].cpu().numpy() * hm_std + hm_mean
                pred_upper = preds_upper[h_name][b].cpu().numpy() * hm_std + hm_mean
                
                # Compute mask for this horizon
                target_valid = np.isfinite(target_h)
                valid_mask = target_valid & dynamic_valid & static_valid
                
                # Deltas
                delta_obs = target_h - most_recent_in
                delta_pred = pred_h - most_recent_in
                delta_lower = pred_lower - most_recent_in
                delta_upper = pred_upper - most_recent_in
                
                # Column 0: Target (show year + horizon offset)
                target_plot = np.where(valid_mask, target_h, np.nan)
                im = axes[row, 0].imshow(target_plot, cmap='turbo', vmin=hm_vmin, vmax=hm_vmax)
                axes[row, 0].set_title(f'Target {h_year} (+{(h_idx+1)*5}yr)')
                axes[row, 0].axis('off')
                plt.colorbar(im, ax=axes[row, 0], fraction=0.046, pad=0.04)
                
                # Column 1: Central Prediction (show year + horizon offset)
                pred_plot = np.where(valid_mask, pred_h, np.nan)
                im = axes[row, 1].imshow(pred_plot, cmap='turbo', vmin=hm_vmin, vmax=hm_vmax)
                axes[row, 1].set_title(f'Pred Central {h_year}')
                axes[row, 1].axis('off')
                plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)
                
                # Column 2: Absolute Error (central)
                error = np.abs(pred_h - target_h)
                error_plot = np.where(valid_mask, error, np.nan)
                im = axes[row, 2].imshow(error_plot, cmap='hot', vmin=0.0, vmax=0.5)
                axes[row, 2].set_title(f'Error {h_year}')
                axes[row, 2].axis('off')
                plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)
                
                # Column 3: Delta Observed
                delta_obs_plot = np.where(valid_mask, delta_obs, np.nan)
                im = axes[row, 3].imshow(delta_obs_plot, cmap='seismic', vmin=-0.3, vmax=0.3)
                axes[row, 3].set_title(f'Δ Obs {h_year}')
                axes[row, 3].axis('off')
                plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)
                
                # Column 4: Delta Predicted (central)
                delta_pred_plot = np.where(valid_mask, delta_pred, np.nan)
                im = axes[row, 4].imshow(delta_pred_plot, cmap='seismic', vmin=-0.3, vmax=0.3)
                axes[row, 4].set_title(f'Δ Pred Central {h_year}')
                axes[row, 4].axis('off')
                plt.colorbar(im, ax=axes[row, 4], fraction=0.046, pad=0.04)
                
                # Column 5: Delta Lower (2.5% quantile)
                delta_lower_plot = np.where(valid_mask, delta_lower, np.nan)
                im = axes[row, 5].imshow(delta_lower_plot, cmap='seismic', vmin=-0.3, vmax=0.3)
                axes[row, 5].set_title(f'Δ Lower 2.5% {h_year}')
                axes[row, 5].axis('off')
                plt.colorbar(im, ax=axes[row, 5], fraction=0.046, pad=0.04)
                
                # Column 6: Delta Upper (97.5% quantile)
                delta_upper_plot = np.where(valid_mask, delta_upper, np.nan)
                im = axes[row, 6].imshow(delta_upper_plot, cmap='seismic', vmin=-0.3, vmax=0.3)
                axes[row, 6].set_title(f'Δ Upper 97.5% {h_year}')
                axes[row, 6].axis('off')
                plt.colorbar(im, ax=axes[row, 6], fraction=0.046, pad=0.04)
            
            # Row 5: Histograms for all horizons
            for h_idx, h_name in enumerate(horizon_names):
                h_year = target_years[h_idx]  # Get actual year for this sample
                target_h = targets_all[h_name][b].cpu().numpy() * hm_std + hm_mean
                pred_h = preds_central[h_name][b].cpu().numpy() * hm_std + hm_mean
                target_valid = np.isfinite(target_h)
                valid_mask = target_valid & dynamic_valid & static_valid
                
                delta_obs = target_h - most_recent_in
                delta_pred = pred_h - most_recent_in
                
                delta_obs_valid = delta_obs[valid_mask]
                delta_pred_valid = delta_pred[valid_mask]
                
                if len(delta_obs_valid) > 0:
                    counts_obs, _ = np.histogram(delta_obs_valid, bins=histogram_bins)
                    counts_pred, _ = np.histogram(delta_pred_valid, bins=histogram_bins)
                    
                    x = np.arange(num_bins)
                    width = 0.35
                    axes[5, h_idx].bar(x - width/2, counts_obs, width, label='Obs', alpha=0.7, color='blue')
                    axes[5, h_idx].bar(x + width/2, counts_pred, width, label='Pred', alpha=0.7, color='red')
                    axes[5, h_idx].set_yscale('log')
                    axes[5, h_idx].set_title(f'Δ Histogram {h_year}')
                    axes[5, h_idx].legend(fontsize=6)
                    axes[5, h_idx].grid(alpha=0.3)
                else:
                    axes[5, h_idx].text(0.5, 0.5, 'No valid', ha='center', va='center')
                    axes[5, h_idx].axis('off')
            
            # Hide the extra columns in histogram row (only 4 histograms)
            axes[5, 4].axis('off')
            axes[5, 5].axis('off')
            axes[5, 6].axis('off')
            
            plt.tight_layout()
            # Convert to numpy array and log (robust for macOS backend)
            fig.canvas.draw()
            img_rgba = np.array(fig.canvas.buffer_rgba())
            img_rgb = img_rgba[..., :3]
            images.append(wandb.Image(img_rgb, caption=f"Sample {b}"))
            plt.close(fig)

        experiment.log({"Predictions_vs_Targets": images})
        
        # ---- Accumulate diffs from ALL batches for hexbin and histogram (per horizon) ----
        print("\nAccumulating changes from all test batches (per horizon)...")
        horizon_names = ['5yr', '10yr', '15yr', '20yr']
        horizon_years = [2005, 2010, 2015, 2020]
        
        # Per-horizon accumulators
        diffs_obs_horizons = {h: [] for h in horizon_names}
        diffs_mod_horizons = {h: [] for h in horizon_names}
        
        for batch_data in all_batches_data:
            input_dynamic_batch = batch_data['input_dynamic']
            input_mask_batch = batch_data['input_mask']
            
            B_batch = input_dynamic_batch.shape[0]
            for b in range(B_batch):
                most_recent_in = (input_dynamic_batch[b, -1, 0].numpy() * hm_std + hm_mean)
                
                # Compute validity mask (same for all horizons)
                input_dynamic_raw = input_dynamic_batch[b].numpy()
                dynamic_valid = np.isfinite(input_dynamic_raw).all(axis=(0, 1))
                input_static_raw = batch_data['input_static'][b].numpy()
                static_valid = np.isfinite(input_static_raw).all(axis=0)
                
                # Process each horizon
                for h_name in horizon_names:
                    target_batch = batch_data[f'target_{h_name}']
                    preds_batch = batch_data[f'preds_{h_name}']
                    
                    # Denormalize
                    target_orig = target_batch[b].numpy() * hm_std + hm_mean
                    pred_orig = preds_batch[b].numpy() * hm_std + hm_mean
                    
                    # Compute validity mask for this horizon
                    target_valid = np.isfinite(target_orig)
                    valid_pred_mask = target_valid & dynamic_valid & static_valid
                    
                    # Calculate changes
                    delta = target_orig - most_recent_in
                    pred_delta = pred_orig - most_recent_in
                    
                    # Accumulate valid pixels only
                    diffs_obs_horizons[h_name].append(delta[valid_pred_mask])
                    diffs_mod_horizons[h_name].append(pred_delta[valid_pred_mask])
        
        print(f"✓ Accumulated changes from {len(all_batches_data)} batches for all horizons")

        # ---- Hexbin plots: Observed vs Predicted HM change (per horizon) ----
        import matplotlib.colors as mcolors
        pmin, pmax = -0.01, 0.2
        
        hexbin_images = []
        for h_name, h_year in zip(horizon_names, horizon_years):
            if len(diffs_obs_horizons[h_name]) > 0:
                diff_obs_all = np.concatenate(diffs_obs_horizons[h_name])
                diff_mod_all = np.concatenate(diffs_mod_horizons[h_name])
                
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
                plt.title(f"Obs vs Pred HM Change - {h_year}")
                plt.legend(frameon=False, loc="upper left")
                plt.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)
                plt.tight_layout()
                fig2.canvas.draw()
                img_rgba2 = np.array(fig2.canvas.buffer_rgba())
                img_rgb2 = img_rgba2[..., :3]
                hexbin_images.append(wandb.Image(img_rgb2, caption=f"{h_year} ({h_name})"))
                plt.close(fig2)
        
        if hexbin_images:
            experiment.log({"Obs_vs_Pred_HM_Change": hexbin_images})
            
        # ---- Histograms: Observed vs Predicted HM change distribution (per horizon) ----
        bins = [-1, -0.05, 0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
        bin_labels = ['-1 to -0.05', '-0.05 to 0', '0 to 0.005', '0.005 to 0.02', 
                      '0.02 to 0.05', '0.05 to 0.1', '0.1 to 0.2', '0.2 to 0.5', '0.5 to 1']
        
        histogram_images = []
        for h_name, h_year in zip(horizon_names, horizon_years):
            if len(diffs_obs_horizons[h_name]) > 0:
                diff_obs_all = np.concatenate(diffs_obs_horizons[h_name])
                diff_mod_all = np.concatenate(diffs_mod_horizons[h_name])
                
                # Compute histograms on full data (not just filtered range)
                obs_hist, _ = np.histogram(diff_obs_all, bins=bins)
                pred_hist, _ = np.histogram(diff_mod_all, bins=bins)
                
                # Create histogram figure
                fig3, ax = plt.subplots(figsize=(12, 6))
                x = np.arange(len(bin_labels))
                width = 0.35
                
                # Plot bars
                ax.bar(x - width/2, obs_hist, width, label='Observed', alpha=0.8, color='#2ecc71')
                ax.bar(x + width/2, pred_hist, width, label='Predicted', alpha=0.8, color='#3498db')
                
                # Formatting
                ax.set_xlabel('Change Bins', fontsize=12, fontweight='bold')
                ax.set_ylabel('Count (log scale)', fontsize=12, fontweight='bold')
                ax.set_title(f'Observed vs Predicted HM Change Distribution - {h_year}', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(bin_labels, rotation=45, ha='right')
                ax.legend(fontsize=11)
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                fig3.canvas.draw()
                img_rgba3 = np.array(fig3.canvas.buffer_rgba())
                img_rgb3 = img_rgba3[..., :3]
                histogram_images.append(wandb.Image(img_rgb3, caption=f"{h_year} ({h_name})"))
                plt.close(fig3)
        
        if histogram_images:
            experiment.log({"HM_Change_Histogram": histogram_images})
        
        # ---- MAE Comparison Plot: Conv-CNN vs Baselines ----
        print("\nCreating MAE comparison plot (Conv-CNN vs Baselines)...")
        if len(mae_conv_cnn) > 0 and len(mae_no_change_list) > 0 and len(mae_linear_list) > 0:
            fig_mae, ax_mae = plt.subplots(figsize=(10, 6))
            
            # Plot three lines
            ax_mae.plot(horizon_labels, mae_conv_cnn, marker='o', linewidth=2.5, 
                       label='Conv-CNN', color='#3498db', markersize=8)
            ax_mae.plot(horizon_labels, mae_no_change_list, marker='s', linewidth=2.5, 
                       label='No Change', color='#e74c3c', markersize=8, linestyle='--')
            ax_mae.plot(horizon_labels, mae_linear_list, marker='^', linewidth=2.5, 
                       label='Linear', color='#f39c12', markersize=8, linestyle='--')
            
            # Formatting
            ax_mae.set_xlabel('Forecast Horizon (years)', fontsize=12, fontweight='bold')
            ax_mae.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
            ax_mae.set_title('MAE vs Forecast Horizon: Model Comparison', fontsize=14, fontweight='bold')
            ax_mae.set_xticks(horizon_labels)
            ax_mae.legend(fontsize=11, loc='upper left')
            ax_mae.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig_mae.canvas.draw()
            img_rgba_mae = np.array(fig_mae.canvas.buffer_rgba())
            img_rgb_mae = img_rgba_mae[..., :3]
            experiment.log({"MAE_Comparison": wandb.Image(img_rgb_mae, 
                           caption="MAE comparison: Conv-CNN vs Baselines")})
            plt.close(fig_mae)
            print("✓ MAE comparison plot created and logged")

        # ---- MSE Comparison Plot: Conv-CNN vs Baselines ----
        print("\nCreating MSE comparison plot (Conv-CNN vs Baselines)...")
        if len(mse_conv_cnn) > 0 and len(mse_no_change_list) > 0 and len(mse_linear_list) > 0:
            fig_mse, ax_mse = plt.subplots(figsize=(10, 6))
            
            # Plot three lines
            ax_mse.plot(horizon_labels, mse_conv_cnn, marker='o', linewidth=2.5, 
                        label='Conv-CNN', color='#3498db', markersize=8)
            ax_mse.plot(horizon_labels, mse_no_change_list, marker='s', linewidth=2.5, 
                        label='No Change', color='#e74c3c', markersize=8, linestyle='--')
            ax_mse.plot(horizon_labels, mse_linear_list, marker='^', linewidth=2.5, 
                        label='Linear', color='#f39c12', markersize=8, linestyle='--')
            
            # Formatting
            ax_mse.set_xlabel('Forecast Horizon (years)', fontsize=12, fontweight='bold')
            ax_mse.set_ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
            ax_mse.set_title('MSE vs Forecast Horizon: Model Comparison', fontsize=14, fontweight='bold')
            ax_mse.set_xticks(horizon_labels)
            ax_mse.legend(fontsize=11, loc='upper left')
            ax_mse.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig_mse.canvas.draw()
            img_rgba_mse = np.array(fig_mse.canvas.buffer_rgba())
            img_rgb_mse = img_rgba_mse[..., :3]
            experiment.log({"MSE_Comparison": wandb.Image(img_rgb_mse, 
                               caption="MSE comparison: Conv-CNN vs Baselines")})
            plt.close(fig_mse)
            print("✓ MSE comparison plot created and logged")

        # Ensure wandb shuts down cleanly
        experiment.finish()
    else:
        if not args.run_full_set_evaluation:
            print("Skipping full test-set evaluation/prediction logging (--run_full_set_evaluation false).")
        elif best_ckpt and use_wandb:
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
    # Snapshot everything prediction needs from the training dataset, so the loaders (and
    # their persistent worker processes) can be released first. At the global extent the
    # blending accumulators need tens of GB, and two folds run concurrently.
    _ds_train = train_loader.dataset
    PREDICT_STATS = {
        'hm_mean': _ds_train.hm_mean,
        'hm_std': _ds_train.hm_std,
        'elev_mean': _ds_train.elev_mean,
        'elev_std': _ds_train.elev_std,
        'include_components': bool(getattr(_ds_train, 'include_components', True)),
        'comp_means': dict(_ds_train.comp_means),
        'comp_stds': dict(_ds_train.comp_stds),
        'static_means': list(_ds_train.static_means),
        'static_stds': list(_ds_train.static_stds),
    }

    def _release_dataloaders():
        import gc
        for name in ('train_loader', 'val_loader', 'test_loader'):
            obj = globals().pop(name, None)
            if obj is not None and hasattr(obj, '_iterator'):
                obj._iterator = None
            del obj
        gc.collect()

    # Hindcast input windows: every 3-year window whose +5yr target is still observed.
    HINDCAST_WINDOWS = [
        (1990, 1995, 2000),
        (1995, 2000, 2005),
        (2000, 2005, 2010),
        (2005, 2010, 2015),
    ]

    def _predict_region_and_write(best_ckpt_path: str, input_years_override=None,
                                  output_prefix=None, infer_model=None):
        import time
        start_time = time.time()

        print("\n" + "="*70)
        print("LARGE-AREA PREDICTION")
        print("="*70)
        
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
            print("⚠ No valid prediction region specified; skipping large-area prediction.")
            return
        
        print(f"Region file: {region_path}")

        # Load region polygon (assume EPSG:4326 if no CRS field)
        with open(region_path, 'r') as f:
            gj = json.load(f)
        # Merge all features into a single geometry collection/polygon list
        geoms = [shape(feat["geometry"]) for feat in gj.get("features", [])]
        if not geoms:
            print("Empty geometry in region GeoJSON; skipping.")
            return
        # Configure prediction years: explicit window (new) > --predict_input_years > legacy branch
        window = input_years_override
        if window is None and args.predict_input_years:
            window = [int(y.strip()) for y in args.predict_input_years.split(',')]
        if window is not None:
            input_years = list(window)
            if len(input_years) != 3:
                raise ValueError(f"Prediction window needs exactly 3 input years, got {input_years}")
            base_year = input_years[-1]
            target_years = tuple(base_year + h for h in (5, 10, 15, 20))
            print(f"Input window: {input_years} -> targets {list(target_years)}")
        elif args.predict_final_year == 2040:
            # Use 2020 as base, predict 2025, 2030, 2035, 2040
            input_years = [2010, 2015, 2020]
            target_years = (2025, 2030, 2035, 2040)
            base_year = 2020
        else:
            # Use 2000 as base, predict 2005, 2010, 2015, 2020
            input_years = [1990, 1995, 2000]
            target_years = (2005, 2010, 2015, 2020)
            base_year = 2000

        # Use base-year HM raster as spatial reference
        year_to_idx = {y: i for i, y in enumerate(years)}
        target_src_path = hm_files[year_to_idx[base_year]]
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
                print("⚠ Region is outside raster extent; skipping.")
                return

            # Prepare accumulators over the bbox window (one per horizon)
            Hwin, Wwin = r1 - r0, c1 - c0
            print(f"Region bounding box: {Hwin} × {Wwin} pixels")
            print(f"  Row range: [{r0}, {r1})")
            print(f"  Col range: [{c0}, {c1})")
            
            # Multi-horizon quantile accumulators (3 quantiles × 4 horizons = 12 outputs)
            horizon_names = ['5yr', '10yr', '15yr', '20yr']
            horizon_years = list(target_years)
            quantile_names = ['lower', 'central', 'upper']  # Updated to match independent heads terminology
            
            # Create accumulators for each horizon-quantile combination.
            # float32 (not float64) and only for horizons that will actually be written:
            # at the global extent each accumulator is 2.7 GB, and a hindcast window whose
            # later horizons fall past the last observed year needs none of them.
            active_horizons = [
                h for h, y in zip(horizon_names, horizon_years)
                if args.predict_max_target_year is None or y <= args.predict_max_target_year
            ]
            if not active_horizons:
                print("⚠ No horizons within --predict_max_target_year; skipping this window.")
                return infer_model
            # Accumulators are allocated after the model is loaded, because how many there
            # are depends on the head family: the spline head adds one per quantile level.
            wsum = np.zeros((Hwin, Wwin), dtype=np.float32)
            nodata_mask_total = np.zeros((Hwin, Wwin), dtype=bool)

            # Stats and config captured from the training dataset before it is released
            # (see PREDICT_STATS below) — prediction must not keep the dataloaders and
            # their worker processes alive, since the global accumulators need the RAM.
            hm_mean, hm_std = PREDICT_STATS['hm_mean'], PREDICT_STATS['hm_std']
            elev_mean, elev_std = PREDICT_STATS['elev_mean'], PREDICT_STATS['elev_std']
            include_components = bool(PREDICT_STATS['include_components'])
            static_list_paths = list(static_files if args.static_channels is None else static_files[:int(args.static_channels)])
            t_idxs = [year_to_idx[y] for y in input_years]

            # CRITICAL: per-variable normalization stats (NOT pooled hm_mean/hm_std)
            comp_means = PREDICT_STATS['comp_means']  # Dict: {var_name: mean}
            comp_stds = PREDICT_STATS['comp_stds']    # Dict: {var_name: std}
            static_means = PREDICT_STATS['static_means']  # List: [mean_0, mean_1, ...]
            static_stds = PREDICT_STATS['static_stds']    # List: [std_0, std_1, ...]
            # HM_VARS from module (not instance attribute)
            HM_VARS = ["AG", "BU", "EX", "FR", "HI", "NS", "PO", "TI", "gdp", "population"]

            # Open all sources
            hm_srcs = [rasterio.open(p) for p in hm_files]
            comp_srcs = {y: [rasterio.open(p) for p in component_files[y]] for y in years} if include_components else {y: [] for y in years}
            stat_srcs = [rasterio.open(p) for p in static_list_paths]
            hm_ctx_src, hm_ctx_bands = None, None
            if (_wants_context(args) and args.hm_context_pattern
                    and _csv_strs(args.hm_context_stats)):
                from prepare_hm_context import band_indices
                hm_path = args.hm_context_pattern.format(year=base_year)
                if not os.path.exists(hm_path):
                    raise FileNotFoundError(
                        f"--hm_context_stats was requested but {hm_path} is missing; "
                        f"run scripts/prepare_hm_context.py first")
                hm_ctx_src = rasterio.open(hm_path)
                hm_ctx_bands = band_indices(hm_ctx_src.tags(),
                                            _csv_strs(args.hm_context_stats),
                                            _csv_ints(args.hm_context_radii) or (3, 30, 100))
                print(f"HM context: {hm_path} bands {hm_ctx_bands}")

            ctx_src = None
            if _wants_context(args) and args.context_pattern:
                ctx_path = args.context_pattern.format(year=base_year)
                if os.path.exists(ctx_path):
                    ctx_src = rasterio.open(ctx_path)
                    print(f"Quantile-head context: {ctx_path}")
                else:
                    print(f"⚠ context raster missing ({ctx_path}); heads will see zeros")

            tile = 128
            stride = int(args.predict_stride)
            from rasterio.windows import Window
            # Precompute a region mask over the bbox for faster per-tile tests
            bbox_transform = ref_transform * Affine.translation(c0, r0)
            bbox_mask = rio_features.geometry_mask([mapping(region_geom)], out_shape=(Hwin, Wwin), transform=bbox_transform, invert=True)

            # Load model for inference (reused across windows when caller supplies it)
            device = next(model.parameters()).device
            if device.type == 'cpu' and torch.cuda.is_available():
                # Lightning may have returned the module to CPU after fit; prediction over a
                # large region on CPU is orders of magnitude slower.
                device = torch.device('cuda')
            if infer_model is None:
                print(f"\nLoading model from checkpoint: {best_ckpt_path}")
                infer_model = SpatioTemporalLightningModule.load_from_checkpoint(best_ckpt_path, map_location=device)
                # The quantile-head context is derived from the normalized HM channel and
                # rescaled by hm_std; these are plain attributes absent from the .ckpt, so
                # without setting them here inference would build the context at a
                # different scale than training did.
                infer_model.hm_mean = PREDICT_STATS['hm_mean']
                infer_model.hm_std = PREDICT_STATS['hm_std']
                infer_model.eval()
                print(f"✓ Model loaded on device: {device}")
            infer_model = infer_model.to(device)

            # The quantile-function raster: one band per u-level, written only by the spline
            # head. The triple is *derived* from this, and 0.025 / 0.975 are levels of this
            # grid, so the two agree exactly rather than approximately.
            qf_u = None
            qf_names = []
            if getattr(args, 'predict_qf_levels', 0) and \
                    getattr(infer_model.model, 'head_family', 'triple') == 'spline':
                from src.models.quantile_spline import output_u_grid, splines_from_output
                qf_u = output_u_grid(int(args.predict_qf_levels))
                qf_names = [f"qf{i:03d}" for i in range(len(qf_u))]
                print(f"  Quantile function: {len(qf_u)} levels, "
                      f"u in [{qf_u[0]:.5f}, {qf_u[-1]:.5f}], "
                      f"{len(qf_u) * len(active_horizons)} accumulators")
            accum_horizons = {
                f"{h}_{q}": np.zeros((Hwin, Wwin), dtype=np.float32)
                for h in active_horizons for q in list(quantile_names) + qf_names
            }

            # Optional restriction mask: skip tiles that do not overlap the requested values.
            # Used for fold hindcasts, where only the held-out fold's pixels are consumed.
            # Every tile overlapping a kept pixel is still processed, so kept pixels get
            # exactly the same blended value as an unrestricted run.
            restrict_win = None
            restrict_values = None
            if args.predict_restrict_mask:
                if not args.predict_restrict_values:
                    raise ValueError("--predict_restrict_mask requires --predict_restrict_values")
                restrict_values = [int(v) for v in str(args.predict_restrict_values).split(',')]
                with rasterio.open(args.predict_restrict_mask) as rsrc:
                    restrict_win = rsrc.read(1, window=Window(c0, r0, Wwin, Hwin))
                restrict_win = np.isin(restrict_win, restrict_values)
                print(f"Restriction mask: {args.predict_restrict_mask} values={restrict_values} "
                      f"({restrict_win.sum():,} of {restrict_win.size:,} px kept)")

            if int(getattr(args, "predict_subsample_blocks", 0)) > 0:
                _B = 128
                _nby, _nbx = (Hwin + _B - 1) // _B, (Wwin + _B - 1) // _B
                _base = restrict_win if restrict_win is not None else bbox_mask
                # Only blocks that carry predictable pixels are candidates; otherwise the
                # sample is mostly ocean and its effective size is a fiction.
                _cand = [(by, bx) for by in range(_nby) for bx in range(_nbx)
                         if _base[by * _B:(by + 1) * _B, bx * _B:(bx + 1) * _B].any()]
                _rng = np.random.default_rng(int(args.predict_subsample_seed))
                _pick = _rng.permutation(len(_cand))[:int(args.predict_subsample_blocks)]
                _keep = np.zeros((Hwin, Wwin), dtype=bool)
                for _i in _pick:
                    _by, _bx = _cand[_i]
                    _keep[_by * _B:(_by + 1) * _B, _bx * _B:(_bx + 1) * _B] = True
                restrict_win = _keep if restrict_win is None else (restrict_win & _keep)
                print(f"SCREEN MODE: {len(_pick)} of {len(_cand)} candidate {_B}px blocks "
                      f"(seed {args.predict_subsample_seed}); "
                      f"{int(restrict_win.sum()):,} px kept for prediction")
            
            def lonlat_grid_for_window(i0: int, j0: int, hi: int, wj: int):
                rows = np.arange(i0, i0 + hi)
                cols = np.arange(j0, j0 + wj)
                rr, cc = np.meshgrid(rows, cols, indexing='ij')
                xs, ys = rasterio.transform.xy(ref_transform, rr, cc)
                xs = np.array(xs); ys = np.array(ys)
                # Ensure xs and ys have shape [hi, wj]
                if xs.ndim == 1:
                    xs = xs.reshape(hi, wj)
                    ys = ys.reshape(hi, wj)
                if ref_crs and ref_crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
                    transformer = Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
                    lon, lat = transformer.transform(xs, ys)
                    lon = np.array(lon); lat = np.array(lat)
                    if lon.ndim == 1:
                        lon = lon.reshape(hi, wj)
                        lat = lat.reshape(hi, wj)
                else:
                    lon, lat = xs, ys
                return np.stack([lon, lat], axis=-1).astype(np.float32)  # [hi, wj, 2]

            # Calculate total tiles for progress tracking
            num_tiles_i = len(range(r0, r1, stride))
            num_tiles_j = len(range(c0, c1, stride))
            total_tiles = num_tiles_i * num_tiles_j
            print(f"\nProcessing {total_tiles:,} tiles ({num_tiles_i} × {num_tiles_j})")
            print(f"  Tile size: {tile} × {tile} pixels")
            print(f"  Stride: {stride} pixels")
            print(f"  Target horizons: {horizon_years}")
            print(f"  Input years: {input_years}")
            print()
            
            # Collect all tile coordinates first
            tile_coords = []
            for i in range(r0, r1, stride):
                for j in range(c0, c1, stride):
                    tile_coords.append((i, j))
            
            batch_size = args.predict_batch_size
            print(f"  Batch size: {batch_size} tiles")
            
            tiles_processed = 0
            tiles_skipped = 0
            tiles_with_valid = 0
            last_percent = -1
            tile_start_time = time.time()

            # Process tiles in batches
            for batch_start in range(0, len(tile_coords), batch_size):
                batch_end = min(batch_start + batch_size, len(tile_coords))
                batch_tiles = tile_coords[batch_start:batch_end]
                
                # Prepare batch data
                batch_inputs_dyn = []
                batch_contexts = []
                batch_hm_contexts = []
                batch_inputs_stat = []
                batch_lonlats = []
                batch_metadata = []  # Store (i, j, hi, wj, li0, lj0, li1, lj1, valid_mask, input_invalid_mask)
                
                for i, j in batch_tiles:
                    hi = min(tile, r1 - i)
                    wj = min(tile, c1 - j)
                    if hi <= 0 or wj <= 0:
                        continue
                    # Local indices in accum arrays
                    li0, lj0 = i - r0, j - c0
                    li1, lj1 = li0 + hi, lj0 + wj
                    submask = bbox_mask[li0:li1, lj0:lj1]
                    if not np.any(submask):
                        tiles_processed += 1
                        tiles_skipped += 1
                        continue
                    # Cheap pre-read rejection (before any raster IO) for restricted runs
                    if restrict_win is not None and not restrict_win[li0:li1, lj0:lj1].any():
                        tiles_processed += 1
                        tiles_skipped += 1
                        continue
                    win = Window(j, i, wj, hi)
                    # Build inputs
                    dyn_ts = []
                    for t_idx, y in zip(t_idxs, input_years):
                        channels = []
                        arr_hm = hm_srcs[t_idx].read(1, window=win, masked=True).filled(np.nan)
                        # Data is already in [0, 1] range
                        channels.append((arr_hm - hm_mean) / hm_std)
                        if include_components and comp_srcs.get(y, []):
                            for var_idx, (var_name, src) in enumerate(zip(HM_VARS, comp_srcs[y])):
                                carr = src.read(1, window=win, masked=True).filled(np.nan)
                                # Replace NaN with 0 BEFORE normalization (missing = no pressure/activity)
                                carr = np.nan_to_num(carr, nan=0.0)
                                # Use per-variable normalization (CRITICAL for GDP/population)
                                channels.append((carr - comp_means[var_name]) / comp_stds[var_name])
                        dyn_ts.append(np.stack(channels, axis=0))  # [C_dyn, hi, wj]
                    input_dynamic_np = np.stack(dyn_ts, axis=0)  # [T, C_dyn, hi, wj]
                    static_chs = []
                    # Static file order: [ele, tas, tasmin, pr, dpi_dsi, iucn_nostrict, iucn_strict]
                    nan_to_zero_static = {0, 4, 5, 6}  # ele, dpi_dsi, iucn_nostrict, iucn_strict
                    for static_idx, src in enumerate(stat_srcs):
                        sarr = src.read(1, window=win, masked=True).filled(np.nan)
                        # Replace NaN with 0 for specific variables (before normalization)
                        if static_idx in nan_to_zero_static:
                            sarr = np.nan_to_num(sarr, nan=0.0)
                        # Use per-variable normalization (CRITICAL for different scales)
                        static_chs.append((sarr - static_means[static_idx]) / static_stds[static_idx])
                    input_static_np = np.stack(static_chs, axis=0) if static_chs else np.zeros((0, hi, wj), dtype=np.float32)

                    # Valid mask for prediction (less strict than training)
                    # Only require HM channel (index 0) to be valid across all timesteps
                    # Component channels can be NaN (will be replaced with 0.0)
                    hm_valid_all_times = np.isfinite(input_dynamic_np[:, 0, :, :]).all(axis=0)  # [H, W]
                    # Only require first static channel (elevation) to be valid
                    stat_valid = np.isfinite(input_static_np[0]) if static_chs else np.ones((hi, wj), dtype=bool)
                    valid_mask = submask & hm_valid_all_times & stat_valid
                    if not np.any(valid_mask):
                        tiles_processed += 1
                        tiles_skipped += 1
                        continue
                    
                    tiles_with_valid += 1
                    tiles_processed += 1
                    
                    # Track which pixels had valid inputs (BEFORE replacing NaN)
                    # This matches the validation code approach (lines 465-467)
                    dynamic_has_nan = ~np.isfinite(input_dynamic_np).all(axis=(0, 1))  # [hi, wj]
                    static_has_nan = ~np.isfinite(input_static_np).all(axis=0) if static_chs else np.zeros((hi, wj), dtype=bool)
                    input_invalid_mask = dynamic_has_nan | static_has_nan  # Pixels to mask in predictions
                    
                    # Add to batch
                    # Replace NaN with 0.0 in normalized space = mean in original space
                    in_dyn = np.nan_to_num(input_dynamic_np, nan=0.0).astype(np.float32)
                    in_stat = np.nan_to_num(input_static_np, nan=0.0).astype(np.float32)
                    lonlat_hw2 = lonlat_grid_for_window(i, j, hi, wj)
                    
                    # Pad to tile size if needed (for edge tiles)
                    if hi < tile or wj < tile:
                        # Pad dynamic: [T, C, hi, wj] -> [T, C, tile, tile]
                        T, C = in_dyn.shape[:2]
                        in_dyn_padded = np.zeros((T, C, tile, tile), dtype=np.float32)
                        in_dyn_padded[:, :, :hi, :wj] = in_dyn
                        in_dyn = in_dyn_padded
                        
                        # Pad static: [C, hi, wj] -> [C, tile, tile]
                        C_stat = in_stat.shape[0]
                        in_stat_padded = np.zeros((C_stat, tile, tile), dtype=np.float32)
                        in_stat_padded[:, :hi, :wj] = in_stat
                        in_stat = in_stat_padded
                        
                        # Pad lonlat: [hi, wj, 2] -> [tile, tile, 2]
                        lonlat_padded = np.zeros((tile, tile, 2), dtype=np.float32)
                        lonlat_padded[:hi, :wj, :] = lonlat_hw2
                        lonlat_hw2 = lonlat_padded
                    
                    if ctx_src is not None:
                        cx = np.stack([
                            np.nan_to_num(ctx_src.read(1, window=win, masked=True).filled(np.nan), nan=0.0),
                            np.nan_to_num(ctx_src.read(2, window=win, masked=True).filled(np.nan), nan=1e4),
                        ], axis=0).astype(np.float32)
                        if hi < tile or wj < tile:
                            padded = np.zeros((2, tile, tile), dtype=np.float32)
                            padded[1] = 1e4
                            padded[:, :hi, :wj] = cx
                            cx = padded
                        batch_contexts.append(cx)
                    if hm_ctx_src is not None:
                        raw = hm_ctx_src.read(hm_ctx_bands, window=win).astype(np.float32)
                        hm_cx = np.where(raw == -32768, 0.0,
                                         raw * np.float32(1.0 / 32767.0)).astype(np.float32)
                        if hi < tile or wj < tile:
                            # Every other source above pads its edge tiles to the full tile;
                            # this one did not, so a region whose extent is not a whole number
                            # of strides handed np.stack a (bands, hi, wj) among (bands, tile,
                            # tile) and it refused. Training never saw this because chips are
                            # always full size -- only the prediction path tiles to an edge.
                            # Zero is what this block already substitutes for the raster's own
                            # -32768 nodata sentinel.
                            padded = np.zeros((len(hm_ctx_bands), tile, tile), dtype=np.float32)
                            padded[:, :hi, :wj] = hm_cx
                            hm_cx = padded
                        batch_hm_contexts.append(hm_cx)
                    batch_inputs_dyn.append(in_dyn)
                    batch_inputs_stat.append(in_stat)
                    batch_lonlats.append(lonlat_hw2)
                    batch_metadata.append((i, j, hi, wj, li0, lj0, li1, lj1, valid_mask, input_invalid_mask))
                
                # Process batch on GPU if we have any valid tiles
                if len(batch_inputs_dyn) > 0:
                    # Stack into batch tensors
                    batch_dyn_tensor = torch.from_numpy(np.stack(batch_inputs_dyn, axis=0)).to(device)  # [B, T, C, H, W]
                    batch_stat_tensor = torch.from_numpy(np.stack(batch_inputs_stat, axis=0)).to(device)  # [B, C, H, W]
                    batch_lonlat_tensor = torch.from_numpy(np.stack(batch_lonlats, axis=0)).to(device)  # [B, H, W, 2]
                    
                    batch_ctx_tensor = (
                        torch.from_numpy(np.stack(batch_contexts, axis=0)).to(device)
                        if batch_contexts else None
                    )
                    batch_hm_tensor = (
                        torch.from_numpy(np.stack(batch_hm_contexts, axis=0)).to(device)
                        if batch_hm_contexts else None
                    )
                    batch_qf = None
                    with torch.no_grad():
                        batch_preds = infer_model(batch_dyn_tensor, batch_stat_tensor,
                                                  lonlat=batch_lonlat_tensor,
                                                  change_context=batch_ctx_tensor,
                                                  hm_context=batch_hm_tensor)  # [B, 12, H, W]
                        if qf_u is not None:
                            # Evaluated once per batch, decoded by the *same* function the
                            # loss uses, so the raster and the objective cannot drift apart.
                            m_ = infer_model.model
                            u_t = torch.as_tensor(qf_u, dtype=batch_preds.dtype,
                                                  device=batch_preds.device)
                            batch_qf = [
                                sp.ppf(u_t).movedim(-1, 1).cpu().numpy()   # [B, n_u, H, W]
                                for sp in splines_from_output(
                                    batch_preds, m_.num_horizons, m_.spline_u_knots,
                                    learn_slopes=m_.spline_learn_slopes,
                                    clamp=m_.spline_clamp())
                            ]
                    
                    # Process each tile in the batch
                    for tile_idx, (i, j, hi, wj, li0, lj0, li1, lj1, valid_mask, input_invalid_mask) in enumerate(batch_metadata):
                        # Extract quantile predictions for this tile (crop to actual size if padded)
                        # batch_preds: [B, 12, H, W] where 12 = 4 horizons × 3 quantiles
                        # Channel ordering: [lower_5yr, central_5yr, upper_5yr, lower_10yr, ...]
                        preds_horizons = {}
                        for h_idx, h_name in enumerate(horizon_names):
                            if h_name not in active_horizons:
                                continue
                            # Extract 3 quantiles for this horizon
                            pred_lower = batch_preds[tile_idx, 3*h_idx, :hi, :wj].detach().cpu().numpy()
                            pred_central = batch_preds[tile_idx, 3*h_idx+1, :hi, :wj].detach().cpu().numpy()
                            pred_upper = batch_preds[tile_idx, 3*h_idx+2, :hi, :wj].detach().cpu().numpy()
                            
                            # Denormalize to [0, 1] scale
                            pred_lower = pred_lower * hm_std + hm_mean
                            pred_central = pred_central * hm_std + hm_mean
                            pred_upper = pred_upper * hm_std + hm_mean
                            
                            # CRITICAL: Mask predictions where inputs had NaN (same as validation code)
                            pred_lower[input_invalid_mask] = np.nan
                            pred_central[input_invalid_mask] = np.nan
                            pred_upper[input_invalid_mask] = np.nan
                            
                            # Store with keys matching accumulator dict
                            preds_horizons[f"{h_name}_lower"] = pred_lower
                            preds_horizons[f"{h_name}_central"] = pred_central
                            preds_horizons[f"{h_name}_upper"] = pred_upper

                            if qf_u is not None:
                                qf = batch_qf[h_idx][tile_idx, :, :hi, :wj]
                                qf = qf * hm_std + hm_mean
                                qf[:, input_invalid_mask] = np.nan
                                for li, lname in enumerate(qf_names):
                                    preds_horizons[f"{h_name}_{lname}"] = qf[li]
                        
                        # Distance-to-edge weights within tile
                        interior = valid_mask.astype(np.uint8)
                        interior[[0, -1], :] = 0
                        interior[:, [0, -1]] = 0
                        weights = distance_transform_edt(interior)
                        weights = np.where(valid_mask, weights, 0.0)
                        
                        if weights.max() > 0:
                            # Accumulate each horizon-quantile combination
                            for key, pred in preds_horizons.items():
                                accum_horizons[key][li0:li1, lj0:lj1] += pred * weights
                            wsum[li0:li1, lj0:lj1] += weights
                        nodata_mask_total[li0:li1, lj0:lj1] |= ~valid_mask
                
                # Progress indicator (after each batch)
                percent = int(100 * tiles_processed / total_tiles)
                if percent != last_percent and percent % 5 == 0:
                    elapsed = time.time() - tile_start_time
                    tiles_per_sec = tiles_processed / elapsed if elapsed > 0 else 0
                    eta_sec = (total_tiles - tiles_processed) / tiles_per_sec if tiles_per_sec > 0 else 0
                    print(f"  Progress: {percent:3d}% ({tiles_processed:,}/{total_tiles:,} tiles) | "
                          f"Speed: {tiles_per_sec:.1f} tiles/s | "
                          f"ETA: {int(eta_sec//60):02d}:{int(eta_sec%60):02d}")
                    last_percent = percent

            # Final blend for all horizon-quantile combinations
            print("\n" + "-"*70)
            print("Blending overlapping tiles for all horizons and quantiles...")
            m = wsum > 0

            # Blend one raster at a time, at the moment it is written, instead of building a
            # dict of every horizon x quantile level first.
            #
            # There are len(active_horizons) * (3 + n_qf_levels) accumulators -- 268 for a
            # four-horizon window at 64 levels. On southern Africa (1.86 Mpx) a full second
            # copy is 2 GB and invisible. On Africa (63.1 Mpx) each array is 0.252 GB, so the
            # copy is 67.6 GB, and because np.full touches every page it is ALL resident,
            # while accum_horizons (np.zeros) stays sparse over ocean. Measured peak was
            # ~98 GB for one fold; two folds in parallel were OOM-killed by the kernel with
            # no traceback. Blending on demand removes that copy entirely: peak becomes the
            # sparse accumulators plus one temporary.
            #
            # The arithmetic is unchanged -- same expression, evaluated later -- so the
            # written values are identical. The quantile levels blend on exactly the same
            # weights as the triple. A weighted average of monotone sequences is monotone, so
            # the blended quantile function is still a quantile function, and because 0.025
            # and 0.975 are levels of the grid the blended bands reproduce the blended
            # lower/upper rasters rather than merely approximating them.
            # In screen mode the kept blocks are exact, but every PROCESSED TILE writes its
            # whole 128 px extent, so a halo around each block also comes out finite -- with
            # incomplete blending, because the tiles that would have contributed to it were
            # skipped. Measured: kept pixels agree with a full run to 3.6e-7 (float32 summation
            # order), the halo to only 3.1e-3, which is the size of the signal. In the ordinary
            # fold hindcast the halo is harmless because it falls outside the fold and the
            # stitcher drops it; here it falls INSIDE the fold, so the scorer would take it.
            _screen_mask = restrict_win if int(
                getattr(args, "predict_subsample_blocks", 0)) > 0 else None

            def _blend(key):
                out_h = np.full((Hwin, Wwin), np.nan, dtype=np.float32)
                out_h[m] = (accum_horizons[key][m] / wsum[m]).astype(np.float32)
                # Clamp predictions to valid range [0, 1]
                out_h[m] = np.clip(out_h[m], 0.0, 1.0)
                if _screen_mask is not None:
                    out_h[~_screen_mask] = np.nan
                return out_h
            
            # Calculate statistics
            num_valid_pixels = m.sum()
            num_total_pixels = Hwin * Wwin
            valid_percent = 100 * num_valid_pixels / num_total_pixels
            
            print(f"✓ Blending complete")
            print(f"  Valid pixels: {num_valid_pixels:,} / {num_total_pixels:,} ({valid_percent:.1f}%)")
            print(f"  Generated 12 predictions (3 quantiles × 4 horizons)")

            # Write GeoTIFF for each horizon-quantile combination
            print("\nWriting output GeoTIFFs...")
            out_profile = ref.profile.copy()
            out_profile.update({
                'height': Hwin,
                'width': Wwin,
                'transform': ref_transform * Affine.translation(c0, r0),
                'count': 1,
                'dtype': 'float32',
                'compress': 'deflate'
            })
            out_dir = Path(args.predict_output_dir) if args.predict_output_dir else (
                Path(os.getcwd()) / 'data' / 'predictions'
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix = output_prefix if output_prefix is not None else (args.predict_output_prefix or "")
            max_year = args.predict_max_target_year

            out_paths = {}
            for h_name, h_year in zip(horizon_names, horizon_years):
                if h_name not in active_horizons:
                    print(f"  · {h_year}: skipped (> --predict_max_target_year {max_year})")
                    continue
                for q_name in quantile_names:
                    key = f"{h_name}_{q_name}"
                    out_path = out_dir / f"{prefix}prediction_{h_year}_{q_name}_blended.tif"
                    with rasterio.open(out_path, 'w', **out_profile) as dst:
                        dst.write(_blend(key), 1)
                    out_paths[key] = out_path
                    print(f"  ✓ {h_year} {q_name}: {out_path}")

                if qf_names:
                    # One multi-band raster per horizon rather than 64 files. int16 x 1/32767
                    # is the ensemble's own storage convention: HM is bounded on [0, 1], so
                    # this is lossless to 3e-5, far below any quantity of interest.
                    qf_profile = out_profile.copy()
                    qf_profile.update(count=len(qf_names), dtype='int16', nodata=-32768,
                                      tiled=True, blockxsize=256, blockysize=256)
                    qf_path = out_dir / f"{prefix}prediction_{h_year}_qf_blended.tif"
                    with rasterio.open(qf_path, 'w', **qf_profile) as dst:
                        for li, lname in enumerate(qf_names):
                            band = _blend(f"{h_name}_{lname}")
                            q = np.where(np.isfinite(band),
                                         np.round(band * 32767.0), -32768)
                            dst.write(np.clip(q, -32768, 32767).astype(np.int16), li + 1)
                            dst.set_band_description(li + 1, f"u={qf_u[li]:.6f}")
                        dst.update_tags(u_levels=",".join(repr(float(v)) for v in qf_u),
                                        scale_factor="3.0518509e-05",
                                        head_family="spline")
                    out_paths[f"{h_name}_qf"] = qf_path
                    print(f"  ✓ {h_year} qf: {qf_path} ({len(qf_names)} bands)")
            
            # Final summary
            elapsed_total = time.time() - start_time
            print("\n" + "="*70)
            print("PREDICTION SUMMARY")
            print("="*70)
            print(f"Total tiles processed: {tiles_processed:,}")
            print(f"  Tiles with valid data: {tiles_with_valid:,}")
            print(f"  Tiles skipped (no data/outside region): {tiles_skipped:,}")
            print(f"Output dimensions: {Hwin} × {Wwin} pixels")
            print(f"Valid output pixels: {num_valid_pixels:,} ({valid_percent:.1f}%)")
            print(f"\nOutput files ({len(out_paths)} written):")
            for h_name, h_year in zip(horizon_names, horizon_years):
                if not any(f"{h_name}_{q}" in out_paths for q in quantile_names):
                    continue
                print(f"  {h_year}:")
                for q_name in quantile_names:
                    key = f"{h_name}_{q_name}"
                    if key in out_paths:
                        print(f"    {q_name}: {out_paths[key]}")
            print(f"\nTotal time: {int(elapsed_total//60):02d}:{int(elapsed_total%60):02d}")
            print("="*70 + "\n")

            # Close sources
            for src in hm_srcs:
                src.close()
            for y in comp_srcs:
                for src in comp_srcs[y]:
                    src.close()
            for src in stat_srcs:
                src.close()
            if ctx_src is not None:
                ctx_src.close()
            if hm_ctx_src is not None:
                hm_ctx_src.close()

            return infer_model

    # Run prediction if requested
    if run_large_area_prediction:
        _release_dataloaders()
        # Use provided checkpoint, or best from training
        pred_checkpoint = checkpoint_path if checkpoint_path else checkpoint_cb.best_model_path
        if pred_checkpoint and args.predict_all_windows:
            # One checkpoint load, all 4 hindcast windows; prefix keeps outputs from colliding
            # (the same target year is produced by several windows).
            base_prefix = args.predict_output_prefix or ""
            cached_model = None
            for win in HINDCAST_WINDOWS:
                win_prefix = f"{base_prefix}w{win[-1]}_"
                print(f"\n{'#'*70}\n# WINDOW {win} -> prefix '{win_prefix}'\n{'#'*70}")
                cached_model = _predict_region_and_write(
                    pred_checkpoint,
                    input_years_override=list(win),
                    output_prefix=win_prefix,
                    infer_model=cached_model,
                )
        elif pred_checkpoint:
            _predict_region_and_write(pred_checkpoint)
        else:
            print("\n⚠️  No checkpoint available for prediction!")
            print("   Provide --checkpoint or train a model (--max_epochs > 0)")
