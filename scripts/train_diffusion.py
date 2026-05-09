"""Train the conditional diffusion U-Net on 20yr Δhm.

Usage:
    python scripts/train_diffusion.py --max_epochs 5 --restrict_to_region config/region_to_predict_small.geojson

Defaults are tuned for the dev region; override for full-scale runs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch

try:
    import lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import WandbLogger
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import WandbLogger

from torchgeo_dataloader import get_dataloader
from src.models.diffusion_lightning import DiffusionLightningModule


def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--split_mask_file",
                   default="data/raw/hm_global/split_mask_1000.tif")
    p.add_argument("--restrict_to_region",
                   default="config/region_to_predict_small.geojson",
                   help="GeoJSON region to confine training/val chips to.")
    p.add_argument("--chip_size", type=int, default=64)
    p.add_argument("--train_chips", type=int, default=256)
    p.add_argument("--val_chips", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--stat_samples", type=int, default=512,
                   help="Random windows per raster for normalization-stat sampling. "
                        "Lower (e.g. 64) for quick smoke runs.")
    p.add_argument("--dhm_stats_cache",
                   default="data/raw/hm_global/dhm_stats_20yr.json")

    # Trainer
    p.add_argument("--max_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--precision", default="32-true",
                   help="Lightning precision: '32-true' (Apple Silicon/CPU safe), "
                        "'16-mixed', or 'bf16-mixed' (CUDA/Ampere+).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Model architecture
    p.add_argument("--base_channels", type=int, default=128)
    p.add_argument("--channel_mults", type=int, nargs="+", default=[1, 2, 2, 4])
    p.add_argument("--attention_head_dim", type=int, default=64)

    # Diffusion
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--ensemble_n", type=int, default=16)
    p.add_argument("--use_ema", action="store_true", default=False,
                   help="Track exponential-moving-average UNet weights and use them at sampling.")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--weighted_sampling", action="store_true", default=False,
                   help="Sample chips proportional to (max |Δhm|)^alpha to oversample "
                        "rare high-change cases.")
    p.add_argument("--weight_alpha", type=float, default=1.0)
    p.add_argument("--pixel_weight_alpha", type=float, default=0.0,
                   help="Per-pixel reweighting on v-prediction MSE: weight ∝ (|Δhm|^α + ε). "
                        "0 = uniform; 1 = linear in |Δhm|. Concentrates gradient on rare pixels.")
    p.add_argument("--pixel_weight_eps", type=float, default=0.05)
    p.add_argument("--pattern_loss_weight", type=float, default=0.0,
                   help="Weight for the multi-scale binary-pattern matching loss. 0 = off.")
    p.add_argument("--pattern_thresholds", type=float, nargs="+", default=[0.05, 0.4],
                   help="Thresholds (raw Δhm space) for soft-binarising in the pattern loss.")
    p.add_argument("--pattern_scales", type=int, nargs="+", default=[8, 16],
                   help="Average-pool kernel sizes for tile-pattern matching.")
    p.add_argument("--pattern_temperature", type=float, default=0.02,
                   help="Sigmoid temperature for soft binarisation.")
    p.add_argument("--cfg_dropout_prob", type=float, default=0.0,
                   help="Probability of dropping conditioning during training (CFG). "
                        "0 = off; 0.1 standard. Enables classifier-free guidance at sampling.")
    p.add_argument("--min_snr_gamma", type=float, default=0.0,
                   help="Hang-2023 min-SNR-γ weighting on v-loss (0=off, 5 standard "
                        "for v-prediction). Upweights low-noise / high-SNR steps.")
    p.add_argument("--dhm_transform", choices=("none", "signed_log1p"), default="none",
                   help="Forward Δhm transform before z-score normalisation. "
                        "'signed_log1p' compresses bulk and stretches the heavy tail "
                        "so diffusion can resolve high-magnitude change events.")
    p.add_argument("--dhm_log_scale", type=float, default=0.05,
                   help="Knee scale for signed_log1p: transform(x) = sign(x) * "
                        "log1p(|x|/scale). Smaller = more tail stretch.")
    p.add_argument("--exloss_lambda", type=float, default=0.0,
                   help="Tier 2B asymmetric Exloss (Gong 2024). 0=off; ~1 standard. "
                        "Replaces pixel_weight_alpha when >0; under-predictions of "
                        "high-magnitude pixels are penalised more than over.")
    p.add_argument("--wasserstein_loss_weight", type=float, default=0.0,
                   help="Tier 2C marginal Wasserstein regulariser (WassDiff). "
                        "0=off; ~0.05 standard. Forces the predicted pixel-value "
                        "histogram to match the target distribution.")
    p.add_argument("--use_magnitude_cond", action="store_true", default=False,
                   help="Tier 2A FIDE-style block-maxima conditioning. Adds a "
                        "per-chip max|Δhm| scalar as an extra conditioning channel.")
    p.add_argument("--m_dropout_prob", type=float, default=0.1,
                   help="Probability of dropping the magnitude scalar to null "
                        "during training (so the model also learns the m-uncond "
                        "distribution). Standard 0.1.")
    p.add_argument("--m_norm_scale", type=float, default=0.5,
                   help="Divisor used to normalise raw |Δhm| max into the "
                        "magnitude conditioning channel. ~max realistic |Δhm|.")
    p.add_argument("--use_mean_head", action="store_true", default=False,
                   help="Tier 3A: enable a small deterministic conv head that "
                        "predicts the conditional-mean Δhm (CorrDiff). The "
                        "diffusion U-Net then learns only the residual, freeing "
                        "its stochastic capacity for rare events.")
    p.add_argument("--mean_head_hidden", type=int, default=64,
                   help="Hidden channel width for the CorrDiff mean head.")
    p.add_argument("--mean_loss_weight", type=float, default=1.0,
                   help="Multiplier on the mean head's MSE loss term.")

    # Location encoder
    p.add_argument("--use_location_encoder", action="store_true", default=True)
    p.add_argument("--no_location_encoder", dest="use_location_encoder",
                   action="store_false")
    p.add_argument("--locenc_out_channels", type=int, default=8)
    p.add_argument("--locenc_legendre_polys", type=int, default=10)

    # W&B
    p.add_argument("--wandb_project", default="spatio-temporal-diffusion")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--disable_wandb", action="store_true")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a checkpoint to resume from.")
    p.add_argument("--default_root_dir", default="models/checkpoints_diffusion")

    return p.parse_args()


def make_loaders(args):
    common = dict(
        chip_size=args.chip_size,
        timesteps=3,
        target_mode="delta_20yr",
        restrict_to_region=args.restrict_to_region,
        dhm_stats_cache=args.dhm_stats_cache,
        split_mask_file=args.split_mask_file,
        include_components=True,
        stat_samples=args.stat_samples,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        dhm_transform=args.dhm_transform,
        dhm_log_scale=args.dhm_log_scale,
    )
    train_loader = get_dataloader(
        batch_size=args.batch_size,
        chips_per_epoch=args.train_chips,
        mode="random",
        split_value=1,
        weighted_sampling=args.weighted_sampling,
        weight_alpha=args.weight_alpha,
        **common,
    )
    # Val loader is uniform: we want val/loss to reflect the natural chip
    # distribution, not the importance-weighted training distribution.
    val_loader = get_dataloader(
        batch_size=args.batch_size,
        chips_per_epoch=args.val_chips,
        mode="random",
        split_value=2,
        **common,
    )
    return train_loader, val_loader


def compute_cond_channels(sample_batch, args):
    """Total conditioning channels = T*C_dyn + C_static + locenc_out + 1 (hm_t)
    + 1 (block-maxima M) when use_magnitude_cond is on."""
    T = sample_batch["input_dynamic"].shape[1]
    C_dyn = sample_batch["input_dynamic"].shape[2]
    C_static = sample_batch["input_static"].shape[1]
    locenc = args.locenc_out_channels if args.use_location_encoder else 0
    extra = 1 if args.use_magnitude_cond else 0
    return T * C_dyn + C_static + locenc + 1 + extra


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    print("Building dataloaders...")
    train_loader, val_loader = make_loaders(args)

    print("Introspecting one batch to determine conditioning channel count...")
    sample_batch = next(iter(train_loader))
    cond_channels = compute_cond_channels(sample_batch, args)
    print(f"  cond_channels = {cond_channels}")

    locenc_kwargs = None
    if args.use_location_encoder:
        locenc_kwargs = dict(
            backbone=("sphericalharmonics", "siren"),
            out_channels=args.locenc_out_channels,
            hparams=dict(legendre_polys=args.locenc_legendre_polys,
                         dim_hidden=64, num_layers=2,
                         optimizer=dict(lr=1e-4, wd=1e-3)),
        )

    train_ds = train_loader.dataset
    module = DiffusionLightningModule(
        cond_channels=cond_channels,
        sample_size=args.chip_size,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        attention_head_dim=args.attention_head_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        ensemble_n=args.ensemble_n,
        location_encoder_kwargs=locenc_kwargs,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        pixel_weight_alpha=args.pixel_weight_alpha,
        pixel_weight_eps=args.pixel_weight_eps,
        pattern_loss_weight=args.pattern_loss_weight,
        pattern_thresholds=tuple(args.pattern_thresholds),
        pattern_scales=tuple(args.pattern_scales),
        pattern_temperature=args.pattern_temperature,
        dhm_mean=float(train_ds.dhm_mean),
        dhm_std=float(train_ds.dhm_std),
        cfg_dropout_prob=args.cfg_dropout_prob,
        min_snr_gamma=args.min_snr_gamma,
        dhm_transform=args.dhm_transform,
        dhm_log_scale=args.dhm_log_scale,
        exloss_lambda=args.exloss_lambda,
        wasserstein_loss_weight=args.wasserstein_loss_weight,
        use_magnitude_cond=args.use_magnitude_cond,
        m_dropout_prob=args.m_dropout_prob,
        m_norm_scale=args.m_norm_scale,
        use_mean_head=args.use_mean_head,
        mean_head_hidden=args.mean_head_hidden,
        mean_loss_weight=args.mean_loss_weight,
    )
    n_params = sum(p.numel() for p in module.parameters())
    print(f"  module param count: {n_params/1e6:.1f}M")

    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=2,
            filename="dhm-diffusion-epoch{epoch:02d}-valloss{val/loss:.4f}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = None
    if not args.disable_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            log_model=True,
        )
        logger.log_hyperparams(vars(args) | {"cond_channels": cond_channels,
                                             "param_count_m": n_params / 1e6})

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        default_root_dir=args.default_root_dir,
        accelerator="auto",
        log_every_n_steps=10,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
