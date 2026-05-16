#!/usr/bin/env python
"""Visualise individual ensemble samples for a single tile — direct read on
sample-to-sample diversity. Loads a checkpoint, picks a tile from the small
region, samples N members, and writes an N+1-panel figure: obs target + each
sample.

Usage:
    python scripts/sample_grid.py \
        --checkpoint <path> \
        --n_samples 16 \
        --tile_row 100 --tile_col 100 \
        --out outputs/sample_grid.png
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

import torch
from torchgeo_dataloader import (
    hm_files, component_files, static_files, years,
    HumanFootprintChipDataset,
)
from src.models.diffusion_lightning import DiffusionLightningModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--tile_row", type=int, default=None,
                   help="Source row in HM_2000 raster space. If None, auto-pick "
                        "a high-change tile.")
    p.add_argument("--tile_col", type=int, default=None)
    p.add_argument("--tile_size", type=int, default=64)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--m_target", type=float, default=0.5)
    p.add_argument("--m_sample_diverse", action="store_true")
    p.add_argument("--m_sample_min", type=float, default=0.1)
    p.add_argument("--m_sample_max", type=float, default=0.8)
    p.add_argument("--residual_scale_pos", type=float, default=1.0)
    p.add_argument("--residual_scale_neg", type=float, default=1.0)
    p.add_argument("--cond_perturb_std", type=float, default=0.0)
    p.add_argument("--out", default="outputs/sample_grid.png")
    p.add_argument("--vmin", type=float, default=-0.1)
    p.add_argument("--vmax", type=float, default=0.4)
    p.add_argument("--vmin_diff", type=float, default=-0.05)
    p.add_argument("--vmax_diff", type=float, default=0.05)
    return p.parse_args()


def load_dataset_for_stats(tile_size):
    ds = HumanFootprintChipDataset(
        hm_files, component_files, static_files,
        chip_size=tile_size, stat_samples=256,
        target_mode="delta_20yr",
    )
    return ds


def get_tile_data(ds, row, col, tile_size):
    """Read one tile from rasters and return (input_dynamic, input_static, lonlat, hm_t_norm, obs_dhm)."""
    base_year = 2000
    input_years = [1990, 1995, 2000]
    year_to_idx = {y: i for i, y in enumerate(years)}
    hm_srcs = [rasterio.open(p) for p in hm_files]
    comp_srcs = {y: [rasterio.open(p) for p in ds._comp_files[y]] for y in years}
    stat_srcs = [rasterio.open(p) for p in static_files]
    try:
        win = Window(col, row, tile_size, tile_size)
        dyn_ts = []
        for t_idx, y in zip([year_to_idx[y] for y in input_years], input_years):
            channels = []
            arr_hm = hm_srcs[t_idx].read(1, window=win, masked=True).filled(np.nan)
            channels.append((arr_hm - ds.hm_mean) / ds.hm_std)
            for var_name, src in zip(ds.hm_vars, comp_srcs[y]):
                carr = np.nan_to_num(src.read(1, window=win, masked=True).filled(np.nan), nan=0.0)
                channels.append((carr - ds.comp_means[var_name]) / ds.comp_stds[var_name])
            dyn_ts.append(np.stack(channels, axis=0))
        input_dynamic = np.stack(dyn_ts, axis=0).astype(np.float32)
        static_chs = []
        for sidx, src in enumerate(stat_srcs):
            sarr = src.read(1, window=win, masked=True).filled(np.nan)
            if sidx in {0, 4, 5, 6}:
                sarr = np.nan_to_num(sarr, nan=0.0)
            static_chs.append((sarr - ds.static_means[sidx]) / ds.static_stds[sidx])
        input_static = np.stack(static_chs, axis=0).astype(np.float32)
        # lonlat
        ref_tr = hm_srcs[0].transform
        rows = np.arange(row, row + tile_size)
        cols = np.arange(col, col + tile_size)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        xs, ys = rasterio.transform.xy(ref_tr, rr, cc)
        ll = np.stack([np.asarray(xs), np.asarray(ys)], axis=-1).astype(np.float32)
        # hm_t at t=2000
        hm_t_norm = input_dynamic[2, 0:1]
        # obs Δhm = HM_2020 - HM_2000
        idx_2020 = year_to_idx[2020]
        hm20 = hm_srcs[idx_2020].read(1, window=win, masked=True).filled(np.nan)
        hm00 = hm_srcs[year_to_idx[2000]].read(1, window=win, masked=True).filled(np.nan)
        obs = (hm20 - hm00).astype(np.float32)
        return input_dynamic, input_static, ll, hm_t_norm, obs
    finally:
        for s in hm_srcs:
            s.close()
        for v in comp_srcs.values():
            for s in v:
                s.close()
        for s in stat_srcs:
            s.close()


def main():
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    module = DiffusionLightningModule.load_from_checkpoint(args.checkpoint, map_location=device)
    module.train(False)

    print("Loading dataset (stats)...")
    ds = load_dataset_for_stats(args.tile_size)

    if args.tile_row is None or args.tile_col is None:
        # auto-pick a high-change tile from the small region
        args.tile_row = 12500
        args.tile_col = 22500
        print(f"Auto-pick: row={args.tile_row} col={args.tile_col}")
    in_dyn, in_stat, ll, hm_t, obs = get_tile_data(ds, args.tile_row, args.tile_col, args.tile_size)
    bd = torch.from_numpy(in_dyn[None]).to(device)
    bs = torch.from_numpy(in_stat[None]).to(device)
    bll = torch.from_numpy(ll[None]).to(device)
    bhm_t = torch.from_numpy(hm_t[None]).to(device)

    print(f"Sampling {args.n_samples} members...")
    with torch.no_grad():
        m_scalar = torch.tensor([args.m_target], device=device, dtype=torch.float32)
        N = args.n_samples
        if args.m_sample_diverse:
            base = module.assemble_conditioning(bd, bs, bll, bhm_t, m_scalar=m_scalar)
            cond = base.repeat(N, 1, 1, 1)
            m_all = torch.empty(N, device=device, dtype=torch.float32).uniform_(
                args.m_sample_min, args.m_sample_max,
            )
            m_norm = (m_all / max(getattr(module, "m_norm_scale", 0.5), 1e-8)).view(N, 1, 1, 1)
            cond[:, -1:, :, :] = m_norm.expand(N, 1, args.tile_size, args.tile_size).to(cond.dtype)
            if args.cond_perturb_std > 0:
                cond = cond + args.cond_perturb_std * torch.randn_like(cond)
            samples = module.sample(
                cond, n_samples=1,
                num_inference_steps=args.num_inference_steps,
                residual_scale_pos=args.residual_scale_pos,
                residual_scale_neg=args.residual_scale_neg,
            )
            samples = samples.view(N, 1, args.tile_size, args.tile_size)
        else:
            cond = module.assemble_conditioning(bd, bs, bll, bhm_t, m_scalar=m_scalar)
            samples = module.sample(
                cond, n_samples=N,
                num_inference_steps=args.num_inference_steps,
                residual_scale_pos=args.residual_scale_pos,
                residual_scale_neg=args.residual_scale_neg,
            )
            samples = samples.view(N, 1, args.tile_size, args.tile_size)
        # denormalise
        samples_raw = module.denormalize(samples.float())
    samples_dhm = samples_raw.cpu().numpy()[:, 0]  # [N, H, W]
    print(f"Sample stats: per-pixel std mean={samples_dhm.std(axis=0).mean():.4f}  max={samples_dhm.std(axis=0).max():.4f}")
    print(f"  per-tile-max across samples: min={samples_dhm.max(axis=(1,2)).min():.4f}  max={samples_dhm.max(axis=(1,2)).max():.4f}")

    # Plot
    cols = 5
    rows = 1 + (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2), constrained_layout=True)
    axes = np.array(axes).reshape(rows, cols)
    # First row: obs (center)
    for j in range(cols):
        axes[0, j].axis("off")
    obs_show = np.where(np.isfinite(obs), obs, np.nan)
    im = axes[0, cols // 2].imshow(obs_show, vmin=args.vmin, vmax=args.vmax, cmap="RdBu_r")
    axes[0, cols // 2].set_title(f"obs Δhm (max={np.nanmax(obs):.3f})")
    # Remaining: samples
    for n in range(N):
        ri, ci = 1 + n // cols, n % cols
        sshow = samples_dhm[n]
        axes[ri, ci].imshow(sshow, vmin=args.vmin, vmax=args.vmax, cmap="RdBu_r")
        axes[ri, ci].set_title(f"s{n} max={sshow.max():.3f}", fontsize=8)
        axes[ri, ci].axis("off")
    # turn off unused
    for n in range(N, rows * cols - cols):
        ri, ci = 1 + n // cols, n % cols
        if ri < rows:
            axes[ri, ci].axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, label="Δhm")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
