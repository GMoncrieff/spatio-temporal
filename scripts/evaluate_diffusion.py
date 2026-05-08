"""Evaluate a diffusion-prediction GeoTIFF against observed Δhm.

Computes:
- Aggregate-tile MAE (mean Δhm per non-overlapping NxN block).
- Per-tile histogram intersection on the rarity-weighted bin schedule used by
  the baseline (matches baselines/convlstm/models/histogram_loss.py).
- 3-panel observation/prediction/error map.
- Per-tile MAE and histogram-intersection summaries.

Writes a markdown report to docs/diffusion_v1_results.md.

Usage:
    python scripts/evaluate_diffusion.py \
        --pred_dir data/predictions_diffusion \
        --predict_region config/region_to_predict_small.geojson \
        --tile_size 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import rasterio
from rasterio import features as rio_features
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, transform as shp_transform
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from torchgeo_dataloader import _resolve

HIST_BIN_EDGES = np.array([-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", default="data/predictions_diffusion")
    p.add_argument("--predict_region",
                   default="config/region_to_predict_small.geojson")
    p.add_argument("--tile_size", type=int, default=16,
                   help="Size of non-overlapping tiles for aggregate metrics.")
    p.add_argument("--output_dir", default="outputs/diffusion_v1")
    p.add_argument("--report", default="docs/diffusion_v1_results.md")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a trained checkpoint. If set, generates a "
                        "5-site × 5-sample comparison figure and pulls the "
                        "exact model config into the report.")
    p.add_argument("--n_sites", type=int, default=5)
    p.add_argument("--n_samples_per_site", type=int, default=5)
    p.add_argument("--num_inference_steps", type=int, default=12)
    p.add_argument("--split_mask_file",
                   default="data/raw/hm_global/split_mask_1000.tif")
    p.add_argument("--dhm_stats_cache",
                   default="data/raw/hm_global/dhm_stats_20yr.json")
    p.add_argument("--wandb_run", default=None,
                   help="Optional W&B run path 'entity/project/runid' to embed.")
    return p.parse_args()


def load_observed_dhm(pred_bounds, pred_shape):
    """Read HM_2000 and HM_2020 over the prediction's geographic window."""
    hm00_path = _resolve("HM_2000_AA_1000.tiff")
    hm20_path = _resolve("HM_2020_AA_1000.tiff")
    out = {}
    for key, path in (("hm00", hm00_path), ("hm20", hm20_path)):
        with rasterio.open(path) as src:
            win = rasterio.windows.from_bounds(*pred_bounds, transform=src.transform)
            arr = src.read(
                1, window=win, masked=True,
                out_shape=pred_shape,  # snap to prediction grid
            ).filled(np.nan)
            out[key] = arr
    dhm_obs = out["hm20"] - out["hm00"]
    return dhm_obs


def region_mask_for(pred_path, region_geojson):
    with rasterio.open(pred_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_h, ref_w = ref.height, ref.width
    with open(region_geojson) as f:
        gj = json.load(f)
    if gj.get("type") == "FeatureCollection":
        geoms = [shape(feat["geometry"]) for feat in gj["features"]]
    elif gj.get("type") == "Feature":
        geoms = [shape(gj["geometry"])]
    else:
        geoms = [shape(gj)]
    if ref_crs and ref_crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
        transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
        geoms = [shp_transform(lambda x, y: transformer.transform(x, y), g) for g in geoms]
    geom = unary_union(geoms)
    mask = rio_features.geometry_mask(
        [mapping(geom)], out_shape=(ref_h, ref_w),
        transform=ref_transform, invert=True,
    )
    return mask  # True inside region


def tile_means(arr, valid, tile_size):
    """Aggregate to non-overlapping tiles. Returns (means, counts)."""
    H, W = arr.shape
    Ht = (H // tile_size) * tile_size
    Wt = (W // tile_size) * tile_size
    a = np.where(valid, arr, np.nan)[:Ht, :Wt]
    n_h, n_w = Ht // tile_size, Wt // tile_size
    a_blocks = a.reshape(n_h, tile_size, n_w, tile_size)
    counts = np.isfinite(a_blocks).sum(axis=(1, 3))
    means = np.nanmean(a_blocks, axis=(1, 3))
    return means, counts


def per_tile_histograms(arr, valid, tile_size, bin_edges):
    """For each non-overlapping tile, return [n_tile_h, n_tile_w, n_bins] histograms."""
    H, W = arr.shape
    Ht = (H // tile_size) * tile_size
    Wt = (W // tile_size) * tile_size
    n_h, n_w = Ht // tile_size, Wt // tile_size
    n_bins = len(bin_edges) - 1
    hists = np.zeros((n_h, n_w, n_bins), dtype=np.float64)
    for ti in range(n_h):
        for tj in range(n_w):
            i0 = ti * tile_size
            j0 = tj * tile_size
            block = arr[i0:i0 + tile_size, j0:j0 + tile_size]
            v = valid[i0:i0 + tile_size, j0:j0 + tile_size]
            vals = block[v & np.isfinite(block)]
            if vals.size == 0:
                continue
            counts, _ = np.histogram(vals, bins=bin_edges)
            total = counts.sum()
            if total > 0:
                hists[ti, tj] = counts / total
    return hists


def histogram_intersection(p, q):
    """Sum over bins of min(p, q). Both shape [..., n_bins], normalized."""
    return np.minimum(p, q).sum(axis=-1)


def panel_map(obs, pred, mask, out_path, vmin, vmax):
    """Three-panel map: observed | predicted | error."""
    err = pred - obs
    show = lambda a: np.where(mask, a, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    cmap = "RdBu_r"
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    for ax, arr, title in [
        (axes[0], obs, "Observed Δhm (2020 − 2000)"),
        (axes[1], pred, "Predicted Δhm (median over 16 samples)"),
        (axes[2], err, "Error (pred − obs)"),
    ]:
        im = ax.imshow(show(arr), cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def metric_summary_plots(tile_mae, tile_xinter, tile_counts, out_path):
    """Two-panel summary: MAE histogram + intersection scatter."""
    keep = tile_counts > 0
    mae_flat = tile_mae[keep]
    xinter_flat = tile_xinter[keep]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].hist(mae_flat, bins=30, color="steelblue", edgecolor="black", linewidth=0.3)
    axes[0].axvline(np.median(mae_flat), color="red", linestyle="--",
                    label=f"median = {np.median(mae_flat):.4f}")
    axes[0].set_xlabel("|mean(pred) − mean(obs)| per tile")
    axes[0].set_ylabel("Tile count")
    axes[0].set_title("Per-tile aggregate-mean MAE")
    axes[0].legend()

    axes[1].hist(xinter_flat, bins=30, color="forestgreen", edgecolor="black", linewidth=0.3)
    axes[1].axvline(np.median(xinter_flat), color="red", linestyle="--",
                    label=f"median = {np.median(xinter_flat):.3f}")
    axes[1].set_xlabel("Histogram intersection (1 = perfect match)")
    axes[1].set_ylabel("Tile count")
    axes[1].set_title("Per-tile histogram intersection")
    axes[1].set_xlim(0, 1)
    axes[1].legend()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def coverage_rate(obs, q025, q975, valid):
    """Fraction of valid pixels with q025 ≤ obs ≤ q975. ~0.95 = well-calibrated."""
    if valid.sum() == 0:
        return float("nan")
    in_band = (obs >= q025) & (obs <= q975) & valid
    return float(in_band.sum() / valid.sum())


def per_bin_coverage(obs, q025, q975, valid, bin_edges):
    """Stratify coverage by Δhm magnitude bin.

    For each bin [low, high), report:
      - n: number of valid pixels whose observed Δhm falls in the bin,
      - coverage: fraction of those pixels where q025 ≤ obs ≤ q975.

    Pixels in the highest bin use closed-on-the-right semantics so the
    end of the schedule (e.g. Δhm == 1.0) is not dropped.
    """
    in_band = (obs >= q025) & (obs <= q975)
    out = []
    n_bins = len(bin_edges) - 1
    for i in range(n_bins):
        low = float(bin_edges[i])
        high = float(bin_edges[i + 1])
        if i < n_bins - 1:
            in_bin = (obs >= low) & (obs < high)
        else:
            in_bin = (obs >= low) & (obs <= high)
        bin_mask = valid & in_bin
        n = int(bin_mask.sum())
        if n == 0:
            cov = float("nan")
        else:
            cov = float((in_band & bin_mask).sum() / n)
        out.append({"bin": [low, high], "n": n, "coverage": cov})
    return out


def model_summary_from_checkpoint(checkpoint_path):
    """Load a checkpoint and pull a model+training summary dict (no torch needed for parts)."""
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {}) or {}
    state = ckpt.get("state_dict", {}) or {}
    n_params = sum(v.numel() for v in state.values()
                   if hasattr(v, "numel") and v.dtype.is_floating_point) if state else None
    return dict(
        cond_channels=hp.get("cond_channels"),
        sample_size=hp.get("sample_size"),
        base_channels=hp.get("base_channels"),
        channel_mults=tuple(hp.get("channel_mults", ())),
        attention_head_dim=hp.get("attention_head_dim"),
        layers_per_block=hp.get("layers_per_block"),
        attention_at_low_two=hp.get("attention_at_low_two"),
        lr=hp.get("lr"),
        weight_decay=hp.get("weight_decay"),
        num_train_timesteps=hp.get("num_train_timesteps"),
        num_inference_steps=hp.get("num_inference_steps"),
        ensemble_n=hp.get("ensemble_n"),
        location_encoder_kwargs=hp.get("location_encoder_kwargs"),
        n_params=n_params,
        epoch=ckpt.get("epoch"),
        global_step=ckpt.get("global_step"),
    )


def sample_at_sites(args, n_sites, n_samples, num_inference_steps):
    """Sample n_samples from the trained model at n_sites random valid chips.

    Returns a list of dicts: {observed: [H,W], samples: [N,H,W], valid_mask: [H,W]}
    in raw Δhm units (denormalized).
    """
    import torch
    from torchgeo_dataloader import get_dataloader
    from src.models.diffusion_lightning import DiffusionLightningModule

    print("Loading checkpoint for sampling at sites...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    module = DiffusionLightningModule.load_from_checkpoint(args.checkpoint, map_location=device)
    module.train(False)

    print(f"Building site loader (region={args.predict_region}, n_sites={n_sites})...")
    loader = get_dataloader(
        batch_size=1,
        chip_size=64,
        chips_per_epoch=n_sites,
        target_mode="delta_20yr",
        restrict_to_region=args.predict_region,
        dhm_stats_cache=args.dhm_stats_cache,
        split_mask_file=args.split_mask_file,
        split_value=2,            # validation split (held-out)
        stat_samples=64,           # quick (we only use Δhm stats here)
        num_workers=0,
        mode="random",
    )
    ds = loader.dataset
    dhm_mean = ds.dhm_mean
    dhm_std = ds.dhm_std

    print(f"Sampling {n_samples} times at each of {n_sites} sites...")
    sites = []
    for batch in loader:
        with torch.no_grad():
            input_dynamic = batch["input_dynamic"].to(device)
            input_static = batch["input_static"].to(device)
            lonlat = batch["lonlat"].to(device)
            hm_t_normalized = batch["hm_t_normalized"].to(device)
            cond = module.assemble_conditioning(
                input_dynamic, input_static, lonlat, hm_t_normalized,
            )
            samples = module.sample(
                cond, n_samples=n_samples,
                num_inference_steps=num_inference_steps,
            )  # [N, B=1, 1, H, W]
        samples_unnorm = samples[:, 0, 0].cpu().float().numpy() * dhm_std + dhm_mean
        target_unnorm = batch["target_dhm"][0, 0].cpu().numpy() * dhm_std + dhm_mean
        sites.append(dict(
            observed=target_unnorm,
            samples=samples_unnorm,
            valid_mask=batch["valid_mask"][0].cpu().numpy(),
        ))
        if len(sites) >= n_sites:
            break
    return sites


def plot_sample_grid(sites, out_path, vmin=-0.1, vmax=0.4):
    """N rows (sites) × (1 + n_samples) cols (observed + samples).

    Color scale is fixed across all panels to make sample-to-sample and
    site-to-site comparisons honest. Defaults span the meaningful Δhm range:
    a small loss of modification on the negative side, a larger gain on the
    positive side (most change in this dataset is positive).
    """
    n_sites = len(sites)
    n_samples = sites[0]["samples"].shape[0]

    masked_obs = [np.where(s["valid_mask"], s["observed"], np.nan) for s in sites]
    masked_samp = [np.where(s["valid_mask"][None], s["samples"], np.nan) for s in sites]

    fig, axes = plt.subplots(
        n_sites, n_samples + 1,
        figsize=((n_samples + 1) * 2.0, n_sites * 2.0),
        constrained_layout=True,
    )
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    for r, site in enumerate(sites):
        axes[r, 0].imshow(masked_obs[r], cmap="RdBu_r", norm=norm)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("Observed", fontsize=10)
        axes[r, 0].set_ylabel(f"site {r + 1}", fontsize=9)
        for c in range(n_samples):
            ax = axes[r, c + 1]
            im = ax.imshow(masked_samp[r][c], cmap="RdBu_r", norm=norm)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"sample {c + 1}", fontsize=10)
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", shrink=0.8,
                        fraction=0.04, pad=0.02)
    cbar.set_label("Δhm")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_bin_coverage_table(per_bin):
    if not per_bin:
        return "| (no q025/q975 rasters present) | | |"
    rows = []
    for entry in per_bin:
        low, high = entry["bin"]
        n = entry["n"]
        cov = entry["coverage"]
        cov_s = f"{cov:.3f}" if cov == cov else "—"
        rows.append(f"| `[{low:>+7.3f}, {high:>+6.3f}]` | {n:,} | {cov_s} |")
    return "\n".join(rows)


def write_report(metrics, args, report_path, panel_path, summary_path,
                 model_info=None, sample_grid_path=None):
    rel = lambda p: os.path.relpath(p, os.path.dirname(report_path))

    model_section = ""
    if model_info is not None:
        locenc = model_info.get("location_encoder_kwargs") or {}
        locenc_str = (
            f"`{locenc.get('backbone', '?')}`, "
            f"out_channels={locenc.get('out_channels', '?')}"
            if locenc else "disabled"
        )
        n_params = model_info.get("n_params")
        params_str = f"{n_params/1e6:.1f} M" if n_params else "n/a"
        model_section = f"""## Model & checkpoint

- Checkpoint: `{args.checkpoint}`
- Stopping epoch: {model_info.get('epoch')} (global step {model_info.get('global_step')})
- Architecture: `ConditionalDiffusionUNet` (`diffusers.UNet2DModel`)
  - sample_size = {model_info.get('sample_size')}
  - base_channels = {model_info.get('base_channels')}
  - channel_mults = {list(model_info.get('channel_mults') or [])}
  - attention_head_dim = {model_info.get('attention_head_dim')}, attention_at_low_two = {model_info.get('attention_at_low_two')}
  - layers_per_block = {model_info.get('layers_per_block')}
  - cond_channels = {model_info.get('cond_channels')} (3 timesteps × 11 dyn + 7 static + locenc + 1 hm_t)
  - parameter count = **{params_str}**
- Diffusion: `DDPMScheduler(prediction_type="v_prediction", beta_schedule="squaredcos_cap_v2")`
  - num_train_timesteps = {model_info.get('num_train_timesteps')}
  - inference sampler: DDIM at {args.num_inference_steps} steps
- LocationEncoder: {locenc_str}
- Optimizer: `AdamW(lr={model_info.get('lr')}, weight_decay={model_info.get('weight_decay')})`
"""
    wandb_section = ""
    if args.wandb_run:
        wandb_section = f"\n- W&B run: [{args.wandb_run}](https://wandb.ai/{args.wandb_run})\n"

    sample_section = ""
    if sample_grid_path is not None:
        sample_section = f"""

## Random samples vs. observed change

5 randomly-drawn validation chips (rows). Column 1 = observed Δhm
(2020 − 2000); columns 2–6 = independent draws from the trained diffusion
model's conditional posterior. Sample variability captures the model's
uncertainty about *where* and *how much* change occurs.

![]({rel(sample_grid_path)})
"""

    md = f"""# Diffusion v1 — Sanity Evaluation on the Small Region

## Setup

- Branch: `diffusion`
- Region: `{args.predict_region}`
- Aggregate-tile size: **{args.tile_size}×{args.tile_size}** pixels (10 km blocks).
- Histogram bins (rarity-weighted, matches baseline `histogram_loss.py`):
  `{HIST_BIN_EDGES.tolist()}`.{wandb_section}
{model_section}
## Results

| Metric | Value |
|---|---|
| Tiles with valid coverage | {metrics['n_tiles_valid']:,} of {metrics['n_tiles_total']:,} |
| Tile-mean MAE (median) | **{metrics['mae_median']:.4f}** |
| Tile-mean MAE (mean) | {metrics['mae_mean']:.4f} |
| Tile-mean MAE (95th %ile) | {metrics['mae_p95']:.4f} |
| Histogram intersection (median) | **{metrics['xinter_median']:.3f}** |
| Histogram intersection (mean) | {metrics['xinter_mean']:.3f} |
| Pearson r (predicted vs. observed tile-mean Δhm) | {metrics['pearson_r']:.3f} |
| Coverage rate (q025 ≤ obs ≤ q975), all bins | {metrics.get('coverage_rate', float('nan')):.3f} (target ≈ 0.95) |

### Coverage stratified by Δhm change bin

How often the observed Δhm falls inside the predicted (q025, q975) band, broken
out by magnitude bin. The overall coverage number is dominated by the no-change
bin, where most pixels live; the harder-to-cover bins are the rare large-change
ones where the model has to express genuine uncertainty.

| Δhm bin | n pixels | Coverage |
|---|---:|---:|
{_render_bin_coverage_table(metrics.get('per_bin_coverage'))}

## Figures

- Map comparison: ![]({rel(panel_path)})
- Tile-level summaries: ![]({rel(summary_path)})
{sample_section}
## Caveats / Notes

- The aggregate tile is the smallest meaningful spatial unit at which we can
  *fairly* compare a generative ensemble to a deterministic observation — pixel
  agreement is not the goal of v1 (see migration plan).
- Prediction stride and ensemble size may have been reduced for iteration
  speed on Apple Silicon; widen them for the data-paper run.
- Coverage rate is computed on the (q025, q975) band — if it drifts far below
  ~0.95, the model is over-confident; if much higher, the bands are too wide.
"""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(md)


def main():
    args = parse_args()
    pred_path = os.path.join(args.pred_dir, "prediction_dhm_2020_median.tif")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}. Run "
            f"scripts/predict_region_diffusion.py first."
        )

    print(f"Loading prediction: {pred_path}")
    with rasterio.open(pred_path) as pr:
        pred = pr.read(1)
        pred_bounds = pr.bounds
        pred_shape = (pr.height, pr.width)

    print("Reading HM 2000/2020 over the same window...")
    obs = load_observed_dhm(pred_bounds, pred_shape)

    print(f"Building region mask from {args.predict_region}...")
    region_mask = region_mask_for(pred_path, args.predict_region)

    valid = (
        np.isfinite(pred) & np.isfinite(obs) & region_mask
    )
    n_valid_pixels = int(valid.sum())
    print(f"Valid pixels (pred ∩ obs ∩ region): {n_valid_pixels:,} / {pred.size:,}")
    if n_valid_pixels == 0:
        raise RuntimeError("No overlap between prediction and observation; "
                           "check region geometry and stats")

    print(f"Aggregating to {args.tile_size}×{args.tile_size} tiles...")
    pred_tile, counts_p = tile_means(pred, valid, args.tile_size)
    obs_tile, counts_o = tile_means(obs, valid, args.tile_size)
    counts = np.minimum(counts_p, counts_o)
    keep = counts > 0
    n_tiles_total = pred_tile.size
    n_tiles_valid = int(keep.sum())

    tile_mae = np.abs(pred_tile - obs_tile)
    mae_flat = tile_mae[keep]

    pred_hists = per_tile_histograms(pred, valid, args.tile_size, HIST_BIN_EDGES)
    obs_hists = per_tile_histograms(obs, valid, args.tile_size, HIST_BIN_EDGES)
    tile_xinter = histogram_intersection(pred_hists, obs_hists)
    xinter_flat = tile_xinter[keep]

    if mae_flat.size and (np.std(pred_tile[keep]) > 0 and np.std(obs_tile[keep]) > 0):
        pearson_r = float(np.corrcoef(pred_tile[keep], obs_tile[keep])[0, 1])
    else:
        pearson_r = float("nan")

    # Coverage: how often the observed Δhm falls inside the predicted band.
    q025_path = os.path.join(args.pred_dir, "prediction_dhm_2020_q025.tif")
    q975_path = os.path.join(args.pred_dir, "prediction_dhm_2020_q975.tif")
    cov = float("nan")
    bin_cov = None
    if os.path.exists(q025_path) and os.path.exists(q975_path):
        with rasterio.open(q025_path) as r:
            q025 = r.read(1)
        with rasterio.open(q975_path) as r:
            q975 = r.read(1)
        cov_valid = valid & np.isfinite(q025) & np.isfinite(q975)
        cov = coverage_rate(obs, q025, q975, cov_valid)
        bin_cov = per_bin_coverage(obs, q025, q975, cov_valid, HIST_BIN_EDGES)

    metrics = dict(
        n_tiles_total=n_tiles_total,
        n_tiles_valid=n_tiles_valid,
        mae_median=float(np.median(mae_flat)),
        mae_mean=float(np.mean(mae_flat)),
        mae_p95=float(np.percentile(mae_flat, 95)),
        xinter_median=float(np.median(xinter_flat)),
        xinter_mean=float(np.mean(xinter_flat)),
        pearson_r=pearson_r,
        coverage_rate=cov,
        per_bin_coverage=bin_cov,
    )
    print("Metrics:")
    for k, v in metrics.items():
        if k == "per_bin_coverage":
            if v is None:
                continue
            print("  per_bin_coverage:")
            for entry in v:
                low, high = entry["bin"]
                cov_v = entry["coverage"]
                cov_s = f"{cov_v:.3f}" if cov_v == cov_v else "nan"  # nan check
                print(f"    [{low:>+7.3f}, {high:>+6.3f}]  n={entry['n']:>10,d}  coverage={cov_s}")
        else:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "map_comparison.png"
    summary_path = out_dir / "tile_metrics.png"
    abs_max = max(abs(np.nanpercentile(obs[valid], 1)),
                  abs(np.nanpercentile(obs[valid], 99)),
                  abs(np.nanpercentile(pred[valid], 1)),
                  abs(np.nanpercentile(pred[valid], 99)))
    panel_map(obs, pred, valid, panel_path, vmin=-abs_max, vmax=abs_max)
    metric_summary_plots(tile_mae, tile_xinter, counts, summary_path)
    print(f"Wrote: {panel_path}")
    print(f"Wrote: {summary_path}")

    sample_grid_path = None
    model_info = None
    if args.checkpoint:
        model_info = model_summary_from_checkpoint(args.checkpoint)
        try:
            sites = sample_at_sites(
                args,
                n_sites=args.n_sites,
                n_samples=args.n_samples_per_site,
                num_inference_steps=args.num_inference_steps,
            )
            sample_grid_path = out_dir / "samples_vs_observed.png"
            plot_sample_grid(sites, sample_grid_path)
            print(f"Wrote: {sample_grid_path}")
        except Exception as e:
            print(f"⚠ Could not generate sample grid: {e}")
            sample_grid_path = None

    write_report(metrics, args, args.report, panel_path, summary_path,
                 model_info=model_info, sample_grid_path=sample_grid_path)
    print(f"Wrote report: {args.report}")

    metrics_json = out_dir / "metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics JSON: {metrics_json}")


if __name__ == "__main__":
    main()
