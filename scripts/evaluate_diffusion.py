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


def per_tile_max(arr, valid, tile_size):
    """Per non-overlapping tile, return the spatial max of valid pixels."""
    H, W = arr.shape
    Ht = (H // tile_size) * tile_size
    Wt = (W // tile_size) * tile_size
    a = np.where(valid, arr, np.nan)[:Ht, :Wt]
    n_h, n_w = Ht // tile_size, Wt // tile_size
    blocks = a.reshape(n_h, tile_size, n_w, tile_size)
    counts = np.isfinite(blocks).sum(axis=(1, 3))
    maxes = np.nanmax(blocks, axis=(1, 3))
    return maxes, counts


def qq_max_of_field(pred, obs, valid, tile_size, q=np.linspace(0.0, 1.0, 51)):
    """Q-Q comparison of per-tile spatial max(Δhm).

    The single most diagnostic metric for "magnitude-correct somewhere in the
    field": for each tile take max(pred) and max(obs); compare their CDFs.
    Slope < 1 in the upper-tail = systematic under-prediction of magnitude.
    """
    pred_max, c_p = per_tile_max(pred, valid, tile_size)
    obs_max, c_o = per_tile_max(obs, valid, tile_size)
    keep = (c_p > 0) & (c_o > 0) & np.isfinite(pred_max) & np.isfinite(obs_max)
    if keep.sum() == 0:
        return None
    pred_q = np.quantile(pred_max[keep], q)
    obs_q = np.quantile(obs_max[keep], q)
    return {
        "quantiles": q.tolist(),
        "pred_q": pred_q.tolist(),
        "obs_q": obs_q.tolist(),
        "n_tiles": int(keep.sum()),
        "pred_max_global": float(np.max(pred_max[keep])),
        "obs_max_global": float(np.max(obs_max[keep])),
    }


def r95p_index(pred, obs, valid, percentile=95):
    """R95p: ratio of total Δhm mass above the p-th percentile of observations.

    >=1.0 means the model captures (or over-states) the upper-tail mass; <<1.0
    means systematic under-prediction. Standard in the precipitation-extremes
    literature (Aich et al. GMD 2026; FuXi-Extreme).
    """
    obs_pos = obs[valid & (obs > 0)]
    if obs_pos.size < 32:
        return None
    p = float(np.percentile(obs_pos, percentile))
    obs_mass = float(np.sum(np.maximum(obs[valid] - p, 0.0)))
    pred_mass = float(np.sum(np.maximum(pred[valid] - p, 0.0)))
    if obs_mass <= 0:
        return None
    return {
        "percentile": percentile,
        "threshold": p,
        "obs_tail_mass": obs_mass,
        "pred_tail_mass": pred_mass,
        "ratio": pred_mass / obs_mass,
    }


def tail_exceedance_metrics(pred, obs, q975, valid, thresholds=(0.05, 0.1, 0.2, 0.4)):
    """For each threshold t, report deterministic POD/FAR/CSI on (pred>t vs obs>t)
    plus ensemble-exceedance support from q975.

    POD = TP/(TP+FN), CSI = TP/(TP+FN+FP). q975-based "soft" exceedance reports
    the fraction of pixels where the predicted upper bound at least crosses t —
    a calibration check on the ensemble's tail width.
    """
    out = []
    for t in thresholds:
        obs_pos = (obs > t) & valid
        pred_pos = (pred > t) & valid
        tp = int((obs_pos & pred_pos).sum())
        fn = int((obs_pos & ~pred_pos).sum())
        fp = int((~obs_pos & pred_pos).sum())
        n_obs = int(obs_pos.sum())
        pod = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else float("nan")
        far = fp / (tp + fp) if (tp + fp) > 0 else float("nan")
        # Ensemble-tail support from q975 (model thinks P(X>t)>2.5%)
        if q975 is not None:
            q975_pos = (q975 > t) & valid
            n_q975 = int(q975_pos.sum())
            soft_pod = (
                int((obs_pos & q975_pos).sum()) / n_obs if n_obs > 0 else float("nan")
            )
        else:
            n_q975 = None
            soft_pod = float("nan")
        out.append({
            "threshold": float(t),
            "n_obs_exceed": n_obs,
            "n_pred_exceed": int(pred_pos.sum()),
            "n_q975_exceed": n_q975,
            "POD": pod,
            "CSI": csi,
            "FAR": far,
            "q975_soft_POD": soft_pod,
        })
    return out


def plot_qq_max_of_field(qq, out_path):
    """Diagonal Q-Q of per-tile max(Δhm). Blue line = perfect; red = our model."""
    if qq is None:
        return
    pred_q = np.asarray(qq["pred_q"])
    obs_q = np.asarray(qq["obs_q"])
    fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    lim_lo = float(min(pred_q.min(), obs_q.min()))
    lim_hi = float(max(pred_q.max(), obs_q.max()))
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "b--", linewidth=1, label="y = x")
    ax.plot(obs_q, pred_q, "r-o", markersize=3, linewidth=1.2, label="pred vs obs")
    ax.set_xlabel("Observed per-tile max(Δhm)")
    ax.set_ylabel("Predicted per-tile max(Δhm)")
    ax.set_title("Q-Q: per-tile spatial maximum")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_tail_exceedance_table(rows):
    if not rows:
        return ""
    header = ("| Threshold | n(obs>t) | n(pred>t) | POD | CSI | FAR | q975 soft-POD |\n"
              "|---|---|---|---|---|---|---|\n")
    body = []
    for r in rows:
        body.append(
            f"| {r['threshold']:>+.3f} | {r['n_obs_exceed']:>10,d} | "
            f"{r['n_pred_exceed']:>10,d} | "
            f"{r['POD']:.3f} | {r['CSI']:.3f} | {r['FAR']:.3f} | "
            f"{r['q975_soft_POD']:.3f} |"
        )
    return "### Tail exceedance (deterministic + q975 ensemble support)\n\n" + header + "\n".join(body) + "\n"


def _render_r95p(r95p):
    if r95p is None:
        return ""
    return (
        f"### R95p (mass above {r95p['percentile']}th percentile of obs)\n\n"
        f"- Threshold: {r95p['threshold']:.4f}\n"
        f"- Observed tail mass: {r95p['obs_tail_mass']:.4f}\n"
        f"- Predicted tail mass: {r95p['pred_tail_mass']:.4f}\n"
        f"- **Ratio (pred / obs):** **{r95p['ratio']:.3f}** "
        f"(<<1 = under-predicting tail; ~1 = calibrated; >1 = over)\n"
    )


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
                 model_info=None, sample_grid_path=None, qq_path=None):
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

    md = f"""# Diffusion Δhm Forecasting — Results on the Small Region

## Setup

- Branch: `diffusion`
- Region: `{args.predict_region}`
- Aggregate-tile size: **{args.tile_size}×{args.tile_size}** pixels (10 km blocks).
- Histogram bins (rarity-weighted, matches baseline `histogram_loss.py`):
  `{HIST_BIN_EDGES.tolist()}`.{wandb_section}

## Iteration history (best-of-run on user-target metrics)

User targets (re-stated): improve **per-bin pixel coverage** (especially
bins > 0.05) and **tile-level histogram intersection** (`xinter_median`).
Pixel-precise magnitude matching is *not* a goal.

| Run | bin > 0.1 cov | bin > 0.2 cov | xinter | Pearson r | MAE med | cov rate | pred_max | Notes |
|---|---|---|---|---|---|---|---|---|
| v8 baseline | 0.000 | 0.000 | 0.860 | 0.561 | 0.0034 | 0.942 | 0.056 | EMA + weighted + min-SNR (stride 32 eval) |
| v10/v11 | 0.000 | 0.000 | 0.55/0.85 | 0.55 | varies | varies | 0.022/0.011 | signed-log target — regression |
| v12 | 0.000 | 0.000 | 0.438 | 0.378 | 0.0050 | 0.949 | 0.056 | 2A+2B+2C — flat on tail |
| **v13** | **0.069** | 0.000 | 0.400 | 0.665 | 0.0074 | 0.954 | **0.208** | **CorrDiff residual — tail wall breaks** |
| v14 | 0.072 | 0.000 | 0.430 | 0.677 | 0.0067 | 0.953 | 0.221 | + tile-aware loss redesign |
| v15 | 0.085 | 0.002 | 0.521 | 0.677 | 0.0042 | 0.959 | 0.214 | + 2× ensemble at inference (n=32) |
| v17 | 0.086 | 0.001 | 0.399 | 0.699 | 0.0097 | 0.958 | 0.206 | + per-tile hist_loss=1.0 — regression |
| v18 | 0.087 | 0.002 | 0.529 | 0.681 | 0.0043 | 0.959 | 0.215 | gentler hist_loss=0.3 |
| v19 | 0.087 | 0.002 | 0.481 | 0.693 | 0.0072 | 0.956 | 0.222 | + TV(masked) — tail preserved, neg-bias worsened |
| **v20** | 0.083 | 0.007 | **0.604** | **0.738** | **0.0034** | 0.958 | 0.356 | **+ strong chip-mean anchor — best balanced** |
| v21 | 0.394 | 0.039 | 0.133 | 0.606 | 0.0299 | 0.483 | 0.505 | + α=1.0 pixel-weighted mean head — tail breaks, bulk breaks |
| v22 | 0.135 | 0.005 | 0.471 | 0.687 | 0.0051 | 0.952 | 0.256 | α=0.3 — mediocre middle |
| v23 | **0.441** | 0.037 | 0.175 | 0.622 | 0.0279 | 0.485 | 0.413 | α=1.0 + 3× bulk anchors — best bin>0.1 cov |
| v24 | 0.247 | 0.042 | 0.343 | 0.534 | 0.0129 | 0.718 | 0.389 | mw=0.3 — best balance bin>0.2 |
| **v25** | 0.239 | **0.048** | 0.246 | 0.412 | 0.0209 | 0.543 | 0.411 | **mw=0.1 — best upper-tail; first q975→0.4** |

v13 was the architectural breakthrough (CorrDiff residual). v20 is
the best **balanced** result. v24 is the best **bin>0.2 / balance**.
v25 has the **best upper-tail** including first non-zero q975 reach
to 0.4. Choice depends on which trade-off you want.

### Production recipes

**v20 — bulk-balanced** (best xinter, MAE, coverage rate, Pearson):
```
Training:
  --use_ema --weighted_sampling --weight_alpha 1
  --pattern_loss_weight 0.3 --pattern_scales 8 16 32
  --tile_mean_loss_weight 5.0 --tile_mean_scales 8 16 32 64
  --wasserstein_loss_weight 0.2
  --hist_loss_weight 0.3 --hist_temperature 0.05 --hist_scales 16 32
  --tv_loss_weight 1.0 --tv_loss_target_floor 0.05
  --min_snr_gamma 5
  --use_magnitude_cond --m_dropout_prob 0.3 --m_norm_scale 0.5
  --use_mean_head --mean_head_hidden 128 --mean_loss_weight 1.0
Inference:
  --ensemble_n 32 --m_target 0.7 --predict_stride 64
```

**v25 — tail-focused** (best bin>0.2 cov, POD@0.2, q975 soft-POD@0.4):
```
Training: v20 +
  --tile_mean_loss_weight 15.0 (vs 5.0)
  --wasserstein_loss_weight 0.5 (vs 0.2)
  --hist_loss_weight 1.0 (vs 0.3)
  --pattern_loss_weight 0.5 (vs 0.3)
  --mean_loss_weight 0.1 (vs 1.0)
  --mean_head_pixel_weight_alpha 1.0 --mean_head_pixel_weight_eps 0.01
Inference: same as v20
```

**v24 — middle ground** (decent bulk + decent tail): v25 settings but
`--mean_loss_weight 0.3`.

The trade-off: stronger mean head pixel weighting (α=1.0) lifts hotspot
predictions but pulls the bulk's median pred from +0.001 to ~+0.02.
Pixel-precise matching is *not* a goal of this work; mean preservation
of the bulk distribution is. v20 maximises the latter; v25 maximises
upper-tail coverage. v24 splits the difference.

Performance note: predict_region_diffusion.py now calls
`torch.mps.empty_cache()` between batches and explicitly deletes
intermediate GPU tensors. Without that, MPS memory accumulates across
batches and the predict run slows from ~40 min to 2-3 h.

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

## Tail diagnostics

These metrics directly answer "is the model under-predicting magnitude
*somewhere in the field*?" — the user's stated success criterion. Methods follow
WassDiff (IEEE TGRS 2025), ExtremeCast (AAAI 2024), and the Aich et al. (GMD
2026) bias-correction work.

{_render_r95p(metrics.get('r95p'))}
{_render_tail_exceedance_table(metrics.get('tail_exceedance'))}
{f"![Q-Q max-of-field]({rel(qq_path)})" if qq_path is not None else ""}

## Figures

- Map comparison: ![]({rel(panel_path)})
- Tile-level summaries: ![]({rel(summary_path)})
{sample_section}
## Caveats / Notes

- The aggregate tile is the smallest meaningful spatial unit at which we can
  *fairly* compare a generative ensemble to a deterministic observation —
  per-pixel magnitude matching is explicitly *not* a goal of this work.
- Coverage rate is computed on the (q025, q975) band — if it drifts far below
  ~0.95, the model is over-confident; if much higher, the bands are too wide.
- Bins ≥ 0.2 still show near-zero coverage across all iterations: those
  events are rare (~1700 pixels region-wide in [0.2, 0.4]) and concentrate
  on a handful of hotspot tiles. Closing this gap likely needs focal-cropped
  training data (chip-centred on hotspots) or a wider U-Net backbone.
- Apple Silicon is fast enough for development at `--ensemble_n 32`; for
  the data-paper run on CUDA, bump to `--ensemble_n 64+` and consider a
  larger network with `base_channels=192`.
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
    else:
        q975 = None

    qq_max = qq_max_of_field(pred, obs, valid, args.tile_size)
    r95p = r95p_index(pred, obs, valid, percentile=95)
    tail_exceed = tail_exceedance_metrics(
        pred, obs, q975, valid, thresholds=(0.05, 0.1, 0.2, 0.4)
    )

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
        qq_max_of_field=qq_max,
        r95p=r95p,
        tail_exceedance=tail_exceed,
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
                cov_s = f"{cov_v:.3f}" if cov_v == cov_v else "nan"
                print(f"    [{low:>+7.3f}, {high:>+6.3f}]  n={entry['n']:>10,d}  coverage={cov_s}")
        elif k == "qq_max_of_field":
            if v is None:
                continue
            print(f"  qq_max_of_field: pred_max_global={v['pred_max_global']:.4f}  "
                  f"obs_max_global={v['obs_max_global']:.4f}  n_tiles={v['n_tiles']}")
        elif k == "r95p":
            if v is None:
                continue
            print(f"  r95p: threshold={v['threshold']:.4f}  ratio(pred/obs)={v['ratio']:.3f}")
        elif k == "tail_exceedance":
            print("  tail_exceedance:")
            for r in v:
                print(f"    t>{r['threshold']:>+.3f}  n_obs={r['n_obs_exceed']:>10,d}  "
                      f"POD={r['POD']:.3f}  CSI={r['CSI']:.3f}  "
                      f"q975-soft-POD={r['q975_soft_POD']:.3f}")
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

    qq_path = out_dir / "qq_max_of_field.png"
    plot_qq_max_of_field(qq_max, qq_path)
    if qq_max is not None:
        print(f"Wrote: {qq_path}")

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
                 model_info=model_info, sample_grid_path=sample_grid_path,
                 qq_path=qq_path if qq_max is not None else None)
    print(f"Wrote report: {args.report}")

    metrics_json = out_dir / "metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics JSON: {metrics_json}")


if __name__ == "__main__":
    main()
