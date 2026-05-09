"""Region prediction with the conditional diffusion U-Net.

For each tile inside the requested GeoJSON region, draw N samples from the
trained diffusion model, aggregate to median / 2.5% / 97.5% / std, and write
4 GeoTIFFs to disk. Distance-transform overlap blending mirrors the baseline.

Usage:
    python scripts/predict_region_diffusion.py \
        --checkpoint <path> \
        --predict_region config/region_to_predict_small.geojson
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import torch
import rasterio
from rasterio import features as rio_features
from rasterio.transform import rowcol, Affine
from rasterio.windows import Window
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, transform as shp_transform
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt

from torchgeo_dataloader import (
    hm_files, component_files, static_files, years,
)
from src.models.diffusion_lightning import DiffusionLightningModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--predict_region",
                   default="config/region_to_predict_small.geojson")
    p.add_argument("--output_dir", default="data/predictions_diffusion")
    p.add_argument("--predict_stride", type=int, default=32)
    p.add_argument("--predict_batch_size", type=int, default=4,
                   help="Number of *tiles* per GPU batch. Each tile expands "
                        "to ensemble_n samples internally.")
    p.add_argument("--ensemble_n", type=int, default=16)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="CFG scale: 1=no guidance (use cond only); >1 amplifies the "
                        "conditional. Only meaningful if the model was trained with "
                        "cfg_dropout_prob > 0.")
    p.add_argument("--tile_size", type=int, default=64)
    p.add_argument("--dhm_stats_cache",
                   default="data/raw/hm_global/dhm_stats_20yr.json")
    p.add_argument("--hm_stats_from_train",
                   default=None,
                   help="Optional path to a JSON dump of the training "
                        "dataset's hm_mean/hm_std/comp_*/static_* stats. "
                        "If unset, recompute fresh stats from a quick sample.")
    p.add_argument("--m_target", type=float, default=None,
                   help="Inference value for the FIDE-style block-maxima "
                        "conditioning channel (raw |Δhm| units). Only used if "
                        "the checkpoint's use_magnitude_cond is True. Typical "
                        "choices: training-set 99th percentile of max|Δhm| (≈0.5).")
    return p.parse_args()


def load_region(geojson_path, ref_crs):
    with open(geojson_path) as f:
        gj = json.load(f)
    if gj.get("type") == "FeatureCollection":
        geoms = [shape(feat["geometry"]) for feat in gj["features"]]
    elif gj.get("type") == "Feature":
        geoms = [shape(gj["geometry"])]
    else:
        geoms = [shape(gj)]
    if not geoms:
        raise ValueError(f"No geometry features in {geojson_path}")
    if ref_crs and ref_crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
        transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
        geoms = [shp_transform(lambda x, y: transformer.transform(x, y), g) for g in geoms]
    return unary_union(geoms)


def quick_stats(stat_samples=512, chip_size=64):
    """Recompute training-time normalization stats from the same raster set.

    The diffusion module's training run cached normalization on its dataset
    instance; we reproduce stats here directly so prediction does not require
    the live training dataset to be in memory.
    """
    from torchgeo_dataloader import HumanFootprintChipDataset
    print("Computing normalization stats from raster sample (slow on first call)...")
    ds = HumanFootprintChipDataset(
        hm_files, component_files, static_files,
        chip_size=chip_size, stat_samples=stat_samples,
        target_mode="delta_20yr",  # match training mode (extends hm_vars with distance vars)
    )
    return dict(
        hm_mean=float(ds.hm_mean), hm_std=float(ds.hm_std),
        comp_means={k: float(v) for k, v in ds.comp_means.items()},
        comp_stds={k: float(v) for k, v in ds.comp_stds.items()},
        static_means=[float(x) for x in ds.static_means],
        static_stds=[float(x) for x in ds.static_stds],
        hm_vars=list(ds.hm_vars),
        comp_files=dict(ds._comp_files),
    )


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    module = DiffusionLightningModule.load_from_checkpoint(args.checkpoint, map_location=device)
    module.train(False)

    # Δhm stats are baked into the checkpoint hyperparameters (and so is the
    # forward transform spec, e.g. dhm_transform="signed_log1p", dhm_log_scale).
    # module.denormalize() inverts both, so we just need to print for visibility.
    print(f"Δhm stats from checkpoint: mean={module.dhm_mean:.6e} std={module.dhm_std:.6e} "
          f"transform={getattr(module, 'dhm_transform', 'none')} "
          f"log_scale={getattr(module, 'dhm_log_scale', float('nan'))}")

    stats = quick_stats(stat_samples=512, chip_size=args.tile_size)
    hm_mean, hm_std = stats["hm_mean"], stats["hm_std"]
    comp_means, comp_stds = stats["comp_means"], stats["comp_stds"]
    static_means, static_stds = stats["static_means"], stats["static_stds"]
    # The dataset extends hm_vars with dist10/dist40 when delta_20yr; use that list
    # both for normalization keys AND to know which raster files to open.
    hm_vars_used = stats["hm_vars"]
    comp_files_used = stats["comp_files"]
    print(f"Component vars in conditioning: {hm_vars_used}")

    base_year = 2000
    input_years = [1990, 1995, 2000]
    year_to_idx = {y: i for i, y in enumerate(years)}
    t_idxs = [year_to_idx[y] for y in input_years]
    target_src_path = hm_files[year_to_idx[base_year]]

    # Region setup
    with rasterio.open(target_src_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_height, ref_width = ref.height, ref.width
        out_profile = ref.profile.copy()

    region_geom = load_region(args.predict_region, ref_crs)
    minx, miny, maxx, maxy = region_geom.bounds
    top_left = rowcol(ref_transform, minx, maxy, op=float)
    bottom_right = rowcol(ref_transform, maxx, miny, op=float)
    r0 = int(max(0, np.floor(min(top_left[0], bottom_right[0]))))
    c0 = int(max(0, np.floor(min(top_left[1], bottom_right[1]))))
    r1 = int(min(ref_height, np.ceil(max(top_left[0], bottom_right[0]))))
    c1 = int(min(ref_width, np.ceil(max(top_left[1], bottom_right[1]))))
    if r1 <= r0 or c1 <= c0:
        raise RuntimeError("Region is outside raster extent.")
    Hwin, Wwin = r1 - r0, c1 - c0
    print(f"Region bbox: {Hwin} × {Wwin} pixels (rows [{r0},{r1}), cols [{c0},{c1}))")

    bbox_transform = ref_transform * Affine.translation(c0, r0)
    bbox_mask = rio_features.geometry_mask(
        [mapping(region_geom)], out_shape=(Hwin, Wwin),
        transform=bbox_transform, invert=True,
    )

    # Accumulators per output statistic
    stat_keys = ["median", "q025", "q975", "std"]
    accum = {k: np.zeros((Hwin, Wwin), dtype=np.float64) for k in stat_keys}
    wsum = np.zeros((Hwin, Wwin), dtype=np.float64)

    # Open all rasters
    hm_srcs = [rasterio.open(p) for p in hm_files]
    # Use the extended component file list from the dataset (includes distance vars).
    comp_srcs = {y: [rasterio.open(p) for p in comp_files_used[y]] for y in years}
    stat_srcs = [rasterio.open(p) for p in static_files]
    nan_to_zero_static = {0, 4, 5, 6}

    def lonlat_grid(i0, j0, hi, wj):
        rows = np.arange(i0, i0 + hi)
        cols = np.arange(j0, j0 + wj)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        xs, ys = rasterio.transform.xy(ref_transform, rr, cc)
        xs = np.asarray(xs); ys = np.asarray(ys)
        if xs.ndim == 1:
            xs = xs.reshape(hi, wj); ys = ys.reshape(hi, wj)
        if ref_crs and ref_crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
            transformer = Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(xs, ys)
            lon = np.asarray(lon); lat = np.asarray(lat)
            if lon.ndim == 1:
                lon = lon.reshape(hi, wj); lat = lat.reshape(hi, wj)
        else:
            lon, lat = xs, ys
        return np.stack([lon, lat], axis=-1).astype(np.float32)

    tile = args.tile_size
    stride = args.predict_stride
    tile_coords = [(i, j) for i in range(r0, r1, stride) for j in range(c0, c1, stride)]
    total = len(tile_coords)
    print(f"Tiles: {total} (size {tile}, stride {stride}, batch {args.predict_batch_size})")

    processed = 0
    sampled = 0
    t_start = time.time()

    for batch_start in range(0, total, args.predict_batch_size):
        batch = tile_coords[batch_start : batch_start + args.predict_batch_size]
        bdyn, bstat, blonlat, bhm_t = [], [], [], []
        meta = []  # (i, j, hi, wj, li0, lj0, valid_mask, input_invalid)

        for (i, j) in batch:
            hi = min(tile, r1 - i)
            wj = min(tile, c1 - j)
            if hi <= 0 or wj <= 0:
                processed += 1
                continue
            li0, lj0 = i - r0, j - c0
            submask = bbox_mask[li0:li0 + hi, lj0:lj0 + wj]
            if not submask.any():
                processed += 1
                continue

            win = Window(j, i, wj, hi)
            dyn_ts = []
            for t_idx, y in zip(t_idxs, input_years):
                channels = []
                arr_hm = hm_srcs[t_idx].read(1, window=win, masked=True).filled(np.nan)
                channels.append((arr_hm - hm_mean) / hm_std)
                # Components (originals + precomputed distance vars when delta_20yr)
                for var_idx, (var_name, src) in enumerate(zip(hm_vars_used, comp_srcs[y])):
                    carr = src.read(1, window=win, masked=True).filled(np.nan)
                    carr = np.nan_to_num(carr, nan=0.0)
                    channels.append((carr - comp_means[var_name]) / comp_stds[var_name])
                dyn_ts.append(np.stack(channels, axis=0))
            input_dynamic = np.stack(dyn_ts, axis=0)  # [T, C_dyn, hi, wj]

            static_chs = []
            for sidx, src in enumerate(stat_srcs):
                sarr = src.read(1, window=win, masked=True).filled(np.nan)
                if sidx in nan_to_zero_static:
                    sarr = np.nan_to_num(sarr, nan=0.0)
                static_chs.append((sarr - static_means[sidx]) / static_stds[sidx])
            input_static = np.stack(static_chs, axis=0)

            hm_valid_all = np.isfinite(input_dynamic[:, 0]).all(axis=0)
            stat_valid = np.isfinite(input_static[0])
            valid_mask = submask & hm_valid_all & stat_valid
            if not valid_mask.any():
                processed += 1
                continue

            input_invalid = (
                ~np.isfinite(input_dynamic).all(axis=(0, 1))
                | ~np.isfinite(input_static).all(axis=0)
            )
            input_dynamic = np.nan_to_num(input_dynamic, nan=0.0).astype(np.float32)
            input_static = np.nan_to_num(input_static, nan=0.0).astype(np.float32)

            ll = lonlat_grid(i, j, hi, wj)
            hm_t_norm = input_dynamic[2, 0:1]  # [1, hi, wj]

            # Pad to tile if at edges
            if hi < tile or wj < tile:
                T, C = input_dynamic.shape[:2]
                pad_dyn = np.zeros((T, C, tile, tile), dtype=np.float32)
                pad_dyn[:, :, :hi, :wj] = input_dynamic
                input_dynamic = pad_dyn
                Cs = input_static.shape[0]
                pad_stat = np.zeros((Cs, tile, tile), dtype=np.float32)
                pad_stat[:, :hi, :wj] = input_static
                input_static = pad_stat
                pad_ll = np.zeros((tile, tile, 2), dtype=np.float32)
                pad_ll[:hi, :wj, :] = ll
                ll = pad_ll
                pad_hm_t = np.zeros((1, tile, tile), dtype=np.float32)
                pad_hm_t[:, :hi, :wj] = hm_t_norm
                hm_t_norm = pad_hm_t

            bdyn.append(input_dynamic); bstat.append(input_static)
            blonlat.append(ll); bhm_t.append(hm_t_norm)
            meta.append((i, j, hi, wj, li0, lj0, valid_mask, input_invalid))

        if not bdyn:
            processed += len(batch)
            continue

        bdyn_t = torch.from_numpy(np.stack(bdyn)).to(device)
        bstat_t = torch.from_numpy(np.stack(bstat)).to(device)
        bll_t = torch.from_numpy(np.stack(blonlat)).to(device)
        bhm_t_t = torch.from_numpy(np.stack(bhm_t)).to(device)

        with torch.no_grad():
            m_scalar = None
            if getattr(module, "use_magnitude_cond", False):
                m_val = float(args.m_target) if args.m_target is not None else 0.0
                m_scalar = torch.full(
                    (bdyn_t.shape[0],), m_val, device=device, dtype=torch.float32,
                )
            cond = module.assemble_conditioning(bdyn_t, bstat_t, bll_t, bhm_t_t,
                                                m_scalar=m_scalar)
            samples = module.sample(
                cond, n_samples=args.ensemble_n,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )  # [N, B, 1, H, W] in normalised (and possibly transformed) model space.
            # module.denormalize handles z-score AND inverse target transform (e.g.
            # signed_log1p) so callers always work in raw Δhm units.
            samples_raw = module.denormalize(samples.float())

        # Aggregate per tile
        samples_dhm = samples_raw.cpu().numpy()
        # samples_dhm: [N, B, 1, H, W]
        med = np.median(samples_dhm, axis=0)[:, 0]              # [B, H, W]
        q025 = np.quantile(samples_dhm, 0.025, axis=0)[:, 0]
        q975 = np.quantile(samples_dhm, 0.975, axis=0)[:, 0]
        std = samples_dhm.std(axis=0)[:, 0]

        for t_idx, (i, j, hi, wj, li0, lj0, valid_mask, input_invalid) in enumerate(meta):
            stats_per_tile = {
                "median": med[t_idx, :hi, :wj],
                "q025": q025[t_idx, :hi, :wj],
                "q975": q975[t_idx, :hi, :wj],
                "std": std[t_idx, :hi, :wj],
            }
            for s in stats_per_tile.values():
                s[input_invalid] = np.nan

            interior = valid_mask.astype(np.uint8)
            interior[[0, -1], :] = 0
            interior[:, [0, -1]] = 0
            weights = distance_transform_edt(interior)
            weights = np.where(valid_mask, weights, 0.0)
            if weights.max() <= 0:
                continue
            for k, arr in stats_per_tile.items():
                accum[k][li0:li0 + hi, lj0:lj0 + wj] += np.where(np.isfinite(arr), arr, 0.0) * weights
            wsum[li0:li0 + hi, lj0:lj0 + wj] += weights
            sampled += 1

        processed += len(batch)
        if processed % max(1, args.predict_batch_size * 4) == 0:
            elapsed = time.time() - t_start
            rate = processed / max(elapsed, 1e-6)
            eta = (total - processed) / max(rate, 1e-6)
            print(f"  {processed}/{total} tiles | {rate:.1f}/s | ETA {int(eta//60):02d}:{int(eta%60):02d}")

    print(f"Sampled {sampled} tiles. Blending and writing...")
    m = wsum > 0
    out = {}
    for k in stat_keys:
        arr = np.full((Hwin, Wwin), np.nan, dtype=np.float32)
        arr[m] = (accum[k][m] / wsum[m]).astype(np.float32)
        out[k] = arr

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_profile.update(dict(
        height=Hwin, width=Wwin,
        transform=ref_transform * Affine.translation(c0, r0),
        count=1, dtype="float32", compress="deflate",
    ))
    paths = {}
    for k in stat_keys:
        path = out_dir / f"prediction_dhm_2020_{k}.tif"
        with rasterio.open(path, "w", **out_profile) as dst:
            dst.write(out[k], 1)
        paths[k] = path
        print(f"  wrote {path}")

    for src in hm_srcs: src.close()
    for s in stat_srcs: s.close()
    for v in comp_srcs.values():
        for s in v: s.close()

    print("Done.")
    return paths


if __name__ == "__main__":
    main()
