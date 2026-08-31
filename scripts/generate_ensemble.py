#!/usr/bin/env python3
"""Phase 3 — generate the correlated ensemble and write it to an icechunk repository.

Marginals come from the **recalibrated** production rasters (Phase 1.5) plus the unmodified
central forecast; correlation structure comes from the Phase 1c variogram fit and the
hindcast horizon autocorrelation. Members are AR(1)-coupled across horizons so that
*change* statistics between two horizons inherit the right correlation.

Members are split across GPUs in blocks aligned to the member-chunk, so two workers never
write the same chunk. Each worker writes through a forked icechunk session and hands its
change record back; the parent merges them and commits once, so the snapshot either exists
complete or not at all.

Storage: an icechunk repository holding one array, ``members``, of shape
``(M, n_horizons, H, W)`` int16, chunks ``(1, 1, 1024, 1024)``, scale 1/32767
(HM is bounded on [0,1] and the observed maximum reaches 1.0), sentinel -32768 for invalid
pixels. ``manifest.json`` records each member's seed and AR(1) chain, so a single member can
be regenerated deterministically and the ensemble can be extended from 50 to 200 members
later without redoing anything.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# Set before torch is imported anywhere (the workers are spawned, so they inherit it).
# At the global grid the circulant-embedding field is 26136 x 40000 after padding and its
# working set peaks at 22.5 GB on a 24 GB card. The default caching allocator reserved
# 24.62 GB to serve that -- 2.1 GB of it fragmentation -- and OOM'd; expandable segments
# reserve 22.53 GB for the identical allocation. Measured, not guessed.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.copula import (  # noqa: E402
    DEFAULT_SCALE, INT16_SENTINEL, Z975, read_shape_artifact, shape_bounds, stack_shapes,
)

# The quantile-function raster's own quantum. Same number as DEFAULT_SCALE, spelled from the
# raster's side: the writer stores round(value / INT16_SCALE) as int16.
INT16_SCALE = 1.0 / 32767.0
from src.ensemble.aggregate import ARRAY_NAME  # noqa: E402
from src.ensemble.validate import DIST_LABELS, distance_band  # noqa: E402

REPO = Path(__file__).parent.parent
# One member per chunk. A member is generated and written whole, so a ten-member chunk means
# ten read-modify-write cycles over the same 20 MB object — invisible in plain zarr, where the
# last write wins and overwrites the file, but icechunk keeps every version until it is
# garbage-collected: the first M=400 null written that way came to 16.7 GB against 2.9 GB for
# the identical array copied in chunk-aligned blocks. One member per chunk also means no two
# workers can ever touch the same chunk, and reading a single member (T3 does) stops
# decompressing nine others.
MEMBER_CHUNK = 1
CHUNK_SHAPE = (MEMBER_CHUNK, 1, 1024, 1024)
N_BANDS = len(DIST_LABELS)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--central_dir", default="data/predictions")
    ap.add_argument("--recal_dir", default="data/predictions/recal")
    ap.add_argument("--central_pattern", default="prediction_{year}_central_blended.tif")
    ap.add_argument("--recal_pattern", default="prediction_{year}_{q}_recal.tif")
    ap.add_argument("--fallback_pattern", default="prediction_{year}_{q}_blended.tif")
    ap.add_argument("--years", default="2025,2030,2035,2040")
    ap.add_argument("--base_year", type=int, default=2020)
    ap.add_argument("--members", type=int, default=50)
    ap.add_argument("--out", default="data/ensemble/members.icechunk")
    ap.add_argument("--variogram_fits", default="data/ensemble/diagnostics/variogram_fits.csv")
    ap.add_argument("--rho_json", default="data/ensemble/residuals/horizon_autocorrelation.json")
    ap.add_argument("--ranges_px", default=None, help="Override, e.g. '8,240'")
    ap.add_argument("--weights", default=None, help="Override, e.g. '0.5,0.4'")
    ap.add_argument("--nugget", type=float, default=None)
    ap.add_argument("--dist_raster", default=None,
                    help="Distance-to-past-change raster. Required for a per-band marginal "
                         "shape; without it a per-band artifact falls back to its pooled "
                         "shape.")
    ap.add_argument("--marginal_shape", default=None,
                    help="JSON of per-horizon empirical marginal shapes "
                         "(scripts/fit_marginal_shape.py). Without it the marginal is the "
                         "two-piece normal, whose body carries ~3x too much moderate "
                         "change for this residual.")
    ap.add_argument("--qf_dir", default=None,
                    help="Directory of the distributional model's 64-band quantile-function "
                         "rasters. With this, the ensemble's marginal IS the model's own "
                         "Q_h(u|x): no two-piece normal, no empirical shape, no width factor. "
                         "Mutually exclusive with --marginal_shape.")
    ap.add_argument("--qf_pattern", default="prediction_{year}_qf.tif")
    ap.add_argument("--horizon_corr", default=None,
                    help="JSON with 'horizons' and a full correlation matrix 'R', from "
                         "src.ensemble.residuals.horizon_correlation_matrix. Replaces the "
                         "AR(1) chain with a separable space x horizon covariance "
                         "C_space(d).R_hh', which carries each horizon's spatial spectrum and "
                         "reproduces R exactly, both by construction. Mutually exclusive with "
                         "the AR(1) path; --rho_json is then only a scoring reference.")
    ap.add_argument("--copula", default="gaussian", choices=["gaussian", "t"],
                    help="Dependence family. 't' scales the correlated Gaussian field by one "
                         "shared chi2_df/df draw per member and maps it through the Student-t "
                         "CDF, which is tail *dependent* where a Gaussian copula is not. Every "
                         "pixel's marginal is unchanged by construction. Requires --qf_dir.")
    ap.add_argument("--copula_w_draw", default="stratified",
                    choices=["stratified", "iid"],
                    help="How the per-member chi2 factor is drawn. 'iid' is the textbook "
                         "sampler and is WRONG at finite M for this purpose: the same M draws "
                         "are reused at every pixel, so the realised across-member mixture is "
                         "the empirical law of those M values, not the target one. Measured at "
                         "M=400, df=7: the realised CDF sits +0.0033 above target at u=0.944, "
                         "which narrows every published interval by ~6% and flatters T2.8. "
                         "'stratified' takes w_m = chi2.ppf((m+0.5)/M)/df in random order, so "
                         "the realised mixture matches the target law exactly at any M while "
                         "staying independent of the field.")
    ap.add_argument("--copula_df", type=float, default=7.0,
                    help="Degrees of freedom for --copula t. 7 was measured, not chosen: the "
                         "observation's standardised position within the member ecoregion-mean "
                         "distribution has kurtosis 4.2-5.2 at h=10/15/20 on the e1 Africa "
                         "hindcast (Gaussian is 3.0), and 3 + 6/(nu-4) inverts to nu = 6.7-8.9, "
                         "7.1 at h=20.")
    ap.add_argument("--spectral_fits", default=None,
                    help="JSON from scripts/fit_field_spectra.py. Matching the observed "
                         "power spectrum rather than the variogram is what gives members "
                         "the right texture: the variogram fit leaves the 3-10px band, "
                         "which holds ~40%% of the observed variance, half empty.")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--wrap_lon", type=lambda x: str(x).lower() == "true", default=None,
                    help="Default: auto (True only for the full-globe grid)")
    ap.add_argument("--independent", action="store_true",
                    help="Build the independent-pixel null ensemble (same marginals, no "
                         "spatial structure) — the honest null for T3.2/T3.3")
    ap.add_argument("--disable_wandb", action="store_true")
    ap.add_argument("--wandb_group", default=None)
    return ap.parse_args(argv)


# --------------------------------------------------------------------------------------
def resolve_paths(args, years):
    """(central, lower, upper) per year, preferring recalibrated bounds."""
    out = {}
    used_recal = {}
    for y in years:
        cen = Path(args.central_dir) / args.central_pattern.format(year=y)
        trio = {"central": cen}
        for q in ("lower", "upper"):
            r = Path(args.recal_dir) / args.recal_pattern.format(year=y, q=q)
            if r.exists():
                trio[q] = r
                used_recal[y] = True
            else:
                trio[q] = Path(args.central_dir) / args.fallback_pattern.format(year=y, q=q)
                used_recal[y] = False
        missing = [str(p) for p in trio.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing rasters for {y}: {missing}")
        out[y] = trio
    return out, used_recal


def load_marginals(paths, years, cache_dir, dist_raster=None):
    """Compact per-valid-pixel marginal parameters, memory-mapped for the workers.

    Two passes so no more than one year's three rasters are resident at a time (each is
    2.7 GB at the global grid), and the compact arrays go to disk rather than being pickled
    into every worker process.

    With ``dist_raster`` the pixel's distance band is cached the same way, in the same
    order, so a per-band marginal shape can be applied by gathering rows.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    valid = None
    profile = None
    for y in years:
        with rasterio.open(paths[y]["central"]) as c:
            cen = c.read(1)
            profile = c.profile.copy()
        v = np.isfinite(cen)
        del cen
        for q in ("lower", "upper"):
            with rasterio.open(paths[y][q]) as s:
                v &= np.isfinite(s.read(1))
        valid = v if valid is None else (valid & v)

    idx = np.flatnonzero(valid.ravel()).astype(np.int64)
    np.save(cache_dir / "idx.npy", idx)

    if dist_raster:
        # Same ordering as idx, so the worker can index it with the flat z vector directly.
        with rasterio.open(dist_raster) as s:
            band = distance_band(s.read(1).astype(np.float64)).ravel()[idx]
        np.save(cache_dir / "band.npy", band.astype(np.int8))

    compact = {}
    for y in years:
        with rasterio.open(paths[y]["central"]) as c:
            cen = c.read(1).ravel()[idx].astype(np.float32)
        with rasterio.open(paths[y]["lower"]) as s:
            low = s.read(1).ravel()[idx].astype(np.float32)
        with rasterio.open(paths[y]["upper"]) as s:
            upp = s.read(1).ravel()[idx].astype(np.float32)
        arrays = {
            "loc": cen,
            "scale_left": np.maximum((cen - low) / Z975, 1e-6).astype(np.float32),
            "scale_right": np.maximum((upp - cen) / Z975, 1e-6).astype(np.float32),
        }
        compact[y] = {}
        for name, arr in arrays.items():
            p = cache_dir / f"{y}_{name}.npy"
            np.save(p, arr)
            compact[y][name] = str(p)
        del arrays, cen, low, upp
    return compact, valid, idx, profile


def load_quantile_functions(qf_paths, years, cache_dir, idx):
    """Compact per-valid-pixel quantile functions, memory-mapped for the workers.

    The distributional model's forecast *is* ``Q_h(u|x)``, so the ensemble's marginal is read
    from the model's own 64-band raster rather than fitted to a triple. Kept int16 exactly as
    stored (scale ``INT16_SCALE``): 64 levels x 35.6M px is 4.6 GB per horizon as int16 and
    9.1 GB as float32, and the worker dequantizes the two bands it gathers instead of the slab.

    Returns ``(paths_by_year, u_levels)``. The u-grid is shared by every raster and every
    pixel — ``output_u_grid`` defines it in one place — which is what makes the lookup a
    ``searchsorted`` on a 1-D tensor instead of a per-pixel search.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out, u_ref = {}, None
    for y in years:
        with rasterio.open(qf_paths[y]) as src:
            tags = src.tags()
            if "u_levels" not in tags:
                raise SystemExit(f"{qf_paths[y]} carries no u_levels tag; not a quantile-"
                                 f"function raster (was --predict_qf_levels 0 set?)")
            u = np.array([float(v) for v in tags["u_levels"].split(",")], dtype=np.float64)
            if np.diff(u).min() <= 0:
                raise SystemExit(f"{qf_paths[y]}: u levels are not strictly increasing")
            if u_ref is None:
                u_ref = u
            elif not np.array_equal(u, u_ref):
                raise SystemExit(f"{qf_paths[y]}: u grid differs from {years[0]}'s; the "
                                 f"horizons would be sampled on different grids")
            p = cache_dir / f"{y}_qf.npy"
            # Band by band into a memmap, never the whole raster. src.read() on the global
            # grid is 64 x 684.4 Mpx x 2 B = 87.6 GB, and the gather that follows it another
            # 23.6 GB -- against 125 GB of DRAM. On Africa the same call is 8.1 GB, which is
            # why it survived every regional run. One band is 1.37 GB and the compacted
            # result is written straight to disk, so the peak is the band.
            q = np.lib.format.open_memmap(p, mode="w+", dtype=np.int16,
                                          shape=(int(u.size), int(idx.size)))
            for bi in range(int(u.size)):
                q[bi] = src.read(bi + 1).reshape(-1)[idx]
            q.flush()
            del q
        out[y] = str(p)
    return out, u_ref


def t_cdf_table(df, zmax=12.0, n=240001):
    """``(grid, T_df(grid))`` for mapping a t-variate to its probability rank on GPU.

    torch has no Student-t CDF, and calling scipy per member on 13.8M pixels would add most
    of an hour to a run. The CDF is smooth, so a dense table plus linear interpolation is
    exact to ~1e-9 at this spacing — and the accuracy that matters is bounded anyway: the
    stored quantile grid truncates at u = 1e-4, and ``T_7(-12) = 1.0e-6`` is already past
    it, so everything beyond the table's ends clamps to the outermost stored quantile
    exactly as a Gaussian draw of ``|z| > 3.72`` does.
    """
    from scipy.stats import t as _t
    g = np.linspace(-zmax, zmax, n).astype(np.float64)
    return g.astype(np.float32), _t.cdf(g, df).astype(np.float32)


def u_from_t_torch(z, w, grid, cdf):
    """``u = T_df(z / sqrt(w))`` — the Student-t copula's probability rank.

    ``z`` is the correlated standard-normal field and ``w`` a single ``chi2_df / df`` draw
    shared by every pixel and every horizon of one member. That is the whole construction:
    a Gaussian copula scaled by one common random factor, which is what makes a t-copula
    tail *dependent* where a Gaussian one is not. Because ``z / sqrt(w)`` is marginally
    ``t_df`` by definition, ``T_df`` of it is exactly uniform — so **every pixel's marginal
    is untouched**, and only the joint behaviour changes. The per-pixel gates (T5, and
    check_qf_ensemble's sandwich) must pass unchanged; if they move, this is wrong.
    """
    import torch
    zt = z * float(1.0 / np.sqrt(w))
    j = torch.searchsorted(grid, zt.contiguous()).clamp_(1, grid.numel() - 1)
    g0, g1 = grid[j - 1], grid[j]
    t = ((zt - g0) / (g1 - g0)).clamp_(0.0, 1.0)
    return cdf[j - 1] + t * (cdf[j] - cdf[j - 1])


def qf_from_z_torch(z, u_levels, Q, scale, clip=(0.0, 1.0), u=None):
    """Map a standard-normal field through each pixel's own quantile function.

    ``u = Phi(z)`` then ``Q(u)`` by linear interpolation between the two stored levels that
    bracket it. Only those two bands are gathered per pixel, so the slab is never dequantized
    whole. Draws beyond the grid's ends (u outside [1e-4, 1-1e-4], i.e. |z| > 3.72) clamp to
    the outermost stored quantile: the raster is the forecast, and extrapolating past it would
    invent tail the model did not emit.
    """
    # Imported here, not at module scope: this file sets PYTORCH_CUDA_ALLOC_CONF before torch
    # is imported anywhere, and a module-level import would defeat that.
    import torch

    # ``u`` is supplied when the copula is not Gaussian; the Gaussian path is unchanged.
    if u is None:
        u = 0.5 * (1.0 + torch.erf(z * 0.7071067811865476))
    j = torch.searchsorted(u_levels, u.contiguous()).clamp_(1, u_levels.numel() - 1)
    u0, u1 = u_levels[j - 1], u_levels[j]
    t = ((u - u0) / (u1 - u0)).clamp_(0.0, 1.0)
    ar = torch.arange(z.numel(), device=z.device)
    q0 = Q[j - 1, ar].to(torch.float32) * scale
    q1 = Q[j, ar].to(torch.float32) * scale
    return torch.clamp(q0 + t * (q1 - q0), clip[0], clip[1])


def with_bounds(shape):
    """Attach the continuation constants so the worker can hold the grids on the GPU."""
    z_hi, z_lo, off_hi, off_lo = shape_bounds(shape)
    return {**shape, "z_hi": z_hi, "z_lo": z_lo, "off_hi": off_hi, "off_lo": off_lo}


def load_shapes(path, have_band):
    """``{horizon: shape}`` for the worker — stacked per band when a raster is available.

    Without a distance raster a per-band artifact collapses to band 0's shape, which for
    these artifacts is the horizon's pooled fit.
    """
    table, banded = read_shape_artifact(path, N_BANDS)
    horizons = sorted({h for h, _ in table})
    if not (banded and have_band):
        return {h: with_bounds(table[(h, 0)]) for h in horizons if (h, 0) in table}, banded
    # One row per band the raster can produce, not per band the artifact happens to carry:
    # the >100 px band is too sparse to fit region-wide but still occurs in the raster, and
    # a short stack is an out-of-bounds gather on the GPU.
    return ({h: stack_shapes([table.get((h, b)) for b in range(N_BANDS)])
             for h in horizons}, banded)


def field_params(args, horizons):
    """Correlation structure per horizon: spectral fit > CLI override > variogram fit."""
    if args.spectral_fits and Path(args.spectral_fits).exists():
        blob = json.load(open(args.spectral_fits))
        by_h = blob.get("by_horizon", {})
        out = {}
        for h in horizons:
            f = by_h.get(str(h)) or next(iter(by_h.values()))
            out[h] = {"ranges_px": f["ranges_px"], "weights": f["weights"],
                      "nugget": f["nugget"], "kernel": blob.get("kernel", "matern"),
                      "nu": blob.get("nu", 0.5)}
        return out
    if args.ranges_px and args.weights:
        ranges = [float(x) for x in args.ranges_px.split(",")]
        weights = [float(x) for x in args.weights.split(",")]
        nug = float(args.nugget if args.nugget is not None else max(0.0, 1 - sum(weights)))
        return {h: {"ranges_px": ranges, "weights": weights, "nugget": nug} for h in horizons}

    fits = pd.read_csv(args.variogram_fits)
    glob = fits[fits["stratum"].astype(str).isin(["ALL", "nan", "None"])]
    if glob.empty:
        glob = fits
    out = {}
    for h in horizons:
        sub = glob[glob["horizon"] == h] if "horizon" in glob else glob
        if sub.empty:
            sub = glob
        row = sub.iloc[0]
        total = row["nugget"] + row["var_short"] + row["var_long"]
        out[h] = {
            "ranges_px": [float(row["range_short_px"]), float(row["range_long_px"])],
            "weights": [float(row["var_short"] / total), float(row["var_long"] / total)],
            "nugget": float(row["nugget"] / total),
        }
    return out


def load_rho(args, horizons):
    rho = {}
    p = Path(args.rho_json)
    if p.exists():
        raw = json.load(open(p))
        rho = {int(k): float(v) for k, v in raw.items() if np.isfinite(float(v))}
    return {h: rho.get(h, 0.9) for h in horizons[1:]}


def assign_members(n_members, n_workers, block=MEMBER_CHUNK):
    """Partition members into chunk-aligned blocks so workers never share a zarr chunk."""
    blocks = [list(range(s, min(s + block, n_members))) for s in range(0, n_members, block)]
    out = [[] for _ in range(n_workers)]
    for i, b in enumerate(blocks):
        out[i % n_workers].extend(b)
    return out


# --------------------------------------------------------------------------------------
def worker(worker_id, gpu, member_ids, cfg, session=None):
    """Generate this worker's members. Returns the forked icechunk session to be merged.

    Members are partitioned on the chunk boundary (``assign_members``), so two workers never
    touch the same chunk and their change sets merge without conflict. Only the parent
    commits — local-filesystem icechunk warns that concurrent *commits* are unsafe, and this
    shape has exactly one.
    """
    import torch
    import zarr

    from src.ensemble.copula import marginal_from_z_torch
    from src.ensemble.fields import generate_correlated_field

    years = cfg["years"]
    horizons = cfg["horizons"]
    H, W = cfg["shape"]
    device = f"cuda:{gpu}" if (gpu is not None and torch.cuda.is_available()) else "cpu"

    compact = cfg["compact"]
    idx_np = np.load(cfg["idx"], mmap_mode="r")
    z_store = zarr.open_group(session.store, mode="r+")[ARRAY_NAME]

    idx_arr = np.asarray(idx_np)
    # int32 on the device, int64 on the host. These two live for the whole worker and would
    # otherwise sit inside the field's peak: at 184.6M valid px int64 costs 1.48 GB and the
    # index fits in int32 (H*W = 684M < 2^31). Torch indexes fine with int32.
    idx_t = torch.as_tensor(idx_arr.astype(np.int32), device=device)
    # Marginals stay pinned on the host and only the horizon in flight is moved to the
    # GPU: all four years at once is ~9 GB, which does not coexist with the FFT working
    # set on a 24 GB card.
    qf_cfg = cfg.get("qf")
    if qf_cfg:
        # int16 on the host, memory-mapped: 64 levels x 35.6M px is 4.6 GB per horizon, and
        # four horizons resident on a 24 GB card would not coexist with the FFT working set.
        # The slab in flight is uploaded per (member, horizon) -- ~0.5 s against the field
        # generation it sits beside.
        qf_cpu = {y: np.load(p, mmap_mode="r") for y, p in qf_cfg["paths"].items()}
        u_levels_t = torch.as_tensor(np.asarray(qf_cfg["u_levels"]), device=device,
                                     dtype=torch.float32)
        marg_cpu = {}
    else:
        qf_cpu, u_levels_t = None, None
        marg_cpu = {
            y: {k: torch.from_numpy(np.load(v)) for k, v in compact[y].items()}
            for y in years
        }
    scatter = np.full(H * W, INT16_SENTINEL, dtype=np.int16)

    # The shape grids are the same for every member, so upload them once rather than on
    # each of the M x n_horizons calls. torch.as_tensor inside apply_shape_torch is then a
    # no-op because device and dtype already match.
    band_t = None
    if cfg.get("band"):
        # int8 holds a distance-band index (0-5) in 0.18 GB instead of 1.48 GB;
        # apply_shape_torch casts it to long itself, and only after the field is freed.
        band_t = torch.as_tensor(np.load(cfg["band"]).astype(np.int8), device=device)
    shapes_dev = {}
    for h, sh in (cfg.get("shapes") or {}).items():
        shapes_dev[h] = {k: (torch.as_tensor(np.asarray(v), device=device,
                                             dtype=torch.float32)
                             if k != "n" and np.ndim(v) > 0 else v)
                         for k, v in sh.items()}

    # One table per worker, not per member. Only built when the copula is not Gaussian.
    t_grid = t_cdf = None
    if cfg.get("copula", "gaussian") == "t":
        import torch as _torch
        _g, _c = t_cdf_table(cfg["copula_df"])
        t_grid = _torch.as_tensor(_g, device=device)
        t_cdf = _torch.as_tensor(_c, device=device)

    # Separable space x horizon: one shared spatial spectrum, and the horizons coupled by a
    # Cholesky factor of the full correlation matrix rather than an AR(1) chain. Under the AR
    # recursion horizon h carries `rho^2 S_{h-1} + (1 - rho^2) S_h`, not S_h; under this it
    # carries the shared spectrum exactly and reproduces R exactly. Both by construction.
    chol_L = None
    if cfg.get("horizon_corr") is not None and not cfg["independent"]:
        import torch as _torch
        chol_L = _torch.as_tensor(
            np.linalg.cholesky(np.asarray(cfg["horizon_corr"], dtype=np.float64)).astype(np.float32),
            device=device)

    for m in member_ids:
        t0 = time.time()
        prev = None
        # One chi2 draw per MEMBER, shared across every horizon and every pixel: the member
        # is one story, so its tail factor has to be one number. Seeded off the member index
        # alone, so a member regenerates identically however the run is sharded.
        w_m = 1.0
        if cfg.get("copula", "gaussian") == "t":
            w_m = float(cfg["copula_w"][m])
        z_sep = None
        if chol_L is not None:
            eps_all = []
            for h_ in horizons:
                fp_ = cfg["field_params"][h_]
                f_ = generate_correlated_field(
                    H, W, fp_["ranges_px"], fp_["weights"], fp_["nugget"],
                    wrap_lon=cfg["wrap_lon"], device=device,
                    seed=cfg["seed"] + 1_000_003 * m + 101 * h_, return_torch=True,
                    kernel=fp_.get("kernel", "gaussian"), nu=fp_.get("nu", 0.5))
                eps_all.append(f_.reshape(-1)[idx_t].clone())
                del f_
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            z_sep = chol_L @ torch.stack(eps_all)
            del eps_all
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        for hi, (h, y) in enumerate(zip(horizons, years)):
            seed = cfg["seed"] + 1_000_003 * m + 101 * h
            if z_sep is not None:
                eps = None          # the fields were drawn above, jointly
            elif cfg["independent"]:
                gen = torch.Generator(device=device)
                gen.manual_seed(seed)
                eps = torch.randn(idx_t.shape[0], generator=gen, device=device)
            else:
                fp = cfg["field_params"][h]
                field = generate_correlated_field(
                    H, W, fp["ranges_px"], fp["weights"], fp["nugget"],
                    wrap_lon=cfg["wrap_lon"], device=device, seed=seed, return_torch=True,
                    kernel=fp.get("kernel", "gaussian"), nu=fp.get("nu", 0.5),
                )
                eps = field.reshape(-1)[idx_t].clone()
                del field
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

            if z_sep is not None:
                z = z_sep[hi]
            elif prev is None:
                z = eps
            else:
                rho = float(cfg["rho"].get(h, 0.9))
                z = rho * prev + np.sqrt(max(0.0, 1 - rho ** 2)) * eps
            prev = z

            if qf_cpu is not None:
                Q = torch.as_tensor(np.ascontiguousarray(qf_cpu[y]), device=device)
                u_m = (u_from_t_torch(z, w_m, t_grid, t_cdf)
                       if t_grid is not None else None)
                vals = qf_from_z_torch(z, u_levels_t, Q, qf_cfg["scale"], u=u_m)
                del u_m
                del Q
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                mg = {k: v.to(device, non_blocking=True) for k, v in marg_cpu[y].items()}
                vals = marginal_from_z_torch(z, mg["loc"], mg["scale_left"], mg["scale_right"],
                                             shape=shapes_dev.get(h), band=band_t)
                del mg
            q = torch.clamp(torch.round(vals / cfg["scale"]), INT16_SENTINEL + 1, 32767)
            q = q.to(torch.int16).cpu().numpy()
            del vals
            scatter[:] = INT16_SENTINEL
            scatter[idx_arr] = q
            z_store[m, hi] = scatter.reshape(H, W)
        print(f"[worker {worker_id} gpu {gpu}] member {m} done in {time.time() - t0:.1f}s", flush=True)
    return session


def main(argv=None):
    args = parse_args(argv)
    years = [int(y) for y in args.years.split(",")]
    horizons = [y - args.base_year for y in years]

    paths, used_recal = resolve_paths(args, years)
    if not all(used_recal.values()):
        print("⚠ Recalibrated bounds not found for every year; falling back to the raw heads "
              "for those years. Phase 1.5 must be final before the production run.")

    if args.qf_dir and args.marginal_shape:
        raise SystemExit("--qf_dir and --marginal_shape are two different marginals; the "
                         "distributional model's quantile function IS its marginal, and "
                         "reshaping it with an empirical fit would undo the point of it.")
    print("Loading marginals ...")
    cache_dir = Path(args.out).parent / (Path(args.out).stem + "_marginals")
    compact, valid, idx, profile = load_marginals(paths, years, cache_dir, args.dist_raster)
    band_path = (cache_dir / "band.npy") if args.dist_raster else None
    H, W = valid.shape
    print(f"  grid {H} x {W}, {idx.size:,} valid pixels ({100 * idx.size / valid.size:.1f}%)")

    wrap = args.wrap_lon
    if wrap is None:
        wrap = bool(W >= 39_000 and abs(profile["transform"].c + 180.0) < 0.05)
    fps = field_params(args, horizons)
    rho = load_rho(args, horizons)
    print(f"  wrap_lon={wrap}  rho={rho}")
    for h in horizons:
        top = sorted(zip(fps[h]['ranges_px'], fps[h]['weights']), key=lambda t: -t[1])[:3]
        print(f"  h={h}: kernel={fps[h].get('kernel','gaussian')} nugget={fps[h]['nugget']:.3f} "
              f"top structures=" + " ".join(f"{r:g}px:{w:.3f}" for r, w in top if w > 0.004))

    import icechunk
    import zarr

    out_path = Path(args.out)
    est_gb = args.members * len(years) * H * W * 2 / 1e9
    print(f"  writing {out_path} — {est_gb:.1f} GB uncompressed "
          f"({100 * idx.size / valid.size:.0f}% populated, expect far less on disk)")
    # Recreating an icechunk repository in place is not a thing, and clearing the old one is
    # the caller's decision, not a silent side effect of pointing --out at it.
    if out_path.exists() and any(out_path.iterdir()):
        raise SystemExit(f"{out_path} already exists and is not empty — remove it first")
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(out_path)))
    session = repo.writable_session("main")
    root = zarr.create_group(session.store)
    store = root.create_array(
        ARRAY_NAME,
        shape=(args.members, len(years), H, W),
        chunks=CHUNK_SHAPE,
        dtype="i2", fill_value=INT16_SENTINEL,
    )
    store.attrs.update({
        "scale": DEFAULT_SCALE, "offset": 0.0, "sentinel": INT16_SENTINEL,
        "years": years, "horizons": horizons, "base_year": args.base_year,
        "members": args.members, "independent_null": bool(args.independent),
        "wrap_lon": bool(wrap),
        "field_params": {str(k): v for k, v in fps.items()},
        "rho": {str(k): v for k, v in rho.items()},
        "seed": args.seed,
        "marginal": ("per-pixel quantile function read from the model (no post-hoc "
                     "reshaping)" if args.qf_dir else
                     "median-spliced two-piece normal; ppf(0.5)=central by construction"),
        "qf_dir": str(args.qf_dir) if args.qf_dir else None,
        "copula": args.copula, "copula_df": float(args.copula_df),
        "sources": {str(y): {k: str(v) for k, v in paths[y].items()} for y in years},
    })

    qf_cfg = None
    if args.qf_dir:
        qf_paths = {y: str(Path(args.qf_dir) / args.qf_pattern.format(year=y)) for y in years}
        missing = [p for p in qf_paths.values() if not Path(p).exists()]
        if missing:
            raise SystemExit(f"--qf_dir given but these are missing: {missing}")
        print(f"  marginal: the model's own quantile function, from {args.qf_dir}")
        qf_map, u_levels = load_quantile_functions(qf_paths, years, cache_dir, idx)
        qf_cfg = {"paths": qf_map, "u_levels": [float(v) for v in u_levels],
                  "scale": float(INT16_SCALE)}
        print(f"    {len(u_levels)} levels, u in [{u_levels[0]:.5f}, {u_levels[-1]:.5f}] "
              f"— draws beyond that clamp to the outermost stored quantile")

    shapes, banded = {}, False
    if args.marginal_shape and Path(args.marginal_shape).exists():
        shapes, has_bands = load_shapes(args.marginal_shape, band_path is not None)
        banded = has_bands and band_path is not None
        kind = "per (horizon x distance band)" if banded else "per horizon"
        print(f"  marginal shape: empirical {kind}, from {args.marginal_shape} "
              f"(horizons {sorted(shapes)})")
        if has_bands and not banded:
            print("  ⚠ the artifact carries per-band shapes but --dist_raster was not given, "
                  "so the pooled shape is used — that is a different marginal family")
    else:
        print("  marginal shape: two-piece normal (no --marginal_shape given)")

    # Which marginal produced this store is not recoverable from the values, and these
    # families are routinely compared against each other on disk. Record it — including the
    # tail bounds, since two artifacts can share a body and differ only there, and that
    # difference is what moves T8.1 and T8.2.
    prov = {
        "source": str(args.marginal_shape) if args.marginal_shape else None,
        "family": ("empirical, per (horizon x distance band)" if banded else
                   "empirical, per horizon" if shapes else
                   "median-spliced two-piece normal"),
        "dist_raster": str(args.dist_raster) if args.dist_raster else None,
    }
    if args.marginal_shape and Path(args.marginal_shape).exists():
        table, _ = read_shape_artifact(args.marginal_shape, N_BANDS)
        bodies = {id(v) for v in table.values()}
        prov["distinct_bodies"] = len(bodies)
        h0 = min(h for h, _ in table)
        prov["u_bound_by_band"] = [table[(h0, b)].get("u_bound") for b in range(N_BANDS)]
        prov["u_bound_lo_by_band"] = [table[(h0, b)].get("u_bound_lo")
                                      for b in range(N_BANDS)]
    store.attrs["marginal_shape"] = prov

    cfg = {
        "years": years, "horizons": horizons, "shape": (H, W), "compact": compact,
        "idx": str(cache_dir / "idx.npy"), "out": str(out_path), "field_params": fps, "rho": rho,
        "seed": args.seed, "scale": DEFAULT_SCALE, "wrap_lon": bool(wrap),
        "independent": bool(args.independent), "shapes": shapes, "qf": qf_cfg,
        "band": str(band_path) if (band_path and banded) else None,
        "copula": args.copula, "copula_df": float(args.copula_df),
        "horizon_corr": None, "copula_w": None,
    }
    if args.copula == "t":
        from scipy.stats import chi2 as _chi2
        _rng = np.random.default_rng(args.seed + 7_777_777)
        if args.copula_w_draw == "stratified":
            _w = _chi2.ppf((np.arange(args.members) + 0.5) / args.members,
                           args.copula_df) / args.copula_df
            _w = _w[_rng.permutation(args.members)]
        else:
            _w = _rng.chisquare(args.copula_df, size=args.members) / args.copula_df
        cfg["copula_w"] = [float(x) for x in _w]
        print(f"  chi2 factor: {args.copula_w_draw}, mean {_w.mean():.4f} "
              f"(target 1.0), min {_w.min():.4f}, max {_w.max():.4f}")
    if args.horizon_corr:
        blob = json.load(open(args.horizon_corr))
        hz = [int(x) for x in blob["horizons"]]
        if hz != list(horizons):
            raise SystemExit(f"--horizon_corr covers {hz}, this run needs {list(horizons)}")
        R_ = np.asarray(blob["R"], dtype=np.float64)
        np.linalg.cholesky(R_)          # refuse a non-PSD matrix here, not inside a worker
        cfg["horizon_corr"] = R_.tolist()
        store.attrs["horizon_corr"] = {"R": R_.tolist(), "horizons": hz,
                                       "source": str(args.horizon_corr)}
        off = R_[np.triu_indices_from(R_, k=1)]
        print(f"  separable space x horizon: R from {args.horizon_corr}, "
              f"off-diagonal {off.min():.3f}-{off.max():.3f}, "
              f"adjacent {', '.join(f'{R_[i, i+1]:.3f}' for i in range(len(hz)-1))}")
    if args.copula == "t":
        if not args.qf_dir:
            raise SystemExit("--copula t needs --qf_dir: it supplies u directly, and the "
                             "two-piece path takes z")
        if args.independent:
            # The null must keep the marginals and drop the dependence. A t-copula's shared
            # chi2 factor IS dependence, so applying it here would give the null domain-scale
            # structure and quietly flatter the correlated ensemble in T3.2/T3.3.
            print("  --copula t ignored for --independent: the null stays Gaussian, which is "
                  "the same marginal and no dependence")
            cfg["copula"] = "gaussian"
            # attrs were written from the CLI value above; record what actually ran, so a
            # reader of the null store is not told it carries a dependence it does not have.
            store.attrs["copula"] = "gaussian"
            store.attrs["copula_requested"] = "t"
        else:
            print(f"  copula: Student-t, df={args.copula_df} "
                  f"(one chi2_{args.copula_df:g}/{args.copula_df:g} factor per member)")

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""] if args.gpus else [None]
    assignment = assign_members(args.members, len(gpus))
    t0 = time.time()
    if len(gpus) == 1:
        session = worker(0, gpus[0], assignment[0], cfg, session=session)
    else:
        # Each worker gets a fork of the session, writes its own member blocks, and hands
        # the change record back; the parent merges them and commits once. A ProcessPool
        # rather than bare Processes because the forked sessions have to come *back*.
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("spawn")
        fork = session.fork()
        with ProcessPoolExecutor(max_workers=len(gpus), mp_context=ctx) as ex:
            futures = [ex.submit(worker, wid, gpu, members, cfg, fork)
                       for wid, (gpu, members) in enumerate(zip(gpus, assignment))]
            done = [f.result() for f in futures]
        session.merge(*done)
    snapshot = session.commit(
        f"{args.members} members x {len(years)} horizons"
        + (" (independent-pixel null)" if args.independent else ""))
    print(f"  committed snapshot {snapshot}")
    elapsed = time.time() - t0

    manifest = {
        "members": [
            {"member": m, "seeds": {str(h): args.seed + 1_000_003 * m + 101 * h for h in horizons},
             "ar1_chain": [f"z_{horizons[0]}"] + [f"z_{h} = rho_{h} z_{p} + sqrt(1-rho^2) eps_{h}"
                                                  for p, h in zip(horizons[:-1], horizons[1:])]}
            for m in range(args.members)
        ],
        "rho": rho, "field_params": {str(k): v for k, v in fps.items()},
        "copula": args.copula, "copula_df": float(args.copula_df),
        "copula_w_draw": args.copula_w_draw,
        "copula_w": ({str(m): float(v) for m, v in enumerate(cfg["copula_w"])}
                     if args.copula == "t" else None),
        "years": years, "wrap_lon": bool(wrap), "independent_null": bool(args.independent),
        "minutes": elapsed / 60.0,
    }
    with open(out_path.parent / (out_path.stem + "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    du = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 1e9
    print(f"\n✓ {args.members} members x {len(years)} horizons in {elapsed / 60:.1f} min "
          f"— {du:.1f} GB on disk")
    print("  NOTE: with M=50 the ensemble's own 2.5/97.5 percentiles carry real Monte-Carlo "
          "noise (they fall between the 1st and 2nd order statistics). T5.2 must be scored "
          "with an MC-scaled tolerance; tail-sensitive per-pixel products want M >= 200.")

    if not args.disable_wandb:
        try:
            import wandb
            run = wandb.init(project="spatio-temporal-convlstm",
                             group=args.wandb_group or "ensemble-generation",
                             job_type="ensemble-generation", tags=["ensemble", "phase3"],
                             config={**vars(args), "n_valid_px": int(idx.size)})
            run.log({"minutes": elapsed / 60.0, "disk_gb": du, "members": args.members})
            run.finish()
        except Exception as e:
            print(f"⚠ W&B unavailable ({e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
