#!/usr/bin/env python3
"""Read the delivered products back and check them against the rasters they came from.

``export_products.py`` verifies its own COG *layout*, which says the container is right and
nothing about the numbers in it. This says the numbers are right, on the files as written, by
sampling windows and comparing four independent things:

  passthrough   lower / mean / upper must equal the published band they were copied from,
                to the bit. They are a format change, not a computation.
  exceedance    gt01 and gt04 are ``1 - F(x)``, so ``Q(1 - p)`` must come back to ``x``.
                That is a ROUND TRIP through the inverse rather than a second spelling of
                the same interpolation, so it catches the failures that actually happen: the
                wrong band, the wrong threshold, the wrong year, a transposed window.
  ordering      ``P(HM > 0.1) >= P(HM > 0.4)`` everywhere, and both inside [0, 1]. A
                one-line check that no rearrangement can pass by accident.
  icechunk      the uint16 store against the float32 source, which must agree to half a
                quantisation step (7.63e-06). Plus the shape, the dtype, the chunk and shard
                grid and the percentile axis being whole — the read-granularity promise is
                part of the deliverable, not an implementation detail.

Why sampled windows rather than whole rasters: a global float32 band is 2.7 GB and the
quantile raster is 175 GB, and a verification that costs more than the export does not get
run. The failures this catches are structural and show in any window that has data.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from export_products import PASSTHROUGH, THRESHOLDS, VARIABLES, src_paths, U16_MAX  # noqa: E402
from score_distributional_model import qf_levels  # noqa: E402

def sample_windows(central_path: Path, n_px: int, seed: int = 0, max_blocks: int = 30000):
    """The raster's OWN tiles, in a seeded shuffled order, keeping the ones with data.

    Not uniformly random offsets. The global grid is 73% ocean, and a smoke run predicts a
    few dozen 128 px blocks of it, so uniform sampling finds nothing and reports success
    having compared zero pixels -- which is exactly how ``make_cogs.verify`` came to pass on
    twelve all-ocean windows. Walking the tiling is aligned to the file's own blocks, so it
    is cheap, and it terminates on the data rather than on a try count.
    """
    rng = np.random.default_rng(seed)
    with rasterio.open(central_path) as src:
        blocks = [w for _, w in src.block_windows(1)]
        order = rng.permutation(len(blocks))[:max_blocks]
        out, seen = [], 0
        for i in order:
            w = blocks[int(i)]
            m = np.isfinite(src.read(1, window=w))
            k = int(m.sum())
            if k < 16:
                continue
            out.append((int(w.row_off), int(w.col_off), int(w.height), int(w.width), m))
            seen += k
            if seen >= n_px:
                break
    if not out:
        raise SystemExit(
            f"{central_path}: no tile carries data. A verification that compares zero "
            f"pixels reports success; this refuses instead.")
    return out


def read_win(path, r0, c0, h, w):
    with rasterio.open(path) as src:
        return src.read(window=rasterio.windows.Window(c0, r0, w, h))


def quantile_at_vec(u, q, levels):
    """``Q(level)`` per pixel for a per-pixel level. q is (n_u, n_px), levels is (n_px,)."""
    j = np.clip(np.searchsorted(u, levels), 1, u.size - 1)
    t = (levels - u[j - 1]) / (u[j] - u[j - 1])
    cols = np.arange(q.shape[1])
    return q[j - 1, cols] + t * (q[j, cols] - q[j - 1, cols])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", required=True, help="the --out_dir given to export_products")
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--mode", choices=["hindcast", "forecast"], required=True)
    ap.add_argument("--base_year", type=int, required=True)
    ap.add_argument("--years", required=True)
    ap.add_argument("--n_px", type=int, default=20000,
                    help="stop sampling once this many valid pixels have been compared")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rt_tol", type=float, default=2e-3,
                    help="tolerance on the exceedance round trip Q(1-p) vs the threshold")
    a = ap.parse_args(argv)

    years = [int(y) for y in a.years.split(",")]
    prod, src_dir = Path(a.products), Path(a.src_dir)
    cog_dir = prod / "cogs"
    bad = []

    for year in years:
        sp = src_paths(src_dir, a.mode, a.base_year, year)
        u = qf_levels(str(sp["qf"]))[0]
        wins = sample_windows(sp["central"], a.n_px, a.seed + year)
        n_seen = 0
        rt_worst = {k: 0.0 for k in THRESHOLDS}
        pass_worst = {k: 0.0 for k in PASSTHROUGH}
        for r0, c0, hh, ww, m in wins:
            n_seen += int(m.sum())
            q = read_win(sp["qf"], r0, c0, hh, ww)[:, m]              # (n_u, n_px)
            for var, band in PASSTHROUGH.items():
                cog = cog_dir / f"hm_{year}_{var}.tif"
                if not cog.is_file():
                    bad.append(f"{cog} missing"); continue
                got = read_win(cog, r0, c0, hh, ww)[0][m]
                want = read_win(sp[band], r0, c0, hh, ww)[0][m]
                ok = np.isfinite(got) & np.isfinite(want)
                if ok.any():
                    pass_worst[var] = max(pass_worst[var],
                                          float(np.abs(got[ok] - want[ok]).max()))
            ps = {}
            for var, thr in THRESHOLDS.items():
                cog = cog_dir / f"hm_{year}_{var}.tif"
                if not cog.is_file():
                    bad.append(f"{cog} missing"); continue
                p = read_win(cog, r0, c0, hh, ww)[0][m]
                ps[var] = p
                if np.nanmin(p) < -1e-6 or np.nanmax(p) > 1 + 1e-6:
                    bad.append(f"{year} {var}: outside [0, 1] "
                               f"({np.nanmin(p):.4g}, {np.nanmax(p):.4g})")
                # Interior only: p == 0 or 1 means the threshold is outside the represented
                # range, where the inverse is not defined and the clamp is the right answer.
                interior = np.isfinite(p) & (p > 1.0 / len(u)) & (p < 1 - 1.0 / len(u))
                if interior.any():
                    back = quantile_at_vec(u, q[:, interior], 1.0 - p[interior])
                    rt_worst[var] = max(rt_worst[var],
                                        float(np.abs(back - thr).max()))
            if "gt01" in ps and "gt04" in ps:
                ok = np.isfinite(ps["gt01"]) & np.isfinite(ps["gt04"])
                viol = int((ps["gt01"][ok] < ps["gt04"][ok] - 1e-7).sum())
                if viol:
                    bad.append(f"{year}: P(>0.1) < P(>0.4) at {viol} pixels")
        for var, d in pass_worst.items():
            if d != 0.0:
                bad.append(f"{year} {var}: COG differs from the published band by {d:.3g}")
        for var, d in rt_worst.items():
            if d > a.rt_tol:
                bad.append(f"{year} {var}: Q(1-p) misses the threshold by {d:.3g} "
                           f"(tol {a.rt_tol:g})")
        # Print the measured worst case, never the word "exact". A line that says the same
        # thing whether or not the check passed is not evidence of anything (rule 28).
        print(f"  {year}: {n_seen:,} px in {len(wins)} windows | passthrough max |d| "
              f"{max(pass_worst.values()):.3g} | round trip gt01 {rt_worst['gt01']:.2e} "
              f"gt04 {rt_worst['gt04']:.2e}")

    # ---------------------------------------------------------------- icechunk
    store = prod / f"{a.mode}_qf.icechunk"
    if store.exists():
        import icechunk
        import zarr
        repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store)))
        root = zarr.open_group(repo.readonly_session("main").store, mode="r")
        hm = root["hm"]
        sy = [int(v) for v in root["year"][:]]
        first = src_paths(src_dir, a.mode, a.base_year, years[0])
        u = qf_levels(str(first["qf"]))[0]
        with rasterio.open(first["qf"]) as s:
            H, W = s.height, s.width
        want_shape = (len(years), len(u), H, W)
        if tuple(hm.shape) != want_shape:
            bad.append(f"icechunk hm shape {tuple(hm.shape)}, expected {want_shape}")
        if hm.dtype != np.uint16:
            bad.append(f"icechunk hm dtype {hm.dtype}, expected uint16")
        if sy != years:
            bad.append(f"icechunk year axis {sy}, expected {years}")
        ch = tuple(hm.chunks)
        if ch[0] != 1 or ch[1] != len(u):
            bad.append(f"icechunk chunks {ch}: the percentile axis must be whole "
                       f"(1, {len(u)}, lat, lon)")
        sh = tuple(getattr(hm, "shards", None) or ())
        if sh and (sh[2] % ch[2] or sh[3] % ch[3]):
            bad.append(f"icechunk shards {sh} are not a whole number of chunks {ch}")
        scale = float(hm.attrs.get("scale_factor", 0.0))
        if abs(scale - 1.0 / U16_MAX) > 1e-12:
            bad.append(f"icechunk scale_factor {scale}, expected {1.0 / U16_MAX}")
        worst = 0.0
        collide = 0
        mis = 0
        for yi, year in enumerate(years):
            sp = src_paths(src_dir, a.mode, a.base_year, year)
            for r0, c0, hh, ww, m in sample_windows(sp["central"], max(2048, a.n_px // 8),
                                                    a.seed + year)[:8]:
                src_q = read_win(sp["qf"], r0, c0, hh, ww)[:, m]
                got = hm[yi, :, r0:r0 + hh, c0:c0 + ww].reshape(len(u), -1)[:, m.ravel()]
                fin = np.isfinite(src_q)
                # Compare against the PRESCRIBED ENCODING, not against the source with a
                # tolerance. The encoding is exactly defined -- round to the nearest 1/65535,
                # then clamp off the fill code -- so the store either holds that integer or
                # it does not, and a bitwise test needs no tolerance to argue about. The old
                # form compared decoded HM against the source within half a step and had to
                # be re-argued the moment the clamp was added, because a clamped 1.0 is a
                # full step away by design.
                # In the SOURCE's own dtype, which is float32. Promoting to float64 first
                # invents precision the raster does not have: recomputed in float64, 252 of
                # 4.19 M sampled codes came out +/-1 different from what float32 arithmetic
                # produces, and the check then failed a store that was byte-for-byte what the
                # encoder prescribes. An expectation must be computed the way the thing it
                # checks computes it.
                want = np.minimum(np.rint(np.clip(src_q, 0.0, 1.0) * U16_MAX), U16_MAX - 1)
                if fin.any():
                    n_off = int((got[fin] != want[fin]).sum())
                    if n_off:
                        mis += n_off
                    dec = got.astype(np.float64) * scale
                    ref = np.clip(src_q.astype(np.float64), 0.0, 1.0)
                    worst = max(worst, float(np.abs(dec[fin] - ref[fin]).max()))
                # The fill value must mean MISSING and nothing else. The round trip above
                # cannot catch a collision, because 65535 * 1/65535 decodes to 1.0 and
                # compares equal to a source of 1.0 -- the value is right and its MEANING is
                # wrong. So check the other half of the encoding: a cell may hold the fill
                # value only where the source is not finite.
                collide += int((got[fin] == U16_MAX).sum())
        half_step = 0.5 / U16_MAX
        # 1.05, not 1.01: half a uint16 step is 7.629e-06 and correct rounding of a float32
        # source lands a hair above it. A real defect here is orders of magnitude larger.
        if collide:
            bad.append(f"icechunk: {collide:,} sampled cells hold the fill value {U16_MAX} "
                       f"while the source is FINITE -- real data written as missing. "
                       f"HM >= {(U16_MAX - 0.5) / U16_MAX:.8f} encodes to the fill value "
                       f"unless the code is clamped to {U16_MAX - 1}.")
        if mis:
            bad.append(f"icechunk: {mis:,} sampled cells do not hold the prescribed code "
                       f"min(rint(clip(q,0,1) * {U16_MAX}), {U16_MAX - 1})")
        # One full step, not half: a source of exactly 1.0 is clamped to 65534 by design and
        # therefore decodes 1/65535 low. That is the documented representable maximum.
        if worst > (1.0 / U16_MAX) * 1.05:
            bad.append(f"icechunk decoded HM off by {worst:.3g}, more than one uint16 "
                       f"step ({1.0 / U16_MAX:.3g})")
        print(f"  icechunk: shape {tuple(hm.shape)} {hm.dtype} chunks {ch} shards {sh or '-'} "
              f"| round trip max |err| {worst:.3g} (half step {half_step:.3g}) "
              f"| fill/finite collisions {collide:,} | code mismatches {mis:,}")
    else:
        print(f"  · no icechunk store at {store} (skipped)")

    if bad:
        print("FATAL:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    print("  ✓ products verified against their source rasters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
