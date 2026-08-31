#!/usr/bin/env python3
"""Assert the separable draw's two by-construction claims against the members it produced.

`Z = chol(R) . E` claims that (a) the horizons come back with correlation `R` and (b) every
horizon keeps the shared spatial spectrum. "By construction" is exactly how a wrong
implementation gets waved through, so both are measured from the stored ensemble.

T4.1 cannot do this job here. Its reference is `rho_json`, which pools adjacent pairs across
windows, while `R` is estimated on pixels finite in *every* horizon in one window — different
quantities (0.692/0.740/0.843 against 0.805/0.765/0.843). A separable run reproduces `R` and is
then marked failing by T4.1. This compares against `R` itself.

    scripts/check_separable_horizon.py --ensemble <members>.icechunk --R <horizon_corr.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.ensemble import aggregate as agg  # noqa: E402
from src.ensemble.residuals import qf_normal_score  # noqa: E402

QF_SCALE = 1.0 / 32767.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--R", required=True)
    ap.add_argument("--qf_dir", default="data/ensemble/exp/af_e1_hind/stitched")
    ap.add_argument("--base_year", type=int, default=2000)
    ap.add_argument("--years", default="2005,2010,2015,2020")
    ap.add_argument("--window", type=int, default=2560,
                    help="Side of the contiguous window the sample is drawn from. Reading "
                         "four whole 64-band quantile rasters is 32 GB of I/O; the statistic "
                         "is a pooled correlation, not a spatial one, so a window is enough.")
    ap.add_argument("--n_points", type=int, default=120_000)
    ap.add_argument("--n_members", type=int, default=60)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args(argv)

    years = [int(y) for y in args.years.split(",")]
    blob = json.load(open(args.R))
    R = np.asarray(blob["R"], dtype=float)

    store, attrs = agg.open_ensemble(args.ensemble)
    M, nH, H, W = store.shape
    print(f"ensemble {store.shape}")

    # One pixel sample, shared by every horizon, so the correlation is measured on a common
    # support exactly as R was estimated.
    from rasterio.windows import Window

    rng = np.random.default_rng(args.seed)
    side = min(args.window, H, W)
    r0 = int(rng.integers(0, max(1, H - side)))
    c0 = int(rng.integers(0, max(1, W - side)))
    win = Window(c0, r0, side, side)
    us, qs, keep = [], [], None
    for y in years:
        p = Path(args.qf_dir) / f"w{args.base_year}_prediction_{y}_qf.tif"
        with rasterio.open(p) as src:
            u = np.array([float(v) for v in src.tags()["u_levels"].split(",")], dtype=np.float64)
            q = src.read(window=win).astype(np.float32)
            nod = src.nodata
        q = (np.where(q == nod, np.nan, q) * np.float32(QF_SCALE)).reshape(q.shape[0], -1)
        ok = np.isfinite(q[0]) & np.isfinite(q[-1])
        keep = ok if keep is None else (keep & ok)
        us.append(u); qs.append(q)
    local = np.flatnonzero(keep)
    if local.size == 0:
        raise SystemExit(f"window at ({r0}, {c0}) is empty; try another --seed")
    local = np.sort(rng.choice(local, size=min(args.n_points, local.size), replace=False))
    rr, cc = np.unravel_index(local, (side, side))
    idx = ((rr + r0) * W + (cc + c0)).astype(np.int64)
    qs = [q[:, local].astype(np.float64) for q in qs]
    print(f"  window ({r0}, {c0}) {side}x{side}; {idx.size} pixels finite in every horizon")

    Z = []
    for hi in range(len(years)):
        X = agg.members_at_points(store, attrs, hi, idx, H, W)[:args.n_members]
        zz = []
        for m in range(X.shape[0]):
            v = X[m].astype(np.float64)
            g = np.isfinite(v)
            z = np.full(v.shape, np.nan)
            if g.any():
                z[g] = qf_normal_score(v[g], us[hi], qs[hi][:, g])
            zz.append(z)
        Z.append(np.concatenate(zz))
    Z = np.stack(Z)
    ok = np.isfinite(Z).all(axis=0)
    got = np.corrcoef(Z[:, ok])

    print("\ntarget R:");    print(np.array2string(R, precision=4, suppress_small=True))
    print("delivered:");     print(np.array2string(got, precision=4, suppress_small=True))
    d = np.abs(got - R)
    print(f"\nmax |delivered - R| = {d.max():.4f} at "
          f"{np.unravel_index(d.argmax(), d.shape)}   (tolerance {args.tol})")
    print(f"adjacent pairs: " + ", ".join(f"{got[i, i+1]:.3f} vs {R[i, i+1]:.3f}"
                                          for i in range(len(years) - 1)))
    ok_flag = bool(d.max() <= args.tol)
    print("✓ the members reproduce R" if ok_flag else "✗ the members do NOT reproduce R")
    return 0 if ok_flag else 1


if __name__ == "__main__":
    sys.exit(main())
