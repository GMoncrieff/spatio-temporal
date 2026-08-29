#!/usr/bin/env python3
"""Prove the ensemble's marginal is the quantile function it was sampled from.

Two instruments sharing no code is how a bug gets localised in this project, and the sampler
and the raster are exactly such a pair: ``generate_ensemble.py --qf_dir`` maps ``Phi(z)``
through the stored quantile function on the GPU, and this reads the members back and asks
whether their empirical distribution is the one the raster describes. A wrong scale, a wrong
pixel ordering, or a level lookup off by one all survive a visual check of the output and none
of them survives this.

``--mode bounds`` is the cheap pre-flight: every member value must lie inside
``[Q(u_min), Q(u_max)]`` at its own pixel, which the sampler guarantees by construction and
which a mis-indexed slab violates immediately. Runs at any M.

``--mode marginal`` is the real test, and it is stated for a **discrete** distribution because
this one is: the quantile function is clipped at HM=0, so a genuine atom sits at zero and a
large share of members take exactly that value. For an atom the right statement is a sandwich,

    P(V < Q(u))  <=  u  <=  P(V <= Q(u)),

not ``P(V <= Q(u)) == u``. Written the naive way this check failed by +0.11 at mid-grid on a
sampler that is exactly correct, with the signature of a real defect: a smooth, one-sided,
mid-range bias vanishing at both tails. When a number looks impossible, suspect the metric.
Needs M large enough for the noise to be smaller than what it looks for -- at M=400 the
standard error on a mid-grid level is ~0.025.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.aggregate import open_ensemble  # noqa: E402

INT16_SCALE = 1.0 / 32767.0
SENTINEL = -32768


def read_qf(path):
    with rasterio.open(path) as src:
        u = np.array([float(v) for v in src.tags()["u_levels"].split(",")], dtype=np.float64)
        q = src.read()
        nod = src.nodata
    return u, q, nod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--qf", required=True, help="the quantile-function raster for one horizon")
    ap.add_argument("--horizon_index", type=int, default=0,
                    help="index of that horizon in the store's horizon axis")
    ap.add_argument("--mode", default="marginal", choices=["bounds", "marginal"])
    ap.add_argument("--n_px", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max |empirical - u| allowed at any checked level, mode=marginal")
    args = ap.parse_args(argv)

    u, q, nod = read_qf(args.qf)
    arr, attrs = open_ensemble(args.ensemble)
    M = arr.shape[0]
    H, W = arr.shape[2], arr.shape[3]
    if q.shape[1:] != (H, W):
        raise SystemExit(f"qf grid {q.shape[1:]} != ensemble grid {(H, W)}")

    rng = np.random.default_rng(args.seed)
    # Sample rows, then valid columns inside them: reading scattered single pixels from an
    # int16 store chunked (1, 1, 1024, 1024) would fetch a 2 MB chunk per pixel.
    rows = rng.choice(H, size=min(64, H), replace=False)
    lo_band = q[0][rows]
    ok = (lo_band != nod) & np.isfinite(lo_band)
    if not ok.any():
        raise SystemExit("no valid pixels in the sampled rows")

    vals, qlo, qhi, qsel = [], [], [], []
    per_row = max(1, args.n_px // len(rows))
    for i, r in enumerate(rows):
        cols = np.flatnonzero(ok[i])
        if cols.size == 0:
            continue
        cols = rng.choice(cols, size=min(per_row, cols.size), replace=False)
        cols.sort()
        block = arr[:, args.horizon_index, r, :]           # (M, W) one row, all members
        v = block[:, cols].astype(np.float64)
        good = np.all(v != SENTINEL, axis=0)
        v = v[:, good] * attrs.get("scale", INT16_SCALE)
        cols = cols[good]
        if cols.size == 0:
            continue
        vals.append(v)
        qlo.append(q[0, r, cols] * INT16_SCALE)
        qhi.append(q[-1, r, cols] * INT16_SCALE)
        qsel.append(q[:, r, cols] * INT16_SCALE)

    if not vals:
        raise SystemExit("no usable pixels sampled")
    V = np.concatenate(vals, axis=1)                       # (M, n_px)
    QLO, QHI = np.concatenate(qlo), np.concatenate(qhi)
    Q = np.concatenate(qsel, axis=1)                       # (levels, n_px)
    n = V.shape[1]
    print(f"ensemble {args.ensemble}\n  M={M}, sampled {n:,} pixels, "
          f"marginal='{attrs.get('marginal', '?')}'")

    if args.mode == "bounds":
        # One int16 quantum of slack: the sampler clamps to the stored end quantiles and both
        # sides are quantised, so an exact-equality test would fail on rounding alone.
        tol = 2 * INT16_SCALE
        below = (V < QLO[None, :] - tol).sum()
        above = (V > QHI[None, :] + tol).sum()
        total = V.size
        print(f"  below Q(u_min): {below:,} / {total:,}   above Q(u_max): {above:,} / {total:,}")
        if below or above:
            raise SystemExit("FAIL: members fall outside the quantile function's own range — "
                             "the slab is mis-indexed or the scale is wrong")
        print("  ✓ every member value lies inside its pixel's stored quantile range")
        return 0

    atom = float((V == Q[0][None, :]).mean())
    print(f"  atom at the lowest stored quantile: {atom:.4f} of member-pixels "
          f"(the quantile function is clipped at HM=0, so this is real, not a bug)")
    print(f"  {'u':>9}{'P(V<Q)':>10}{'P(V<=Q)':>10}{'miss':>9}")
    worst = 0.0
    for j, level in enumerate(u):
        lo = float((V < Q[j][None, :]).mean())
        hi = float((V <= Q[j][None, :]).mean())
        # Distance from u to the sandwich, zero when it lies inside.
        miss = max(0.0, lo - level, level - hi)
        worst = max(worst, miss)
        if j % max(1, len(u) // 12) == 0 or miss > args.tol:
            print(f"  {level:>9.5f}{lo:>10.5f}{hi:>10.5f}{miss:>+9.5f}")
    se = float(np.sqrt(0.25 / M))
    print(f"  worst distance outside [P(V<Q), P(V<=Q)] = {worst:.5f}  "
          f"(binomial SE at u=0.5 is {se:.5f} for M={M})")
    if worst > args.tol:
        raise SystemExit(f"FAIL: the sampled marginal is not the stored one (worst {worst:.4f} "
                         f"> tol {args.tol})")
    print("  ✓ the members reproduce the quantile function they were sampled from")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
