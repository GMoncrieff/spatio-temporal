#!/usr/bin/env python3
"""Does a width-factor configuration keep the far field's extreme tail?

The interval score is an average over pixels, so it is structurally blind to a defect that
lives in a thousandth of them. That blindness cost this project a scorecard: a biome-keyed
calibration won the held-out interval score by 0.8% while halving the far band's p99.9
half-width, and `P(Δ>0.05)` beyond 100 px collapsed from 0.845 of observed to 0.042. The
average far-field pixel got *wider* the whole time.

The reason is what class averages do. A class that mixes remote quiet land with the rare
remote pixel that has a genuinely wide interval hands both the same factor, so the tail is
pulled toward the middle. Only the pixels in that tail can reach +0.05 in remote country, so
losing them removes the entire far-field change signal.

This measures the thing directly: per distance band, the ratio of the corrected half-width's
upper percentiles to the uncorrected ones. A configuration that leaves a band alone scores
1.000 there by construction. Anything much below 1 at p99.9 in the far bands will break T8,
and it will do so without moving any pooled number enough to notice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.ensemble.validate import (  # noqa: E402
    DHAT_BINS, DIST_LABELS, HM_BINS, distance_band,
)
from score_width_variants import factor_arrays, load_factors  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--variants", nargs="+", required=True, help="label=path.json")
    ap.add_argument("--percentiles", default="50,99,99.9")
    args = ap.parse_args(argv)

    qs = [float(q) for q in args.percentiles.split(",")]
    man = pd.read_csv(args.manifest)
    row = man[man["horizon"].astype(int) == args.horizon].iloc[0]

    with rasterio.open(row["path_w_up"]) as s:
        wu = s.read(1).astype(np.float64)
    with rasterio.open(row["path_dhat"]) as s:
        dhat = s.read(1).astype(np.float64)
    with rasterio.open(row["path_hm_t0"]) as s:
        hm0 = s.read(1).astype(np.float64)
    with rasterio.open(row["path_dist_past_change"]) as s:
        band = distance_band(s.read(1).astype(np.float64)).astype(np.int64)
    ok = np.isfinite(wu) & (wu > 0) & np.isfinite(dhat) & np.isfinite(hm0)
    d_idx = np.digitize(dhat, DHAT_BINS[1:-1])
    hm_idx = np.digitize(hm0, HM_BINS[1:-1])

    print(f"h={args.horizon}: upper half-width percentile ratio vs the uncorrected heads")
    print(f"  1.000 = band untouched. Watch p99.9 in the far bands — that is the tail T8 needs.\n")
    hdr = f"{'variant':>10} {'band':>8} " + " ".join(f"{'p'+str(q):>9}" for q in qs)
    print(hdr)
    for spec in args.variants:
        label, _, path = spec.partition("=")
        factors = load_factors(path)
        k_up, _ = factor_arrays(factors, args.horizon, band, d_idx, hm_idx, tag=label)
        new = wu * k_up
        for b, blbl in enumerate(DIST_LABELS):
            m = ok & (band == b)
            if m.sum() < 1000:
                continue
            cells = []
            for q in qs:
                a = np.percentile(wu[m], q)
                c = np.percentile(new[m], q)
                cells.append(f"{(c / a if a > 0 else np.nan):>9.3f}")
            print(f"{label:>10} {blbl:>8} " + " ".join(cells))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
