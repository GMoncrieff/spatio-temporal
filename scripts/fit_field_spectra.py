#!/usr/bin/env python3
"""Fit the Phase 2 field structure by matching the observed power spectrum.

The variogram fit is done at lags, where two long structures can absorb the fit while
leaving the mid-frequency band empty. Measured against the observed residual field, the
variogram-fitted Gaussian mixture carried 4.9x too much power beyond 50 px and 0.47x what
it should at 3-10 px — the band holding 47% of the observed variance — which reads
visually as smooth blobs where the truth is structured speckle.

Fitting the radial spectrum instead, over a fixed range basis with a Matérn kernel, brings
those to 0.94x and 0.88x.

The field is fitted on the *standardized* residual (Y − central)/sigma, because that is
precisely what the copula's normal score z represents.
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

from src.ensemble.fields import DEFAULT_BASIS_RANGES, fit_spectral_mixture, radial_power_spectrum

BANDS = [(0, 0.02), (0.02, 0.1), (0.1, 0.35), (0.35, 0.71)]
BAND_LABELS = [">50px", "10-50px", "3-10px", "1-3px"]


def band_shares(field):
    k, _, total = radial_power_spectrum(field)
    total = total / total.sum()
    return [float(total[(k >= lo) & (k < hi)].sum()) for lo, hi in BANDS]


def standardized_residual(row, clip=5.0):
    with rasterio.open(row["path_res_native"]) as s:
        res = s.read(1).astype(np.float64)
    with rasterio.open(row["path_w_up"]) as s:
        wu = s.read(1).astype(np.float64)
    with rasterio.open(row["path_w_lo"]) as s:
        wl = s.read(1).astype(np.float64)
    sigma = np.where(res >= 0, np.maximum(wu, 1e-6), np.maximum(wl, 1e-6)) / 1.959964
    e = res / sigma
    # Clip before the FFT: a handful of pixels with near-degenerate widths would otherwise
    # dominate the spectrum through their magnitude alone.
    return np.where(np.isfinite(e), np.clip(e, -clip, clip), np.nan)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", default="matern", choices=["matern", "gaussian"])
    ap.add_argument("--nu", type=float, default=0.5)
    ap.add_argument("--ranges", default=",".join(str(r) for r in
                                                 (1.5, 2.5, 4, 6, 9, 14, 25, 50, 120, 300)))
    args = ap.parse_args(argv)

    ranges = tuple(float(x) for x in args.ranges.split(","))
    man = pd.read_csv(args.manifest)
    fits = {}
    print(f"{'tag':>12} {'nugget':>7} | " + " ".join(f"{b:>9}" for b in BAND_LABELS))
    for _, row in man.iterrows():
        h = int(row["horizon"])
        tag = f"w{int(row['base_year'])}_h{h}"
        e = standardized_residual(row)
        if np.isfinite(e).sum() < 10_000:
            continue
        fit = fit_spectral_mixture(e, ranges_px=ranges, kernel=args.kernel, nu=args.nu)
        obs = band_shares(e)
        fit["observed_band_shares"] = obs
        fit["band_labels"] = BAND_LABELS
        # Keep the best fit per horizon (windows agree closely; the longest record wins).
        fits.setdefault(str(h), fit)
        print(f"{tag:>12} {fit['nugget']:>7.3f} | " + " ".join(f"{v:>9.4f}" for v in obs))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"kernel": args.kernel, "nu": args.nu, "by_horizon": fits}, f, indent=2)
    print(f"\n✓ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
