#!/usr/bin/env python3
"""Fit the Phase 2 field structure by matching the observed power spectrum.

The variogram fit is done at lags, where two long structures can absorb the fit while
leaving the mid-frequency band empty. Measured against the observed residual field, the
variogram-fitted Gaussian mixture carried 4.9x too much power beyond 50 px and 0.47x what
it should at 3-10 px — the band holding 47% of the observed variance — which reads
visually as smooth blobs where the truth is structured speckle.

Fitting the radial spectrum instead, over a fixed range basis with a Matérn kernel, brings
those to 0.94x and 0.88x.

Two fit spaces are available. ``--fit_space width`` (the default, and what every card up to
2026-08-29 was built on) uses the *standardized* residual ``(Y - central)/sigma`` with sigma
read off the interval half-widths. ``--fit_space pit`` uses the observation's own probability
rank under the forecast, ``Phi^-1(F_qf(Y))`` -- which is what the copula's normal score
actually *is*, rather than a three-point approximation to it.

The width form uses Q(0.025), Q(0.5) and Q(0.975) and assumes the rest is symmetric. The
model emits sixty-four levels, HM is bounded below, 40% of Africa sits in [0, 0.01) and the
atom at HM = 0 is real, so the assumption fails hardest in the pixels that dominate the
count -- and those pixels are what set the nugget and the shortest-range weight. Measured on
the e1 Africa hindcast (``scripts/diag_pit_vs_width_residual.py``): the two spaces put
0.10-0.15 of total variance in different places on the 4-50 px band, against a
realisation-noise floor of 0.044, and the width field carries a 0.62-sigma region-wide mean
at h=20 where the PIT field carries 0.03.
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
from src.ensemble.residuals import pit_normal_field

BANDS = [(0, 0.02), (0.02, 0.1), (0.1, 0.35), (0.35, 0.71)]
BAND_LABELS = [">50px", "10-50px", "3-10px", "1-3px"]

# The band of correlation ranges that still carries structure at the separation T3.2 scores
# (mean 28.5 px on Africa). Below it the nugget and the ~1.5 px component are white noise to
# the variogram score; above it a component is a constant across the pair. Reported by
# --report_windows so the identifiability question has a number attached to it.
RESOLVABLE_PX = (4.0, 50.0)

# The production basis, so a diagnostic and the fitter cannot drift apart.
BASIS_DEFAULT = (1.5, 2.5, 4.0, 6.0, 9.0, 14.0, 25.0, 50.0, 120.0, 300.0)


def weight_shares(ranges_px, weights, nugget):
    """(nugget, <4px, 4-50px, >50px) shares of the fitted mixture's unit variance."""
    r, w = np.asarray(ranges_px, float), np.asarray(weights, float)
    lo, hi = RESOLVABLE_PX
    return (float(nugget), float(w[r < lo].sum()),
            float(w[(r >= lo) & (r <= hi)].sum()), float(w[r > hi].sum()))


def band_shares(field):
    k, _, total = radial_power_spectrum(field)
    total = total / total.sum()
    return [float(total[(k >= lo) & (k < hi)].sum()) for lo, hi in BANDS]


def add_long_scale_component(fit, long_range, long_weight):
    """Reserve variance for a scale the spectral fit is structurally blind to.

    fit_spectral_mixture removes the field mean before the FFT, so the k=0 component — a
    region-wide offset, which is what an ecoregion-mean error mostly is — cannot appear in
    the fitted mixture. Measured on southern Africa: the pure spectral fit puts ~5% of
    variance beyond 300 px and lands at a spread-skill ratio of 0.564, i.e. the ensemble
    understates ecoregion-mean uncertainty twofold, while the texture it does capture is
    correct. Adding an explicit long component restores the aggregate without touching the
    fine structure (0.40 -> spread-skill 1.038, ecoregion coverage 0.611 -> 0.889/1.000,
    and the 3-10 px band stays at 0.73x of observed).
    """
    keep = 1.0 - float(long_weight)
    total = sum(fit["weights"]) + fit["nugget"]
    out = dict(fit)
    out["weights"] = [w / total * keep for w in fit["weights"]] + [float(long_weight)]
    out["ranges_px"] = list(fit["ranges_px"]) + [float(long_range)]
    out["nugget"] = fit["nugget"] / total * keep
    out["long_scale"] = {"range_px": float(long_range), "weight": float(long_weight)}
    return out


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


def fit_field(row, args, clip=5.0):
    """The field Phase 2 fits its spectrum to, in whichever space was asked for."""
    if args.fit_space == "width":
        return standardized_residual(row, clip=clip)
    qf = Path(args.qf_dir) / args.qf_pattern.format(
        base=int(row["base_year"]), year=int(row["target_year"]))
    if not qf.exists():
        raise SystemExit(f"--fit_space pit needs {qf}, which is missing")
    z = pit_normal_field(str(qf), row["path_observed"])
    # Clipped on the same argument as the width form: the PIT grid truncates at u = 1e-4,
    # so |z| never exceeds 3.72 by construction and this is a no-op at 5.0 -- kept so the
    # two spaces differ in the transform and in nothing else.
    return np.where(np.isfinite(z), np.clip(z, -clip, clip), np.nan)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", default="matern", choices=["matern", "gaussian"])
    ap.add_argument("--nu", type=float, default=0.5)
    ap.add_argument("--ranges", default=",".join(str(r) for r in BASIS_DEFAULT))
    ap.add_argument("--fit_space", default="width", choices=["width", "pit"],
                    help="'width' = (Y-central)/sigma, today's behaviour. 'pit' = "
                         "Phi^-1(F_qf(Y)), the observation's own probability rank under the "
                         "forecast, which is the quantity the copula's latent represents. "
                         "'pit' needs --qf_dir.")
    ap.add_argument("--qf_dir", default=None,
                    help="Directory of the 64-band quantile-function rasters; required by "
                         "--fit_space pit.")
    ap.add_argument("--qf_pattern", default="w{base}_prediction_{year}_qf.tif")
    ap.add_argument("--report_windows", action="store_true",
                    help="Print the fitted mixture for EVERY window, not just the one that "
                         "survives `fits.setdefault`, plus the spread across origins at a "
                         "fixed horizon. Diagnostic only; does not change what is written.")
    ap.add_argument("--long_range", type=float, default=1500.0,
                    help="Range of the aggregate-scale component (px)")
    ap.add_argument("--long_weight", type=float, default=0.40,
                    help="Variance share given to that component. The spectral fit cannot "
                         "see it: fit_spectral_mixture subtracts the field mean before the "
                         "FFT, so region-wide offsets are invisible to it. The 0.40 default "
                         "was hand-set on SOUTHERN Africa, where it gave a spread-skill "
                         "ratio of 1.038 (0.0 gave 0.564), against the post-hoc product's "
                         "residuals. MEASURED on the e1 Africa hindcast 2026-08-29, the "
                         "quantity it stands in for — the across-origin variance of the "
                         "region-wide mean residual, as a share of total variance — is "
                         "0.002-0.009, not 0.40; and Africa reads a spread-skill of 1.713 "
                         "at 0.40, i.e. over-dispersed, with ecoregion coverage pinned at "
                         "1.000. Do not carry this default between regions or models "
                         "without re-measuring it.")
    args = ap.parse_args(argv)

    if args.fit_space == "pit" and not args.qf_dir:
        raise SystemExit("--fit_space pit requires --qf_dir")
    ranges = tuple(float(x) for x in args.ranges.split(","))
    man = pd.read_csv(args.manifest)
    print(f"fit space: {args.fit_space}")
    fits = {}
    by_h_windows = {}
    print(f"{'tag':>12} {'nugget':>7} | " + " ".join(f"{b:>9}" for b in BAND_LABELS))
    for _, row in man.iterrows():
        h = int(row["horizon"])
        tag = f"w{int(row['base_year'])}_h{h}"
        e = fit_field(row, args)
        if np.isfinite(e).sum() < 10_000:
            continue
        fit = fit_spectral_mixture(e, ranges_px=ranges, kernel=args.kernel, nu=args.nu)
        # Recorded before the long component is appended: that component is not fitted, so
        # including it would report the same hand-set 0.40 in every window and hide the
        # spread the NNLS solution actually has.
        by_h_windows.setdefault(h, []).append(
            (tag, weight_shares(fit["ranges_px"], fit["weights"], fit["nugget"])))
        if args.long_weight > 0:
            fit = add_long_scale_component(fit, args.long_range, args.long_weight)
        obs = band_shares(e)
        fit["observed_band_shares"] = obs
        fit["band_labels"] = BAND_LABELS
        # Keep the best fit per horizon (windows agree closely; the longest record wins).
        fits.setdefault(str(h), fit)
        print(f"{tag:>12} {fit['nugget']:>7.3f} | " + " ".join(f"{v:>9.4f}" for v in obs))

    if args.report_windows:
        # `fits.setdefault` above keeps only the first window per horizon and discards the
        # rest, so the fitter has never been asked how much its solution moves between
        # forecast origins. With ~5 live weights and few independent residual maps, several
        # mixtures can reproduce one spectrum -- and a spread here that rivals the effect a
        # variant produces means the mixture is not identifiable at this sample size.
        print(f"\n=== fitted mixture across forecast origins (long component excluded) ===")
        print(f"{'':>12} {'nugget':>8} {'<4px':>8} {'4-50px':>8} {'>50px':>8}")
        for h in sorted(by_h_windows):
            wins = by_h_windows[h]
            for tag, sh in wins:
                print(f"{tag:>12} " + " ".join(f"{v:>8.4f}" for v in sh))
            if len(wins) > 1:
                a_ = np.array([sh for _, sh in wins])
                print(f"{'  spread h=' + str(h):>12} " +
                      " ".join(f"{v:>8.4f}" for v in a_.max(axis=0) - a_.min(axis=0)) +
                      f"   (n={len(wins)} origins, sd " +
                      " ".join(f"{v:.4f}" for v in a_.std(axis=0)) + ")")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"kernel": args.kernel, "nu": args.nu, "fit_space": args.fit_space,
                   "long_weight": args.long_weight, "long_range": args.long_range,
                   "by_horizon": fits}, f, indent=2)
    print(f"\n✓ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
