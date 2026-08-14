#!/usr/bin/env python3
"""How much spatial structure does the residual have at the lags T3.2 scores?

T3.2 asks the copula ensemble to score at least 30% below an independent-pixel ensemble
with identical marginals. After the sampling fix (pairs inside the correlation range,
spread-weighted points) it measures 16.5% and still fails, and the natural next move is to
tune the field's short-scale content down until it passes.

That move is only correct if the shortfall is the generator's. This asks the prior
question of the data instead: an ensemble calibrated to the residual reproduces the
residual's correlation structure, so it can differ from an independent-pixel ensemble only
in the fraction of variance that is still *correlated* at the separation being scored.
Where the residual is mostly decorrelated, a faithful ensemble and an unfaithful one look
alike, and the target is unreachable without misrepresenting the data.

The measurement is the empirical variogram of the standardized residual, read as
``gamma(d) / sill`` — the share already decorrelated at separation ``d``. One minus that is
the entire budget T3.2 has to work with.

This deliberately measures rather than simulates. An earlier version of this script built a
synthetic ensemble from the fitted spectrum and scored T3.2 on it to estimate a ceiling; it
returned 1.6-2.9% against a production measurement of 16.5%, in two different geometries,
so it was not a valid model of the production scoring and its "ceiling" was discarded. The
variogram of the real residual needs no such stand-in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.ensemble.fields import empirical_variogram_from_field  # noqa: E402


def decorrelated_shares(field, lags, max_lag_px=200, n_bins=40, n_pairs=400_000,
                        sill_from_px=150.0):
    """``gamma(d)/sill`` at each requested lag, plus the sill it was normalized by."""
    c, g, n = empirical_variogram_from_field(np.nan_to_num(field, nan=0.0),
                                             max_lag_px=max_lag_px, n_bins=n_bins,
                                             n_pairs=n_pairs)
    ok = np.isfinite(g) & (n > 50)
    c, g = c[ok], g[ok]
    if c.size == 0:
        return {d: np.nan for d in lags}, np.nan
    far = c > sill_from_px
    sill = float(np.mean(g[far])) if far.any() else float(g[-1])
    out = {}
    for d in lags:
        out[d] = float(g[int(np.argmin(np.abs(c - d)))] / sill) if sill > 0 else np.nan
    return out, sill


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lags", default="5,10,25,50",
                    help="Separations (px) to report the decorrelated share at")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from fit_field_spectra import standardized_residual

    lags = [float(x) for x in args.lags.split(",")]
    man = pd.read_csv(args.manifest)
    rows = []
    header = f"{'tag':>11} " + " ".join(f"{'g/sill@' + str(int(d)) + 'px':>13}" for d in lags)
    print(header)
    for _, r in man.iterrows():
        e = standardized_residual(r)
        if np.isfinite(e).sum() < 10_000:
            continue
        shares, sill = decorrelated_shares(e, lags)
        tag = f"w{int(r['base_year'])}_h{int(r['horizon'])}"
        rows.append({"tag": tag, "horizon": int(r["horizon"]), "sill": sill,
                     **{f"decorrelated_at_{int(d)}px": shares[d] for d in lags}})
        print(f"{tag:>11} " + " ".join(f"{shares[d]:>13.3f}" for d in lags))

    df = pd.DataFrame(rows)
    print()
    for d in lags:
        col = f"decorrelated_at_{int(d)}px"
        m = float(df[col].mean())
        print(f"  at {int(d):>3} px: {m:.1%} of residual variance already decorrelated "
              f"-> only {1 - m:.1%} is structure T3.2 can reward")

    d_score = 25.0 if 25.0 in lags else lags[len(lags) // 2]
    budget = 1.0 - float(df[f"decorrelated_at_{int(d_score)}px"].mean())
    print()
    print(f"T3.2 is scored near {int(d_score)} px, where the structure budget is "
          f"{budget:.1%} of variance.")
    print("  A 30% reduction in variogram score against an independent-pixel null asks the")
    print("  ensemble to be better structured than the residual it is calibrated to. Passing")
    print("  it means removing short-scale power the data demonstrably has, which is exactly")
    print("  what T3.4 (spectrum within 1.5x) exists to prevent — the two targets conflict.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\n✓ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
