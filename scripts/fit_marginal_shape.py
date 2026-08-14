#!/usr/bin/env python3
"""Fit the per-horizon marginal shape from this model's own hindcast residuals.

The Phase 3 marginal is a median-spliced two-piece normal: it pins the 2.5/50/97.5
quantiles to the published triple and fills everything between them with Gaussian mass.
Measured on the k=5 residuals, the filling is badly wrong:

    P(|e| > 0.25)  P(|e| > 0.5)  P(|e| > 1)  P(|e| > 1.96)   kurtosis
      Gaussian   0.803          0.617         0.317       0.050            3
      h = 5      0.342          0.194         0.088       0.034        20366
      h = 20     0.492          0.254         0.082       0.015          928

At the 95% point the interval is about right — that is what T1.1 measures, mildly
over-covering. In the body the residual is three times more concentrated than the Gaussian
the marginal assumes, and the ensemble inherits that: members scatter 3.5x too much
moderate change over ground the observation leaves flat, which the scorecard reports four
separate times (T6.1 3.6x, T8.1 up to 6.9x, T6.5, T7.3).

This replaces the *shape* and nothing else. The output maps a normal score through the
residual's own standardized quantile function, normalized so u = 0.025/0.5/0.975 still land
exactly on lower/central/upper — so T5.1 and T5.2 hold by construction, and a residual that
really were Gaussian would produce the identity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.ensemble.copula import Z975, apply_shape, fit_residual_shape  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_knots", type=int, default=513)
    ap.add_argument("--clip", type=float, default=8.0,
                    help="Clip the standardized residual before fitting; a handful of "
                         "near-degenerate widths would otherwise set the whole tail")
    args = ap.parse_args(argv)

    from fit_field_spectra import standardized_residual

    man = pd.read_csv(args.manifest)
    per_h: dict[int, list] = {}
    for _, row in man.iterrows():
        e = standardized_residual(row, clip=args.clip)
        e = e[np.isfinite(e)]
        if e.size < 10_000:
            continue
        per_h.setdefault(int(row["horizon"]), []).append(e)

    out = {"by_horizon": {}}
    print(f"{'h':>3} {'n':>12} | {'P(|e|>0.25)':>12} {'P(|e|>0.5)':>11} {'P(|e|>1)':>9}"
          f" | {'gauss':>6} {'gauss':>6} {'gauss':>6}")
    for h in sorted(per_h):
        e = np.concatenate(per_h[h])
        shape = fit_residual_shape(e, n_knots=args.n_knots)
        if shape is None:
            print(f"{h:>3} too few samples; skipping")
            continue
        out["by_horizon"][str(h)] = shape

        # Report what the shape will do, by pushing a standard normal through it — this is
        # exactly what the copula does with z, so the numbers are the members' own.
        z = np.random.default_rng(0).standard_normal(400_000)
        shaped = apply_shape(z, shape)
        def frac(a, k):
            return float((np.abs(a) > k * Z975).mean())

        print(f"{h:>3} {e.size:>12,} | {frac(shaped, 0.25):>12.3f} "
              f"{frac(shaped, 0.5):>11.3f} {frac(shaped, 1.0):>9.3f} | "
              f"{frac(z, 0.25):>6.3f} {frac(z, 0.5):>6.3f} {frac(z, 1.0):>6.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"\n✓ {args.out} ({len(out['by_horizon'])} horizons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
