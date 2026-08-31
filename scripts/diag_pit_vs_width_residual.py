#!/usr/bin/env python3
"""D6 — is the spectrum fitted in the space the copula actually samples in?

The copula's latent is a standard normal ``z``, and the review's §5.1 prescribes fitting the
dependence model to the out-of-fold **PIT normal score** of the observation,

    z = Phi^-1( F_{h,p}(y_{h,p}) ),

using the exact learned CDF. This pipeline does not do that. `fit_field_spectra.py:64-75`
fits a *width-standardised* residual instead,

    e = (y - central) / sigma,   sigma = max(w_up, w_lo) / 1.959964,

which reads three points off the marginal -- Q(0.025), Q(0.5), Q(0.975) -- and assumes the
rest is symmetric and normal. The model emits 64 levels; 61 of them are unused, and the
symmetry assumption is exactly wrong for this field: HM is bounded below, 40% of Africa sits
in the [0, 0.01) stratum, and 5.4% of member-pixels land on the atom at HM = 0.

Where the two differ most is where sigma is smallest -- the quiet pixels -- and those set the
nugget and the shortest-range weight, which together carry 0.37 of the latent variance and
contribute nothing at the separation T3.2 is scored at. So this is worth a number before it
is worth a variant.

Both fields are fitted with the same estimator, on the same basis, over the same pixels.

    scripts/diag_pit_vs_width_residual.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.ensemble.fields import fit_spectral_mixture  # noqa: E402
from src.ensemble.residuals import pit_normal_field  # noqa: E402
from fit_field_spectra import BASIS_DEFAULT, standardized_residual, weight_shares  # noqa: E402

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/ensemble/exp/af_e1_hind/residuals/manifest.csv")
    ap.add_argument("--stitched", default="data/ensemble/exp/af_e1_hind/stitched")
    ap.add_argument("--horizons", default="5,20")
    args = ap.parse_args(argv)

    man = pd.read_csv(args.manifest)
    print(f"{'':>34} {'nugget':>8} {'<4px':>8} {'4-50px':>8} {'>50px':>8}")
    for h in [int(x) for x in args.horizons.split(",")]:
        row = man[man.horizon == h].iloc[0]
        by, ty = int(row["base_year"]), int(row["target_year"])
        qf = f"{args.stitched}/w{by}_prediction_{ty}_qf.tif"

        e = standardized_residual(row)
        z = pit_normal_field(qf, row["path_observed"])
        # Score both on exactly the pixels both have, so the comparison is the space and
        # nothing else.
        both = np.isfinite(e) & np.isfinite(z)
        e = np.where(both, e, np.nan)
        z = np.where(both, z, np.nan).astype(np.float64)

        print(f"\n  h={h} (w{by}), {int(both.sum())} px in both")
        print(f"    width-standardised e: mean {np.nanmean(e):+.4f} var {np.nanvar(e):.4f}")
        print(f"    PIT normal score z  : mean {np.nanmean(z):+.4f} var {np.nanvar(z):.4f}"
              f"   <- must be ~N(0,1) if the marginal is calibrated")
        for label, f in (("(y-central)/sigma  [production]", e),
                         ("Phi^-1(F_qf(y))    [review 5.1]", z)):
            fit = fit_spectral_mixture(f, ranges_px=BASIS_DEFAULT, kernel="matern", nu=0.5)
            sh = weight_shares(fit["ranges_px"], fit["weights"], fit["nugget"])
            print(f"{label:>34} " + " ".join(f"{v:>8.4f}" for v in sh))
    print("\nDECISION (rule set before the measurement): the two spaces move the resolvable\n"
          "4-50 px share by more than the estimator's own realisation noise (0.044, from D1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
