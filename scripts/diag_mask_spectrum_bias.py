#!/usr/bin/env python3
"""D1 — does the spectral fitter recover a known spectrum through this region's mask?

`fit_spectral_mixture` removes the field mean and then **zero-fills** every invalid pixel
before the FFT (`src/ensemble/fields.py:214-218`). There is no taper, no window, and no
mask-aware correction. Zero-filling multiplies the field by the mask, which convolves the
true spectrum with the mask's own — and on this project's residual rasters the mask is not a
gentle coastline. The Africa hindcast is a **holdout stitch**, so its finite set is the k=5
fold checkerboard on 512 px blocks intersected with the continent: 21.85% of the grid, with
strong structure of its own.

The variogram path in the same module refuses to do this ("NaN endpoints are dropped rather
than zero-filled: filling would deflate the sill", `fields.py:296-299`), so the hazard is
already understood here; it is the spectral path that fills.

This script does what the estimator has never been asked to do: synthesise a field with a
**known** mixture, fit it unmasked as a control, then fit it again through the real mask, and
report where the variance moved. It separates the two parts of the mask, because they are
different objects and only one of them is an artifact of how we validate:

    land         ocean and nodata -- present in any product raster
    checkerboard the held-out fold geometry -- present only because we score out of fold

The quantity that matters is the share of variance on ranges the T3.2 gate can see. T3.2 is
scored at a mean pair separation of 28.5 px, where the nugget and the 1.5 px component carry
no correlation and a 1500 px component is a constant. Only the middle of the basis does work.

    scripts/diag_mask_spectrum_bias.py --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.fields import fit_spectral_mixture, generate_correlated_field  # noqa: E402

# The production basis, from scripts/fit_field_spectra.py:84-85.
BASIS = (1.5, 2.5, 4.0, 6.0, 9.0, 14.0, 25.0, 50.0, 120.0, 300.0)
# The band that carries correlation at T3.2's scored separation.
RESOLVABLE = (4.0, 50.0)
# The baseline's own h=20 NNLS solution, un-rescaled by the appended long component's
# keep = 1 - 0.40 factor, so this is what the fitter actually returned on real residuals.
TRUTH_RANGES = (1.5, 4.0, 6.0, 25.0, 50.0)
TRUTH_WEIGHTS = (0.30220, 0.12276, 0.15301, 0.13485, 0.21842)
TRUTH_NUGGET = 0.06876


def shares(ranges, weights, nugget):
    """(nugget, short <4px, resolvable 4-50px, long >50px) shares of unit variance."""
    r = np.asarray(ranges, dtype=float)
    w = np.asarray(weights, dtype=float)
    lo, hi = RESOLVABLE
    return (float(nugget),
            float(w[r < lo].sum()),
            float(w[(r >= lo) & (r <= hi)].sum()),
            float(w[r > hi].sum()))


def fit(field, label):
    f = fit_spectral_mixture(field, ranges_px=BASIS, kernel="matern", nu=0.5)
    return label, shares(f["ranges_px"], f["weights"], f["nugget"]), f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--residual",
                    default="data/ensemble/exp/af_e1_hind/residuals/w2000_h20_res_native.tif",
                    help="Raster whose finite mask is the one under test")
    ap.add_argument("--continent", default="data/ensemble/region/africa/ecoregion.tif",
                    help="Land mask, used only to separate 'coastline' from 'held-out fold "
                         "geometry'. The fold mask is no use for this: it labels every pixel "
                         "of the grid 1-5, ocean included.")
    ap.add_argument("--reps", type=int, default=3,
                    help="Independent realisations; a single periodogram is noisy")
    ap.add_argument("--device", default="cuda:0", help="'cpu' or a torch device")
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args(argv)

    with rasterio.open(args.residual) as s:
        finite = np.isfinite(s.read(1))
        H, W = finite.shape
    with rasterio.open(args.continent) as s:
        continent = s.read(1) > 0
    # The residual's finite set is the land mask intersected with the folds this hindcast
    # held out, so the two rows below separate the part any product raster carries from the
    # part that exists only because we score out of fold.
    print(f"grid {H} x {W}")
    print(f"  land           {continent.mean():.4f} finite")
    print(f"  + checkerboard {finite.mean():.4f} finite   "
          f"({finite.sum() / max(continent.sum(), 1):.3f} of the continent)")

    t = shares(TRUTH_RANGES, TRUTH_WEIGHTS, TRUTH_NUGGET)
    print(f"\ntruth mixture: ranges {TRUTH_RANGES} weights {TRUTH_WEIGHTS} nugget {TRUTH_NUGGET}")

    rows = {}
    for rep in range(args.reps):
        z = generate_correlated_field(
            H, W, TRUTH_RANGES, TRUTH_WEIGHTS, TRUTH_NUGGET, wrap_lon=False,
            device=args.device, seed=args.seed + rep, kernel="matern", nu=0.5)
        z = np.asarray(z, dtype=np.float64)
        for label, mask in (("unmasked (control)", None),
                            ("land only", continent),
                            ("land + fold checkerboard", finite)):
            f = z if mask is None else np.where(mask, z, np.nan)
            rows.setdefault(label, []).append(fit(f, label)[1])
        del z

    hdr = f"{'':>26} {'nugget':>8} {'<4px':>8} {'4-50px':>8} {'>50px':>8}"
    print("\n" + hdr)
    print(f"{'TRUTH':>26} " + " ".join(f"{v:>8.4f}" for v in t))
    base = None
    for label, vals in rows.items():
        a = np.array(vals)
        m, sd = a.mean(axis=0), a.std(axis=0)
        print(f"{label:>26} " + " ".join(f"{v:>8.4f}" for v in m)
              + f"   (sd over {args.reps} reps: " + " ".join(f"{v:.4f}" for v in sd) + ")")
        if base is None:
            base = m

    full = np.array(rows["land + fold checkerboard"]).mean(axis=0)
    cont = np.array(rows["land only"]).mean(axis=0)
    print(f"\nmovement out of the resolvable 4-50 px band, against the unmasked control:")
    print(f"  land only (coastline)     {cont[2] - base[2]:+.4f}")
    print(f"  land + fold checkerboard  {full[2] - base[2]:+.4f}")
    print(f"  of which the checkerboard {full[2] - cont[2]:+.4f}")
    print(f"\nmovement into nugget + <4px:")
    print(f"  land + fold checkerboard  {(full[0] + full[1]) - (base[0] + base[1]):+.4f}")
    moved = abs(full[2] - base[2])
    print(f"\nDECISION (rule set before the measurement): >= 0.05 of total variance moved out "
          f"of 4-50 px\n  measured {moved:.4f} -> "
          f"{'mask-aware estimation TAKES a slot' if moved >= 0.05 else 'mask variant DOES NOT earn a slot'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
