#!/usr/bin/env python3
"""Half-width multipliers that make the published interval the residual's own 95% interval.

The marginal shape normalizes each side by that side's own 2.5/97.5 quantile — which is what
pins T5.2 — so the fitted shape is the residual's distribution **stretched to fill the
published interval**. Any T5.2-preserving marginal therefore re-injects whatever width error
the bounds carry, and no change to the shape can reach it. Setting the half-width to the
residual's own 97.5 quantile makes the stretch exactly 1.0, and makes coverage 0.95 within
that class by construction rather than the measured 0.98-0.99.

**Two class axes, both measured to be necessary.**

*Distance to past change* is the axis the marginal failures organise along, but only out to
10 px. Beyond that, narrowing overshoots (10-30 px goes from 1.67x observed to 0.43x at
h=10) or makes the +0.05 threshold unreachable altogether (30-100 px moves it from 3.5 to
11.5 shape units while the fit stops at the clip). Those bands keep their published width.

*Predicted change* is the axis the **coverage** failures organise along, and leaving it out
is what a band-only factor gets wrong. Within the 0-1 px band at h=20 the factor runs 0.75
where Delta-hat is (0.01,0.05] and 0.95 where it is (0.05,0.15]; applying the band average
of 0.77 over-narrows the high-change class by 24%, which took T1.2 to 0.74-0.91 and T1.3 to
0.74/0.79 against gates of 0.95+/-0.03 and >=0.92. The high-change class is the one the
whole project exists to get right, so it cannot absorb the band average.

Cells thinner than ``--min_count`` shrink to their band's factor and then to 1.0, rather
than being fitted on noise — the same cell -> stratum -> horizon hierarchy Phase 1.5 uses.

Output feeds ``apply_recalibration.py --width_factors``. Rebuild the residuals against that
output before refitting the marginal shape: the shape normalizes to whatever half-width it
is fitted on, and fitting it to the old widths while generating from the new ones is a
silent T5.2 break.
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

from src.ensemble.copula import Z975  # noqa: E402
from src.ensemble.validate import (  # noqa: E402
    DHAT_BINS, DHAT_LABELS, DIST_LABELS, HM_BINS, HM_LABELS, distance_band,
)

N_BANDS = len(DIST_LABELS)
N_DHAT = len(DHAT_LABELS)
N_HM = len(HM_LABELS)


def unstretch(e, min_count):
    """(k_up, k_lo, n) — the residual's own 97.5/2.5 quantiles in half-width units.

    **Not centred on the median.** ``fit_residual_shape`` centres before taking quantiles
    because T5.1 requires the shape to map u = 0.5 to exactly zero; a *width* factor must
    not, because the published interval is centred on the central forecast, not on the
    residual's median, so it has to cover the residual's bias as well as its spread.

    The difference is not cosmetic. Where the model over-predicts change — Delta-hat in
    (0.05,0.15], the class the whole project exists to get right — the standardized residual
    has median -0.85, and centring reads its 2.5 percentile as -1.28 (k_lo = 0.65) instead
    of the true -2.13 (k_lo = 1.09). Acting on the centred number narrows a bound that was
    already slightly too tight, and coverage in that class fell from 0.95 to 0.73 with the
    misses all on the low side.
    """
    e = e[np.isfinite(e)]
    if e.size < min_count:
        return None
    k_up = float(np.quantile(e, 0.975)) / Z975
    k_lo = abs(float(np.quantile(e, 0.025))) / Z975
    if not (k_up > 0 and k_lo > 0):
        return None
    return k_up, k_lo, int(e.size)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bands", default="0,1,2",
                    help="Distance bands to narrow. Default 0,1,2 (out to 10 px); beyond "
                         "that narrowing measurably hurts — see the module docstring.")
    ap.add_argument("--min_count", type=int, default=2_000,
                    help="Cells thinner than this take the band factor outright.")
    ap.add_argument("--shrink_n", type=int, default=2_000,
                    help="Shrinkage scale toward the band factor: weight n/(n + shrink_n) "
                         "on the cell's own estimate. Kept small deliberately — the band "
                         "factor is *biased* for the high-change class (0.77 against its "
                         "own 0.95), not merely noisier, so shrinking hard toward it "
                         "reintroduces the error this axis exists to remove.")
    ap.add_argument("--clip", type=float, default=8.0)
    ap.add_argument("--floor", type=float, default=0.25,
                    help="Never narrow below this. A factor near zero would be fitted on a "
                         "class whose residual is degenerate, not on a real width error.")
    args = ap.parse_args(argv)

    bands = {int(x) for x in args.bands.split(",") if x.strip() != ""}
    man = pd.read_csv(args.manifest)

    # Three nested class levels, each shrunk toward its parent: the band is the root, the
    # predicted-change bin sits under it, and the HM level under that. Measured at h=20,
    # within a fixed (band x dhat) cell the factor still varies 1.4-81x across HM bins, so
    # the third level is carrying signal the first two do not.
    per_cell: dict[tuple[int, int, int], list] = {}
    per_band: dict[tuple[int, int], list] = {}
    per_hm: dict[tuple[int, int, int, int], list] = {}
    for _, row in man.iterrows():
        h = int(row["horizon"])
        with rasterio.open(row["path_res_native"]) as s:
            res = s.read(1).astype(np.float64)
        with rasterio.open(row["path_w_up"]) as s:
            wu = s.read(1).astype(np.float64)
        with rasterio.open(row["path_w_lo"]) as s:
            wl = s.read(1).astype(np.float64)
        with rasterio.open(row["path_dhat"]) as s:
            dhat = s.read(1).astype(np.float64)
        with rasterio.open(row["path_dist_past_change"]) as s:
            band = distance_band(s.read(1).astype(np.float64))

        ok = (np.isfinite(res) & np.isfinite(wu) & np.isfinite(wl) & np.isfinite(dhat)
              & (wu > 0) & (wl > 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            e = np.clip(res / (np.where(res >= 0, wu, wl) / Z975), -args.clip, args.clip)
        d_idx = np.digitize(dhat, DHAT_BINS[1:-1])
        with rasterio.open(row["path_hm_t0"]) as s:
            hm_idx = np.digitize(s.read(1).astype(np.float64), HM_BINS[1:-1])
        for b in sorted(bands):
            mb = ok & (band == b)
            if mb.any():
                per_band.setdefault((h, b), []).append(e[mb])
            for d in range(N_DHAT):
                m = mb & (d_idx == d)
                if not m.any():
                    continue
                per_cell.setdefault((h, b, d), []).append(e[m])
                for j in range(N_HM):
                    mj = m & (hm_idx == j)
                    if mj.any():
                        per_hm.setdefault((h, b, d, j), []).append(e[mj])

    def shrink(parts, parent):
        """Fit this cell and pull it toward its parent by n/(n + shrink_n).

        Returns ``(k_up, k_lo, n, fitted)``. ``fitted`` is False when the cell was too thin
        and simply inherited its parent — reporting an inherited value as if it were
        measured is how a table of 360 classes comes to look better resolved than it is.
        """
        fit = unstretch(np.concatenate(parts), args.min_count) if parts else None
        if fit is None:
            return (None if parent is None else (*parent[:3], False))
        k_up, k_lo, n = fit
        if parent is not None:
            lam = n / (n + args.shrink_n)
            k_up = lam * k_up + (1 - lam) * parent[0]
            k_lo = lam * k_lo + (1 - lam) * parent[1]
        return max(k_up, args.floor), max(k_lo, args.floor), n, True

    out: dict[str, dict] = {"by_horizon": {}}
    print(f"{'h':>3} {'band':>9} {'dhat':>15} {'HM_t0':>12} {'n':>9} {'k_up':>6} {'k_lo':>6}")
    n_leaf = n_fitted = 0
    for (h, b) in sorted(per_band):
        band_fit = unstretch(np.concatenate(per_band[(h, b)]), args.min_count)
        if band_fit is None:
            continue
        band_fit = (max(band_fit[0], args.floor), max(band_fit[1], args.floor),
                    band_fit[2], True)
        for d in range(N_DHAT):
            dfit = shrink(per_cell.get((h, b, d)), band_fit)
            for j in range(N_HM):
                got = shrink(per_hm.get((h, b, d, j)), dfit)
                if got is None:
                    continue
                k_up, k_lo, n, fitted = got
                out["by_horizon"].setdefault(str(h), {}).setdefault(str(b), {}) \
                   .setdefault(str(d), {})[str(j)] = [round(k_up, 4), round(k_lo, 4)]
                n_leaf += 1
                n_fitted += fitted
                if fitted:
                    print(f"{h:>3} {DIST_LABELS[b]:>9} {DHAT_LABELS[d]:>15} "
                          f"{HM_LABELS[j]:>12} {n:>9,} {k_up:>6.3f} {k_lo:>6.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n✓ {args.out} — {n_leaf} (band x predicted-change x HM) classes, of which "
          f"{n_fitted} fitted on their own pixels and {n_leaf - n_fitted} inherited from a "
          f"parent; bands {sorted(bands)} narrowed, the rest keep their published width")
    return 0


if __name__ == "__main__":
    sys.exit(main())
