#!/usr/bin/env python3
"""Fit the marginal shape from this model's own hindcast residuals.

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

**Fitted per (horizon x distance-to-past-change band).** Pooled over pixels, one shape per
horizon is a mixture of two incompatible regimes. Fitted per band at h=20, ``q(0.999)`` runs
5.99 / 8.60 / 11.20 / 20.13 / 26.37 from the 0-1 px band outward, while ``q(0.9)`` barely
moves (0.65-1.00): **the band axis carries tail shape, not body shape.** The pooled fit sits
at 11.9, too fat for the near field and far too thin for the far field, which is why the
observation's rank among members is pinned at 0 near past change and at M beyond 30 px.

Because it is the tail that differs, ``--u_bound`` matters as much as the fit itself: it is
where the fitted quantile function gives way to a unit-slope continuation. Left at the
default 0.975 the far band's real tail is truncated and the per-band fit scores *worse* than
the two-piece normal it replaces (predicted P(delta>0.05) 0.00012 against 0.0024 observed,
versus the two-piece's 0.00029). See scripts/predict_change_rates.py, which sweeps it in
seconds rather than at 35 min per configuration.
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

from src.ensemble.copula import Z975, apply_shape, fit_residual_shape  # noqa: E402
from src.ensemble.validate import DIST_LABELS, distance_band  # noqa: E402

N_BANDS = len(DIST_LABELS)


def per_band(spec, name):
    """Parse a flag that is either one value for every band or one value per band."""
    vals = [float(v) for v in str(spec).split(",")]
    if len(vals) == 1:
        return vals * N_BANDS
    if len(vals) != N_BANDS:
        raise SystemExit(f"--{name} needs 1 or {N_BANDS} comma-separated values, got {len(vals)}")
    return vals


def report(tag, n, shape, rng):
    """What the shape will do, by pushing a standard normal through it.

    This is exactly what the copula does with z, so the numbers are the members' own.
    """
    z = rng.standard_normal(200_000)
    s = apply_shape(z, shape)
    ug = np.asarray(shape["u"])
    qv = np.asarray(shape["q"])

    def frac(a, k):
        return float((np.abs(a) > k * Z975).mean())

    print(f"{tag:>14} {n:>11,} | {frac(s, 0.25):>6.3f} {frac(s, 0.5):>6.3f} "
          f"{frac(s, 1.0):>6.3f} | {np.interp(0.9, ug, qv):>7.3f} "
          f"{np.interp(0.99, ug, qv):>7.3f} {np.interp(0.999, ug, qv):>8.3f} | "
          f"{shape['u_bound']:>6.4f} {float(np.abs(s).max()):>8.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_knots", type=int, default=513)
    ap.add_argument("--clip", default="8.0",
                    help="Clip the standardized residual before fitting; a handful of "
                         "near-degenerate widths would otherwise set the whole tail. One "
                         "value, or one per distance band.")
    ap.add_argument("--u_bound", default="0.975",
                    help="Where the fitted shape gives way to the unit-slope continuation "
                         "on the UPPER side. One value, or one per distance band. 0.975 "
                         "reproduces the two-piece normal's tail exactly.")
    ap.add_argument("--u_bound_lo", default=None,
                    help="Same, lower side; defaults to 1 - u_bound (symmetric). The two "
                         "tails of this residual fail in opposite directions — the upper "
                         "is too thin, the lower already 3-28x too hot — so raising them "
                         "together buys T8.1 and loses T6.2/T6.3/T6.5/T8.4. Hold this at "
                         "0.025 while raising --u_bound.")
    ap.add_argument("--by_band", type=lambda x: (str(x).lower() == "true"), nargs="?",
                    const=True, default=True,
                    help="Fit per (horizon x distance band). False reproduces the earlier "
                         "pooled-over-pixels fit.")
    ap.add_argument("--pooled_body", type=lambda x: (str(x).lower() == "true"), nargs="?",
                    const=True, default=False,
                    help="Give every band the pooled body but keep its own --u_bound. "
                         "Measured: the band axis pays on the tail bound, not on the shape "
                         "— per-band bodies score worse everywhere, while a bound raised "
                         "in the near field and held at 0.975 beyond 100 px is the only "
                         "way to gain the far-field tail without inventing change in "
                         "remote stable country (T8.2).")
    ap.add_argument("--min_count", type=int, default=10_000,
                    help="Bands with fewer finite residual pixels fall back to the pooled "
                         "shape for that horizon.")
    args = ap.parse_args(argv)

    from fit_field_spectra import standardized_residual

    clips = per_band(args.clip, "clip")
    bounds = per_band(args.u_bound, "u_bound")
    bounds_lo = (per_band(args.u_bound_lo, "u_bound_lo") if args.u_bound_lo
                 else [1.0 - b for b in bounds])
    # Every band's grid carries every band's bound, so the union grid the GPU gathers from
    # is the common grid and each bound stays an exact knot rather than an interpolation.
    extra = sorted(set(bounds) | set(bounds_lo) | {1.0 - b for b in bounds})

    man = pd.read_csv(args.manifest)
    pooled: dict[int, list] = {}
    banded: dict[tuple[int, int], list] = {}
    missing_dist = 0
    for _, row in man.iterrows():
        h = int(row["horizon"])
        path = str(row.get("path_dist_past_change", "") or "")
        band = None
        if args.by_band:
            if path and Path(path).exists():
                with rasterio.open(path) as s:
                    band = distance_band(s.read(1).astype(np.float64))
            else:
                missing_dist += 1

        ep = standardized_residual(row, clip=clips[0])
        ok = np.isfinite(ep)
        if int(ok.sum()) >= 10_000:
            pooled.setdefault(h, []).append(ep[ok])
        if band is None:
            continue
        for b in range(N_BANDS):
            # Recompute only when this band's clip differs — the clip is what decides how
            # much of the far band's real tail survives, so it has to be per band, but the
            # reads behind it are not free.
            e = ep if clips[b] == clips[0] else standardized_residual(row, clip=clips[b])
            m = np.isfinite(e) & (band == b)
            if m.any():
                banded.setdefault((h, b), []).append(e[m])

    if missing_dist:
        print(f"⚠ --by_band requested but {missing_dist} manifest rows have no "
              f"path_dist_past_change; those rows contribute to the pooled fit only")

    out = {"by_horizon": {}}
    print(f"{'horizon/band':>14} {'n':>11} | {'>0.25':>6} {'>0.5':>6} {'>1':>6} | "
          f"{'q(.9)':>7} {'q(.99)':>7} {'q(.999)':>8} | {'bound':>6} {'max|S|':>8}")
    rng = np.random.default_rng(0)
    for h in sorted(pooled):
        e = np.concatenate(pooled[h])
        shape = fit_residual_shape(e, n_knots=args.n_knots, u_bound=bounds[0],
                                   u_bound_lo=bounds_lo[0], extra_knots=extra)
        if shape is None:
            print(f"{h:>14} too few samples; skipping")
            continue
        entry = {"pooled": shape}
        report(f"h={h} pooled", e.size, shape, rng)

        if args.by_band:
            by_band, fallback = {}, []
            for b in range(N_BANDS):
                if args.pooled_body:
                    # Same body, this band's bound. The grid already carries every bound as
                    # an exact knot, so only the field the continuation reads changes.
                    by_band[str(b)] = {**shape, "u_bound": bounds[b],
                                       "u_bound_lo": bounds_lo[b]}
                    report(f"  {DIST_LABELS[b]}", e.size, by_band[str(b)], rng)
                    continue
                parts = banded.get((h, b))
                eb = np.concatenate(parts) if parts else np.empty(0)
                sb = fit_residual_shape(eb, n_knots=args.n_knots, min_count=args.min_count,
                                        u_bound=bounds[b], u_bound_lo=bounds_lo[b],
                                        extra_knots=extra)
                if sb is None:
                    fallback.append(f"{DIST_LABELS[b]} (n={eb.size:,})")
                    continue
                by_band[str(b)] = sb
                report(f"  {DIST_LABELS[b]}", eb.size, sb, rng)
            if by_band:
                entry["by_band"] = by_band
            if fallback:
                print(f"{'':>14} pooled fallback for: {', '.join(fallback)}")
        out["by_horizon"][str(h)] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    n_band = sum(len(v.get("by_band", {})) for v in out["by_horizon"].values())
    print(f"\n✓ {args.out} ({len(out['by_horizon'])} horizons, {n_band} band shapes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
