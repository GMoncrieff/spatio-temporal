#!/usr/bin/env python3
"""What rate of change will the members produce? Answered in closed form, before generating.

`scripts/member_distance_relationship.py` scores an ensemble by asking, per (year x
distance band), how often a member exceeds +0.05 and how that compares with the
observation. Getting one answer costs ~3 min of generation plus 18-32 min of validation,
which is too slow to design a marginal against.

It does not have to be sampled. The members differ only in the correlated normal field, and
the quantity being scored is a *marginal* one, so the member mean is available exactly:

    delta_m(pixel) = dhat + sigma * S(z),   z ~ N(0, 1),  sigma = w_up/Z975 or w_lo/Z975
    E_m P(delta > thr) = mean over pixels of  P( S(z) > Z975 * (thr - dhat) / w )
                       = mean over pixels of  1 - Phi( S^-1( Z975 * (thr - dhat) / w ) )

The correlation between pixels changes how much members *scatter* around that mean; it does
not move the mean. So this predicts the centre of the member distribution — which is the
thing the observation's rank is pinned against — in about thirty seconds.

Two uses:

1. **Sweeping the marginal.** ``--sweep_u_bound`` refits the per-band shape at each
   candidate tail bound and prints what each would produce. The optimum is interior: at the
   default 0.975 the 30-100 px band predicts 0.00012 against 0.0024 observed, and with the
   bound removed entirely it predicts 0.0065.

2. **Checking the generator.** This and the sampled ensemble reach the same number by
   completely different paths — a closed-form CDF here, 400 sampled int16 members there. If
   they disagree beyond Monte-Carlo error, the bug is in the generation path, and that is
   worth knowing before a scorecard is read as a finding.

Verified against the recorded k=5 numbers: observed 0.1767/0.0605/0.0234/0.0078/0.0024
across bands at h=20 and a two-piece far band of 0.000291 against the 0.0003 measured from
400 members.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.ensemble import copula as cop  # noqa: E402
from src.ensemble.validate import DIST_LABELS, distance_band  # noqa: E402

N_BANDS = len(DIST_LABELS)


def exceed_prob(t, shape, upper=True):
    """P(S(z) > t) for z ~ N(0,1), or P(S(z) < t) when ``upper`` is False.

    Inverting the shape rather than applying it is what makes this exact: the threshold is
    fixed and it is the *normal score that reaches it* we need. Applying the shape to a
    grid of z and counting would be a second Monte-Carlo error on top of the one this
    script exists to avoid.
    """
    t = np.asarray(t, dtype=np.float64)
    z = t if shape is None else cop.invert_shape(t, shape)
    if shape is not None:
        # invert_shape interpolates on the fitted grid; past its edge the quantile function
        # says nothing, so the probability there is zero rather than extrapolated.
        qv = np.asarray(shape["q"], dtype=np.float64)
        z_hi, z_lo, off_hi, off_lo = cop.shape_bounds(shape)
        beyond_hi = (t > qv[-1]) & (t > z_hi + off_hi)
        beyond_lo = (t < qv[0]) & (t < z_lo + off_lo)
        z = np.where(beyond_hi, np.inf, np.where(beyond_lo, -np.inf, z))
    return 1.0 - norm.cdf(z) if upper else norm.cdf(z)


def load_cells(manifest, base_year, all_windows, clip):
    """Per (horizon, band): the standardized residual and the two thresholds, in z units."""
    man = pd.read_csv(manifest)
    if not all_windows:
        man = man[man["base_year"].astype(int) == int(base_year)]
    if man.empty:
        raise SystemExit(f"no manifest rows for base_year={base_year}")

    cells: dict[tuple[int, int], dict[str, list]] = {}
    for _, row in man.iterrows():
        h = int(row["horizon"])
        arr = {}
        for key in ("res_native", "w_up", "w_lo", "dhat", "hm_t0"):
            with rasterio.open(row[f"path_{key}"]) as s:
                arr[key] = s.read(1).astype(np.float64)
        with rasterio.open(row["path_dist_past_change"]) as s:
            band = distance_band(s.read(1).astype(np.float64))

        res, wu, wl = arr["res_native"], arr["w_up"], arr["w_lo"]
        dhat, hm0 = arr["dhat"], arr["hm_t0"]
        ok = (np.isfinite(res) & np.isfinite(wu) & np.isfinite(wl) & np.isfinite(dhat)
              & np.isfinite(hm0) & (wu > 0) & (wl > 0))
        sigma = np.where(res >= 0, wu, wl) / cop.Z975
        with np.errstate(invalid="ignore", divide="ignore"):
            e = np.clip(res / sigma, -clip, clip)
        for b in range(N_BANDS):
            m = ok & (band == b)
            if not m.any():
                continue
            c = cells.setdefault((h, b), {k: [] for k in
                                          ("e", "obs", "hm0", "dhat", "wu", "wl")})
            c["e"].append(e[m])
            c["obs"].append(dhat[m] + res[m])
            c["hm0"].append(hm0[m])
            c["dhat"].append(dhat[m])
            c["wu"].append(wu[m])
            c["wl"].append(wl[m])
    return {k: {kk: np.concatenate(vv) for kk, vv in v.items()} for k, v in cells.items()}


def thresholds(cell, thr, width=(1.0, 1.0)):
    """The threshold in shaped-normal-score units, per pixel, with the HM clip applied.

    ``S(z)`` is scaled by the *upper* half-width when it is positive and the lower one when
    negative, so which half-width converts the threshold depends on which side of the
    central forecast the threshold falls.
    """
    gap = thr - cell["dhat"]
    w = np.where(gap >= 0, cell["wu"] * width[0], cell["wl"] * width[1])
    t = cop.Z975 * gap / np.maximum(w, 1e-12)
    # Members are clipped to HM in [0, 1], so a threshold outside that range is unreachable
    # — a real zero, not a small probability. BOTH ends matter: land already at HM < 0.01
    # cannot fall by 0.01, and the measured members are exactly 0.000000 there. Handling
    # only the upper clip made this script predict 0.00069 where the ensemble produces none.
    unreachable = (cell["hm0"] + thr >= 1.0) if thr >= 0 else (cell["hm0"] + thr <= 0.0)
    return np.where(unreachable, np.inf if thr >= 0 else -np.inf, t)


def evaluate(cells, shapes, hi_thr, lo_thr, widths=None):
    rows = []
    for (h, b), c in sorted(cells.items()):
        shape = shapes.get((h, b)) if shapes else None
        w = (1.0, 1.0) if widths is None else widths.get((h, b), (1.0, 1.0))
        t_hi = thresholds(c, hi_thr, w)
        t_lo = thresholds(c, lo_thr, w)
        rows.append({
            "horizon": h, "band": DIST_LABELS[b], "n": c["e"].size,
            "obs_hi": float((c["obs"] > hi_thr).mean()),
            "pred_hi": float(exceed_prob(t_hi, shape, upper=True).mean()),
            "obs_lo": float((c["obs"] < lo_thr).mean()),
            "pred_lo": float(exceed_prob(t_lo, shape, upper=False).mean()),
        })
    return pd.DataFrame(rows)


def narrow(cell, k_up, k_lo, clip):
    """The standardized residual and the two half-widths after narrowing the interval.

    Narrowing is not a post-hoc rescale of the members — it changes the *published* bounds,
    so the shape has to be refitted against the narrowed widths too. Fitting the shape to
    one width and generating against another is the mismatch that would silently break
    T5.2, and it is why this returns both together.
    """
    res = cell["obs"] - cell["dhat"]
    k = np.where(res >= 0, k_up, k_lo)
    sigma = np.where(res >= 0, cell["wu"] * k_up, cell["wl"] * k_lo) / cop.Z975
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.clip(res / np.maximum(sigma, 1e-12), -clip, clip)
    return e[np.isfinite(e)], k


def fit_shapes(cells, n_knots, bounds, min_count, bounds_lo=None, widths=None, clip=8.0,
               pooled_body=True):
    """Refit every (horizon x band) shape at the given per-band tail bounds and widths.

    ``pooled_body`` defaults to True because that is what the production fit ships
    (``fit_marginal_shape.py --pooled_body``), and an instrument that scores a different
    marginal from the one that will be generated mis-ranks configurations: on the same
    residuals a per-band body scores 0.347 where the pooled body scores 0.288, which is
    larger than the differences being compared.
    """
    bounds_lo = bounds_lo or [1.0 - b for b in bounds]
    extra = sorted(set(bounds) | set(bounds_lo) | {1.0 - b for b in bounds})
    per_cell = {}
    for (h, b), c in cells.items():
        e = c["e"]
        if widths is not None:
            e, _ = narrow(c, *widths.get((h, b), (1.0, 1.0)), clip)
        per_cell[(h, b)] = e

    out = {}
    for (h, b), e in per_cell.items():
        src = (np.concatenate([v for (hh, _), v in per_cell.items() if hh == h])
               if pooled_body else e)
        s = cop.fit_residual_shape(src, n_knots=n_knots, min_count=min_count,
                                   u_bound=bounds[b], u_bound_lo=bounds_lo[b],
                                   extra_knots=extra)
        if s is not None:
            out[(h, b)] = s
    return out


def unstretch_factors(cells, n_bands):
    """Per (horizon x band), the factor that makes the interval the residual's own 95% one.

    `fit_residual_shape` normalizes each side by that side's own 2.5/97.5 quantile, which is
    what pins T5.2 — so the fitted shape is the residual's distribution *stretched* to fill
    the published interval, and the stretch is exactly the interval's excess width. Setting
    the half-width to the residual's own 97.5 quantile makes the stretch 1.0, which also
    makes per-band coverage 0.95 by construction rather than the measured 0.982-0.990.

    Not centred on the median: the interval is centred on the central forecast, so it must
    cover the residual's bias as well as its spread. Centring here reads a class where the
    model over-predicts as needing a *narrower* lower bound than it does — see
    ``scripts/fit_width_factors.py``.
    """
    out = {}
    for (h, b), c in cells.items():
        e = c["e"]
        out[(h, b)] = (float(np.quantile(e, 0.975)) / cop.Z975,
                       abs(float(np.quantile(e, 0.025))) / cop.Z975)
    return out


def show(df, label):
    print(f"\n=== {label} ===")
    print(f"{'h':>3} {'band':>9} {'n':>10} | {'obs P(>hi)':>11} {'pred':>10} {'ratio':>7}"
          f" | {'obs P(<lo)':>11} {'pred':>10} {'ratio':>7}")
    for _, r in df.iterrows():
        rh = r["pred_hi"] / r["obs_hi"] if r["obs_hi"] > 0 else np.nan
        rl = r["pred_lo"] / r["obs_lo"] if r["obs_lo"] > 0 else np.nan
        print(f"{int(r['horizon']):>3} {r['band']:>9} {int(r['n']):>10,} | "
              f"{r['obs_hi']:>11.6f} {r['pred_hi']:>10.6f} {rh:>7.2f} | "
              f"{r['obs_lo']:>11.6f} {r['pred_lo']:>10.6f} {rl:>7.2f}")


def score(df):
    """One number per configuration: mean |log ratio| over cells, hi and lo together.

    A ratio, because the rates span four orders of magnitude across bands and an absolute
    error would score only the near field. Logged, because being 3x too hot and 3x too cold
    are the same size of mistake — and the failure this exists to fix is 2-7x hot in the
    near bands and 25x cold in the far one, at the same time.
    """
    v = []
    for col_o, col_p in (("obs_hi", "pred_hi"), ("obs_lo", "pred_lo")):
        o, p = df[col_o].to_numpy(), df[col_p].to_numpy()
        m = (o > 0) & np.isfinite(o) & np.isfinite(p)
        v.append(np.abs(np.log10(np.maximum(p[m], 1e-9) / o[m])))
    return float(np.concatenate(v).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--marginal_shape", default=None,
                    help="Shape artifact to score. Omitted, the two-piece normal is scored.")
    ap.add_argument("--base_year", type=int, default=2000)
    ap.add_argument("--all_windows", type=lambda x: (str(x).lower() == "true"), nargs="?",
                    const=True, default=False,
                    help="Pool every base year. Off by default so the rows line up with "
                         "member_distance_relationship.py, which scores one window.")
    ap.add_argument("--hi_thr", type=float, default=0.05)
    ap.add_argument("--lo_thr", type=float, default=-0.01)
    ap.add_argument("--clip", type=float, default=8.0)
    ap.add_argument("--n_knots", type=int, default=513)
    ap.add_argument("--min_count", type=int, default=10_000)
    ap.add_argument("--sweep_u_bound", default=None,
                    help="Comma-separated tail bounds to refit and compare, e.g. "
                         "0.975,0.99,0.999. Ignores --marginal_shape.")
    ap.add_argument("--sweep_width", default=None,
                    help="Comma-separated multipliers on the per-band unstretch factor, "
                         "e.g. 1.0,1.1,1.25. 1.0 sets each band's half-width to the "
                         "residual's own 97.5 quantile, which makes per-band coverage 0.95 "
                         "by construction; above 1.0 keeps some of the current excess. "
                         "The shape is refitted against the narrowed widths, because "
                         "narrowing changes the published bounds the shape normalizes to.")
    ap.add_argument("--width_u_bound", default="0.999,0.999,0.999,0.999,0.999,0.975",
                    help="Tail bound used with --sweep_width; defaults to the measured "
                         "configuration.")
    ap.add_argument("--width_u_bound_lo", type=float, default=0.025)
    ap.add_argument("--narrow_bands", default="0,1,2",
                    help="Which distance bands to narrow. Default 0,1,2 (out to 10 px) is "
                         "measured, not arbitrary: narrowing 10-30 px overshoots at every "
                         "horizon (h=10 1.67 -> 0.43 x observed), and narrowing 30-100 px "
                         "puts +0.05 past what the fitted shape can reach at all — the "
                         "threshold moves from 3.5 to 11.5 shape units while the fit stops "
                         "at the clip. Raising the clip recovers only part of it and still "
                         "scores worse than leaving that band alone.")
    ap.add_argument("--out_width_json", default=None,
                    help="Write the chosen per-(horizon x band) factors here, for "
                         "apply_recalibration.py --width_factors.")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args(argv)

    cells = load_cells(args.manifest, args.base_year, args.all_windows, args.clip)
    print(f"loaded {len(cells)} (horizon x band) cells, "
          f"{sum(c['e'].size for c in cells.values()):,} pixels")

    frames = []
    if args.sweep_width:
        import json as _json
        base = unstretch_factors(cells, N_BANDS)
        bounds = [float(x) for x in args.width_u_bound.split(",")]
        if len(bounds) == 1:
            bounds *= N_BANDS
        blo = [args.width_u_bound_lo] * N_BANDS
        print("\nper-band unstretch factors (multiplier 1.0 uses these directly):")
        for (h, b), (ku, kl) in sorted(base.items()):
            if h == max(k[0] for k in base):
                print(f"   h={h} {DIST_LABELS[b]:>9}  up {ku:.3f}  down {kl:.3f}")
        nb = {int(x) for x in args.narrow_bands.split(",") if x.strip() != ""}
        print(f"narrowing bands {sorted(nb)}; the rest keep their published width")
        # multiplier 0 is the incumbent: no narrowing at all, same tail bound
        for mult in [0.0] + [float(x) for x in args.sweep_width.split(",")]:
            widths = (None if mult == 0.0 else
                      {k: ((v[0] * mult, v[1] * mult) if k[1] in nb else (1.0, 1.0))
                       for k, v in base.items()})
            shapes = fit_shapes(cells, args.n_knots, bounds, args.min_count,
                                bounds_lo=blo, widths=widths, clip=args.clip)
            df = evaluate(cells, shapes, args.hi_thr, args.lo_thr, widths=widths)
            df["config"] = "no narrowing" if mult == 0.0 else f"unstretch x {mult}"
            frames.append(df)
            show(df, df["config"].iloc[0])
            print(f"\nmean |log10 ratio|: {score(df):.3f}")
            if args.out_width_json and mult == 1.0 and widths:
                blob = {"by_horizon": {}}
                for (h, b), (ku, kl) in sorted(widths.items()):
                    blob["by_horizon"].setdefault(str(h), {})[str(b)] = [ku, kl]
                Path(args.out_width_json).parent.mkdir(parents=True, exist_ok=True)
                _json.dump(blob, open(args.out_width_json, "w"), indent=1)
                print(f"✓ {args.out_width_json}")
        print("\n--- summary ---")
        for df in frames:
            print(f"{df['config'].iloc[0]:>22}  mean |log10 ratio| {score(df):.3f}")
    elif args.sweep_u_bound:
        base = evaluate(cells, None, args.hi_thr, args.lo_thr)
        base["config"] = "two-piece normal"
        frames.append(base)
        show(base, "two-piece normal")
        print(f"\nmean |log10 ratio|: {score(base):.3f}")
        for ub in [float(x) for x in args.sweep_u_bound.split(",")]:
            shapes = fit_shapes(cells, args.n_knots, [ub] * N_BANDS, args.min_count)
            df = evaluate(cells, shapes, args.hi_thr, args.lo_thr)
            df["config"] = f"band shape, u_bound={ub}"
            frames.append(df)
            show(df, f"band shape, u_bound={ub}")
            print(f"\nmean |log10 ratio|: {score(df):.3f}")
        print("\n--- summary ---")
        for df in frames:
            print(f"{df['config'].iloc[0]:>32}  mean |log10 ratio| {score(df):.3f}")
    else:
        shapes = (cop.read_shape_artifact(args.marginal_shape, N_BANDS)[0]
                  if args.marginal_shape else None)
        label = args.marginal_shape or "two-piece normal"
        df = evaluate(cells, shapes, args.hi_thr, args.lo_thr)
        df["config"] = label
        frames.append(df)
        show(df, label)
        print(f"\nmean |log10 ratio|: {score(df):.3f}")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames).to_csv(args.out_csv, index=False)
        print(f"✓ {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
