#!/usr/bin/env python3
"""One stratified read of a model experiment, central field *and* intervals, in ~40 s.

The model phase runs many configurations, and the two things that decide each of them —
where the central error lives, and how wrong the interval widths are per class — both come
straight off the stitched prediction rasters. Neither needs an ensemble, a recalibration or
a residual-building stage, so neither should wait for one.

Three tables, all stratified, because pooled numbers hid every defect in the last phase:

  * **central** — RMSE, MAE, bias, skill against persistence and the slope of observed on
    predicted, per horizon and per (distance band | predicted change | HM level).
  * **bias** — the *median standardized residual* by predicted-change class, which is the
    named central-head defect: -0.6 to -0.8 half-widths where the model predicts most.
  * **width** — the unstretch factors k_up / k_lo per class: the multiplier that would make
    the published interval the residual's own 95% interval. **A model whose width heads are
    right needs k = 1 everywhere**, so `mean|ln k|` over classes is a single number for how
    much post-hoc width correction the model is still asking for. That is this phase's
    primary quantile metric.

Conventions are imported, never re-derived: `distance_band` (right=True), `DHAT_BINS`,
`HM_BINS` and `Z975` all come from the modules that own them, and `e` is standardized the
same way `fit_width_factors.py` standardizes it — uncentred, by the side's own half-width —
so the numbers here and there are the same numbers.

Usage:
    scripts/score_model_experiment.py --label e5_all_k5 \
        --stitched_dir data/ensemble/exp/e5_all_k5/stitched --folds 1,2
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

from src.strata import (  # noqa: E402
    DHAT_BINS, DHAT_LABELS, DIST_LABELS, HM_BINS, HM_LABELS, distance_band,
)

# The published bounds are the 2.5/97.5 percentiles, so half-width / Z975 is the scale of the
# normal that would place them there. One definition, here, since the copula module is gone.
Z975 = 1.959963985
from diagnose_central_field import (  # noqa: E402
    HORIZONS, MAX_OBSERVED_YEAR, WINDOWS, HM_DIR, _read, _read_like,
)

REGION_ROOT = Path("data/ensemble/region/southern_africa")
CLIP = 8.0


def rows_from_stitched(stitched_dir: Path, context_pattern: str) -> list[dict]:
    out = []
    for window in WINDOWS:
        base = window[-1]
        for h in HORIZONS:
            year = base + h
            if year > MAX_OBSERVED_YEAR:
                continue
            paths = {q: stitched_dir / f"w{base}_prediction_{year}_{q}.tif"
                     for q in ("central", "lower", "upper")}
            if not all(p.exists() for p in paths.values()):
                continue
            out.append({
                "window": "-".join(str(y) for y in window), "base_year": base,
                "target_year": year, "horizon": h,
                **{f"path_{q}": str(p) for q, p in paths.items()},
                "path_observed": str(HM_DIR / f"HM_{year}_AA_1000.tiff"),
                "path_baseline": str(HM_DIR / f"HM_{base}_AA_1000.tiff"),
                "path_context": context_pattern.format(year=base),
            })
    if not out:
        raise SystemExit(f"no stitched rasters under {stitched_dir}")
    return out


def load_row(row: dict, fold_sel: np.ndarray | None):
    with rasterio.open(row["path_central"]) as src:
        ref = {"transform": src.transform, "width": src.width, "height": src.height}
    central = _read(row["path_central"])
    lower = _read(row["path_lower"])
    upper = _read(row["path_upper"])
    observed = _read_like(row["path_observed"], ref)
    hm_t0 = _read_like(row["path_baseline"], ref)
    dist = _read_like(row["path_context"], ref, band=2)

    ok = (np.isfinite(central) & np.isfinite(observed) & np.isfinite(hm_t0)
          & np.isfinite(lower) & np.isfinite(upper) & np.isfinite(dist))
    # A degenerate half-width makes the standardized residual explode for a reason that is
    # about the denominator, not the model, so those pixels are dropped from the width
    # tables (they stay in the central tables, where no division happens).
    w_up = upper - central
    w_lo = central - lower
    ok_w = ok & (w_up > 0) & (w_lo > 0)
    if fold_sel is not None:
        ok = ok & fold_sel
        ok_w = ok_w & fold_sel

    resid = observed - central
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.clip(resid / (np.where(resid >= 0, w_up, w_lo) / Z975), -CLIP, CLIP)

    return dict(
        central=central[ok], observed=observed[ok], hm_t0=hm_t0[ok],
        lower=lower[ok], upper=upper[ok], dist=dist[ok], resid=resid[ok],
        e=e[ok_w], band_w=distance_band(dist[ok_w]),
        dhat_w=np.digitize((central - hm_t0)[ok_w], DHAT_BINS[1:-1]),
        hm_w=np.digitize(hm_t0[ok_w], HM_BINS[1:-1]),
        w_up=w_up[ok_w], w_lo=w_lo[ok_w],
        band=distance_band(dist[ok]),
        dhat_idx=np.digitize((central - hm_t0)[ok], DHAT_BINS[1:-1]),
        hm_idx=np.digitize(hm_t0[ok], HM_BINS[1:-1]),
    )


def central_stats(resid, obs_change, pred_change, covered):
    n = int(resid.size)
    if n == 0:
        return {"n": 0}
    mse = float(np.mean(resid ** 2))
    mse_p = float(np.mean(obs_change ** 2))
    out = {
        "n": n, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(resid))),
        "bias": float(np.mean(resid)),
        "skill": float(1.0 - mse / mse_p) if mse_p > 0 else np.nan,
        "coverage": float(np.mean(covered)),
        "sd_pred_change": float(np.std(pred_change)),
        "sd_obs_change": float(np.std(obs_change)),
    }
    if n >= 3 and np.std(pred_change) > 1e-12 and np.std(obs_change) > 1e-12:
        out["corr"] = float(np.corrcoef(pred_change, obs_change)[0, 1])
        out["slope"] = float(np.polyfit(pred_change, obs_change, 1)[0])
    else:
        out["corr"] = np.nan
        out["slope"] = np.nan
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stitched_dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out_dir", default=None, help="defaults to <stitched_dir>/../score")
    ap.add_argument("--folds", default=None,
                    help="comma-separated fold ids; screening runs may only be scored on "
                         "the folds they held out")
    ap.add_argument("--fold_mask", default=str(REGION_ROOT / "fold_mask.tif"))
    ap.add_argument("--context_pattern",
                    default="data/raw/hm_global/change_context_w{year}_1000.tif")
    ap.add_argument("--min_count", type=int, default=2000,
                    help="classes thinner than this are reported but excluded from the "
                         "|ln k| summary, which otherwise reads quantiles off noise")
    args = ap.parse_args(argv)

    stitched = Path(args.stitched_dir)
    out_dir = Path(args.out_dir) if args.out_dir else stitched.parent / "score"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = rows_from_stitched(stitched, args.context_pattern)

    with rasterio.open(rows[0]["path_central"]) as src:
        ref = {"transform": src.transform, "width": src.width, "height": src.height}
    fold_sel = None
    if args.folds:
        want = sorted({int(x) for x in args.folds.split(",")})
        fm = _read_like(args.fold_mask, ref)
        fold_sel = np.isin(np.nan_to_num(fm, nan=0).astype(int), want)
        print(f"folds {want}: {int(fold_sel.sum()):,} px")

    central_recs, bias_recs = [], []
    # Standardized residuals pooled across windows within a horizon, exactly as
    # fit_width_factors.py pools them, so the k values are comparable to its output.
    pool: dict[tuple, list] = {}

    for row in rows:
        d = load_row(row, fold_sel)
        meta = {"label": args.label, "window": row["window"], "horizon": row["horizon"]}
        obs_change = d["observed"] - d["hm_t0"]
        pred_change = d["central"] - d["hm_t0"]
        covered = ((d["observed"] >= d["lower"]) & (d["observed"] <= d["upper"])).astype(float)
        print(f"  {row['window']} h={row['horizon']:>2}  {d['resid'].size:>9,} px")

        central_recs.append({**meta, "stratum": "pooled", "bin": "all",
                             **central_stats(d["resid"], obs_change, pred_change, covered)})
        for name, idx, labels in (("distance", d["band"], DIST_LABELS),
                                  ("dhat", d["dhat_idx"], DHAT_LABELS),
                                  ("hm_t0", d["hm_idx"], HM_LABELS)):
            for b, lab in enumerate(labels):
                sel = idx == b
                if not sel.any():
                    continue
                central_recs.append({**meta, "stratum": name, "bin": lab,
                                     **central_stats(d["resid"][sel], obs_change[sel],
                                                     pred_change[sel], covered[sel]),
                                     "frac": float(sel.mean())})

        # The named defect: median standardized residual by predicted-change class.
        for b, lab in enumerate(DHAT_LABELS):
            sel = d["dhat_w"] == b
            if sel.sum() < args.min_count:
                continue
            bias_recs.append({**meta, "dhat": lab, "n": int(sel.sum()),
                              "median_e": float(np.median(d["e"][sel])),
                              "mean_e": float(np.mean(d["e"][sel]))})

        h = row["horizon"]
        for b in range(len(DIST_LABELS)):
            mb = d["band_w"] == b
            if not mb.any():
                continue
            pool.setdefault((h, b, -1, -1), []).append(d["e"][mb])
            for dd in range(len(DHAT_LABELS)):
                m = mb & (d["dhat_w"] == dd)
                if not m.any():
                    continue
                pool.setdefault((h, b, dd, -1), []).append(d["e"][m])
                for j in range(len(HM_LABELS)):
                    mj = m & (d["hm_w"] == j)
                    if mj.any():
                        pool.setdefault((h, b, dd, j), []).append(d["e"][mj])

    width_recs = []
    for (h, b, dd, j), parts in sorted(pool.items()):
        e = np.concatenate(parts)
        if e.size < 30:
            continue
        k_up = float(np.quantile(e, 0.975)) / Z975
        k_lo = abs(float(np.quantile(e, 0.025))) / Z975
        width_recs.append({
            "label": args.label, "horizon": h, "band": DIST_LABELS[b],
            "dhat": "all" if dd < 0 else DHAT_LABELS[dd],
            "hm_t0": "all" if j < 0 else HM_LABELS[j],
            "n": int(e.size), "k_up": k_up, "k_lo": k_lo,
            "median_e": float(np.median(e)),
        })

    cdf = pd.DataFrame(central_recs)
    bdf = pd.DataFrame(bias_recs)
    wdf = pd.DataFrame(width_recs)

    # Pixel-weighted roll-up per horizon, the headline row.
    pooled = cdf[cdf["stratum"] == "pooled"]
    by_h = []
    for h, g in pooled.groupby("horizon"):
        w = g["n"].to_numpy(float)
        by_h.append({
            "label": args.label, "horizon": int(h), "n": int(w.sum()),
            "rmse": float(np.sqrt(np.sum(w * g["rmse"] ** 2) / w.sum())),
            "mae": float(np.sum(w * g["mae"]) / w.sum()),
            "bias": float(np.sum(w * g["bias"]) / w.sum()),
            "skill": float(np.sum(w * g["skill"]) / w.sum()),
            "corr": float(np.sum(w * g["corr"]) / w.sum()),
            "slope": float(np.sum(w * g["slope"]) / w.sum()),
            "coverage": float(np.sum(w * g["coverage"]) / w.sum()),
        })
    hdf = pd.DataFrame(by_h)

    # How much post-hoc width correction is the model still asking for? Leaf classes only
    # (band x dhat x HM), thick enough to read a 2.5% quantile from.
    #
    # A class whose residual sits entirely on one side gives a non-positive k on the other,
    # and log() of that is not a width error — it is a class with no upper (or lower) tail
    # to measure. fit_width_factors.unstretch drops those; so does this, and it says how
    # many, because silently averaging over a NaN is how a summary stops meaning anything.
    def ln_k(frame):
        vals = np.concatenate([frame["k_up"].to_numpy(), frame["k_lo"].to_numpy()]) \
            if len(frame) else np.array([])
        good = vals[np.isfinite(vals) & (vals > 0)]
        return np.log(good), int(vals.size - good.size)

    leaf = wdf[(wdf["dhat"] != "all") & (wdf["hm_t0"] != "all") & (wdf["n"] >= args.min_count)]
    band = wdf[(wdf["dhat"] == "all") & (wdf["n"] >= args.min_count)]
    lk, n_drop_leaf = ln_k(leaf)
    lb, _ = ln_k(band)
    if lk.size == 0:
        lk = np.array([np.nan])
    if lb.size == 0:
        lb = np.array([np.nan])
    summary = {
        "label": args.label,
        "stitched_dir": str(stitched),
        "folds": args.folds,
        "n_leaf_classes": int(len(leaf)),
        "n_leaf_sides_dropped": n_drop_leaf,
        "width_mean_abs_ln_k_leaf": float(np.mean(np.abs(lk))),
        "width_max_abs_ln_k_leaf": float(np.max(np.abs(lk))),
        "width_frac_within_20pct_leaf": float(np.mean(np.abs(lk) < np.log(1.2))),
        "width_mean_abs_ln_k_band": float(np.mean(np.abs(lb))),
        "central_rmse": {int(r.horizon): r.rmse for r in hdf.itertuples()},
        "central_skill": {int(r.horizon): r.skill for r in hdf.itertuples()},
        "coverage": {int(r.horizon): r.coverage for r in hdf.itertuples()},
    }

    cdf.to_csv(out_dir / f"central_{args.label}.csv", index=False)
    bdf.to_csv(out_dir / f"bias_{args.label}.csv", index=False)
    wdf.to_csv(out_dir / f"width_{args.label}.csv", index=False)
    hdf.to_csv(out_dir / f"by_horizon_{args.label}.csv", index=False)
    (out_dir / f"summary_{args.label}.json").write_text(json.dumps(summary, indent=2))

    fmt = lambda v: f"{v:.5f}"
    print(f"\n=== {args.label} · pooled by horizon ===")
    print(hdf.to_string(index=False, float_format=fmt))

    print(f"\n=== central RMSE by distance band ===")
    dist = cdf[cdf["stratum"] == "distance"]
    print(dist.pivot_table(index="bin", columns="horizon", values="rmse",
                           aggfunc=lambda s: np.sqrt(np.mean(s ** 2)))
          .reindex(DIST_LABELS).to_string(float_format=fmt))

    print(f"\n=== median standardized residual by predicted-change class ===")
    if len(bdf):
        print(bdf.pivot_table(index="dhat", columns="horizon", values="median_e")
              .reindex([l for l in DHAT_LABELS if l in set(bdf["dhat"])])
              .to_string(float_format=lambda v: f"{v:+.3f}"))

    print(f"\n=== width factor k_up by distance band (1.0 = the model needs no correction) ===")
    print(band.pivot_table(index="band", columns="horizon", values="k_up")
          .reindex([b for b in DIST_LABELS if b in set(band["band"])])
          .to_string(float_format=fmt))

    print(f"\n=== coverage by distance band (target 0.95) ===")
    print(dist.pivot_table(index="bin", columns="horizon", values="coverage")
          .reindex(DIST_LABELS).to_string(float_format=fmt))

    print(f"\nWIDTH SUMMARY  mean|ln k| leaf {summary['width_mean_abs_ln_k_leaf']:.4f} "
          f"| band {summary['width_mean_abs_ln_k_band']:.4f} "
          f"| within 20%: {summary['width_frac_within_20pct_leaf']:.3f} "
          f"over {summary['n_leaf_classes']} classes "
          f"({n_drop_leaf} one-sided sides dropped)")
    print(f"✓ wrote {out_dir}/summary_{args.label}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
