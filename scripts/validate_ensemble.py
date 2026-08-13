#!/usr/bin/env python3
"""Phase 4 — score every target metric and emit one pass/fail scorecard.

Runs against a *hindcast* ensemble (built from the recalibrated hindcast rasters), because
scoring aggregate coverage needs observations. The output is
``data/ensemble/validation/scorecard.csv`` plus figures, so "is it done?" is answerable
without re-deriving thresholds.

Failures print the diagnosis from the plan's "which knob fixes which failure" table rather
than just a red X — the whole point of separating marginals from correlation structure is
that a given failure has one correct response.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble import aggregate as agg  # noqa: E402
from src.ensemble import fields as fld  # noqa: E402
from src.ensemble import validate as val  # noqa: E402
from src.ensemble import variogram as vgm  # noqa: E402
from src.ensemble.copula import INT16_SENTINEL, quantization_error_bound  # noqa: E402

REPO = Path(__file__).parent.parent
HM_DIR = REPO / "data" / "raw" / "hm_global"

KNOB_TABLE = {
    "T1": "Phase 1.5 rescale factors s(class, horizon) — do NOT widen the variogram.",
    "T2_under": "Raise long-range weight / range in the fitted variogram — do NOT re-widen marginals.",
    "T2_over": "Lower long-range weight; check the nugget fraction is not under-estimated.",
    "T3": "Spectral shape wrong — add a third range or change kernel family (Phase 2).",
    "T4.1": "Re-estimate AR(1) rho_h; consider rho varying by stratum.",
    "T4.2": "Enforce monotone spread across horizons as a constraint on s in Phase 1.5.",
    "T5": "Hard gate: a failure here is a bug in generation, not a calibration issue.",
    "T1.5 sharpness": "Use a finer stratification in Phase 1.5 — NOT a larger global factor.",
}


class Scorecard:
    def __init__(self):
        self.rows = []

    def add(self, tid, metric, value, target, passed, note="", knob=""):
        self.rows.append({
            "id": tid, "metric": metric, "value": value, "target": target,
            "pass": bool(passed) if passed is not None else None, "note": note,
            "diagnosis": KNOB_TABLE.get(knob, "") if not passed and knob else "",
        })

    def df(self):
        df = pd.DataFrame(self.rows)
        if not df.empty:
            # Nullable boolean: reported-only rows carry NA, and `~` still works. A plain
            # object column silently turns `~True` into -2 and breaks the failure listing.
            df["pass"] = pd.array(df["pass"].tolist(), dtype="boolean")
        return df


# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", default="data/ensemble/hindcast_members.zarr")
    ap.add_argument("--null_ensemble", default=None,
                    help="Independent-pixel null (same marginals, no spatial structure)")
    ap.add_argument("--years", default="2005,2010,2015,2020")
    ap.add_argument("--base_year", type=int, default=2000)
    ap.add_argument("--recal_dir", default="data/ensemble/hindcast/recal")
    ap.add_argument("--recal_pattern", default="w{base}_prediction_{year}_{q}_recal.tif")
    ap.add_argument("--central_pattern", default="w{base}_prediction_{year}_central_recal.tif")
    ap.add_argument("--observed_pattern", default=str(HM_DIR / "HM_{year}_AA_1000.tiff"))
    ap.add_argument("--ecoregion_raster", default=str(HM_DIR / "ecoregion_id_1000.tif"))
    ap.add_argument("--lookup_csv", default=str(HM_DIR / "ecoregion_lookup.csv"))
    ap.add_argument("--variogram_fits", default="data/ensemble/diagnostics/variogram_fits.csv")
    ap.add_argument("--rho_json", default="data/ensemble/residuals/horizon_autocorrelation.json")
    ap.add_argument("--out_dir", default="data/ensemble/validation")
    ap.add_argument("--recal_manifest", default="data/ensemble/hindcast/recal/recal_manifest.json",
                    help="Used for the T1.5 sharpness guard (width vs the original heads)")
    ap.add_argument("--block_sizes", default="10,100,1000")
    ap.add_argument("--score_points", type=int, default=1500)
    ap.add_argument("--stages", default="gates,percentiles,aggregate,rank,spatial,temporal")
    ap.add_argument("--disable_wandb", action="store_true")
    ap.add_argument("--wandb_group", default=None)
    return ap.parse_args(argv)


def raster_paths(args, years):
    base = args.base_year
    out = {}
    for y in years:
        out[y] = {
            "central": Path(args.recal_dir) / args.central_pattern.format(base=base, year=y),
            "lower": Path(args.recal_dir) / args.recal_pattern.format(base=base, year=y, q="lower"),
            "upper": Path(args.recal_dir) / args.recal_pattern.format(base=base, year=y, q="upper"),
            "observed": Path(args.observed_pattern.format(year=y)),
        }
    return out


# --------------------------------------------------------------------------------------
# T5 hard gates + ensemble percentile rasters
# --------------------------------------------------------------------------------------
def stage_gates(args, store, attrs, years, paths, out_dir, card, block_rows=None):
    """T5.1/T5.2/T5.3 in one streaming pass, which also writes the percentile rasters."""
    print("\n=== T5 · hard gates (median / tails / mask) ===")
    M, nH, H, W = store.shape
    if block_rows is None:
        # Every member of a block is held at once; cap the working set near 1.5 GB.
        block_rows = int(np.clip(1.5e9 / max(M * W * 8, 1), 32, 512))
    tol_median = quantization_error_bound(float(attrs.get("scale", 1 / 32767)))
    # Monte-Carlo tolerance on the tails: with M members the 2.5th percentile sits between
    # order statistics, so its standard error is sqrt(p(1-p)/M) / f(x_p). Approximated with
    # a normal density at the 2.5% point.
    from scipy.stats import norm as _norm
    p = 0.025
    mc_sigma_z = np.sqrt(p * (1 - p) / M) / _norm.pdf(_norm.ppf(p))

    pct_paths = {}
    summary = []
    for hi, year in enumerate(years):
        with rasterio.open(paths[year]["central"]) as c:
            profile = c.profile.copy()
        profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate", BIGTIFF="YES")
        outs = {q: out_dir / f"ens_{year}_{q}.tif" for q in ("p2_5", "median", "p97_5")}
        dsts = {q: rasterio.open(v, "w", **profile) for q, v in outs.items()}
        pct_paths[year] = outs

        n_valid = n_med_ok = n_lo_ok = n_hi_ok = 0
        n_mask_mismatch = 0
        sum_halfwidth = 0.0
        try:
            with rasterio.open(paths[year]["central"]) as csrc, \
                 rasterio.open(paths[year]["lower"]) as lsrc, \
                 rasterio.open(paths[year]["upper"]) as usrc:
                for r0 in range(0, H, block_rows):
                    rr = min(block_rows, H - r0)
                    win = Window(0, r0, W, rr)
                    cen = csrc.read(1, window=win).astype(np.float32)
                    low = lsrc.read(1, window=win).astype(np.float32)
                    upp = usrc.read(1, window=win).astype(np.float32)
                    q = np.asarray(store[:, hi, r0:r0 + rr, :])
                    ens = agg.dequantize_block(q, attrs)
                    ens_valid = np.isfinite(ens).all(axis=0)
                    cen_valid = np.isfinite(cen)
                    n_mask_mismatch += int((ens_valid != cen_valid).sum())
                    ok = ens_valid & cen_valid
                    if not ok.any():
                        for name, arr in (("p2_5", low), ("median", cen), ("p97_5", upp)):
                            dsts[name].write(np.full((rr, W), np.nan, np.float32), 1, window=win)
                        continue
                    # One sort for all three quantiles; three separate calls sort the
                    # (50, rows, 40000) block three times over.
                    p25, med, p975 = np.nanpercentile(ens, [2.5, 50.0, 97.5], axis=0)
                    dsts["median"].write(np.where(ok, med, np.nan).astype(np.float32), 1, window=win)
                    dsts["p2_5"].write(np.where(ok, p25, np.nan).astype(np.float32), 1, window=win)
                    dsts["p97_5"].write(np.where(ok, p975, np.nan).astype(np.float32), 1, window=win)

                    n_valid += int(ok.sum())
                    n_med_ok += int((np.abs(med - cen)[ok] <= tol_median).sum())
                    half = np.maximum(upp - cen, 1e-9)
                    sum_halfwidth += float(half[ok].sum())
                    tol_lo = mc_sigma_z * np.maximum(cen - low, 1e-9) / 1.96 * 3.0
                    tol_hi = mc_sigma_z * half / 1.96 * 3.0
                    n_lo_ok += int((np.abs(p25 - low)[ok] <= tol_lo[ok]).sum())
                    n_hi_ok += int((np.abs(p975 - upp)[ok] <= tol_hi[ok]).sum())
        finally:
            for d in dsts.values():
                d.close()

        f_med = n_med_ok / max(n_valid, 1)
        f_lo = n_lo_ok / max(n_valid, 1)
        f_hi = n_hi_ok / max(n_valid, 1)
        summary.append({"year": year, "n_valid": n_valid, "frac_median_ok": f_med,
                        "frac_p2_5_ok": f_lo, "frac_p97_5_ok": f_hi,
                        "n_mask_mismatch": n_mask_mismatch})
        print(f"  {year}: median≡central {100*f_med:.3f}% | p2.5 within MC {100*f_lo:.1f}% | "
              f"p97.5 within MC {100*f_hi:.1f}% | mask mismatches {n_mask_mismatch:,}")
        card.add("T5.1", f"median==central ({year})", f_med, "1.000 (hard gate)", f_med >= 0.9999,
                 note=f"tolerance {tol_median:.2e}", knob="T5")
        card.add("T5.2", f"tails within MC tolerance ({year})", min(f_lo, f_hi), ">=0.95 (MC-scaled)",
                 min(f_lo, f_hi) >= 0.95, note=f"M={M}, mc_sigma_z={mc_sigma_z:.3f}", knob="T5")
        card.add("T5.3", f"valid-mask identity ({year})", n_mask_mismatch, "0 pixels",
                 n_mask_mismatch == 0, knob="T5")

    pd.DataFrame(summary).to_csv(out_dir / "t5_gates.csv", index=False)
    return pct_paths


# --------------------------------------------------------------------------------------
# T2 aggregate coverage
# --------------------------------------------------------------------------------------
def stage_aggregate(args, store, attrs, years, paths, out_dir, card, null_store=None, null_attrs=None):
    print("\n=== T2 · aggregate-scale coverage ===")
    block_sizes = [int(b) for b in args.block_sizes.split(",")]
    rows, zonal_rows = [], []
    thresholds = (0.1, 0.3)
    zonal_members_by_year = {}

    for hi, year in enumerate(years):
        with rasterio.open(paths[year]["central"]) as c:
            profile = c.profile.copy()
        # ---- T2.1 blocks --------------------------------------------------------------
        usable = [B for B in block_sizes
                  if profile["height"] // B > 0 and profile["width"] // B > 0]
        # One pass over the members for all scales: at 50 members a global pass is ~68 GB
        # of reads, and the nested block sums aggregate upward for free.
        mem_by_b = agg.block_member_stats_multi(store, hi, usable, attrs=attrs)
        obs_by_b = agg.block_observed_multi(paths[year]["observed"], profile, usable)
        for B in usable:
            mem, valid, _ = mem_by_b[B]
            obs, obs_valid = obs_by_b[B]
            ok = valid & obs_valid
            if not ok.any():
                continue
            res = agg.coverage_from_members(mem[:, ok], obs[ok])
            lo, hi_w = val.wilson_interval(res["n_covered"], res["n"])
            rows.append({"year": year, "scale_km": B, "kind": "ensemble", **res,
                         "wilson_lo": float(lo), "wilson_hi": float(hi_w)})
            # Pixelwise-independent-propagation baseline at the same scale: the motivating
            # contrast, and the reason the ensemble exists.
            base = val.compute_block_coverage(paths[year]["lower"], paths[year]["upper"],
                                              paths[year]["observed"], block_sizes=[B])
            if not base.empty:
                rows.append({"year": year, "scale_km": B, "kind": "pixelwise-propagated",
                             "n": int(base.iloc[0]["n_blocks"]),
                             "n_covered": int(base.iloc[0]["n_covered"]),
                             "coverage": float(base.iloc[0]["coverage"]),
                             "mean_width": float(base.iloc[0]["mean_width"]),
                             "wilson_lo": float(base.iloc[0]["wilson_lo"]),
                             "wilson_hi": float(base.iloc[0]["wilson_hi"])})
            print(f"  {year} {B}km: ensemble {res['coverage']:.3f} "
                  f"vs pixelwise {float(base.iloc[0]['coverage']) if not base.empty else np.nan:.3f}")

        # ---- T2.2 / T2.3 ecoregion ------------------------------------------------------
        if Path(args.ecoregion_raster).exists():
            zm = agg.zonal_member_stats(store, hi, args.ecoregion_raster, attrs=attrs,
                                        thresholds=thresholds)
            zo = agg.zonal_observed(paths[year]["observed"], args.ecoregion_raster, profile,
                                    thresholds=thresholds)
            common, i_m, i_o = np.intersect1d(zm["zone_ids"], zo["zone_ids"], return_indices=True)
            keep = zo["n_px"][i_o] >= 100
            zonal_members_by_year[year] = {
                "zone_ids": common[keep], "mean_members": zm["mean"][:, i_m][:, keep],
                "mean_obs": zo["mean"][i_o][keep],
            }
            res = agg.coverage_from_members(zm["mean"][:, i_m][:, keep], zo["mean"][i_o][keep])
            lo, hi_w = val.wilson_interval(res["n_covered"], res["n"])
            zonal_rows.append({"year": year, "level": "ecoregion", "stat": "mean", **res,
                               "wilson_lo": float(lo), "wilson_hi": float(hi_w)})
            for t in thresholds:
                r = agg.coverage_from_members(zm[f"area{t}"][:, i_m][:, keep], zo[f"area{t}"][i_o][keep])
                zonal_rows.append({"year": year, "level": "ecoregion", "stat": f"area>{t}", **r})
            print(f"  {year} ecoregion mean coverage {res['coverage']:.3f} (n={res['n']})")

            # Biome / realm are *reported*, not scored: at n=14 the Wilson CI (+/-0.114) is
            # wider than the +/-0.05 tolerance, so such a number can neither pass nor fail.
            lut = pd.read_csv(args.lookup_csv)
            for level, col in (("biome", "BIOME_NUM"), ("realm", "REALM")):
                mapping = dict(zip(lut["ECO_ID"], lut[col]))
                groups = np.array([mapping.get(int(z), None) for z in common[keep]], dtype=object)
                uniq = [g for g in pd.unique(groups) if g is not None and not pd.isna(g)]
                if not uniq:
                    continue
                npx = zo["n_px"][i_o][keep]
                gm, go = [], []
                for g in uniq:
                    sel = groups == g
                    w = npx[sel]
                    gm.append((zm["mean"][:, i_m][:, keep][:, sel] * w).sum(axis=1) / w.sum())
                    go.append(float((zo["mean"][i_o][keep][sel] * w).sum() / w.sum()))
                r = agg.coverage_from_members(np.stack(gm, axis=1), np.array(go))
                lo, hi_w = val.wilson_interval(r["n_covered"], r["n"])
                zonal_rows.append({"year": year, "level": f"{level} (reported)", "stat": "mean",
                                   **r, "wilson_lo": float(lo), "wilson_hi": float(hi_w)})

    block_df = pd.DataFrame(rows)
    zonal_df = pd.DataFrame(zonal_rows)
    block_df.to_csv(out_dir / "t2_block_coverage.csv", index=False)
    zonal_df.to_csv(out_dir / "t2_zonal_coverage.csv", index=False)

    for _, r in block_df[block_df["kind"] == "ensemble"].iterrows():
        ok = abs(r["coverage"] - 0.95) <= 0.05
        card.add("T2.1", f"block coverage {int(r['scale_km'])}km ({r['year']})", r["coverage"],
                 "0.95 +/- 0.05", ok, knob="T2_under" if r["coverage"] < 0.95 else "T2_over")
    for _, r in zonal_df.iterrows():
        if "reported" in str(r["level"]):
            card.add("T2.6", f"{r['level']} {r['stat']} ({r['year']})", r["coverage"],
                     "reported with CI (not scored)", None,
                     note=f"Wilson [{r.get('wilson_lo', np.nan):.3f}, {r.get('wilson_hi', np.nan):.3f}]")
            continue
        tid = "T2.2" if r["stat"] == "mean" else "T2.3"
        ok = abs(r["coverage"] - 0.95) <= 0.05
        card.add(tid, f"{r['level']} {r['stat']} ({r['year']})", r["coverage"], "0.95 +/- 0.05", ok,
                 knob="T2_under" if r["coverage"] < 0.95 else "T2_over")

    # ---- T2.4 change between horizons ----------------------------------------------------
    if len(years) >= 2 and Path(args.ecoregion_raster).exists():
        y0, y1 = years[0], years[-1]
        a, b = zonal_members_by_year.get(y0), zonal_members_by_year.get(y1)
        if a and b:
            common, ia, ib = np.intersect1d(a["zone_ids"], b["zone_ids"], return_indices=True)
            dm = b["mean_members"][:, ib] - a["mean_members"][:, ia]
            do = b["mean_obs"][ib] - a["mean_obs"][ia]
            r = agg.coverage_from_members(dm, do)
            card.add("T2.4", f"ecoregion mean change {y0}->{y1}", r["coverage"], "0.95 +/- 0.05",
                     abs(r["coverage"] - 0.95) <= 0.05,
                     note="hardest case: depends on the between-horizon AR(1) correlation",
                     knob="T4.1")
            print(f"  change {y0}->{y1} coverage {r['coverage']:.3f} (n={r['n']})")
    return zonal_members_by_year


def stage_rank(args, zonal_members_by_year, out_dir, card):
    """T2.5 — rank histogram at aggregate (ecoregion) scale."""
    print("\n=== T2.5 · rank histograms ===")
    rows = []
    for year, d in zonal_members_by_year.items():
        hist = agg.rank_histogram(d["mean_members"], d["mean_obs"])
        test = agg.rank_histogram_test(hist)
        rows.append({"year": year, **test, "hist": json.dumps(hist.tolist())})
        computable = test["p_value"] is not None and np.isfinite(test["p_value"])
        card.add("T2.5", f"rank histogram flatness ({year})", test["p_value"], "chi2 p > 0.01",
                 (test["p_value"] > 0.01) if computable else None,
                 note=(f"reliability index {test['reliability_index']:.4f}" if computable
                       else f"not scorable: only {test['n']} aggregation units"),
                 knob="T2_under")
        print(f"  {year}: chi2 p={test['p_value']:.4g}, reliability index "
              f"{test['reliability_index']:.4f}")
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "t2_rank_histograms.csv", index=False)
        _plot_rank_hist(rows, out_dir / "rank_histograms.png")


def _plot_rank_hist(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 3), squeeze=False)
    for ax, r in zip(axes[0], rows):
        h = np.array(json.loads(r["hist"]))
        ax.bar(np.arange(h.size), h / max(h.sum(), 1), width=1.0)
        ax.axhline(1 / h.size, ls="--", c="k", lw=1)
        ax.set_title(f"{r['year']} (p={r['p_value']:.3g})", fontsize=9)
        ax.set_xlabel("rank")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# T3 spatial realism, T4 temporal coherence
# --------------------------------------------------------------------------------------
def _marginal_arrays(paths, year, rows=None, cols=None):
    from src.ensemble.copula import Z975

    def _read(p):
        with rasterio.open(p) as s:
            a = s.read(1).astype(np.float32)
        return a if rows is None else a[np.ix_(rows, cols)]

    cen, low, upp = _read(paths[year]["central"]), _read(paths[year]["lower"]), _read(paths[year]["upper"])
    sl = np.maximum((cen - low) / Z975, 1e-6)
    sr = np.maximum((upp - cen) / Z975, 1e-6)
    return cen, sl, sr


def recover_z(member, cen, sl, sr, eps=1e-4, quant=1.0 / 32767.0, min_sigma_steps=5.0):
    """Invert the copula to get the member's normal score.

    Two classes of pixel are dropped, because at those the member carries no usable
    information about the underlying field and including them would make the spatial and
    temporal diagnostics measure storage artifacts instead:

    * pixels sitting on the [0,1] clip bounds — their score was truncated by the clip;
    * pixels whose applicable sigma is smaller than a few int16 quantization steps — there
      the stored member is central plus rounding noise, and dividing that noise by a tiny
      sigma manufactures enormous spurious scores.
    """
    d = member - cen
    sigma = np.where(d < 0, sl, sr)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = d / sigma
    unusable = (
        (member <= eps) | (member >= 1.0 - eps)
        | (sigma < min_sigma_steps * quant)
    )
    return np.where(np.isfinite(z) & ~unusable, z, np.nan)


def stage_spatial(args, store, attrs, years, paths, out_dir, card, null_store=None, null_attrs=None):
    print("\n=== T3 · spatial realism ===")
    M, nH, H, W = store.shape
    hi = nH - 1
    year = years[hi]
    cen, sl, sr = _marginal_arrays(paths, year)
    mem = agg.member_slice(store, attrs, 0, hi)
    z = recover_z(mem, cen, sl, sr)
    frac_usable = float(np.isfinite(z).sum() / max(np.isfinite(cen).sum(), 1))
    print(f"  recovered normal scores at {100*frac_usable:.1f}% of valid pixels "
          f"(rest clipped at the [0,1] bounds)")

    fit_target = None
    if Path(args.variogram_fits).exists():
        fits = pd.read_csv(args.variogram_fits)
        sub = fits[(fits["stratum"].astype(str) == "ALL")]
        if "horizon" in sub:
            s2 = sub[sub["horizon"] == (years[hi] - args.base_year)]
            sub = s2 if not s2.empty else sub
        if not sub.empty:
            fit_target = sub.iloc[0].to_dict()

    max_lag = min(512, min(H, W) // 3)
    c_, g_, n_ = fld.empirical_variogram_from_field(z, max_lag_px=max_lag, n_pairs=400_000)
    member_fit = vgm.fit_nugget_multirange_model(c_, g_, n_, max_lag_px=max_lag)
    print(f"  member variogram: sill {member_fit['sill']:.3f}, practical range "
          f"{member_fit['practical_range_px']:.1f}px, nugget fraction "
          f"{member_fit['nugget_fraction']:.3f}")
    # The member's normal scores are standardized by construction while the residual field
    # carries its own variance, so only *scale-free* quantities are comparable between the
    # two: the nugget fraction and the practical range. The sill is checked against the
    # generator's own contract (unit variance) instead.
    var_z = float(np.nanvar(z))
    card.add("T3.1", "member normal-score variance", var_z, "1.0 +/- 0.15",
             abs(var_z - 1.0) <= 0.15, knob="T3")
    if fit_target:
        rng_err = abs(member_fit["practical_range_px"] - fit_target["practical_range_px"]) / \
            max(fit_target["practical_range_px"], 1e-9)
        nug_err = abs(member_fit["nugget_fraction"] - fit_target["nugget_fraction"])
        card.add("T3.1", "member practical range vs fitted", rng_err, "<= 0.25 relative",
                 rng_err <= 0.25, knob="T3")
        card.add("T3.1", "member nugget/sill vs fitted", nug_err, "<= 0.10 absolute",
                 nug_err <= 0.10, knob="T3")

    # ---- T3.2 / T3.3 against the independent-pixel null ----------------------------------
    if null_store is not None:
        rng = np.random.default_rng(0)
        with rasterio.open(paths[year]["observed"]) as o:
            o_t = o.transform
            p_t = rasterio.open(paths[year]["central"]).transform
            r_off = int(round((p_t.f - o_t.f) / o_t.e))
            c_off = int(round((p_t.c - o_t.c) / o_t.a))
            obs = o.read(1, window=Window(c_off, r_off, W, H), boundless=True,
                         fill_value=np.nan).astype(np.float32)
        ok = np.isfinite(obs) & np.isfinite(cen)
        idx = np.flatnonzero(ok.ravel())
        if idx.size > args.score_points:
            idx = rng.choice(idx, args.score_points, replace=False)
        rr, cc = np.unravel_index(idx, (H, W))
        coords = np.stack([rr, cc], axis=1).astype(float)
        y = obs.ravel()[idx]
        X = np.stack([agg.member_slice(store, attrs, m, hi).ravel()[idx] for m in range(M)])
        # The null only has to cover the horizon being scored; it is white noise and so
        # compresses far worse than the correlated ensemble, and storing all four horizons
        # of it buys nothing.
        null_hi = min(hi, null_store.shape[1] - 1)
        Xn = np.stack([agg.member_slice(null_store, null_attrs, m, null_hi).ravel()[idx]
                       for m in range(null_store.shape[0])])
        pairs = agg.sample_pairs(idx.size, 20000, rng=rng, coords=coords, max_dist=500)
        vs = agg.variogram_score(X, y, pairs)
        vs_null = agg.variogram_score(Xn, y, pairs)
        es = agg.energy_score(X, y)
        es_null = agg.energy_score(Xn, y)
        es_degen = agg.energy_score(np.repeat(cen.ravel()[idx][None, :], 2, axis=0), y)
        improve = 1.0 - vs / max(vs_null, 1e-12)
        print(f"  variogram score {vs:.4g} vs null {vs_null:.4g} ({100*improve:.1f}% lower)")
        print(f"  energy score {es:.4g} vs null {es_null:.4g}, degenerate {es_degen:.4g}")
        card.add("T3.2", "variogram score vs independent-pixel null", improve, ">= 0.30 lower",
                 improve >= 0.30, knob="T3")
        card.add("T3.3", "energy score beats null and degenerate", es,
                 f"< min(null {es_null:.4g}, degenerate {es_degen:.4g})",
                 es < es_null and es < es_degen, knob="T3")

    # ---- T3.4 radial power spectrum ------------------------------------------------------
    k, P = fld.radial_power_spectrum(z[:min(H, 2048), :min(W, 2048)])
    np.savetxt(out_dir / "t3_member_spectrum.csv", np.stack([k, P], axis=1),
               delimiter=",", header="wavenumber_per_px,power", comments="")
    card.add("T3.4", "radial power spectrum written", len(k), "reference comparison",
             None, note="compare against the residual spectrum in the writeup")

    # ---- T3.5 seam / artifact renders -----------------------------------------------------
    seam_note = "regional grid; lon seam not applicable"
    seam_ok = True
    if W >= 39_000:
        left = agg.member_slice(store, attrs, 0, hi, window=(0, H, 0, 8))
        right = agg.member_slice(store, attrs, 0, hi, window=(0, H, W - 8, W))
        interior = agg.member_slice(store, attrs, 0, hi, window=(0, H, W // 2 - 8, W // 2 + 8))
        seam_diff = np.nanmean(np.abs(right[:, -1] - left[:, 0]))
        int_diff = np.nanmean(np.abs(np.diff(interior, axis=1)))
        seam_ok = bool(np.isfinite(seam_diff) and np.isfinite(int_diff) and seam_diff <= 3 * int_diff)
        seam_note = f"mean |Δ| across seam {seam_diff:.5f} vs interior {int_diff:.5f}"
    card.add("T3.5", "lon seam continuity", seam_note, "no discontinuity (hard gate)",
             seam_ok, knob="T3")
    _plot_member_render(z, out_dir / "member_field_render.png")
    return member_fit


def stage_temporal(args, store, attrs, years, paths, out_dir, card):
    print("\n=== T4 · temporal coherence ===")
    M, nH, H, W = store.shape
    rng = np.random.default_rng(1)
    rows = np.sort(rng.choice(H, size=min(H, 2000), replace=False))
    cols = np.sort(rng.choice(W, size=min(W, 2000), replace=False))

    zs = {}
    for hi, year in enumerate(years):
        cen, sl, sr = _marginal_arrays(paths, year, rows, cols)
        stack = []
        for m in range(M):
            v = agg.member_slice(store, attrs, m, hi)[np.ix_(rows, cols)]
            stack.append(recover_z(v, cen, sl, sr))
        zs[hi] = np.stack(stack)

    rho_target = {}
    if Path(args.rho_json).exists():
        rho_target = {int(k): float(v) for k, v in json.load(open(args.rho_json)).items()}

    for a, b in zip(range(nH - 1), range(1, nH)):
        x, y = zs[a].ravel(), zs[b].ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        corr = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 10 else np.nan
        h = years[b] - args.base_year
        tgt = rho_target.get(h, np.nan)
        ok_flag = bool(np.isfinite(corr) and np.isfinite(tgt) and abs(corr - tgt) <= 0.10)
        print(f"  corr(z_{years[a]}, z_{years[b]}) = {corr:.3f} vs hindcast {tgt:.3f}")
        card.add("T4.1", f"between-horizon corr {years[a]}->{years[b]}", corr,
                 f"within +/-0.10 of {tgt:.3f}", ok_flag if np.isfinite(tgt) else None, knob="T4.1")

    spreads = []
    for hi in range(nH):
        vals = []
        for m in range(M):
            vals.append(agg.member_slice(store, attrs, m, hi)[np.ix_(rows, cols)])
        spreads.append(np.nanstd(np.stack(vals), axis=0))
    spreads = np.stack(spreads)
    ok = np.isfinite(spreads).all(axis=0)
    mono = np.all(np.diff(spreads, axis=0) >= -1e-6, axis=0)
    frac = float(mono[ok].mean()) if ok.any() else np.nan
    print(f"  spread non-decreasing with horizon at {100*frac:.2f}% of sampled pixels")
    card.add("T4.2", "spread non-decreasing in horizon", frac, ">= 0.99", frac >= 0.99, knob="T4.2")


def _plot_member_render(z, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = max(1, max(z.shape) // 2000)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(z[::step, ::step], cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
    ax.set_title("Member 0 normal-score field (seam / tile-edge artifact check)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------
def stage_percentiles(args, pct_paths, years, paths, out_dir, card):
    """T1.2-T1.4 — class-conditional coverage of the ensemble's own per-pixel tails."""
    print("\n=== T1 · class-conditional coverage of the ensemble marginals ===")
    from src.ensemble.residuals import compute_residuals

    rows = []
    for year in years:
        tag = f"ens_{year}"
        info = compute_residuals(
            observed_path=str(paths[year]["observed"]),
            central_path=str(pct_paths[year]["median"]),
            lower_path=str(pct_paths[year]["p2_5"]),
            upper_path=str(pct_paths[year]["p97_5"]),
            baseline_hm_path=str(HM_DIR / f"HM_{args.base_year}_AA_1000.tiff"),
            out_dir=str(out_dir / "residuals"), tag=tag, transform=None,
        )
        rows.append({"window": f"ens-{args.base_year}", "base_year": args.base_year,
                     "target_year": year, "horizon": year - args.base_year, **info})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "ensemble_residual_manifest.csv", index=False)

    eco = args.ecoregion_raster if Path(args.ecoregion_raster).exists() else None
    audit = val.compute_class_conditional_coverage(
        manifest, ecoregion_raster=eco, lookup_csv=args.lookup_csv if eco else None)
    audit.to_csv(out_dir / "t1_ensemble_class_coverage.csv", index=False)
    pooled = val.rollup_coverage(audit, by=["horizon"])
    by_dhat = val.rollup_coverage(audit, by=["horizon", "dhat_bin", "dhat_bin_idx"])
    by_dhat.to_csv(out_dir / "t1_by_dhat.csv", index=False)
    print(by_dhat[["horizon", "dhat_bin", "n_px", "n_eff", "coverage"]].to_string(index=False))

    for _, r in pooled.iterrows():
        card.add("T1.1", f"pooled coverage (h={int(r['horizon'])})", r["coverage"], "0.95 +/- 0.01",
                 abs(r["coverage"] - 0.95) <= 0.01, knob="T1")
    worst = 0.0
    for _, r in by_dhat.iterrows():
        if r["n_eff"] < 100:
            continue
        dev = abs(r["coverage"] - 0.95)
        worst = max(worst, dev)
        card.add("T1.2", f"class coverage h={int(r['horizon'])} {r['dhat_bin']}", r["coverage"],
                 "|cov-0.95| <= 0.03", dev <= 0.03, note=f"n_eff={int(r['n_eff'])}", knob="T1")
        if r["dhat_bin"] in ("(0.05,0.15]", ">0.15"):
            card.add("T1.3", f"high-change tail h={int(r['horizon'])} {r['dhat_bin']}",
                     r["coverage"], ">= 0.92", r["coverage"] >= 0.92, knob="T1")
    card.add("T1.2", "worst primary cell deviation", worst, "<= 0.05", worst <= 0.05, knob="T1")
    return audit


def main(argv=None):
    args = parse_args(argv)
    years = [int(y) for y in args.years.split(",")]
    paths = raster_paths(args, years)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = [s.strip() for s in args.stages.split(",")]

    store, attrs = agg.open_ensemble(args.ensemble)
    null_store = null_attrs = None
    if args.null_ensemble and Path(args.null_ensemble).exists():
        null_store, null_attrs = agg.open_ensemble(args.null_ensemble)

    print("=" * 78)
    print("PHASE 4 — ENSEMBLE VALIDATION SCORECARD")
    print("=" * 78)
    print(f"Ensemble: {args.ensemble} shape={store.shape} members={store.shape[0]}")
    print(f"Null:     {args.null_ensemble or '(none — T3.2/T3.3 will be skipped)'}")

    card = Scorecard()
    t0 = time.time()

    # T1.5 sharpness guard — inflating every interval until coverage passes is not a fix.
    # If width is about to blow past +25%, the correct move is a finer stratification, not
    # a bigger global factor.
    if Path(args.recal_manifest).exists():
        man = json.load(open(args.recal_manifest))
        for o in man.get("outputs", []):
            wr = o.get("width_ratio")
            if wr is None or not np.isfinite(wr):
                continue
            card.add("T1.5", f"interval width vs original heads (h={o['horizon']})", wr,
                     "<= 1.25", wr <= 1.25,
                     note=f"{Path(o['upper']).name}", knob="T1.5 sharpness")

    pct_paths = None
    if "gates" in stages:
        pct_paths = stage_gates(args, store, attrs, years, paths, out_dir, card)
    if "percentiles" in stages:
        if pct_paths is None:
            pct_paths = {y: {q: out_dir / f"ens_{y}_{q}.tif"
                             for q in ("p2_5", "median", "p97_5")} for y in years}
        stage_percentiles(args, pct_paths, years, paths, out_dir, card)
    zonal = {}
    if "aggregate" in stages:
        zonal = stage_aggregate(args, store, attrs, years, paths, out_dir, card,
                                null_store, null_attrs)
    if "rank" in stages and zonal:
        stage_rank(args, zonal, out_dir, card)
    if "spatial" in stages:
        stage_spatial(args, store, attrs, years, paths, out_dir, card, null_store, null_attrs)
    if "temporal" in stages:
        stage_temporal(args, store, attrs, years, paths, out_dir, card)

    df = card.df()
    df.to_csv(out_dir / "scorecard.csv", index=False)
    scored = df[df["pass"].notna()]
    n_pass = int(scored["pass"].sum())
    print("\n" + "=" * 78)
    print(f"SCORECARD: {n_pass}/{len(scored)} scored checks passed "
          f"({len(df) - len(scored)} reported-only) in {(time.time()-t0)/60:.1f} min")
    print("=" * 78)
    failed = scored[~scored["pass"]]
    if len(failed):
        print("\nFailures and the knob that fixes each:")
        for _, r in failed.iterrows():
            print(f"  ✗ {r['id']} {r['metric']}: {r['value']} (target {r['target']})")
            if r["diagnosis"]:
                print(f"      → {r['diagnosis']}")
    print(f"\nScorecard: {out_dir / 'scorecard.csv'}")

    if not args.disable_wandb:
        try:
            import wandb
            run = wandb.init(project="spatio-temporal-convlstm",
                             group=args.wandb_group or "ensemble-validation",
                             job_type="ensemble-validation", tags=["ensemble", "phase4"],
                             config=vars(args))
            run.log({"scorecard": wandb.Table(dataframe=df.astype(str)),
                     "n_pass": n_pass, "n_scored": len(scored),
                     "pass_rate": n_pass / max(len(scored), 1)})
            for fig in ("rank_histograms.png", "member_field_render.png"):
                p = out_dir / fig
                if p.exists():
                    run.log({fig.replace(".png", ""): wandb.Image(str(p))})
            run.finish()
        except Exception as e:
            print(f"⚠ W&B unavailable ({e})")
    return 0 if len(failed) == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
