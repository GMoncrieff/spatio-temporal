#!/usr/bin/env python3
"""Measure where the central forecast's error actually lives.

The uncertainty work so far has audited the *intervals*. The central field they are
built around has never been audited, so this script asks the same questions of it that
Phase 1d asked of the quantile heads: not "how big is the error" but "which pixels own
it", stratified by covariates available at prediction time.

Three decompositions, all on the out-of-sample hindcast residuals:

  * by distance to past change — the covariate that turned out to explain the quantile
    heads' failure, so it is the first thing to check on the central head too;
  * by predicted change magnitude — whether the model's own confidence tracks its error;
  * by observed change magnitude — diagnostic only (not available at prediction time),
    but it is what separates "the model is bad at change" from "the model is bad at
    standing still".

Plus two questions that a plain error table cannot answer:

  * **Reconstruction error.** The central head predicts absolute HM, not change, so it
    must reproduce HM_t0 through a 16-channel trunk before it can add anything. The
    error on pixels that did not change is that reconstruction cost, measured directly.
  * **Change skill.** Regressing observed change on predicted change gives the slope the
    model is actually operating at; a slope well below 1 means it is shrinking its own
    signal, which no amount of interval calibration can fix.

Everything is scored against persistence (Δ̂ = 0), because for a variable whose median
20-year change is 0.0001 that is the baseline any forecast has to beat to be worth its
uncertainty layer.
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

REGION_ROOT = Path("data/ensemble/region/southern_africa")

# Bands from the T8 table in the plan, so the numbers line up with the ones already
# measured for the quantile heads.
DIST_EDGES = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, np.inf]
DIST_LABELS = ["0-1px", "1-3px", "3-10px", "10-30px", "30-100px", ">100px"]

# Δ̂ bin edges fixed from the measured change distribution (plan, Phase 1d).
DHAT_EDGES = [-np.inf, -0.01, 0.001, 0.01, 0.05, 0.15, np.inf]
DHAT_LABELS = ["<=-0.01", "(-0.01,0.001]", "(0.001,0.01]", "(0.01,0.05]", "(0.05,0.15]", ">0.15"]

# Same edges applied to the observation, for the diagnostic-only cut.
DOBS_LABELS = DHAT_LABELS

HM_EDGES = [0.0, 0.01, 0.1, 0.3, 0.6, 1.01]
HM_LABELS = ["[0,0.01)", "[0.01,0.1)", "[0.1,0.3)", "[0.3,0.6)", "[0.6,1]"]

# "Did not change" for the reconstruction-cost measurement. 0.001 is an order of
# magnitude below the smallest bin edge anywhere else in the project.
STATIC_THRESHOLD = 0.001


HM_DIR = Path("data/raw/hm_global")
WINDOWS = [(1990, 1995, 2000), (1995, 2000, 2005), (2000, 2005, 2010), (2005, 2010, 2015)]
HORIZONS = (5, 10, 15, 20)
MAX_OBSERVED_YEAR = 2020


def _read(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None and np.isfinite(src.nodata):
            arr = np.where(arr == src.nodata, np.nan, arr)
    return arr


def _read_like(path: str | Path, profile: dict, band: int = 1) -> np.ndarray:
    """Read a global raster on the reference raster's grid.

    Everything downstream is a southern-Africa crop while the HM and context rasters are
    global, so the window offset is derived from the two transforms rather than assumed.
    """
    with rasterio.open(path) as src:
        col_off = int(round((profile["transform"].c - src.transform.c) / src.transform.a))
        row_off = int(round((profile["transform"].f - src.transform.f) / src.transform.e))
        arr = src.read(band,
                       window=rasterio.windows.Window(col_off, row_off,
                                                      profile["width"], profile["height"]),
                       boundless=True, fill_value=np.nan).astype(np.float64)
        nod = src.nodata
    if nod is not None and np.isfinite(nod):
        arr = np.where(arr == nod, np.nan, arr)
    # HM rasters carry negative sentinels for ocean in places.
    return np.where(arr < -1e6, np.nan, arr)


def manifest_from_stitched(stitched_dir: Path, context_pattern: str) -> pd.DataFrame:
    """Build the diagnostic's input table straight from a stitched prediction directory.

    Deriving the observation and the baseline here — rather than depending on a separate
    residual-building stage — removes the step where a regional experiment and a global
    covariate raster can silently disagree about which pixels they describe.
    """
    rows = []
    for window in WINDOWS:
        base = window[-1]
        for h in HORIZONS:
            target_year = base + h
            if target_year > MAX_OBSERVED_YEAR:
                continue
            paths = {q: stitched_dir / f"w{base}_prediction_{target_year}_{q}.tif"
                     for q in ("central", "lower", "upper")}
            if not all(p.exists() for p in paths.values()):
                continue
            rows.append({
                "window": "-".join(str(y) for y in window),
                "base_year": base, "target_year": target_year, "horizon": h,
                "path_central": str(paths["central"]),
                "path_lower": str(paths["lower"]),
                "path_upper": str(paths["upper"]),
                "path_observed": str(HM_DIR / f"HM_{target_year}_AA_1000.tiff"),
                "path_baseline": str(HM_DIR / f"HM_{base}_AA_1000.tiff"),
                "path_context": context_pattern.format(year=base),
            })
    if not rows:
        raise SystemExit(f"no stitched rasters found under {stitched_dir}")
    return pd.DataFrame(rows)


def _bin(values: np.ndarray, edges: list[float], labels: list[str]) -> np.ndarray:
    """Right-open binning that returns label indices, -1 for out of range/NaN."""
    idx = np.digitize(values, edges[1:-1], right=True)
    idx = np.where(np.isfinite(values), idx, -1)
    return idx


def _stats(resid: np.ndarray, obs_change: np.ndarray, pred_change: np.ndarray,
           covered: np.ndarray | None = None) -> dict:
    """Error stats for one stratum.

    ``resid`` is observed − central, so it is simultaneously the error of the absolute
    forecast and the error of the change forecast (the HM_t0 term cancels).
    """
    n = int(resid.size)
    if n == 0:
        return {"n": 0}
    mse = float(np.mean(resid ** 2))
    # Persistence predicts no change at all, so its error is the observed change itself.
    mse_persist = float(np.mean(obs_change ** 2))
    out = {
        "n": n,
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(mse)),
        "bias": float(np.mean(resid)),
        "rmse_persistence": float(np.sqrt(mse_persist)),
        # Positive means the model beats "nothing will change"; negative means it does not.
        "skill_vs_persistence": float(1.0 - mse / mse_persist) if mse_persist > 0 else np.nan,
        "mean_obs_change": float(np.mean(obs_change)),
        "mean_pred_change": float(np.mean(pred_change)),
        "sd_obs_change": float(np.std(obs_change)),
        "sd_pred_change": float(np.std(pred_change)),
    }
    if n >= 3 and np.std(pred_change) > 1e-12 and np.std(obs_change) > 1e-12:
        out["corr_pred_obs"] = float(np.corrcoef(pred_change, obs_change)[0, 1])
        # Slope of observed on predicted: 1.0 is a change signal at the right amplitude,
        # < 1 means the model over-states the change it does predict, > 1 means it
        # under-states it.
        slope, intercept = np.polyfit(pred_change, obs_change, 1)
        out["slope_obs_on_pred"] = float(slope)
        out["intercept_obs_on_pred"] = float(intercept)
    else:
        out["corr_pred_obs"] = np.nan
        out["slope_obs_on_pred"] = np.nan
        out["intercept_obs_on_pred"] = np.nan
    if covered is not None and covered.size == resid.size:
        out["coverage"] = float(np.mean(covered))
    return out


def analyse_row(row: pd.Series, fold_sel: np.ndarray | None = None) -> tuple[list[dict], dict]:
    """Return per-stratum records plus the pooled record for one (window, horizon).

    ``fold_sel`` restricts scoring to a subset of the fold mask, which is how a
    single-fold screening run stays comparable to the full k=5 baseline: the same
    pixels are scored in both.
    """
    with rasterio.open(row["path_central"]) as src:
        ref_profile = {"transform": src.transform, "width": src.width, "height": src.height}
    central = _read(row["path_central"])
    lower = _read(row["path_lower"])
    upper = _read(row["path_upper"])

    if "path_baseline" in row and isinstance(row.get("path_baseline"), str):
        # Stitched-directory mode: everything is derived on the prediction raster's grid.
        observed = _read_like(row["path_observed"], ref_profile)
        hm_t0 = _read_like(row["path_baseline"], ref_profile)
        dist = _read_like(row["path_context"], ref_profile, band=2)
    else:
        # Manifest mode. Some manifests record the *global* observed raster even when
        # everything else is a regional crop; the residual is observed − central by
        # construction, so reconstructing from it is exact and on the region's grid.
        with rasterio.open(row["path_observed"]) as src:
            observed_shape = (src.height, src.width)
        if observed_shape == central.shape:
            observed = _read(row["path_observed"])
        else:
            observed = central + _read(row["path_res_native"])
        hm_t0 = _read(row["path_hm_t0"])
        dist = _read(row["path_dist_past_change"])

    valid = (np.isfinite(central) & np.isfinite(observed) & np.isfinite(hm_t0)
             & np.isfinite(lower) & np.isfinite(upper) & np.isfinite(dist))
    if fold_sel is not None:
        valid &= fold_sel

    central = central[valid]
    observed = observed[valid]
    hm_t0 = hm_t0[valid]
    lower = lower[valid]
    upper = upper[valid]
    dist = dist[valid]

    resid = observed - central
    obs_change = observed - hm_t0
    pred_change = central - hm_t0
    covered = ((observed >= lower) & (observed <= upper)).astype(np.float64)

    meta = {"window": row["window"], "horizon": int(row["horizon"]),
            "base_year": int(row["base_year"]), "target_year": int(row["target_year"])}

    records: list[dict] = []

    pooled = {**meta, "stratum": "pooled", "bin": "all",
              **_stats(resid, obs_change, pred_change, covered)}
    records.append(pooled)

    # The cost of reproducing HM_t0 through the trunk, isolated: pixels where the truth
    # did not move, so a perfect model would emit exactly HM_t0.
    static = np.abs(obs_change) < STATIC_THRESHOLD
    if static.sum() > 0:
        recon = {
            **meta, "stratum": "reconstruction", "bin": f"|dobs|<{STATIC_THRESHOLD}",
            "n": int(static.sum()),
            "mae": float(np.mean(np.abs(resid[static]))),
            "rmse": float(np.sqrt(np.mean(resid[static] ** 2))),
            "bias": float(np.mean(resid[static])),
            "rmse_persistence": float(np.sqrt(np.mean(obs_change[static] ** 2))),
            "skill_vs_persistence": np.nan,
            "mean_pred_change": float(np.mean(pred_change[static])),
            "sd_pred_change": float(np.std(pred_change[static])),
            "coverage": float(np.mean(covered[static])),
            "frac_of_valid": float(static.mean()),
        }
        records.append(recon)

    cuts = [
        ("distance", _bin(dist, DIST_EDGES, DIST_LABELS), DIST_LABELS),
        ("dhat", _bin(pred_change, DHAT_EDGES, DHAT_LABELS), DHAT_LABELS),
        ("dobs", _bin(obs_change, DHAT_EDGES, DOBS_LABELS), DOBS_LABELS),
        ("hm_t0", _bin(hm_t0, HM_EDGES, HM_LABELS), HM_LABELS),
    ]
    for name, idx, labels in cuts:
        for b, label in enumerate(labels):
            sel = idx == b
            if sel.sum() == 0:
                continue
            records.append({
                **meta, "stratum": name, "bin": label,
                **_stats(resid[sel], obs_change[sel], pred_change[sel], covered[sel]),
                "frac_of_valid": float(sel.mean()),
            })

    # Variance of the residual explained by a linear function of the baseline level —
    # a direct test of whether the error is a level-dependent reconstruction artifact.
    summary = dict(meta)
    summary["r2_resid_on_hm_t0"] = float(np.corrcoef(hm_t0, resid)[0, 1] ** 2)
    summary["r2_resid_on_dhat"] = float(np.corrcoef(pred_change, resid)[0, 1] ** 2)
    summary["frac_static_px"] = float(static.mean())
    # How much of the true change signal the model reproduces at all.
    summary["var_ratio_pred_obs"] = float(np.var(pred_change) / np.var(obs_change)) if np.var(obs_change) > 0 else np.nan
    return records, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region_root", default=str(REGION_ROOT))
    ap.add_argument("--manifest", default=None,
                    help="defaults to <region_root>/manifest.csv")
    ap.add_argument("--stitched_dir", default=None,
                    help="score a stitched prediction directory directly, deriving the "
                         "observation, baseline and distance covariate from the global "
                         "rasters on the prediction grid (skips the manifest entirely)")
    ap.add_argument("--context_pattern",
                    default="data/raw/hm_global/change_context_w{year}_1000.tif",
                    help="full-raster past-change context; band 2 is distance in pixels")
    ap.add_argument("--restrict_mask", default=None,
                    help="restrict scoring to a mask raster (e.g. the production split "
                         "mask), so two configurations are compared on identical pixels")
    ap.add_argument("--restrict_values", default=None,
                    help="comma-separated values of --restrict_mask to keep")
    ap.add_argument("--out_dir", default=None,
                    help="defaults to <region_root>/central_diag")
    ap.add_argument("--label", default="baseline",
                    help="tag written into every row, so configurations stay separable")
    ap.add_argument("--fold_mask", default=None,
                    help="restrict scoring to particular folds (defaults to "
                         "<region_root>/fold_mask.tif when --folds is given)")
    ap.add_argument("--folds", default=None,
                    help="comma-separated fold ids to score, e.g. '1,2'. Screening runs "
                         "train on a subset of folds, so only those folds' pixels are "
                         "out-of-sample and only they may be scored.")
    ap.add_argument("--wandb_project", default="spatio-temporal-convlstm")
    ap.add_argument("--wandb_group", default="central-field")
    ap.add_argument("--disable_wandb", action="store_true")
    args = ap.parse_args(argv)

    region_root = Path(args.region_root)
    manifest_path = Path(args.manifest) if args.manifest else region_root / "manifest.csv"
    out_dir = Path(args.out_dir) if args.out_dir else region_root / "central_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stitched_dir:
        manifest = manifest_from_stitched(Path(args.stitched_dir), args.context_pattern)
        manifest_path = Path(args.stitched_dir)
        print(f"stitched dir: {manifest_path} ({len(manifest)} window x horizon rows)")
    else:
        manifest = pd.read_csv(manifest_path)
        print(f"manifest: {manifest_path} ({len(manifest)} rows)")

    with rasterio.open(manifest.iloc[0]["path_central"]) as src:
        ref_profile = {"transform": src.transform, "width": src.width, "height": src.height}

    fold_sel = None
    if args.folds:
        fold_ids = {int(x) for x in args.folds.split(",")}
        fm_path = Path(args.fold_mask) if args.fold_mask else region_root / "fold_mask.tif"
        with rasterio.open(fm_path) as src:
            fm = src.read(1) if (src.height, src.width) == (ref_profile["height"], ref_profile["width"]) \
                else _read_like(fm_path, ref_profile)
        fold_sel = np.isin(np.nan_to_num(fm, nan=0).astype(int), sorted(fold_ids))
        print(f"folds {sorted(fold_ids)} from {fm_path}: {int(fold_sel.sum()):,} px selected")

    if args.restrict_mask:
        keep = {int(x) for x in args.restrict_values.split(",")} if args.restrict_values else None
        rm = _read_like(args.restrict_mask, ref_profile)
        rsel = np.isin(np.nan_to_num(rm, nan=-1).astype(int), sorted(keep)) if keep else np.isfinite(rm)
        fold_sel = rsel if fold_sel is None else (fold_sel & rsel)
        print(f"restrict {args.restrict_mask} values {args.restrict_values}: "
              f"{int(rsel.sum()):,} px; combined {int(fold_sel.sum()):,} px")

    all_records: list[dict] = []
    summaries: list[dict] = []
    for _, row in manifest.iterrows():
        print(f"  {row['window']} h={row['horizon']} -> {row['target_year']}")
        recs, summary = analyse_row(row, fold_sel=fold_sel)
        all_records.extend(recs)
        summaries.append(summary)

    df = pd.DataFrame(all_records)
    df.insert(0, "label", args.label)
    sdf = pd.DataFrame(summaries)
    sdf.insert(0, "label", args.label)

    # Byte-identical numbers across supposedly different configurations have twice meant a
    # run scored the wrong checkpoint, so record what was actually read.
    prov = {"manifest": str(manifest_path), "folds": args.folds,
            "central_rasters": sorted({str(r) for r in manifest["path_central"]})[:4]}
    print(f"provenance: central rasters from {Path(manifest.iloc[0]['path_central']).parent}")

    detail_path = out_dir / f"central_error_{args.label}.csv"
    summary_path = out_dir / f"central_summary_{args.label}.csv"
    df.to_csv(detail_path, index=False)
    sdf.to_csv(summary_path, index=False)

    # Pooled-by-horizon roll-up, weighted by pixel count, so h=10 can be compared to the
    # rest on the same footing as the coverage table it is anomalous in.
    pooled = df[df["stratum"] == "pooled"].copy()
    by_h = []
    for h, g in pooled.groupby("horizon"):
        w = g["n"].to_numpy(dtype=float)
        by_h.append({
            "label": args.label, "horizon": int(h), "n": int(w.sum()),
            "n_windows": int(len(g)),
            "rmse": float(np.sqrt(np.sum(w * g["rmse"] ** 2) / w.sum())),
            "mae": float(np.sum(w * g["mae"]) / w.sum()),
            "bias": float(np.sum(w * g["bias"]) / w.sum()),
            "rmse_persistence": float(np.sqrt(np.sum(w * g["rmse_persistence"] ** 2) / w.sum())),
            "coverage": float(np.sum(w * g["coverage"]) / w.sum()),
            "corr_pred_obs": float(np.sum(w * g["corr_pred_obs"]) / w.sum()),
            "slope_obs_on_pred": float(np.sum(w * g["slope_obs_on_pred"]) / w.sum()),
        })
    hdf = pd.DataFrame(by_h)
    hdf["skill_vs_persistence"] = 1.0 - (hdf["rmse"] ** 2) / (hdf["rmse_persistence"] ** 2)
    horizon_path = out_dir / f"central_by_horizon_{args.label}.csv"
    hdf.to_csv(horizon_path, index=False)

    print(f"\n=== pooled by horizon ({args.label}) ===")
    print(hdf.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print(f"\n=== by distance to past change (pixel-weighted over windows) ===")
    dist = df[df["stratum"] == "distance"]
    piv = dist.pivot_table(index="bin", columns="horizon", values="rmse",
                           aggfunc=lambda s: np.sqrt(np.mean(s ** 2)))
    piv = piv.reindex(DIST_LABELS)
    print(piv.to_string(float_format=lambda v: f"{v:.5f}"))

    print(f"\n=== change-signal calibration (slope of observed on predicted) ===")
    print(pooled[["window", "horizon", "corr_pred_obs", "slope_obs_on_pred",
                  "sd_pred_change", "sd_obs_change", "skill_vs_persistence", "coverage"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    recon = df[df["stratum"] == "reconstruction"]
    if len(recon):
        print(f"\n=== reconstruction cost on unchanged pixels (|dobs| < {STATIC_THRESHOLD}) ===")
        print(recon[["window", "horizon", "n", "frac_of_valid", "rmse", "mae", "bias",
                     "sd_pred_change", "coverage"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    if not args.disable_wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, group=args.wandb_group,
                             job_type="central-diagnostics", name=f"central-{args.label}",
                             config={"label": args.label, "region_root": str(region_root)})
            run.log({"central/by_horizon": wandb.Table(dataframe=hdf),
                     "central/detail": wandb.Table(dataframe=df),
                     "central/summary": wandb.Table(dataframe=sdf)})
            for _, r in hdf.iterrows():
                run.log({f"central/h{int(r['horizon'])}/rmse": r["rmse"],
                         f"central/h{int(r['horizon'])}/skill": r["skill_vs_persistence"],
                         f"central/h{int(r['horizon'])}/coverage": r["coverage"]})
            run.finish()
        except Exception as exc:  # pragma: no cover - logging must never fail the run
            print(f"  ⚠ wandb logging skipped: {exc}")

    manifest_out = {
        "label": args.label,
        "detail": str(detail_path),
        "summary": str(summary_path),
        "by_horizon": str(horizon_path),
        "n_manifest_rows": int(len(manifest)),
        **prov,
    }
    (out_dir / f"central_diag_{args.label}.json").write_text(json.dumps(manifest_out, indent=2))
    print(f"\n✓ wrote {detail_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
