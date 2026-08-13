#!/usr/bin/env python3
"""Phase 1 + 1.5 driver — diagnostics, variograms, and the recalibration decision.

Runs, in order:
  1b  coverage vs aggregation scale (blocks + ecoregion/biome/realm zonal), the motivating
      "collapse toward 0 at large scales" figure
  1c  biome-stratified variogram fits on the rank-Gaussian residuals, plus the (weak)
      h=20 extrapolation check
  1d  class-conditional coverage audit — the evidence Phase 1.5 acts on
  1.5 conformal scale factors, the keep/global/stratified decision, and the leave-one-
      fold-out coverage table that is reported as the headline

Everything is written under ``data/ensemble/`` and mirrored to W&B as tables and figures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble import calibrate as cal  # noqa: E402
from src.ensemble import validate as val  # noqa: E402
from src.ensemble import variogram as vgm  # noqa: E402

REPO = Path(__file__).parent.parent
HM_DIR = REPO / "data" / "raw" / "hm_global"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/ensemble/residuals/manifest.csv")
    ap.add_argument("--out_dir", default="data/ensemble")
    ap.add_argument("--fold_mask", default=str(HM_DIR / "fold_mask_1000.tif"))
    ap.add_argument("--ecoregion_raster", default=str(HM_DIR / "ecoregion_id_1000.tif"))
    ap.add_argument("--lookup_csv", default=str(HM_DIR / "ecoregion_lookup.csv"))
    ap.add_argument("--block_sizes", default="1,10,100,1000")
    ap.add_argument("--n_pairs", type=int, default=300_000)
    ap.add_argument("--max_lag_px", type=int, default=1024)
    ap.add_argument("--score_cap", type=int, default=8000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n0", type=float, default=200.0)
    ap.add_argument("--refit_scores", action="store_true",
                    help="Re-stream the residual rasters instead of using cached conformal scores")
    ap.add_argument("--stages", default="coverage,variogram,audit,calibrate")
    ap.add_argument("--disable_wandb", action="store_true")
    ap.add_argument("--wandb_group", default=None)
    return ap.parse_args(argv)


def stage_coverage(args, manifest, out_dir, run):
    """1b — block and zonal coverage of the raw pixelwise intervals."""
    print("\n=== 1b · coverage vs aggregation scale ===")
    block_sizes = [int(b) for b in args.block_sizes.split(",")]
    frames, zonal_frames = [], []
    for _, row in manifest.iterrows():
        tag = f"w{int(row['base_year'])}_h{int(row['horizon'])}"
        df = val.compute_block_coverage(
            row["path_lower"], row["path_upper"], row["path_observed"],
            block_sizes=block_sizes, pred_central_path=row["path_central"],
        )
        df["label"] = tag
        df["horizon"] = int(row["horizon"])
        frames.append(df)
        for kind, sub in df.groupby("kind"):
            print(f"  {tag} [{kind}]: " + ", ".join(
                f"{int(r.scale_px)}km={r.coverage:.3f}"
                for r in sub.itertuples() if np.isfinite(r.coverage)))

        if Path(args.ecoregion_raster).exists():
            z = val.compute_zonal_coverage(
                row["path_lower"], row["path_upper"], row["path_observed"],
                args.ecoregion_raster, lookup_csv=args.lookup_csv,
            )
            z["label"] = tag
            z["horizon"] = int(row["horizon"])
            zonal_frames.append(z)

    block_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = out_dir / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    block_df.to_csv(out / "block_coverage.csv", index=False)
    fig = None
    if not block_df.empty:
        fig = val.plot_coverage_vs_scale(
            block_df, out / "coverage_vs_scale.png",
            title="Pixelwise intervals: coverage collapses with aggregation scale",
        )

    zonal_summary = pd.DataFrame()
    if zonal_frames:
        zdf = pd.concat(zonal_frames, ignore_index=True)
        zdf.to_csv(out / "zonal_coverage_ecoregion.csv", index=False)
        pieces = []
        for label, sub in zdf.groupby("label"):
            # Ecoregion is the *scored* unit (n≈846 → Wilson half-width ±0.015, so a ±0.05
            # target is meaningful). Biome (n=14) and realm (n=8) are reported with CIs
            # only — at n=14 the CI is ±0.114, wider than the tolerance, so a biome-level
            # number can neither pass nor fail honestly.
            eco = val.summarize_zonal(sub)
            eco["level"], eco["label"] = "ecoregion (scored)", label
            pieces.append(eco)
            for lvl, col in (("biome (reported)", "BIOME_NAME"), ("realm (reported)", "REALM")):
                if col not in sub:
                    continue
                coarse = val.rollup_zonal(sub, col)
                if coarse.empty:
                    continue
                s = val.summarize_zonal(coarse)
                s["level"], s["label"] = lvl, label
                pieces.append(s)
        zonal_summary = pd.concat(pieces, ignore_index=True)
        zonal_summary.to_csv(out / "zonal_coverage_summary.csv", index=False)
        print(zonal_summary.to_string(index=False))

    if run is not None:
        import wandb
        payload = {}
        if not block_df.empty:
            payload["block_coverage"] = wandb.Table(dataframe=block_df)
        if fig:
            payload["coverage_vs_scale"] = wandb.Image(str(fig))
        if not zonal_summary.empty:
            payload["zonal_coverage"] = wandb.Table(dataframe=zonal_summary)
        run.log(payload)
    return block_df


def stage_variogram(args, manifest, out_dir, run):
    """1c — biome-stratified multi-scale variogram fits on the rank-Gaussian residuals."""
    print("\n=== 1c · variogram fitting ===")
    out = out_dir / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    strat_raster = args.ecoregion_raster if Path(args.ecoregion_raster).exists() else None
    all_fits, global_by_h = [], {}
    for _, row in manifest.iterrows():
        h = int(row["horizon"])
        tag = f"w{int(row['base_year'])}_h{h}"
        gfit_df = vgm.fit_stratified(
            row["path_res_z"], stratum_raster=None, lookup_csv=None, strata=None,
            n_pairs=args.n_pairs, max_lag_px=args.max_lag_px,
        )
        gfit = gfit_df.iloc[0].to_dict()
        gfit.update({"label": tag, "horizon": h, "stratum": "ALL"})
        all_fits.append(gfit)
        global_by_h[h] = gfit
        print(f"  {tag} global: sill={gfit.get('sill', np.nan):.4f} "
              f"range_long={gfit.get('range_long_px', np.nan):.1f}px "
              f"nugget_frac={gfit.get('nugget_fraction', np.nan):.3f} r2={gfit.get('r2', np.nan):.3f}")

        if strat_raster is not None:
            sdf = vgm.fit_stratified(
                row["path_res_z"], stratum_raster=strat_raster, lookup_csv=args.lookup_csv,
                n_pairs=max(60_000, args.n_pairs // 4), max_lag_px=args.max_lag_px,
            )
            sdf["label"] = tag
            sdf["horizon"] = h
            all_fits.extend(sdf.to_dict("records"))
            flagged = sdf[sdf["flag"].astype(str) != ""]
            if len(flagged):
                print(f"    {len(flagged)}/{len(sdf)} biome fits flagged "
                      f"(roll these up to realm): {sorted(flagged['stratum'].dropna().tolist())}")

    fits = pd.DataFrame(all_fits)
    fits.to_csv(out / "variogram_fits.csv", index=False)

    extrap = {}
    if global_by_h:
        extrap = vgm.extrapolate_h20_check(global_by_h, global_by_h.get(20))
        vgm.to_json(extrap, out / "variogram_h20_extrapolation.json")
        print(f"  h=20 extrapolation check: {extrap}")

    if run is not None:
        import wandb
        run.log({"variogram_fits": wandb.Table(dataframe=fits.astype(str)),
                 "variogram_h20_extrapolation": json.dumps(extrap, default=float)})
    return fits


def stage_audit(args, manifest, out_dir, run):
    """1d — class-conditional coverage of the current heads."""
    print("\n=== 1d · class-conditional coverage audit ===")
    out = out_dir / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    eco = args.ecoregion_raster if Path(args.ecoregion_raster).exists() else None
    audit = val.compute_class_conditional_coverage(
        manifest, ecoregion_raster=eco, lookup_csv=args.lookup_csv if eco else None,
    )
    audit.to_csv(out / "coverage_audit.csv", index=False)

    pooled = val.rollup_coverage(audit, by=["horizon"])
    primary = val.rollup_coverage(audit, by=["horizon", "dhat_bin", "dhat_bin_idx"])
    hm_level = val.rollup_coverage(audit, by=["horizon", "hm_bin"])
    pooled.to_csv(out / "coverage_pooled.csv", index=False)
    primary.to_csv(out / "coverage_by_dhat.csv", index=False)
    hm_level.to_csv(out / "coverage_by_hm_level.csv", index=False)

    print("\nPooled coverage (the number the repo has been relying on):")
    print(pooled[["horizon", "n_px", "n_eff", "coverage"]].to_string(index=False))
    print("\nClass-conditional coverage by predicted change:")
    print(primary[["horizon", "dhat_bin", "n_px", "n_eff", "coverage",
                   "frac_below", "frac_above"]].to_string(index=False))

    fig = val.plot_coverage_heatmap(audit, out / "coverage_audit_heatmap.png") if not audit.empty else None
    if run is not None:
        import wandb
        payload = {"coverage_audit": wandb.Table(dataframe=audit),
                   "coverage_pooled": wandb.Table(dataframe=pooled),
                   "coverage_by_dhat": wandb.Table(dataframe=primary)}
        if fig:
            payload["coverage_audit_heatmap"] = wandb.Image(str(fig))
        run.log(payload)
    return audit


def stage_calibrate(args, manifest, out_dir, run, audit=None):
    """1.5a/b/d — conformal factors, decision, and LOFO-CV coverage."""
    print("\n=== 1.5 · conformal recalibration ===")
    out = out_dir / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    eco = args.ecoregion_raster if Path(args.ecoregion_raster).exists() else None
    # Collecting the scores streams every residual raster; refitting from the cached
    # reservoirs is instantaneous, which is what makes iterating on the stratification
    # practical at all.
    store_path = out / "conformal_scores.npz"
    if store_path.exists() and not args.refit_scores:
        print(f"  loading cached conformal scores from {store_path}")
        store = cal.ScoreStore.load(store_path, cap=args.score_cap)
    else:
        store = cal.collect_conformal_scores(
            manifest,
            fold_mask_path=args.fold_mask if Path(args.fold_mask).exists() else None,
            ecoregion_raster=eco, lookup_csv=args.lookup_csv if eco else None,
            cap=args.score_cap,
        )
        store.save(store_path)
        print(f"  cached conformal scores -> {store_path}")
    factors = cal.fit_scale_factors(store, alpha=args.alpha, n0=args.n0)
    if factors.empty:
        print("  ⚠ no conformal scores collected; skipping")
        return None

    decision = cal.decide_recalibration(factors, audit)
    print(f"  DECISION: {decision['decision']} — {decision['reason']}")
    print(f"  factors: mean {decision['mean_s']:.3f}, range [{decision['min_s']:.3f}, "
          f"{decision['max_s']:.3f}], relative spread {decision['relative_spread']:.3f}")

    applied = factors
    if decision["decision"] == "keep":
        applied = cal.as_identity(factors)
    elif decision["decision"] == "global":
        applied = cal.collapse_to_global(factors)
    applied.to_csv(out / "scale_factors.csv", index=False)
    factors.to_csv(out / "scale_factors_raw.csv", index=False)
    with open(out / "recalibration_decision.json", "w") as f:
        json.dump(decision, f, indent=2, default=float)

    lofo = cal.lofo_coverage(store, alpha=args.alpha, n0=args.n0)
    lofo_pooled = cal.pool_lofo(lofo, by=["horizon", "dhat_bin"]) if not lofo.empty else pd.DataFrame()
    if not lofo.empty:
        lofo.to_csv(out / "lofo_coverage_cells.csv", index=False)
        lofo_pooled.to_csv(out / "lofo_coverage.csv", index=False)
        print("\nLeave-one-fold-out coverage after recalibration (the headline table):")
        print(lofo_pooled[["horizon", "dhat_bin", "n_px", "n_eff", "coverage",
                           "wilson_lo", "wilson_hi"]].to_string(index=False))

    if run is not None:
        import wandb
        payload = {
            "scale_factors": wandb.Table(dataframe=applied),
            "recalibration_decision": decision["decision"],
            "recalibration_mean_s": decision["mean_s"],
            "recalibration_relative_spread": decision["relative_spread"],
        }
        if not lofo_pooled.empty:
            payload["lofo_coverage"] = wandb.Table(dataframe=lofo_pooled)
        run.log(payload)
    return {"factors": applied, "decision": decision, "lofo": lofo_pooled}


def main(argv=None):
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"✗ Residual manifest not found: {manifest_path}. Run scripts/run_hindcast_folds.py first.")
        return 1
    manifest = pd.read_csv(manifest_path)
    out_dir = Path(args.out_dir)
    stages = [s.strip() for s in args.stages.split(",")]

    run = None
    if not args.disable_wandb:
        try:
            import wandb
            run = wandb.init(
                project="spatio-temporal-convlstm",
                group=args.wandb_group or f"diagnostics-{time.strftime('%Y%m%d-%H%M%S')}",
                job_type="ensemble-diagnostics",
                tags=["ensemble", "phase1", "calibration"],
                config=vars(args),
            )
        except Exception as e:
            print(f"⚠ W&B unavailable ({e}); continuing without it")

    print("=" * 78)
    print("PHASE 1 + 1.5 — DIAGNOSTICS AND RECALIBRATION")
    print("=" * 78)
    print(f"Residual maps: {len(manifest)} (windows {sorted(manifest['base_year'].unique())})")

    audit = None
    if "coverage" in stages:
        stage_coverage(args, manifest, out_dir, run)
    if "variogram" in stages:
        stage_variogram(args, manifest, out_dir, run)
    if "audit" in stages:
        audit = stage_audit(args, manifest, out_dir, run)
    if "calibrate" in stages:
        if audit is None:
            p = out_dir / "calibration" / "coverage_audit.csv"
            audit = pd.read_csv(p) if p.exists() else None
        stage_calibrate(args, manifest, out_dir, run, audit=audit)

    if run is not None:
        run.finish()
    print("\n✓ Diagnostics complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
