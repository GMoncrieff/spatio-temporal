#!/usr/bin/env python3
"""Phase 0d — fold-CV hindcast orchestration.

For each spatial fold, retrain the model with that fold held out entirely, predict every
hindcast input window, then stitch the per-fold rasters into genuinely out-of-sample
global rasters and derive residuals plus the class covariates Phase 1.5 needs.

Training is driven by shelling out to ``scripts/train_lightning.py`` rather than
reimplementing it, so fold-CV models are trained with exactly the same hyperparameters,
loss weights and schedule as the production checkpoint.

Both GPUs are used: folds run concurrently, one process pinned per GPU.

Stages (``--stage``): ``train`` | ``stitch`` | ``residuals`` | ``all``.

Example (global, k=5):
    python scripts/run_hindcast_folds.py --stage all \
        --region config/region_to_predict_large.geojson --gpus 0,1

Smoke test (southern Africa, 2 folds, 1 epoch):
    python scripts/run_hindcast_folds.py --stage all --folds 1,2 --max_epochs 1 \
        --region config/region_to_predict_small.geojson --windows 2000 --train_chips 16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.residuals import (  # noqa: E402
    HORIZONS,
    QUANTILES,
    RankGaussianTransform,
    append_manifest,
    compute_residuals,
    fit_rank_gaussian_transform,
    horizon_autocorrelation,
    sample_hm_values,
    read_manifest,
)

REPO = Path(__file__).parent.parent
HM_DIR = REPO / "data" / "raw" / "hm_global"
FOLD_MASK = HM_DIR / "fold_mask_1000.tif"
ALL_WINDOWS = [(1990, 1995, 2000), (1995, 2000, 2005), (2000, 2005, 2010), (2005, 2010, 2015)]
MAX_OBSERVED_YEAR = 2020

# Hyperparameters of the production checkpoint (artifacts/model-khrpthgy:v0). Fold models
# must match it or later median/spread consistency checks compare different models.
PRODUCTION_HPARAMS = dict(
    hidden_dim=64,
    num_layers=4,
    kernel_size=3,
    locenc_out_channels=8,
    locenc_legendre_polys=10,
    ssim_weight=0.2,
    laplacian_weight=0.3,
    histogram_weight=1.0,
    histogram_lambda_w2=0.1,
    histogram_warmup_epochs=0,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all", choices=["all", "train", "stitch", "residuals"])
    p.add_argument("--folds", default="1,2,3,4,5", help="Comma-separated fold ids to run")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--fold_mask", default=str(FOLD_MASK))
    p.add_argument("--gpus", default="0,1", help="Comma-separated GPU ids; folds run one per GPU")
    p.add_argument("--region", default="config/region_to_predict_large.geojson")
    p.add_argument("--windows", default="all",
                   help="'all' or comma-separated base years, e.g. '2000,2005'")
    p.add_argument("--output_root", default="data/ensemble/hindcast")
    p.add_argument("--residual_dir", default="data/ensemble/residuals")
    p.add_argument("--transform_json", default="data/ensemble/rank_gaussian.json")
    p.add_argument("--norm_stats_json", default="data/ensemble/norm_stats.json")
    p.add_argument("--log_dir", default="data/ensemble/logs")
    # Training passthrough
    p.add_argument("--max_epochs", type=int, default=150)
    p.add_argument("--train_chips", type=int, default=100)
    p.add_argument("--val_chips", type=int, default=40)
    p.add_argument("--val_stride", type=int, default=512,
                   help="Grid stride for validation chips. Grid mode ignores --val_chips, so "
                        "this is what actually bounds per-epoch validation cost.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--predict_stride", type=int, default=64)
    p.add_argument("--predict_batch_size", type=int, default=32)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--disable_wandb", action="store_true")
    p.add_argument("--wandb_group", default=None, help="W&B group (default: hindcast-<timestamp>)")
    p.add_argument("--keep_fold_rasters", action="store_true",
                   help="Keep per-fold prediction rasters after stitching (they are large)")
    p.add_argument("--dry_run", action="store_true", help="Print the fold commands and exit")
    return p.parse_args(argv)


def selected_windows(spec):
    if spec == "all":
        return ALL_WINDOWS
    bases = {int(x) for x in spec.split(",")}
    return [w for w in ALL_WINDOWS if w[-1] in bases]


# --------------------------------------------------------------------------------------
# Stage 1 — fold training + prediction
# --------------------------------------------------------------------------------------
def build_fold_command(args, fold, windows):
    pred_dir = Path(args.output_root) / "preds"
    cmd = [
        args.python, str(REPO / "scripts" / "train_lightning.py"),
        "--max_epochs", str(args.max_epochs),
        "--train_chips", str(args.train_chips),
        "--val_chips", str(args.val_chips),
        "--val_stride", str(args.val_stride),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--devices", "1",
        "--fold_mask", args.fold_mask,
        "--exclude_fold", str(fold),
        "--n_folds", str(args.n_folds),
        "--norm_stats_json", args.norm_stats_json,
        "--run_full_set_evaluation", "False",
        "--run_large_area_prediction", "True",
        "--predict_region", args.region,
        "--predict_stride", str(args.predict_stride),
        "--predict_batch_size", str(args.predict_batch_size),
        "--predict_output_dir", str(pred_dir),
        "--predict_max_target_year", str(MAX_OBSERVED_YEAR),
        "--predict_restrict_mask", args.fold_mask,
        "--predict_restrict_values", str(fold),
    ]
    if len(windows) == len(ALL_WINDOWS):
        # train_lightning loops the windows itself and appends "w{base}_" to the prefix,
        # so one checkpoint load covers all four.
        cmd += ["--predict_all_windows", "True", "--predict_output_prefix", f"fold{fold}_"]
    else:
        if len(windows) != 1:
            raise ValueError("Use --windows all or exactly one base year")
        base = windows[0][-1]
        cmd += [
            "--predict_input_years", ",".join(str(y) for y in windows[0]),
            "--predict_output_prefix", f"fold{fold}_w{base}_",
        ]
    for k, v in PRODUCTION_HPARAMS.items():
        cmd += [f"--{k}", str(v)]
    if args.disable_wandb:
        cmd += ["--disable_wandb"]
    else:
        cmd += [
            "--wandb_group", args.wandb_group,
            "--wandb_run_name", f"hindcast-fold{fold}",
            "--wandb_tags", f"ensemble,hindcast,fold{fold}",
        ]
    return cmd


def run_folds(args, folds, windows):
    """Run fold trainings across GPUs, at most one process per GPU at a time."""
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = list(folds)
    running = {}  # gpu -> (proc, fold, log_handle, t0)
    results = {}
    t_start = time.time()

    while pending or running:
        for gpu in list(gpus):
            if gpu in running or not pending:
                continue
            fold = pending.pop(0)
            cmd = build_fold_command(args, fold, windows)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_path = log_dir / f"hindcast_fold{fold}.log"
            fh = open(log_path, "w")
            print(f"[orchestrator] fold {fold} -> GPU {gpu} | log {log_path}")
            proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT)
            running[gpu] = (proc, fold, fh, time.time())

        time.sleep(10)
        for gpu, (proc, fold, fh, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            elapsed = time.time() - t0
            status = "ok" if rc == 0 else f"FAILED (rc={rc})"
            print(f"[orchestrator] fold {fold} finished on GPU {gpu} in {elapsed/60:.1f} min — {status}")
            results[fold] = {"returncode": rc, "minutes": elapsed / 60.0, "gpu": gpu}
            del running[gpu]

    total = (time.time() - t_start) / 60.0
    print(f"[orchestrator] all folds done in {total:.1f} min wall clock")
    failed = [f for f, r in results.items() if r["returncode"] != 0]
    if failed:
        raise RuntimeError(f"Folds failed: {failed} — see {log_dir}")
    return results


# --------------------------------------------------------------------------------------
# Stage 2 — stitch
# --------------------------------------------------------------------------------------
def stitch_all(args, folds, windows):
    from src.ensemble.residuals import stitch_fold_predictions

    pred_dir = Path(args.output_root) / "preds"
    out_dir = Path(args.output_root) / "stitched"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for window in windows:
        base = window[-1]
        for h in HORIZONS:
            target_year = base + h
            if target_year > MAX_OBSERVED_YEAR:
                continue
            for q in QUANTILES:
                fold_paths = {}
                for f in folds:
                    p = pred_dir / f"fold{f}_w{base}_prediction_{target_year}_{q}_blended.tif"
                    if p.exists():
                        fold_paths[f] = str(p)
                if not fold_paths:
                    print(f"  ⚠ no fold rasters for w{base} {target_year} {q}; skipping")
                    continue
                out_path = out_dir / f"w{base}_prediction_{target_year}_{q}.tif"
                info = stitch_fold_predictions(fold_paths, args.fold_mask, str(out_path))
                print(f"  ✓ w{base} {target_year} {q}: {info['n_valid_px']:,} px -> {out_path}")
                written.append(info)
    if not args.keep_fold_rasters:
        n = 0
        for p in pred_dir.glob("fold*_w*_prediction_*.tif"):
            p.unlink()
            n += 1
        if n:
            print(f"  · removed {n} per-fold rasters (use --keep_fold_rasters to retain)")
    return written


# --------------------------------------------------------------------------------------
# Stage 3 — residuals
# --------------------------------------------------------------------------------------
def ensure_transform(args):
    path = Path(args.transform_json)
    if path.exists():
        print(f"Using existing rank-Gaussian transform: {path}")
        return RankGaussianTransform.from_json(path)
    print("Fitting rank-Gaussian transform from observed HM ...")
    samples = []
    for year in (2005, 2010, 2015, 2020):
        f = HM_DIR / f"HM_{year}_AA_1000.tiff"
        if f.exists():
            samples.append(sample_hm_values(f, n_windows=100, window_size=256, random_seed=42 + year))
    sample = np.concatenate(samples)
    tr = fit_rank_gaussian_transform(sample)
    tr.to_json(path)
    print(f"  p0 (exact-zero mass) = {tr.p0:.4f}, {len(tr.knot_values)} knots -> {path}")
    return tr


def residuals_all(args, windows):
    tr = ensure_transform(args)
    stitched = Path(args.output_root) / "stitched"
    res_dir = Path(args.residual_dir)
    manifest = res_dir / "manifest.csv"
    if manifest.exists():
        manifest.unlink()

    rows = []
    for window in windows:
        base = window[-1]
        for h in HORIZONS:
            target_year = base + h
            if target_year > MAX_OBSERVED_YEAR:
                continue
            paths = {q: stitched / f"w{base}_prediction_{target_year}_{q}.tif" for q in QUANTILES}
            if not all(p.exists() for p in paths.values()):
                print(f"  ⚠ missing stitched rasters for w{base} h{h}; skipping")
                continue
            observed = HM_DIR / f"HM_{target_year}_AA_1000.tiff"
            baseline = HM_DIR / f"HM_{base}_AA_1000.tiff"
            tag = f"w{base}_h{h}"
            info = compute_residuals(
                observed_path=str(observed),
                central_path=str(paths["central"]),
                lower_path=str(paths["lower"]),
                upper_path=str(paths["upper"]),
                baseline_hm_path=str(baseline),
                out_dir=str(res_dir),
                tag=tag,
                transform=tr,
            )
            row = {
                "window": f"{window[0]}-{window[1]}-{window[2]}",
                "base_year": base,
                "target_year": target_year,
                "horizon": h,
                "path_central": str(paths["central"]),
                "path_lower": str(paths["lower"]),
                "path_upper": str(paths["upper"]),
                "path_observed": str(observed),
                **info,
            }
            append_manifest(manifest, row)
            rows.append(row)
            print(f"  ✓ {tag}: {info['n_valid_px']:,} valid residual px")

    if rows:
        rho = horizon_autocorrelation(manifest)
        rho_path = res_dir / "horizon_autocorrelation.json"
        with open(rho_path, "w") as f:
            json.dump(rho, f, indent=2)
        print(f"  ✓ horizon autocorrelation rho = "
              f"{ {k: round(v, 4) for k, v in rho.items()} } -> {rho_path}")
    return rows


# --------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    folds = [int(f) for f in args.folds.split(",")]
    windows = selected_windows(args.windows)
    if args.wandb_group is None:
        args.wandb_group = f"hindcast-{time.strftime('%Y%m%d-%H%M%S')}"

    print("=" * 78)
    print("PHASE 0 — FOLD-CV HINDCAST HARNESS")
    print("=" * 78)
    print(f"Folds:        {folds}")
    print(f"Windows:      {[w[-1] for w in windows]} (base years)")
    print(f"Region:       {args.region}")
    print(f"GPUs:         {args.gpus}")
    print(f"Stage:        {args.stage}")
    print(f"W&B group:    {args.wandb_group}")
    print("=" * 78)

    if args.dry_run:
        for f in folds:
            print(" ".join(build_fold_command(args, f, windows)))
        return 0

    run = None
    if not args.disable_wandb:
        try:
            import wandb

            run = wandb.init(
                project="spatio-temporal-convlstm",
                group=args.wandb_group,
                job_type="hindcast-orchestration",
                name=f"{args.wandb_group}-orchestrator",
                tags=["ensemble", "hindcast", "phase0"],
                config={**vars(args), "windows": [w[-1] for w in windows], "folds": folds},
            )
        except Exception as e:
            print(f"⚠ W&B unavailable ({e}); continuing without it")

    t0 = time.time()
    if args.stage in ("all", "train"):
        results = run_folds(args, folds, windows)
        if run is not None:
            import wandb

            tbl = wandb.Table(columns=["fold", "gpu", "minutes", "returncode"])
            for f, r in sorted(results.items()):
                tbl.add_data(f, r["gpu"], round(r["minutes"], 2), r["returncode"])
            run.log({"fold_training": tbl,
                     "fold_minutes_total": sum(r["minutes"] for r in results.values())})

    if args.stage in ("all", "stitch"):
        print("\n--- Stitching fold predictions ---")
        stitch_all(args, folds, windows)

    if args.stage in ("all", "residuals"):
        print("\n--- Computing residuals ---")
        rows = residuals_all(args, windows)
        if run is not None and rows:
            import wandb

            df = read_manifest(Path(args.residual_dir) / "manifest.csv")
            run.log({"residual_manifest": wandb.Table(dataframe=df)})

    elapsed = (time.time() - t0) / 60
    print(f"\n✓ Phase 0 stage '{args.stage}' complete in {elapsed:.1f} min")
    if run is not None:
        run.log({"phase0_minutes": elapsed})
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
