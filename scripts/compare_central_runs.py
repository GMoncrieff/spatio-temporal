#!/usr/bin/env python3
"""Side-by-side central-field comparison across configurations.

Reads the ``central_by_horizon_<label>.csv`` and ``central_error_<label>.csv`` files that
``diagnose_central_field.py`` writes, and prints one table per metric with configurations
as columns — the shape that makes a regression obvious instead of arithmetic.

It also refuses to be fooled by the failure mode that has bitten this project twice: if
two configurations produce byte-identical numbers, that is reported as a probable
wrong-checkpoint error rather than as a null result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["rmse", "mae", "bias", "skill_vs_persistence", "corr_pred_obs",
           "slope_obs_on_pred", "coverage"]


def load(paths_and_labels):
    frames = []
    for label, path in paths_and_labels:
        p = Path(path)
        f = p / f"central_by_horizon_{label}.csv"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        df = pd.read_csv(f)
        df["label"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_detail(paths_and_labels):
    frames = []
    for label, path in paths_and_labels:
        f = Path(path) / f"central_error_{label}.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["label"] = label
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+",
                    help="label=dir pairs, e.g. baseline=data/.../central_diag "
                         "e1_residual=data/ensemble/exp/e1_residual/central_diag")
    ap.add_argument("--reference", default=None,
                    help="label to express the others as a ratio of (default: the first)")
    args = ap.parse_args(argv)

    pairs = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"expected label=dir, got {spec!r}")
        label, path = spec.split("=", 1)
        pairs.append((label, path))
    labels = [lab for lab, _ in pairs]
    ref = args.reference or labels[0]

    df = load(pairs)
    pd.set_option("display.width", 220)

    for metric in METRICS:
        if metric not in df.columns:
            continue
        piv = df.pivot_table(index="horizon", columns="label", values=metric)
        piv = piv.reindex(columns=[lab for lab in labels if lab in piv.columns])
        print(f"\n=== {metric} ===")
        print(piv.to_string(float_format=lambda v: f"{v:.5f}"))
        if metric in ("rmse", "mae") and ref in piv.columns and len(piv.columns) > 1:
            rel = piv.div(piv[ref], axis=0)
            print(f"  (relative to {ref})")
            print(rel.to_string(float_format=lambda v: f"{v:.3f}"))

    # Pixel counts, so a comparison across different valid masks is visible rather than
    # silently unfair.
    npiv = df.pivot_table(index="horizon", columns="label", values="n")
    print("\n=== scored pixels ===")
    print(npiv.to_string(float_format=lambda v: f"{v:,.0f}"))
    if npiv.std(axis=1).max() > 0:
        print("  ⚠ configurations were scored on different pixel counts; "
              "restrict them to a common mask before reading the deltas as model effects")

    # The trap: identical numbers mean the same checkpoint was scored twice.
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            va = df[df.label == a].set_index("horizon")["rmse"]
            vb = df[df.label == b].set_index("horizon")["rmse"]
            common = va.index.intersection(vb.index)
            if len(common) and np.allclose(va[common], vb[common], rtol=0, atol=1e-12):
                print(f"\n  ⚠ {a} and {b} produced byte-identical RMSE at every horizon — "
                      f"almost certainly the same rasters were scored twice, not a null result")

    detail = load_detail(pairs)
    if detail is not None:
        recon = detail[detail.stratum == "reconstruction"]
        if len(recon):
            print("\n=== reconstruction cost on unchanged pixels (RMSE where |dobs| < 0.001) ===")
            piv = recon.pivot_table(index="horizon", columns="label", values="rmse",
                                    aggfunc=lambda s: float(np.sqrt(np.mean(np.asarray(s) ** 2))))
            print(piv.reindex(columns=[l for l in labels if l in piv.columns])
                  .to_string(float_format=lambda v: f"{v:.5f}"))
        dist = detail[detail.stratum == "distance"]
        if len(dist):
            print("\n=== RMSE by distance to past change (h=5) ===")
            piv = dist[dist.horizon == 5].pivot_table(
                index="bin", columns="label", values="rmse",
                aggfunc=lambda s: float(np.sqrt(np.mean(np.asarray(s) ** 2))))
            order = ["0-1px", "1-3px", "3-10px", "10-30px", "30-100px", ">100px"]
            piv = piv.reindex([b for b in order if b in piv.index])
            print(piv.reindex(columns=[l for l in labels if l in piv.columns])
                  .to_string(float_format=lambda v: f"{v:.5f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
