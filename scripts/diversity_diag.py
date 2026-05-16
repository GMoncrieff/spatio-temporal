#!/usr/bin/env python
"""Quick diversity diagnostic: compare std/median/q025/q975 rasters between
two prediction directories. Prints per-pixel std mean/p95 and median range.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import rasterio


def stats_for_dir(d: Path) -> dict:
    out = {}
    for key in ["median", "std", "q025", "q975"]:
        p = d / f"prediction_dhm_2020_{key}.tif"
        if not p.exists():
            return None
        with rasterio.open(p) as r:
            arr = r.read(1)
        v = np.isfinite(arr)
        x = arr[v]
        out[key] = {
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "p5": float(np.percentile(x, 5)),
            "p95": float(np.percentile(x, 95)),
            "min": float(x.min()),
            "max": float(x.max()),
        }
    # Sample envelope width = q975 - q025
    with rasterio.open(d / "prediction_dhm_2020_q025.tif") as r:
        q025 = r.read(1)
    with rasterio.open(d / "prediction_dhm_2020_q975.tif") as r:
        q975 = r.read(1)
    v = np.isfinite(q025) & np.isfinite(q975)
    w = (q975 - q025)[v]
    out["envelope_width"] = {
        "mean": float(w.mean()),
        "p95": float(np.percentile(w, 95)),
        "max": float(w.max()),
    }
    # Hotspot vs flat std (does diversity concentrate at hotspots?)
    with rasterio.open(d / "prediction_dhm_2020_median.tif") as r:
        med = r.read(1)
    with rasterio.open(d / "prediction_dhm_2020_std.tif") as r:
        std = r.read(1)
    v = np.isfinite(med) & np.isfinite(std)
    if v.any():
        hot = v & (np.abs(med) > 0.05)
        flat = v & (np.abs(med) < 0.01)
        h_mean = float(std[hot].mean()) if hot.any() else float("nan")
        f_mean = float(std[flat].mean()) if flat.any() else float("nan")
        out["hotspot_std"] = {
            "hotspot_mean": h_mean,
            "flat_mean": f_mean,
            "ratio": (h_mean / f_mean) if (np.isfinite(h_mean) and np.isfinite(f_mean) and f_mean > 0) else float("nan"),
        }
    return out


def main():
    dirs = [Path(x) for x in sys.argv[1:]]
    if not dirs:
        print("usage: diversity_diag.py DIR1 [DIR2 ...]")
        sys.exit(1)
    results = []
    for d in dirs:
        r = stats_for_dir(d)
        results.append((d, r))
        if r is None:
            print(f"{d}: missing files")
            continue
    # Print as table
    cols = ["std_mean", "std_p95", "envelope_mean", "envelope_p95",
            "median_mean", "median_p5", "median_p95", "median_max",
            "std_hot", "std_flat", "hot/flat"]
    print(f"{'dir':<35s}", *(f"{c:>12s}" for c in cols))
    for d, r in results:
        if r is None:
            continue
        h = r.get("hotspot_std", {"hotspot_mean": float("nan"), "flat_mean": float("nan"), "ratio": float("nan")})
        row = [
            r["std"]["mean"], r["std"]["p95"],
            r["envelope_width"]["mean"], r["envelope_width"]["p95"],
            r["median"]["mean"], r["median"]["p5"], r["median"]["p95"],
            r["median"]["max"],
            h["hotspot_mean"], h["flat_mean"], h["ratio"],
        ]
        s = f"{str(d):<35s}" + "".join(f"{v:>+12.4f}" for v in row)
        print(s)


if __name__ == "__main__":
    main()
