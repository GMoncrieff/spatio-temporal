#!/usr/bin/env python
"""PIT histograms for the two promotion candidates, at three binning resolutions.

Why three. A PIT histogram's bumpiness is partly the forecast and partly the bin count -- the
RMS deviation from uniform grows as sqrt(B) under a *calibrated* forecast, so a single
resolution cannot tell a real feature from sampling noise. Three resolutions can: a feature
that keeps its position and relative height as the bins are refined is being resolved; one
that appears only at the finest is noise. This is the same test ``qf_diagnostics.pit_structure``
reports as ``pit_rms_se_B`` and ``pit_growth_vs_noise``; this script draws the histograms
those numbers summarise, which the scorecard only ever rendered at a single 20 bins.

Cost. A full pass over 23.4 M pixels x 4 horizons x 2 models means re-reading both 64-band
quantile rasters end to end -- about an hour per model. The histogram does not need that: it
reads every Nth band of rows (``--row_stride``), which is a spatially systematic sample rather
than a cherry-picked one, and reports the pixel count it actually used on the figure so the
sample is never mistaken for the population.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

from score_distributional_model import pit, read_qf, _read_like_band  # noqa: E402
from diagnose_central_field import HM_DIR, _read_like  # noqa: E402
from src.qf_diagnostics import pit_structure           # noqa: E402

EXP = Path("/mnt/hdd1/spatio-temporal/data/conv_spline/exp")
FOLD_MASK = "data/raw/hm_global/fold_mask_b4_1000.tif"
FOLDS = (1, 2)
MODELS = [("E2a", "E2a_pwl_freescale_s42", "#1F6FEB"),
          ("E1v", "E1v_pwl_neglog_s42", "#D98324")]
WINDOWS = [2000, 2005, 2010, 2015]
HORIZONS = [5, 10, 15, 20]
MAX_OBSERVED_YEAR = 2020
RESOLUTIONS = [10, 20, 50]

INK, MUTED, GRID = "#1A1D24", "#6B7280", "#D8DCE4"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.7, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def collect(run: str, row_stride: int, band: int = 256):
    """PIT values per horizon, sampled on every ``row_stride``-th band of rows."""
    d = EXP / run / "stitched"
    per_h = {h: [] for h in HORIZONS}
    for base in WINDOWS:
        for h in HORIZONS:
            year = base + h
            if year > MAX_OBSERVED_YEAR:
                continue
            qf_path = d / f"w{base}_prediction_{year}_qf.tif"
            if not qf_path.exists():
                continue
            with rasterio.open(qf_path) as src:
                H, W = src.height, src.width
                ref = {"transform": src.transform, "width": W, "height": H}
            sel_full = np.isin(_read_like(FOLD_MASK, ref), FOLDS)
            obs_p = str(HM_DIR / f"HM_{year}_AA_1000.tiff")
            for k, r0 in enumerate(range(0, H, band)):
                if k % row_stride:
                    continue
                nr = min(band, H - r0)
                u, q = read_qf(str(qf_path), r0, nr)
                y = _read_like_band(obs_p, ref, r0, nr)
                keep = sel_full[r0:r0 + nr] & np.isfinite(y) & np.isfinite(q).all(axis=0)
                if not keep.any():
                    continue
                per_h[h].append(pit(u, q[:, keep], y[keep]).astype(np.float32))
                del u, q, y, keep
    return {h: (np.concatenate(v) if v else np.empty(0, np.float32))
            for h, v in per_h.items()}


def figure(pits, path, row_stride):
    fig, axes = plt.subplots(len(RESOLUTIONS), len(HORIZONS),
                             figsize=(13.2, 2.55 * len(RESOLUTIONS)), squeeze=False)
    for ri, B in enumerate(RESOLUTIONS):
        edges = np.linspace(0.0, 1.0, B + 1)
        centres = 0.5 * (edges[1:] + edges[:-1])
        for ci, h in enumerate(HORIZONS):
            ax = axes[ri][ci]
            for short, _, colour in MODELS:
                p = pits[short][h]
                if p.size == 0:
                    continue
                dens = np.histogram(p, bins=B, range=(0.0, 1.0))[0] / p.size * B
                # Step outlines, not filled bars: two filled histograms hide each other.
                ax.step(np.concatenate([[0.0], centres, [1.0]]),
                        np.concatenate([[dens[0]], dens, [dens[-1]]]),
                        where="mid", color=colour, lw=1.6, label=short)
            ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, None)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
            ax.set_axisbelow(True)
            if ri == 0:
                ax.set_title(f"h = {h} yr", loc="left", fontsize=9.5, weight="bold")
            if ci == 0:
                ax.set_ylabel(f"{B} bins\ndensity", fontsize=8.5)
            if ri == len(RESOLUTIONS) - 1:
                ax.set_xlabel("PIT")
    n = int(sum(pits["E2a"][h].size for h in HORIZONS))
    h_, l_ = axes[0][0].get_legend_handles_labels()
    fig.legend(h_ + [plt.Line2D([], [], color=MUTED, ls="--", lw=1.0)],
               l_ + ["uniform (calibrated)"], loc="upper right", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.995, 0.998))
    fig.suptitle(f"PIT histograms at three binning resolutions  ·  "
                 f"every {row_stride}ᵗʰ band of rows, {n:,} pixel-horizons sampled",
                 x=0.008, y=0.995, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/conv_spline/promotion")
    ap.add_argument("--row_stride", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pits = {}
    for short, run, _ in MODELS:
        print(f"  {short} ...", flush=True)
        pits[short] = collect(run, a.row_stride)
        print(f"    {sum(v.size for v in pits[short].values()):,} pixel-horizons")

    figure(pits, out / "pit_resolutions.png", a.row_stride)

    stats = {}
    for short, _, _ in MODELS:
        stats[short] = {}
        for h in HORIZONS:
            p = pits[short][h]
            s = pit_structure(p, bins=tuple(RESOLUTIONS)) if p.size else {}
            stats[short][h] = {
                "n": int(p.size),
                "mean": float(p.mean()) if p.size else None,
                **{f"rms_se_{b}": s.get(f"pit_rms_se_{b}") for b in RESOLUTIONS},
                **{f"peak_u_{b}": s.get(f"pit_peak_u_{b}") for b in RESOLUTIONS},
                "growth_vs_noise": s.get("pit_growth_vs_noise"),
                "frac_below_half": float((p < 0.5).mean()) if p.size else None,
            }
    (out / "pit_stats.json").write_text(json.dumps(stats, indent=1))
    print("wrote", out / "pit_resolutions.png")


if __name__ == "__main__":
    main()
