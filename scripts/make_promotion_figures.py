#!/usr/bin/env python
"""Figures and metrics for the E1v vs E2a promotion comparison.

Reads the two stitched hindcasts and the observed HM rasters and writes, into --out_dir:

  pixels_1.png .. pixels_3.png   three 3x3 blocks -- 9 pixels, 3 views each
  chips.png                      6 x 128 px chips: observed / E2a / E1v mean HM change
  chips_scatter.png              observed vs predicted change, per chip
  crps_by_horizon.png            CRPS and CRPS skill
  calibration.png                PIT, coverage, reliability
  tails.png                      tail reach, rare-change rates, distance bands
  metrics.json                   every number the page quotes

Both models are drawn from the SAME nine pixels and the SAME six chips: the pixel walk is
seeded off the fold mask rather than off either run's own finite pixels, so the panels stack.
Nothing here interprets anything -- it plots what is on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import TwoSlopeNorm           # noqa: E402

from score_distributional_model import qf_levels, read_qf, _read_band, _read_like_band  # noqa: E402
from diagnose_central_field import HM_DIR, _read_like  # noqa: E402

EXP = Path("/mnt/hdd1/spatio-temporal/data/conv_spline/exp")
SCORES = Path("data/conv_spline/scores")
FOLD_MASK = "data/raw/hm_global/fold_mask_b4_1000.tif"
FOLDS = (1, 2)

# The two candidates. Order is fixed everywhere: colour follows the model, never its rank.
MODELS = [
    ("E2a", "E2a_pwl_freescale_s42", "#1F6FEB"),
    ("E1v", "E1v_pwl_neglog_s42", "#D98324"),
]
BASE_YEAR, HORIZON = 2000, 20
TARGET_YEAR = BASE_YEAR + HORIZON

INK = "#1A1D24"
MUTED = "#6B7280"
GRID = "#D8DCE4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def paths(run: str) -> dict:
    d = EXP / run / "stitched"
    return {
        "central": d / f"w{BASE_YEAR}_prediction_{TARGET_YEAR}_central.tif",
        "lower": d / f"w{BASE_YEAR}_prediction_{TARGET_YEAR}_lower.tif",
        "upper": d / f"w{BASE_YEAR}_prediction_{TARGET_YEAR}_upper.tif",
        "qf": d / f"w{BASE_YEAR}_prediction_{TARGET_YEAR}_qf.tif",
        "observed": HM_DIR / f"HM_{TARGET_YEAR}_AA_1000.tiff",
        "baseline": HM_DIR / f"HM_{BASE_YEAR}_AA_1000.tiff",
    }


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------- nine pixels

def pick_pixels(n=9, seed=0, n_candidates=6000):
    """Nine pixels both models can be read at, drawn off the FOLD MASK.

    Not off either run's own finite pixels: drawing from one model's would bias the pair
    toward wherever that model happens to be defined, and the panels would not stack.
    """
    p = paths(MODELS[0][1])
    with rasterio.open(p["central"]) as src:
        H, W = src.height, src.width
        ref = {"transform": src.transform, "width": W, "height": H}
    sel = np.isin(_read_like(FOLD_MASK, ref), FOLDS)
    rc = np.argwhere(sel)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rc))[:n_candidates]

    out = []
    for i in order:
        r, c = int(rc[i][0]), int(rc[i][1])
        obs = _read_like_band(str(p["observed"]), ref, r, 1)[0, c]
        hm0 = _read_like_band(str(p["baseline"]), ref, r, 1)[0, c]
        if not (np.isfinite(obs) and np.isfinite(hm0)):
            continue
        rec = {"r": r, "c": c, "observed": float(obs), "hm_t0": float(hm0), "q": {}}
        ok = True
        for short, run, _ in MODELS:
            pr = paths(run)
            cen = _read_band(str(pr["central"]), r, 1)[0, c]
            u, q = read_qf(str(pr["qf"]), r, 1)
            q = q[:, 0, c]
            if not (np.isfinite(cen) and np.isfinite(q).all()):
                ok = False
                break
            rec["q"][short] = (u, q, float(cen))
        if ok:
            out.append(rec)
        if len(out) == n:
            break
    if len(out) < n:
        raise SystemExit(f"only found {len(out)} usable pixels")
    return out


def _window(u, ladders, obs, u_lo=0.02, u_hi=0.98, pad=0.18):
    """A view window on the BODY of the ladder, cut on u rather than on value.

    Q(u) runs to the clamp at both ends, so a full-range axis spends its width on the last
    half-percent and flattens everything a reader came for. Cutting on the VALUES does not
    help -- the published u-grid is deliberately tail-dense, so even their 99th percentile
    sits out near the clamp. The window is Q(0.02) to Q(0.98) across both models, widened to
    include the observation, padded. Stated on the page: a clipped axis that does not say so
    is a lie by omission.
    """
    i_lo = int(np.argmin(np.abs(np.asarray(u) - u_lo)))
    i_hi = int(np.argmin(np.abs(np.asarray(u) - u_hi)))
    lo = min(float(np.asarray(q)[i_lo]) for q in ladders)
    hi = max(float(np.asarray(q)[i_hi]) for q in ladders)
    lo, hi = min(lo, obs), max(hi, obs)
    span = max(hi - lo, 1e-4)
    return lo - pad * span, hi + pad * span


def pixel_figure(recs, path, title):
    """Three pixels x three views. Row = pixel, column = view."""
    fig, axes = plt.subplots(3, 3, figsize=(11.6, 9.0))
    for row, rec in enumerate(recs):
        hm0, obs = rec["hm_t0"], rec["observed"]
        obs_ch = obs - hm0
        ugrid = rec["q"][MODELS[0][0]][0]
        qs = [rec["q"][s][1] for s, _, _ in MODELS]
        hm_lo, hm_hi = _window(ugrid, qs, obs)
        ch_lo, ch_hi = _window(ugrid, [q - hm0 for q in qs], obs_ch)

        # 1. predicted CHANGE -- the quantile function re-expressed as change from HM(t0)
        ax = axes[row][0]
        for short, _, colour in MODELS:
            u, q, _ = rec["q"][short]
            ax.plot(q - hm0, u, color=colour, lw=1.8, label=short)
        ax.axvline(obs_ch, color=INK, lw=1.2, ls="--", label="observed")
        ax.axvline(0.0, color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(ch_lo, ch_hi)
        ax.set_xlabel("predicted change in HM")
        ax.set_ylabel("u")
        _style(ax)

        # 2. predicted HM -- the quantile function in absolute HM, with the two anchors
        ax = axes[row][1]
        for short, _, colour in MODELS:
            u, q, cen = rec["q"][short]
            ax.plot(u, q, color=colour, lw=1.8, label=short)
            ax.scatter([0.5], [cen], s=18, color=colour, zorder=5,
                       edgecolor="white", linewidth=0.8)
        ax.axhline(obs, color=INK, lw=1.2, ls="--")
        ax.axhline(hm0, color=MUTED, lw=1.0, ls=":")
        ax.set_ylim(hm_lo, hm_hi)
        ax.set_xlabel("u")
        ax.set_ylabel("predicted HM")
        _style(ax)

        # 3. cumulative probability at the observed value and across HM
        ax = axes[row][2]
        for short, _, colour in MODELS:
            u, q, _ = rec["q"][short]
            ax.plot(q, u, color=colour, lw=1.8, label=short)
            pit = float(np.interp(obs, q, u))
            ax.scatter([obs], [pit], s=26, color=colour, zorder=6,
                       edgecolor="white", linewidth=0.9)
            ax.annotate(f"F={pit:.3f}", (obs, pit), textcoords="offset points",
                        xytext=(6, -2 + 11 * (short == "E1v")), fontsize=7.5, color=colour)
        ax.axvline(obs, color=INK, lw=1.2, ls="--")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(hm_lo, hm_hi)
        ax.set_xlabel("HM")
        ax.set_ylabel("F(HM)")
        _style(ax)

        axes[row][0].text(0.0, 1.07,
                          f"r{rec['r']} c{rec['c']}   HM\u2080 = {hm0:.4f}   "
                          f"observed change {obs_ch:+.4f}",
                          transform=axes[row][0].transAxes, fontsize=8.2, color=INK,
                          ha="left", va="bottom", family="monospace")

    for col, t in enumerate(["predicted change", "predicted HM  (Q(u))",
                             "cumulative probability"]):
        bb = axes[0][col].get_position()
        fig.text(bb.x0, 0.958, t, fontsize=9.5, weight="bold", ha="left", va="bottom")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(title, x=0.012, y=0.992, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.935], h_pad=3.0)
    fig.savefig(path, dpi=155)
    plt.close(fig)


# ----------------------------------------------------------------- chips

def pick_chips(n=6, size=128, seed=11, n_candidates=900):
    """Six 128 px windows inside the fold mask, spread over observed-change magnitude."""
    p = paths(MODELS[0][1])
    with rasterio.open(p["central"]) as src:
        H, W = src.height, src.width
        ref = {"transform": src.transform, "width": W, "height": H}
    sel = np.isin(_read_like(FOLD_MASK, ref), FOLDS)
    rng = np.random.default_rng(seed)

    cands = []
    for _ in range(n_candidates):
        r = int(rng.integers(0, H - size))
        c = int(rng.integers(0, W - size))
        if sel[r:r + size, c:c + size].mean() < 0.85:
            continue
        obs = _read_like_band(str(p["observed"]), ref, r, size)[:, c:c + size]
        hm0 = _read_like_band(str(p["baseline"]), ref, r, size)[:, c:c + size]
        if not (np.isfinite(obs).mean() > 0.95 and np.isfinite(hm0).mean() > 0.95):
            continue
        ch = obs - hm0
        cands.append((r, c, float(np.nanmean(np.abs(ch))), float(np.nanmax(ch))))
    if len(cands) < n:
        raise SystemExit(f"only {len(cands)} candidate chips")
    cands.sort(key=lambda t: t[2])
    idx = np.linspace(0, len(cands) - 1, n).round().astype(int)
    return [cands[i] for i in idx]


def chip_figure(chips, path, size=128):
    p0 = paths(MODELS[0][1])
    with rasterio.open(p0["central"]) as src:
        ref = {"transform": src.transform, "width": src.width, "height": src.height}

    fig, axes = plt.subplots(len(chips), 3, figsize=(8.4, 2.75 * len(chips)))
    store = []
    for row, (r, c, _, _) in enumerate(chips):
        obs = _read_like_band(str(p0["observed"]), ref, r, size)[:, c:c + size]
        hm0 = _read_like_band(str(p0["baseline"]), ref, r, size)[:, c:c + size]
        obs_ch = obs - hm0
        preds = {}
        for short, run, _ in MODELS:
            cen = _read_band(str(paths(run)["central"]), r, size)[:, c:c + size]
            preds[short] = cen - hm0

        # The scale must cover everything DRAWN, not just the observed panel. Taking it from
        # observed alone made a chip where little happened saturate both prediction panels
        # to flat colour -- a scale that hides the only thing the row is there to show.
        allv = np.concatenate([obs_ch.ravel()] + [preds[s].ravel() for s, _, _ in MODELS])
        allv = allv[np.isfinite(allv)]
        vmax = float(np.nanpercentile(np.abs(allv), 99.5)) if allv.size else 0.01
        vmax = max(vmax, 1e-4)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        panels = [("observed", obs_ch)] + [(s, preds[s]) for s, _, _ in MODELS]
        for col, (name, arr) in enumerate(panels):
            ax = axes[row][col]
            im = ax.imshow(arr, cmap="RdBu_r", norm=norm, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID)
            if row == 0:
                ax.set_title(name, loc="left", fontsize=9, weight="bold")
            if col == 0:
                ax.set_ylabel(f"r{r} c{c}", fontsize=7.6, color=MUTED, family="monospace")
            ax.text(0.03, 0.03, f"mean {np.nanmean(arr):+.4f}", transform=ax.transAxes,
                    fontsize=7, color=INK, family="monospace",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.4))
        cb = fig.colorbar(im, ax=list(axes[row]), fraction=0.021, pad=0.012)
        cb.ax.tick_params(labelsize=6.5)
        cb.outline.set_edgecolor(GRID)
        cb.set_label("HM change", fontsize=7, color=MUTED)

        store.append({"r": r, "c": c,
                      "observed_mean": float(np.nanmean(obs_ch)),
                      **{f"{s}_mean": float(np.nanmean(preds[s])) for s, _, _ in MODELS}})

    fig.suptitle(f"Mean HM change {BASE_YEAR}→{TARGET_YEAR}, six 128 px chips",
                 x=0.02, ha="left", fontsize=11, weight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return store


def chip_scatter(chips, path, size=128):
    """Per-pixel observed vs predicted change inside the same six chips."""
    p0 = paths(MODELS[0][1])
    with rasterio.open(p0["central"]) as src:
        ref = {"transform": src.transform, "width": src.width, "height": src.height}
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.6))
    for k, (r, c, _, _) in enumerate(chips):
        ax = axes[k // 3][k % 3]
        obs = _read_like_band(str(p0["observed"]), ref, r, size)[:, c:c + size]
        hm0 = _read_like_band(str(p0["baseline"]), ref, r, size)[:, c:c + size]
        obs_ch = (obs - hm0).ravel()
        for short, run, colour in MODELS:
            cen = _read_band(str(paths(run)["central"]), r, size)[:, c:c + size]
            pr = (cen - hm0).ravel()
            m = np.isfinite(obs_ch) & np.isfinite(pr)
            ax.scatter(obs_ch[m], pr[m], s=2.0, alpha=0.18, color=colour,
                       linewidths=0, label=short)
        lim = np.nanpercentile(np.abs(obs_ch[np.isfinite(obs_ch)]), 99.5)
        lim = max(float(lim), 1e-4)
        ax.plot([-lim, lim], [-lim, lim], color=INK, lw=0.9, ls="--")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"r{r} c{c}", loc="left", fontsize=8.5, family="monospace", color=MUTED)
        ax.set_xlabel("observed change")
        ax.set_ylabel("predicted change")
        _style(ax)
    for col, t in enumerate(["predicted change", "predicted HM  (Q(u))",
                             "cumulative probability"]):
        bb = axes[0][col].get_position()
        fig.text(bb.x0, 0.958, t, fontsize=9.5, weight="bold", ha="left", va="bottom")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.995, 0.997), markerscale=6)
    fig.suptitle("Observed vs predicted change, every pixel in the six chips",
                 x=0.02, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- metric charts

def load(run):
    return (pd.read_csv(SCORES / f"dist_{run}.csv"),
            pd.read_csv(SCORES / f"central_{run}.csv"),
            json.load(open(SCORES / f"summary_{run}.json")))


def wmean(df, col):
    w = df["n"].to_numpy(float)
    v = df[col].to_numpy(float)
    m = np.isfinite(v) & (w > 0)
    return float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else np.nan


def crps_figure(data, path):
    hs = [5, 10, 15, 20]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    for short, _, colour in MODELS:
        d, c, s = data[short]
        axes[0].plot(hs, [s[f"crps{h}"] for h in hs], "o-", color=colour, lw=2, ms=6,
                     label=short, mec="white", mew=1.2)
        axes[1].plot(hs, [s[f"crps_skill{h}"] for h in hs], "o-", color=colour, lw=2, ms=6,
                     label=short, mec="white", mew=1.2)
        axes[2].plot(hs, [s[f"rmse{h}"] for h in hs], "o-", color=colour, lw=2, ms=6,
                     label=short, mec="white", mew=1.2)
    p = data["E2a"][0]
    pers = [wmean(p[(p.stratum == "pooled") & (p.horizon == h)], "crps_persistence")
            for h in hs]
    axes[0].plot(hs, pers, "s--", color=MUTED, lw=1.4, ms=5, label="persistence")
    for ax, t, yl in zip(axes,
                         ["CRPS (raw HM)", "CRPS skill vs persistence", "RMSE of E[Q]"],
                         ["CRPS", "1 − CRPS/CRPS$_p$", "RMSE"]):
        ax.set_title(t, loc="left", fontsize=9.5, weight="bold")
        ax.set_xlabel("horizon (years)")
        ax.set_ylabel(yl)
        ax.set_xticks(hs)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def calibration_figure(data, path):
    hs = [5, 10, 15, 20]
    levels = [(0.50, "cov50", "width50"), (0.80, "cov80", "width80"),
              (0.95, "cov95", "width95"), (0.99, "cov99", "width99")]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.5))

    for short, _, colour in MODELS:
        d, _, s = data[short]
        axes[0].plot(hs, [s[f"pit_mean_{h}"] for h in hs], "o-", color=colour, lw=2, ms=6,
                     label=short, mec="white", mew=1.2)
        axes[1].plot(hs, [s[f"pit_ks{h}"] for h in hs], "o-", color=colour, lw=2, ms=6,
                     mec="white", mew=1.2)
        p = d[(d.stratum == "pooled")]
        nominal = [lv for lv, _, _ in levels]
        got = [wmean(p[p.horizon == 20], cov) for _, cov, _ in levels]
        axes[2].plot(nominal, got, "o-", color=colour, lw=2, ms=6, mec="white", mew=1.2)
        axes[3].plot(hs, [wmean(p[p.horizon == h], "width95") for h in hs], "o-",
                     color=colour, lw=2, ms=6, mec="white", mew=1.2)

    axes[0].axhline(0.5, color=MUTED, ls="--", lw=1.1)
    axes[0].set_title("PIT mean  (0.50 = calibrated)", loc="left", fontsize=9.5, weight="bold")
    axes[0].set_ylabel("mean PIT"); axes[0].set_xlabel("horizon (years)")
    axes[0].set_xticks(hs); axes[0].legend(frameon=False, fontsize=8.5)

    axes[1].set_title("PIT KS distance  (0 = uniform)", loc="left", fontsize=9.5, weight="bold")
    axes[1].set_ylabel("KS"); axes[1].set_xlabel("horizon (years)"); axes[1].set_xticks(hs)

    axes[2].plot([0.4, 1.0], [0.4, 1.0], color=MUTED, ls="--", lw=1.1)
    axes[2].set_title("Coverage vs nominal, h=20", loc="left", fontsize=9.5, weight="bold")
    axes[2].set_xlabel("nominal"); axes[2].set_ylabel("empirical")
    axes[2].set_xticks([0.5, 0.8, 0.95, 0.99])

    axes[3].set_title("95% interval width", loc="left", fontsize=9.5, weight="bold")
    axes[3].set_ylabel("width (HM)"); axes[3].set_xlabel("horizon (years)")
    axes[3].set_xticks(hs)
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tail_figure(data, path):
    hs = [5, 10, 15, 20]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.6))

    for short, _, colour in MODELS:
        d, _, s = data[short]
        p = d[d.stratum == "pooled"]
        axes[0].plot(hs, [wmean(p[p.horizon == h], "tail_reach_median") for h in hs],
                     "o-", color=colour, lw=2, ms=6, label=short, mec="white", mew=1.2)
        axes[1].plot(hs, [s[f"far_tail_excess_{h}"] for h in hs], "o-", color=colour,
                     lw=2, ms=6, mec="white", mew=1.2)

    # rare, large change: predicted vs observed P(change > 0.05), by distance band
    dist_order = ["0-1", "1-3", "3-10", "10-30", "30-100", ">100"]
    x = np.arange(len(dist_order))
    w = 0.36
    d0 = data["E2a"][0]
    dd = d0[(d0.stratum == "distance") & (d0.horizon == 20)]
    obs = [wmean(dd[dd["bin"] == b], "pgt005_obs") for b in dist_order]
    axes[2].bar(x, obs, width=0.82, color=GRID, edgecolor="none", label="observed")
    for k, (short, _, colour) in enumerate(MODELS):
        d, _, _ = data[short]
        dd = d[(d.stratum == "distance") & (d.horizon == 20)]
        pr = [wmean(dd[dd["bin"] == b], "pgt005_pred") for b in dist_order]
        axes[2].bar(x + (k - 0.5) * w, pr, width=w, color=colour, edgecolor="white",
                    linewidth=0.8, label=short)
    axes[2].set_xticks(x); axes[2].set_xticklabels(dist_order, fontsize=7.5)
    axes[2].set_title("P(change > 0.05) by distance to past change, h=20",
                      loc="left", fontsize=9.5, weight="bold")
    axes[2].set_xlabel("distance band (px)"); axes[2].set_ylabel("probability")
    axes[2].legend(frameon=False, fontsize=8)

    # CRPS skill by distance band -- remote land
    for short, _, colour in MODELS:
        d, _, _ = data[short]
        dd = d[(d.stratum == "distance") & (d.horizon == 20)]
        sk = [wmean(dd[dd["bin"] == b], "crps_skill") for b in dist_order]
        axes[3].plot(x, sk, "o-", color=colour, lw=2, ms=6, mec="white", mew=1.2,
                     label=short)
    axes[3].axhline(0.0, color=MUTED, ls="--", lw=1.1)
    axes[3].set_xticks(x); axes[3].set_xticklabels(dist_order, fontsize=7.5)
    axes[3].set_title("CRPS skill by distance band, h=20", loc="left",
                      fontsize=9.5, weight="bold")
    axes[3].set_xlabel("distance band (px)"); axes[3].set_ylabel("CRPS skill")

    axes[0].set_title("Tail reach (median)", loc="left", fontsize=9.5, weight="bold")
    axes[0].set_xlabel("horizon (years)"); axes[0].set_ylabel("reach")
    axes[0].set_xticks(hs); axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].axhline(1.0, color=MUTED, ls="--", lw=1.1)
    axes[1].set_title("Far-tail excess  (1.0 = calibrated)", loc="left",
                      fontsize=9.5, weight="bold")
    axes[1].set_xlabel("horizon (years)"); axes[1].set_ylabel("observed / expected")
    axes[1].set_xticks(hs)
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def obs_change_figure(data, path):
    """CRPS skill and coverage across observed-change magnitude -- the rare-event axis."""
    order = ["<-0.01", "[-0.01,0.001)", "[0.001,0.01)", "[0.01,0.05)", ">=0.05"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    d0 = data["E2a"][0]
    dd = d0[(d0.stratum == "obs_change") & (d0.horizon == 20)]
    counts = [float(dd[dd["bin"] == b]["n"].sum()) for b in order]
    axes[0].bar(x, counts, color=GRID, edgecolor="none")
    axes[0].set_yscale("log")
    axes[0].set_title("pixels per band, h=20", loc="left", fontsize=9.5, weight="bold")
    axes[0].set_ylabel("pixels")

    for short, _, colour in MODELS:
        d, _, _ = data[short]
        dd = d[(d.stratum == "obs_change") & (d.horizon == 20)]
        axes[1].plot(x, [wmean(dd[dd["bin"] == b], "crps_skill") for b in order], "o-",
                     color=colour, lw=2, ms=6, mec="white", mew=1.2, label=short)
        axes[2].plot(x, [wmean(dd[dd["bin"] == b], "cov95") for b in order], "o-",
                     color=colour, lw=2, ms=6, mec="white", mew=1.2)
    axes[1].axhline(0.0, color=MUTED, ls="--", lw=1.1)
    axes[1].set_title("CRPS skill by observed change, h=20", loc="left",
                      fontsize=9.5, weight="bold")
    axes[1].set_ylabel("CRPS skill"); axes[1].legend(frameon=False, fontsize=8.5)
    axes[2].axhline(0.95, color=MUTED, ls="--", lw=1.1)
    axes[2].set_title("95% coverage by observed change, h=20", loc="left",
                      fontsize=9.5, weight="bold")
    axes[2].set_ylabel("empirical coverage")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7, rotation=18, ha="right")
        ax.set_xlabel("observed 20-yr change in HM")
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/conv_spline/promotion")
    ap.add_argument("--pixel_seed", type=int, default=0)
    ap.add_argument("--chip_seed", type=int, default=11)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = {short: load(run) for short, run, _ in MODELS}

    print("metric charts ...")
    crps_figure(data, out / "crps_by_horizon.png")
    calibration_figure(data, out / "calibration.png")
    tail_figure(data, out / "tails.png")
    obs_change_figure(data, out / "obs_change.png")

    print("nine pixels ...")
    recs = pick_pixels(seed=a.pixel_seed)
    for i in range(3):
        pixel_figure(recs[i * 3:(i + 1) * 3], out / f"pixels_{i + 1}.png",
                     f"Pixels {i * 3 + 1}–{i * 3 + 3} of 9  ·  "
                     f"base {BASE_YEAR}, target {TARGET_YEAR} (h={HORIZON})")

    print("six chips ...")
    chips = pick_chips(seed=a.chip_seed)
    chip_stats = chip_figure(chips, out / "chips.png")
    chip_scatter(chips, out / "chips_scatter.png")

    metrics = {
        "base_year": BASE_YEAR, "target_year": TARGET_YEAR, "horizon": HORIZON,
        "models": {short: {"run": run, "colour": col} for short, run, col in MODELS},
        "summary": {short: data[short][2] for short, _, _ in MODELS},
        "pixels": [{"r": r["r"], "c": r["c"], "hm_t0": r["hm_t0"],
                    "observed": r["observed"],
                    "observed_change": r["observed"] - r["hm_t0"],
                    **{f"{s}_pit": float(np.interp(r["observed"], r["q"][s][1], r["q"][s][0]))
                       for s, _, _ in MODELS},
                    **{f"{s}_central": r["q"][s][2] for s, _, _ in MODELS}}
                   for r in recs],
        "chips": chip_stats,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
