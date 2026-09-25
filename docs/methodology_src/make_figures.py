#!/usr/bin/env python
"""Figures for docs/methodology.pdf that are not already produced by the scorecards.

    python docs/methodology_src/make_figures.py

Reads the fold mask, the stitched E2a hindcast and the E2a scorecard CSVs; writes PNGs to
docs/methodology_src/figs/. The remaining figures are copied from docs/global/scores/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.windows import Window

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from score_distributional_model import qf_levels  # noqa: E402

FIG = REPO / "docs/methodology_src/figs"
FOLD_MASK = REPO / "data/raw/hm_global/fold_mask_b4_1000.tif"
STITCHED = Path("/mnt/hdd1/spatio-temporal/data/conv_spline/global/g_E2a_hind/stitched")
SCORES = REPO / "docs/global/scores"
HM = REPO / "data/raw/hm_global"

BLUE, ORANGE, GREY, OCEAN = "#2a78d6", "#eb6834", "#c9c7c0", "#ffffff"
INK, INK2 = "#1b1b1b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 220})


def fold_roles():
    """One fold-CV round: fold 1 held out, fold 2 validates, folds 3-5 train."""
    r0, c0, n = 7300, 21500, 2048                         # ~20.3N..2.2N, 13.5E..31.9E
    with rasterio.open(FOLD_MASK) as s:
        g = s.read(1, out_shape=(s.height // 10, s.width // 10), resampling=rasterio.enums.Resampling.mode)
        z = s.read(1, window=Window(c0, r0, n, n))
    # The fold mask also assigns ids over the ocean (its validity mask read the HM raster's
    # finite 3.4e38 sentinel as valid), so roles are drawn on LAND only: that is what is scored.
    with rasterio.open(HM / "HM_2000_AA_1000.tiff") as h:
        lg = h.read(1, out_shape=g.shape, resampling=rasterio.enums.Resampling.nearest)
        lz = h.read(1, window=Window(c0, r0, n, n))
    g = np.where(np.isfinite(lg) & (lg < 1e30), g, 0)
    z = np.where(np.isfinite(lz) & (lz < 1e30), z, 0)
    def roles(a):
        out = np.zeros(a.shape, np.uint8)                 # 0 ocean/invalid
        out[np.isin(a, (3, 4, 5))] = 1                    # training
        out[a == 2] = 2                                   # validation
        out[a == 1] = 3                                   # held out
        return out
    cmap = ListedColormap([OCEAN, GREY, ORANGE, BLUE])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [2.35, 1]})
    ax[0].imshow(roles(g), cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
                 extent=(-180, 180, -70.002, 83.997))
    ax[0].add_patch(plt.Rectangle((13.5, 83.997 - (r0 + n) * 0.009), n * 0.009, n * 0.009,
                                  fill=False, ec=INK, lw=1.0))
    ax[0].set_xlabel("longitude"); ax[0].set_ylabel("latitude")
    ax[0].set_title("(a) one cross-validation round, global", loc="left", fontsize=9)
    ax[1].imshow(roles(z), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    for k in range(0, n + 1, 512):
        ax[1].axhline(k - 0.5, color="white", lw=0.4); ax[1].axvline(k - 0.5, color="white", lw=0.4)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title("(b) 2048 px zoom: 512 px super-blocks", loc="left", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in (BLUE, ORANGE, GREY)]
    fig.legend(handles, ["held-out fold (predicted, scored)", "validation fold", "training folds (3 of 5)"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "fold_roles.png", bbox_inches="tight"); plt.close(fig)


def quantile_ladder(row=9976, col=14889):
    """One pixel's exported quantile function at +20 yr, with the model's 15 knots."""
    qp = STITCHED / "w2000_prediction_2020_qf.tif"
    u = qf_levels(str(qp))[0]
    with rasterio.open(qp) as s:
        q = s.read(window=Window(col, row, 1, 1))[:, 0, 0].astype(float)
        sc = float(s.tags().get("scale", 1.0) or 1.0)
    q = q * sc if sc != 1.0 else q
    def px(p):
        with rasterio.open(p) as s:
            v = float(s.read(1, window=Window(col, row, 1, 1))[0, 0])
        return v
    y0, y1 = px(HM / "HM_2000_AA_1000.tiff"), px(HM / "HM_2020_AA_1000.tiff")
    knots = np.array([0.0, 0.001, 0.005, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.999, 1.0])
    from scipy.stats import norm
    z = norm.ppf(u)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for k in knots[1:-1]:
        ax.axvline(norm.ppf(k), color=GREY, lw=0.7, zorder=0)
    ax.plot(z, q, color=BLUE, lw=2, label=r"$Q_{20}(u)$ exported on 64 levels")
    ax.axhline(y0, color=INK2, lw=1, ls=":", label=f"HM in 2000 (persistence) = {y0:.3f}")
    ax.axhline(y1, color=ORANGE, lw=1.4, label=f"observed HM in 2020 = {y1:.3f}")
    ticks = [0.001, 0.025, 0.25, 0.5, 0.75, 0.975, 0.999]
    ax.set_xticks(norm.ppf(ticks)); ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel(r"quantile level $u$ (probit-spaced axis; grey lines = the 15 model knots)")
    ax.set_ylabel("HM")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "quantile_ladder.png"); plt.close(fig)
    return {"row": row, "col": col, "hm2000": y0, "hm2020": y1,
            "q025": float(np.interp(0.025, u, q)), "q50": float(np.interp(0.5, u, q)),
            "q975": float(np.interp(0.975, u, q))}


def band_skill():
    """CRPS and MSE skill, and 95% coverage, by distance to past change at +20 yr."""
    d = pd.read_csv(SCORES / "dist_g_E2a_hind.csv")
    c = pd.read_csv(SCORES / "central_g_E2a_hind.csv")
    d = d[(d.stratum == "distance") & (d.horizon == 20)].set_index("bin")
    c = c[(c.stratum == "distance") & (c.horizon == 20)].set_index("bin")
    bands = ["0-1", "1-3", "3-10", "10-30", "30-100", ">100"]
    x = np.arange(len(bands)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2))
    ax[0].bar(x - w / 2 - 0.01, d.loc[bands, "crps_skill"], w, color=BLUE, label="CRPS skill")
    ax[0].bar(x + w / 2 + 0.01, c.loc[bands, "skill"], w, color=ORANGE, label="MSE skill")
    ax[0].axhline(0, color=INK2, lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(bands); ax[0].set_xlabel("distance to past change (px)")
    ax[0].set_ylabel("skill vs persistence"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("(a) skill at +20 yr", loc="left", fontsize=9)
    ax[1].bar(x, c.loc[bands, "coverage"], 0.6, color=BLUE)
    ax[1].axhline(0.95, color=INK2, lw=1, ls="--"); ax[1].text(5.45, 0.953, "nominal 0.95", ha="right",
                                                               va="bottom", fontsize=8, color=INK2)
    ax[1].set_ylim(0.6, 1.0); ax[1].set_xticks(x); ax[1].set_xticklabels(bands)
    ax[1].set_xlabel("distance to past change (px)"); ax[1].set_ylabel("95% interval coverage")
    ax[1].set_title("(b) coverage at +20 yr", loc="left", fontsize=9)
    for i, b in enumerate(bands):
        ax[1].text(i, c.loc[b, "coverage"] - 0.012, f"{c.loc[b, 'coverage']:.3f}", ha="center", va="top",
                   fontsize=7.5, color="white", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "band_skill.png"); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fold_roles(); print("fold_roles.png")
    print("quantile_ladder.png", quantile_ladder())
    band_skill(); print("band_skill.png")
    import shutil
    for f in ("densities_g_E2a_hind.png", "pit_g_E2a_hind.png", "chips_g_E2a_hind.png"):
        shutil.copy2(SCORES / f, FIG / f); print("copied", f)
