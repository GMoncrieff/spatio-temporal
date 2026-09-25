#!/usr/bin/env python3
"""A detailed, self-contained scorecard for a scored hindcast: accuracy, calibration, PIT.

Distinct from ``score_distributional_model.py``, which COMPUTES the metrics over 184.6 M
pixels in five hours. This only READS what that wrote (``dist_*.csv``, ``central_*.csv``)
and adds four figures that need nothing more than a handful of windowed raster reads plus a
subsampled PIT pass. Re-deriving the metrics here would be a second implementation of every
one of them, which this project has been bitten by (rule 2): one definition, imported.

WHAT IS AND IS NOT HERE. The picket-fence gates -- needle mass, implied-density ceilings,
degenerate-segment fractions -- are deliberately absent. At global scale 98.8% of land sits
ON the physical HM=0 boundary, so those columns are dominated by the boundary rather than by
the monotonicity artefact they were built to catch, and ranking on them would be ranking on
the boundary (rule 25). Accuracy, skill, coverage, calibration, PIT and far-field reach are
what this reports.

THE TARGET YEAR. Everything verified here is the hindcast's +20 yr horizon, base 2000 ->
2020. HM observations exist for 1990-2020 only, so the forecast product's 2025-2040 has no
observed change, no PIT and no actual-vs-predicted map: those three panels can only be built
where truth exists.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from score_distributional_model import (  # noqa: E402
    HORIZONS, pit, qf_levels, read_qf, _read_band, _read_like_band,
)
from src.qf_plots import implied_density, write_scorecard  # noqa: E402


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _f(x, nd=4):
    """A number for a table cell, or an em dash. Never a bare NaN: a blank-looking cell and
    a genuinely missing metric must not read the same."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.0f}" if nd == 0 else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- tables

def pooled(df, h):
    sub = df[(df["stratum"] == "pooled") & (df["horizon"] == h)]
    return sub.iloc[0] if len(sub) else None


def accuracy_table(dist, cen):
    """CRPS and RMSE beside the persistence forecast each is skilled against.

    ``skill = 1 - mse/mse_persistence`` (score_model_experiment.central_stats), so the
    persistence RMSE is recovered exactly as ``rmse / sqrt(1 - skill)`` rather than
    recomputed -- the two must not be able to disagree.
    """
    rows = []
    for h in HORIZONS:
        d, c = pooled(dist, h), pooled(cen, h)
        if d is None or c is None:
            continue
        rmse_p = (c["rmse"] / np.sqrt(1.0 - c["skill"])
                  if np.isfinite(c["skill"]) and c["skill"] < 1 else np.nan)
        rows.append([f"+{h} yr", _f(d["n"], 0), _f(d["crps"], 6),
                     _f(d["crps_persistence"], 6), _f(d["crps_skill"]),
                     _f(c["rmse"], 6), _f(rmse_p, 6), _f(c["skill"]),
                     _f(c["mae"], 6), _f(c["bias"], 6), _f(c["corr"]), _f(c["slope"])])
    return ("Accuracy and skill against persistence",
            ["horizon", "n", "CRPS", "CRPS persist", "CRPS skill",
             "RMSE", "RMSE persist", "RMSE skill", "MAE", "bias", "corr", "slope"], rows)


def calibration_table(dist):
    """Nominal coverage against realised, at four levels, with the interval widths.

    Coverage alone cannot say whether an interval is right: an interval can cover 95% by
    being enormous. The width is printed beside it so the two are read together.
    """
    rows = []
    for h in HORIZONS:
        d = pooled(dist, h)
        if d is None:
            continue
        rows.append([f"+{h} yr",
                     _f(d["cov50"]), _f(d["width50"], 5),
                     _f(d["cov80"]), _f(d["width80"], 5),
                     _f(d["cov95"]), _f(d["width95"], 5),
                     _f(d["cov99"]), _f(d["width99"], 5)])
    return ("Calibration: realised coverage and interval width",
            ["horizon", "cov 50%", "width 50%", "cov 80%", "width 80%",
             "cov 95%", "width 95%", "cov 99%", "width 99%"], rows)


def pit_table(dist):
    """The PIT summary. ``pit_rms_se`` is in standard errors, so it grows with n by
    construction: at 184.6 M pixels it says 'detectably non-uniform', not 'badly
    calibrated'. Read the mean and the tail masses for the size of the miss."""
    rows = []
    for h in HORIZONS:
        d = pooled(dist, h)
        if d is None:
            continue
        rows.append([f"+{h} yr", _f(d["pit_mean"]), _f(d["pit_ks"]),
                     _f(d["pit_lt_0025"], 5), _f(d["pit_gt_0975"], 5),
                     _f(d["pit_lt_0001"], 6), _f(d["pit_gt_0999"], 6),
                     _f(d["pit_rms_se_20"], 1), _f(d["pit_rms_se_60"], 1)])
    return ("PIT: centring and tail mass  (uniform = mean 0.50; nominal 0.025 / 0.025 / "
            "0.001 / 0.001)",
            ["horizon", "PIT mean", "PIT KS", "P(u<.025)", "P(u>.975)",
             "P(u<.001)", "P(u>.999)", "RMS/SE 20 bins", "RMS/SE 60 bins"], rows)


def farfield_table(dist, cen, h):
    """Skill by distance to past change -- the axis a pooled average cannot see (rule 14).

    ``tail_reach`` is how far beyond the 95% interval the far quantiles run; it is reported
    here because the far field is the band the learned tails were promoted to fix.
    """
    d = dist[(dist["stratum"] == "distance") & (dist["horizon"] == h)]
    c = cen[(cen["stratum"] == "distance") & (cen["horizon"] == h)]
    c = c.set_index("bin")
    rows = []
    for _, r in d.iterrows():
        cc = c.loc[r["bin"]] if r["bin"] in c.index else None
        rows.append([str(r["bin"]), _f(r["n"], 0), _f(r["crps"], 6), _f(r["crps_skill"]),
                     _f(cc["rmse"], 6) if cc is not None else "—",
                     _f(cc["skill"]) if cc is not None else "—",
                     _f(r["cov95"]), _f(r["width95"], 5),
                     _f(r["tail_reach_median"], 2), _f(r["pit_mean"])])
    return (f"Far field: by distance to past change, +{h} yr",
            ["band (px)", "n", "CRPS", "CRPS skill", "RMSE", "RMSE skill",
             "cov 95%", "width 95%", "tail reach", "PIT mean"], rows)


def reach_table(dist):
    rows = []
    for h in HORIZONS:
        d = pooled(dist, h)
        if d is None:
            continue
        rows.append([f"+{h} yr", _f(d["tail_reach_median"], 3), _f(d["tail_reach_p90"], 3),
                     _f(d["halfwidth_p999_median"], 5), _f(d["width95"], 5)])
    return ("Far-field reach: how far the extreme quantiles run beyond the 95% interval",
            ["horizon", "reach median", "reach p90", "half-width p99.9", "width 95%"], rows)


# --------------------------------------------------------------------------- sampling

def sample_pixels(paths, mask_path, n=9, seed=7, max_draws=400_000):
    """``n`` scorable pixels drawn uniformly at random from the scored mask.

    Rejection sampling, NOT ``np.argwhere(mask)``: the global mask holds 679.9 M True
    entries, so argwhere alone is 10.9 GB and the permutation another 5.4. Drawing (r, c)
    uniformly and keeping those the mask accepts is uniform over exactly the same set.
    """
    with rasterio.open(paths["central"]) as src:
        H, W = src.height, src.width
        ref = {"transform": src.transform, "width": W, "height": H}
    u = qf_levels(str(paths["qf"]))[0]
    rng = np.random.default_rng(seed)
    msrc = rasterio.open(mask_path)
    out = {"u": u, "qf": [], "hm0": [], "obs": [], "label": []}
    draws = 0
    while len(out["qf"]) < n and draws < max_draws:
        draws += 1
        r, c = int(rng.integers(0, H)), int(rng.integers(0, W))
        if int(msrc.read(1, window=rasterio.windows.Window(c, r, 1, 1))[0, 0]) <= 0:
            continue
        cen = _read_band(str(paths["central"]), r, 1)[0, c]
        obs = _read_like_band(str(paths["observed"]), ref, r, 1)[0, c]
        hm0 = _read_like_band(str(paths["baseline"]), ref, r, 1)[0, c]
        if not (np.isfinite(cen) and np.isfinite(obs) and np.isfinite(hm0)):
            continue
        q = read_qf(str(paths["qf"]), r, 1)[1][:, 0, c]
        if not np.isfinite(q).all():
            continue
        out["qf"].append(q); out["hm0"].append(hm0); out["obs"].append(obs)
        out["label"].append(f"r{r} c{c}")
    msrc.close()
    if not out["qf"]:
        raise SystemExit("found no scorable pixel")
    out["qf"] = np.stack(out["qf"], axis=1)
    out["hm0"] = np.asarray(out["hm0"]); out["obs"] = np.asarray(out["obs"])
    out["n_draws"] = draws
    return out


def sample_chips(paths, mask_path, n=9, size=256, seed=11, min_active=0.02,
                 max_draws=6000):
    """``n`` square chips that actually contain change, for the actual-vs-predicted maps.

    Chips are drawn at random and kept only if at least ``min_active`` of their scorable
    pixels moved by more than 0.01 HM. That is a DELIBERATE bias and is stated on the figure:
    drawn uniformly, nine chips of a 73%-ocean grid whose land is 98.8% unchanged wilderness
    would be nine blank squares, which shows nothing about whether the model puts change in
    the right place.
    """
    with rasterio.open(paths["central"]) as src:
        H, W = src.height, src.width
        ref = {"transform": src.transform, "width": W, "height": H}
        tr = src.transform
    rng = np.random.default_rng(seed)
    chips, draws = [], 0
    while len(chips) < n and draws < max_draws:
        draws += 1
        r = int(rng.integers(0, H - size)); c = int(rng.integers(0, W - size))
        win = rasterio.windows.Window(c, r, size, size)
        with rasterio.open(mask_path) as m:
            if (m.read(1, window=win) > 0).mean() < 0.99:
                continue
        obs = _read_win(paths["observed"], win, ref)
        hm0 = _read_win(paths["baseline"], win, ref)
        cen = _read_win(paths["central"], win, ref)
        ok = np.isfinite(obs) & np.isfinite(hm0) & np.isfinite(cen)
        if ok.mean() < 0.5:
            continue
        d_obs = np.where(ok, obs - hm0, np.nan)
        if np.nanmean(np.abs(d_obs) > 0.01) < min_active:
            continue
        lon = tr.c + (c + size / 2) * tr.a
        lat = tr.f + (r + size / 2) * tr.e
        chips.append({"obs": d_obs, "pred": np.where(ok, cen - hm0, np.nan),
                      "label": f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'} "
                               f"{abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"})
    if not chips:
        raise SystemExit("found no chip with change; lower --min_active")
    return chips, draws


def _read_win(path, win, ref):
    with rasterio.open(str(path)) as s:
        if (s.height, s.width) == (ref["height"], ref["width"]) and s.transform == ref["transform"]:
            return s.read(1, window=win).astype(np.float64)
    # Different grid: fall back to the scorer's own aligned band reader and slice.
    band = _read_like_band(str(path), ref, int(win.row_off), int(win.height))
    return band[:, int(win.col_off):int(win.col_off) + int(win.width)].astype(np.float64)


def pit_subsampled(paths, mask_path, row_chunk=512, every=4):
    """PIT over every ``every``-th row band, using the scorer's own ``pit``.

    A full pass is 75 minutes and this figure does not need one: a histogram of a few tens of
    millions of pixels is not improved by a few hundred million. Bands are evenly spaced down
    the grid, so latitude coverage stays even. The pixel count is printed on the figure --
    a histogram whose n is not stated cannot be judged.
    """
    with rasterio.open(paths["central"]) as src:
        H, W = src.height, src.width
        ref = {"transform": src.transform, "width": W, "height": H}
    u = qf_levels(str(paths["qf"]))[0]
    vals = []
    for i, r0 in enumerate(range(0, H, row_chunk)):
        if i % every:
            continue
        nr = min(row_chunk, H - r0)
        obs = _read_like_band(str(paths["observed"]), ref, r0, nr)
        with rasterio.open(mask_path) as m:
            msk = m.read(1, window=rasterio.windows.Window(0, r0, W, nr)) > 0
        _, q = read_qf(str(paths["qf"]), r0, nr)
        ok = np.isfinite(obs) & msk & np.isfinite(q).all(axis=0)
        if not ok.any():
            del q
            continue
        vals.append(pit(u, q[:, ok], obs[ok]).astype(np.float32))
        del q
    if not vals:
        raise SystemExit("PIT subsample found no scorable pixel")
    return np.concatenate(vals)


# --------------------------------------------------------------------------- figures

def change_density_figure(s, path, target, base):
    plt = _mpl()
    dens = implied_density(s["u"], s["qf"])
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.5))
    for k, ax in enumerate(axes.ravel()):
        if k >= s["qf"].shape[1]:
            ax.axis("off"); continue
        chg = s["qf"][:, k] - s["hm0"][k]
        ax.stairs(dens[:, k], chg, lw=1.0, color="#1f4e79")
        ax.axvline(0.0, color="#2e86c1", lw=2.2, ls=(0, (4, 3)))
        ax.axvline(s["obs"][k] - s["hm0"][k], color="#e67e22", lw=1.4)
        ax.set_yscale("log")
        lo, hi = np.percentile(chg, [2.0, 98.0])
        pad = max(hi - lo, 1e-4) * 0.15
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(f"{s['label'][k]}   observed change "
                     f"{s['obs'][k] - s['hm0'][k]:+.4f}", fontsize=8.5)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.15, lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel(f"HM change, {base} to {target}", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted density", fontsize=8)
    fig.suptitle(f"Predicted distribution of HM CHANGE, {base}→{target}, "
                 f"nine random scored pixels", fontsize=11)
    fig.text(0.5, 0.005, "orange = observed change   blue dashed = no change (persistence)   "
                         "log density axis; the distribution is piecewise constant because "
                         "the raster is a piecewise-linear quantile function",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


def level_density_figure(s, path, target, base):
    plt = _mpl()
    dens = implied_density(s["u"], s["qf"])
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.5))
    for k, ax in enumerate(axes.ravel()):
        if k >= s["qf"].shape[1]:
            ax.axis("off"); continue
        lev = s["qf"][:, k]
        ax.stairs(dens[:, k], lev, lw=1.0, color="#1f4e79")
        # Dashed under solid: at most pixels HM barely moves, so these two land on top of
        # each other and a same-style pair shows only whichever was drawn last.
        ax.axvline(s["hm0"][k], color="#2e86c1", lw=2.2, ls=(0, (4, 3)))
        ax.axvline(s["obs"][k], color="#e67e22", lw=1.4)
        ax.set_yscale("log")
        lo, hi = np.percentile(lev, [2.0, 98.0])
        lo = min(lo, s["hm0"][k], s["obs"][k]); hi = max(hi, s["hm0"][k], s["obs"][k])
        pad = max(hi - lo, 1e-4) * 0.15
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(f"{s['label'][k]}   HM {base} {s['hm0'][k]:.4f} → "
                     f"{target} {s['obs'][k]:.4f}", fontsize=8.5)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.15, lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("HM level", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted density", fontsize=8)
    fig.suptitle(f"Predicted distribution of HM LEVEL at {target}, "
                 f"same nine pixels", fontsize=11)
    fig.text(0.5, 0.005, f"blue = observed HM at {base} (the baseline the model starts from)"
                         f"   orange = observed HM at {target} (the truth)",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


def pit_bins_figure(p, path, target, bins=(8, 32)):
    plt = _mpl()
    fig, axes = plt.subplots(1, len(bins), figsize=(6.2 * len(bins), 4.0), squeeze=False)
    for ax, nb in zip(axes[0], bins):
        c, edges = np.histogram(p, bins=nb, range=(0.0, 1.0))
        centres = 0.5 * (edges[:-1] + edges[1:])
        d = c / c.sum() * nb
        ax.bar(centres, d, width=1.0 / nb, color="#1f4e79", edgecolor="white", linewidth=0.5)
        ax.axhline(1.0, color="#c0392b", ls="--", lw=1.1)
        ax.set_xlim(0, 1)
        ax.set_title(f"{nb} bins   PIT mean {p.mean():.4f}   n = {p.size:,}", fontsize=9.5)
        ax.set_xlabel("PIT  u* = Q$^{-1}$(y)", fontsize=9)
        ax.tick_params(labelsize=8); ax.grid(alpha=0.15, lw=0.5, axis="y")
    axes[0][0].set_ylabel("density (uniform = 1)", fontsize=9)
    fig.suptitle(f"PIT at {target} (+20 yr), two bin widths", fontsize=11)
    fig.text(0.5, 0.005, "dashed red = uniform, the calibrated reference. A coarse and a fine "
                         "binning together separate a real feature from a binning artefact: "
                         "structure that survives both is in the forecast.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


def chips_figure(chips, path, target, base, min_active):
    """Nine chips, observed beside predicted, a shared scale within each pair.

    Paired left-right rather than stacked: the eye compares two squares side by side far
    better than one above the other, and it halves the figure's height.
    """
    plt = _mpl()
    n = len(chips)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 6, figsize=(15.0, 2.65 * rows))
    axes = np.atleast_2d(axes)
    for k, ch in enumerate(chips):
        r, blk = divmod(k, 3)
        ax_o, ax_p = axes[r][blk * 2], axes[r][blk * 2 + 1]
        both = np.concatenate([ch["obs"][np.isfinite(ch["obs"])].ravel(),
                               ch["pred"][np.isfinite(ch["pred"])].ravel()])
        vmax = max(float(np.percentile(np.abs(both), 99.0)) if both.size else 0.05, 1e-3)
        for ax, key, tag in ((ax_o, "obs", f"{ch['label']}  observed"),
                             (ax_p, "pred", "predicted")):
            im = ax.imshow(ch[key], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            # The location goes in the TITLE, not a y-label: a y-label on the left panel sits
            # directly on top of the previous pair's colourbar ticks.
            ax.set_title(tag, fontsize=8, pad=3)
        cb = fig.colorbar(im, ax=[ax_o, ax_p], fraction=0.046, pad=0.02,
                          format="%+.2f")
        cb.ax.tick_params(labelsize=6)
    for k in range(n, rows * 3):
        r, blk = divmod(k, 3)
        axes[r][blk * 2].axis("off"); axes[r][blk * 2 + 1].axis("off")
    fig.suptitle(f"Mean HM change {base}\u2192{target}: observed vs predicted, "
                 f"nine chips of {chips[0]['obs'].shape[0]} px", fontsize=11)
    fig.text(0.5, 0.01,
             f"Red = increase, blue = decrease; the scale is shared WITHIN each pair (99th "
             f"pct of |change|) and differs BETWEEN pairs. Chips are NOT uniform samples: "
             f"each was required to have more than {min_active:.0%} of its pixels move by "
             f"over 0.01 HM, because nine uniform draws of a 73%-ocean grid whose land is "
             f"almost all unchanged would show nothing.",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(path, dpi=115, bbox_inches="tight"); plt.close(fig)
    return path


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stitched_dir", required=True)
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--label", required=True,
                    help="the label the scoring run used; it keys dist_<label>.csv and the "
                         "figure filenames")
    ap.add_argument("--title", default=None,
                    help="display name for the page. Separate from --label because --label "
                         "must keep matching the CSVs on disk.")
    ap.add_argument("--fold_mask", required=True)
    ap.add_argument("--base_year", type=int, default=2000)
    ap.add_argument("--target_year", type=int, default=2020)
    ap.add_argument("--hm_dir", default="data/raw/hm_global")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig_dir", default=None)
    ap.add_argument("--pit_every", type=int, default=4,
                    help="use every Nth row band for the PIT figure (1 = a full pass)")
    ap.add_argument("--row_chunk", type=int, default=512)
    ap.add_argument("--chip_px", type=int, default=256)
    ap.add_argument("--min_active", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    sd, scd = Path(a.stitched_dir), Path(a.scores_dir)
    hm = Path(a.hm_dir)
    b, t = a.base_year, a.target_year
    paths = {"central": sd / f"w{b}_prediction_{t}_central.tif",
             "lower": sd / f"w{b}_prediction_{t}_lower.tif",
             "upper": sd / f"w{b}_prediction_{t}_upper.tif",
             "qf": sd / f"w{b}_prediction_{t}_qf.tif",
             "observed": hm / f"HM_{t}_AA_1000.tiff",
             "baseline": hm / f"HM_{b}_AA_1000.tiff"}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))

    dist = pd.read_csv(scd / f"dist_{a.label}.csv")
    cen = pd.read_csv(scd / f"central_{a.label}.csv")
    fig_dir = Path(a.fig_dir or scd); fig_dir.mkdir(parents=True, exist_ok=True)
    out = Path(a.out or (scd / f"scorecard_detailed_{a.label}.html"))

    print(f"target {t} (base {b}, +{t - b} yr)")
    print("  sampling nine pixels ...", flush=True)
    s = sample_pixels(paths, a.fold_mask, seed=a.seed)
    print(f"    {s['qf'].shape[1]} pixels from {s['n_draws']} draws")
    print("  sampling nine chips ...", flush=True)
    chips, cdraws = sample_chips(paths, a.fold_mask, size=a.chip_px, seed=a.seed + 4,
                                 min_active=a.min_active)
    print(f"    {len(chips)} chips from {cdraws} draws")
    print(f"  PIT over every {a.pit_every}th row band ...", flush=True)
    p = pit_subsampled(paths, a.fold_mask, a.row_chunk, a.pit_every)
    print(f"    {p.size:,} pixels, PIT mean {p.mean():.4f}")

    figs = [
        ("Predicted distribution of HM change, nine pixels",
         change_density_figure(s, fig_dir / f"dist_change_{a.label}.png", t, b)),
        ("Predicted distribution of HM level, same nine pixels",
         level_density_figure(s, fig_dir / f"dist_level_{a.label}.png", t, b)),
        (f"PIT at {t}, 8 and 32 bins",
         pit_bins_figure(p, fig_dir / f"pit_bins_{a.label}.png", t)),
        (f"Observed vs predicted mean HM change, nine chips",
         chips_figure(chips, fig_dir / f"chips_{a.label}.png", t, b, a.min_active)),
    ]

    h_last = max(HORIZONS)
    tables = [accuracy_table(dist, cen), calibration_table(dist), pit_table(dist),
              reach_table(dist), farfield_table(dist, cen, h_last),
              farfield_table(dist, cen, 5)]
    meta = {"target": f"{b} → {t}", "horizons": "+5/+10/+15/+20 yr",
            "pixels scored": f"{int(pooled(dist, h_last)['n']):,}",
            "PIT figure n": f"{p.size:,} (every {a.pit_every}th row band)",
            "stitched": str(sd)}
    notes = (
        "Out-of-sample: a k=5 holdout mosaic, each pixel predicted by the fold whose "
        "training excluded it.",
        "CRPS and RMSE skill are both against <b>persistence</b> (HM unchanged from the base "
        "year), not against zero: the median 20-year HM change is 0.0001, so skill against "
        "zero would be meaningless.",
        "The picket-fence gates are deliberately not reported here. At global scale 98.8% of "
        "land sits on the physical HM=0 boundary, so needle mass and implied-density "
        "ceilings measure the boundary rather than the monotonicity artefact they were "
        "built for.",
        f"Verification stops at {t}: HM observations exist for 1990–2020 only, so the "
        "forecast product's 2025–2040 has no observed change, no PIT and no "
        "actual-vs-predicted map.",
        f"<b>The PIT figure and the PIT table are different samples, and their means differ "
        f"slightly on purpose.</b> The table is the full scoring pass over every scored "
        f"pixel; the figure is every {a.pit_every}th row band ({p.size:,} px), which is "
        f"enough for a histogram and does not need the extra hours. Both numbers are printed "
        f"where they are used rather than reconciled into one.",
    )
    write_scorecard(out, f"{a.title or a.label} — detailed scorecard", meta, tables,
                    figs, notes)
    print(f"\n  wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
