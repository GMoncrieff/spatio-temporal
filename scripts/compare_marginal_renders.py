#!/usr/bin/env python3
"""Observed change beside members from two marginal families, in one figure.

The two families differ only in how mass is distributed *between* the published bounds —
the bounds, the central forecast and the correlated field are identical — so comparing them
across two separate W&B runs makes the eye do work it is bad at. These panels put them on
the same rows, same windows, same member indices, same colour scale.

Three views, because the defect is invisible in the first and obvious in the other two:

  * the signed change field, which is what "looks realistic" usually means and which a
    diverging colour map renders almost identically for both families;
  * the same field thresholded — "did this pixel change by more than X" — which is the
    quantity T6.1 and T8.1 actually score, and where the two-piece normal's excess of
    moderate change becomes plain;
  * the change distribution on a log density, where the body difference is quantitative.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble import aggregate as agg  # noqa: E402

HM_DIR = Path("data/raw/hm_global")


def read_like(path, ref, band=1):
    with rasterio.open(path) as s:
        co = int(round((ref["transform"].c - s.transform.c) / s.transform.a))
        ro = int(round((ref["transform"].f - s.transform.f) / s.transform.e))
        a = s.read(band, window=Window(co, ro, ref["width"], ref["height"]),
                   boundless=True, fill_value=np.nan).astype(np.float64)
    return np.where(a < -1e6, np.nan, a)


def pick_windows(d_obs, size=768, n=2, min_land=0.6, n_candidates=400):
    """The busiest window and a quiet one — the two regimes that fail differently.

    **Candidates must be substantially on land.** Requiring merely one finite pixel lets an
    almost-entirely-ocean window win the "quiet" slot on a nanmean taken over a handful of
    coastal pixels — which renders as a blank map with a few coloured pixels in one corner,
    and says nothing about the forecast. Measured on Africa: the unguarded pick returned a
    window whose maps were empty while its member histograms were fully populated, which is
    the tell that the crop, not the field, was degenerate. ``spread_windows`` in
    ``plot_forecast_panel.py`` already guarded this at 0.6; the guard belongs here, where
    both scripts read it from.

    The candidate count is raised with it: at 60 draws a 60%-land requirement can leave the
    quiet slot filled by whatever survived rather than by the quietest land window.
    """
    H, W = d_obs.shape
    rng = np.random.default_rng(0)
    best, quiet = None, None
    for _ in range(n_candidates):
        r0 = int(rng.integers(0, max(1, H - size)))
        c0 = int(rng.integers(0, max(1, W - size)))
        sub = d_obs[r0:r0 + size, c0:c0 + size]
        finite = np.isfinite(sub)
        if finite.mean() < min_land:
            continue
        score = float(np.nanmean(np.abs(sub)))
        if best is None or score > best[0]:
            best = (score, (r0, c0))
        if quiet is None or score < quiet[0]:
            quiet = (score, (r0, c0))
    out = {}
    if best:
        out["high_change"] = best[1]
    if quiet and n > 1:
        out["quiet"] = quiet[1]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble_a", required=True)
    ap.add_argument("--label_a", default="two-piece normal")
    ap.add_argument("--ensemble_b", required=True)
    ap.add_argument("--label_b", default="empirical shape")
    ap.add_argument("--recal_dir", required=True)
    ap.add_argument("--base_year", type=int, default=2000)
    ap.add_argument("--years", default="2005,2010,2015,2020")
    ap.add_argument("--members", default="0,1,2")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--wandb_project", default="spatio-temporal-convlstm")
    ap.add_argument("--wandb_group", default="marginal-comparison")
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--disable_wandb", action="store_true")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = [int(y) for y in args.years.split(",")]
    mids = [int(m) for m in args.members.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sa, aa = agg.open_ensemble(args.ensemble_a)
    sb, ab = agg.open_ensemble(args.ensemble_b)
    print(f"{args.label_a}: M={sa.shape[0]}   {args.label_b}: M={sb.shape[0]}")
    if sa.shape[0] != sb.shape[0]:
        print("  ⚠ different member counts — percentile-sensitive panels are not comparable")

    run = None
    if not args.disable_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, group=args.wandb_group,
                         job_type="marginal-renders",
                         name=args.wandb_run_name or f"{args.label_a} vs {args.label_b}",
                         config={"ensemble_a": args.ensemble_a, "ensemble_b": args.ensemble_b,
                                 "label_a": args.label_a, "label_b": args.label_b,
                                 "members_a": int(sa.shape[0]), "members_b": int(sb.shape[0])})

    figures, rows = {}, []
    for hi, year in enumerate(years):
        cen_path = Path(args.recal_dir) / f"w{args.base_year}_prediction_{year}_central_recal.tif"
        with rasterio.open(cen_path) as s:
            ref = {"transform": s.transform, "width": s.width, "height": s.height}
            cen = s.read(1).astype(np.float64)
        hm0 = read_like(HM_DIR / f"HM_{args.base_year}_AA_1000.tiff", ref)
        obs = read_like(HM_DIR / f"HM_{year}_AA_1000.tiff", ref)
        valid = np.isfinite(cen) & np.isfinite(hm0) & np.isfinite(obs)
        d_obs_full = np.where(valid, obs - hm0, np.nan)

        for wname, (r0, c0) in pick_windows(d_obs_full, size=args.size).items():
            sl = (slice(r0, r0 + args.size), slice(c0, c0 + args.size))
            win = (r0, r0 + args.size, c0, c0 + args.size)
            d_obs = d_obs_full[sl]
            d_cen = np.where(valid, cen - hm0, np.nan)[sl]
            h0 = hm0[sl]
            A = np.stack([agg.member_slice(sa, aa, m, hi, window=win) - h0 for m in mids])
            B = np.stack([agg.member_slice(sb, ab, m, hi, window=win) - h0 for m in mids])

            step = max(1, args.size // 900)
            vmax = max(float(np.nanpercentile(np.abs(d_obs), 99.5)), 0.02)

            # ---- view 1 + 2: signed field on top, thresholded below -------------------
            ncol = 2 + len(mids)
            fig, axes = plt.subplots(3, ncol, figsize=(3.0 * ncol, 9.6))
            panels = [("observed Δ", d_obs), ("central forecast Δ", d_cen)]
            for i, m in enumerate(mids):
                panels.append((f"{args.label_a}\nmember {m}", A[i]))
            for ax, (t, arr) in zip(axes[0], panels):
                ax.imshow(arr[::step, ::step], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          interpolation="nearest")
                ax.set_title(t, fontsize=8)
                ax.axis("off")
            panels_b = [("observed Δ", d_obs), ("central forecast Δ", d_cen)]
            for i, m in enumerate(mids):
                panels_b.append((f"{args.label_b}\nmember {m}", B[i]))
            for ax, (t, arr) in zip(axes[1], panels_b):
                ax.imshow(arr[::step, ::step], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          interpolation="nearest")
                ax.set_title(t, fontsize=8)
                ax.axis("off")

            # Thresholded: the quantity the failing targets actually score.
            def frac(a, thr, sign=1):
                v = a[np.isfinite(a)]
                return float((v > thr).mean()) if sign > 0 else float((v < thr).mean())

            thr_panels = [
                ("observed", d_obs), ("central", d_cen),
                (args.label_a, A[0]), (args.label_b, B[0]),
            ]
            for ax, (t, arr) in zip(axes[2], thr_panels):
                m = (arr[::step, ::step] > 0.05).astype(float)
                ax.imshow(m, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"{t}: Δ > +0.05\n{100*frac(arr, 0.05):.2f}% of px", fontsize=8)
                ax.axis("off")
            for ax in axes[2][len(thr_panels):]:
                ax.axis("off")

            fig.suptitle(
                f"{wname} · target {year} — {args.label_a} (row 1) vs {args.label_b} (row 2)\n"
                f"row 3: thresholded at +0.05, the quantity T6.1/T8.1 score",
                fontsize=11)
            fig.tight_layout()
            p = out_dir / f"marginal_{wname}_{year}.png"
            fig.savefig(p, dpi=125)
            plt.close(fig)
            figures[f"marginal/{wname}_{year}"] = str(p)

            # ---- view 3: change distribution ------------------------------------------
            fig2, ax = plt.subplots(figsize=(6.4, 4.2))
            bins = np.linspace(-3 * vmax, 3 * vmax, 121)
            for arr, lbl, st in ((d_obs, "observed", dict(color="k", lw=2.0)),
                                 (A.ravel(), args.label_a, dict(color="C3", lw=1.5)),
                                 (B.ravel(), args.label_b, dict(color="C0", lw=1.5)),
                                 (d_cen, "central forecast", dict(color="0.5", lw=1.0, ls=":"))):
                v = arr[np.isfinite(arr)]
                if v.size == 0:
                    continue
                h, edges = np.histogram(v, bins=bins, density=True)
                ax.step(0.5 * (edges[1:] + edges[:-1]), np.maximum(h, 1e-6), where="mid",
                        label=lbl, **st)
            ax.set_yscale("log")
            ax.set_xlabel("Δ HM")
            ax.set_ylabel("density (log)")
            ax.set_title(f"{wname} · {year}: change distribution", fontsize=10)
            ax.legend(fontsize=8, frameon=False)
            fig2.tight_layout()
            p2 = out_dir / f"marginal_dist_{wname}_{year}.png"
            fig2.savefig(p2, dpi=125)
            plt.close(fig2)
            figures[f"marginal_dist/{wname}_{year}"] = str(p2)

            rows.append({
                "window": wname, "year": year,
                "obs_frac_gt_0.05": frac(d_obs, 0.05),
                f"{args.label_a}_frac_gt_0.05": frac(A, 0.05),
                f"{args.label_b}_frac_gt_0.05": frac(B, 0.05),
                "obs_frac_lt_-0.01": frac(d_obs, -0.01, -1),
                f"{args.label_a}_frac_lt_-0.01": frac(A, -0.01, -1),
                f"{args.label_b}_frac_lt_-0.01": frac(B, -0.01, -1),
            })
            print(f"  {wname} {year}: Δ>+0.05 obs {100*frac(d_obs,0.05):.2f}% | "
                  f"{args.label_a} {100*frac(A,0.05):.2f}% | {args.label_b} {100*frac(B,0.05):.2f}%"
                  f"   ||   Δ<-0.01 obs {100*frac(d_obs,-0.01,-1):.2f}% | "
                  f"{args.label_a} {100*frac(A,-0.01,-1):.2f}% | {args.label_b} {100*frac(B,-0.01,-1):.2f}%")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "marginal_render_stats.csv", index=False)

    if run is not None:
        import wandb
        run.log({k: wandb.Image(v) for k, v in figures.items()})
        run.log({"marginal/stats": wandb.Table(dataframe=df)})
        print(f"\n✓ logged {len(figures)} figures to {run.url}")
        run.finish()
    print(f"✓ {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
