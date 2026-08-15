#!/usr/bin/env python3
"""Change-distribution density, split by distance to past change.

The pooled Δ density says the members carry too much moderate change; it cannot say
*where*. Distance to past change is the covariate that decides whether change is possible
at all — measured on southern Africa, P(Δ>0.05) falls 0.222 → 0.080 → 0.023 → 0.0068 →
0.0010 → 0.0000 across the T8 bands — so splitting the density by that band shows whether a
marginal misplaces change or merely mis-scales it.

Each panel overlays, on a log density:

  * the observation, which is the target shape;
  * the central forecast, which carries no spread and so shows what the model alone says;
  * members from each marginal family being compared.

Bands with too few pixels are skipped rather than drawn as noise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble import aggregate as agg  # noqa: E402

HM_DIR = Path("data/raw/hm_global")
DIST_EDGES = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, np.inf]
DIST_LABELS = ["0-1 px", "1-3 px", "3-10 px", "10-30 px", "30-100 px", ">100 px"]
MIN_PX = 2000


def read_like(path, ref, band=1):
    with rasterio.open(path) as s:
        co = int(round((ref["transform"].c - s.transform.c) / s.transform.a))
        ro = int(round((ref["transform"].f - s.transform.f) / s.transform.e))
        a = s.read(band, window=Window(co, ro, ref["width"], ref["height"]),
                   boundless=True, fill_value=np.nan).astype(np.float64)
    return np.where(a < -1e6, np.nan, a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensembles", nargs="+", required=True,
                    help="label=path pairs, e.g. 'two-piece=...zarr' 'shaped=...zarr'")
    ap.add_argument("--recal_dir", required=True)
    ap.add_argument("--dist_raster", required=True)
    ap.add_argument("--dist_band", type=int, default=1)
    ap.add_argument("--base_year", type=int, default=2000)
    ap.add_argument("--years", default="2005,2010,2015,2020")
    ap.add_argument("--members", type=int, default=12,
                    help="Members pooled per family; the density needs mass in the tails")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--wandb_project", default="spatio-temporal-convlstm")
    ap.add_argument("--wandb_group", default="marginal-comparison")
    ap.add_argument("--wandb_run_name", default="change density by distance to past change")
    ap.add_argument("--disable_wandb", action="store_true")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ens = []
    for spec in args.ensembles:
        if "=" not in spec:
            raise SystemExit(f"expected label=path, got {spec!r}")
        lab, path = spec.split("=", 1)
        st, at = agg.open_ensemble(path)
        ens.append((lab, st, at))
        print(f"  {lab}: M={st.shape[0]} from {path}")

    years = [int(y) for y in args.years.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colours = ["C3", "C0", "C2", "C4"]

    run = None
    if not args.disable_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, group=args.wandb_group,
                         job_type="change-density", name=args.wandb_run_name,
                         config={"ensembles": args.ensembles, "members": args.members})

    figures, rows = {}, []
    for hi, year in enumerate(years):
        cen_path = Path(args.recal_dir) / f"w{args.base_year}_prediction_{year}_central_recal.tif"
        with rasterio.open(cen_path) as s:
            ref = {"transform": s.transform, "width": s.width, "height": s.height}
            cen = s.read(1).astype(np.float64)
        hm0 = read_like(HM_DIR / f"HM_{args.base_year}_AA_1000.tiff", ref)
        obs = read_like(HM_DIR / f"HM_{year}_AA_1000.tiff", ref)
        dist = read_like(args.dist_raster, ref, band=args.dist_band)
        valid = (np.isfinite(cen) & np.isfinite(hm0) & np.isfinite(obs) & np.isfinite(dist))

        d_obs = (obs - hm0)[valid]
        d_cen = (cen - hm0)[valid]
        dband = np.digitize(dist, DIST_EDGES[1:-1], right=True)[valid]

        fam = []
        for lab, st, at in ens:
            n = min(args.members, st.shape[0])
            stack = np.stack([(agg.member_slice(st, at, m, hi) - hm0)[valid] for m in range(n)])
            fam.append((lab, stack))

        fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6), sharex=True)
        drawn = 0
        for b, (ax, lbl) in enumerate(zip(axes.ravel(), DIST_LABELS)):
            sel = dband == b
            n_px = int(sel.sum())
            if n_px < MIN_PX:
                ax.text(0.5, 0.5, f"{lbl}\n{n_px:,} px — too few", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="0.4")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            drawn += 1
            o = d_obs[sel]
            lim = max(float(np.nanpercentile(np.abs(o), 99.9)), 0.05)
            bins = np.linspace(-lim, lim, 121)

            def draw(v, label, **st):
                v = v[np.isfinite(v)]
                if v.size == 0:
                    return
                h, e = np.histogram(v, bins=bins, density=True)
                ax.step(0.5 * (e[1:] + e[:-1]), np.maximum(h, 1e-6), where="mid",
                        label=label, **st)

            draw(o, "observed", color="k", lw=2.0)
            draw(d_cen[sel], "central forecast", color="0.55", lw=1.1, ls=":")
            rec = {"year": year, "band": lbl, "n_px": n_px,
                   "obs_P(>0.05)": float((o > 0.05).mean()),
                   "obs_P(<-0.01)": float((o < -0.01).mean())}
            for i, (lab, stack) in enumerate(fam):
                v = stack[:, sel]
                draw(v.ravel(), lab, color=colours[i % len(colours)], lw=1.4)
                rec[f"{lab}_P(>0.05)"] = float((v > 0.05).mean())
                rec[f"{lab}_P(<-0.01)"] = float((v < -0.01).mean())
            rows.append(rec)

            ax.set_yscale("log")
            ax.set_title(f"{lbl}  ({n_px:,} px)", fontsize=10)
            ax.tick_params(labelsize=8)
            if b == 0:
                ax.legend(fontsize=8, frameon=False)
            if b >= 3:
                ax.set_xlabel("Δ HM", fontsize=9)
            if b % 3 == 0:
                ax.set_ylabel("density (log)", fontsize=9)

        fig.suptitle(f"Change distribution by distance to past change — target {year}\n"
                     "change is a near-neighbour phenomenon; a marginal can mis-scale it "
                     "or misplace it, and only this split tells them apart", fontsize=12)
        fig.tight_layout()
        p = out_dir / f"density_by_distance_{year}.png"
        fig.savefig(p, dpi=125)
        plt.close(fig)
        figures[f"density_by_distance/{year}"] = str(p)
        print(f"  {year}: {drawn} bands drawn -> {p}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "density_by_distance.csv", index=False)

    # Ratios are what the T8 targets are stated in, so print them rather than raw rates.
    labs = [lab for lab, _, _ in ens]
    print(f"\n{'year':>5} {'band':>10} {'n_px':>10} {'obs P(>.05)':>12} " +
          " ".join(f"{l + ' ratio':>18}" for l in labs))
    for _, r in df.iterrows():
        cells = []
        for l in labs:
            o = r["obs_P(>0.05)"]
            cells.append(f"{(r[f'{l}_P(>0.05)'] / o if o > 0 else np.inf):>18.2f}")
        print(f"{int(r.year):>5} {r.band:>10} {int(r.n_px):>10,} {r['obs_P(>0.05)']:>12.5f} "
              + " ".join(cells))

    if run is not None:
        import wandb
        run.log({k: wandb.Image(v) for k, v in figures.items()})
        run.log({"density_by_distance/stats": wandb.Table(dataframe=df)})
        print(f"\n✓ logged {len(figures)} figures to {run.url}")
        run.finish()
    print(f"✓ {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
