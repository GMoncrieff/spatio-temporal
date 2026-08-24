#!/usr/bin/env python3
"""One figure per site for the forward product: how the forecast grows with lead time.

    row per horizon (2025 / 2030 / 2035 / 2040), five columns:

        central Δ | lower 2.5% Δ | upper 97.5% Δ | interval width | one member's Δ

**Why this is not `plot_forecast_panel.py`.** That figure is built around the observation —
row 1 opens with observed Δ and row 3 overlays the observed histogram on every member. A
forward forecast has no observation, so those panels cannot be drawn and the comparison the
figure exists to make does not exist. Deleting them would leave a hindcast panel with holes
in it, which reads as a rendering failure rather than as a forecast.

What can be judged without truth is *how the product behaves with lead time*, and that is
what the rows are for: the central forecast should grow, the interval should widen, and the
widening should be monotone — the last is T4.2, which is enforced per pixel by
`apply_recalibration.enforce_horizon_monotonicity` and is therefore visible here as a
property that must hold in every panel of column 4.

**One member, not five.** Column 5 shows the *same* member at all four horizons, chosen as
the highest-change member at the longest lead. A member is one story about the future, and
the horizons within it are coupled by the measured AR(1) ρ — so a member that develops fast
by 2040 must already be developing by 2025. Showing five different members at one horizon
(what the hindcast panel does) answers "how wide is the ensemble"; showing one member across
horizons answers "is a member a coherent story", which is the property ρ exists to create
and the one a forward product depends on.

**Sites are picked on predicted change**, since there is no observed field to rank them by.
The ranking uses Δ at the longest horizon, so "busiest" means the model expects the most
development there.

Scales: **one per column, shared down the rows**, each set from the longest horizon. Rows
must share or growth with lead time is normalised away — that is the comparison the figure
exists to make. Columns must not: the central Δ and the upper bound differ by a factor of
three or more, so a single scale set from the central field drives the bound and member
columns to solid colour at the long leads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.ensemble import aggregate as agg  # noqa: E402

from compare_marginal_renders import HM_DIR, read_like  # noqa: E402
from plot_forecast_panel import spread_windows  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--recal_dir", required=True)
    ap.add_argument("--label", default="global forecast")
    ap.add_argument("--base_year", type=int, default=2020,
                    help="Year the change is measured from (the last observed HM)")
    ap.add_argument("--years", default="2025,2030,2035,2040")
    ap.add_argument("--pattern", default="prediction_{year}_{q}_recal.tif",
                    help="Production rasters carry no w{base}_ prefix; hindcast ones do")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--n_sites", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--wandb_project", default="spatio-temporal-convlstm")
    ap.add_argument("--wandb_group", default="forecast-horizons")
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--disable_wandb", action="store_true")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = [int(y) for y in args.years.split(",")]
    recal = Path(args.recal_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store, attrs = agg.open_ensemble(args.ensemble)
    M = int(store.shape[0])
    if store.shape[1] < len(years):
        raise SystemExit(f"ensemble carries {store.shape[1]} horizons, {len(years)} requested")
    print(f"{args.label}: M={M}, {len(years)} horizons from {args.ensemble}")

    def qpath(year, q):
        return recal / args.pattern.format(base=args.base_year, year=year, q=q)

    # Reference grid and the baseline the change is measured from.
    with rasterio.open(qpath(years[0], "central")) as s:
        ref = {"transform": s.transform, "width": s.width, "height": s.height}
    hm0_full = read_like(HM_DIR / f"HM_{args.base_year}_AA_1000.tiff", ref)

    # Load every horizon once: four rows x three quantiles is small enough to hold, and the
    # scales have to be set from the longest lead before anything is drawn.
    fields = {}
    for year in years:
        cen = read_like(qpath(year, "central"), ref)
        low = read_like(qpath(year, "lower"), ref)
        upp = read_like(qpath(year, "upper"), ref)
        valid = np.isfinite(cen) & np.isfinite(hm0_full)
        fields[year] = {
            "d_cen": np.where(valid, cen - hm0_full, np.nan),
            "d_low": np.where(valid, low - hm0_full, np.nan),
            "d_upp": np.where(valid, upp - hm0_full, np.nan),
            "width": np.where(valid, upp - low, np.nan),
        }
        print(f"  {year}: loaded")

    # No observation to rank sites by, so rank on predicted change at the longest lead.
    sites = spread_windows(fields[years[-1]]["d_cen"], size=args.size, n=args.n_sites,
                           seed=args.seed)
    print(f"  sites (ranked on predicted Δ at {years[-1]}): "
          f"{ {k: v for k, v in sites.items()} }")

    run = None
    if not args.disable_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, group=args.wandb_group,
                         job_type="forecast-horizons",
                         name=args.wandb_run_name or f"forecast horizons — {args.label}",
                         config={**vars(args), "members": M})

    written = []
    for sname, (r0, c0) in sites.items():
        sl = (slice(r0, r0 + args.size), slice(c0, c0 + args.size))
        win = (r0, r0 + args.size, c0, c0 + args.size)
        h0 = hm0_full[sl]

        # One member, followed across horizons: pick the extreme at the longest lead, then
        # show that same index everywhere so the AR(1) coupling is what the column displays.
        li = len(years) - 1
        means = np.full(M, np.nan)
        for m in range(M):
            means[m] = float(np.nanmean(agg.member_slice(store, attrs, m, li, window=win) - h0))
        mi = int(np.nanargmax(means))
        print(f"  {sname}: tracking member {mi} (mean Δ {means[mi]:+.5f} at {years[li]}, "
              f"rank {M}/{M})")

        # One scale per COLUMN, shared down the rows.
        #
        # Rows are the comparison this figure exists to make, so they must share a scale or
        # growth with lead time is normalised away. Columns must not: the central Δ reaches
        # ~0.05 here while the upper bound reaches ~0.17, so a single scale set from the
        # central field drives the bound and member columns to solid colour at the long
        # leads -- three of five columns carrying no information, which is what the first
        # render of this figure did. Each column is a different quantity with its own range.
        step = max(1, args.size // 900)
        mem_last = agg.member_slice(store, attrs, mi, li, window=win) - h0
        col_src = {
            0: fields[years[li]]["d_cen"][sl],
            1: fields[years[li]]["d_low"][sl],
            2: fields[years[li]]["d_upp"][sl],
            3: fields[years[li]]["width"][sl],
            4: mem_last,
        }
        col_vmax = {c: max(float(np.nanpercentile(np.abs(a), 99.5)), 0.02)
                    for c, a in col_src.items()}

        nrow, ncol = len(years), 5
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.35 * nrow))
        axes = np.atleast_2d(axes)

        for ri, year in enumerate(years):
            f = fields[year]
            mem = mem_last if ri == li else agg.member_slice(store, attrs, mi, ri,
                                                              window=win) - h0
            panels = [
                (f["d_cen"][sl], "central Δ", "div"),
                (f["d_low"][sl], "lower 2.5% Δ", "div"),
                (f["d_upp"][sl], "upper 97.5% Δ", "div"),
                (f["width"][sl], "interval width", "seq"),
                (mem, f"member {mi} Δ", "div"),
            ]
            for ci, (arr, title, kind) in enumerate(panels):
                ax = axes[ri, ci]
                v = col_vmax[ci]
                if kind == "div":
                    im = ax.imshow(arr[::step, ::step], cmap="RdBu_r",
                                   vmin=-v, vmax=v, interpolation="nearest")
                else:
                    im = ax.imshow(arr[::step, ::step], cmap="magma",
                                   vmin=0.0, vmax=v, interpolation="nearest")
                ax.axis("off")
                if ri == 0:
                    ax.set_title(title, fontsize=10)
                if ci == 0:
                    ax.text(-0.06, 0.5, f"{year}\n+{year - args.base_year} yr",
                            transform=ax.transAxes, rotation=90, va="center", ha="center",
                            fontsize=10)
                mean_v = float(np.nanmean(arr))
                ax.set_xlabel("")
                ax.text(0.5, -0.045, f"mean {mean_v:+.4f}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=7.5)
                if ri == nrow - 1:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.85)

        fig.suptitle(
            f"{args.label} — {sname} site, {args.size}x{args.size} px @ (row {r0}, col {c0})\n"
            f"change from HM {args.base_year}; member {mi} is the same story at every lead "
            f"(horizons coupled by measured AR(1) ρ)",
            fontsize=11)
        fig.tight_layout(rect=(0.01, 0.0, 1, 0.95))
        p = out_dir / f"forecast_horizons_{sname}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)
        print(f"  ✓ {p}")
        if run is not None:
            import wandb
            run.log({f"forecast_horizons/{sname}": wandb.Image(str(p))})

    if run is not None:
        run.finish()
    print(f"\n✓ {len(written)} figures -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
