#!/usr/bin/env python3
"""One figure for the forecast product: what was published, what members look like, and how
each member's change distribution compares with the observation.

    row 1   observed Δ | central forecast Δ | lower 2.5% Δ | upper 97.5% Δ | (blank)
    row 2   3 random members | the highest-change member | the lowest-change member
    row 3   the change histogram of each member above, observed overlaid in every panel

Row 1 is the published product — the triple the ensemble is built to reproduce exactly
(T5.1/T5.2) — shown as *change* rather than level so it sits on the same scale as the
members. Row 2 is what one plausible realisation of the world looks like under the measured
marginal. Row 3 is the same members as distributions, because the eye cannot judge from a
map whether a field carries twice as much moderate change as reality: that is exactly the
defect this phase existed to fix, and it is visible in a histogram and invisible in a map.

The high- and low-change members are the extremes of the window's mean Δ across all M
members, so row 2 brackets the ensemble instead of sampling its front — three arbitrary
indices say nothing about spread, which is the failure mode the per-member diagnostics
exist to catch.

All map panels share one diverging colour scale, set from the observation. The published
bounds are wider than the observation and will saturate; that is the point of showing them.
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

from compare_marginal_renders import HM_DIR, pick_windows, read_like  # noqa: E402


def spread_windows(d_obs, size=768, n=3, n_candidates=400, seed=0):
    """``n`` windows spanning the observed-change spectrum, busiest first.

    ``pick_windows`` returns the two extremes, which is the right choice when the question
    is "where do the two regimes fail differently". When the question is instead "what does
    this model look like across the domain", two extremes are misleading — they are the
    tails. This samples candidate windows, ranks them by mean |Δ observed|, and returns
    evenly spaced quantiles of that ranking so the set brackets *and* fills the range.

    Candidates must be substantially on land: a window that is 90% ocean scores as quiet for
    a reason that has nothing to do with the forecast.
    """
    H, W = d_obs.shape
    rng = np.random.default_rng(seed)
    cands = []
    for _ in range(n_candidates):
        r0 = int(rng.integers(0, max(1, H - size)))
        c0 = int(rng.integers(0, max(1, W - size)))
        sub = d_obs[r0:r0 + size, c0:c0 + size]
        finite = np.isfinite(sub)
        if finite.mean() < 0.6:
            continue
        cands.append((float(np.nanmean(np.abs(sub))), (r0, c0)))
    if not cands:
        return pick_windows(d_obs, size=size)
    cands.sort(key=lambda t: -t[0])
    n = min(n, len(cands))
    idx = [int(round(i * (len(cands) - 1) / max(n - 1, 1))) for i in range(n)]
    names = ["busiest", "mid-change", "quietest"] if n == 3 else [f"q{i + 1}" for i in range(n)]
    return {names[i] if i < len(names) else f"w{i}": cands[j][1] for i, j in enumerate(idx)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--label", default="measured marginal")
    ap.add_argument("--recal_dir", required=True)
    ap.add_argument("--base_year", type=int, default=2000)
    ap.add_argument("--years", default="2005,2010,2015,2020")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0,
                    help="Picks the random members; fixed so the figure is reproducible.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_windows", type=int, default=0,
                    help="Instead of the default busiest/quietest pair, take this many "
                         "windows spanning the observed-change spectrum — busiest, then "
                         "evenly spaced quantiles down to the quietest. Use when the point "
                         "is to see the same model across several regimes.")
    ap.add_argument("--wandb_project", default="spatio-temporal-convlstm")
    ap.add_argument("--wandb_group", default="marginal-comparison")
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--disable_wandb", action="store_true")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = [int(y) for y in args.years.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store, attrs = agg.open_ensemble(args.ensemble)
    M = int(store.shape[0])
    print(f"{args.label}: M={M} from {args.ensemble}")

    run = None
    if not args.disable_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, group=args.wandb_group,
                         job_type="forecast-panel",
                         name=args.wandb_run_name or f"forecast panel — {args.label}",
                         config={**vars(args), "members": M})

    figures = {}
    for hi, year in enumerate(years):
        stem = f"w{args.base_year}_prediction_{year}"
        with rasterio.open(Path(args.recal_dir) / f"{stem}_central_recal.tif") as s:
            ref = {"transform": s.transform, "width": s.width, "height": s.height}
            cen = s.read(1).astype(np.float64)
        low = read_like(Path(args.recal_dir) / f"{stem}_lower_recal.tif", ref)
        upp = read_like(Path(args.recal_dir) / f"{stem}_upper_recal.tif", ref)
        hm0 = read_like(HM_DIR / f"HM_{args.base_year}_AA_1000.tiff", ref)
        obs = read_like(HM_DIR / f"HM_{year}_AA_1000.tiff", ref)
        valid = np.isfinite(cen) & np.isfinite(hm0) & np.isfinite(obs)
        d_obs_full = np.where(valid, obs - hm0, np.nan)

        if args.n_windows > 0:
            wins = spread_windows(d_obs_full, size=args.size, n=args.n_windows)
        else:
            wins = pick_windows(d_obs_full, size=args.size)
        for wname, (r0, c0) in wins.items():
            sl = (slice(r0, r0 + args.size), slice(c0, c0 + args.size))
            win = (r0, r0 + args.size, c0, c0 + args.size)
            h0 = hm0[sl]
            d_obs = d_obs_full[sl]
            d_cen = np.where(valid, cen - hm0, np.nan)[sl]
            d_low = np.where(valid, low - hm0, np.nan)[sl]
            d_upp = np.where(valid, upp - hm0, np.nan)[sl]

            # Rank every member by how much change it made in this window, so the two
            # extremes can bracket the three random draws.
            means = np.full(M, np.nan)
            for m in range(M):
                v = agg.member_slice(store, attrs, m, hi, window=win) - h0
                means[m] = float(np.nanmean(v))
            hi_m, lo_m = int(np.nanargmax(means)), int(np.nanargmin(means))
            rng = np.random.default_rng(args.seed)
            pool = [m for m in range(M) if m not in (hi_m, lo_m)]
            rand = sorted(rng.choice(pool, size=min(args.n_random, len(pool)),
                                     replace=False).tolist())
            picks = [(m, f"random member {i + 1}\n(index {m})")
                     for i, m in enumerate(rand)]
            picks.append((hi_m, f"highest-change member\n(index {hi_m}, rank {M}/{M})"))
            picks.append((lo_m, f"lowest-change member\n(index {lo_m}, rank 1/{M})"))
            fields = [agg.member_slice(store, attrs, m, hi, window=win) - h0
                      for m, _ in picks]

            step = max(1, args.size // 900)
            vmax = max(float(np.nanpercentile(np.abs(d_obs), 99.5)), 0.02)
            ncol = len(picks)
            fig, axes = plt.subplots(3, ncol, figsize=(3.2 * ncol, 10.2))

            def show(ax, arr, title):
                ax.imshow(arr[::step, ::step], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          interpolation="nearest")
                ax.set_title(title, fontsize=8)
                ax.axis("off")

            row1 = [(d_obs, "observed Δ"), (d_cen, "central forecast Δ"),
                    (d_low, "lower 2.5% Δ"), (d_upp, "upper 97.5% Δ")]
            for ax, (arr, t) in zip(axes[0], row1):
                show(ax, arr, t)
            for ax in axes[0][len(row1):]:
                ax.axis("off")

            for ax, arr, (_, t) in zip(axes[1], fields, picks):
                show(ax, arr, t)

            # Row 3: each member's change distribution against the observation's.
            bins = np.linspace(-3 * vmax, 3 * vmax, 121)
            ctr = 0.5 * (bins[1:] + bins[:-1])
            o = d_obs[np.isfinite(d_obs)]
            h_obs, _ = np.histogram(o, bins=bins, density=True)
            for ax, arr, (m, _) in zip(axes[2], fields, picks):
                v = arr[np.isfinite(arr)]
                h_mem, _ = np.histogram(v, bins=bins, density=True)
                ax.step(ctr, np.maximum(h_obs, 1e-6), where="mid", color="k", lw=2.0,
                        label="observed")
                ax.step(ctr, np.maximum(h_mem, 1e-6), where="mid", color="C0", lw=1.4,
                        label=f"member {m}")
                ax.set_yscale("log")
                ax.set_xlabel("Δ HM", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.set_title(f"Δ>+0.05: obs {100 * float((o > 0.05).mean()):.2f}% vs "
                             f"member {100 * float((v > 0.05).mean()):.2f}%", fontsize=8)
                ax.legend(fontsize=7, frameon=False)
            axes[2][0].set_ylabel("density (log)", fontsize=8)

            fig.suptitle(
                f"{wname} · target {year} · {args.label} (M={M})\n"
                f"row 1 published product · row 2 members · row 3 their change "
                f"distributions against the observation — maps share one scale set from the "
                f"observation, so the published bounds saturate", fontsize=11)
            fig.tight_layout()
            p = out_dir / f"forecast_panel_{wname}_{year}.png"
            fig.savefig(p, dpi=125)
            plt.close(fig)
            figures[f"forecast_panel/{wname}_{year}"] = str(p)
            print(f"  {wname} {year}: random {rand}, high {hi_m} "
                  f"(mean Δ {means[hi_m]:+.5f}), low {lo_m} ({means[lo_m]:+.5f})")

    if run is not None:
        import wandb
        run.log({k: wandb.Image(v) for k, v in figures.items()})
        print(f"✓ logged {len(figures)} figures to {run.url}")
        run.finish()
    print(f"✓ {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
