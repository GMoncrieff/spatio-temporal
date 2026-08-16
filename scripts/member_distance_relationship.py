#!/usr/bin/env python3
"""Does each *individual* member reproduce the observed change-vs-distance relationship?

The pooled density answers a weaker question than it appears to. Pooling members mixes
within-member spatial structure with across-member variance, so a pool can match the
observation while no single member does — or miss it while every member is fine. A member
is meant to be one plausible realisation of the world, so the relationship has to hold
*inside* each member.

For every member this computes, per distance-to-past-change band:

  * ``P(Delta > 0.05)`` and ``P(Delta < -0.01)`` — the T8.1/T8.4 quantities;
  * the near/far contrast ``P(0-1px) / P(3-10px)``, which is the *shape* of the decay and
    is what "change is clustered near past change" actually means.

The observation is a single number per band, so the test is whether it falls inside the
across-member spread — and where in it. The rank of the observation among the members is
reported directly: for a calibrated ensemble it is uniform on 0..M, and a rank pinned at 0
or M means every member is on the same side of the truth, which a mean ratio can hide.
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
from src.ensemble.validate import distance_band  # noqa: E402

HM_DIR = Path("data/raw/hm_global")
DIST_LABELS = ["0-1 px", "1-3 px", "3-10 px", "10-30 px", "30-100 px", ">100 px"]
MIN_PX = 2000


def read_like(path, ref, band=1):
    with rasterio.open(path) as s:
        co = int(round((ref["transform"].c - s.transform.c) / s.transform.a))
        ro = int(round((ref["transform"].f - s.transform.f) / s.transform.e))
        a = s.read(band, window=Window(co, ro, ref["width"], ref["height"]),
                   boundless=True, fill_value=np.nan).astype(np.float64)
    return np.where(a < -1e6, np.nan, a)


def band_stats(delta, dband, n_bands, hi_thr=0.05, lo_thr=-0.01):
    """P(>hi) and P(<lo) per band for one field."""
    hi = np.full(n_bands, np.nan)
    lo = np.full(n_bands, np.nan)
    for b in range(n_bands):
        sel = dband == b
        n = int(sel.sum())
        if n < MIN_PX:
            continue
        v = delta[sel]
        hi[b] = float((v > hi_thr).mean())
        lo[b] = float((v < lo_thr).mean())
    return hi, lo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensembles", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--recal_dir", required=True)
    ap.add_argument("--dist_raster", required=True)
    ap.add_argument("--dist_band", type=int, default=1)
    ap.add_argument("--base_year", type=int, default=2000)
    ap.add_argument("--years", default="2005,2010,2015,2020")
    ap.add_argument("--members", type=int, default=100)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--wandb_project", default="spatio-temporal-convlstm")
    ap.add_argument("--wandb_group", default="marginal-comparison")
    ap.add_argument("--wandb_run_name",
                    default="member-wise change vs distance to past change")
    ap.add_argument("--disable_wandb", action="store_true")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ens = []
    for spec in args.ensembles:
        lab, path = spec.split("=", 1)
        st, at = agg.open_ensemble(path)
        ens.append((lab, st, at))
        print(f"  {lab}: M={st.shape[0]}")

    years = [int(y) for y in args.years.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = len(DIST_LABELS)
    colours = {0: "C3", 1: "C0", 2: "C2"}

    run = None
    if not args.disable_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, group=args.wandb_group,
                         job_type="member-distance", name=args.wandb_run_name,
                         config={"ensembles": args.ensembles, "members": args.members})

    figures, rows = {}, []
    for hi_idx, year in enumerate(years):
        cen_path = Path(args.recal_dir) / f"w{args.base_year}_prediction_{year}_central_recal.tif"
        with rasterio.open(cen_path) as s:
            ref = {"transform": s.transform, "width": s.width, "height": s.height}
            cen = s.read(1).astype(np.float64)
        hm0 = read_like(HM_DIR / f"HM_{args.base_year}_AA_1000.tiff", ref)
        obs = read_like(HM_DIR / f"HM_{year}_AA_1000.tiff", ref)
        dist = read_like(args.dist_raster, ref, band=args.dist_band)
        valid = np.isfinite(cen) & np.isfinite(hm0) & np.isfinite(obs) & np.isfinite(dist)

        d_obs = (obs - hm0)[valid]
        d_cen = (cen - hm0)[valid]
        dband = distance_band(dist)[valid]

        o_hi, o_lo = band_stats(d_obs, dband, nb)
        c_hi, c_lo = band_stats(d_cen, dband, nb)

        per_family = {}
        for lab, st, at in ens:
            M = min(args.members, st.shape[0])
            m_hi = np.full((M, nb), np.nan)
            m_lo = np.full((M, nb), np.nan)
            for m in range(M):
                dm = (agg.member_slice(st, at, m, hi_idx) - hm0)[valid]
                m_hi[m], m_lo[m] = band_stats(dm, dband, nb)
            per_family[lab] = (m_hi, m_lo, M)
            print(f"  {year} {lab}: {M} members scored")

        # ---- figure: per-band member spread with the observation overlaid ---------------
        fams = list(per_family)
        fig, axes = plt.subplots(2, nb, figsize=(3.0 * nb, 8.0))
        for b in range(nb):
            for r, (stat, obs_v, cen_v, name) in enumerate(
                    ((0, o_hi, c_hi, "P(Δ > +0.05)"), (1, o_lo, c_lo, "P(Δ < −0.01)"))):
                ax = axes[r, b]
                if not np.isfinite(obs_v[b]):
                    ax.text(0.5, 0.5, "too few px", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="0.5")
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                data, labels = [], []
                for fi, lab in enumerate(fams):
                    arr = per_family[lab][stat][:, b]
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        data.append(arr); labels.append(lab)
                if data:
                    bp = ax.boxplot(data, labels=[l[:9] for l in labels], widths=0.55,
                                    showfliers=False, patch_artist=True)
                    for fi, patch in enumerate(bp["boxes"]):
                        patch.set_facecolor(colours.get(fi, "C7")); patch.set_alpha(0.45)
                ax.axhline(obs_v[b], color="k", lw=2.0, label="observed")
                ax.axhline(cen_v[b], color="0.55", lw=1.0, ls=":", label="central")
                ax.set_yscale("symlog", linthresh=1e-4)
                ax.tick_params(labelsize=7)
                if b == 0:
                    ax.set_ylabel(name, fontsize=9)
                    ax.legend(fontsize=7, frameon=False)
                if r == 0:
                    ax.set_title(DIST_LABELS[b], fontsize=9)
        fig.suptitle(
            f"Per-member change vs distance to past change — target {year}\n"
            "boxes are the spread across individual members; the black line is the single "
            "observed value each member should be a plausible draw around", fontsize=12)
        fig.tight_layout()
        p = out_dir / f"member_distance_{year}.png"
        fig.savefig(p, dpi=125)
        plt.close(fig)
        figures[f"member_distance/{year}"] = str(p)

        # ---- table: rank of the observation among the members --------------------------
        for lab in fams:
            m_hi, m_lo, M = per_family[lab]
            for b in range(nb):
                if not np.isfinite(o_hi[b]):
                    continue
                col = m_hi[:, b][np.isfinite(m_hi[:, b])]
                col_lo = m_lo[:, b][np.isfinite(m_lo[:, b])]
                if col.size == 0:
                    continue
                rank = int((col < o_hi[b]).sum())
                rows.append({
                    "year": year, "family": lab, "band": DIST_LABELS[b], "M": int(col.size),
                    "obs_P_hi": o_hi[b], "member_mean_P_hi": float(col.mean()),
                    "member_p5_P_hi": float(np.percentile(col, 5)),
                    "member_p95_P_hi": float(np.percentile(col, 95)),
                    "obs_rank_hi": rank, "obs_inside_90pct_hi":
                        bool(np.percentile(col, 5) <= o_hi[b] <= np.percentile(col, 95)),
                    "obs_P_lo": o_lo[b],
                    "member_mean_P_lo": float(col_lo.mean()) if col_lo.size else np.nan,
                    "obs_rank_lo": int((col_lo < o_lo[b]).sum()) if col_lo.size else -1,
                })

        # ---- the decay shape, per member -----------------------------------------------
        if np.isfinite(o_hi[0]) and np.isfinite(o_hi[2]) and o_hi[2] > 0:
            obs_ratio = o_hi[0] / o_hi[2]
            line = [f"  {year} near/far contrast P(0-1px)/P(3-10px): observed {obs_ratio:.1f}"]
            for lab in fams:
                m_hi, _, _ = per_family[lab]
                r = m_hi[:, 0] / np.maximum(m_hi[:, 2], 1e-12)
                r = r[np.isfinite(r)]
                line.append(f"| {lab} {np.median(r):.1f} [{np.percentile(r,5):.1f}, "
                            f"{np.percentile(r,95):.1f}]")
            print(" ".join(line))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "member_distance_stats.csv", index=False)

    print(f"\n{'year':>5} {'family':>17} {'band':>10} {'observed':>9} {'mem mean':>9} "
          f"{'[p5, p95]':>21} {'rank/M':>9} {'inside':>7}")
    for _, r in df.iterrows():
        print(f"{int(r.year):>5} {r.family:>17} {r.band:>10} {r.obs_P_hi:>9.5f} "
              f"{r.member_mean_P_hi:>9.5f} "
              f"{'[' + format(r.member_p5_P_hi, '.5f') + ', ' + format(r.member_p95_P_hi, '.5f') + ']':>21} "
              f"{str(r.obs_rank_hi) + '/' + str(r.M):>9} {'yes' if r.obs_inside_90pct_hi else 'NO':>7}")

    inside = df["obs_inside_90pct_hi"].mean() if len(df) else np.nan
    print(f"\nObservation inside the member 5-95% range in {100*inside:.0f}% of "
          f"(year x band x family) cells — a calibrated ensemble would give ~90%.")

    if run is not None:
        import wandb
        run.log({k: wandb.Image(v) for k, v in figures.items()})
        run.log({"member_distance/stats": wandb.Table(dataframe=df)})
        run.log({"member_distance/frac_inside_90pct": float(inside)})
        print(f"✓ logged {len(figures)} figures to {run.url}")
        run.finish()
    print(f"✓ {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
