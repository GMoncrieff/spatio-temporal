#!/usr/bin/env python3
"""What the failing realism targets look like, side by side with the observation.

T6.1, T6.5, T8.1 and T7.3 all fail, and the scorecard reports them as four separate rows.
They are one defect seen four ways: the members carry more small-to-moderate change than
the world does, scattered over ground the observation leaves flat.

The renders in ``validate_ensemble.py`` show the raw change fields, where a signed colour
map makes a 0.02 change and a 0.2 change look similar. These panels threshold instead —
"did this pixel change by more than X" — which is the quantity the targets are actually
about, and the excess becomes obvious rather than inferable.
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
from src.ensemble.validate import DIST_LABELS, distance_band  # noqa: E402

HM_DIR = Path("data/raw/hm_global")


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
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--central", required=True, help="central raster for the target year")
    ap.add_argument("--base_year", type=int, default=2000)
    ap.add_argument("--target_year", type=int, default=2020)
    ap.add_argument("--horizon_idx", type=int, default=3)
    ap.add_argument("--dist_raster", default=None)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    store, attrs = agg.open_ensemble(args.ensemble)
    with rasterio.open(args.central) as s:
        ref = {"transform": s.transform, "width": s.width, "height": s.height}
        cen = s.read(1).astype(np.float64)
    hm0 = read_like(HM_DIR / f"HM_{args.base_year}_AA_1000.tiff", ref)
    obs = read_like(HM_DIR / f"HM_{args.target_year}_AA_1000.tiff", ref)
    valid = np.isfinite(cen) & np.isfinite(hm0) & np.isfinite(obs)

    d_obs = np.where(valid, obs - hm0, np.nan)
    d_cen = np.where(valid, cen - hm0, np.nan)
    n_mem = min(args.members, store.shape[0])
    mem = np.stack([agg.member_slice(store, attrs, m, args.horizon_idx) - hm0
                    for m in range(n_mem)])
    mem = np.where(valid[None], mem, np.nan)

    step = max(1, max(d_obs.shape) // 1100)

    def frac(a, thr, sign=1):
        v = a[np.isfinite(a)]
        return float((v > thr).mean()) if sign > 0 else float((v < thr).mean())

    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.4))

    def show(ax, mask, title, colour):
        m = np.where(np.isfinite(mask), mask, np.nan)
        ax.imshow(np.isfinite(m) & (m > 0), cmap=colour, vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Row 1 — moderate increase, the band the targets actually fail in.
    for ax, (lbl, arr) in zip(
            axes[0],
            [("observed", d_obs), ("central forecast", d_cen),
             ("member 0", mem[0]), ("member 1", mem[1])]):
        sub = arr[::step, ::step]
        show(ax, (sub > 0.05).astype(float), f"{lbl}: Δ > +0.05\n"
             f"{100*frac(arr, 0.05):.2f}% of pixels", "Reds")

    # Row 2 — small decreases, where T6.1 fails hardest.
    for ax, (lbl, arr) in zip(
            axes[1],
            [("observed", d_obs), ("central forecast", d_cen),
             ("member 0", mem[0]), ("member 1", mem[1])]):
        sub = arr[::step, ::step]
        show(ax, (sub < -0.01).astype(float), f"{lbl}: Δ < −0.01\n"
             f"{100*frac(arr, -0.01, sign=-1):.2f}% of pixels", "Blues")

    fig.suptitle(
        f"Where the members put change that the world does not — target {args.target_year}\n"
        "Top: moderate increase (T8.1). Bottom: small decrease (T6.1). "
        "Percentages are the share of valid pixels.", fontsize=11)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"✓ {args.out}")

    # The same story as a table, by distance to past change — T8.1's actual cut.
    if args.dist_raster and Path(args.dist_raster).exists():
        dist = read_like(args.dist_raster, ref, band=1)
        if not np.isfinite(dist).any():
            dist = read_like(args.dist_raster, ref, band=2)
        idx = distance_band(dist)
        print(f"\n{'band(px)':>9} {'n':>10} {'obs P(Δ>.05)':>13} {'mem P(Δ>.05)':>13} {'ratio':>7}"
              f" {'obs P(Δ<-.01)':>14} {'mem P(Δ<-.01)':>14} {'ratio':>7}")
        for b, lbl in enumerate(DIST_LABELS):
            sel = (idx == b) & valid
            if sel.sum() < 500:
                continue
            o_hi = float((d_obs[sel] > 0.05).mean())
            m_hi = float((mem[:, sel] > 0.05).mean())
            o_lo = float((d_obs[sel] < -0.01).mean())
            m_lo = float((mem[:, sel] < -0.01).mean())
            print(f"{lbl:>9} {int(sel.sum()):>10,} {o_hi:>13.5f} {m_hi:>13.5f} "
                  f"{(m_hi/o_hi if o_hi>0 else np.inf):>7.2f} {o_lo:>14.5f} {m_lo:>14.5f} "
                  f"{(m_lo/o_lo if o_lo>0 else np.inf):>7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
