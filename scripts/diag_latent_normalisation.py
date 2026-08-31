#!/usr/bin/env python3
"""D3 — are the members and their null on the same latent scale?

T3.2 and T3.3 score the correlated ensemble *against* the independent-pixel null. The
comparison is only about correlation if the two share a marginal, and on this pipeline they
do not quite:

  * a correlated member's latent field is standardised per realisation --
    ``field = (field - field.mean()) / field.std()`` (`src/ensemble/fields.py:166-170`) --
    with the moments taken over the **full H x W grid**, 78% of which is invalid on this
    holdout-stitched hindcast. The valid subset is therefore not standard normal.
  * the null draws ``torch.randn`` on the valid pixels directly
    (`scripts/generate_ensemble.py:406-407`) and is **not standardised at all**.

T3.1 already reports the first number as 1.117 against a 1.0 +/- 0.15 contract. What it does
not report is the second, and the difference between them sits inside the denominator of the
row that fails hardest. A member ensemble 12% wider than its own null loses variogram score
for a reason that has nothing to do with spatial structure.

This measures both on the same pixels, through the same quantile function, sharing no code
with the generator.

    scripts/diag_latent_normalisation.py
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
from validate_ensemble import qf_recover_z  # noqa: E402

QF_INT16_SCALE = 1.0 / 32767.0


def latent_variance(store, attrs, hi, idx, H, W, u, q, n_members, label):
    """Mean and variance of the recovered normal scores, per member and pooled."""
    X = agg.members_at_points(store, attrs, hi, idx, H, W)[:n_members]
    per_var, per_mean, keep = [], [], np.isfinite(q[0]) & np.isfinite(q[-1])
    for m in range(X.shape[0]):
        v = X[m].astype(np.float64)
        good = keep & np.isfinite(v)
        if good.sum() < 100:
            continue
        z = qf_recover_z(v[good], u, q[:, good])
        z = z[np.isfinite(z)]
        per_var.append(z.var())
        per_mean.append(z.mean())
    per_var, per_mean = np.array(per_var), np.array(per_mean)
    print(f"  {label:>28}  var {per_var.mean():.4f} +/- {per_var.std():.4f}   "
          f"mean {per_mean.mean():+.4f} +/- {per_mean.std():.4f}   "
          f"(M={len(per_var)}, n={int(keep.sum())} px)")
    return per_var.mean()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    hdd = "/mnt/hdd1/spatio-temporal/data/ensemble/exp/af_e1_hind"
    ap.add_argument("--members", default=f"{hdd}/members_m400.icechunk")
    ap.add_argument("--null", default=f"{hdd}/null_m400.icechunk")
    ap.add_argument("--qf", default="data/ensemble/exp/af_e1_hind/stitched/"
                                    "w2000_prediction_2020_qf.tif")
    ap.add_argument("--horizon_index", type=int, default=3)
    ap.add_argument("--n_points", type=int, default=200_000)
    ap.add_argument("--n_members", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    store, attrs = agg.open_ensemble(args.members)
    nstore, nattrs = agg.open_ensemble(args.null)
    M, nH, H, W = store.shape
    print(f"members {store.shape}   null {nstore.shape}")

    # The qf raster is the marginal both ensembles were drawn from, so it is also the map
    # back to normal scores. Read once, then sample the same pixels from both stores.
    with rasterio.open(args.qf) as s:
        u = np.array([float(v) for v in s.tags()["u_levels"].split(",")], dtype=np.float64)
        q = s.read().astype(np.float32)
        nod = s.nodata
    q = (np.where(q == nod, np.nan, q) * np.float32(QF_INT16_SCALE)).reshape(q.shape[0], -1)

    rng = np.random.default_rng(args.seed)
    valid = np.flatnonzero(np.isfinite(q[0]) & np.isfinite(q[-1]))
    idx = np.sort(rng.choice(valid, size=min(args.n_points, valid.size), replace=False))
    qs = q[:, idx].astype(np.float64)
    del q

    print(f"\nlatent normal scores, recovered through Q at h={args.horizon_index}:")
    vm = latent_variance(store, attrs, args.horizon_index, idx, H, W, u, qs,
                         args.n_members, "correlated members")
    hn = min(args.horizon_index, nstore.shape[1] - 1)
    vn = latent_variance(nstore, nattrs, hn, idx, H, W, u, qs,
                         args.n_members, "independent-pixel null")

    print(f"\nT3.1 contract is 1.0 +/- 0.15 on the members; the null has no row at all.")
    print(f"  members {vm:.4f}   null {vn:.4f}   difference {vm - vn:+.4f}")
    print(f"\nDECISION (rule set before the measurement): > 0.05 apart\n"
          f"  measured {abs(vm - vn):.4f} -> "
          f"{'latent-normalisation variant TAKES a slot' if abs(vm - vn) > 0.05 else 'DOES NOT earn a slot'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
