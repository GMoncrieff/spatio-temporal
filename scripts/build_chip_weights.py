#!/usr/bin/env python3
"""Per-chip sampling weights from the input-window past-change rasters.

Stratified batch sampling needs a number per candidate chip position saying how much change
that chip has seen. It is computed **only from input-window covariates** -- band 1 of
``change_context_w{year}_1000.tif``, which the model already reads -- so the sampler cannot
see the target and there is no leak, whatever weight it ends up using.

Computed once for the whole grid and cached: the table is indexed by chip position, and every
fold draws from the same table, so a fold's own weights never depend on which fold it is.

Motivation, measured on the first baseline run: on the 75% of pixels whose observed 20-year
change is in [-0.01, 0.001) the model's CRPS is **worse than persistence by 1.25** and its 95%
interval covers 0.996 of observations, while real decreases sit at PIT 0.023 with 29% coverage
and real large increases at PIT 0.905 with 65%. The optimiser is spending itself on the quiet
majority. That is what this exists to change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

HM_DIR = Path("data/raw/hm_global")


def chip_max(path: Path, chip: int, block_chips: int = 16) -> np.ndarray:
    """Per-chip max |past change| on a ``chip``-px lattice, streamed by row block."""
    with rasterio.open(path) as src:
        H, W = src.height, src.width
        n_i, n_j = H // chip, W // chip
        out = np.zeros((n_i, n_j), dtype=np.float32)
        rows = chip * block_chips
        for bi in range(0, n_i, block_chips):
            r0 = bi * chip
            nr = min(rows, H - r0)
            m = nr // chip
            if m == 0:
                continue
            a = src.read(1, window=Window(0, r0, n_j * chip, m * chip))
            a = np.abs(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0))
            out[bi:bi + m] = a.reshape(m, chip, n_j, chip).max(axis=(1, 3))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chip_size", type=int, default=128)
    ap.add_argument("--windows", default="2000,2005,2010,2015")
    ap.add_argument("--context_pattern",
                    default=str(HM_DIR / "change_context_w{year}_1000.tif"))
    ap.add_argument("--out", default="data/ensemble/chip_weights_128.npz")
    args = ap.parse_args(argv)

    chip = int(args.chip_size)
    total = None
    for year in [int(y) for y in args.windows.split(",")]:
        p = Path(args.context_pattern.format(year=year))
        if not p.exists():
            raise SystemExit(f"missing {p}")
        w = chip_max(p, chip)
        # Max across input windows: a chip that was ever active is worth presenting, and the
        # training sampler draws its end year uniformly over all four windows anyway.
        total = w if total is None else np.maximum(total, w)
        print(f"  w{year}: chips {w.shape}, max {w.max():.4f}, "
              f"frac > 0.01 {(w > 0.01).mean():.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, w=total.astype(np.float32), chip_size=chip)
    nz = total[total > 0]
    print(f"\n✓ {out}  {total.shape}  mean {total.mean():.5f}  "
          f"median(nonzero) {np.median(nz) if nz.size else 0:.5f}  max {total.max():.4f}")
    for q in (0.5, 0.9, 0.99, 0.999):
        print(f"    q{q}: {np.quantile(total, q):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
