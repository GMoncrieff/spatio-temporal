"""Combine k fold-CV prediction rasters into one raster per (window, horizon, quantile).

Lifted verbatim from the ensemble branch's ``src/ensemble/residuals.py``, which also held the
rank-Gaussian transform and the residual harness that the recalibration layer needed. Nothing
here depends on either, so the stitcher lives on its own: rasterio and numpy only.

Two modes, and the difference matters for how the output may be used.

``holdout`` is the honest one — each pixel takes the prediction from the fold whose training
excluded it, so the whole raster is out of sample. It is also a hard mosaic, and the join is
measurable: across a fold boundary at +20 yr the step against the step inside a fold is 1.08x
for the central field and the lower bound, and 2.01x for the upper, because the upper is the
one head the folds disagree on.

``mean`` averages every fold at every pixel. Seamless, and in sample at every pixel, since four
of five folds trained on any given location. Its interval is also narrower than any single
fold's, because averaging discards the between-fold spread rather than adding it. Display only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# Horizon offsets in years, in model channel order.
HORIZONS = (5, 10, 15, 20)
QUANTILES = ("lower", "central", "upper")


def _aligned_window(ref_profile, mask_src):
    """Window of ``mask_src`` matching the reference raster's extent (both on the HM grid)."""
    t = ref_profile["transform"]
    mt = mask_src.transform
    col_off = int(round((t.c - mt.c) / mt.a))
    row_off = int(round((t.f - mt.f) / mt.e))
    return Window(col_off, row_off, ref_profile["width"], ref_profile["height"]), row_off, col_off

def stitch_fold_predictions(
    fold_paths: dict,
    fold_mask_path: str,
    out_path: str,
    block_rows: int = 1024,
    nodata=np.nan,
    mode: str = "holdout",
):
    """Combine per-fold rasters into one raster, either out-of-sample or seamless.

    Each fold's raster covers the same extent (fold membership controlled *training* data,
    not prediction extent).

    ``mode="holdout"`` (default) keeps, at each pixel, the prediction from the fold whose
    training excluded it — ``fold f`` values where ``fold_mask == f``. This is what makes
    the hindcast honest and it is the only mode anything scored should use.

    It is also, unavoidably, a hard mosaic. The k=5 mask is a **128 px checkerboard**, so
    adjacent tiles come from different models, and wherever those models disagree the join
    is visible. Measured on Africa at +20 yr, the step across a fold boundary against the
    step inside a fold is 1.08x for the central field and for the lower bound — nothing —
    and **2.01x for the upper bound**, because the upper is the one head the five folds do
    not agree on (mean pairwise |fold_i - fold_j| of 0.0383 against 0.0053 central and
    0.0040 lower). A single fold's own blended raster is seamless at every period from 64
    to 1024 px, so neither the model nor the overlap blending is involved.

    ``mode="mean"`` averages every fold at every pixel instead. There is no mosaic and no
    seam, and no pixel is out of sample any more — which is exactly right for a *forward*
    product, where no geography was held out to begin with, and exactly wrong for scoring.
    Note the averaged interval is narrower than any single fold's would be, because it
    discards the between-fold spread rather than adding it.

    Parameters
    ----------
    fold_paths
        ``{fold_id: path}`` for a single (window, horizon, quantile) combination.
    mode
        ``"holdout"`` for evaluation, ``"mean"`` for a seamless product raster.
    """
    if mode not in ("holdout", "mean"):
        raise ValueError(f"unknown stitch mode {mode!r} (expected 'holdout' or 'mean')")
    fold_ids = sorted(fold_paths)
    srcs = {f: rasterio.open(fold_paths[f]) for f in fold_ids}
    ref = srcs[fold_ids[0]]
    profile = ref.profile.copy()
    profile.update(dtype="float32", count=1, nodata=nodata, compress="deflate", BIGTIFF="YES")
    for f, s in srcs.items():
        if (s.height, s.width) != (ref.height, ref.width) or s.transform != ref.transform:
            raise ValueError(f"Fold {f} raster geometry differs from fold {fold_ids[0]}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with rasterio.open(fold_mask_path) as mask_src:
        mask_win, row_off, col_off = _aligned_window(profile, mask_src)
        with rasterio.open(out_path, "w", **profile) as dst:
            for r0 in range(0, ref.height, block_rows):
                rows = min(block_rows, ref.height - r0)
                out = np.full((rows, ref.width), np.nan, dtype=np.float32)
                fmask = mask_src.read(
                    1, window=Window(col_off, row_off + r0, ref.width, rows), boundless=True, fill_value=0
                )
                if mode == "mean":
                    acc = np.zeros((rows, ref.width), dtype=np.float64)
                    cnt = np.zeros((rows, ref.width), dtype=np.int32)
                    for f in fold_ids:
                        vals = srcs[f].read(1, window=Window(0, r0, ref.width, rows))
                        ok = np.isfinite(vals)
                        acc[ok] += vals[ok]
                        cnt[ok] += 1
                    got = cnt > 0
                    out[got] = (acc[got] / cnt[got]).astype(np.float32)
                else:
                    for f in fold_ids:
                        sel = fmask == f
                        if not sel.any():
                            continue
                        vals = srcs[f].read(1, window=Window(0, r0, ref.width, rows))
                        out[sel] = vals[sel]
                n_written += int(np.isfinite(out).sum())
                dst.write(out, 1, window=Window(0, r0, ref.width, rows))
    for s in srcs.values():
        s.close()
    return {"path": str(out_path), "n_valid_px": n_written, "mode": mode}

