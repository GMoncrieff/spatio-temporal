"""Fold-mosaic stitching -- the one piece of the deleted ensemble layer the model phase needs.

``stitch_fold_predictions`` joins the k=5 fold predictions into one raster. Two modes, and the
distinction is load-bearing: ``holdout`` takes each pixel from the fold that did NOT train on
it (the only mode anything scored may use), while ``mean`` averages every fold (seamless, but
in-sample, so it is for display rasters and products only). Never score a mean-stitched raster.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

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
    # Multi-band and integer sources pass straight through: the quantile-function raster the
    # distributional head writes is 64 int16 bands, and the fold select is per pixel and
    # entirely band-agnostic. Single-band float sources keep the historical float32/NaN
    # profile exactly, so the scored product is untouched.
    n_bands = int(ref.count)
    is_float = np.issubdtype(np.dtype(ref.dtypes[0]), np.floating)
    out_dtype = "float32" if is_float else ref.dtypes[0]
    out_nodata = nodata if is_float else ref.nodata
    profile.update(dtype=out_dtype, count=n_bands, nodata=out_nodata,
                   compress="deflate", BIGTIFF="YES")
    for f, s in srcs.items():
        if (s.height, s.width) != (ref.height, ref.width) or s.transform != ref.transform:
            raise ValueError(f"Fold {f} raster geometry differs from fold {fold_ids[0]}")

    # Dataset tags and band descriptions travel with the data. The quantile-function raster
    # carries its u grid in a tag, and that tag *is* the contract: without it the levels the
    # bands stand for are unknown, and a reader that guessed them would silently score a
    # different grid from the one the model wrote. Nothing else in this pipeline had tags, so
    # the omission was invisible until the first multi-band stitch.
    src_tags = {k: v for k, v in ref.tags().items() if k != "AREA_OR_POINT"}
    src_descs = list(ref.descriptions)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with rasterio.open(fold_mask_path) as mask_src:
        mask_win, row_off, col_off = _aligned_window(profile, mask_src)
        with rasterio.open(out_path, "w", **profile) as dst:
            if src_tags:
                dst.update_tags(**src_tags)
            for bi, desc in enumerate(src_descs):
                if desc:
                    dst.set_band_description(bi + 1, desc)
            fill = np.nan if is_float else out_nodata

            def _valid(a):
                return np.isfinite(a) if is_float else (a != out_nodata)

            for r0 in range(0, ref.height, block_rows):
                rows = min(block_rows, ref.height - r0)
                win = Window(0, r0, ref.width, rows)
                out = np.full((n_bands, rows, ref.width), fill, dtype=out_dtype)
                fmask = mask_src.read(
                    1, window=Window(col_off, row_off + r0, ref.width, rows), boundless=True, fill_value=0
                )
                if mode == "mean":
                    acc = np.zeros((n_bands, rows, ref.width), dtype=np.float64)
                    cnt = np.zeros((n_bands, rows, ref.width), dtype=np.int32)
                    for f in fold_ids:
                        vals = srcs[f].read(window=win)
                        ok = _valid(vals)
                        acc[ok] += vals[ok]
                        cnt[ok] += 1
                    got = cnt > 0
                    mean_vals = acc[got] / cnt[got]
                    out[got] = (mean_vals if is_float
                                else np.rint(mean_vals)).astype(out_dtype)
                else:
                    for f in fold_ids:
                        sel = fmask == f
                        if not sel.any():
                            continue
                        vals = srcs[f].read(window=win)
                        out[:, sel] = vals[:, sel]
                # Counted on the first band so the number keeps its historical meaning
                # (valid pixels), rather than becoming valid pixels times band count.
                n_written += int(_valid(out[0]).sum())
                dst.write(out, window=win)
    for s in srcs.values():
        s.close()
    return {"path": str(out_path), "n_valid_px": n_written, "mode": mode}


# --------------------------------------------------------------------------------------
# Residuals + class covariates
# --------------------------------------------------------------------------------------
