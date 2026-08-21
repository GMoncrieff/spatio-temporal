"""Phase 0 — out-of-sample residual harness.

Two things live here:

1. :class:`RankGaussianTransform` — a hurdle rank-Gaussian transform for HM. HM is bounded
   on [0, 1] with a real ~1.4% atom at exactly zero and a heavy right skew, so a plain
   logit is undefined at the endpoints and does nothing about the atom. Zeros are mapped
   to a deterministic-per-pixel uniform draw on (0, p0) rather than to a single constant,
   because a constant would put a degenerate spike at one z value and break the zero-lag
   behaviour of the Phase 1c variogram fit.

2. Fold stitching and residual computation — turning k fold-CV prediction rasters into one
   genuinely out-of-sample raster per (window, horizon, quantile), then into residuals in
   both raw HM units (what Phase 1.5 calibrates) and rank-Gaussian units (what Phase 1c/2
   model as a Gaussian field), plus the class-covariate rasters Phase 1.5 needs.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.stats import norm

# Horizon offsets in years, in model channel order.
HORIZONS = (5, 10, 15, 20)
QUANTILES = ("lower", "central", "upper")
EPS = 1e-6


# --------------------------------------------------------------------------------------
# Deterministic per-pixel uniforms
# --------------------------------------------------------------------------------------
def hash_uniform(index: np.ndarray, salt: int = 0) -> np.ndarray:
    """Deterministic uniform(0,1) draw keyed on an integer index (splitmix64).

    Used so a given pixel's zero-atom position is reproducible across calls and across
    processes, instead of being resampled every time the transform is applied.
    """
    idx = np.asarray(index, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = idx + np.uint64(0x9E3779B97F4A7C15) + np.uint64(salt & 0xFFFFFFFFFFFFFFFF)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def pixel_index(rows: np.ndarray, cols: np.ndarray, width: int) -> np.ndarray:
    """Global pixel index for hashing: row * width + col."""
    return np.asarray(rows, dtype=np.int64) * np.int64(width) + np.asarray(cols, dtype=np.int64)


def window_index_grid(row_off: int, col_off: int, height: int, width: int, raster_width: int):
    """Index grid for a raster window, in the same global numbering as :func:`pixel_index`."""
    rr = np.arange(row_off, row_off + height, dtype=np.int64)[:, None]
    cc = np.arange(col_off, col_off + width, dtype=np.int64)[None, :]
    return rr * np.int64(raster_width) + cc


# --------------------------------------------------------------------------------------
# Rank-Gaussian transform with explicit zero-atom handling
# --------------------------------------------------------------------------------------
@dataclass
class RankGaussianTransform:
    """Hurdle rank-Gaussian transform for HM values on [0, 1].

    Attributes
    ----------
    p0
        P(X == 0), the exact-zero atom mass.
    knot_probs, knot_values
        Interpolated empirical CDF of the *positive* part, F_pos.
    """

    p0: float
    knot_probs: np.ndarray
    knot_values: np.ndarray

    # ---- construction -----------------------------------------------------------------
    @classmethod
    def fit(cls, sample: np.ndarray, n_knots: int = 1500) -> "RankGaussianTransform":
        x = np.asarray(sample, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("Cannot fit RankGaussianTransform on an empty sample")
        n_total = x.size
        pos = x[x > 0]
        p0 = float((n_total - pos.size) / n_total)
        if pos.size < 2:
            raise ValueError("Sample has fewer than 2 positive values; cannot fit F_pos")

        probs = np.linspace(0.0, 1.0, int(n_knots))
        values = np.quantile(pos, probs)
        # Collapse ties so np.interp sees a strictly increasing abscissa; keep the mean
        # probability of each tied group (the standard mid-rank convention).
        uniq, inverse = np.unique(values, return_inverse=True)
        if uniq.size < values.size:
            summed = np.bincount(inverse, weights=probs, minlength=uniq.size)
            counts = np.bincount(inverse, minlength=uniq.size)
            probs = summed / counts
            values = uniq
        return cls(p0=p0, knot_probs=np.asarray(probs), knot_values=np.asarray(values))

    # ---- serialization ----------------------------------------------------------------
    def to_json(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "p0": self.p0,
                    "knot_probs": self.knot_probs.tolist(),
                    "knot_values": self.knot_values.tolist(),
                },
                f,
            )

    @classmethod
    def from_json(cls, path) -> "RankGaussianTransform":
        with open(path) as f:
            d = json.load(f)
        return cls(
            p0=float(d["p0"]),
            knot_probs=np.asarray(d["knot_probs"], dtype=np.float64),
            knot_values=np.asarray(d["knot_values"], dtype=np.float64),
        )

    # ---- transform --------------------------------------------------------------------
    def cdf(self, x: np.ndarray, index: np.ndarray | None = None) -> np.ndarray:
        """Uniform-scale image of ``x`` (the copula scale)."""
        x = np.asarray(x, dtype=np.float64)
        u = np.empty_like(x)
        finite = np.isfinite(x)
        zero = finite & (x <= 0)
        posm = finite & (x > 0)

        if index is None:
            # Fall back to the atom midpoint; only sensible for one-off summaries, not for
            # residual fields (see the class docstring).
            u[zero] = self.p0 / 2.0
        else:
            idx = np.asarray(index)
            u[zero] = np.clip(hash_uniform(idx[zero]) * self.p0, EPS, self.p0 - EPS if self.p0 > 2 * EPS else self.p0 / 2)

        f_pos = np.interp(x[posm], self.knot_values, self.knot_probs, left=0.0, right=1.0)
        u[posm] = self.p0 + (1.0 - self.p0) * f_pos
        u[~finite] = np.nan
        return np.clip(u, EPS, 1.0 - EPS)

    def forward(self, x: np.ndarray, index: np.ndarray | None = None) -> np.ndarray:
        """HM values -> approximately standard-normal scores."""
        return norm.ppf(self.cdf(x, index=index))

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Standard-normal scores -> HM values on [0, 1]."""
        z = np.asarray(z, dtype=np.float64)
        u = norm.cdf(z)
        return self.inverse_cdf(u)

    def inverse_cdf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        x = np.empty_like(u)
        finite = np.isfinite(u)
        atom = finite & (u <= self.p0)
        posm = finite & (u > self.p0)
        x[atom] = 0.0
        v = (u[posm] - self.p0) / max(1.0 - self.p0, EPS)
        x[posm] = np.interp(
            v, self.knot_probs, self.knot_values,
            left=self.knot_values[0], right=self.knot_values[-1],
        )
        x[~finite] = np.nan
        return np.clip(x, 0.0, 1.0)


def sample_hm_values(raster_path, n_windows: int = 400, window_size: int = 256, random_seed: int = 42):
    """Windowed random sample of an HM raster (mirrors the dataset's sampling pattern).

    A full read is ~684M pixels; the fit only needs a representative sample.
    """
    rng = np.random.default_rng(random_seed)
    out = []
    with rasterio.open(raster_path) as src:
        H, W = src.height, src.width
        for _ in range(n_windows):
            i = int(rng.integers(0, max(1, H - window_size)))
            j = int(rng.integers(0, max(1, W - window_size)))
            arr = src.read(1, window=Window(j, i, window_size, window_size), masked=True).filled(np.nan)
            arr = arr[np.isfinite(arr) & (arr >= 0)]
            if arr.size:
                out.append(arr)
    if not out:
        raise RuntimeError(f"No valid pixels sampled from {raster_path}")
    return np.concatenate(out)


def fit_rank_gaussian_transform(observed_hm_sample, n_knots: int = 1500) -> RankGaussianTransform:
    return RankGaussianTransform.fit(observed_hm_sample, n_knots=n_knots)


# --------------------------------------------------------------------------------------
# Fold stitching
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# Residuals + class covariates
# --------------------------------------------------------------------------------------
def compute_residuals(
    observed_path: str,
    central_path: str,
    lower_path: str,
    upper_path: str,
    baseline_hm_path: str,
    out_dir: str,
    tag: str,
    transform: RankGaussianTransform | None = None,
    block_rows: int = 1024,
):
    """Write residual and class-covariate rasters for one (window, horizon).

    Outputs (all float32 GeoTIFF, NaN where any input is invalid):
      ``{tag}_res_native.tif``  observed − central, in raw HM units (Phase 1.5 calibrates here)
      ``{tag}_res_z.tif``       T(observed) − T(central), rank-Gaussian units (Phase 1c/2 fit here)
      ``{tag}_dhat.tif``        central − HM_t0, the predicted change (primary class covariate)
      ``{tag}_hm_t0.tif``       baseline HM level (secondary class covariate)
      ``{tag}_w_up.tif``        upper − central
      ``{tag}_w_lo.tif``        central − lower
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = ["res_native", "res_z", "dhat", "hm_t0", "w_up", "w_lo"]
    with rasterio.open(central_path) as ref:
        profile = ref.profile.copy()
        H, W = ref.height, ref.width
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate", BIGTIFF="YES")

    srcs = {
        "obs": rasterio.open(observed_path),
        "central": rasterio.open(central_path),
        "lower": rasterio.open(lower_path),
        "upper": rasterio.open(upper_path),
        "hm0": rasterio.open(baseline_hm_path),
    }
    # Observed/baseline rasters are global; predictions may be a sub-window of them.
    obs_win, obs_row_off, obs_col_off = _aligned_window(profile, srcs["obs"])
    _, hm0_row_off, hm0_col_off = _aligned_window(profile, srcs["hm0"])

    dsts = {n: rasterio.open(out_dir / f"{tag}_{n}.tif", "w", **profile) for n in names}
    n_valid = 0
    try:
        for r0 in range(0, H, block_rows):
            rows = min(block_rows, H - r0)
            win = Window(0, r0, W, rows)
            obs = srcs["obs"].read(
                1, window=Window(obs_col_off, obs_row_off + r0, W, rows), boundless=True, fill_value=np.nan
            ).astype(np.float64)
            hm0 = srcs["hm0"].read(
                1, window=Window(hm0_col_off, hm0_row_off + r0, W, rows), boundless=True, fill_value=np.nan
            ).astype(np.float64)
            central = srcs["central"].read(1, window=win).astype(np.float64)
            lower = srcs["lower"].read(1, window=win).astype(np.float64)
            upper = srcs["upper"].read(1, window=win).astype(np.float64)

            obs = np.where(obs < 0, np.nan, obs)
            hm0 = np.where(hm0 < 0, np.nan, hm0)
            valid = np.isfinite(obs) & np.isfinite(central) & np.isfinite(lower) & np.isfinite(upper)
            n_valid += int(valid.sum())

            res_native = np.where(valid, obs - central, np.nan)
            if transform is not None:
                # Same pixel index for both sides, so a pixel that is zero in both maps to
                # the same z and contributes an exact zero residual.
                idx = window_index_grid(
                    obs_row_off + r0, obs_col_off, rows, W, srcs["obs"].width
                )
                z_obs = transform.forward(np.where(valid, obs, np.nan), index=idx)
                z_cen = transform.forward(np.where(valid, central, np.nan), index=idx)
                res_z = np.where(valid, z_obs - z_cen, np.nan)
            else:
                res_z = np.full_like(res_native, np.nan)

            blocks = {
                "res_native": res_native,
                "res_z": res_z,
                "dhat": np.where(valid & np.isfinite(hm0), central - hm0, np.nan),
                "hm_t0": np.where(valid & np.isfinite(hm0), hm0, np.nan),
                "w_up": np.where(valid, upper - central, np.nan),
                "w_lo": np.where(valid, central - lower, np.nan),
            }
            for n, arr in blocks.items():
                dsts[n].write(arr.astype(np.float32), 1, window=win)
    finally:
        for d in dsts.values():
            d.close()
        for s in srcs.values():
            s.close()

    return {
        "n_valid_px": n_valid,
        **{f"path_{n}": str(out_dir / f"{tag}_{n}.tif") for n in names},
    }


MANIFEST_FIELDS = [
    "window", "base_year", "target_year", "horizon", "n_valid_px",
    "path_res_native", "path_res_z", "path_dhat", "path_hm_t0", "path_w_up", "path_w_lo",
    "path_central", "path_lower", "path_upper", "path_observed", "path_dist_past_change",
]


def append_manifest(manifest_path, row: dict):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    exists = manifest_path.exists()
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_manifest(manifest_path):
    import pandas as pd

    return pd.read_csv(manifest_path)


def _stripe_pairs(path_a, path_b, stripe_h: int, stride: int, offset: int):
    """Every valid (z_{h-5}, z_h) pair inside evenly spaced full-width row stripes."""
    with rasterio.open(path_a) as sa, rasterio.open(path_b) as sb:
        H, W = sa.height, sa.width
        out_a, out_b = [], []
        for r0 in range(offset % stride, H, stride):
            rr = min(stripe_h, H - r0)
            if rr <= 0:
                continue
            win = Window(0, r0, W, rr)
            za = sa.read(1, window=win).ravel()
            zb = sb.read(1, window=win).ravel()
            ok = np.isfinite(za) & np.isfinite(zb)
            if ok.any():
                out_a.append(za[ok])
                out_b.append(zb[ok])
    if not out_a:
        return None
    return np.concatenate(out_a), np.concatenate(out_b)


def horizon_autocorrelation(manifest, n_sample_px: int = 2_000_000, random_seed: int = 42,
                            block: int = 512, max_attempts_per_block: int = 400,
                            report: bool = False, method: str = "stripes",
                            stripe_h: int = 64, stride: int = 512, offset: int = 0):
    """corr(z_h, z_{h-5}) of the residual field, pooled over windows.

    Feeds the AR(1) coupling in Phase 2 (``rho_by_horizon``).

    ``n_sample_px`` is a target number of **valid pixel pairs**, and blocks are drawn until it
    is met. The previous version converted it to a block count instead --
    ``n_sample_px // block**2``, which is *seven* 512 px blocks at the default -- and drew them
    uniformly over the whole grid. On southern Africa's 1.86 Mpx those seven blocks covered
    most of the region and the estimate looked stable. On the global grid, which is 73%
    invalid, they land mostly in ocean and the few that hit land are one correlation length
    across, so the estimate was noise: measured over three seeds at two sample sizes, rho(h=15)
    ranged 0.319 to 0.827 and did not converge when the sample was doubled. An empty or
    nearly-empty block now costs an attempt rather than a share of the sample.

    Targeting valid pairs was necessary but not sufficient: at 20M pairs the estimate still
    moved 0.11-0.16 between seeds, because ~130 blocks of 512 px is an effective sample of
    *blocks*, and the residual field is correlated well inside one. ``method="stripes"``
    (the default) instead reads evenly spaced full-width row stripes, so the sample spans every
    longitude and both hemispheres and is **deterministic** -- no seed, nothing to converge.
    ``method="blocks"`` keeps the random path for callers that want it.
    """
    import pandas as pd

    df = manifest if isinstance(manifest, pd.DataFrame) else read_manifest(manifest)
    rng = np.random.default_rng(random_seed)
    out, diag = {}, {}
    for h in (10, 15, 20):
        windows = []
        _ = rng  # random path is only used when method="blocks"
        for window, grp in df.groupby("window"):
            a = grp[grp["horizon"] == h - 5]
            b = grp[grp["horizon"] == h]
            if not a.empty and not b.empty:
                windows.append((a.iloc[0]["path_res_z"], b.iloc[0]["path_res_z"]))
        if not windows:
            out[h] = float("nan")
            continue

        if method == "stripes":
            pa_all, pb_all, n_pairs = [], [], 0
            for pa, pb in windows:
                got = _stripe_pairs(pa, pb, stripe_h, stride, offset)
                if got is not None:
                    pa_all.append(got[0]); pb_all.append(got[1]); n_pairs += got[0].size
            if pa_all:
                a = np.concatenate(pa_all); b = np.concatenate(pb_all)
                out[h] = float(np.corrcoef(a, b)[0, 1])
            else:
                out[h] = float("nan")
            diag[h] = {"pairs": n_pairs, "windows": len(windows),
                       "stripe_h": stripe_h, "stride": stride, "offset": offset}
            continue

        target = max(1, n_sample_px // len(windows))
        pairs, n_pairs, n_blocks, n_empty = [], 0, 0, 0
        for pa, pb in windows:
            got = 0
            with rasterio.open(pa) as sa, rasterio.open(pb) as sb:
                H, W = sa.height, sa.width
                attempts = 0
                cap = max_attempts_per_block * max(1, target // (block * block) + 1)
                while got < target and attempts < cap:
                    attempts += 1
                    i = int(rng.integers(0, max(1, H - block)))
                    j = int(rng.integers(0, max(1, W - block)))
                    win = Window(j, i, min(block, W - j), min(block, H - i))
                    za = sa.read(1, window=win).ravel()
                    zb = sb.read(1, window=win).ravel()
                    ok = np.isfinite(za) & np.isfinite(zb)
                    n_ok = int(ok.sum())
                    if n_ok <= 10:
                        n_empty += 1
                        continue
                    pairs.append(np.stack([za[ok], zb[ok]], axis=0))
                    got += n_ok
                    n_blocks += 1
            n_pairs += got

        if pairs:
            allp = np.concatenate(pairs, axis=1)
            out[h] = float(np.corrcoef(allp[0], allp[1])[0, 1])
        else:
            out[h] = float("nan")
        diag[h] = {"pairs": n_pairs, "blocks": n_blocks, "empty_draws": n_empty,
                   "windows": len(windows)}
    return (out, diag) if report else out
