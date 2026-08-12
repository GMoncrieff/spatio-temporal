"""Phase 1c — stratified variogram estimation and multi-scale fitting.

Fitted on the *rank-Gaussian* residual (``res_z``), not raw HM units: the field generator
in Phase 2 needs an approximately Gaussian, homoscedastic residual, and fitting a spread
model in one space while applying it in another is the classic way to get this wrong.

Strata are **biomes (14)**, not ecoregions (846). Aggregation coverage wants many polygons
for statistical power; variogram fitting wants few, large, internally homogeneous strata —
each fit needs enough same-stratum pairs out to the long-range scale, and an 846-way split
starves the long lags. Realm (8) is the fallback for biomes whose fit fails to converge or
whose fitted range exceeds the stratum's own extent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

KM_PER_DEG = 111.32


def pixel_km(transform, lat: float = 0.0):
    """Approximate pixel size in km at a given latitude (EPSG:4326 grid)."""
    dx = abs(transform.a) * KM_PER_DEG * np.cos(np.deg2rad(lat))
    dy = abs(transform.e) * KM_PER_DEG
    return dx, dy


def load_field(residual_raster, stratum_raster=None, stratum_lut=None):
    """Read a residual raster (and an aligned stratum raster) once, into memory.

    The pair sampler is called many times — per horizon, per stratum — so re-reading
    windows from a compressed GeoTIFF each time dominates the runtime. Even the global
    grid is only ~2.7 GB as float32, which this machine holds comfortably.
    """
    with rasterio.open(residual_raster) as src:
        z = src.read(1).astype(np.float32)
        p_t = src.transform
        H, W = src.height, src.width
    st = None
    if stratum_raster is not None:
        with rasterio.open(stratum_raster) as ssrc:
            s_t = ssrc.transform
            r_off = int(round((p_t.f - s_t.f) / s_t.e))
            c_off = int(round((p_t.c - s_t.c) / s_t.a))
            st = ssrc.read(1, window=Window(c_off, r_off, W, H), boundless=True, fill_value=0)
        if stratum_lut is not None:
            st = stratum_lut[np.clip(st, 0, len(stratum_lut) - 1)]
    return z, st


def sample_pixel_pairs(
    residual_raster,
    n_pairs: int = 400_000,
    max_lag_px: int = 1024,
    n_bins: int = 24,
    stratum_raster=None,
    stratum_values=None,
    stratum_lut=None,
    patch: int = 1024,
    random_seed: int = 42,
    block_source=None,
    stratum_source=None,
):
    """Distance-binned empirical semivariogram from randomly sampled pixel pairs.

    Exhaustive pairing is infeasible at ~150M pixels, so pairs are drawn inside random
    patches: random (point, lagged point) pairs are taken from a patch of the field. When a
    stratum is given, both members of a pair must fall in it.

    ``block_source`` / ``stratum_source`` accept preloaded arrays (see :func:`load_field`).

    Returns ``(bin_centres_px, gamma, counts)`` where gamma is the semivariance
    ``0.5 * mean((z_i - z_j)^2)``.
    """
    rng = np.random.default_rng(random_seed)
    edges = np.unique(np.round(np.geomspace(1, max_lag_px, n_bins + 1)).astype(int))
    sums = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)

    if block_source is None:
        block_source, stratum_source = load_field(residual_raster, stratum_raster, stratum_lut)
    H, W = block_source.shape

    strat_mask = None
    if stratum_source is not None and stratum_values is not None:
        strat_mask = np.isin(stratum_source, stratum_values)
        if strat_mask.sum() < 200:
            return np.array([]), np.array([]), np.array([])

    drawn = 0
    attempts = 0
    per_patch = max(2000, n_pairs // 50)
    max_attempts = 400
    while drawn < n_pairs and attempts < max_attempts:
        attempts += 1
        ph, pw = min(patch, H), min(patch, W)
        i = int(rng.integers(0, max(1, H - ph)))
        j = int(rng.integers(0, max(1, W - pw)))
        z = block_source[i:i + ph, j:j + pw]
        ok = np.isfinite(z)
        if strat_mask is not None:
            ok = ok & strat_mask[i:i + ph, j:j + pw]
        n_ok = int(ok.sum())
        if n_ok < 100:
            continue

        ai, aj = np.nonzero(ok)
        sel = rng.integers(0, ai.size, size=per_patch)
        ai, aj = ai[sel], aj[sel]
        lag = np.round(np.exp(rng.uniform(
            0, np.log(max(2, min(max_lag_px, min(ph, pw) - 1))), per_patch))).astype(int)
        theta = rng.uniform(0, 2 * np.pi, per_patch)
        bi = ai + np.round(lag * np.sin(theta)).astype(int)
        bj = aj + np.round(lag * np.cos(theta)).astype(int)
        inb = (bi >= 0) & (bi < ph) & (bj >= 0) & (bj < pw)
        ai, aj, bi, bj, lag = ai[inb], aj[inb], bi[inb], bj[inb], lag[inb]
        if ai.size == 0:
            continue
        good = ok[bi, bj]
        ai, aj, bi, bj, lag = ai[good], aj[good], bi[good], bj[good], lag[good]
        if ai.size == 0:
            continue

        d2 = 0.5 * (z[ai, aj].astype(np.float64) - z[bi, bj].astype(np.float64)) ** 2
        idx = np.digitize(lag, edges) - 1
        valid = (idx >= 0) & (idx < len(counts))
        np.add.at(sums, idx[valid], d2[valid])
        np.add.at(counts, idx[valid], 1)
        drawn += int(valid.sum())

    centres = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = sums / counts
    keep = counts > 30
    return centres[keep], gamma[keep], counts[keep]


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------
def nugget_two_range_model(h, nugget, var1, len1, var2, len2):
    """Nugget + two Gaussian-kernel structures (the model Phase 2 can synthesize exactly)."""
    h = np.asarray(h, dtype=np.float64)
    return (
        nugget
        + var1 * (1.0 - np.exp(-(h ** 2) / (len1 ** 2)))
        + var2 * (1.0 - np.exp(-(h ** 2) / (len2 ** 2)))
    )


def fit_nugget_multirange_model(centres, gamma, counts=None, max_lag_px=None, use_gstools=True):
    """Fit nugget + short-range + long-range structures to a binned variogram.

    gstools supplies the initial guess and a single-structure sanity fit; the two-range
    refinement is a weighted least-squares on top, because Phase 2 synthesizes exactly this
    kernel mixture and a gstools covariance model with one structure cannot express it.
    """
    from scipy.optimize import curve_fit

    centres = np.asarray(centres, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    ok = np.isfinite(centres) & np.isfinite(gamma)
    centres, gamma = centres[ok], gamma[ok]
    if centres.size < 5:
        raise ValueError("Too few variogram bins to fit")
    w = np.sqrt(counts[ok]) if counts is not None else np.ones_like(centres)

    sill0 = float(np.nanmax(gamma))
    max_lag = max_lag_px or float(centres.max())

    gstools_summary = None
    if use_gstools:
        try:
            import gstools as gs

            model = gs.Gaussian(dim=2)
            _, _, r2 = model.fit_variogram(centres, gamma, nugget=True, return_r2=True)
            gstools_summary = {
                "gstools_var": float(model.var),
                "gstools_len_scale": float(model.len_scale),
                "gstools_nugget": float(model.nugget),
                "gstools_r2": float(r2),
            }
        except Exception as e:  # pragma: no cover - optional dependency path
            gstools_summary = {"gstools_error": str(e)}

    p0 = [
        max(1e-6, 0.15 * sill0),          # nugget
        0.45 * sill0, max(2.0, 0.05 * max_lag),   # short range
        0.45 * sill0, max(5.0, 0.5 * max_lag),    # long range
    ]
    bounds = (
        [0.0, 0.0, 1.0, 0.0, 2.0],
        [sill0 * 1.5, sill0 * 3, max_lag * 2, sill0 * 3, max_lag * 20],
    )
    try:
        popt, _ = curve_fit(
            nugget_two_range_model, centres, gamma, p0=p0, bounds=bounds,
            sigma=1.0 / np.maximum(w, 1e-6), maxfev=20000,
        )
        converged = True
    except Exception:
        popt, converged = np.array(p0), False

    nugget, var1, len1, var2, len2 = popt
    if len1 > len2:  # keep "short" first for a stable, interpretable parameterization
        var1, len1, var2, len2 = var2, len2, var1, len1
    sill = nugget + var1 + var2
    pred = nugget_two_range_model(centres, nugget, var1, len1, var2, len2)
    ss_res = float(np.sum(w * (gamma - pred) ** 2))
    ss_tot = float(np.sum(w * (gamma - np.average(gamma, weights=w)) ** 2))

    return {
        "nugget": float(nugget),
        "var_short": float(var1), "range_short_px": float(len1),
        "var_long": float(var2), "range_long_px": float(len2),
        "sill": float(sill),
        "nugget_fraction": float(nugget / sill) if sill > 0 else np.nan,
        # Practical range of a Gaussian structure: where gamma reaches 95% of its sill.
        "practical_range_px": float(np.sqrt(3.0) * len2),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "converged": bool(converged),
        **(gstools_summary or {}),
    }


def fit_stratified(
    residual_raster,
    stratum_raster=None,
    lookup_csv=None,
    strata=None,
    n_pairs: int = 300_000,
    max_lag_px: int = 1024,
    random_seed: int = 42,
    min_r2: float = 0.5,
):
    """Fit one variogram model per stratum (biome), flagging fits that should roll up.

    A fit is flagged when it fails to converge, has low R², or its practical range exceeds
    the max lag sampled (which means the stratum is smaller than its own fitted structure).
    """
    from .validate import biome_lut

    lut_arr = None
    if stratum_raster is not None and lookup_csv is not None:
        lut_arr, _, _ = biome_lut(lookup_csv)
        if strata is None:
            strata = sorted(int(b) for b in np.unique(lut_arr) if b > 0)

    # One read of the residual (and stratum) raster serves every stratum's fit.
    field, strat = load_field(residual_raster, stratum_raster, lut_arr)

    rows = []
    todo = [(None, None)] if strata is None else [(s, [s]) for s in strata]
    for name, values in todo:
        centres, gamma, counts = sample_pixel_pairs(
            None, n_pairs=n_pairs, max_lag_px=max_lag_px,
            stratum_values=values, random_seed=random_seed,
            block_source=field, stratum_source=strat,
        )
        if centres.size < 5:
            rows.append({"stratum": name, "converged": False, "flag": "insufficient_pairs"})
            continue
        fit = fit_nugget_multirange_model(centres, gamma, counts, max_lag_px=max_lag_px)
        flag = ""
        if not fit["converged"]:
            flag = "no_convergence"
        elif fit["r2"] < min_r2:
            flag = "low_r2"
        elif fit["practical_range_px"] > max_lag_px:
            flag = "range_exceeds_extent"
        rows.append({"stratum": name, "n_bins": len(centres), "n_pairs": int(counts.sum()),
                     "flag": flag, **fit})
    return pd.DataFrame(rows)


def extrapolate_h20_check(fitted_params_by_horizon: dict, h20_fit: dict | None = None):
    """Fit sigma(h) / range(h) growth from h=5,10,15 and predict h=20.

    Only one h=20 residual map exists across all folds (it is observed only from the
    [1990,95,00] window), so this is a *single validation point*, not a distribution —
    weak evidence, and any writeup should say so.
    """
    hs = sorted(h for h in fitted_params_by_horizon if h != 20)
    if len(hs) < 2:
        return {"status": "insufficient_horizons"}

    def _fit_pred(key):
        y = np.array([fitted_params_by_horizon[h][key] for h in hs], dtype=float)
        x = np.array(hs, dtype=float)
        good = np.isfinite(y) & (y > 0)
        if good.sum() < 2:
            return np.nan, np.nan
        # Power law in lead time: log y = a + b log h
        b, a = np.polyfit(np.log(x[good]), np.log(y[good]), 1)
        return float(np.exp(a + b * np.log(20.0))), float(b)

    out = {"status": "ok", "horizons_used": hs}
    for key in ("sill", "practical_range_px", "nugget"):
        pred, slope = _fit_pred(key)
        out[f"pred_{key}_h20"] = pred
        out[f"slope_log_{key}"] = slope
        if h20_fit is not None and key in h20_fit:
            obs = h20_fit[key]
            out[f"obs_{key}_h20"] = obs
            out[f"ratio_{key}"] = float(pred / obs) if obs else np.nan
    out["evidence"] = "single h=20 map; weak evidence, not proof"
    return out


def save_fits(df: pd.DataFrame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def fits_to_field_params(fit_row, px_km: float = 1.0):
    """Convert a fitted row into Phase 2 ``generate_correlated_field`` arguments."""
    total = fit_row["nugget"] + fit_row["var_short"] + fit_row["var_long"]
    if total <= 0:
        raise ValueError("Degenerate variogram fit (zero sill)")
    return {
        "ranges_px": [float(fit_row["range_short_px"]), float(fit_row["range_long_px"])],
        "weights": [float(fit_row["var_short"] / total), float(fit_row["var_long"] / total)],
        "nugget": float(fit_row["nugget"] / total),
        "sill": float(total),
    }


def to_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return path
