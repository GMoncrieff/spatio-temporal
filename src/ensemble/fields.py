"""Phase 2 — correlated residual field generation.

Fields are synthesized by circulant embedding: build the target covariance's discrete
spectrum ``S``, then ``y = irfft2(sqrt(S) * rfft2(white noise))`` has exactly that
covariance (up to the circulant wrap). The covariance model is the one Phase 1c fits —
nugget plus a mixture of Gaussian structures — and because each Gaussian structure is
separable, ``S`` is a sum of outer products of 1-D spectra and never has to be materialized
as a full covariance array.

Longitude wraps for free (the global grid is periodic in lon at ±180°); latitude is *not*
periodic, so the row axis is padded and cropped to keep the wrap from folding the Arctic
into the Antarctic.

The FFTs run on GPU through torch when one is available — at the global grid size
(17111 × 40000 ≈ 684M px) that is the difference between seconds and minutes per field.
"""

from __future__ import annotations

import numpy as np

try:  # torch is a hard dependency of the project, but keep the import defensive
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def _pad_for(ranges_px, wrap: bool):
    """Rows/cols of padding needed so the circulant wrap does not contaminate the field."""
    if wrap:
        return 0
    return int(np.ceil(3.0 * max(ranges_px)))


def _next_fast_len(n: int) -> int:
    """Round up to a length with small prime factors (much faster FFTs)."""
    from scipy.fft import next_fast_len

    return int(next_fast_len(n))


def _matern_spectrum_2d(Hp, Wp, length, nu, xp, device=None):
    """Analytic 2-D spectral density of a Matérn covariance, on the rFFT grid.

    ``S(k) ∝ (alpha² + k²)^-(nu+1)`` with ``alpha = sqrt(2 nu)/L``. nu = 0.5 is the
    exponential covariance; nu -> infinity recovers the Gaussian.

    Why this matters here: the Gaussian kernel is infinitely smooth at the origin and its
    spectrum decays as exp(-k²), so it cannot produce mid-frequency roughness. Measured
    against the observed residual field, Gaussian-kernel members carry ~2x too much power
    beyond 50 px and 5-7x too little between 3 and 50 px. A Matérn tail is a power law and
    fixes that directly. Working from the spectrum also removes the need for the covariance
    to be separable, which exponential kernels are not.
    """
    half = Wp // 2 + 1
    if xp == "torch":
        fy = torch.fft.fftfreq(Hp, device=device).reshape(-1, 1)
        fx = torch.fft.rfftfreq(Wp, device=device).reshape(1, -1)
        k2 = (2 * np.pi) ** 2 * (fy ** 2 + fx ** 2)
        alpha2 = (2.0 * nu) / float(length) ** 2
        S = (alpha2 + k2) ** (-(nu + 1.0))
        return (S / S.sum()).to(torch.float32)
    fy = np.fft.fftfreq(Hp).reshape(-1, 1)
    fx = np.fft.rfftfreq(Wp).reshape(1, -1)
    k2 = (2 * np.pi) ** 2 * (fy ** 2 + fx ** 2)
    alpha2 = (2.0 * nu) / float(length) ** 2
    S = (alpha2 + k2) ** (-(nu + 1.0))
    return (S / S.sum()).astype(np.float32)


def _spectrum_1d(n: int, ranges_px, backend, device, dtype):
    """Per-axis spectra of exp(-d^2 / L^2) on a circulant grid of length n."""
    d = np.arange(n)
    d = np.minimum(d, n - d).astype(np.float64)
    out = []
    for L in ranges_px:
        g = np.exp(-(d ** 2) / float(L) ** 2)
        if backend == "torch":
            t = torch.as_tensor(g, device=device, dtype=dtype)
            out.append(torch.fft.fft(t).real)
        else:
            from scipy.fft import fft

            out.append(np.real(fft(g)))
    return out


def generate_correlated_field(
    H: int,
    W: int,
    ranges_px,
    weights,
    nugget: float = 0.0,
    rng=None,
    wrap_lon: bool = True,
    device=None,
    seed=None,
    return_torch: bool = False,
    kernel: str = "gaussian",
    nu: float = 0.5,
):
    """Unit-variance Gaussian field with covariance ``sum_i w_i exp(-h^2/L_i^2) + nugget*delta``.

    Parameters
    ----------
    ranges_px, weights
        Correlation lengths in pixels and their variance shares. ``sum(weights) + nugget``
        is normalized to 1, since only the *correlation* structure matters — the copula in
        Phase 3 sets the marginals, so the sill cancels.
    wrap_lon
        True for the global grid (periodic in longitude). Latitude is padded regardless.
    kernel, nu
        ``"gaussian"`` (the original, very smooth) or ``"matern"`` with smoothness ``nu``
        (0.5 = exponential). Matérn is what puts mid-frequency roughness in the field; the
        Gaussian kernel is 5-7x deficient between 3 and 50 px against the observed
        residual spectrum.
    """
    ranges_px = [float(r) for r in np.atleast_1d(ranges_px)]
    weights = [float(w) for w in np.atleast_1d(weights)]
    if len(ranges_px) != len(weights):
        raise ValueError("ranges_px and weights must have the same length")
    total = sum(weights) + float(nugget)
    if total <= 0:
        raise ValueError("weights + nugget must be positive")
    weights = [w / total for w in weights]
    nugget = float(nugget) / total

    pad_r = _pad_for(ranges_px, wrap=False)          # latitude is never periodic
    pad_c = _pad_for(ranges_px, wrap=wrap_lon)
    Hp = _next_fast_len(H + 2 * pad_r)
    Wp = W if (wrap_lon and pad_c == 0) else _next_fast_len(W + 2 * pad_c)

    use_torch = _HAS_TORCH and (device is not None and str(device) != "cpu")
    backend = "torch" if use_torch else "numpy"

    if backend == "torch":
        dev = torch.device(device)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed if seed is not None else np.random.SeedSequence().entropy % (2**63)))
        noise = torch.randn((Hp, Wp), generator=gen, device=dev, dtype=torch.float32)
        gy = gx = None
        if kernel != "matern":
            gy = _spectrum_1d(Hp, ranges_px, backend, dev, torch.float64)
            gx = _spectrum_1d(Wp, ranges_px, backend, dev, torch.float64)
        half = Wp // 2 + 1
        # The spectrum is accumulated in float32 and the transforms are done in place:
        # at the global grid every full-size temporary is ~1.5-3 GB, and holding one more
        # than necessary is the difference between fitting on a 24 GB card and not.
        S = torch.full((Hp, half), float(nugget), device=dev, dtype=torch.float32)
        if kernel == "matern":
            for w, L in zip(weights, ranges_px):
                S.add_(_matern_spectrum_2d(Hp, Wp, L, nu, "torch", dev), alpha=float(w) * Hp * half)
        else:
            for w, sy, sx in zip(weights, gy, gx):
                S.add_(torch.outer(sy, sx[:half]).to(torch.float32), alpha=float(w))
        del gy, gx
        S.clamp_(min=0.0).sqrt_()
        spec = torch.fft.rfft2(noise)
        del noise
        spec.mul_(S)
        del S
        field = torch.fft.irfft2(spec, s=(Hp, Wp))
        del spec
        field = field[pad_r:pad_r + H, :W] if Wp == W else field[pad_r:pad_r + H, pad_c:pad_c + W]
        # Circulant embedding gives unit variance analytically; renormalize against the
        # realized sample so the copula sees exactly standard-normal scores.
        field = (field - field.mean()) / field.std().clamp_min(1e-12)
        return field if return_torch else field.detach().cpu().numpy()

    from scipy.fft import irfft2, rfft2

    rng = rng if rng is not None else np.random.default_rng(seed)
    noise = rng.standard_normal((Hp, Wp)).astype(np.float32)
    gy = gx = None
    if kernel != "matern":
        gy = _spectrum_1d(Hp, ranges_px, backend, None, None)
        gx = _spectrum_1d(Wp, ranges_px, backend, None, None)
    half = Wp // 2 + 1
    S = np.full((Hp, half), nugget, dtype=np.float64)
    if kernel == "matern":
        for w, L in zip(weights, ranges_px):
            S += float(w) * _matern_spectrum_2d(Hp, Wp, L, nu, "numpy") * Hp * half
    else:
        for w, sy, sx in zip(weights, gy, gx):
            S += w * np.outer(sy, sx[:half])
    S = np.sqrt(np.clip(S, 0.0, None)).astype(np.float32)
    field = irfft2(rfft2(noise) * S, s=(Hp, Wp))
    field = field[pad_r:pad_r + H, :W] if Wp == W else field[pad_r:pad_r + H, pad_c:pad_c + W]
    sd = field.std()
    return ((field - field.mean()) / (sd if sd > 0 else 1.0)).astype(np.float32)


DEFAULT_BASIS_RANGES = (2.0, 5.0, 12.0, 30.0, 80.0, 200.0)


def fit_spectral_mixture(field, ranges_px=DEFAULT_BASIS_RANGES, kernel: str = "matern",
                         nu: float = 0.5, n_bins: int = 60, weight_by_power: bool = True):
    """Fit mixture weights + nugget by matching the observed *radial power spectrum*.

    Variograms are fitted at lags, where a couple of long structures can absorb the fit
    while leaving the mid-frequency band empty — which is exactly the failure here: the
    variogram-fitted field carries 4.9x too much power beyond 50 px and half what it should
    at 3-10 px, the band holding 47% of the observed variance. Matching the spectrum
    targets the quantity the eye actually reads as texture.

    Non-negative least squares over a fixed range basis, so there is no optimizer to get
    stuck and the weights are guaranteed to describe a valid covariance.
    """
    from scipy.optimize import nnls

    f = np.asarray(field, dtype=np.float64)
    finite = np.isfinite(f)
    if finite.sum() < 100:
        raise ValueError("Too few finite pixels to fit a spectrum")
    f = np.where(finite, f - np.nanmean(f), 0.0)
    H, W = f.shape

    k_obs, _, p_obs = radial_power_spectrum(f, n_bins=n_bins)
    p_obs = p_obs / p_obs.sum()

    # Model bases on the same grid and binning, so no analytic normalisation is needed.
    cols = []
    for L in ranges_px:
        S = (_matern_spectrum_2d(H, W, L, nu, "numpy") if kernel == "matern"
             else _gaussian_spectrum_2d(H, W, L))
        cols.append(_bin_spectrum(S, H, W, n_bins, k_obs))
    cols.append(_bin_spectrum(np.ones((H, W // 2 + 1), dtype=np.float32), H, W, n_bins, k_obs))

    A = np.stack(cols, axis=1)
    A = A / np.maximum(A.sum(axis=0, keepdims=True), 1e-30)
    w = np.sqrt(np.maximum(p_obs, 0)) if weight_by_power else np.ones_like(p_obs)
    coef, _ = nnls(A * w[:, None], p_obs * w)
    if coef.sum() <= 0:
        raise ValueError("Spectral fit produced no power")
    coef = coef / coef.sum()
    return {"ranges_px": [float(r) for r in ranges_px],
            "weights": [float(c) for c in coef[:-1]],
            "nugget": float(coef[-1]), "kernel": kernel, "nu": float(nu)}


def _gaussian_spectrum_2d(Hp, Wp, length):
    fy = np.fft.fftfreq(Hp).reshape(-1, 1)
    fx = np.fft.rfftfreq(Wp).reshape(1, -1)
    k2 = (2 * np.pi) ** 2 * (fy ** 2 + fx ** 2)
    S = np.exp(-k2 * float(length) ** 2 / 4.0)
    return (S / S.sum()).astype(np.float32)


def _bin_spectrum(S, H, W, n_bins, k_ref):
    """Total power per radial annulus, on the same bins radial_power_spectrum uses."""
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    edges = np.linspace(0, k.max(), n_bins + 1)[1:]
    idx = np.digitize(k.ravel(), edges)
    power = np.bincount(idx, weights=np.asarray(S, dtype=np.float64).ravel(),
                        minlength=n_bins + 1)[:n_bins]
    counts = np.bincount(idx, minlength=n_bins + 1)[:n_bins]
    return power[counts > 0][: len(k_ref)]


def apply_ar1_horizon_coupling(z_by_horizon, rho_by_horizon, rng=None):
    """Couple per-horizon fields with an AR(1) chain: z_h = rho_h z_{h-5} + sqrt(1-rho^2) eps_h.

    ``z_by_horizon`` holds independent unit-variance fields; the returned dict has the same
    marginal variance but the between-horizon correlation the hindcast residuals show. This
    is what makes *change* statistics (T2.4) come out right — they depend on getting the
    correlation between two horizons correct, not just each horizon's spread.
    """
    horizons = sorted(z_by_horizon)
    out = {horizons[0]: z_by_horizon[horizons[0]]}
    for prev, h in zip(horizons[:-1], horizons[1:]):
        rho = float(rho_by_horizon.get(h, 0.0))
        rho = float(np.clip(rho, -0.999, 0.999))
        eps = z_by_horizon[h]
        out[h] = rho * out[prev] + np.sqrt(1.0 - rho ** 2) * eps
    return out


def empirical_variogram_from_field(field, max_lag_px=256, n_bins=20, n_pairs=200_000, seed=0):
    """Quick semivariogram of a generated field, for the T3.1 comparison."""
    rng = np.random.default_rng(seed)
    H, W = field.shape
    edges = np.unique(np.round(np.geomspace(1, max_lag_px, n_bins + 1)).astype(int))
    sums = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    ai = rng.integers(0, H, n_pairs)
    aj = rng.integers(0, W, n_pairs)
    lag = np.round(np.exp(rng.uniform(0, np.log(max_lag_px), n_pairs))).astype(int)
    theta = rng.uniform(0, 2 * np.pi, n_pairs)
    bi = np.clip(ai + np.round(lag * np.sin(theta)).astype(int), 0, H - 1)
    bj = np.clip(aj + np.round(lag * np.cos(theta)).astype(int), 0, W - 1)
    real_lag = np.round(np.hypot(bi - ai, bj - aj)).astype(int)
    # NaN endpoints are dropped rather than zero-filled: filling would deflate the sill by
    # the fraction of missing pixels instead of leaving it unbiased.
    d2 = 0.5 * (field[ai, aj] - field[bi, bj]) ** 2
    ok = np.isfinite(d2)
    idx = np.digitize(real_lag[ok], edges) - 1
    valid = (idx >= 0) & (idx < len(counts))
    np.add.at(sums, idx[valid], d2[ok][valid])
    np.add.at(counts, idx[valid], 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    keep = counts > 30
    with np.errstate(invalid="ignore"):
        gamma = sums / counts
    return centres[keep], gamma[keep], counts[keep]


def radial_power_spectrum(field, n_bins: int = 40):
    """Radially averaged power spectrum (T3.4), returned as (wavenumber_px^-1, power)."""
    from scipy.fft import rfft2

    f = np.asarray(field, dtype=np.float64)
    f = f - np.nanmean(f)
    f = np.nan_to_num(f)
    F = rfft2(f)
    P = (F.real ** 2 + F.imag ** 2)
    H, W = f.shape
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    kmax = k.max()
    edges = np.linspace(0, kmax, n_bins + 1)[1:]
    idx = np.digitize(k.ravel(), edges)
    power = np.bincount(idx, weights=P.ravel(), minlength=n_bins + 1)[:n_bins]
    counts = np.bincount(idx, minlength=n_bins + 1)[:n_bins]
    centres = 0.5 * (np.r_[0, edges[:-1]] + edges)
    ok = counts > 0
    # Return both the radial *mean* power (the conventional RAPS) and the *total* power in
    # each annulus. Only the total is a variance share: high-k annuli contain many more
    # modes, so summing radial means across bands silently under-weights fine scales.
    return centres[ok], (power[ok] / counts[ok]), power[ok]
