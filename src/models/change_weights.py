"""Class weights that teach the quantile heads where change is possible.

The measured problem: the ConvLSTM's central prediction decays 5x with distance from past
change (and goes slightly negative in the far field), but the upper quantile head decays
only 1.9x — from 0.125 to 0.065 — while the observed probability of a >0.05 HM increase
falls from 0.222 to *exactly zero* over 495,640 pixels. The trunk knows where change can
happen; the head does not.

That is an optimization artifact, not a capacity limit. The head sees the same trunk
features, but it is trained by a pooled pinball loss in which the far field is the
overwhelming majority of pixels and contributes almost no gradient signal about its own
tail — the loss is minimised by a globally reasonable width. Weighting each distance band
to contribute equally makes the head fit the *conditional* quantile in each band instead.

Distance bands are computed with dilations of the past-change mask (max-pool), so they cost
one pooling op per radius on the GPU and need no external raster.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Radii in pixels (~km). These mirror the T8 bands, whose observed P(change > 0.05) runs
# 0.222 / 0.080 / 0.023 / 0.0068 / 0.0010 / 0.0000.
DEFAULT_RADII = (1, 3, 10, 30)
DEFAULT_CHANGE_THRESHOLD = 0.01


def distance_band(past_change: torch.Tensor, radii=DEFAULT_RADII,
                  threshold: float = DEFAULT_CHANGE_THRESHOLD) -> torch.Tensor:
    """Band index 0..len(radii): 0 = touching past change, last = beyond every radius.

    ``past_change`` is HM_t0 − HM_{t0−2 steps} in raw HM units, shaped [B, 1, H, W].
    """
    seed = (past_change > threshold).float()
    band = torch.full_like(seed, float(len(radii)))
    for i, r in enumerate(reversed(radii)):
        k = int(2 * r + 1)
        near = F.max_pool2d(seed, kernel_size=k, stride=1, padding=int(r)) > 0
        band = torch.where(near, torch.full_like(band, float(len(radii) - 1 - i)), band)
    return band.long()


def class_balanced_weights(
    past_change: torch.Tensor,
    mask: torch.Tensor,
    target_change: torch.Tensor | None = None,
    radii=DEFAULT_RADII,
    threshold: float = DEFAULT_CHANGE_THRESHOLD,
    change_edges=(0.01, 0.05),
    max_weight: float = 50.0,
) -> torch.Tensor:
    """Per-pixel weights making each (distance band [x change bin]) contribute equally.

    Normalized to mean 1 over valid pixels, so the overall loss scale — and therefore the
    learning rate that was tuned against it — is unchanged; only the *balance* between
    classes moves.
    """
    band = distance_band(past_change, radii=radii, threshold=threshold)
    cls = band
    n_cls = len(radii) + 1
    if target_change is not None:
        bins = torch.zeros_like(band)
        for e in change_edges:
            bins = bins + (target_change.abs() > e).long()
        cls = band * (len(change_edges) + 1) + bins
        n_cls *= len(change_edges) + 1

    valid = mask.bool()
    weights = torch.ones_like(past_change)
    if valid.sum() == 0:
        return weights

    flat = cls[valid].reshape(-1)
    counts = torch.bincount(flat, minlength=n_cls).clamp(min=1).float()
    inv = flat.numel() / (counts * (counts > 0).sum().clamp(min=1))
    inv = inv.clamp(max=max_weight)
    w = inv[cls.clamp(0, n_cls - 1)]
    w = torch.where(valid, w, torch.zeros_like(w))
    denom = w[valid].mean().clamp(min=1e-8)
    return w / denom


CONTEXT_RADII = (1, 3, 10, 30, 100)
N_CONTEXT_CHANNELS = len(CONTEXT_RADII) + 3


def quantile_context_from_distance(dist: torch.Tensor, past_change: torch.Tensor,
                                   hm_now: torch.Tensor | None = None,
                                   radii=CONTEXT_RADII):
    """Context features from a *full-raster* distance-to-past-change band.

    Occupancy within radius r is simply ``dist <= r``, so no pooling is involved and the
    chip boundary plays no part. This is the whole reason the distance is precomputed on
    the full raster (scripts/prepare_change_context.py): deriving it from a 128 px chip
    made the 100 px radius saturate into "is there any change in this chip" — measured at
    0.752 mean occupancy for a single past-change pixel — which is an artifact of framing,
    not geography.

    Returns [B, len(radii) + 3, H, W]: occupancy per radius, log1p distance, the signed
    past change, and the current HM level.
    """
    feats = [(dist <= float(r)).to(dist.dtype) for r in radii]
    feats.append(torch.log1p(dist.clamp(min=0.0)) / 10.0)
    feats.append(past_change)
    feats.append(hm_now if hm_now is not None else torch.zeros_like(past_change))
    return torch.cat(feats, dim=1)


def quantile_context(past_change: torch.Tensor, hm_now: torch.Tensor | None = None,
                     radii=CONTEXT_RADII, threshold: float = DEFAULT_CHANGE_THRESHOLD):
    """Chip-local fallback via max-pool dilations.

    Only correct when the chip is much larger than the biggest radius; retained for tests
    and for callers with no precomputed context raster. Prefer
    :func:`quantile_context_from_distance`.
    """
    seed = (past_change > threshold).float()
    feats = []
    for r in radii:
        feats.append(F.max_pool2d(seed, kernel_size=int(2 * r + 1), stride=1, padding=int(r)))
    feats.append(torch.zeros_like(past_change))
    feats.append(past_change)
    feats.append(hm_now if hm_now is not None else torch.zeros_like(past_change))
    return torch.cat(feats, dim=1)


def past_change_from_inputs(input_dynamic: torch.Tensor, hm_std: float = 1.0,
                            hm_channel: int = 0) -> torch.Tensor:
    """HM_t0 − HM_{t0−2 steps} in raw units, from the normalized dynamic input stack.

    ``input_dynamic`` is [B, T, C, H, W] with the HM channel normalized; the difference of
    two normalized values is the raw difference divided by hm_std, so multiplying back
    recovers raw HM units without needing the mean.
    """
    now = input_dynamic[:, -1, hm_channel:hm_channel + 1]
    then = input_dynamic[:, 0, hm_channel:hm_channel + 1]
    return (now - then) * float(hm_std)
