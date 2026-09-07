"""Stratification bins for the conv-spline scorecard. One definition of each, used everywhere.

Extracted from the deleted ensemble layer because these are properties of *the data*, not of
the post-hoc chain that used to live on top of it. Every one of them cost the project real
time when it existed at more than one call site; see ``distance_band`` in particular.
"""

from __future__ import annotations

import numpy as np

# Bin edges fixed from the measured HM change distribution (2000->2020, 18.9M px), so that
# no bin is degenerate. The top two Delta-hat bins are small but are the point of the audit.
DHAT_BINS = [-np.inf, -0.01, 0.001, 0.01, 0.05, 0.15, np.inf]
DHAT_LABELS = ["<=-0.01", "(-0.01,0.001]", "(0.001,0.01]", "(0.01,0.05]", "(0.05,0.15]", ">0.15"]

HM_BINS = [0.0, 0.01, 0.1, 0.3, 0.6, 1.0001]
HM_LABELS = ["[0,0.01)", "[0.01,0.1)", "[0.1,0.3)", "[0.3,0.6)", "[0.6,1]"]

# Distance (px ~ km) to the nearest pixel that changed by >0.01 in the previous decade.
# P(future change > 0.05) runs 0.177 / 0.061 / 0.023 / 0.0078 / 0.0024 / 0.0000 across these
# bands on southern Africa -- a 70x gradient. Computable from the input years alone, so it is
# admissible as a prediction-time stratum.
DIST_BINS = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, np.inf]
DIST_LABELS = ["0-1", "1-3", "3-10", "10-30", "30-100", ">100"]

# Observed SIGNED change. Signed rather than absolute because the negative bin is a finding
# in its own right: 18.1% of land shows a small decrease and HM does not meaningfully
# decrease, so that bin is measurement noise and a forecast leaking probability into it is
# the zero-crossing defect. The core stratum matters more in this phase than it ever has:
# 68% of land moves by less than 0.01, and that is where the density defect lives.
OBS_MAG_BINS = [-np.inf, -0.01, 0.001, 0.01, 0.05, np.inf]
OBS_MAG_LABELS = ["<-0.01", "[-0.01,0.001)", "[0.001,0.01)", "[0.01,0.05)", ">=0.05"]


def distance_band(dist):
    """Band index of a distance-to-past-change raster. The one definition; use it.

    ``right=True`` is load-bearing rather than cosmetic. The distance comes from an exact
    Euclidean transform, so ``dist == 1.0`` and ``dist == 3.0`` are not measure-zero events
    but two of the most populated values on the raster, and the convention decides which
    side of a band edge they fall on. Scored the other way the 0-1 px band's observed
    P(change > 0.05) reads 0.222 rather than 0.177 -- a 26% difference. Half the call sites
    in this repository once used each convention.
    """
    return np.digitize(np.asarray(dist), DIST_BINS[1:-1], right=True).astype(np.int8)


def dhat_bin(dhat):
    """Bin index of the predicted change."""
    return np.clip(np.digitize(np.asarray(dhat), DHAT_BINS[1:-1]), 0, len(DHAT_LABELS) - 1)


def hm_bin(hm0):
    """Bin index of the baseline HM level."""
    return np.clip(np.digitize(np.asarray(hm0), HM_BINS[1:-1]), 0, len(HM_LABELS) - 1)


def obs_mag_bin(change):
    """Bin index of the observed signed change."""
    return np.clip(np.digitize(np.asarray(change), OBS_MAG_BINS[1:-1]),
                   0, len(OBS_MAG_LABELS) - 1)
