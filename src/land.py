"""What counts as land: one predicate, used by the fold mask and the normalisation statistics.

The HM rasters declare nodata = 3.4e38, a FINITE value. The fold-mask builder tested
``~isnan(x) & (x >= 0)``, which 3.4e38 passes, so 99.33% of the grid -- ocean included --
got a fold id. The observed-HM exporter met the same sentinel from the other side (its
``isfinite`` test would have written the ocean as HM = 1). Land is where HM is a number,
non-negative, and not the raster's declared nodata.
"""
import numpy as np


def hm_land(arr, nodata=None):
    """Boolean land mask for an HM array read raw (not masked), given its declared nodata."""
    arr = np.asarray(arr)
    land = np.isfinite(arr) & (arr >= 0)
    if nodata is not None and np.isfinite(nodata):
        land &= arr != np.asarray(nodata, dtype=arr.dtype)
    return land
