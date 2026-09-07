"""Prediction rasters must be BigTIFF.

GDAL's BIGTIFF default is IF_NEEDED, which cannot promote a COMPRESSED raster because it
cannot predict the compressed size. Every large-area prediction output was therefore a classic
TIFF capped at 4 GiB. The 64-band quantile function over all 184.6M global land pixels wants
~15 GB: the forward model died at `TIFFAppendToStrip: Maximum TIFF file size exceeded` in band
9 of 34 and left four rasters that read back as all-finite ZEROS past the failure point --
plausible numbers, no error, and an existence check passed them.

The fold hindcast escaped by 27 MB: its largest quantile raster is 4,267,991,319 bytes against
a 4,294,967,296 ceiling. That margin is luck, not design, and it is what this guards.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

SRC = (REPO / "scripts" / "train_lightning.py").read_text()
CEILING = 4 * 1024 ** 3


def test_float_output_profile_declares_bigtiff():
    blk = SRC.split("out_profile.update({")[1].split("})")[0]
    assert "'BIGTIFF': 'YES'" in blk, "the triple's writer would cap at 4 GiB"


def test_quantile_output_profile_declares_bigtiff():
    # The call spans several lines and contains nested parens, so take a source window
    # rather than trying to split on ')'.
    i = SRC.index("qf_profile.update(")
    blk = SRC[i:i + 400]
    assert "BIGTIFF='YES'" in blk, "the 64-band quantile writer would cap at 4 GiB"


def test_stitcher_still_declares_bigtiff():
    # The stitcher always had it, which is why the stitched path never failed. If it is ever
    # dropped, the global stitched quantile raster (~15 GB) breaks the same way.
    src = (REPO / "src" / "stitch.py").read_text()
    assert 'BIGTIFF="YES"' in src


def _tiff_version(path):
    """42 = classic TIFF (4 GiB ceiling), 43 = BigTIFF. Bytes 2-3 of the header."""
    with open(path, "rb") as fh:
        head = fh.read(4)
    endian = "<" if head[:2] == b"II" else ">"
    return int(np.frombuffer(head[2:4], dtype=np.dtype(endian + "u2"))[0])


def test_bigtiff_flag_actually_changes_the_container(tmp_path):
    """The flag must be load-bearing, not decorative.

    IF_NEEDED *does* promote an uncompressed raster, because GDAL can compute that size up
    front -- which is why a naive control passes and proves nothing. Compression is what
    defeats it: GDAL cannot predict the compressed size, keeps classic TIFF, and the ceiling
    is only discovered mid-write. These are the two cases that actually differ.
    """
    prof = dict(driver="GTiff", height=1024, width=70000, count=1, dtype="uint8",
                crs="EPSG:4326", transform=Affine(0.01, 0, 0, 0, -0.01, 0),
                compress="deflate")
    a = tmp_path / "compressed_default.tif"
    with rasterio.open(a, "w", **prof) as dst:
        dst.write(np.zeros((1024, 70000), dtype=np.uint8), 1)
    assert _tiff_version(a) == 42, "compressed + IF_NEEDED should stay classic TIFF"

    b = tmp_path / "compressed_bigtiff.tif"
    with rasterio.open(b, "w", **dict(prof, BIGTIFF="YES")) as dst:
        dst.write(np.zeros((1024, 70000), dtype=np.uint8), 1)
    assert _tiff_version(b) == 43, "BIGTIFF=YES must produce a BigTIFF container"


def test_the_hindcast_margin_is_recorded():
    """Not a code test: a note that the surviving hindcast was 0.63% from the same failure,
    so nobody later reads 'the hindcast was fine' as 'the hindcast was safe'."""
    largest_fold_qf = 4_267_991_319
    assert largest_fold_qf < CEILING
    assert (CEILING - largest_fold_qf) / CEILING < 0.01, "margin was under 1% — luck, not design"
