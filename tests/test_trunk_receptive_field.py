"""E2c widens the trunk's receptive field with --kernel_size 5, and the run must be able to
prove it did.

The ConvLSTM sees three timesteps, and every 3x3 convolution along the path from an input
pixel to the last layer's last hidden state adds reach -- the layer-to-layer steps and the
time recurrence alike -- so the radius is ``(k-1)/2 * (sum(d) + (T-1) * max(d))``. At E2a's
4 layers x 3x3 that is 6 px. A stale comment in the code said ~10.

The formula is checked against a gradient probe that shares no code with it (rule 3: the
second implementation earns its keep), and the log line that reports the radius is read off
the CONSTRUCTED module, never off the flags (rule 28). The forecast command used to spell
``--kernel_size 3`` as a literal before TRAIN_ARGS; argparse keeps the last occurrence so it
was harmless for E2a, and it is precisely the second spelling that drifts. The fingerprint
check is what would see it.
"""
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src.models.convlstm import ConvLSTM  # noqa: E402
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor  # noqa: E402

BASE = os.path.join(ROOT, "scripts", "conv_spline_base.sh")

TRUNK = ("Trunk:             ConvLSTM 4 layers x 64, kernel {k}, dilation 1,1,1,1, "
         "receptive radius {r} px over 3 timesteps\n")


def _probe(m, T=3, S=97):
    """Radius of the input region the centre output pixel depends on, by autograd."""
    torch.manual_seed(0)
    x = torch.randn(1, T, m.input_dim, S, S, dtype=torch.float64, requires_grad=True)
    out, _ = m(x)
    out[0][:, -1][0, :, S // 2, S // 2].sum().backward()
    nz = (x.grad.abs().sum(dim=(0, 1, 2)) > 0).nonzero()
    return int((nz - S // 2).abs().max())


@pytest.mark.parametrize("k,L,d,want", [
    (3, 4, 1, 6),              # E2a
    (5, 4, 1, 12),             # E2c
    (7, 4, 1, 18),
    (3, 10, 1, 12),
    (3, 4, [1, 2, 3, 4], 18),
])
def test_receptive_radius_matches_a_gradient_probe(k, L, d, want):
    m = ConvLSTM(2, 4, (k, k), L, batch_first=True, dilation=d).double()
    assert m.receptive_radius(timesteps=3) == want
    assert _probe(m) == want


def _predictor(kernel_size):
    return SpatioTemporalPredictor(hidden_dim=8, kernel_size=kernel_size, num_layers=4,
                                   num_static_channels=1, num_dynamic_channels=1,
                                   use_location_encoder=False)


def test_trunk_banner_reads_the_kernel_off_the_module():
    from train_lightning import _trunk_banner
    line = _trunk_banner(_predictor(5))
    assert line.startswith("Trunk:")
    assert "kernel 5" in line and "receptive radius 12 px" in line


def test_trunk_banner_control_is_e2a():
    from train_lightning import _trunk_banner
    line = _trunk_banner(_predictor(3))
    assert "kernel 3" in line and "receptive radius 6 px" in line


# --------------------------------------------------------------------- the log check

def _verify_trunk(log, flags):
    script = f'source "{BASE}"\nverify_trunk_fingerprint "{log}" probe "{flags}"\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode == 0


def _log(tmp_path, name, k, r):
    p = tmp_path / name
    p.write_text(TRUNK.format(k=k, r=r))
    return p


E2C_FLAGS = "--head_family pwl --free_scale True --mu_mse_weight 0.0 --kernel_size 5"
E2A_FLAGS = "--head_family pwl --free_scale True --mu_mse_weight 0.0"


def test_trunk_fingerprint_accepts_e2c(tmp_path):
    assert _verify_trunk(_log(tmp_path, "e2c.log", 5, 12), E2C_FLAGS)


def test_trunk_fingerprint_rejects_a_3x3_trunk_when_5_was_asked(tmp_path):
    """The drift the forecast command's old --kernel_size 3 literal could have caused."""
    assert not _verify_trunk(_log(tmp_path, "k3.log", 3, 6), E2C_FLAGS)


def test_trunk_fingerprint_defaults_to_3_when_the_flags_name_no_kernel(tmp_path):
    assert _verify_trunk(_log(tmp_path, "e2a.log", 3, 6), E2A_FLAGS)
    assert not _verify_trunk(_log(tmp_path, "e2a_k5.log", 5, 12), E2A_FLAGS)


def test_trunk_fingerprint_fails_closed_on_a_missing_log(tmp_path):
    assert not _verify_trunk(tmp_path / "nope.log", E2C_FLAGS)


def test_trunk_fingerprint_fails_closed_on_a_log_with_no_banner(tmp_path):
    p = tmp_path / "empty.log"
    p.write_text("LOSS WEIGHTS\n")
    assert not _verify_trunk(p, E2C_FLAGS)
