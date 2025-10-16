"""
Test histogram loss for pixel-level change distributions.
"""

import torch
import numpy as np
from src.models.histogram_loss import HistogramLoss, compute_histogram


def test_compute_histogram():
    """Test histogram computation from continuous changes."""
    bin_edges = torch.tensor([-1.0, -0.05, -0.005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    
    B, H, W = 2, 32, 32
    changes = torch.randn(B, H, W) * 0.1  # Most values in [-0.3, 0.3]
    
    counts, proportions = compute_histogram(changes, bin_edges)
    
    # Check shapes
    assert counts.shape == (B, 9), f"Expected counts shape (2, 9), got {counts.shape}"
    assert proportions.shape == (B, 9), f"Expected proportions shape (2, 9), got {proportions.shape}"
    
    # Check that proportions sum to 1
    prop_sums = proportions.sum(dim=1)
    assert torch.allclose(prop_sums, torch.ones(B), atol=1e-5), "Proportions should sum to 1"
    
    # Check that all proportions are non-negative
    assert (proportions >= 0).all(), "All proportions should be non-negative"
    
    print("✓ Histogram computation test passed")


def test_histogram_loss():
    """Test histogram loss computation."""
    bin_edges = torch.tensor([-1.0, -0.05, -0.005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    
    loss_fn = HistogramLoss(bin_edges, lambda_w2=0.1)
    
    B, H, W = 4, 32, 32
    changes_obs = torch.randn(B, H, W) * 0.1
    changes_pred = torch.randn(B, H, W) * 0.1
    
    # Compute loss
    total_loss, ce_loss, w2_loss, p_obs, p_pred = loss_fn(changes_obs, changes_pred)
    
    # Check that losses are scalars and positive
    assert total_loss.ndim == 0, "Total loss should be scalar"
    assert ce_loss.ndim == 0, "CE loss should be scalar"
    assert w2_loss.ndim == 0, "W2 loss should be scalar"
    assert total_loss > 0, "Total loss should be positive"
    
    # Check histogram shapes
    assert p_obs.shape == (B, 9), f"Expected p_obs shape (4, 9), got {p_obs.shape}"
    assert p_pred.shape == (B, 9), f"Expected p_pred shape (4, 9), got {p_pred.shape}"
    
    print("✓ Histogram loss test passed")
    print(f"  Total loss: {total_loss.item():.6f}")
    print(f"  CE loss: {ce_loss.item():.6f}")
    print(f"  W2 loss: {w2_loss.item():.6f}")


def test_histogram_with_mask():
    """Test histogram computation with validity mask."""
    bin_edges = torch.tensor([-1.0, -0.05, -0.005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    
    loss_fn = HistogramLoss(bin_edges, lambda_w2=0.1)
    
    B, H, W = 2, 32, 32
    changes_obs = torch.randn(B, H, W) * 0.1
    changes_pred = torch.randn(B, H, W) * 0.1
    
    # Create mask (50% valid pixels)
    mask = torch.rand(B, H, W) > 0.5
    
    # Compute loss with mask
    total_loss, ce_loss, w2_loss, p_obs, p_pred = loss_fn(changes_obs, changes_pred, mask=mask)
    
    # Check that losses are finite
    assert torch.isfinite(total_loss), "Total loss should be finite"
    assert torch.isfinite(ce_loss), "CE loss should be finite"
    assert torch.isfinite(w2_loss), "W2 loss should be finite"
    
    print("✓ Histogram with mask test passed")


def test_class_balanced_weights():
    """Test that class weights are computed correctly."""
    bin_edges = torch.tensor([-1.0, -0.05, -0.005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    
    loss_fn = HistogramLoss(bin_edges, lambda_w2=0.1, smoothing=1e-3)
    
    # Create imbalanced distribution (most values in one bin)
    B = 4
    p_obs = torch.zeros(B, 9)
    p_obs[:, 4] = 0.8  # 80% in bin 4
    p_obs[:, 3] = 0.1  # 10% in bin 3
    p_obs[:, 5] = 0.1  # 10% in bin 5
    
    # Compute class weights
    weights = loss_fn.compute_class_weights(p_obs)
    
    # Check that weights are higher for rare classes
    assert weights[4] < weights[0], "Frequent class should have lower weight"
    assert weights[4] < weights[8], "Frequent class should have lower weight"
    
    # Check that weights sum to num_bins (normalized)
    assert torch.isclose(weights.sum(), torch.tensor(9.0), atol=1e-4), "Weights should sum to num_bins"
    
    print("✓ Class-balanced weights test passed")
    print(f"  Weights: {weights.numpy()}")


if __name__ == "__main__":
    test_compute_histogram()
    test_histogram_loss()
    test_histogram_with_mask()
    test_class_balanced_weights()
    print("\n✅ All histogram loss tests passed!")
