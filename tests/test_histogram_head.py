"""
Test histogram prediction head functionality.
"""

import torch
import pytest
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor
from src.models.histogram_loss import HistogramLoss, compute_observed_histogram


def test_histogram_head_forward():
    """Test that histogram head produces correct output shapes."""
    # Create model with histogram head
    model = SpatioTemporalPredictor(
        hidden_dim=16,
        num_static_channels=3,
        num_dynamic_channels=9,
        num_layers=2,
        kernel_size=3,
        use_location_encoder=False,
        use_histogram_head=True,
        histogram_bins=[-1.0, -0.05, -0.005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    )
    
    # Create dummy inputs
    B, T, C_dyn, H, W = 2, 3, 9, 32, 32
    C_static = 3
    
    input_dynamic = torch.randn(B, T, C_dyn, H, W)
    input_static = torch.randn(B, C_static, H, W)
    
    # Forward pass
    preds, hist_probs = model(input_dynamic, input_static)
    
    # Check shapes
    assert preds.shape == (B, 1, H, W), f"Expected preds shape {(B, 1, H, W)}, got {preds.shape}"
    assert hist_probs.shape == (B, 9), f"Expected hist_probs shape {(B, 9)}, got {hist_probs.shape}"
    
    # Check that histogram probabilities sum to 1
    prob_sums = hist_probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(B), atol=1e-5), "Histogram probabilities should sum to 1"
    
    print("✓ Histogram head forward pass test passed")


def test_histogram_loss():
    """Test histogram loss computation."""
    bin_edges = torch.tensor([-1.0, -0.05, -0.005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    bin_midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    loss_fn = HistogramLoss(bin_midpoints, lambda_w2=0.1)
    
    # Create dummy distributions
    B = 4
    num_bins = 9
    
    # Observed distribution (uniform)
    p_obs = torch.ones(B, num_bins) / num_bins
    
    # Predicted distribution (slightly different)
    p_hat = torch.softmax(torch.randn(B, num_bins), dim=1)
    
    # Compute loss
    total_loss, ce_loss, w2_loss = loss_fn(p_obs, p_hat)
    
    # Check that losses are scalars and positive
    assert total_loss.ndim == 0, "Total loss should be scalar"
    assert ce_loss.ndim == 0, "CE loss should be scalar"
    assert w2_loss.ndim == 0, "W2 loss should be scalar"
    assert total_loss > 0, "Total loss should be positive"
    
    print("✓ Histogram loss test passed")


def test_compute_observed_histogram():
    """Test observed histogram computation from continuous changes."""
    bin_edges = torch.tensor([-1.0, -0.05, -0.005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    
    B, H, W = 2, 32, 32
    
    # Create dummy changes with known distribution
    changes = torch.randn(B, H, W) * 0.1  # Most values in [-0.3, 0.3]
    
    # Compute histogram
    p_obs = compute_observed_histogram(changes, bin_edges)
    
    # Check shape
    assert p_obs.shape == (B, 9), f"Expected shape {(B, 9)}, got {p_obs.shape}"
    
    # Check that probabilities sum to 1
    prob_sums = p_obs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(B), atol=1e-5), "Histogram probabilities should sum to 1"
    
    # Check that all probabilities are non-negative
    assert (p_obs >= 0).all(), "All probabilities should be non-negative"
    
    print("✓ Observed histogram computation test passed")


def test_histogram_with_mask():
    """Test observed histogram computation with validity mask."""
    bin_edges = torch.tensor([-1.0, -0.05, -0.005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    
    B, H, W = 2, 32, 32
    
    # Create dummy changes
    changes = torch.randn(B, H, W) * 0.1
    
    # Create mask (50% valid pixels)
    mask = torch.rand(B, H, W) > 0.5
    
    # Compute histogram with mask
    p_obs = compute_observed_histogram(changes, bin_edges, mask=mask)
    
    # Check shape
    assert p_obs.shape == (B, 9), f"Expected shape {(B, 9)}, got {p_obs.shape}"
    
    # Check that probabilities sum to 1
    prob_sums = p_obs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(B), atol=1e-5), "Histogram probabilities should sum to 1"
    
    print("✓ Histogram with mask test passed")


if __name__ == "__main__":
    test_histogram_head_forward()
    test_histogram_loss()
    test_compute_observed_histogram()
    test_histogram_with_mask()
    print("\n✅ All histogram head tests passed!")
