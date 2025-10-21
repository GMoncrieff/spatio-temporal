"""
Test histogram loss for pixel-level change distributions.
"""

import torch
import numpy as np
from src.models.histogram_loss import HistogramLoss, compute_histogram


def test_compute_histogram():
    """Test histogram computation from continuous changes."""
    bin_edges = torch.tensor([-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0])
    
    B, H, W = 2, 32, 32
    changes = torch.randn(B, H, W) * 0.1  # Most values in [-0.3, 0.3]
    
    counts, proportions = compute_histogram(changes, bin_edges)
    
    # Check shapes
    assert counts.shape == (B, 8), f"Expected counts shape (2, 8), got {counts.shape}"
    assert proportions.shape == (B, 8), f"Expected proportions shape (2, 8), got {proportions.shape}"
    
    # Check that proportions sum to 1
    prop_sums = proportions.sum(dim=1)
    assert torch.allclose(prop_sums, torch.ones(B), atol=1e-5), "Proportions should sum to 1"
    
    # Check that all proportions are non-negative
    assert (proportions >= 0).all(), "All proportions should be non-negative"
    
    print("✓ Histogram computation test passed")


def test_histogram_loss():
    """Test histogram loss computation."""
    bin_edges = torch.tensor([-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0])
    
    loss_fn = HistogramLoss(bin_edges)
    
    B, H, W = 4, 32, 32
    changes_obs = torch.randn(B, H, W) * 0.1
    changes_pred = torch.randn(B, H, W) * 0.1
    
    # Compute loss (returns w2_loss, p_obs, p_pred)
    w2_loss, p_obs, p_pred = loss_fn(changes_obs, changes_pred, horizon_idx=0)
    
    # Check that loss is scalar and positive
    assert w2_loss.ndim == 0, "W2 loss should be scalar"
    assert w2_loss > 0, "W2 loss should be positive"
    
    # Check histogram shapes
    assert p_obs.shape == (B, 8), f"Expected p_obs shape (4, 8), got {p_obs.shape}"
    assert p_pred.shape == (B, 8), f"Expected p_pred shape (4, 8), got {p_pred.shape}"
    
    print("✓ Histogram loss test passed")
    print(f"  W2 loss: {w2_loss.item():.6f}")


def test_histogram_with_mask():
    """Test histogram computation with validity mask."""
    bin_edges = torch.tensor([-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0])
    
    loss_fn = HistogramLoss(bin_edges)
    
    B, H, W = 2, 32, 32
    changes_obs = torch.randn(B, H, W) * 0.1
    changes_pred = torch.randn(B, H, W) * 0.1
    
    # Create mask (50% valid pixels)
    mask = torch.rand(B, H, W) > 0.5
    
    # Compute loss with mask
    w2_loss, p_obs, p_pred = loss_fn(changes_obs, changes_pred, mask=mask, horizon_idx=0)
    
    # Check that loss is finite
    assert torch.isfinite(w2_loss), "W2 loss should be finite"
    
    print("✓ Histogram with mask test passed")
    print(f"  W2 loss: {w2_loss.item():.6f}")


def test_class_balanced_weights():
    """Test that bin weights can be set and used correctly."""
    bin_edges = torch.tensor([-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0])
    
    # Create custom weights (4 horizons, 8 bins)
    custom_weights = torch.ones(4, 8)
    custom_weights[0, 3] = 2.0  # Higher weight for bin 3 in horizon 0
    
    loss_fn = HistogramLoss(bin_edges, bin_weights=custom_weights)
    
    # Check that weights were set correctly
    assert loss_fn.bin_weights.shape == (4, 8), "Weights should be [4, 8]"
    assert loss_fn.bin_weights[0, 3] == 2.0, "Custom weight should be set"
    
    # Test that loss can be computed with custom weights
    B, H, W = 2, 32, 32
    changes_obs = torch.randn(B, H, W) * 0.1
    changes_pred = torch.randn(B, H, W) * 0.1
    
    w2_loss, p_obs, p_pred = loss_fn(changes_obs, changes_pred, horizon_idx=0)
    assert torch.isfinite(w2_loss), "Loss should be finite"
    
    print("✓ Class-balanced weights test passed")
    print(f"  Weights shape: {loss_fn.bin_weights.shape}")
    print(f"  Custom weight [0,3]: {loss_fn.bin_weights[0, 3].item()}")


def test_warmup_epochs():
    """Test that histogram loss respects warmup epochs."""
    from src.models.lightning_module import SpatioTemporalLightningModule
    
    model = SpatioTemporalLightningModule(
        hidden_dim=16,
        num_static_channels=3,
        num_dynamic_channels=9,
        num_layers=2,
        histogram_weight=0.5,
        histogram_warmup_epochs=10,
    )
    
    # Check that warmup is set correctly
    assert model.histogram_warmup_epochs == 10, "Warmup epochs should be 10"
    assert model.histogram_weight == 0.5, "Histogram weight should be 0.5"
    
    print("✓ Warmup epochs test passed")
    print(f"  Warmup epochs: {model.histogram_warmup_epochs}")
    print(f"  Histogram weight: {model.histogram_weight}")


if __name__ == "__main__":
    print("Testing histogram loss with new 8-bin configuration...\n")
    test_compute_histogram()
    test_histogram_loss()
    test_histogram_with_mask()
    test_class_balanced_weights()
    test_warmup_epochs()
    print("\n✅ All histogram loss tests passed!")
    print("\nBin configuration:")
    print("  Bin 1: decrease (< -0.005)")
    print("  Bin 2: no change (-0.005 to 0.005)")
    print("  Bin 3: tiny increase (0.005 to 0.02)")
    print("  Bin 4: small increase (0.02 to 0.1)")
    print("  Bin 5: moderate increase (0.1 to 0.2)")
    print("  Bin 6: large increase (0.2 to 0.4)")
    print("  Bin 7: very large increase (0.4 to 0.6)")
    print("  Bin 8: extreme increase (> 0.6)")
