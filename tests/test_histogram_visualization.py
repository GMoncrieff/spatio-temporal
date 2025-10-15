"""
Test histogram visualization in prediction plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor
from src.models.histogram_loss import compute_observed_histogram


def test_histogram_visualization():
    """Test that histogram visualization code works."""
    
    # Create model with histogram head
    model = SpatioTemporalPredictor(
        hidden_dim=16,
        num_static_channels=3,
        num_dynamic_channels=9,
        num_layers=2,
        kernel_size=3,
        use_location_encoder=False,
        use_histogram_head=True,
    )
    model.eval()
    
    # Create dummy inputs
    B, T, C_dyn, H, W = 1, 3, 9, 32, 32
    C_static = 3
    
    input_dynamic = torch.randn(B, T, C_dyn, H, W)
    input_static = torch.randn(B, C_static, H, W)
    
    # Forward pass
    with torch.no_grad():
        preds, hist_probs = model(input_dynamic, input_static)
    
    # Get histogram bins
    histogram_bins = model.histogram_bins.numpy()
    num_bins = len(histogram_bins) - 1
    
    # Create dummy changes
    delta = torch.randn(1, H, W) * 0.1
    pred_delta = torch.randn(1, H, W) * 0.1
    mask = torch.ones(1, H, W, dtype=torch.bool)
    
    # Compute histograms
    p_obs = compute_observed_histogram(delta, model.histogram_bins, mask=mask)
    p_pred_pixels = compute_observed_histogram(pred_delta, model.histogram_bins, mask=mask)
    
    # Extract arrays
    p_obs = p_obs[0].numpy()
    hist_probs_sample = hist_probs[0].numpy()
    p_pred_pixels = p_pred_pixels[0].numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Histogram 1: Observed (log scale)
    axes[0].bar(range(num_bins), p_obs, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Observed Change Histogram')
    axes[0].set_xlabel('Bin')
    axes[0].set_ylabel('Proportion (log scale)')
    axes[0].set_yscale('log')
    axes[0].set_ylim([1e-4, 1.0])
    axes[0].grid(alpha=0.3, which='both')
    
    # Histogram 2: Histogram head (log scale)
    axes[1].bar(range(num_bins), hist_probs_sample, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_title('Histogram Head Prediction')
    axes[1].set_xlabel('Bin')
    axes[1].set_ylabel('Proportion (log scale)')
    axes[1].set_yscale('log')
    axes[1].set_ylim([1e-4, 1.0])
    axes[1].grid(alpha=0.3, which='both')
    
    # Histogram 3: Pixel predictions binned (log scale)
    axes[2].bar(range(num_bins), p_pred_pixels, alpha=0.7, color='green', edgecolor='black')
    axes[2].set_title('Pixel Predictions (Binned)')
    axes[2].set_xlabel('Bin')
    axes[2].set_ylabel('Proportion (log scale)')
    axes[2].set_yscale('log')
    axes[2].set_ylim([1e-4, 1.0])
    axes[2].grid(alpha=0.3, which='both')
    
    # Overlay comparison (log scale)
    x = np.arange(num_bins)
    width = 0.25
    axes[3].bar(x - width, p_obs, width, label='Observed', alpha=0.7, color='blue')
    axes[3].bar(x, hist_probs_sample, width, label='Hist Head', alpha=0.7, color='red')
    axes[3].bar(x + width, p_pred_pixels, width, label='Pixel', alpha=0.7, color='green')
    axes[3].set_title('Comparison')
    axes[3].set_xlabel('Bin')
    axes[3].set_ylabel('Proportion (log scale)')
    axes[3].set_yscale('log')
    axes[3].set_ylim([1e-4, 1.0])
    axes[3].legend()
    axes[3].grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('/tmp/test_histogram_viz.png', dpi=100)
    plt.close()
    
    print("✓ Histogram visualization test passed")
    print(f"  Saved test plot to /tmp/test_histogram_viz.png")
    print(f"  Observed histogram sum: {p_obs.sum():.4f}")
    print(f"  Hist head prediction sum: {hist_probs_sample.sum():.4f}")
    print(f"  Pixel binned sum: {p_pred_pixels.sum():.4f}")
    
    # Verify sums are close to 1
    assert abs(p_obs.sum() - 1.0) < 0.01, "Observed histogram should sum to 1"
    assert abs(hist_probs_sample.sum() - 1.0) < 0.01, "Hist head should sum to 1"
    assert abs(p_pred_pixels.sum() - 1.0) < 0.01, "Pixel binned should sum to 1"
    
    return True


if __name__ == "__main__":
    test_histogram_visualization()
    print("\n✅ Histogram visualization test passed!")
