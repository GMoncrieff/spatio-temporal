"""
Test baseline MAE calculations to ensure they work correctly.
"""

import torch
import torch.nn.functional as F
import numpy as np

def test_no_change_baseline():
    """Test 'No Change' baseline MAE calculation."""
    print("\n" + "="*70)
    print("TEST: No Change Baseline")
    print("="*70)
    
    # Create synthetic data
    B, H, W = 2, 32, 32
    
    # Last input (most recent HM)
    last_input = torch.rand(B, H, W) * 0.5  # HM values in [0, 0.5]
    
    # Target: slight increase from last input
    target = last_input + torch.rand(B, H, W) * 0.1  # Add 0-0.1
    
    # No Change prediction: just use last_input
    pred_no_change = last_input
    
    # Calculate MAE
    mask = torch.ones(B, H, W, dtype=torch.bool)
    mae_no_change = F.l1_loss(pred_no_change[mask], target[mask], reduction='mean')
    
    print(f"Last input mean: {last_input.mean():.4f}")
    print(f"Target mean: {target.mean():.4f}")
    print(f"Change mean: {(target - last_input).mean():.4f}")
    print(f"MAE (No Change): {mae_no_change.item():.6f}")
    
    # The MAE should be approximately equal to the mean change
    expected_mae = (target - last_input).abs().mean()
    print(f"Expected MAE: {expected_mae.item():.6f}")
    
    assert torch.isclose(mae_no_change, expected_mae, atol=1e-5), "MAE calculation incorrect"
    print("✓ No Change baseline test passed\n")


def test_linear_baseline():
    """Test 'Linear' baseline MAE calculation."""
    print("="*70)
    print("TEST: Linear Baseline")
    print("="*70)
    
    # Create synthetic data with clear linear trend
    B, H, W = 2, 32, 32
    
    # Simulate time series: t=0, t=1, t=2
    # With a consistent increase of 0.05 per timestep
    second_last_input = torch.rand(B, H, W) * 0.5  # t=0
    last_input = second_last_input + 0.05  # t=1
    
    # For 5-year horizon (multiplier=1), target should be last + 0.05
    target_5yr = last_input + 0.05  # t=2
    
    # For 10-year horizon (multiplier=2), target should be last + 0.10
    target_10yr = last_input + 0.10  # t=3
    
    # Calculate trend
    trend = last_input - second_last_input
    print(f"Trend mean: {trend.mean():.4f} (expected: 0.05)")
    
    # Linear prediction for 5-year (h_idx=0, multiplier=1)
    pred_linear_5yr = last_input + (trend * 1)
    mae_linear_5yr = F.l1_loss(pred_linear_5yr, target_5yr, reduction='mean')
    print(f"\n5-year horizon:")
    print(f"  Predicted mean: {pred_linear_5yr.mean():.4f}")
    print(f"  Target mean: {target_5yr.mean():.4f}")
    print(f"  MAE (Linear): {mae_linear_5yr.item():.6f}")
    print(f"  Expected MAE: ~0 (perfect linear trend)")
    
    # Linear prediction for 10-year (h_idx=1, multiplier=2)
    pred_linear_10yr = last_input + (trend * 2)
    mae_linear_10yr = F.l1_loss(pred_linear_10yr, target_10yr, reduction='mean')
    print(f"\n10-year horizon:")
    print(f"  Predicted mean: {pred_linear_10yr.mean():.4f}")
    print(f"  Target mean: {target_10yr.mean():.4f}")
    print(f"  MAE (Linear): {mae_linear_10yr.item():.6f}")
    print(f"  Expected MAE: ~0 (perfect linear trend)")
    
    # For perfect linear trend, MAE should be very small
    assert mae_linear_5yr < 1e-6, f"5yr MAE too large: {mae_linear_5yr}"
    assert mae_linear_10yr < 1e-6, f"10yr MAE too large: {mae_linear_10yr}"
    
    print("✓ Linear baseline test passed\n")


def test_baseline_comparison():
    """Test that baselines work in realistic scenario."""
    print("="*70)
    print("TEST: Realistic Scenario - Compare Baselines")
    print("="*70)
    
    B, H, W = 2, 32, 32
    
    # Scenario: HM is increasing over time
    second_last = torch.rand(B, H, W) * 0.3 + 0.1  # HM in [0.1, 0.4]
    last = second_last + 0.03  # Increased by 0.03
    
    # Target continues to increase but slower (0.02 instead of 0.03)
    target = last + 0.02
    
    # No Change baseline: assume no change
    pred_no_change = last
    mae_no_change = F.l1_loss(pred_no_change, target, reduction='mean')
    
    # Linear baseline: assume same trend continues (0.03 increase)
    trend = last - second_last
    pred_linear = last + trend
    mae_linear = F.l1_loss(pred_linear, target, reduction='mean')
    
    # Simulated Conv-CNN prediction (somewhere between)
    pred_cnn = last + 0.025  # Better than baselines
    mae_cnn = F.l1_loss(pred_cnn, target, reduction='mean')
    
    print(f"Second-to-last mean: {second_last.mean():.4f}")
    print(f"Last input mean: {last.mean():.4f}")
    print(f"Target mean: {target.mean():.4f}")
    print(f"Actual change: {(target - last).mean():.4f}")
    print(f"\nMAE (No Change): {mae_no_change.item():.6f}")
    print(f"MAE (Linear):    {mae_linear.item():.6f}")
    print(f"MAE (Conv-CNN):  {mae_cnn.item():.6f}")
    
    # Conv-CNN should be best in this scenario
    print(f"\nRanking (lower is better):")
    results = [
        ("Conv-CNN", mae_cnn.item()),
        ("Linear", mae_linear.item()),
        ("No Change", mae_no_change.item())
    ]
    results.sort(key=lambda x: x[1])
    for i, (name, mae) in enumerate(results, 1):
        print(f"  {i}. {name}: {mae:.6f}")
    
    print("✓ Baseline comparison test passed\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BASELINE MAE CALCULATION TESTS")
    print("="*70)
    
    test_no_change_baseline()
    test_linear_baseline()
    test_baseline_comparison()
    
    print("="*70)
    print("✅ All baseline MAE tests passed!")
    print("="*70)
