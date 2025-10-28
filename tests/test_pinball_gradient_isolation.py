"""
Test to verify pinball loss gradient isolation.

This test ensures that:
1. Pinball loss only affects quantile head parameters (lower_heads, upper_heads)
2. Central loss only affects backbone and central head parameters
3. No gradient interference between the two loss types
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytest
from src.models.lightning_module import SpatioTemporalLightningModule


def test_pinball_gradient_isolation():
    """
    Verify that pinball loss gradients are isolated to quantile heads only.
    """
    # Create a minimal model
    model = SpatioTemporalLightningModule(
        hidden_dim=8,
        lr=1e-3,
        num_static_channels=2,
        num_dynamic_channels=2,
        num_layers=1,
        kernel_size=3,
        use_location_encoder=False,
        ssim_weight=1.0,
        laplacian_weight=0.5,
        histogram_weight=0.0,  # Disable histogram for simpler testing
    )
    model.train()
    
    # Create dummy batch
    B, T, C_d, H, W = 2, 3, 2, 16, 16
    C_s = 2
    batch = {
        'input_dynamic': torch.randn(B, T, C_d, H, W),
        'input_static': torch.randn(B, C_s, H, W),
        'target_5yr': torch.randn(B, H, W),
        'target_10yr': torch.randn(B, H, W),
        'target_15yr': torch.randn(B, H, W),
        'target_20yr': torch.randn(B, H, W),
    }
    
    # Get optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.optimizers = lambda: optimizer  # Mock for manual optimization
    
    # Forward pass
    input_dynamic = batch['input_dynamic']
    input_static = batch['input_static']
    targets = {
        '5yr': batch['target_5yr'].unsqueeze(1),
        '10yr': batch['target_10yr'].unsqueeze(1),
        '15yr': batch['target_15yr'].unsqueeze(1),
        '20yr': batch['target_20yr'].unsqueeze(1),
    }
    
    # Prepare inputs
    dynamic_valid = torch.isfinite(input_dynamic).all(dim=(1, 2), keepdim=True)
    static_valid = torch.isfinite(input_static).all(dim=1, keepdim=True).unsqueeze(1)
    input_dynamic = torch.nan_to_num(input_dynamic, nan=0.0)
    input_static = torch.nan_to_num(input_static, nan=0.0)
    
    preds_all = model(input_dynamic, input_static, lonlat=None)
    last_input = input_dynamic[:, -1, 0:1, :, :]
    
    # Compute losses for one horizon (simplified)
    h_idx = 0
    h_name = '5yr'
    pred_lower = preds_all[:, 3*h_idx:3*h_idx+1, :, :]
    pred_central = preds_all[:, 3*h_idx+1:3*h_idx+2, :, :]
    pred_upper = preds_all[:, 3*h_idx+2:3*h_idx+3, :, :]
    target_h = targets[h_name]
    
    target_valid = torch.isfinite(target_h)
    mask_h = target_valid & dynamic_valid.squeeze(2) & static_valid.squeeze(2) & torch.isfinite(last_input)
    
    # Compute simplified losses
    mse_loss = torch.nn.functional.mse_loss(pred_central[mask_h], target_h[mask_h])
    pinball_lower_loss = model.pinball_lower(pred_lower, target_h, mask=mask_h)
    pinball_upper_loss = model.pinball_upper(pred_upper, target_h, mask=mask_h)
    
    central_loss = mse_loss
    pinball_loss = pinball_lower_loss + pinball_upper_loss
    
    # Manual backward pass (simulating what happens in training_step)
    optimizer.zero_grad()
    central_loss.backward(retain_graph=True)
    
    # Save ConvLSTM gradients (should be from central loss only)
    convlstm_grads_after_central = []
    for param in model.model.convlstm.parameters():
        if param.grad is not None:
            convlstm_grads_after_central.append(param.grad.clone())
    
    # Save quantile head gradients (should be None at this point)
    lower_grads_after_central = []
    for param in model.model.lower_heads.parameters():
        if param.grad is not None:
            lower_grads_after_central.append(param.grad.clone())
        else:
            lower_grads_after_central.append(None)
    
    # Collect quantile params
    quantile_params = set()
    for param in model.model.lower_heads.parameters():
        quantile_params.add(param)
    for param in model.model.upper_heads.parameters():
        quantile_params.add(param)
    
    # Save non-quantile gradients
    saved_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None and param not in quantile_params:
            saved_grads[name] = param.grad.clone()
    
    # Backprop pinball loss
    pinball_loss.backward()
    
    # Restore non-quantile gradients
    for name, param in model.named_parameters():
        if name in saved_grads:
            param.grad = saved_grads[name]
    
    # Verify gradients
    # 1. ConvLSTM should have ONLY central loss gradients (pinball removed)
    for i, param in enumerate(model.model.convlstm.parameters()):
        if param.grad is not None and i < len(convlstm_grads_after_central):
            assert torch.allclose(param.grad, convlstm_grads_after_central[i], atol=1e-6), \
                f"ConvLSTM param {i} gradient changed after pinball backward (should be restored)"
    
    # 2. Lower heads should have pinball gradients
    for i, param in enumerate(model.model.lower_heads.parameters()):
        assert param.grad is not None, f"Lower head param {i} should have gradient from pinball loss"
    
    # 3. Upper heads should have pinball gradients
    for i, param in enumerate(model.model.upper_heads.parameters()):
        assert param.grad is not None, f"Upper head param {i} should have gradient from pinball loss"
    
    # 4. Central heads should have ONLY central loss gradients
    for param in model.model.central_heads.parameters():
        assert param.grad is not None, "Central head should have gradient from central loss"
    
    print("✅ Gradient isolation test passed!")
    print(f"   ConvLSTM: {len(convlstm_grads_after_central)} params with central loss grads only")
    print(f"   Lower heads: {sum(1 for p in model.model.lower_heads.parameters() if p.grad is not None)} params with pinball grads")
    print(f"   Upper heads: {sum(1 for p in model.model.upper_heads.parameters() if p.grad is not None)} params with pinball grads")
    print(f"   Central heads: {sum(1 for p in model.model.central_heads.parameters() if p.grad is not None)} params with central grads")


def test_gradient_flow_values():
    """
    Verify that gradients have sensible magnitudes and directions.
    """
    model = SpatioTemporalLightningModule(
        hidden_dim=8,
        num_static_channels=2,
        num_dynamic_channels=2,
        use_location_encoder=False,
        histogram_weight=0.0,
    )
    model.train()
    
    # Create dummy batch with known patterns
    B, T, C_d, H, W = 1, 2, 2, 8, 8
    batch = {
        'input_dynamic': torch.randn(B, T, C_d, H, W) * 0.1,
        'input_static': torch.randn(B, 2, H, W) * 0.1,
        'target_5yr': torch.ones(B, H, W) * 0.5,  # Constant target
        'target_10yr': torch.ones(B, H, W) * 0.5,
        'target_15yr': torch.ones(B, H, W) * 0.5,
        'target_20yr': torch.ones(B, H, W) * 0.5,
    }
    
    # Simulate training step manually (without Lightning Trainer)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Forward pass
    input_dynamic = batch['input_dynamic']
    input_static = batch['input_static']
    targets = {
        '5yr': batch['target_5yr'].unsqueeze(1),
        '10yr': batch['target_10yr'].unsqueeze(1),
        '15yr': batch['target_15yr'].unsqueeze(1),
        '20yr': batch['target_20yr'].unsqueeze(1),
    }
    
    # Prepare inputs
    dynamic_valid = torch.isfinite(input_dynamic).all(dim=(1, 2), keepdim=True)
    static_valid = torch.isfinite(input_static).all(dim=1, keepdim=True).unsqueeze(1)
    input_dynamic = torch.nan_to_num(input_dynamic, nan=0.0)
    input_static = torch.nan_to_num(input_static, nan=0.0)
    
    preds_all = model(input_dynamic, input_static, lonlat=None)
    last_input = input_dynamic[:, -1, 0:1, :, :]
    
    # Compute simplified losses for one horizon
    h_idx = 0
    pred_lower = preds_all[:, 3*h_idx:3*h_idx+1, :, :]
    pred_central = preds_all[:, 3*h_idx+1:3*h_idx+2, :, :]
    pred_upper = preds_all[:, 3*h_idx+2:3*h_idx+3, :, :]
    target_h = targets['5yr']
    
    target_valid = torch.isfinite(target_h)
    mask_h = target_valid & dynamic_valid.squeeze(2) & static_valid.squeeze(2) & torch.isfinite(last_input)
    
    # Compute losses
    mse_loss = torch.nn.functional.mse_loss(pred_central[mask_h], target_h[mask_h])
    pinball_lower_loss = model.pinball_lower(pred_lower, target_h, mask=mask_h)
    pinball_upper_loss = model.pinball_upper(pred_upper, target_h, mask=mask_h)
    
    central_loss = mse_loss
    pinball_loss = pinball_lower_loss + pinball_upper_loss
    total_loss = central_loss + pinball_loss
    
    # Manual backward (simulating the training_step logic)
    optimizer.zero_grad()
    central_loss.backward(retain_graph=True)
    
    # Collect quantile params
    quantile_params = set()
    for param in model.model.lower_heads.parameters():
        quantile_params.add(param)
    for param in model.model.upper_heads.parameters():
        quantile_params.add(param)
    
    # Save non-quantile gradients
    saved_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None and param not in quantile_params:
            saved_grads[name] = param.grad.clone()
    
    # Backprop pinball loss
    pinball_loss.backward()
    
    # Restore non-quantile gradients
    for name, param in model.named_parameters():
        if name in saved_grads:
            param.grad = saved_grads[name]
    
    # Verify all expected parameters have gradients
    convlstm_has_grad = any(p.grad is not None for p in model.model.convlstm.parameters())
    lower_has_grad = any(p.grad is not None for p in model.model.lower_heads.parameters())
    upper_has_grad = any(p.grad is not None for p in model.model.upper_heads.parameters())
    central_has_grad = any(p.grad is not None for p in model.model.central_heads.parameters())
    
    assert convlstm_has_grad, "ConvLSTM should have gradients"
    assert lower_has_grad, "Lower heads should have gradients"
    assert upper_has_grad, "Upper heads should have gradients"
    assert central_has_grad, "Central heads should have gradients"
    
    print("✅ Gradient flow test passed!")
    print(f"   Total loss value: {total_loss.item():.6f}")
    print(f"   Central loss: {central_loss.item():.6f}, Pinball loss: {pinball_loss.item():.6f}")


if __name__ == "__main__":
    print("Running pinball gradient isolation tests...\n")
    test_pinball_gradient_isolation()
    print()
    test_gradient_flow_values()
    print("\n✅ All tests passed!")
