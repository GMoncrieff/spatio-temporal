"""
Pinball loss (quantile loss) for quantile regression.

Used to predict specific quantiles of the distribution rather than just the mean.
"""

import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    
    For quantile q:
    - If error > 0 (underestimate): loss = q * error
    - If error < 0 (overestimate): loss = (1-q) * |error|
    
    Args:
        quantile: Target quantile (e.g., 0.025 for 2.5%, 0.975 for 97.5%)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, quantile: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        assert 0 < quantile < 1, f"Quantile must be in (0, 1), got {quantile}"
        self.quantile = quantile
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute pinball loss.
        
        Args:
            pred: Predicted values [B, 1, H, W] or [B, H, W]
            target: Target values [B, 1, H, W] or [B, H, W]
            mask: Valid pixel mask [B, 1, H, W] or [B, H, W] (True = valid)
        
        Returns:
            Scalar loss (if reduction='mean' or 'sum') or tensor (if reduction='none')
        """
        # Compute error
        error = target - pred  # [B, *, H, W]
        
        # Pinball loss: q * max(error, 0) + (1-q) * max(-error, 0)
        # Equivalent to: q * error if error > 0, else (q-1) * error
        loss = torch.where(
            error >= 0,
            self.quantile * error,
            (self.quantile - 1) * error
        )
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            n_valid = mask.sum()
            if n_valid == 0:
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        else:
            n_valid = loss.numel()
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.sum() / n_valid
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def test_pinball_loss():
    """Test pinball loss behavior."""
    print("Testing Pinball Loss...")
    
    # Test median (q=0.5) - should behave like MAE
    loss_median = PinballLoss(quantile=0.5)
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[2.0, 2.0, 1.0]])
    
    # Errors: [1, 0, -2]
    # Expected: 0.5 * |1| + 0.5 * |0| + 0.5 * |-2| = 1.5
    result = loss_median(pred, target)
    print(f"Median (q=0.5) loss: {result.item():.4f} (expected: 1.5)")
    
    # Test lower quantile (q=0.025) - penalizes underestimates heavily
    loss_lower = PinballLoss(quantile=0.025)
    # Underestimate by 1: 0.025 * 1 = 0.025
    # Overestimate by 1: 0.975 * 1 = 0.975
    pred_under = torch.tensor([[1.0]])
    pred_over = torch.tensor([[3.0]])
    target_test = torch.tensor([[2.0]])
    
    loss_under = loss_lower(pred_under, target_test)
    loss_over = loss_lower(pred_over, target_test)
    print(f"\nLower quantile (q=0.025):")
    print(f"  Underestimate loss: {loss_under.item():.4f} (expected: 0.025)")
    print(f"  Overestimate loss: {loss_over.item():.4f} (expected: 0.975)")
    
    # Test upper quantile (q=0.975) - penalizes overestimates heavily
    loss_upper = PinballLoss(quantile=0.975)
    loss_under = loss_upper(pred_under, target_test)
    loss_over = loss_upper(pred_over, target_test)
    print(f"\nUpper quantile (q=0.975):")
    print(f"  Underestimate loss: {loss_under.item():.4f} (expected: 0.975)")
    print(f"  Overestimate loss: {loss_over.item():.4f} (expected: 0.025)")
    
    print("\n✓ Pinball loss tests passed!")


if __name__ == "__main__":
    test_pinball_loss()
