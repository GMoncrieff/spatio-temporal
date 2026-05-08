"""
Histogram-based loss for comparing distributions of pixel-level changes.

Uses class-balanced cross-entropy and Wasserstein-2 distance to compare
observed vs predicted change distributions at the tile level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_histogram(changes, bin_edges, mask=None):
    """
    Compute histogram from continuous pixel changes.
    
    Args:
        changes: [B, H, W] - continuous change values (target - last_input)
        bin_edges: [num_bins + 1] - histogram bin edges
        mask: [B, H, W] - optional boolean mask for valid pixels
        
    Returns:
        counts: [B, num_bins] - histogram counts per sample
        proportions: [B, num_bins] - normalized proportions (sum to 1 per sample)
    """
    B, H, W = changes.shape
    num_bins = len(bin_edges) - 1
    device = changes.device
    
    # Initialize histogram
    counts = torch.zeros(B, num_bins, device=device, dtype=torch.float32)
    
    for b in range(B):
        change_tile = changes[b]  # [H, W]
        
        # Apply mask if provided
        if mask is not None:
            valid_mask = mask[b]  # [H, W]
            change_values = change_tile[valid_mask]  # [N_valid]
        else:
            change_values = change_tile.reshape(-1)  # [H*W]
        
        # Skip if no valid pixels
        if change_values.numel() == 0:
            # Uniform distribution as fallback
            counts[b] = 1.0
            continue
        
        # Compute histogram using manual binning
        for i in range(num_bins):
            left_edge = bin_edges[i]
            right_edge = bin_edges[i + 1]
            
            if i == num_bins - 1:
                # Last bin includes right edge
                in_bin = (change_values >= left_edge) & (change_values <= right_edge)
            else:
                in_bin = (change_values >= left_edge) & (change_values < right_edge)
            
            counts[b, i] = in_bin.sum().float()
    
    # Normalize to proportions
    total_counts = counts.sum(dim=1, keepdim=True)
    total_counts = torch.clamp(total_counts, min=1.0)  # Avoid division by zero
    proportions = counts / total_counts
    
    return counts, proportions


class HistogramLoss(nn.Module):
    """
    Rarity-weighted Wasserstein-2 loss for comparing histograms of pixel-level changes.
    
    Uses only W2 distance with rarity weights to ensure all bins contribute equally.
    """
    
    def __init__(self, bin_edges, bin_weights=None, num_horizons=4):
        """
        Args:
            bin_edges: Tensor [num_bins+1] with bin edge values
            bin_weights: Optional tensor [num_horizons, num_bins] with rarity weights per horizon
            num_horizons: Number of forecast horizons (default: 4)
        """
        super().__init__()
        self.register_buffer('bin_edges', bin_edges)
        self.num_bins = len(bin_edges) - 1
        self.num_horizons = num_horizons
        
        # Compute bin midpoints for Wasserstein distance
        bin_midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.register_buffer('bin_midpoints', bin_midpoints)
        
        # Set bin weights per horizon (default: uniform for all horizons)
        if bin_weights is None:
            bin_weights = torch.ones(num_horizons, self.num_bins)
        self.register_buffer('bin_weights', bin_weights)
    
    def set_bin_weights(self, bin_weights, horizon_idx=None):
        """
        Update bin weights for specific horizon or all horizons.
        
        Args:
            bin_weights: Tensor [num_bins] or [num_horizons, num_bins]
            horizon_idx: Optional int, if provided updates only that horizon
        """
        if horizon_idx is not None:
            # Update specific horizon
            self.bin_weights[horizon_idx] = bin_weights.to(self.bin_edges.device)
        else:
            # Update all horizons
            self.bin_weights = bin_weights.to(self.bin_edges.device)
    
    def wasserstein2_loss_weighted(self, p_obs, p_pred, horizon_idx=0):
        """
        Compute rarity-weighted Wasserstein-2 distance.
        
        Args:
            p_obs: [B, num_bins] - observed distribution
            p_pred: [B, num_bins] - predicted distribution
            horizon_idx: int - which horizon's weights to use
            
        Returns:
            Scalar rarity-weighted W2 distance
        """
        # Compute cumulative distributions
        cdf_obs = torch.cumsum(p_obs, dim=1)  # [B, num_bins]
        cdf_pred = torch.cumsum(p_pred, dim=1)  # [B, num_bins]
        
        # Compute bin widths (distance between consecutive midpoints)
        bin_widths = self.bin_midpoints[1:] - self.bin_midpoints[:-1]  # [num_bins-1]
        # Pad to match num_bins
        bin_widths = torch.cat([bin_widths, bin_widths[-1:]])  # [num_bins]
        
        # Get weights for this horizon
        weights_h = self.bin_weights[horizon_idx]  # [num_bins]
        
        # Apply rarity weights to emphasize rare bins
        # W2^2 distance with bin weights
        weighted_diff = ((cdf_obs - cdf_pred) ** 2) * bin_widths.unsqueeze(0) * weights_h.unsqueeze(0)
        w2_weighted = weighted_diff.sum(dim=1).mean()  # [B] -> scalar
        
        return w2_weighted
    
    def forward(self, changes_obs, changes_pred, mask=None, horizon_idx=0):
        """
        Compute rarity-weighted W2 histogram loss.
        
        Args:
            changes_obs: [B, H, W] - observed changes (target - last_input)
            changes_pred: [B, H, W] - predicted changes (pred - last_input)
            mask: [B, H, W] - validity mask
            horizon_idx: int - which horizon (0=5yr, 1=10yr, 2=15yr, 3=20yr)
            
        Returns:
            w2_loss: Rarity-weighted Wasserstein-2 loss
            p_obs: Observed histogram proportions (for logging)
            p_pred: Predicted histogram proportions (for logging)
        """
        # Compute histograms
        counts_obs, p_obs = compute_histogram(changes_obs, self.bin_edges, mask=mask)
        counts_pred, p_pred = compute_histogram(changes_pred, self.bin_edges, mask=mask)
        
        # Rarity-weighted Wasserstein-2 loss using horizon-specific weights
        w2_loss = self.wasserstein2_loss_weighted(p_obs, p_pred, horizon_idx=horizon_idx)
        
        return w2_loss, p_obs, p_pred
