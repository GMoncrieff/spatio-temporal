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
    Loss for comparing histograms of pixel-level changes.
    
    Combines:
    1. Class-balanced Cross-Entropy: Weights bins by inverse frequency
    2. Wasserstein-2 Distance: Metric on probability distributions
    """
    
    def __init__(self, bin_edges, lambda_w2=0.1, smoothing=1e-3, label_smoothing=0.05):
        """
        Args:
            bin_edges: Tensor [num_bins+1] with bin edge values
            lambda_w2: Weight for Wasserstein-2 term (default: 0.1)
            smoothing: Smoothing factor for class weights (default: 1e-3)
            label_smoothing: Label smoothing factor to prevent extreme log probs (default: 0.05)
        """
        super().__init__()
        self.register_buffer('bin_edges', bin_edges)
        self.lambda_w2 = lambda_w2
        self.smoothing = smoothing
        self.label_smoothing = label_smoothing
        self.num_bins = len(bin_edges) - 1
        
        # Compute bin midpoints for Wasserstein distance
        bin_midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.register_buffer('bin_midpoints', bin_midpoints)
    
    def compute_class_weights(self, p_obs):
        """
        Compute inverse frequency weights for class-balanced CE.
        
        Args:
            p_obs: [B, num_bins] - observed proportions
            
        Returns:
            weights: [num_bins] - class weights (inverse frequency)
        """
        # Average proportion across batch
        avg_proportions = p_obs.mean(dim=0)  # [num_bins]
        
        # Inverse frequency with smoothing
        weights = 1.0 / (avg_proportions + self.smoothing)
        
        # Normalize weights to sum to num_bins (so average weight is 1.0)
        weights = weights * self.num_bins / weights.sum()
        
        return weights
    
    def wasserstein2_loss(self, p_obs, p_pred):
        """
        Compute Wasserstein-2 distance between two discrete distributions.
        
        For 1D distributions with ordered bins, uses cumulative distribution difference.
        
        Args:
            p_obs: [B, num_bins] - observed distribution
            p_pred: [B, num_bins] - predicted distribution
            
        Returns:
            Scalar Wasserstein-2 distance (squared)
        """
        # Compute cumulative distributions
        cdf_obs = torch.cumsum(p_obs, dim=1)  # [B, num_bins]
        cdf_pred = torch.cumsum(p_pred, dim=1)  # [B, num_bins]
        
        # Compute bin widths (distance between consecutive midpoints)
        bin_widths = self.bin_midpoints[1:] - self.bin_midpoints[:-1]  # [num_bins-1]
        # Pad to match num_bins
        bin_widths = torch.cat([bin_widths, bin_widths[-1:]])  # [num_bins]
        
        # W2^2 distance (squared for differentiability)
        w2_sq = torch.sum(((cdf_obs - cdf_pred) ** 2) * bin_widths.unsqueeze(0), dim=1)  # [B]
        
        return w2_sq.mean()
    
    def forward(self, changes_obs, changes_pred, mask=None):
        """
        Compute histogram loss between observed and predicted changes.
        
        Args:
            changes_obs: [B, H, W] - observed changes (target - last_input)
            changes_pred: [B, H, W] - predicted changes (pred - last_input)
            mask: [B, H, W] - validity mask
            
        Returns:
            total_loss: Combined loss
            ce_loss: Class-balanced CE component
            w2_loss: Wasserstein-2 component
            p_obs: Observed histogram proportions (for logging)
            p_pred: Predicted histogram proportions (for logging)
        """
        # Compute histograms
        counts_obs, p_obs = compute_histogram(changes_obs, self.bin_edges, mask=mask)
        counts_pred, p_pred = compute_histogram(changes_pred, self.bin_edges, mask=mask)
        
        # Apply label smoothing to predicted probabilities to prevent extreme log values
        if self.label_smoothing > 0:
            p_pred = (1 - self.label_smoothing) * p_pred + self.label_smoothing / self.num_bins
        
        # Compute class weights from observed distribution
        class_weights = self.compute_class_weights(p_obs)  # [num_bins]
        
        # Class-balanced Cross-Entropy
        # Treat as classification: p_obs is target, p_pred is prediction
        # Weight each bin by inverse frequency
        log_p_pred = torch.log(p_pred + 1e-8)  # [B, num_bins]
        weighted_ce = -p_obs * log_p_pred * class_weights.unsqueeze(0)  # [B, num_bins]
        ce_loss = weighted_ce.sum(dim=1).mean()  # Average over batch
        
        # Wasserstein-2 loss
        w2_loss = self.wasserstein2_loss(p_obs, p_pred)
        
        # Combined loss (normalized by num_bins for scale compatibility with other losses)
        total_loss = (ce_loss + self.lambda_w2 * w2_loss) / self.num_bins
        
        return total_loss, ce_loss, w2_loss, p_obs, p_pred
