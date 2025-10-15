"""
Histogram loss for tile-level distribution prediction.

Combines cross-entropy and Wasserstein-2 distance to train a histogram head
that predicts the distribution of change magnitudes across a tile.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HistogramLoss(nn.Module):
    """
    Loss for histogram prediction head.
    
    Combines:
    1. Cross-entropy: CE(p_obs, p_hat) - standard classification loss
    2. Wasserstein-2: W2(p_obs, p_hat) - metric on probability distributions
    
    The Wasserstein distance uses squared ground distances between bin midpoints,
    making it sensitive to how far apart the predicted and observed distributions are.
    """
    
    def __init__(self, bin_midpoints, lambda_w2=0.1):
        """
        Args:
            bin_midpoints: Tensor of shape [num_bins] with midpoint values for each bin
            lambda_w2: Weight for Wasserstein-2 term (default: 0.1)
        """
        super().__init__()
        self.register_buffer('bin_midpoints', bin_midpoints)
        self.lambda_w2 = lambda_w2
        self.num_bins = len(bin_midpoints)
        
        # Precompute pairwise squared distances between bin midpoints
        # D[i, j] = (midpoint_i - midpoint_j)^2
        midpoints_expanded = bin_midpoints.unsqueeze(0)  # [1, num_bins]
        pairwise_diffs = midpoints_expanded - midpoints_expanded.T  # [num_bins, num_bins]
        self.register_buffer('pairwise_sq_dists', pairwise_diffs ** 2)
    
    def wasserstein2_loss(self, p_obs, p_hat):
        """
        Compute Wasserstein-2 distance between two discrete distributions.
        
        Uses the squared ground distance matrix between bin midpoints.
        
        W2(p, q) = sqrt(sum_ij D[i,j] * T[i,j])
        
        where T is the optimal transport plan. For 1D distributions with ordered bins,
        we can use a simpler closed-form based on cumulative distributions.
        
        Args:
            p_obs: [B, num_bins] - observed distribution (ground truth)
            p_hat: [B, num_bins] - predicted distribution
            
        Returns:
            Scalar Wasserstein-2 distance (squared, for differentiability)
        """
        # For 1D distributions, W2^2 = sum_i (CDF_p(i) - CDF_q(i))^2 * (x_{i+1} - x_i)^2
        # Simplified: use cumulative sum difference
        
        # Compute cumulative distributions
        cdf_obs = torch.cumsum(p_obs, dim=1)  # [B, num_bins]
        cdf_hat = torch.cumsum(p_hat, dim=1)  # [B, num_bins]
        
        # Compute bin widths (distance between consecutive midpoints)
        bin_widths = self.bin_midpoints[1:] - self.bin_midpoints[:-1]  # [num_bins-1]
        # Pad to match num_bins
        bin_widths = torch.cat([bin_widths, bin_widths[-1:]])  # [num_bins]
        
        # W2^2 distance (squared for differentiability)
        w2_sq = torch.sum(((cdf_obs - cdf_hat) ** 2) * bin_widths.unsqueeze(0), dim=1)  # [B]
        
        return w2_sq.mean()
    
    def forward(self, p_obs, p_hat):
        """
        Compute combined histogram loss.
        
        Args:
            p_obs: [B, num_bins] - observed histogram (proportions, sum to 1)
            p_hat: [B, num_bins] - predicted histogram (probabilities from softmax)
            
        Returns:
            Combined loss: CE + lambda_w2 * W2
        """
        # Cross-entropy loss
        # KL divergence is equivalent to CE when p_obs is the target
        ce_loss = F.kl_div(
            torch.log(p_hat + 1e-8),  # log probabilities
            p_obs,  # target distribution
            reduction='batchmean'
        )
        
        # Wasserstein-2 loss
        w2_loss = self.wasserstein2_loss(p_obs, p_hat)
        
        # Combined loss
        total_loss = ce_loss + self.lambda_w2 * w2_loss
        
        return total_loss, ce_loss, w2_loss


def compute_observed_histogram(changes, bin_edges, mask=None):
    """
    Compute observed histogram from continuous pixel changes.
    
    Args:
        changes: [B, H, W] - continuous change values (target - last_input)
        bin_edges: [num_bins + 1] - histogram bin edges
        mask: [B, H, W] - optional boolean mask for valid pixels
        
    Returns:
        p_obs: [B, num_bins] - observed histogram proportions (sum to 1 per sample)
    """
    B, H, W = changes.shape
    num_bins = len(bin_edges) - 1
    device = changes.device
    
    # Initialize histogram
    p_obs = torch.zeros(B, num_bins, device=device, dtype=torch.float32)
    
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
            p_obs[b] = 1.0 / num_bins
            continue
        
        # Compute histogram using torch.histc or manual binning
        # torch.histc doesn't support custom bin edges, so we use manual binning
        for i in range(num_bins):
            left_edge = bin_edges[i]
            right_edge = bin_edges[i + 1]
            
            if i == num_bins - 1:
                # Last bin includes right edge
                in_bin = (change_values >= left_edge) & (change_values <= right_edge)
            else:
                in_bin = (change_values >= left_edge) & (change_values < right_edge)
            
            p_obs[b, i] = in_bin.sum().float()
        
        # Normalize to proportions
        total_count = p_obs[b].sum()
        if total_count > 0:
            p_obs[b] = p_obs[b] / total_count
        else:
            p_obs[b] = 1.0 / num_bins
    
    return p_obs
