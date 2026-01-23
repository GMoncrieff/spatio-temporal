"""
Precompute histogram bin weights for each prediction horizon.

This script:
1. Loads training data batches
2. Computes histogram of change values for each horizon
3. Calculates inverse frequency weights for balanced loss

Output: data/cache/histogram_weights.pkl
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Constants
CACHE_DIR = os.path.join("data", "cache")
OUTPUT_FILE = os.path.join(CACHE_DIR, "histogram_weights.pkl")


def compute_histogram_weights(
    train_loader: DataLoader,
    num_batches: int = 10,
    device: str = "cpu"
):
    """Compute histogram bin weights from training data."""
    print("=" * 70)
    print("COMPUTING HISTOGRAM BIN WEIGHTS")
    print("=" * 70)
    print(f"Number of batches to sample: {num_batches}")
    print(f"Device: {device}")
    print()
    
    # Histogram bins (same as in lightning_module.py)
    histogram_bins = np.array([
        -1.0, -0.005, 0.005, 0.020, 0.100, 0.200, 0.400, 0.600, 1.0
    ])
    num_bins = len(histogram_bins) - 1
    
    horizon_names = ['5yr', '10yr', '15yr', '20yr']
    
    # Collect change values for each horizon
    horizon_changes = {h: [] for h in horizon_names}
    
    print("Collecting change values from training batches...")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        print(f"  Processing batch {batch_idx + 1}/{num_batches}...")
        
        input_dynamic = batch['input_dynamic'].to(device)
        last_input = input_dynamic[:, -1, 0]  # [B, H, W]
        
        for h_name in horizon_names:
            target_key = f'target_{h_name}'
            if target_key not in batch:
                continue
            
            target = batch[target_key].to(device)
            
            # Compute mask for valid pixels
            mask = torch.isfinite(target) & torch.isfinite(last_input)
            
            if mask.sum() == 0:
                continue
            
            # Compute change
            delta = (target - last_input)[mask]
            horizon_changes[h_name].extend(delta.cpu().numpy().tolist())
    
    print()
    
    # Compute histogram and weights for each horizon
    weights_dict = {}
    
    for h_name in horizon_names:
        changes = np.array(horizon_changes[h_name])
        
        if len(changes) == 0:
            print(f"WARNING: No valid changes for {h_name} horizon")
            weights_dict[h_name] = np.ones(num_bins, dtype=np.float32)
            continue
        
        # Compute histogram
        counts, _ = np.histogram(changes, bins=histogram_bins)
        total = counts.sum()
        
        # Compute weights (inverse frequency)
        proportions = counts / total
        weights = np.zeros(num_bins, dtype=np.float32)
        
        for i in range(num_bins):
            if proportions[i] > 0:
                weights[i] = 1.0 / (num_bins * proportions[i])
            else:
                weights[i] = 0.0
        
        weights_dict[h_name] = weights
        
        # Print summary
        print(f"--- {h_name} Horizon ---")
        print("Bin | Count  | Proportion | Weight")
        print("-" * 50)
        
        bin_labels = [
            "[-1.000, -0.005)",
            "[-0.005, +0.005)",
            "[+0.005, +0.020)",
            "[+0.020, +0.100)",
            "[+0.100, +0.200)",
            "[+0.200, +0.400)",
            "[+0.400, +0.600)",
            "[+0.600, +1.000)"
        ]
        
        for i in range(num_bins):
            print(f" {i}  | {counts[i]:6d} | {proportions[i]:10.4f} | {weights[i]:6.3f}  {bin_labels[i]}")
        
        print()
    
    # Package results
    result = {
        "weights": weights_dict,
        "histogram_bins": histogram_bins,
        "num_bins": num_bins,
        "horizon_names": horizon_names,
        "num_batches": num_batches,
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Precompute histogram weights")
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=os.path.join("scripts", "notebooks", "hm.icechunk"),
        help="Path to Zarr dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for sampling"
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of batches to sample"
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="auto",
        choices=["auto", "aws", "m1_mac"],
        help="Platform for dataloader settings"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILE,
        help="Output pickle file"
    )
    
    args = parser.parse_args()
    
    # Import dataloader
    from torchgeo_dataloader import get_dataloader
    
    # Create cache directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Create training dataloader
    print("Creating training dataloader...")
    train_loader = get_dataloader(
        split="train",
        batch_size=args.batch_size,
        zarr_path=args.zarr_path,
        platform=args.platform,
        chips_per_epoch=args.num_batches * args.batch_size,
    )
    print()
    
    # Compute histogram weights
    result = compute_histogram_weights(
        train_loader=train_loader,
        num_batches=args.num_batches,
        device="cpu"
    )
    
    # Save to pickle
    print(f"Saving histogram weights to: {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print()
    print("=" * 70)
    print("HISTOGRAM WEIGHTS SAVED")
    print("=" * 70)
    print(f"Output file: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.2f} KB")
    print()


if __name__ == "__main__":
    main()
