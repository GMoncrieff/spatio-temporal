"""
Master script to run all preprocessing steps.

This script runs all preprocessing in the correct order:
1. Normalization statistics
2. Valid chips (requires stats for consistency)
3. Histogram weights (requires dataloader with cached stats/chips)

Usage:
    python scripts/preprocess/run_all_preprocessing.py
    python scripts/preprocess/run_all_preprocessing.py --force  # Recompute all
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

CACHE_DIR = os.path.join("data", "cache")
PREPROCESS_DIR = os.path.join("scripts", "preprocess")


def file_exists(filepath):
    """Check if file exists and is not empty."""
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0


def run_script(script_name, args=None):
    """Run a preprocessing script."""
    script_path = os.path.join(PREPROCESS_DIR, script_name)
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("=" * 70)
    
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all preprocessing steps")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cache exists"
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=os.path.join("scripts", "notebooks", "hm.icechunk"),
        help="Path to Zarr dataset"
    )
    parser.add_argument(
        "--chip_size",
        type=int,
        default=128,
        help="Chip size"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Stride between chips"
    )
    parser.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.5,
        help="Minimum valid pixel ratio"
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="auto",
        choices=["auto", "aws", "m1_mac"],
        help="Platform for histogram weight computation"
    )
    
    args = parser.parse_args()
    
    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    print("=" * 70)
    print("PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Force recompute: {args.force}")
    print()
    
    # Step 1: Normalization statistics
    stats_file = os.path.join(CACHE_DIR, "normalization_stats.pkl")
    
    if args.force or not file_exists(stats_file):
        print("\n[1/3] Computing normalization statistics...")
        run_script("precompute_dataset_stats.py", [
            "--zarr_path", args.zarr_path,
            "--chip_size", str(args.chip_size),
        ])
    else:
        print(f"\n[1/3] Normalization statistics already exist: {stats_file}")
        print("      Use --force to recompute")
    
    # Step 2: Valid chips
    valid_chips_file = os.path.join(
        CACHE_DIR,
        f"valid_chips_{args.chip_size}_{args.stride}_{args.min_valid_ratio}.pkl"
    )
    
    if args.force or not file_exists(valid_chips_file):
        print("\n[2/3] Computing valid chips...")
        run_script("precompute_valid_chips.py", [
            "--zarr_path", args.zarr_path,
            "--chip_size", str(args.chip_size),
            "--stride", str(args.stride),
            "--min_valid_ratio", str(args.min_valid_ratio),
        ])
    else:
        print(f"\n[2/3] Valid chips already exist: {valid_chips_file}")
        print("      Use --force to recompute")
    
    # Step 3: Histogram weights
    histogram_file = os.path.join(CACHE_DIR, "histogram_weights.pkl")
    
    if args.force or not file_exists(histogram_file):
        print("\n[3/3] Computing histogram weights...")
        run_script("precompute_histogram_weights.py", [
            "--zarr_path", args.zarr_path,
            "--platform", args.platform,
        ])
    else:
        print(f"\n[3/3] Histogram weights already exist: {histogram_file}")
        print("      Use --force to recompute")
    
    # Summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print("\nCached files:")
    
    for filename in os.listdir(CACHE_DIR):
        filepath = os.path.join(CACHE_DIR, filename)
        if os.path.isfile(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  {filename}: {size_kb:.2f} KB")
    
    print()


if __name__ == "__main__":
    main()
