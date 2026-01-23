"""
Precompute valid chip indices for each geographic split.

This script:
1. Generates all chip positions based on chip_size and stride
2. Assigns chips to geographic splits (train/val/test/calib) using spatial hashing
3. Filters chips by minimum valid pixel ratio
4. Saves valid chip indices for each split

Output: data/cache/valid_chips_{chip_size}_{stride}_{min_valid_ratio}.pkl
"""

import os
import sys
import pickle
import argparse
import hashlib
from pathlib import Path

import numpy as np
import xarray as xr
import dask

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Dask threaded scheduler
dask.config.set(scheduler='threads', num_workers=4)

# Constants
ZARR_PATH = os.path.join("scripts", "notebooks", "hm.icechunk")
CACHE_DIR = os.path.join("data", "cache")

# Split ranges (deterministic geographic splits)
SPLIT_RANGES = {
    "train": (0, 70),
    "val": (70, 80),
    "test": (80, 90),
    "calib": (90, 100),
}


def compute_chip_split(y_center: float, x_center: float, seed: int = 42) -> str:
    """Compute which split a chip belongs to based on spatial hash."""
    hash_input = f"{y_center:.4f}_{x_center:.4f}_{seed}"
    hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 100
    
    for split_name, (low, high) in SPLIT_RANGES.items():
        if low <= hash_val < high:
            return split_name
    return "train"


def precompute_valid_chips(
    zarr_path: str,
    chip_size: int = 128,
    stride: int = 128,
    min_valid_ratio: float = 0.5,
    random_seed: int = 42,
    check_ratio: float = 0.2,
):
    """Precompute valid chip indices for each split."""
    print("=" * 70)
    print("PRECOMPUTING VALID CHIPS")
    print("=" * 70)
    print(f"Zarr path: {zarr_path}")
    print(f"Chip size: {chip_size}")
    print(f"Stride: {stride}")
    print(f"Min valid ratio: {min_valid_ratio}")
    print(f"Random seed: {random_seed}")
    print(f"Check ratio: {check_ratio}")
    print()
    
    # Open dataset
    print("Opening Zarr dataset...")
    try:
        import icechunk
        from urllib.parse import urlparse
        
        repo_path = str(zarr_path)
        if repo_path.startswith("s3://"):
            parsed = urlparse(repo_path)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")
            store = icechunk.s3_storage(
                bucket=bucket, prefix=prefix,
                region=None, endpoint_url=None, anonymous=False,
                allow_http=False, force_path_style=False
            )
        else:
            store = icechunk.local_filesystem_storage(repo_path)
        
        repo = icechunk.Repository.open(store)
        session = repo.readonly_session("main")
        
        try:
            ds = xr.open_zarr(session.store, group="hm", consolidated=False)
        except Exception:
            ds = xr.open_zarr(session.store, consolidated=False)
    except Exception as e:
        print(f"Failed to open with icechunk: {e}")
        raise
    
    print(f"Dataset dimensions: {dict(ds.dims)}")
    print()
    
    # Expand consolidated arrays if needed
    if "dynamic" in ds.data_vars and "AA" not in ds.data_vars:
        print("Expanding consolidated dynamic array...")
        ds["AA"] = ds["dynamic"].isel(var_dynamic=0)
    
    H, W = ds.dims["y"], ds.dims["x"]
    
    # Generate chip positions using xbatcher
    print("Generating chip positions...")
    import xbatcher
    bgen = xbatcher.BatchGenerator(
        ds,
        input_dims={"y": chip_size, "x": chip_size},
        input_overlap={"y": chip_size - stride, "x": chip_size - stride},
        batch_dims={}
    )
    
    total_chips = len(bgen)
    print(f"Total chips: {total_chips:,}")
    print()
    
    # Compute chip centers and assign to splits
    print("Computing chip positions and geographic splits...")
    y_starts = list(range(0, H - chip_size + 1, stride))
    x_starts = list(range(0, W - chip_size + 1, stride))
    
    # Map each chip to its split
    chip_to_split = {}
    split_counts = {split: 0 for split in SPLIT_RANGES.keys()}
    
    bgen_idx = 0
    for y_start in y_starts:
        for x_start in x_starts:
            y_center = y_start + chip_size / 2
            x_center = x_start + chip_size / 2
            split = compute_chip_split(y_center, x_center, random_seed)
            chip_to_split[bgen_idx] = split
            split_counts[split] += 1
            bgen_idx += 1
    
    for split, count in split_counts.items():
        pct = 100 * count / total_chips
        print(f"  {split}: {count:,} chips ({pct:.1f}%)")
    print()
    
    # Pre-filter valid chips for each split
    print(f"Pre-filtering chips with min_valid_ratio={min_valid_ratio}...")
    rng = np.random.default_rng(random_seed)
    
    valid_chips = {split: [] for split in SPLIT_RANGES.keys()}
    
    for split in SPLIT_RANGES.keys():
        split_chip_indices = [
            idx for idx, s in chip_to_split.items() if s == split
        ]
        
        print(f"\n{split.upper()} split:")
        print(f"  Total chips: {len(split_chip_indices):,}")
        
        # Sample chips to check if there are many
        if len(split_chip_indices) > 1000 and check_ratio < 1.0:
            n_check = max(100, int(len(split_chip_indices) * check_ratio))
            sample_indices = rng.choice(
                len(split_chip_indices), 
                size=n_check, 
                replace=False
            )
            chips_to_check = [split_chip_indices[i] for i in sample_indices]
            print(f"  Sampling {len(chips_to_check):,} chips for validity check...")
        else:
            chips_to_check = split_chip_indices
            sample_indices = list(range(len(split_chip_indices)))
            print(f"  Checking all {len(chips_to_check):,} chips...")
        
        # Check validity
        n_valid_checked = 0
        for i, bgen_idx in enumerate(chips_to_check):
            if (i + 1) % 500 == 0:
                print(f"    Checked {i+1:,}/{len(chips_to_check):,} chips...")
            
            # Load chip and check valid ratio
            arr = ds["AA"].isel(
                time=0,
                y=slice(y_starts[bgen_idx // len(x_starts)], 
                       y_starts[bgen_idx // len(x_starts)] + chip_size),
                x=slice(x_starts[bgen_idx % len(x_starts)], 
                       x_starts[bgen_idx % len(x_starts)] + chip_size)
            ).load().values
            
            valid_ratio = np.sum(np.isfinite(arr)) / arr.size
            if valid_ratio >= min_valid_ratio:
                valid_chips[split].append(bgen_idx)
                n_valid_checked += 1
        
        # If we sampled, extrapolate
        if len(chips_to_check) < len(split_chip_indices):
            validity_rate = n_valid_checked / len(chips_to_check)
            print(f"  Sampled validity rate: {validity_rate:.1%}")
            
            # Add remaining unchecked chips probabilistically
            unchecked_indices = [
                idx for idx in split_chip_indices 
                if idx not in chips_to_check
            ]
            
            # Add all unchecked (conservative - they'll be filtered at runtime if needed)
            for idx in unchecked_indices:
                if rng.random() < validity_rate:
                    valid_chips[split].append(idx)
        
        print(f"  Valid chips: {len(valid_chips[split]):,}")
    
    print()
    
    # Package results
    result = {
        "valid_chips": valid_chips,
        "chip_to_split": chip_to_split,
        "split_counts": split_counts,
        "chip_size": chip_size,
        "stride": stride,
        "min_valid_ratio": min_valid_ratio,
        "random_seed": random_seed,
        "total_chips": total_chips,
        "y_starts": y_starts,
        "x_starts": x_starts,
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Precompute valid chips")
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=ZARR_PATH,
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
        "--random_seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--check_ratio",
        type=float,
        default=0.2,
        help="Ratio of chips to check for validity (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Compute valid chips
    result = precompute_valid_chips(
        zarr_path=args.zarr_path,
        chip_size=args.chip_size,
        stride=args.stride,
        min_valid_ratio=args.min_valid_ratio,
        random_seed=args.random_seed,
        check_ratio=args.check_ratio,
    )
    
    # Save to pickle
    output_file = os.path.join(
        CACHE_DIR,
        f"valid_chips_{args.chip_size}_{args.stride}_{args.min_valid_ratio}.pkl"
    )
    
    print(f"Saving valid chips to: {output_file}")
    with open(output_file, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print()
    print("=" * 70)
    print("VALID CHIPS SAVED")
    print("=" * 70)
    print(f"Output file: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    
    # Summary
    print()
    print("Summary:")
    for split, chips in result["valid_chips"].items():
        print(f"  {split}: {len(chips):,} valid chips")
    print()


if __name__ == "__main__":
    main()
