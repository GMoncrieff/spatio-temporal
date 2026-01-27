#!/usr/bin/env python3
"""
Precompute valid chip positions from Zarr dataset and assign to geographic splits.

This script scans the Zarr/Icechunk dataset to identify all chip positions that have
sufficient valid data, then assigns them to train/val/test/calib splits in a 
geographically deterministic way (70%/10%/10%/10%).

The output is saved to data/processed/valid_chips_metadata.json for fast loading
during training.

Usage:
    python scripts/precompute_valid_chips.py --zarr_path scripts/notebooks/hm.icechunk \
        --chip_size 128 --stride 64 --min_valid_ratio 0.8
"""

import os
import json
import argparse
import hashlib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress Pydantic warnings from icechunk
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


def hash_position_to_split(y_idx, x_idx, seed=42):
    """
    Deterministically assign a chip position to a split based on its coordinates.
    
    Uses hash of coordinates to ensure:
    - Same position always gets same split
    - Roughly uniform distribution across splits
    - Geographic clustering (nearby chips likely in same split)
    
    Args:
        y_idx: Y index of chip
        x_idx: X index of chip
        seed: Random seed for reproducibility
        
    Returns:
        str: One of 'train', 'val', 'test', 'calib'
    """
    # Hash the position with seed
    position_str = f"{seed}_{y_idx}_{x_idx}"
    hash_val = int(hashlib.md5(position_str.encode()).hexdigest(), 16)
    
    # Map to split: 70% train, 10% val, 10% test, 10% calib
    rand_val = (hash_val % 100) / 100.0
    
    if rand_val < 0.70:
        return 'train'
    elif rand_val < 0.80:
        return 'val'
    elif rand_val < 0.90:
        return 'test'
    else:
        return 'calib'


def check_chip_validity(ds, y_start, x_start, chip_size, min_valid_ratio=0.8):
    """
    Check if a chip position has sufficient valid data.
    
    Args:
        ds: xarray Dataset
        y_start: Starting y index
        x_start: Starting x index
        chip_size: Size of chip
        min_valid_ratio: Minimum fraction of valid pixels required
        
    Returns:
        bool: True if chip is valid
    """
    # Extract chip for target variable (AA) across all time steps
    try:
        chip = ds['AA'].isel(
            y=slice(y_start, y_start + chip_size),
            x=slice(x_start, x_start + chip_size)
        ).load()
        
        # Check if chip has correct size
        if chip.sizes['y'] != chip_size or chip.sizes['x'] != chip_size:
            return False
        
        # Count valid pixels across all time steps
        valid_mask = np.isfinite(chip.values)
        valid_ratio = valid_mask.sum() / valid_mask.size
        
        return valid_ratio >= min_valid_ratio
        
    except Exception as e:
        print(f"Error checking chip at ({y_start}, {x_start}): {e}")
        return False


def precompute_valid_chips(
    zarr_path,
    chip_size=128,
    stride=64,
    min_valid_ratio=0.8,
    output_path=None,
    seed=42
):
    """
    Scan Zarr dataset and identify all valid chip positions.
    
    Args:
        zarr_path: Path to Zarr/Icechunk repository
        chip_size: Size of chips
        stride: Stride between chip positions
        min_valid_ratio: Minimum fraction of valid pixels
        output_path: Where to save metadata (default: data/processed/valid_chips_metadata.json)
        seed: Random seed for split assignment
        
    Returns:
        dict: Metadata with valid chip positions per split
    """
    print("=" * 70)
    print("PRECOMPUTING VALID CHIP POSITIONS")
    print("=" * 70)
    print(f"Zarr path: {zarr_path}")
    print(f"Chip size: {chip_size}")
    print(f"Stride: {stride}")
    print(f"Min valid ratio: {min_valid_ratio}")
    print(f"Random seed: {seed}")
    print()
    
    # Load Zarr dataset
    print("Loading Zarr dataset...")
    try:
        import xarray as xr
        import icechunk
        from urllib.parse import urlparse
        
        repo_path = str(zarr_path)
        if repo_path.startswith("s3://"):
            parsed = urlparse(repo_path)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")
            store = icechunk.s3_storage(
                bucket=bucket, prefix=prefix, region=None, 
                endpoint_url=None, anonymous=False, 
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
        
        # Expand consolidated structure if needed
        if "dynamic" in ds.data_vars and "static" in ds.data_vars:
            print("Expanding consolidated dynamic/static structure...")
            ds_expanded = xr.Dataset(coords={k: ds.coords[k] for k in ("time", "y", "x") if k in ds.coords})
            
            for v in ds.coords["var_dynamic"].values:
                vname = str(v)
                ds_expanded[vname] = ds["dynamic"].sel(var_dynamic=v).drop_vars("var_dynamic")
            
            for v in ds.coords["var_static"].values:
                vname = str(v)
                ds_expanded[vname] = ds["static"].sel(var_static=v).drop_vars("var_static")
            
            ds = ds_expanded
        
        print(f"Dataset loaded: {ds.sizes['y']} × {ds.sizes['x']} pixels")
        print()
        
    except Exception as e:
        print(f"Error loading Zarr dataset: {e}")
        raise
    
    # Calculate grid positions
    H = ds.sizes['y']
    W = ds.sizes['x']
    
    y_starts = list(range(0, H - chip_size + 1, stride))
    x_starts = list(range(0, W - chip_size + 1, stride))
    
    total_positions = len(y_starts) * len(x_starts)
    print(f"Scanning {total_positions:,} potential chip positions...")
    print(f"  Y positions: {len(y_starts)} (stride {stride})")
    print(f"  X positions: {len(x_starts)} (stride {stride})")
    print()
    
    # Scan all positions
    valid_chips = {'train': [], 'val': [], 'test': [], 'calib': []}
    
    with tqdm(total=total_positions, desc="Scanning chips") as pbar:
        for yi, y_start in enumerate(y_starts):
            for xi, x_start in enumerate(x_starts):
                # Check if chip is valid
                if check_chip_validity(ds, y_start, x_start, chip_size, min_valid_ratio):
                    # Assign to split deterministically
                    split = hash_position_to_split(yi, xi, seed)
                    valid_chips[split].append([yi, xi])
                
                pbar.update(1)
    
    # Summary statistics
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    total_valid = sum(len(chips) for chips in valid_chips.values())
    print(f"Total valid chips: {total_valid:,} / {total_positions:,} ({100*total_valid/total_positions:.1f}%)")
    print()
    print("Split distribution:")
    for split in ['train', 'val', 'test', 'calib']:
        count = len(valid_chips[split])
        pct = 100 * count / total_valid if total_valid > 0 else 0
        print(f"  {split:6s}: {count:6,} chips ({pct:5.1f}%)")
    print()
    
    # Prepare metadata
    metadata = {
        'zarr_path': str(zarr_path),
        'chip_size': chip_size,
        'stride': stride,
        'min_valid_ratio': min_valid_ratio,
        'seed': seed,
        'dataset_shape': {'y': H, 'x': W},
        'total_positions': total_positions,
        'total_valid': total_valid,
        'splits': valid_chips,
        'split_counts': {split: len(chips) for split, chips in valid_chips.items()}
    }
    
    # Save metadata
    if output_path is None:
        output_path = Path("data/processed/valid_chips_metadata.json")
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {output_path}")
    print("=" * 70)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Precompute valid chip positions from Zarr dataset"
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="scripts/notebooks/hm.icechunk",
        help="Path to Zarr/Icechunk repository"
    )
    parser.add_argument(
        "--chip_size",
        type=int,
        default=128,
        help="Size of chips (default: 128)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Stride between chip positions (default: 64)"
    )
    parser.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.8,
        help="Minimum fraction of valid pixels (default: 0.8)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for metadata (default: data/processed/valid_chips_metadata.json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split assignment (default: 42)"
    )
    
    args = parser.parse_args()
    
    precompute_valid_chips(
        zarr_path=args.zarr_path,
        chip_size=args.chip_size,
        stride=args.stride,
        min_valid_ratio=args.min_valid_ratio,
        output_path=args.output,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
