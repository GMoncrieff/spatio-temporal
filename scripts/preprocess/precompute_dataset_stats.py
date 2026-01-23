"""
Precompute normalization statistics for the Zarr dataset.

This script computes per-variable mean/std for:
- HM target variable (AA)
- Dynamic covariates (AG, BU, EX, FR, HI, NS, PO, TI, gdp, population)
- Static variables (ele, tas, tasmin, pr, dpi_dsi, iucn_nostrict, iucn_strict)

Output: data/cache/normalization_stats.pkl
"""

import os
import sys
import pickle
import argparse
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
OUTPUT_FILE = os.path.join(CACHE_DIR, "normalization_stats.pkl")

years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
HM_VARS = ["AG", "BU", "EX", "FR", "HI", "NS", "PO", "TI", "gdp", "population"]


def compute_normalization_stats(
    zarr_path: str,
    chip_size: int = 128,
    stat_samples: int = 100,
    random_seed: int = 42
):
    """Compute per-variable normalization statistics."""
    print("=" * 70)
    print("COMPUTING NORMALIZATION STATISTICS")
    print("=" * 70)
    print(f"Zarr path: {zarr_path}")
    print(f"Chip size: {chip_size}")
    print(f"Samples per variable: {stat_samples}")
    print(f"Random seed: {random_seed}")
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
    print(f"Data variables: {list(ds.data_vars)}")
    print()
    
    # Expand consolidated arrays if needed
    if "dynamic" in ds.data_vars and "static" in ds.data_vars:
        print("Expanding consolidated dynamic/static arrays...")
        dynamic_vars = ["AA"] + HM_VARS
        for var_name in dynamic_vars:
            if var_name not in ds.data_vars:
                var_idx = dynamic_vars.index(var_name)
                ds[var_name] = ds["dynamic"].isel(var_dynamic=var_idx)
        
        static_var_names = [
            "ele", "tas", "tasmin", "pr", "dpi_dsi", 
            "iucn_nostrict", "iucn_strict"
        ]
        for var_name in static_var_names:
            if var_name not in ds.data_vars:
                var_idx = static_var_names.index(var_name)
                ds[var_name] = ds["static"].isel(var_static=var_idx)
    
    H, W = ds.dims["y"], ds.dims["x"]
    T = ds.dims["time"]
    
    rng = np.random.default_rng(random_seed)
    
    # 1. HM (AA) statistics
    print("Computing HM (AA) statistics...")
    hm_samples = []
    per_time = max(1, stat_samples // T)
    
    for t_idx in range(T):
        for _ in range(per_time):
            i = rng.integers(0, max(1, H - chip_size))
            j = rng.integers(0, max(1, W - chip_size))
            arr = ds["AA"].isel(
                time=t_idx,
                y=slice(i, i + chip_size),
                x=slice(j, j + chip_size)
            ).load().values
            hm_samples.append(arr)
    
    hm_stack = np.stack(hm_samples, axis=0)
    hm_mean = float(np.nanmean(hm_stack))
    hm_std = float(np.nanstd(hm_stack)) + 1e-8
    print(f"  AA: mean={hm_mean:.6f}, std={hm_std:.6f}")
    print()
    
    # 2. Static variable statistics
    print("Computing static variable statistics...")
    static_var_names = [
        "ele", "tas", "tasmin", "pr", "dpi_dsi", 
        "iucn_nostrict", "iucn_strict"
    ]
    
    static_means = []
    static_stds = []
    
    for var_name in static_var_names:
        samples = []
        for _ in range(max(1, stat_samples // len(static_var_names))):
            i = rng.integers(0, max(1, H - chip_size))
            j = rng.integers(0, max(1, W - chip_size))
            arr = ds[var_name].isel(
                y=slice(i, i + chip_size),
                x=slice(j, j + chip_size)
            ).load().values
            samples.append(arr)
        
        stack = np.stack(samples, axis=0)
        mean = float(np.nanmean(stack))
        std = float(np.nanstd(stack)) + 1e-8
        static_means.append(mean)
        static_stds.append(std)
        print(f"  {var_name}: mean={mean:.6f}, std={std:.6f}")
    
    print()
    
    # 3. Component covariate statistics
    print("Computing component covariate statistics...")
    comp_means = {}
    comp_stds = {}
    
    for var_name in HM_VARS:
        samples = []
        per_var = max(1, stat_samples // (len(HM_VARS) * T))
        
        for t_idx in range(T):
            for _ in range(per_var):
                i = rng.integers(0, max(1, H - chip_size))
                j = rng.integers(0, max(1, W - chip_size))
                arr = ds[var_name].isel(
                    time=t_idx,
                    y=slice(i, i + chip_size),
                    x=slice(j, j + chip_size)
                ).load().values
                samples.append(arr)
        
        stack = np.stack(samples, axis=0)
        mean = float(np.nanmean(stack))
        std = float(np.nanstd(stack)) + 1e-8
        comp_means[var_name] = mean
        comp_stds[var_name] = std
        print(f"  {var_name}: mean={mean:.6f}, std={std:.6f}")
    
    print()
    
    # Package results
    stats = {
        "hm_mean": hm_mean,
        "hm_std": hm_std,
        "static_means": static_means,
        "static_stds": static_stds,
        "static_var_names": static_var_names,
        "comp_means": comp_means,
        "comp_stds": comp_stds,
        "comp_var_names": HM_VARS,
        "chip_size": chip_size,
        "stat_samples": stat_samples,
        "random_seed": random_seed,
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Precompute normalization statistics")
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
        help="Chip size for sampling"
    )
    parser.add_argument(
        "--stat_samples",
        type=int,
        default=100,
        help="Number of samples per variable"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILE,
        help="Output pickle file"
    )
    
    args = parser.parse_args()
    
    # Create cache directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Compute statistics
    stats = compute_normalization_stats(
        zarr_path=args.zarr_path,
        chip_size=args.chip_size,
        stat_samples=args.stat_samples,
        random_seed=args.random_seed
    )
    
    # Save to pickle
    print(f"Saving statistics to: {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print()
    print("=" * 70)
    print("NORMALIZATION STATISTICS SAVED")
    print("=" * 70)
    print(f"Output file: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.2f} KB")
    print()


if __name__ == "__main__":
    main()
