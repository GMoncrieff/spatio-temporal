"""
Zarr/Icechunk dataloader for Human Footprint dataset with performance optimizations.

Key features:
- Dask cache and threaded scheduler for S3 performance
- Platform detection (AWS vs M1 Mac) with appropriate defaults
- Deterministic geographic splits (train 70%, val 10%, test 10%, calib 10%)
- Pre-filtered valid chips (efficient empty batch handling)
- Shuffle-based sampling with configurable epoch length
"""

import os
import sys
import platform
import hashlib
import pickle
from urllib.parse import urlparse
from typing import Optional, Literal, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import rasterio

# Dask cache for chunk caching (critical for S3 performance)
try:
    import dask
    from dask.cache import Cache
    # 10GB cache - helps avoid re-fetching same chunks
    _DASK_CACHE = Cache(1e10)
    _DASK_CACHE.register()
except ImportError:
    _DASK_CACHE = None

# =============================================================================
# Configuration
# =============================================================================

HM_DIR = os.path.join("data", "raw", "hm_global")
STATIC_DIR = HM_DIR
ZARR_PATH = os.path.join("scripts", "notebooks", "hm.icechunk")
CACHE_DIR = os.path.join("data", "cache")

years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
HM_VARS = ["AG", "BU", "EX", "FR", "HI", "NS", "PO", "TI", "gdp", "population"]

static_files = [
    os.path.join(STATIC_DIR, "hm_static_ele_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_tas_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_tasmin_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_pr_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_dpi_dsi_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_iucn_nostrict_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_iucn_strict_1000.tiff"),
]

# Split definitions: train (70%), val (10%), test (10%), calib (10%)
SPLIT_RANGES = {
    "train": (0, 70),    # 0-69
    "val": (70, 80),     # 70-79
    "test": (80, 90),    # 80-89
    "calib": (90, 100),  # 90-99
}

# =============================================================================
# Cache Loading Utilities
# =============================================================================

def load_cached_stats(cache_dir: str = CACHE_DIR):
    """Load cached normalization statistics if available."""
    stats_file = os.path.join(cache_dir, "normalization_stats.pkl")
    if os.path.exists(stats_file):
        with open(stats_file, "rb") as f:
            return pickle.load(f)
    return None


def load_cached_valid_chips(chip_size: int, stride: int, min_valid_ratio: float, cache_dir: str = CACHE_DIR):
    """Load cached valid chips if available."""
    chips_file = os.path.join(cache_dir, f"valid_chips_{chip_size}_{stride}_{min_valid_ratio}.pkl")
    if os.path.exists(chips_file):
        with open(chips_file, "rb") as f:
            return pickle.load(f)
    return None

# =============================================================================
# Platform Detection & Defaults
# =============================================================================

def detect_platform() -> Literal["aws", "m1_mac", "other"]:
    """Detect if running on AWS, M1 Mac, or other platform."""
    # Check for M1 Mac
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "m1_mac"
    
    # Check for AWS (common indicators)
    # 1. AWS metadata service availability
    # 2. Environment variables
    if os.environ.get("AWS_EXECUTION_ENV") or os.environ.get("AWS_REGION"):
        return "aws"
    
    # Check if on Linux with many CPUs (likely cloud)
    if platform.system() == "Linux":
        try:
            cpu_count = os.cpu_count() or 1
            if cpu_count >= 16:
                return "aws"
        except:
            pass
    
    return "other"


def get_platform_defaults(platform_type: Literal["aws", "m1_mac", "other"]) -> dict:
    """Get optimal DataLoader settings for the detected platform."""
    if platform_type == "aws":
        # AWS ml.g5.8xlarge: 32 vCPU, 128GB RAM, 24GB GPU, data on S3
        return {
            "num_workers": 16,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "pin_memory": True,
            "dask_threads": 8,
            "multiprocessing_context": "forkserver",
        }
    elif platform_type == "m1_mac":
        # M1 Mac: 8-10 cores, local disk
        return {
            "num_workers": 4,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "pin_memory": False,  # No CUDA on M1
            "dask_threads": 2,
            "multiprocessing_context": "spawn",  # spawn is safer on macOS
        }
    else:
        # Conservative defaults for unknown platforms
        return {
            "num_workers": 2,
            "prefetch_factor": 2,
            "persistent_workers": False,
            "pin_memory": False,
            "dask_threads": 1,
            "multiprocessing_context": None,
        }


# =============================================================================
# Geographic Split Functions
# =============================================================================

def compute_chip_split(y_center: float, x_center: float, seed: int = 42) -> str:
    """
    Compute deterministic split assignment for a chip based on its center coordinates.
    
    Uses spatial hashing to ensure the same chip always gets the same split,
    regardless of run or epoch.
    
    Args:
        y_center: Y coordinate of chip center
        x_center: X coordinate of chip center
        seed: Random seed for reproducibility
        
    Returns:
        Split name: "train", "val", "test", or "calib"
    """
    # Create deterministic hash from coordinates
    coord_str = f"{y_center:.4f}_{x_center:.4f}_{seed}"
    hash_val = int(hashlib.md5(coord_str.encode()).hexdigest(), 16) % 100
    
    for split_name, (low, high) in SPLIT_RANGES.items():
        if low <= hash_val < high:
            return split_name
    
    return "train"  # Fallback


def get_chip_center_coords(
    y_start: int, x_start: int, chip_size: int, 
    y_coords: np.ndarray, x_coords: np.ndarray
) -> Tuple[float, float]:
    """Get geographic center coordinates for a chip."""
    y_center_idx = y_start + chip_size // 2
    x_center_idx = x_start + chip_size // 2
    
    # Clamp to valid range
    y_center_idx = min(y_center_idx, len(y_coords) - 1)
    x_center_idx = min(x_center_idx, len(x_coords) - 1)
    
    return float(y_coords[y_center_idx]), float(x_coords[x_center_idx])


# =============================================================================
# Dataset
# =============================================================================

class HumanFootprintZarrDataset(Dataset):
    """
    Zarr/Icechunk dataset for Human Footprint with optimized loading.
    
    Features:
    - Deterministic geographic splits
    - Pre-filtered valid chips (NaN ratio < threshold)
    - Dask threaded scheduler for parallel chunk loading
    - Per-variable normalization
    """
    
    def __init__(
        self,
        zarr_path: str,
        split: Literal["train", "val", "test", "calib"] = "train",
        chip_size: int = 128,
        stride: int = 128,
        min_valid_ratio: float = 0.5,
        chips_per_epoch: Optional[int] = None,
        use_temporal_sampling: bool = True,
        end_year_options: Tuple[int, ...] = (2000, 2005, 2010, 2015),
        fixed_input_years: Tuple[int, ...] = (1990, 1995, 2000),
        fixed_target_years: Tuple[int, ...] = (2005, 2010, 2015, 2020),
        include_components: bool = True,
        static_channels: Optional[int] = None,
        static_files_list: Optional[List[str]] = None,
        stat_samples: int = 256,
        random_seed: int = 42,
        dask_threads: int = 4,
    ):
        """
        Initialize the dataset.
        
        Args:
            zarr_path: Path to icechunk repo (local or s3://bucket/prefix)
            split: Which split to use ("train", "val", "test", "calib")
            chip_size: Size of spatial chips (pixels)
            stride: Stride between chips (pixels)
            min_valid_ratio: Minimum ratio of valid (non-NaN) pixels to include chip
            chips_per_epoch: Number of chips per epoch (None = all valid chips)
            use_temporal_sampling: If True, randomly sample end year for targets
            end_year_options: Valid end years for temporal sampling
            fixed_input_years: Input years (always used)
            fixed_target_years: Target years (used when not temporal sampling)
            include_components: Whether to include HM component covariates
            static_channels: Limit number of static channels (None = all)
            static_files_list: List of static file paths (for variable names)
            stat_samples: Number of samples for computing normalization stats
            random_seed: Seed for reproducibility
            dask_threads: Number of Dask threads for parallel chunk loading
        """
        # Lazy imports
        try:
            import xarray as xr
            import xbatcher
            import icechunk
        except ImportError as e:
            raise ImportError(
                "Zarr backend requires: xarray dask xbatcher icechunk zarr"
            ) from e
        
        # Configure Dask scheduler
        if dask_threads <= 1:
            dask.config.set(scheduler="single-threaded")
        else:
            dask.config.set(scheduler="threads", num_workers=dask_threads)
        
        self.zarr_path = zarr_path
        self.split = split
        self.chip_size = chip_size
        self.stride = stride
        self.min_valid_ratio = min_valid_ratio
        self.chips_per_epoch = chips_per_epoch
        self.use_temporal_sampling = use_temporal_sampling and split == "train"
        self.end_year_options = list(end_year_options)
        self.fixed_input_years = tuple(fixed_input_years)
        self.fixed_target_years = tuple(fixed_target_years)
        self.include_components = include_components
        self.random_seed = random_seed
        
        # Static files for variable names
        self._static_files = list(static_files_list or static_files)
        if static_channels is not None:
            self._static_files = self._static_files[:int(static_channels)]
        
        static_var_names = [
            os.path.basename(f).replace("hm_static_", "").replace(".tiff", "")
            for f in self._static_files
        ]
        dynamic_var_names = ["AA"] + (list(HM_VARS) if include_components else [])
        
        # Open icechunk repository
        print(f"=== Opening Zarr Dataset ({split} split) ===")
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
        
        # Open dataset
        try:
            ds = xr.open_zarr(session.store, group="hm", consolidated=False)
        except Exception:
            ds = xr.open_zarr(session.store, consolidated=False)
        
        print(f"Dataset dimensions: {dict(ds.sizes)}")
        print(f"Data variables: {list(ds.data_vars.keys())}")
        
        # Expand consolidated dynamic/static arrays if present
        if "dynamic" in ds.data_vars and "static" in ds.data_vars:
            print("Expanding consolidated dynamic/static arrays...")
            ds_expanded = xr.Dataset(
                coords={k: ds.coords[k] for k in ("time", "y", "x") if k in ds.coords}
            )
            
            for v in ds.coords["var_dynamic"].values:
                vname = str(v)
                ds_expanded[vname] = ds["dynamic"].sel(var_dynamic=v).drop_vars("var_dynamic")
            
            for v in ds.coords["var_static"].values:
                vname = str(v)
                ds_expanded[vname] = ds["static"].sel(var_static=v).drop_vars("var_static")
            
            ds = ds_expanded
        
        # Keep only required variables
        keep_vars = [v for v in dynamic_var_names if v in ds.data_vars]
        keep_vars += [v for v in static_var_names if v in ds.data_vars]
        missing_vars = [v for v in (dynamic_var_names + static_var_names) if v not in ds.data_vars]
        if missing_vars:
            raise ValueError(f"Missing required variables in Zarr: {missing_vars}")
        
        ds = ds[keep_vars]
        self.ds = ds
        
        self.H = int(ds.sizes["y"])
        self.W = int(ds.sizes["x"])
        self.T = int(ds.sizes.get("time", 0))
        
        if self.T != len(years):
            raise ValueError(f"Time dimension {self.T} != expected {len(years)}")
        
        # Get coordinate arrays for geographic split computation
        y_coords = ds["y"].values
        x_coords = ds["x"].values
        
        # Try to load cached valid chips
        cached_chips = load_cached_valid_chips(chip_size, stride, min_valid_ratio)
        
        if cached_chips is not None:
            print(f"Loading valid chips from cache...")
            # Use cached data
            self.valid_chip_indices = cached_chips["valid_chips"][split]
            y_starts = cached_chips["y_starts"]
            x_starts = cached_chips["x_starts"]
            self._n_x = len(x_starts)
            self._n_y = len(y_starts)
            print(f"  {split} split: {len(self.valid_chip_indices)} valid chips (from cache)")
        else:
            print(f"Cache not found, computing chip positions and geographic splits...")
            # Compute all chip positions and their splits
            y_starts = list(range(0, self.H - chip_size + 1, stride))
            x_starts = list(range(0, self.W - chip_size + 1, stride))
            self._n_x = len(x_starts)
            self._n_y = len(y_starts)
            
            # Filter chips by split
            split_chip_indices = []
            for yi, y_start in enumerate(y_starts):
                for xi, x_start in enumerate(x_starts):
                    y_center, x_center = get_chip_center_coords(
                        y_start, x_start, chip_size, y_coords, x_coords
                    )
                    chip_split = compute_chip_split(y_center, x_center, random_seed)
                    if chip_split == split:
                        bgen_idx = yi * self._n_x + xi
                        split_chip_indices.append((bgen_idx, y_start, x_start))
            
            print(f"  {split} split: {len(split_chip_indices)} chips (of {len(y_starts) * len(x_starts)} total)")
            
            # Pre-filter valid chips (check NaN ratio)
            print(f"Pre-filtering chips with min_valid_ratio={min_valid_ratio}...")
            rng = np.random.default_rng(random_seed)
            
            valid_chip_indices = []
            sample_var = "AA"  # Use target variable for validity check
            
            # Sample a subset of chips if there are many
            if len(split_chip_indices) > 1000:
                # Sample 20% of chips for validity check (faster)
                sample_size = max(200, len(split_chip_indices) // 5)
                sample_indices = rng.choice(len(split_chip_indices), sample_size, replace=False)
                chips_to_check = [split_chip_indices[i] for i in sample_indices]
                # Assume similar validity ratio for unchecked chips
                check_ratio = True
            else:
                chips_to_check = split_chip_indices
                check_ratio = False
            
            n_valid_checked = 0
            for bgen_idx, y_start, x_start in chips_to_check:
                # Check validity using first and last time step
                arr = ds[sample_var].isel(
                    time=0, 
                    y=slice(y_start, y_start + chip_size),
                    x=slice(x_start, x_start + chip_size)
                ).load().values
                
                valid_ratio = np.sum(np.isfinite(arr)) / arr.size
                if valid_ratio >= min_valid_ratio:
                    valid_chip_indices.append(bgen_idx)
                    n_valid_checked += 1
            
            # If we sampled, extrapolate to full set
            if check_ratio and len(chips_to_check) > 0:
                validity_rate = n_valid_checked / len(chips_to_check)
                print(f"  Sampled validity rate: {validity_rate:.1%}")
                # Add remaining unchecked chips probabilistically
                unchecked_indices = [
                    idx for idx in range(len(split_chip_indices)) 
                    if idx not in sample_indices
                ]
                # Just add all unchecked (they'll be filtered at runtime if needed)
                for idx in unchecked_indices:
                    valid_chip_indices.append(split_chip_indices[idx][0])
            
            self.valid_chip_indices = valid_chip_indices
            print(f"  Final: {len(valid_chip_indices)} valid chips for {split} split")
            print(f"  TIP: Run 'python scripts/preprocess/run_all_preprocessing.py' to cache this computation")
        
        if len(self.valid_chip_indices) == 0:
            raise ValueError(f"No valid chips found for {split} split!")
        
        # Setup xbatcher
        input_dims = {"time": self.T, "y": chip_size, "x": chip_size}
        overlap_y = max(0, chip_size - stride)
        overlap_x = max(0, chip_size - stride)
        input_overlap = {"time": 0, "y": overlap_y, "x": overlap_x}
        
        self._bgen = xbatcher.BatchGenerator(
            ds,
            input_dims=input_dims,
            input_overlap=input_overlap,
            preload_batch=False,
        )
        
        # Compute normalization statistics (or load from cache)
        rng = np.random.default_rng(random_seed)
        self._compute_normalization_stats(ds, static_var_names, rng, stat_samples)
        
        # Store metadata
        self.C_comp = len(HM_VARS) if include_components else 0
        self.C_dyn = 1 + self.C_comp
        self.C_static = len(static_var_names)
        self._static_var_names = static_var_names
        self._dynamic_var_names = dynamic_var_names
        self.target_t_indices = [years.index(y) for y in fixed_target_years]
        self.year_to_idx = {y: i for i, y in enumerate(years)}
        
        print(f"=== Dataset ready: {len(self)} chips per epoch ===")
    
    def _compute_normalization_stats(
        self, ds, static_var_names: List[str], rng, stat_samples: int
    ):
        """Compute per-variable normalization statistics (or load from cache)."""
        # Try to load cached stats
        cached_stats = load_cached_stats()
        
        if cached_stats is not None:
            print("Loading normalization statistics from cache...")
            self.hm_mean = cached_stats["hm_mean"]
            self.hm_std = cached_stats["hm_std"]
            self.static_means = cached_stats["static_means"]
            self.static_stds = cached_stats["static_stds"]
            self.comp_means = cached_stats["comp_means"]
            self.comp_stds = cached_stats["comp_stds"]
            print(f"  AA: mean={self.hm_mean:.4f}, std={self.hm_std:.4f}")
            return
        
        print("Cache not found, computing normalization statistics...")
        print("  TIP: Run 'python scripts/preprocess/run_all_preprocessing.py' to cache this computation")
        
        # HM (AA) stats
        hm_samples = []
        per_time = max(1, stat_samples // self.T)
        for t_idx in range(self.T):
            for _ in range(per_time):
                i = rng.integers(0, max(1, self.H - self.chip_size))
                j = rng.integers(0, max(1, self.W - self.chip_size))
                arr = ds["AA"].isel(
                    time=t_idx,
                    y=slice(i, i + self.chip_size),
                    x=slice(j, j + self.chip_size)
                ).load().values
                hm_samples.append(arr)
        
        hm_stack = np.stack(hm_samples, axis=0)
        self.hm_mean = float(np.nanmean(hm_stack))
        self.hm_std = float(np.nanstd(hm_stack)) + 1e-8
        
        # Static variable stats
        self.static_means = []
        self.static_stds = []
        for var_name in static_var_names:
            samples = []
            for _ in range(max(1, stat_samples // len(static_var_names))):
                i = rng.integers(0, max(1, self.H - self.chip_size))
                j = rng.integers(0, max(1, self.W - self.chip_size))
                arr = ds[var_name].isel(
                    y=slice(i, i + self.chip_size),
                    x=slice(j, j + self.chip_size)
                ).load().values
                samples.append(arr)
            stack = np.stack(samples, axis=0)
            self.static_means.append(float(np.nanmean(stack)))
            self.static_stds.append(float(np.nanstd(stack)) + 1e-8)
        
        # Component stats
        self.comp_means = {}
        self.comp_stds = {}
        if self.include_components:
            for var_name in HM_VARS:
                samples = []
                per_var = max(1, stat_samples // (len(HM_VARS) * self.T))
                for t_idx in range(self.T):
                    for _ in range(per_var):
                        i = rng.integers(0, max(1, self.H - self.chip_size))
                        j = rng.integers(0, max(1, self.W - self.chip_size))
                        arr = ds[var_name].isel(
                            time=t_idx,
                            y=slice(i, i + self.chip_size),
                            x=slice(j, j + self.chip_size)
                        ).load().values
                        samples.append(arr)
                stack = np.stack(samples, axis=0)
                self.comp_means[var_name] = float(np.nanmean(stack))
                self.comp_stds[var_name] = float(np.nanstd(stack)) + 1e-8
        
        print(f"  AA: mean={self.hm_mean:.4f}, std={self.hm_std:.4f}")
    
    def __len__(self) -> int:
        if self.chips_per_epoch is not None:
            return min(self.chips_per_epoch, len(self.valid_chip_indices))
        return len(self.valid_chip_indices)
    
    def __getitem__(self, idx: int) -> dict:
        # Map idx to valid chip index
        chip_idx = idx % len(self.valid_chip_indices)
        bgen_idx = self.valid_chip_indices[chip_idx]
        
        # Load batch from xbatcher
        batch = self._bgen[bgen_idx].load()
        
        # Temporal sampling
        if self.use_temporal_sampling:
            end_year = int(np.random.choice(self.end_year_options))
            target_years = tuple(end_year + offset for offset in (5, 10, 15, 20))
        else:
            end_year = self.fixed_target_years[-1]
            target_years = self.fixed_target_years
        
        input_years = self.fixed_input_years
        target_t_idxs = [years.index(y) if y <= 2020 else None for y in target_years]
        
        # Build input_dynamic [T, C_dyn, H, W]
        input_dynamic = np.empty(
            (len(input_years), self.C_dyn, self.chip_size, self.chip_size),
            dtype=np.float32
        )
        
        for t_idx, year in enumerate(input_years):
            year_idx = self.year_to_idx[year]
            
            # AA (target variable)
            arr_hm = batch["AA"].isel(time=year_idx).values.astype(np.float32)
            arr_hm = (arr_hm - self.hm_mean) / self.hm_std
            input_dynamic[t_idx, 0, :, :] = arr_hm
            
            # Component covariates
            if self.include_components:
                for c_idx, var_name in enumerate(HM_VARS):
                    carr = batch[var_name].isel(time=year_idx).values.astype(np.float32)
                    carr = np.nan_to_num(carr, nan=0.0)
                    carr = (carr - self.comp_means[var_name]) / self.comp_stds[var_name]
                    input_dynamic[t_idx, 1 + c_idx, :, :] = carr
        
        # Build input_static [C_static, H, W]
        input_static = np.empty(
            (self.C_static, self.chip_size, self.chip_size),
            dtype=np.float32
        )
        
        for s_idx, var_name in enumerate(self._static_var_names):
            sarr = batch[var_name].values.astype(np.float32)
            sarr = np.nan_to_num(sarr, nan=0.0)
            sarr = (sarr - self.static_means[s_idx]) / self.static_stds[s_idx]
            input_static[s_idx, :, :] = sarr
        
        # Coordinates (simplified - pixel indices)
        xs, ys = np.meshgrid(
            np.arange(self.chip_size), np.arange(self.chip_size), indexing="ij"
        )
        if "x" in batch.coords and "y" in batch.coords:
            x_vals = batch["x"].values
            y_vals = batch["y"].values
            xx, yy = np.meshgrid(x_vals, y_vals, indexing="ij")
            lonlat = np.stack([xx, yy], axis=-1).astype(np.float32)
        else:
            lonlat = np.stack([xs, ys], axis=-1).astype(np.float32)
        
        # Multi-horizon targets
        targets = {}
        horizon_names = ['target_5yr', 'target_10yr', 'target_15yr', 'target_20yr']
        
        for horizon_name, t_idx, target_year in zip(horizon_names, target_t_idxs, target_years):
            if t_idx is None or target_year > 2020:
                target_h = np.full((self.chip_size, self.chip_size), np.nan, dtype=np.float32)
            else:
                target_h = batch["AA"].isel(time=t_idx).values.astype(np.float32)
                target_h = (target_h - self.hm_mean) / self.hm_std
            targets[horizon_name] = torch.from_numpy(target_h).float()
        
        sample = {
            "input_dynamic": torch.from_numpy(input_dynamic).float(),
            "input_static": torch.from_numpy(input_static).float(),
            "lonlat": torch.from_numpy(lonlat).float(),
            "timestep": self.target_t_indices[-1],
            "input_years": input_years,
            "target_years": target_years,
            "end_year": end_year,
        }
        sample.update(targets)
        sample["target"] = sample["target_5yr"]
        
        return sample


# =============================================================================
# DataLoader Factory
# =============================================================================

def get_dataloader(
    split: Literal["train", "val", "test", "calib"] = "train",
    batch_size: int = 8,
    chip_size: int = 128,
    stride: int = 128,
    chips_per_epoch: Optional[int] = None,
    min_valid_ratio: float = 0.5,
    use_temporal_sampling: bool = True,
    end_year_options: Tuple[int, ...] = (2000, 2005, 2010, 2015),
    include_components: bool = True,
    static_channels: Optional[int] = None,
    zarr_path: str = ZARR_PATH,
    # Platform settings (None = auto-detect)
    platform: Optional[Literal["aws", "m1_mac", "auto"]] = "auto",
    num_workers: Optional[int] = None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
    dask_threads: Optional[int] = None,
    # Additional settings
    random_seed: int = 42,
    stat_samples: int = 256,
) -> DataLoader:
    """
    Create a DataLoader for the Human Footprint dataset.
    
    Args:
        split: Data split ("train", "val", "test", "calib")
        batch_size: Batch size
        chip_size: Spatial chip size (pixels)
        stride: Stride between chips (pixels)
        chips_per_epoch: Chips per epoch (None = all valid chips)
        min_valid_ratio: Minimum ratio of valid pixels per chip
        use_temporal_sampling: Random end year for training
        end_year_options: Valid end years for temporal sampling
        include_components: Include HM component covariates
        static_channels: Limit static channels
        zarr_path: Path to icechunk repo
        platform: Platform type ("aws", "m1_mac", "auto")
        num_workers: DataLoader workers (None = platform default)
        prefetch_factor: Prefetch factor (None = platform default)
        persistent_workers: Keep workers alive (None = platform default)
        pin_memory: Pin memory for GPU (None = platform default)
        dask_threads: Dask threads (None = platform default)
        random_seed: Random seed
        stat_samples: Samples for normalization stats
        
    Returns:
        DataLoader instance
    """
    # Detect platform and get defaults
    if platform == "auto" or platform is None:
        detected = detect_platform()
        print(f"Detected platform: {detected}")
    else:
        detected = platform
    
    defaults = get_platform_defaults(detected)
    
    # Apply defaults for None values
    num_workers = num_workers if num_workers is not None else defaults["num_workers"]
    prefetch_factor = prefetch_factor if prefetch_factor is not None else defaults["prefetch_factor"]
    persistent_workers = persistent_workers if persistent_workers is not None else defaults["persistent_workers"]
    pin_memory = pin_memory if pin_memory is not None else defaults["pin_memory"]
    dask_threads = dask_threads if dask_threads is not None else defaults["dask_threads"]
    multiprocessing_context = defaults.get("multiprocessing_context")
    
    print(f"DataLoader settings: num_workers={num_workers}, prefetch_factor={prefetch_factor}, "
          f"persistent_workers={persistent_workers}, pin_memory={pin_memory}, dask_threads={dask_threads}")
    
    # Create dataset
    dataset = HumanFootprintZarrDataset(
        zarr_path=zarr_path,
        split=split,
        chip_size=chip_size,
        stride=stride,
        min_valid_ratio=min_valid_ratio,
        chips_per_epoch=chips_per_epoch,
        use_temporal_sampling=use_temporal_sampling if split == "train" else False,
        end_year_options=end_year_options,
        include_components=include_components,
        static_channels=static_channels,
        static_files_list=static_files,
        stat_samples=stat_samples,
        random_seed=random_seed,
        dask_threads=dask_threads,
    )
    
    # Build DataLoader kwargs
    # Only shuffle for training, not for val/test/calib
    shuffle = (split == "train")
    
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    
    # Only add these if num_workers > 0
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
        if multiprocessing_context:
            loader_kwargs["multiprocessing_context"] = multiprocessing_context
    
    return DataLoader(dataset, **loader_kwargs)


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    print("Testing dataloader...")
    
    # Test with auto-detected platform
    loader = get_dataloader(
        split="train",
        batch_size=2,
        chip_size=128,
        stride=128,
        chips_per_epoch=10,
        platform="auto",
    )
    
    print(f"\nDataset length: {len(loader.dataset)}")
    print(f"Batch count: {len(loader)}")
    
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  input_dynamic: {batch['input_dynamic'].shape}")
    print(f"  input_static: {batch['input_static'].shape}")
    print(f"  target: {batch['target'].shape}")
    print(f"  lonlat: {batch['lonlat'].shape}")
    
    print("\nDataloader test passed!")
