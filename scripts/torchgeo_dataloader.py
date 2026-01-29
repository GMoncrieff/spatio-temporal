"""
Optimized Zarr dataloader with xbatcher, Dask caching, and multi-worker support.

Performance optimizations:
- Dask cache (10GB) for repeated chunk access
- Dask threaded scheduler for parallel I/O within workers
- xbatcher BatchGenerator for efficient chip generation
- prefetch_factor for GPU pipeline optimization
- forkserver multiprocessing for S3 compatibility
- Lazy dataset opening per worker to avoid connection issues

Usage:
    loader = get_dataloader(
        split='train',
        batch_size=32,
        num_workers=8,
        prefetch_factor=3,
        dask_threads=4,
    )
"""

import os
import json
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import multiprocessing

# Configure Dask at module level
import dask

# Try to enable Dask cache (requires cachey package)
# This significantly speeds up repeated chunk access for overlapping chips
# IMPORTANT: Use a limited cache size to prevent unbounded memory growth
_DASK_CACHE = None
try:
    from dask.cache import Cache
    # Reduced from 10GB to 2GB to prevent memory leaks
    # Cache will evict old entries when full (LRU policy)
    _DASK_CACHE = Cache(2e9)  # 2GB cache (was 10GB)
    _DASK_CACHE.register()
    print("Dask cache enabled (2GB)")
except ImportError:
    print("Warning: cachey not installed, Dask cache disabled. Install with: pip install cachey")

# Paths
HM_DIR = os.path.join("data", "raw", "hm_global")
STATIC_DIR = HM_DIR
ZARR_PATH = os.path.join("scripts", "notebooks", "hm.icechunk")

# List of years for which we have human footprint data
years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
hm_files = [os.path.join(HM_DIR, f"HM_{year}_AA_1000.tiff") for year in years]

# Time-varying HM covariates
HM_VARS = ["AG", "BU", "EX", "FR", "HI", "NS", "PO", "TI", "gdp", "population"]
component_files = {
    y: [os.path.join(HM_DIR, f"HM_{y}_{v}_1000.tiff") for v in HM_VARS]
    for y in years
}
static_files = [
    os.path.join(STATIC_DIR, "hm_static_ele_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_tas_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_tasmin_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_pr_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_dpi_dsi_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_iucn_nostrict_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_iucn_strict_1000.tiff"),
]


def _configure_dask_for_worker(dask_threads: int = 4):
    """Configure Dask scheduler for a worker process."""
    if dask_threads <= 1:
        dask.config.set(scheduler="synchronous")
    else:
        dask.config.set(scheduler="threads", num_workers=dask_threads)


def _worker_init_fn(worker_id: int):
    """
    Initialize worker process with proper Dask configuration.
    
    Called once per worker when using num_workers > 0.
    Sets up Dask cache and threading for parallel I/O.
    """
    import warnings
    
    # Suppress Pydantic warnings in worker processes
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Each worker gets its own smaller cache (500MB) if cachey is available
    # Reduced to prevent memory leaks with multiple workers
    try:
        from dask.cache import Cache
        worker_cache = Cache(5e8)  # 500MB per worker (was 2GB)
        worker_cache.register()
    except ImportError:
        pass  # Cache not available, continue without
    
    # Configure threading (will be overridden by dataset if dask_threads specified)
    dask.config.set(scheduler="threads", num_workers=2)

class HumanFootprintZarrChipDataset(torch.utils.data.Dataset):
    """
    Optimized PyTorch Dataset for Human Footprint Zarr data using xbatcher.
    
    Features:
    - Uses xbatcher BatchGenerator for efficient chip extraction
    - Pre-computed valid chip filtering for train/val/test/calib splits
    - Lazy dataset opening for multi-worker compatibility
    - Dask threading for parallel I/O
    - Per-variable normalization
    """
    
    def __init__(
        self,
        zarr_path: str,
        split: str = 'train',
        valid_chips_metadata_path: str = 'data/processed/valid_chips_metadata.json',
        chip_size: int = 128,
        timesteps: int = 3,
        chips_per_epoch: Optional[int] = None,
        fixed_input_years: Tuple[int, ...] = (1990, 1995, 2000),
        fixed_target_years: Tuple[int, ...] = (2005, 2010, 2015, 2020),
        use_temporal_sampling: bool = True,
        end_year_options: Tuple[int, ...] = (2000, 2005, 2010, 2015),
        stat_samples: int = 256,
        random_seed: int = 42,
        include_components: bool = True,
        static_channels: Optional[int] = None,
        static_files: Optional[List[str]] = None,
        dask_threads: int = 4,
    ):
        """
        Initialize the dataset.
        
        Args:
            zarr_path: Path to Zarr/Icechunk repository (local or s3://)
            split: One of 'train', 'val', 'test', 'calib'
            valid_chips_metadata_path: Path to pre-computed valid chips JSON
            chip_size: Spatial size of chips
            timesteps: Number of input timesteps
            chips_per_epoch: Limit chips per epoch (None = use all)
            dask_threads: Number of Dask threads for parallel I/O
        """
        # Store config for lazy initialization
        self._zarr_path = zarr_path
        self._split = split
        self._chip_size = chip_size
        self._timesteps = timesteps
        self._dask_threads = dask_threads
        self._random_seed = random_seed
        self._include_components = bool(include_components)
        self._static_files_arg = static_files
        self._static_channels = static_channels
        
        # Temporal configuration
        if len(fixed_input_years) != 3:
            raise ValueError("Multi-horizon setup expects exactly 3 input timesteps")
        self._use_temporal_sampling = use_temporal_sampling and split == 'train'
        self._end_year_options = list(end_year_options)
        self._fixed_input_years = tuple(fixed_input_years)
        self._fixed_target_years = tuple(fixed_target_years)
        self._target_t_indices = [years.index(y) for y in fixed_target_years]
        self._year_to_idx = {y: i for i, y in enumerate(years)}
        
        # Load pre-computed valid chip metadata
        metadata_path = Path(valid_chips_metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Valid chips metadata not found at {metadata_path}. "
                f"Run 'python scripts/precompute_valid_chips.py' first."
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Validate metadata matches current settings
        if metadata['chip_size'] != chip_size:
            raise ValueError(
                f"Chip size mismatch: metadata has {metadata['chip_size']}, "
                f"but requested {chip_size}. Re-run precompute_valid_chips.py."
            )
        
        self._stride = metadata['stride']
        self._dataset_shape = metadata['dataset_shape']
        
        # Get valid chip positions for this split
        all_valid_positions = metadata['splits'][split]
        
        # Optionally limit number of chips per epoch
        if chips_per_epoch is not None and chips_per_epoch < len(all_valid_positions):
            self._all_valid_positions = all_valid_positions
            self._chips_per_epoch = chips_per_epoch
            rng = np.random.default_rng(random_seed)
            indices = rng.choice(len(all_valid_positions), size=chips_per_epoch, replace=False)
            self._valid_positions = [all_valid_positions[i] for i in indices]
            print(f"Loaded {len(all_valid_positions):,} valid chips for split '{split}', sampling {chips_per_epoch:,} per epoch")
        else:
            self._valid_positions = all_valid_positions
            self._all_valid_positions = all_valid_positions
            self._chips_per_epoch = None
            print(f"Loaded {len(self._valid_positions):,} valid chips for split '{split}'")

        # Store static file config
        self._static_files_list = list(static_files if static_files is not None else [])
        if static_channels is not None:
            self._static_files_list = self._static_files_list[:int(static_channels)]

        self._static_var_names = [
            os.path.basename(f).replace("hm_static_", "").replace(".tiff", "") 
            for f in self._static_files_list
        ]
        self._dynamic_var_names = ["AA"] + (list(HM_VARS) if self._include_components else [])
        
        # Lazy initialization flags
        self._initialized = False
        self._ds = None
        self._bgen = None
        self._batch_index_map = None
        
        # Normalization stats (computed on first access)
        self._hm_mean = None
        self._hm_std = None
        self._static_means = None
        self._static_stds = None
        self._comp_means = None
        self._comp_stds = None
        
        # Channel counts
        self.C_comp = len(HM_VARS) if self._include_components else 0
        self.C_dyn = 1 + self.C_comp
        self.C_static = len(self._static_var_names)
        
        # Initialize on main process (for normalization stats)
        # Workers will re-initialize their own connections
        self._initialize_dataset(stat_samples)
    
    def _open_zarr_store(self):
        """Open Zarr store via Icechunk. Called once per worker."""
        import xarray as xr
        import xbatcher
        import icechunk
        
        # Configure Dask for this worker
        _configure_dask_for_worker(self._dask_threads)
        
        repo_path = str(self._zarr_path)
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
        
        return ds
    
    def _expand_consolidated_dataset(self, ds):
        """Expand consolidated dynamic/static structure into separate variables."""
        import xarray as xr
        
        if "dynamic" not in ds.data_vars or "static" not in ds.data_vars:
            return ds
        
        ds_expanded = xr.Dataset(
            coords={k: ds.coords[k] for k in ("time", "y", "x") if k in ds.coords}
        )
        
        if "var_dynamic" not in ds.coords:
            raise ValueError("Expected coord 'var_dynamic' when 'dynamic' variable is present")
        if "var_static" not in ds.coords:
            raise ValueError("Expected coord 'var_static' when 'static' variable is present")
        
        for v in ds.coords["var_dynamic"].values:
            vname = str(v)
            ds_expanded[vname] = ds["dynamic"].sel(var_dynamic=v).drop_vars("var_dynamic")
        
        for v in ds.coords["var_static"].values:
            vname = str(v)
            ds_expanded[vname] = ds["static"].sel(var_static=v).drop_vars("var_static")
        
        return ds_expanded
    
    def _initialize_dataset(self, stat_samples: int = 256):
        """Initialize dataset and compute normalization statistics."""
        import xarray as xr
        import xbatcher
        
        print("=== Zarr Dataset Info ===")
        ds = self._open_zarr_store()
        
        # Print dataset info
        print(f"Dataset dimensions: {dict(ds.sizes)}")
        print(f"Data variables: {list(ds.data_vars.keys())}")
        print(f"Coordinates: {list(ds.coords.keys())}")
        if "var_dynamic" in ds.coords:
            print(f"Dynamic variables: {list(ds.coords['var_dynamic'].values)}")
        if "var_static" in ds.coords:
            print(f"Static variables: {list(ds.coords['var_static'].values)}")
        print("========================")
        
        # Expand consolidated structure
        ds = self._expand_consolidated_dataset(ds)
        
        # Filter to required variables
        keep_vars = [v for v in self._dynamic_var_names if v in ds.data_vars]
        keep_vars += [v for v in self._static_var_names if v in ds.data_vars]
        missing_vars = [v for v in (self._dynamic_var_names + self._static_var_names) if v not in ds.data_vars]
        if missing_vars:
            raise ValueError(f"Missing required variables in Zarr: {missing_vars}")
        ds = ds[keep_vars]
        
        H = int(ds.sizes["y"])
        W = int(ds.sizes["x"])
        T = int(ds.sizes.get("time", 0))
        
        if T != len(years):
            raise ValueError(f"Unexpected time dimension: {T}, expected {len(years)}")
        
        # Create xbatcher BatchGenerator
        # This aligns with Zarr chunks and enables efficient batch extraction
        bgen = xbatcher.BatchGenerator(
            ds,
            input_dims={"y": self._chip_size, "x": self._chip_size},
            input_overlap={"y": self._chip_size - self._stride, "x": self._chip_size - self._stride},
            preload_batch=False,  # Lazy loading - critical for performance
        )
        
        # Build mapping from our valid positions to xbatcher batch indices
        # xbatcher generates batches in row-major order
        n_y_batches = len(range(0, H - self._chip_size + 1, self._stride))
        n_x_batches = len(range(0, W - self._chip_size + 1, self._stride))
        
        # Map (yi, xi) -> xbatcher batch index
        self._batch_index_map = {}
        for yi, xi in self._valid_positions:
            batch_idx = yi * n_x_batches + xi
            if batch_idx < len(bgen):
                self._batch_index_map[len(self._batch_index_map)] = batch_idx
        
        print(f"xbatcher: {len(bgen)} total batches, {len(self._batch_index_map)} valid for split '{self._split}'")
        
        # Compute normalization statistics using xbatcher
        self._compute_normalization_stats(ds, bgen, stat_samples)
        
        # Store for __getitem__
        self._ds = ds
        self._bgen = bgen
        self._initialized = True
    
    def _get_stats_cache_path(self):
        """Generate cache file path based on dataset configuration."""
        import hashlib
        
        # Create unique cache key from dataset parameters
        cache_key_parts = [
            str(self._zarr_path),
            str(self._split),
            str(self._chip_size),
            str(self._stride),
            str(self._include_components),
            str(len(self._static_var_names)),
            str(self._random_seed),
        ]
        cache_key = "_".join(cache_key_parts)
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        
        cache_dir = Path("data/processed/stats_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"norm_stats_{self._split}_{cache_hash}.json"
    
    def _load_cached_stats(self):
        """Load normalization statistics from cache if available."""
        cache_path = self._get_stats_cache_path()
        
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            
            self._hm_mean = cached['hm_mean']
            self._hm_std = cached['hm_std']
            self._static_means = cached['static_means']
            self._static_stds = cached['static_stds']
            self._comp_means = cached['comp_means']
            self._comp_stds = cached['comp_stds']
            
            print(f"✓ Loaded cached normalization statistics from {cache_path.name}")
            return True
        except Exception as e:
            print(f"Warning: Failed to load cached stats: {e}")
            return False
    
    def _save_stats_to_cache(self):
        """Save normalization statistics to cache."""
        cache_path = self._get_stats_cache_path()
        
        try:
            cached = {
                'hm_mean': self._hm_mean,
                'hm_std': self._hm_std,
                'static_means': self._static_means,
                'static_stds': self._static_stds,
                'comp_means': self._comp_means,
                'comp_stds': self._comp_stds,
                'zarr_path': str(self._zarr_path),
                'split': self._split,
                'chip_size': self._chip_size,
                'stride': self._stride,
                'include_components': self._include_components,
                'num_static_vars': len(self._static_var_names),
                'random_seed': self._random_seed,
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cached, f, indent=2)
            
            print(f"✓ Saved normalization statistics to cache: {cache_path.name}")
        except Exception as e:
            print(f"Warning: Failed to save stats to cache: {e}")
    
    def _compute_normalization_stats(self, ds, bgen, stat_samples: int):
        """Compute per-variable normalization statistics from sampled batches."""
        # Try to load from cache first
        if self._load_cached_stats():
            return
        
        print("Computing normalization statistics from data...")
        rng = np.random.default_rng(self._random_seed)
        
        # Sample valid batch indices
        n_samples = min(len(self._batch_index_map), max(64, stat_samples))
        sampled_local_indices = rng.choice(len(self._batch_index_map), size=n_samples, replace=False)
        
        # Collect HM samples
        hm_samples = []
        T = len(years)
        for local_idx in sampled_local_indices:
            batch_idx = self._batch_index_map[local_idx]
            batch = bgen[batch_idx].load()
            t_idx = int(rng.integers(0, T))
            arr = batch["AA"].isel(time=t_idx).values
            hm_samples.append(arr)
        
        hm_stack = np.stack(hm_samples, axis=0)
        self._hm_mean = float(np.nanmean(hm_stack))
        self._hm_std = float(np.nanstd(hm_stack)) + 1e-8
        
        # Static variable stats
        self._static_means = []
        self._static_stds = []
        if self._static_var_names:
            print("Computing per-variable normalization statistics for static layers...")
            n_static_samples = min(len(self._batch_index_map), max(32, stat_samples))
            sampled_static = rng.choice(len(self._batch_index_map), size=n_static_samples, replace=False)
            
            for var_name in self._static_var_names:
                var_samples = []
                for local_idx in sampled_static:
                    batch_idx = self._batch_index_map[local_idx]
                    batch = bgen[batch_idx].load()
                    arr = batch[var_name].values
                    var_samples.append(arr)
                var_stack = np.stack(var_samples, axis=0)
                self._static_means.append(float(np.nanmean(var_stack)))
                self._static_stds.append(float(np.nanstd(var_stack)) + 1e-8)
            
            print("Static normalization stats:")
            for idx, var_name in enumerate(self._static_var_names):
                print(f"  {var_name}: mean={self._static_means[idx]:.6e}, std={self._static_stds[idx]:.6e}")
        
        # Component variable stats
        self._comp_means = {}
        self._comp_stds = {}
        if self._include_components:
            print("Computing per-variable normalization statistics for components...")
            n_comp_samples = min(len(self._batch_index_map), max(32, stat_samples))
            sampled_comp = rng.choice(len(self._batch_index_map), size=n_comp_samples, replace=False)
            
            for var_name in HM_VARS:
                var_samples = []
                for local_idx in sampled_comp:
                    batch_idx = self._batch_index_map[local_idx]
                    batch = bgen[batch_idx].load()
                    t_idx = int(rng.integers(0, T))
                    arr = batch[var_name].isel(time=t_idx).values
                    var_samples.append(arr)
                var_stack = np.stack(var_samples, axis=0)
                self._comp_means[var_name] = float(np.nanmean(var_stack))
                self._comp_stds[var_name] = float(np.nanstd(var_stack)) + 1e-8
            
            print("Component normalization stats:")
            for var_name in HM_VARS:
                print(f"  {var_name}: mean={self._comp_means[var_name]:.6e}, std={self._comp_stds[var_name]:.6e}")
        
        # Save computed stats to cache for future runs
        self._save_stats_to_cache()
    
    def _ensure_initialized(self):
        """Ensure dataset is initialized (lazy init for worker processes)."""
        if self._ds is None or self._bgen is None:
            # Re-initialize in worker process
            import xarray as xr
            import xbatcher
            
            _configure_dask_for_worker(self._dask_threads)
            
            ds = self._open_zarr_store()
            ds = self._expand_consolidated_dataset(ds)
            
            keep_vars = [v for v in self._dynamic_var_names if v in ds.data_vars]
            keep_vars += [v for v in self._static_var_names if v in ds.data_vars]
            ds = ds[keep_vars]
            
            bgen = xbatcher.BatchGenerator(
                ds,
                input_dims={"y": self._chip_size, "x": self._chip_size},
                input_overlap={"y": self._chip_size - self._stride, "x": self._chip_size - self._stride},
                preload_batch=False,
            )
            
            self._ds = ds
            self._bgen = bgen
    
    # Public properties for backward compatibility
    @property
    def include_components(self):
        return self._include_components
    
    @property
    def hm_mean(self):
        return self._hm_mean
    
    @property
    def hm_std(self):
        return self._hm_std
    
    @property
    def static_means(self):
        return self._static_means
    
    @property
    def static_stds(self):
        return self._static_stds
    
    @property
    def comp_means(self):
        return self._comp_means
    
    @property
    def comp_stds(self):
        return self._comp_stds
    
    @property
    def elev_mean(self):
        """Elevation mean (first static variable) for backward compatibility."""
        return self._static_means[0] if self._static_means else 0.0
    
    @property
    def elev_std(self):
        """Elevation std (first static variable) for backward compatibility."""
        return self._static_stds[0] if self._static_stds else 1.0
    
    def __len__(self):
        return len(self._batch_index_map)
    
    def __del__(self):
        """Clean up resources when dataset is destroyed."""
        # Close xarray dataset to release file handles and caches
        if hasattr(self, '_ds') and self._ds is not None:
            try:
                self._ds.close()
            except:
                pass
        
        # Delete batch generator to free Dask task graphs
        if hasattr(self, '_bgen') and self._bgen is not None:
            try:
                del self._bgen
            except:
                pass

    def __getitem__(self, idx):
        """Get a single chip by index using xbatcher."""
        # Ensure initialized (handles worker process lazy init)
        self._ensure_initialized()
        
        # Periodic cache clearing to prevent memory accumulation in persistent workers
        # Clear every 500 batches to balance performance vs memory
        if not hasattr(self, '_batch_counter'):
            self._batch_counter = 0
        self._batch_counter += 1
        
        if self._batch_counter % 500 == 0:
            import gc
            try:
                from dask.cache import Cache
                for cache in Cache._caches:
                    cache.clear()
            except:
                pass
            gc.collect()
        
        # Map our index to xbatcher batch index
        batch_idx = self._batch_index_map[idx]
        
        # Load batch via xbatcher (leverages Dask cache and threading)
        batch = self._bgen[batch_idx].load()
        
        # Temporal sampling
        if self._use_temporal_sampling:
            end_year = int(np.random.choice(self._end_year_options))
            target_years = tuple(end_year + offset for offset in (5, 10, 15, 20))
        else:
            end_year = self._fixed_target_years[-1]
            target_years = self._fixed_target_years
        
        input_years = self._fixed_input_years
        target_t_idxs = [years.index(y) if y <= 2020 else None for y in target_years]
        
        # Build input dynamic tensor [T, C, H, W]
        input_dynamic = np.empty(
            (self._timesteps, self.C_dyn, self._chip_size, self._chip_size),
            dtype=np.float32
        )
        
        for t_idx, year in enumerate(input_years):
            # HM target variable
            arr_hm = batch["AA"].isel(time=t_idx).values
            arr_hm = (arr_hm - self._hm_mean) / self._hm_std
            input_dynamic[t_idx, 0, :, :] = arr_hm
            
            # Component covariates
            if self._include_components:
                for c_idx, var_name in enumerate(HM_VARS):
                    carr = batch[var_name].isel(time=t_idx).values
                    carr = np.nan_to_num(carr, nan=0.0)
                    carr = (carr - self._comp_means[var_name]) / self._comp_stds[var_name]
                    input_dynamic[t_idx, 1 + c_idx, :, :] = carr
        
        # Build static tensor [C, H, W]
        input_static = np.empty(
            (self.C_static, self._chip_size, self._chip_size),
            dtype=np.float32
        )
        for static_idx, var_name in enumerate(self._static_var_names):
            sarr = batch[var_name].values
            sarr = np.nan_to_num(sarr, nan=0.0)
            sarr = (sarr - self._static_means[static_idx]) / self._static_stds[static_idx]
            input_static[static_idx, :, :] = sarr
        
        # Coordinates
        x_coords = batch["x"].values
        y_coords = batch["y"].values
        xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
        lonlat = np.stack([xx, yy], axis=-1).astype(np.float32)
        
        # Multi-horizon targets
        targets = {}
        horizon_names = ['target_5yr', 'target_10yr', 'target_15yr', 'target_20yr']
        
        for horizon_name, t_idx, target_year in zip(horizon_names, target_t_idxs, target_years):
            if t_idx is None or target_year > 2020:
                target_h = np.full((self._chip_size, self._chip_size), np.nan, dtype=np.float32)
            else:
                target_h = batch["AA"].isel(time=t_idx).values
                target_h = (target_h - self._hm_mean) / self._hm_std
            targets[horizon_name] = torch.from_numpy(target_h).float()
        
        sample = {
            "input_dynamic": torch.from_numpy(input_dynamic).float(),
            "input_static": torch.from_numpy(input_static).float(),
            "lonlat": torch.from_numpy(lonlat).float(),
            "timestep": self._target_t_indices[-1],
            "input_years": input_years,
            "target_years": target_years,
            "end_year": end_year,
        }
        sample.update(targets)
        sample["target"] = sample["target_5yr"]
        return sample


def get_dataloader(
    split: str = 'train',
    batch_size: int = 1,
    chip_size: int = 128,
    timesteps: int = 3,
    chips_per_epoch: Optional[int] = None,
    fixed_input_years: Tuple[int, ...] = (1990, 1995, 2000),
    fixed_target_years: Tuple[int, ...] = (2005, 2010, 2015, 2020),
    use_temporal_sampling: bool = True,
    end_year_options: Tuple[int, ...] = (2000, 2005, 2010, 2015),
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
    dask_threads: int = 4,
    stat_samples: int = 256,
    include_components: bool = True,
    static_channels: Optional[int] = None,
    zarr_path: str = ZARR_PATH,
    valid_chips_metadata_path: str = 'data/processed/valid_chips_metadata.json',
):
    """
    Create an optimized DataLoader for Human Footprint dataset.
    
    Performance features:
    - xbatcher for efficient chip extraction aligned with Zarr chunks
    - Dask cache (10GB) for repeated chunk access
    - Dask threading for parallel I/O
    - prefetch_factor for GPU pipeline optimization
    - forkserver multiprocessing for S3 compatibility
    
    Args:
        split: Which split to use ('train', 'val', 'test', 'calib')
        batch_size: Batch size
        chip_size: Size of spatial chips
        chips_per_epoch: Limit chips per epoch (None = use all valid chips)
        num_workers: Number of DataLoader workers
        prefetch_factor: Batches to prefetch per worker (default: 2, recommended: 3)
        dask_threads: Dask threads per worker for parallel I/O
        zarr_path: Path to Zarr/Icechunk repository (local or s3://)
        valid_chips_metadata_path: Path to pre-computed valid chips metadata
        
    Returns:
        DataLoader configured for optimal performance
    
    Note:
        - Shuffle is True for train split, False for others
        - All chips guaranteed to have valid data (no empty chips)
        - For S3 data, recommend: num_workers=8, prefetch_factor=3, dask_threads=4
    """
    # Configure main process Dask
    _configure_dask_for_worker(dask_threads)
    
    ds = HumanFootprintZarrChipDataset(
        zarr_path=zarr_path,
        split=split,
        valid_chips_metadata_path=valid_chips_metadata_path,
        chip_size=chip_size,
        timesteps=timesteps,
        chips_per_epoch=chips_per_epoch,
        fixed_input_years=fixed_input_years,
        fixed_target_years=fixed_target_years,
        use_temporal_sampling=use_temporal_sampling,
        end_year_options=end_year_options,
        stat_samples=stat_samples,
        include_components=include_components,
        static_channels=static_channels,
        static_files=static_files,
        dask_threads=dask_threads,
    )
    
    # Always shuffle for train, never for val/test/calib
    shuffle = (split == 'train')
    
    # DataLoader kwargs
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    
    # Add performance optimizations for multi-worker setup
    if num_workers > 0:
        # Use forkserver for S3 compatibility (avoids connection pool issues)
        loader_kwargs["multiprocessing_context"] = "forkserver"
        loader_kwargs["worker_init_fn"] = _worker_init_fn
        
        # Prefetch more batches to hide I/O latency
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    
    return DataLoader(ds, **loader_kwargs)


if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("TESTING OPTIMIZED XBATCHER DATALOADER")
    print("=" * 60)
    
    # Test with optimizations
    t0 = time.time()
    loader = get_dataloader(
        split='train',
        batch_size=4,
        chip_size=128,
        timesteps=3,
        num_workers=0,  # Single worker for testing
        dask_threads=4,
    )
    t1 = time.time()
    print(f"\nDataset initialization: {t1-t0:.2f}s")
    print(f"Dataset size: {len(loader.dataset):,} chips")
    print(f"Batches per epoch: {len(loader):,}")
    
    # Test batch loading
    print("\nLoading first batch...")
    t0 = time.time()
    batch = next(iter(loader))
    t1 = time.time()
    print(f"First batch load time: {t1-t0:.2f}s")
    print(f"  input_dynamic: {batch['input_dynamic'].shape}")
    print(f"  input_static: {batch['input_static'].shape}")
    print(f"  target: {batch['target'].shape}")
    
    # Test multiple batches
    print("\nLoading 5 more batches...")
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= 5:
            break
    t1 = time.time()
    print(f"5 batches loaded in: {t1-t0:.2f}s ({(t1-t0)/5:.2f}s per batch)")
    
    print("\n✅ Dataloader test complete!")
