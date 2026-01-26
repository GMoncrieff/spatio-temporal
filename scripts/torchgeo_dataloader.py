import os
from urllib.parse import urlparse
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset
import torch
from pyproj import Transformer

# Paths (updated to hm_medium dataset)
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
   # os.path.join(STATIC_DIR, "hm_static_ele_asp_cosin_1000.tiff"),
   # os.path.join(STATIC_DIR, "hm_static_ele_asp_sin_1000.tiff"),
   # os.path.join(STATIC_DIR, "hm_static_ele_slope_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_tas_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_tasmin_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_pr_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_dpi_dsi_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_iucn_nostrict_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_iucn_strict_1000.tiff"),
]

import numpy as np
import rasterio

# Removed HumanFootprintChipDataset - no longer used for batch loading
# rasterio kept for split_mask and validity_mask reading

class HumanFootprintZarrChipDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        zarr_path,
        split='train',
        valid_chips_metadata_path='data/processed/valid_chips_metadata.json',
        chip_size=128,
        timesteps=3,
        chips_per_epoch=None,
        fixed_input_years=(1990, 1995, 2000),
        fixed_target_years=(2005, 2010, 2015, 2020),
        use_temporal_sampling=True,
        end_year_options=(2000, 2005, 2010, 2015),
        stat_samples=256,
        random_seed=42,
        include_components=True,
        static_channels=None,
        static_files=None,
    ):
        try:
            import xarray as xr
            import xbatcher
            import icechunk
        except Exception as e:
            raise ImportError(
                "Zarr backend requires xarray, xbatcher, and icechunk. Install: xarray dask xbatcher icechunk zarr"
            ) from e

        self.zarr_path = zarr_path
        self.split = split
        self.chip_size = chip_size
        self.timesteps = timesteps
        if len(fixed_input_years) != 3:
            raise ValueError("Multi-horizon setup expects exactly 3 input timesteps (1990, 1995, 2000)")
        self.use_temporal_sampling = use_temporal_sampling and split == 'train'
        self.end_year_options = list(end_year_options)
        self.fixed_input_years = tuple(fixed_input_years)
        self.fixed_target_years = tuple(fixed_target_years)
        self.target_t_indices = [years.index(y) for y in fixed_target_years]
        self.year_to_idx = {y: i for i, y in enumerate(years)}
        self.include_components = bool(include_components)
        
        # Load pre-computed valid chip positions
        import json
        from pathlib import Path
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
        
        # Get valid chip positions for this split
        all_valid_positions = metadata['splits'][split]
        self.stride = metadata['stride']
        self.random_seed = random_seed
        
        # Optionally limit number of chips per epoch
        if chips_per_epoch is not None and chips_per_epoch < len(all_valid_positions):
            # Sample a subset for this epoch (will be re-sampled each epoch via __len__)
            self.all_valid_chip_positions = all_valid_positions
            self.chips_per_epoch = chips_per_epoch
            # Initial random sample
            rng = np.random.default_rng(random_seed)
            indices = rng.choice(len(all_valid_positions), size=chips_per_epoch, replace=False)
            self.valid_chip_positions = [all_valid_positions[i] for i in indices]
            print(f"Loaded {len(all_valid_positions):,} valid chips for split '{split}', sampling {chips_per_epoch:,} per epoch")
        else:
            # Use all valid chips
            self.valid_chip_positions = all_valid_positions
            self.all_valid_chip_positions = all_valid_positions
            self.chips_per_epoch = None
            print(f"Loaded {len(self.valid_chip_positions):,} valid chips for split '{split}'")

        self._static_files = list(static_files if static_files is not None else [])
        if static_channels is not None:
            self._static_files = self._static_files[: int(static_channels)]

        static_var_names = [
            os.path.basename(f).replace("hm_static_", "").replace(".tiff", "") for f in self._static_files
        ]
        dynamic_var_names = ["AA"] + (list(HM_VARS) if self.include_components else [])

        repo_path = str(self.zarr_path)
        print("=== Zarr Dataset Info ===")
        if repo_path.startswith("s3://"):
            parsed = urlparse(repo_path)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")
            store = icechunk.s3_storage(bucket=bucket, prefix=prefix, region=None, endpoint_url=None, anonymous=False, allow_http=False, force_path_style=False)
        else:
            store = icechunk.local_filesystem_storage(repo_path)

        repo = icechunk.Repository.open(store)
        session = repo.readonly_session("main")

        try:
            ds = xr.open_zarr(session.store, group="hm", consolidated=False)
        except Exception:
            ds = xr.open_zarr(session.store, consolidated=False)

        # Print dataset info immediately after loading
        print(f"Dataset dimensions: {dict(ds.sizes)}")
        print(f"Data variables: {list(ds.data_vars.keys())}")
        print(f"Coordinates: {list(ds.coords.keys())}")
        if "var_dynamic" in ds.coords:
            print(f"Dynamic variables (var_dynamic): {list(ds.coords['var_dynamic'].values)}")
        if "var_static" in ds.coords:
            print(f"Static variables (var_static): {list(ds.coords['var_static'].values)}")
        print("========================")

        if "dynamic" in ds.data_vars and "static" in ds.data_vars:
            ds_expanded = xr.Dataset(coords={k: ds.coords[k] for k in ("time", "y", "x") if k in ds.coords})

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

            ds = ds_expanded

        keep_vars = [v for v in dynamic_var_names if v in ds.data_vars]
        keep_vars += [v for v in static_var_names if v in ds.data_vars]
        missing_vars = [v for v in (dynamic_var_names + static_var_names) if v not in ds.data_vars]
        if len(missing_vars) > 0:
            raise ValueError(f"Missing required variables in Zarr: {missing_vars}")
        ds = ds[keep_vars]
        self.ds = ds

        self.H = int(ds.sizes["y"])
        self.W = int(ds.sizes["x"])
        self.T = int(ds.sizes.get("time", 0))

        if self.T != len(years):
            raise ValueError(f"Unexpected time dimension length in Zarr: {self.T}, expected {len(years)}")

        # Sample from valid chip positions for normalization statistics
        rng = np.random.default_rng(random_seed)
        hm_samples = []
        total_samps = min(len(self.valid_chip_positions), max(64, int(stat_samples)))
        
        # Sample random valid chips
        sampled_positions = rng.choice(len(self.valid_chip_positions), size=total_samps, replace=False)
        
        for pos_idx in sampled_positions:
            yi, xi = self.valid_chip_positions[pos_idx]
            y_start = yi * self.stride
            x_start = xi * self.stride
            
            # Sample one random time step for this position
            t_idx = int(rng.integers(0, self.T))
            arr = (
                ds["AA"]
                .isel(time=t_idx, y=slice(y_start, y_start + self.chip_size), x=slice(x_start, x_start + self.chip_size))
                .load()
                .values
            )
            hm_samples.append(arr)
        hm_stack_samp = np.stack(hm_samples, axis=0)
        self.hm_mean = np.nanmean(hm_stack_samp)
        self.hm_std = np.nanstd(hm_stack_samp) + 1e-8

        self.static_means = []
        self.static_stds = []
        if len(static_var_names) > 0:
            print("Computing per-variable normalization statistics for static layers...")
            total_static_samps = min(len(self.valid_chip_positions), max(32, int(stat_samples)))
            sampled_positions_static = rng.choice(len(self.valid_chip_positions), size=total_static_samps, replace=False)
            
            for static_idx, var_name in enumerate(static_var_names):
                static_samples = []
                for pos_idx in sampled_positions_static:
                    yi, xi = self.valid_chip_positions[pos_idx]
                    y_start = yi * self.stride
                    x_start = xi * self.stride
                    sarr = (
                        ds[var_name]
                        .isel(y=slice(y_start, y_start + self.chip_size), x=slice(x_start, x_start + self.chip_size))
                        .load()
                        .values
                    )
                    static_samples.append(sarr)
                static_stack = np.stack(static_samples, axis=0)
                self.static_means.append(np.nanmean(static_stack))
                self.static_stds.append(np.nanstd(static_stack) + 1e-8)
            print("Static normalization stats:")
            for idx, var_name in enumerate(static_var_names):
                print(f"  {var_name}: mean={self.static_means[idx]:.6e}, std={self.static_stds[idx]:.6e}")

        self.elev_mean = self.static_means[0] if self.static_means else 0.0
        self.elev_std = self.static_stds[0] if self.static_stds else 1.0

        self.comp_means = {}
        self.comp_stds = {}
        if self.include_components:
            print("Computing per-variable normalization statistics for components...")
            total_comp_samps = min(len(self.valid_chip_positions), max(32, int(stat_samples)))
            sampled_positions_comp = rng.choice(len(self.valid_chip_positions), size=total_comp_samps, replace=False)
            
            for var_name in HM_VARS:
                var_samples = []
                for pos_idx in sampled_positions_comp:
                    yi, xi = self.valid_chip_positions[pos_idx]
                    y_start = yi * self.stride
                    x_start = xi * self.stride
                    
                    # Sample one random time step
                    t_idx = int(rng.integers(0, self.T))
                    arr = (
                        ds[var_name]
                        .isel(time=t_idx, y=slice(y_start, y_start + self.chip_size), x=slice(x_start, x_start + self.chip_size))
                        .load()
                        .values
                    )
                    var_samples.append(arr)
                var_stack = np.stack(var_samples, axis=0)
                self.comp_means[var_name] = np.nanmean(var_stack)
                self.comp_stds[var_name] = np.nanstd(var_stack) + 1e-8
            print("Component normalization stats:")
            for var_name in HM_VARS:
                print(f"  {var_name}: mean={self.comp_means[var_name]:.6e}, std={self.comp_stds[var_name]:.6e}")

        self.C_comp = len(HM_VARS) if self.include_components else 0
        self.C_dyn = 1 + self.C_comp
        self.C_static = len(static_var_names)
        self._static_var_names = static_var_names

    def __len__(self):
        return len(self.valid_chip_positions)

    def __getitem__(self, idx):
        # Get chip position from pre-computed valid positions
        yi, xi = self.valid_chip_positions[idx]
        y_start = yi * self.stride
        x_start = xi * self.stride
        
        # Extract chip directly from dataset
        chip = self.ds.isel(
            y=slice(y_start, y_start + self.chip_size),
            x=slice(x_start, x_start + self.chip_size)
        ).load()

        # Multi-horizon: always use fixed input years (1990, 1995, 2000)
        input_years = self.fixed_input_years

        # Temporal sampling for training: random end year determines targets
        if self.use_temporal_sampling:
            end_year = int(np.random.choice(self.end_year_options))
            target_years = tuple(end_year + offset for offset in (5, 10, 15, 20))
        else:
            end_year = self.fixed_target_years[-1]  # 2020
            target_years = self.fixed_target_years

        target_t_idxs = [years.index(y) if y <= 2020 else None for y in target_years]

        # Load input dynamic data (3 timesteps)
        input_dynamic = np.empty((self.timesteps, self.C_dyn, self.chip_size, self.chip_size), dtype=np.float32)
        for t_idx, year in enumerate(input_years):
            # HM target
            arr_hm = chip["AA"].isel(time=t_idx).values
            arr_hm = (arr_hm - self.hm_mean) / self.hm_std
            input_dynamic[t_idx, 0, :, :] = arr_hm

            # Components if enabled
            if self.include_components:
                for c_idx, var_name in enumerate(HM_VARS):
                    carr = chip[var_name].isel(time=t_idx).values
                    carr = np.nan_to_num(carr, nan=0.0)
                    carr = (carr - self.comp_means[var_name]) / self.comp_stds[var_name]
                    input_dynamic[t_idx, 1 + c_idx, :, :] = carr

        # Load static data (same for all timesteps)
        input_static = np.empty((self.C_static, self.chip_size, self.chip_size), dtype=np.float32)
        for static_idx, var_name in enumerate(self._static_var_names):
            sarr = chip[var_name].values
            sarr = np.nan_to_num(sarr, nan=0.0)
            sarr = (sarr - self.static_means[static_idx]) / self.static_stds[static_idx]
            input_static[static_idx, :, :] = sarr

        # Coordinates from actual chip position
        x_coords = chip["x"].values
        y_coords = chip["y"].values
        xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
        lonlat = np.stack([xx, yy], axis=-1).astype(np.float32)

        # Multi-horizon targets
        targets = {}
        horizon_names = ['target_5yr', 'target_10yr', 'target_15yr', 'target_20yr']

        for horizon_name, t_idx, target_year in zip(horizon_names, target_t_idxs, target_years):
            if t_idx is None or target_year > 2020:
                target_h = np.full((self.chip_size, self.chip_size), np.nan, dtype=np.float32)
            else:
                target_h = chip["AA"].isel(time=t_idx).values
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


def get_dataloader(
    split='train',
    batch_size=1,
    chip_size=128,
    timesteps=3,
    chips_per_epoch=None,
    fixed_input_years=(1990, 1995, 2000),
    fixed_target_years=(2005, 2010, 2015, 2020),
    use_temporal_sampling=True,
    end_year_options=(2000, 2005, 2010, 2015),
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    stat_samples=256,
    include_components=True,
    static_channels=None,
    zarr_path=ZARR_PATH,
    valid_chips_metadata_path='data/processed/valid_chips_metadata.json',
):
    """
    Create a DataLoader for Human Footprint dataset using pre-computed valid chips.
    
    Args:
        split: Which split to use ('train', 'val', 'test', 'calib')
        batch_size: Batch size
        chip_size: Size of spatial chips
        chips_per_epoch: Number of chips to sample per epoch (None = use all valid chips)
        valid_chips_metadata_path: Path to pre-computed valid chips metadata
        zarr_path: Path to Zarr/Icechunk repository
        
    Note:
        - Shuffle is always True for train split, False for others
        - All chips are guaranteed to have valid data (no empty chips)
        - Splits are geographically deterministic
        - If chips_per_epoch is specified, a random subset is sampled each epoch
    """
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
    )
    
    # Always shuffle for train, never for val/test/calib
    shuffle = (split == 'train')
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

if __name__ == "__main__":
    # Test with pre-computed valid chips
    loader = get_dataloader(
        split='train',
        batch_size=2,
        chip_size=128,
        timesteps=3
    )
    batch = next(iter(loader))
    print(f"input_dynamic shape: {batch['input_dynamic'].shape}")
    print(f"input_static shape: {batch['input_static'].shape}")
    print(f"target shape: {batch['target'].shape}")
    print(f"Dataset size: {len(loader.dataset)} chips")
    print(f"Shuffle enabled: {loader.shuffle}")
